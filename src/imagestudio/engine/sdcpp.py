"""stable-diffusion.cpp engine: one binary, GGUF weights, CPU or any GPU.

The binary is found in this order: ``IMAGESTUDIO_SD_BIN``, the studio's own
``tools/sdcpp/<tag>/`` folder, then ``sd-cli`` on PATH. ``install()`` downloads the pinned
release for this OS and architecture (hash-verified) after consent.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

from imagestudio import download as dl
from imagestudio.config import ENV_SD_BIN, ENV_SD_FLAVOR, ENV_THREADS, Settings
from imagestudio.engine.base import EngineError, EngineResult, GenerateSpec, Progress, ProgressFn
from imagestudio.registry import SDCPP_TAG, EngineAsset, find_engine_asset

_STEP_RE = re.compile(r"(\d+)/(\d+)\s*-\s*([\d.]+)\s*(s/it|it/s)")
_TAKING_RE = re.compile(r"taking\s+([\d.]+)\s*s", re.IGNORECASE)
_VERSION_RE = re.compile(r"commit\s+([0-9a-f]+)", re.IGNORECASE)

ConsentFn = Callable[[str, int], bool]


def default_flavor() -> str:
    return os.environ.get(ENV_SD_FLAVOR, "cpu").lower()


class SdCppEngine:
    name = "stable-diffusion.cpp"

    def __init__(self, settings: Settings, flavor: str | None = None) -> None:
        self.settings = settings
        self.flavor = flavor or default_flavor()
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._cancelled = False

    # -- locating / installing ------------------------------------------------------------

    @property
    def install_dir(self) -> Path:
        return self.settings.tools_dir / "sdcpp" / f"{SDCPP_TAG}-{self.flavor}"

    def asset(self) -> EngineAsset | None:
        import platform

        return find_engine_asset(sys.platform, platform.machine(), self.flavor)

    def binary(self) -> Path | None:
        env = os.environ.get(ENV_SD_BIN)
        if env and Path(env).exists():
            return Path(env)
        a = self.asset()
        if a:
            p = self.install_dir / a.exe
            if p.exists():
                return p
        for name in ("sd-cli", "sd"):
            found = shutil.which(name)
            if found:
                return Path(found)
        return None

    def available(self) -> bool:
        return self.binary() is not None

    def version(self) -> str | None:
        b = self.binary()
        if not b:
            return None
        try:
            out = subprocess.run(
                [str(b), "--version"], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        text = (out.stdout or "") + (out.stderr or "")
        m = _VERSION_RE.search(text)
        return f"stable-diffusion.cpp {m.group(1)}" if m else text.strip().splitlines()[0:1] or None

    def install(
        self, consent: ConsentFn, progress: Callable[[int, int], None] | None = None
    ) -> Path:
        """Download and unpack the pinned release. Returns the CLI path."""
        a = self.asset()
        if a is None:
            import platform

            raise EngineError(
                f"no pinned stable-diffusion.cpp build for {sys.platform}/{platform.machine()} "
                f"flavor {self.flavor!r}. Build it from source and set {ENV_SD_BIN}, or put "
                "sd-cli on PATH."
            )
        exe = self.install_dir / a.exe
        if exe.exists():
            return exe
        if not consent(f"stable-diffusion.cpp {SDCPP_TAG} ({a.asset}, MIT licence)", a.size):
            raise EngineError("download of the engine was not approved")
        zip_path = self.settings.tools_dir / a.asset
        dl.download(a.url, zip_path, a.sha256, a.size, progress=progress)
        self.install_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for member in z.infolist():
                # Flatten: releases put files at the root, but guard against nested dirs
                # and path traversal.
                name = Path(member.filename).name
                if not name or member.is_dir() or ".." in member.filename:
                    continue
                with z.open(member) as src, (self.install_dir / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if sys.platform != "win32":
            for p in self.install_dir.iterdir():
                if p.suffix in ("", ".so", ".dylib") or p.name.startswith("sd-"):
                    p.chmod(p.stat().st_mode | 0o111)
        zip_path.unlink(missing_ok=True)
        if not exe.exists():
            raise EngineError(f"{a.exe} missing after unpacking {a.asset}")
        return exe

    # -- running --------------------------------------------------------------------------

    def build_args(self, spec: GenerateSpec) -> list[str]:
        b = self.binary()
        if b is None:
            raise EngineError(
                "stable-diffusion.cpp is not installed; run `imagestudio install-engine`"
            )
        threads = spec.threads or int(os.environ.get(ENV_THREADS, "0") or 0)
        args = [
            str(b),
            "-M",
            "img_gen",
            "-p",
            spec.prompt,
            "-W",
            str(spec.width),
            "-H",
            str(spec.height),
            "--steps",
            str(spec.steps),
            "--cfg-scale",
            str(spec.cfg_scale),
            "--sampling-method",
            spec.sampler,
            "-s",
            str(spec.seed),
            "-o",
            str(spec.output),
            "--log-level",
            "info",
        ]
        if spec.negative_prompt:
            args += ["-n", spec.negative_prompt]
        if threads > 0:
            args += ["-t", str(threads)]
        for flag, path in spec.component_paths.items():
            args += [flag, str(path)]
        args += list(spec.extra_args)
        return args

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()

    def generate(self, spec: GenerateSpec, on_progress: ProgressFn | None = None) -> EngineResult:
        args = self.build_args(spec)
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        first_step_at: float | None = None
        last_step_at: float | None = None
        load_seconds: float | None = None
        sampling_seconds: float | None = None
        phase = "loading"
        tail: list[str] = []

        def emit(p: Progress) -> None:
            if on_progress:
                on_progress(p)

        with self._lock:
            self._cancelled = False
            try:
                self._proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(args[0]).parent),
                )
            except OSError as e:
                raise EngineError(f"could not start {args[0]}: {e}") from e
        proc = self._proc
        assert proc.stdout is not None
        emit(Progress("loading", message="starting stable-diffusion.cpp"))

        buf = b""
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            # Progress bars are redrawn with \r; log lines end with \n.
            while True:
                idx = min((i for i in (buf.find(b"\r"), buf.find(b"\n")) if i >= 0), default=-1)
                if idx < 0:
                    break
                line, buf = buf[:idx].decode("utf-8", "replace").strip(), buf[idx + 1 :]
                if not line:
                    continue
                now = time.monotonic()
                m = _STEP_RE.search(line)
                if m:
                    step, total, rate, unit = (
                        int(m.group(1)),
                        int(m.group(2)),
                        float(m.group(3)),
                        m.group(4),
                    )
                    sps = rate if unit == "s/it" else (1.0 / rate if rate else None)
                    if first_step_at is None:
                        first_step_at = now
                        load_seconds = load_seconds or (now - t0)
                    last_step_at = now
                    phase = "sampling"
                    emit(Progress("sampling", step, total, sps, line[-80:]))
                    continue
                tail.append(line[-200:])
                del tail[:-40]
                low = line.lower()
                if "loading tensors completed" in low or "total params memory" in low:
                    mt = _TAKING_RE.search(line)
                    if mt:
                        load_seconds = float(mt.group(1))
                    phase = "encoding"
                    emit(Progress(phase, message=line[-120:]))
                elif "sampling completed" in low:
                    mt = _TAKING_RE.search(line)
                    if mt:
                        sampling_seconds = float(mt.group(1))
                    phase = "decoding"
                    emit(Progress(phase, message=line[-120:]))
                elif "decode_first_stage completed" in low or ("latent" in low and "decod" in low):
                    phase = "saving"
                    emit(Progress(phase, message=line[-120:]))
                elif "[error]" in low or "error:" in low:
                    emit(Progress(phase, message=line[-160:]))
        rc = proc.wait()
        with self._lock:
            self._proc = None
            cancelled = self._cancelled
        total = time.monotonic() - t0
        if cancelled:
            raise EngineError("cancelled")
        if rc != 0 or not spec.output.exists():
            msg = "\n".join(tail[-12:])
            raise EngineError(
                f"stable-diffusion.cpp exited with code {rc}"
                + (" and produced no image" if not spec.output.exists() else "")
                + (f":\n{msg}" if msg else "")
            )
        if sampling_seconds is None and first_step_at and last_step_at:
            sampling_seconds = last_step_at - first_step_at
        return EngineResult(
            output=spec.output,
            load_seconds=load_seconds or 0.0,
            sampling_seconds=sampling_seconds or 0.0,
            total_seconds=total,
            log_tail=tail[-12:],
        )
