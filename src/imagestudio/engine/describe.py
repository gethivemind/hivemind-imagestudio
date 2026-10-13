"""Image to text with llama.cpp (MIT) and a permissively licensed vision-language model.

The studio downloads the pinned llama.cpp release for this OS (hash-verified) and runs
``llama-mtmd-cli`` once per request: the text model plus its multimodal projector, the image,
and a prompt. Everything stays on the machine. Used for captions, "write a prompt that would
make this", and alt text.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imagestudio import download as dl
from imagestudio.config import ENV_THREADS, Settings
from imagestudio.engine.base import EngineError
from imagestudio.registry import LLAMA_TAG, EngineAsset, find_llama_asset

ConsentFn = Callable[[str, int], bool]
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

PROMPTS = {
    "describe": (
        "Describe this image in two or three sentences: the subject, the setting, the style "
        "and the lighting. Be concrete and neutral."
    ),
    "prompt": (
        "Write a single text-to-image prompt that would recreate this image as closely as "
        "possible. Mention subject, composition, setting, style, medium, lighting, colours and "
        "mood. Output only the prompt, no preamble."
    ),
    "alt": (
        "Write one sentence of alt text for this image, for a screen reader. Output only the "
        "sentence."
    ),
    "tags": (
        "List 8 to 12 short comma-separated tags for this image: subjects, style, mood, colours."
    ),
}


@dataclass
class DescribeSpec:
    image: Path
    model_path: Path
    mmproj_path: Path
    task: str = "describe"  # describe | prompt | alt | tags | custom
    question: str | None = None  # custom task: the exact instruction
    max_tokens: int = 300
    threads: int | None = None


@dataclass
class DescribeResult:
    text: str
    seconds: float
    model: str


class DescribeEngine:
    name = "llama.cpp"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def install_dir(self) -> Path:
        return self.settings.tools_dir / "llama" / LLAMA_TAG

    def asset(self) -> EngineAsset | None:
        import platform

        return find_llama_asset(sys.platform, platform.machine())

    def binary(self) -> Path | None:
        env = os.environ.get("IMAGESTUDIO_LLAMA_BIN")
        if env and Path(env).exists():
            return Path(env)
        a = self.asset()
        if a:
            for p in (self.install_dir / a.exe, self.install_dir / LLAMA_TAG / a.exe):
                if p.exists():
                    return p
        found = shutil.which("llama-mtmd-cli")
        return Path(found) if found else None

    def available(self) -> bool:
        return self.binary() is not None

    def version(self) -> str | None:
        return f"llama.cpp {LLAMA_TAG}" if self.available() else None

    def install(
        self, consent: ConsentFn, progress: Callable[[int, int], None] | None = None
    ) -> Path:
        a = self.asset()
        if a is None:
            import platform

            raise EngineError(
                f"no pinned llama.cpp build for {sys.platform}/{platform.machine()}; set "
                "IMAGESTUDIO_LLAMA_BIN to a llama-mtmd-cli of your own"
            )
        existing = self.binary()
        if existing is not None and existing.is_relative_to(self.install_dir):
            return existing
        if not consent(f"llama.cpp {LLAMA_TAG} ({a.asset}, MIT licence)", a.size):
            raise EngineError("download of llama.cpp was not approved")
        archive = self.settings.tools_dir / a.asset
        dl.download(a.url, archive, a.sha256, a.size, progress=progress)
        self.install_dir.mkdir(parents=True, exist_ok=True)
        if a.asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                for m in z.infolist():
                    name = Path(m.filename).name
                    if not name or m.is_dir() or ".." in m.filename:
                        continue
                    with z.open(m) as src, (self.install_dir / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    name = Path(m.name).name
                    if not m.isfile() or not name or ".." in m.name:
                        continue
                    f = t.extractfile(m)
                    if f is None:
                        continue
                    with (self.install_dir / name).open("wb") as dst:
                        shutil.copyfileobj(f, dst)
        if sys.platform != "win32":
            for p in self.install_dir.iterdir():
                p.chmod(p.stat().st_mode | 0o111)
        archive.unlink(missing_ok=True)
        exe = self.install_dir / a.exe
        if not exe.exists():
            raise EngineError(f"{a.exe} missing after unpacking {a.asset}")
        return exe

    def describe(self, spec: DescribeSpec) -> DescribeResult:
        b = self.binary()
        if b is None:
            raise EngineError(
                "llama.cpp is not installed; run `imagestudio install-engine --vision`"
            )
        instruction = spec.question or PROMPTS.get(spec.task) or PROMPTS["describe"]
        threads = spec.threads or int(os.environ.get(ENV_THREADS, "0") or 0)
        args = [
            str(b),
            "-m",
            str(spec.model_path),
            "--mmproj",
            str(spec.mmproj_path),
            "--image",
            str(spec.image),
            "-p",
            instruction,
            "-n",
            str(spec.max_tokens),
            "--temp",
            "0.2",
            "-c",
            "4096",
            "--no-mmproj-offload",
        ]
        if threads > 0:
            args += ["-t", str(threads)]
        t0 = time.monotonic()
        try:
            out = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
                cwd=str(b.parent),
            )
        except subprocess.TimeoutExpired as e:
            raise EngineError("describe timed out after 15 minutes") from e
        except OSError as e:
            raise EngineError(f"could not start {b}: {e}") from e
        if out.returncode != 0:
            tail = "\n".join((out.stderr or "").strip().splitlines()[-8:])
            raise EngineError(f"llama-mtmd-cli exited with {out.returncode}:\n{tail}")
        text = _ANSI.sub("", out.stdout or "").strip()
        # llama-mtmd-cli prints only the answer on stdout; logs go to stderr.
        text = text.replace("<|im_end|>", "").strip()
        if not text:
            raise EngineError("the vision model returned no text")
        return DescribeResult(text=text, seconds=time.monotonic() - t0, model=spec.model_path.stem)
