"""A fake engine for tests and dry runs: writes a tiny valid PNG, reports progress, needs
no models and no network."""

from __future__ import annotations

import struct
import time
import zlib

from imagestudio.engine.base import EngineResult, GenerateSpec, Progress, ProgressFn, UpscaleSpec


def tiny_png(width: int = 8, height: int = 8, rgb: tuple[int, int, int] = (90, 120, 200)) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class FakeEngine:
    name = "fake"

    def __init__(self, step_delay: float = 0.0, fail: bool = False) -> None:
        self.step_delay = step_delay
        self.fail = fail
        self.calls: list[GenerateSpec] = []
        self._cancel = False

    def available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "fake-0"

    def cancel(self) -> None:
        self._cancel = True

    def generate(self, spec: GenerateSpec, on_progress: ProgressFn | None = None) -> EngineResult:
        self.calls.append(spec)
        self._cancel = False
        t0 = time.monotonic()
        if on_progress:
            on_progress(Progress("loading", message="fake load"))
        if self.fail:
            raise RuntimeError("fake engine failure")
        t1 = time.monotonic()
        for i in range(1, spec.steps + 1):
            if self.step_delay:
                time.sleep(self.step_delay)
            if self._cancel:
                from imagestudio.engine.base import EngineError

                raise EngineError("cancelled")
            if on_progress:
                on_progress(Progress("sampling", i, spec.steps, self.step_delay))
        t2 = time.monotonic()
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        spec.output.write_bytes(tiny_png(min(spec.width, 16), min(spec.height, 16)))
        if on_progress:
            on_progress(Progress("saving", spec.steps, spec.steps))
        return EngineResult(
            output=spec.output,
            load_seconds=t1 - t0,
            sampling_seconds=t2 - t1,
            total_seconds=time.monotonic() - t0,
            log_tail=["fake engine"],
        )

    def upscale(self, spec: UpscaleSpec, on_progress: ProgressFn | None = None) -> EngineResult:
        self.calls.append(spec)
        t0 = time.monotonic()
        if on_progress:
            on_progress(Progress("sampling", 1, 1, message="fake upscale"))
        if self.step_delay:
            time.sleep(self.step_delay)
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        spec.output.write_bytes(tiny_png(32, 32, (200, 120, 90)))
        return EngineResult(
            spec.output, 0.0, time.monotonic() - t0, time.monotonic() - t0, ["fake"]
        )


class FakeBackgroundEngine:
    name = "fake-bg"

    def __init__(self) -> None:
        self.calls: list = []

    def available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "fake-bg-0"

    def remove_background(self, spec, on_progress: ProgressFn | None = None) -> EngineResult:
        self.calls.append(spec)
        if on_progress:
            on_progress(Progress("sampling", 1, 1, message="fake matte"))
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        spec.output.write_bytes(tiny_png(16, 16, (10, 200, 10)))
        return EngineResult(spec.output, 0.0, 0.01, 0.01, ["fake"])
