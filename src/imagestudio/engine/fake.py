"""A fake engine for tests and dry runs: writes a tiny valid PNG, reports progress, needs
no models and no network."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

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


class FakeDescribeEngine:
    name = "fake-vision"

    def __init__(self) -> None:
        self.calls: list = []

    def available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "fake-vision-0"

    def describe(self, spec):
        from imagestudio.engine.describe import DescribeResult

        self.calls.append(spec)
        return DescribeResult(
            text=f"[{spec.task}] a fake description of {spec.image.name}",
            seconds=0.01,
            model="fake",
        )

    def text_binary(self):
        return Path("fake-llama-completion")

    def complete(self, spec):
        from imagestudio.engine.describe import DescribeResult

        self.calls.append(spec)
        text = (
            '{"prompt": "a fake, detailed prompt about '
            + spec.user.splitlines()[-1][:40]
            + '", "negative": "blurry, text", "notes": "assumed daylight"}'
        )
        return DescribeResult(text=text, seconds=0.01, model="fake")


class FakeClipEngine:
    """Deterministic unit vectors: images by file bytes, text by its words (so identical
    images are duplicates and a query that repeats a prompt's words ranks that image first)."""

    name = "fake-clip"

    def __init__(self) -> None:
        self.calls: list = []

    def available(self) -> bool:
        return True

    @staticmethod
    def _vec(seed: bytes):
        import hashlib

        import numpy as np

        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "big"))
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_image(self, image, vision_model):
        from PIL import Image

        self.calls.append(("image", str(image)))
        with Image.open(image) as im:  # pixels, not bytes: PNG text chunks differ per job
            return self._vec(im.convert("RGB").tobytes())

    def embed_text(self, text, text_model, vocab, merges):
        self.calls.append(("text", text))
        return self._vec(text.lower().encode("utf-8"))


class FakeFaceDetector:
    """Finds one face in the middle of any image (or none, when ``faces`` is empty)."""

    name = "fake-faces"

    def __init__(self, faces=None) -> None:
        self.faces = faces if faces is not None else [(4, 4, 6, 6)]
        self.calls: list = []

    def available(self) -> bool:
        return True

    def detect(self, image, model_path, threshold: float = 0.7):
        self.calls.append((image, model_path))
        return list(self.faces)


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
