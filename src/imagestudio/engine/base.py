"""The engine interface. stable-diffusion.cpp is the first implementation; ComfyUI can be
added later behind the same protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class GenerateSpec:
    """Everything an engine needs to render one image. Paths are already resolved."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    sampler: str
    output: Path
    component_paths: dict[str, Path]  # engine flag -> weight file
    extra_args: tuple[str, ...] = ()
    threads: int | None = None
    preview_path: Path | None = None  # engine writes a cheap preview here after each step


@dataclass
class UpscaleSpec:
    """Upscale one image with an ESRGAN-family model (output size = input × scale)."""

    input: Path
    output: Path
    model_path: Path
    repeats: int = 1  # run the 4x model this many times
    threads: int | None = None


@dataclass
class Progress:
    phase: str  # loading | encoding | sampling | decoding | saving
    step: int = 0
    total_steps: int = 0
    seconds_per_step: float | None = None
    message: str = ""


@dataclass
class EngineResult:
    output: Path
    load_seconds: float
    sampling_seconds: float
    total_seconds: float
    log_tail: list[str] = field(default_factory=list)
    peak_rss_bytes: int = 0  # highest resident memory of the engine process, if measured


ProgressFn = Callable[[Progress], None]


class EngineError(RuntimeError):
    pass


class Engine(Protocol):
    name: str

    def available(self) -> bool: ...

    def version(self) -> str | None: ...

    def generate(
        self, spec: GenerateSpec, on_progress: ProgressFn | None = None
    ) -> EngineResult: ...

    def upscale(self, spec: UpscaleSpec, on_progress: ProgressFn | None = None) -> EngineResult: ...
