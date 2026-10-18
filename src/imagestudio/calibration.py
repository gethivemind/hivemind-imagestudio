"""Per-machine timing calibration.

The ETA for a job is ``load_seconds + steps * megapixels * seconds_per_step_megapixel``,
where both coefficients are learned from real runs on this machine (exponential moving
average per model variant) and stored in ``calibration.json`` in the data dir. Until the
first run completes, a heuristic from the core count seeds the estimate and the ETA is
labelled ``uncalibrated``.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from imagestudio.hardware import Hardware
from imagestudio.registry import ModelSpec

# Seed heuristics. A 1024 px FLUX step on an 8-physical-core laptop CPU takes roughly
# 100 s in stable-diffusion.cpp; weights load from disk at roughly 200 MB/s.
_BASE_SECONDS_PER_STEP_MP_8_CORES = 100.0
_DISK_BYTES_PER_SECOND = 200 * 1024 * 1024
_GPU_SECONDS_PER_STEP_MP = 1.5  # a mid-range discrete GPU
_ALPHA = 0.5  # EMA weight of the newest observation
# A sampling step never costs less than this many megapixels' worth of work: attention and
# per-step overhead do not shrink with the image. 256 px is 0.065 MP but costs ~0.2 MP.
_MIN_STEP_MEGAPIXELS = 0.2


@dataclass
class Coefficients:
    load_seconds: float
    seconds_per_step_megapixel: float
    samples: int = 0

    @property
    def calibrated(self) -> bool:
        return self.samples > 0


@dataclass
class Estimate:
    seconds: float
    calibrated: bool
    basis: str

    def to_dict(self) -> dict:
        return asdict(self) | {"human": human_duration(self.seconds)}


def human_duration(seconds: float) -> str:
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


class Calibration:
    def __init__(self, path: Path, hardware: Hardware) -> None:
        self.path = path
        self.hardware = hardware
        self._data: dict[str, Coefficients] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = {k: Coefficients(**v) for k, v in raw.items()}
            except (OSError, ValueError, TypeError):
                self._data = {}

    _save_lock = threading.Lock()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # atomic: another process (UI, MCP server) may be reading at the same moment; the lock
        # and the per-call temp name keep two threads of this process from tripping over each
        # other, and the retry covers Windows refusing to replace a file a reader still holds
        with self._save_lock:
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(
                json.dumps({k: asdict(v) for k, v in self._data.items()}, indent=2),
                encoding="utf-8",
            )
            for attempt in range(20):
                try:
                    tmp.replace(self.path)
                    return
                except PermissionError:
                    if attempt == 19:
                        tmp.unlink(missing_ok=True)
                        raise
                    time.sleep(0.05)

    def _seed(self, model: ModelSpec, weight_bytes: int) -> Coefficients:
        if self.hardware.usable_gpu is not None:
            per_step = _GPU_SECONDS_PER_STEP_MP * model.compute_factor
        else:
            cores = max(1, self.hardware.physical_cores)
            per_step = _BASE_SECONDS_PER_STEP_MP_8_CORES * (8 / cores) * model.compute_factor
        load = weight_bytes / _DISK_BYTES_PER_SECOND
        return Coefficients(load_seconds=load, seconds_per_step_megapixel=per_step, samples=0)

    def coefficients(self, model: ModelSpec, variant: str, weight_bytes: int) -> Coefficients:
        return self._data.get(f"{model.name}:{variant}") or self._seed(model, weight_bytes)

    def estimate(
        self,
        model: ModelSpec,
        variant: str,
        weight_bytes: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float = 1.0,
    ) -> Estimate:
        c = self.coefficients(model, variant, weight_bytes)
        mp = max(_MIN_STEP_MEGAPIXELS, (width * height) / 1_000_000)
        passes = 2 if cfg_scale > 1.0 else 1
        seconds = c.load_seconds + steps * mp * c.seconds_per_step_megapixel * passes
        basis = (
            f"calibrated from {c.samples} run(s) on this machine"
            if c.calibrated
            else "uncalibrated: heuristic from core count; the first real run replaces it"
        )
        return Estimate(seconds=seconds, calibrated=c.calibrated, basis=basis)

    def record(
        self,
        model: ModelSpec,
        variant: str,
        weight_bytes: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        load_seconds: float,
        sampling_seconds: float,
    ) -> None:
        """Fold one finished job into the coefficients for its model variant."""
        mp = max(_MIN_STEP_MEGAPIXELS, (width * height) / 1_000_000)
        passes = 2 if cfg_scale > 1.0 else 1
        per_step = sampling_seconds / max(1e-6, steps * mp * passes)
        key = f"{model.name}:{variant}"
        prev = self._data.get(key)
        if prev is None or prev.samples == 0:
            new = Coefficients(load_seconds, per_step, 1)
        else:
            new = Coefficients(
                load_seconds=prev.load_seconds * (1 - _ALPHA) + load_seconds * _ALPHA,
                seconds_per_step_megapixel=(
                    prev.seconds_per_step_megapixel * (1 - _ALPHA) + per_step * _ALPHA
                ),
                samples=prev.samples + 1,
            )
        self._data[key] = new
        self._save()
