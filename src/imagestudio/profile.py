"""The machine profile written by ``imagestudio benchmark``: what this machine measured, and
the preset it can run within a target time. Like a game's "detect settings": a recommendation
the user may override, never a cap on what the engine will attempt."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GIB = 1024**3


@dataclass
class MachineProfile:
    measured_at: str
    cpu: str
    physical_cores: int
    ram_total_gib: float
    gpu: str | None
    budget_gib: float
    model: str
    variant: str
    benchmark_preset: str
    width: int
    height: int
    steps: int
    wall_seconds: float
    sampling_seconds: float
    seconds_per_step_megapixel: float
    peak_rss_gib: float | None
    free_ram_gib_at_start: float
    swapped: bool
    target_minutes: float
    recommended_preset: str | None
    preset_estimates: dict[str, float] = field(default_factory=dict)  # preset -> seconds
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> MachineProfile | None:
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def recommend(preset_estimates: dict[str, float], target_seconds: float) -> str | None:
    """The highest-quality preset whose estimate meets the target; None if none does."""
    for name in ("best", "balanced", "fast"):
        est = preset_estimates.get(name)
        if est is not None and est <= target_seconds:
            return name
    return None
