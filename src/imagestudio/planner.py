"""Turns a request (model, preset, size) into a concrete plan: which weights, which
settings, how long it will take on this machine, and why.

Policy (ADR-051): quality is a setting, never a consequence. The only thing that can change
which weights run is whether they physically fit in memory, and when that happens the plan
says so in ``fit_note`` and every job record carries it. Nothing here lowers resolution or
steps on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from imagestudio.calibration import Calibration, Estimate
from imagestudio.hardware import GIB, Hardware
from imagestudio.models import ModelStore
from imagestudio.registry import MODELS, PRESET_NAMES, ModelSpec, Preset, Variant

HEADROOM_BYTES = 2 * GIB  # for the OS and the caller
WORKING_BYTES_PER_MEGAPIXEL = int(1.5 * GIB)  # compute buffers at 1 MP, roughly linear
MAX_SIDE = 2048
MIN_SIDE = 256
SIDE_MULTIPLE = 16


class PlanError(ValueError):
    pass


@dataclass
class Fit:
    variant: Variant
    required_bytes: int
    budget_bytes: int

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.budget_bytes


@dataclass
class Plan:
    model: ModelSpec
    variant: Variant
    preset: Preset
    width: int
    height: int
    steps: int
    cfg_scale: float
    estimate: Estimate
    fit_note: str | None = None
    installed: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model.name,
            "variant": self.variant.name,
            "preset": self.preset.name,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "estimate": self.estimate.to_dict(),
            "fit_note": self.fit_note,
            "installed": self.installed,
            "warnings": list(self.warnings),
        }


def memory_budget(hw: Hardware) -> int:
    """Bytes available for weights plus working memory.

    A discrete GPU with known VRAM adds its VRAM (sd.cpp can keep weights in RAM and stream
    them). Integrated graphics share system RAM, so it adds nothing.
    """
    budget = hw.ram_total_bytes - HEADROOM_BYTES
    gpu = hw.usable_gpu
    if gpu and gpu.vram_bytes:
        budget += gpu.vram_bytes
    return max(0, budget)


def required_bytes(variant: Variant, width: int, height: int) -> int:
    mp = (width * height) / 1_000_000
    return variant.download_bytes + int(WORKING_BYTES_PER_MEGAPIXEL * max(mp, 0.25))


def fit_variants(model: ModelSpec, hw: Hardware, width: int, height: int) -> list[Fit]:
    budget = memory_budget(hw)
    return [Fit(v, required_bytes(v, width, height), budget) for v in model.ranked_variants]


def best_fitting_variant(model: ModelSpec, hw: Hardware, width: int, height: int) -> Fit | None:
    for f in fit_variants(model, hw, width, height):
        if f.fits:
            return f
    return None


def snap_side(value: int, name: str) -> int:
    if not MIN_SIDE <= value <= MAX_SIDE:
        raise PlanError(f"{name} must be between {MIN_SIDE} and {MAX_SIDE}, got {value}")
    return (value // SIDE_MULTIPLE) * SIDE_MULTIPLE


class Planner:
    def __init__(self, hw: Hardware, store: ModelStore, calibration: Calibration) -> None:
        self.hw = hw
        self.store = store
        self.calibration = calibration

    def resolve_model(self, name: str | None, width: int, height: int) -> ModelSpec:
        """``None``/``"auto"`` picks the strongest catalogue model that is installed and fits;
        failing that, the strongest installed one (the fit is then reported as a warning)."""
        if name and name != "auto":
            if name not in MODELS:
                raise PlanError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
            return MODELS[name]
        ranked = sorted(MODELS.values(), key=lambda m: m.quality_rank)
        for m in ranked:
            if self.store.installed_variants(m) and best_fitting_variant(m, self.hw, width, height):
                return m
        for m in ranked:
            if self.store.installed_variants(m):
                return m
        return next(
            (m for m in ranked if best_fitting_variant(m, self.hw, width, height)), ranked[-1]
        )

    def plan(
        self,
        prompt_model: str | None,
        preset_name: str = "best",
        width: int | None = None,
        height: int | None = None,
        variant_name: str | None = None,
    ) -> Plan:
        if preset_name not in PRESET_NAMES:
            raise PlanError(f"unknown preset {preset_name!r}; one of {', '.join(PRESET_NAMES)}")
        w0 = snap_side(width, "width") if width else None
        h0 = snap_side(height, "height") if height else None
        model = self.resolve_model(prompt_model, w0 or 1024, h0 or 1024)
        preset = model.presets[preset_name]
        w = w0 or preset.width
        h = h0 or preset.height

        warnings: list[str] = []
        fit_note: str | None = None
        installed_names = {v.name for v in self.store.installed_variants(model)}

        if variant_name:
            variant = model.variant(variant_name)
            fit = next(f for f in fit_variants(model, self.hw, w, h) if f.variant is variant)
            if not fit.fits:
                warnings.append(
                    f"{model.name} [{variant.name}] needs about "
                    f"{fit.required_bytes / GIB:.1f} GiB but this machine has a budget of "
                    f"{fit.budget_bytes / GIB:.1f} GiB; it may swap heavily or fail."
                )
        else:
            fits = fit_variants(model, self.hw, w, h)
            top = fits[0]
            chosen = next((f for f in fits if f.fits), None)
            if chosen is None:
                raise PlanError(
                    f"{model.display_name} does not fit in this machine's memory in any "
                    f"quantisation: the smallest set needs about "
                    f"{fits[-1].required_bytes / GIB:.1f} GiB, the budget is "
                    f"{top.budget_bytes / GIB:.1f} GiB. Choose another model."
                )
            variant = chosen.variant
            if chosen is not top:
                fit_note = (
                    f"Running {model.display_name} as [{variant.name}] rather than "
                    f"[{top.variant.name}]: the higher-fidelity set needs about "
                    f"{top.required_bytes / GIB:.1f} GiB and this machine has a budget of "
                    f"{top.budget_bytes / GIB:.1f} GiB. {variant.note} This is a memory fit, "
                    "not a speed choice; resolution and steps are unchanged."
                )
            # Prefer the best installed variant that fits over one that would need a download,
            # and say when a better one is available.
            if variant.name not in installed_names:
                installed_fitting = [
                    f.variant for f in fits if f.fits and f.variant.name in installed_names
                ]
                if installed_fitting:
                    wanted = variant
                    variant = installed_fitting[0]
                    if variant.rank > wanted.rank:
                        warnings.append(
                            f"Using the installed [{variant.name}] set; the higher-fidelity "
                            f"[{wanted.name}] set fits this machine but is not downloaded "
                            f"(run `imagestudio download-model {model.name} --variant "
                            f"{wanted.name}`)."
                        )

        estimate = self.calibration.estimate(
            model, variant.name, variant.download_bytes, w, h, preset.steps, preset.cfg_scale
        )
        if (
            self.hw.ram_available_bytes < required_bytes(variant, w, h)
            and self.hw.usable_gpu is None
        ):
            warnings.append(
                f"Only {self.hw.ram_available_bytes / GIB:.1f} GiB of RAM is free right now; "
                f"the job needs about {required_bytes(variant, w, h) / GIB:.1f} GiB. Close "
                "other programs or expect swapping to slow it down."
            )
        return Plan(
            model=model,
            variant=variant,
            preset=preset,
            width=w,
            height=h,
            steps=preset.steps,
            cfg_scale=preset.cfg_scale,
            estimate=estimate,
            fit_note=fit_note,
            installed=variant.name in installed_names,
            warnings=warnings,
        )

    def preset_table(self, model: ModelSpec) -> list[dict]:
        """Presets with the expected time on this machine, for ``capabilities()``."""
        out = []
        for p in model.presets.values():
            try:
                plan = self.plan(model.name, p.name)
            except PlanError as e:
                out.append({"preset": p.name, "tradeoff": p.tradeoff, "error": str(e)})
                continue
            out.append(
                {
                    "preset": p.name,
                    "width": plan.width,
                    "height": plan.height,
                    "steps": plan.steps,
                    "cfg_scale": plan.cfg_scale,
                    "tradeoff": p.tradeoff,
                    "variant": plan.variant.name,
                    "expected_time": plan.estimate.to_dict(),
                    "fit_note": plan.fit_note,
                    "installed": plan.installed,
                }
            )
        return out
