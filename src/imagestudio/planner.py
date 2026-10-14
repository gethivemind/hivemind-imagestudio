"""Turns a request (model, preset, size) into a concrete plan: which weights, which
settings, how long it will take on this machine, and why.

Policy (ADR-051): quality is a setting, never a consequence. The only thing that can change
which weights run is whether they physically fit in memory, and when that happens the plan
says so in ``fit_note`` and every job record carries it. Nothing here lowers resolution or
steps on its own.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from imagestudio.calibration import Calibration, Estimate
from imagestudio.hardware import GIB, Hardware
from imagestudio.models import ModelStore
from imagestudio.registry import MODELS, PRESET_NAMES, ModelSpec, Preset, Variant

HEADROOM_BYTES = 2 * GIB  # for the OS and the caller
WORKING_BYTES_PER_MEGAPIXEL = int(1.5 * GIB)  # compute buffers at 1 MP, roughly linear
MAX_SIDE = 2048
MAX_STEPS = 100
MAX_CFG = 20.0
SAMPLERS = (
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
)
MIN_SIDE = 256
SIDE_MULTIPLE = 16


class PlanError(ValueError):
    pass


UNLOCK_HINT = (
    "Over-budget runs are locked by default. To try anyway, set "
    'IMAGESTUDIO_ALLOW_OVER_BUDGET=1 or put {"allow_over_budget": true} in config.json '
    "in the data directory; the run may swap for hours or fail to allocate."
)


@dataclass
class Load:
    """The performance meter: what the job needs against what the machine has."""

    required_bytes: int
    budget_bytes: int
    available_bytes: int

    @property
    def score(self) -> int:
        """0-100+ : percent of the machine's memory budget this job needs."""
        if self.budget_bytes <= 0:
            return 999
        return round(100 * self.required_bytes / self.budget_bytes)

    @property
    def label(self) -> str:
        if self.score > 100:
            return "over budget"
        if self.required_bytes > self.available_bytes:
            return "will swap"
        if self.score >= 70:
            return "tight"
        return "comfortable"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "required_gib": round(self.required_bytes / GIB, 1),
            "budget_gib": round(self.budget_bytes / GIB, 1),
            "available_gib": round(self.available_bytes / GIB, 1),
            "meaning": {
                "comfortable": "fits in free memory with room to spare",
                "tight": "fits the budget; close to the machine's limit",
                "will swap": "fits total memory but not what is free now; expect it to be "
                "several times slower",
                "over budget": "needs more memory than this machine has; locked unless "
                "allow_over_budget is set",
            }[self.label],
        }


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
    sampler: str
    estimate: Estimate
    fit_note: str | None = None
    installed: bool = True
    warnings: list[str] = field(default_factory=list)
    low_memory: bool = False  # weights exceed free RAM: engine memory-maps them
    load: Load | None = None
    over_budget: bool = False

    def to_dict(self) -> dict:
        return {
            "model": self.model.name,
            "variant": self.variant.name,
            "preset": self.preset.name,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "sampler": self.sampler,
            "estimate": self.estimate.to_dict(),
            "fit_note": self.fit_note,
            "installed": self.installed,
            "warnings": list(self.warnings),
            "low_memory": self.low_memory,
            "load": self.load.to_dict() if self.load else None,
            "over_budget": self.over_budget,
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
    def __init__(
        self,
        hw: Hardware,
        store: ModelStore,
        calibration: Calibration,
        allow_over_budget: bool = False,
    ) -> None:
        self.allow_over_budget = allow_over_budget
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
        ranked = sorted(
            (m for m in MODELS.values() if m.role == "generate"), key=lambda m: m.quality_rank
        )
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
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
    ) -> Plan:
        """``steps``, ``cfg_scale`` and ``sampler`` are advanced overrides of the preset."""
        if preset_name not in PRESET_NAMES:
            raise PlanError(f"unknown preset {preset_name!r}; one of {', '.join(PRESET_NAMES)}")
        if steps is not None and not 1 <= steps <= MAX_STEPS:
            raise PlanError(f"steps must be between 1 and {MAX_STEPS}, got {steps}")
        if cfg_scale is not None and not 0.0 <= cfg_scale <= MAX_CFG:
            raise PlanError(f"cfg_scale must be between 0 and {MAX_CFG}, got {cfg_scale}")
        if sampler is not None and sampler not in SAMPLERS:
            raise PlanError(f"unknown sampler {sampler!r}; one of {', '.join(SAMPLERS)}")
        w0 = snap_side(width, "width") if width else None
        h0 = snap_side(height, "height") if height else None
        model = self.resolve_model(prompt_model, w0 or 1024, h0 or 1024)
        preset = model.presets[preset_name]
        w = w0 or preset.width
        h = h0 or preset.height
        n_steps = steps if steps is not None else preset.steps
        cfg = cfg_scale if cfg_scale is not None else preset.cfg_scale
        chosen_sampler = sampler or model.sampler

        warnings: list[str] = []
        fit_note: str | None = None
        over_budget = False
        installed_names = {v.name for v in self.store.installed_variants(model)}

        if variant_name:
            variant = model.variant(variant_name)
            fit = next(f for f in fit_variants(model, self.hw, w, h) if f.variant is variant)
            if not fit.fits:
                msg = (
                    f"{model.display_name} [{variant.name}] at {w}x{h} needs about "
                    f"{fit.required_bytes / GIB:.1f} GiB; this machine's budget is "
                    f"{fit.budget_bytes / GIB:.1f} GiB."
                )
                if not self.allow_over_budget:
                    raise PlanError(f"{msg} {UNLOCK_HINT}")
                over_budget = True
                warnings.append(f"{msg} Running over budget because allow_over_budget is set.")
        else:
            fits = fit_variants(model, self.hw, w, h)
            top = fits[0]
            chosen = next((f for f in fits if f.fits), None)
            if self.allow_over_budget:
                # The unlock is set: a higher-fidelity set that is already downloaded beats a
                # smaller one that would have to be fetched. Never a silent downgrade.
                have = next((f for f in fits if f.variant.name in installed_names), None)
                if have is not None and (chosen is None or have.variant.rank < chosen.variant.rank):
                    if not have.fits:
                        over_budget = True
                        warnings.append(
                            f"{model.display_name} [{have.variant.name}] at {w}x{h} needs about "
                            f"{have.required_bytes / GIB:.1f} GiB; the budget is "
                            f"{have.budget_bytes / GIB:.1f} GiB. Running over budget because "
                            "allow_over_budget is set."
                        )
                    chosen = have
            if chosen is None:
                msg = (
                    f"{model.display_name} does not fit in this machine's memory in any "
                    f"quantisation at {w}x{h}: the smallest set needs about "
                    f"{fits[-1].required_bytes / GIB:.1f} GiB, the budget is "
                    f"{top.budget_bytes / GIB:.1f} GiB."
                )
                if not self.allow_over_budget:
                    raise PlanError(f"{msg} Choose another model or a smaller size. {UNLOCK_HINT}")
                chosen = fits[-1]
                over_budget = True
                warnings.append(f"{msg} Running over budget because allow_over_budget is set.")
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
            model, variant.name, variant.download_bytes, w, h, n_steps, cfg
        )
        low_memory = False
        if (
            self.hw.ram_available_bytes < required_bytes(variant, w, h)
            and self.hw.usable_gpu is None
        ):
            warnings.append(
                f"Only {self.hw.ram_available_bytes / GIB:.1f} GiB of RAM is free right now; "
                f"the job needs about {required_bytes(variant, w, h) / GIB:.1f} GiB. Close "
                "other programs or expect swapping to slow it down. The weights will be "
                "memory-mapped so the OS pages them from the model files instead of the swap file."
            )
            low_memory = True
        if model.family == "flux" and n_steps > preset.steps:
            warnings.append(
                f"{model.display_name} is step-distilled: {preset.steps} steps is its full "
                f"count and {n_steps} will take proportionally longer without improving "
                "the image."
            )
        load = Load(
            required_bytes=required_bytes(variant, w, h),
            budget_bytes=memory_budget(self.hw),
            available_bytes=self.hw.ram_available_bytes
            + ((self.hw.usable_gpu.vram_bytes or 0) if self.hw.usable_gpu else 0),
        )
        return Plan(
            model=model,
            variant=variant,
            preset=preset,
            width=w,
            height=h,
            steps=n_steps,
            cfg_scale=cfg,
            sampler=chosen_sampler,
            estimate=estimate,
            fit_note=fit_note,
            installed=variant.name in installed_names,
            warnings=warnings,
            low_memory=low_memory,
            load=load,
            over_budget=over_budget,
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
                    "load": plan.load.to_dict() if plan.load else None,
                }
            )
        return out
