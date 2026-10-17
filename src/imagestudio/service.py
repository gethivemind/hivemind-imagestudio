"""The studio: wires hardware, models, calibration, planner, engine and the job queue.
Both the MCP server and the CLI talk to this one object."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from imagestudio import __version__
from imagestudio.calibration import Calibration, human_duration
from imagestudio.config import ENV_ALLOW_OVER_BUDGET, ENV_SD_FLAVOR, Settings
from imagestudio.engine.base import Engine
from imagestudio.engine.sdcpp import SdCppEngine
from imagestudio.hardware import Hardware, detect
from imagestudio.jobs import Job, JobStore, Worker
from imagestudio.models import ModelStore
from imagestudio.planner import (
    GIB,
    Plan,
    PlanError,
    Planner,
    fit_variants,
    required_bytes,
)
from imagestudio.planner import (
    memory_budget as memory_budget_bytes,
)
from imagestudio.profile import MachineProfile, now_iso, recommend
from imagestudio.promptparams import (
    apply_aspect,
    parse,
    reproducible,
    without_batch_and_seed,
)
from imagestudio.registry import (
    CUSTOM_FAMILIES,
    EXCLUDED_BY_POLICY,
    LICENCES,
    MODELS,
    SDCPP_TAG,
    SEARCH_MODELS,
    TOOL_MODELS,
    VISION_MODELS,
    ModelSpec,
    build_custom_model,
    default_tool_model,
    default_vision_model,
    edit_models,
    register_custom_models,
)

MAX_SEED = 2**31 - 1


_ASSIST_SYSTEM = (
    "You write prompts for a local text-to-image model, {display_name}. Reply with JSON only, "
    'on one line: {{"prompt": "...", "negative": "...", "notes": "..."}}. '
    "prompt: {style}. "
    "negative: a short comma-separated list of what to keep out, or an empty string "
    "(negative prompts are {negatives} by this model). "
    "notes: one short sentence on what you assumed or changed, written to the user. "
    "Keep everything the user asked for; invent only what is needed to make a complete "
    "picture. Nothing explicit or hateful. No preamble, no markdown."
)


class StudioError(RuntimeError):
    pass


def job_title(job: Job) -> str:
    """A short generic title: when, which preset, which size. Prompts can be long."""
    from datetime import datetime

    try:
        when = datetime.fromisoformat(job.finished_at or job.created_at).astimezone()
        stamp = when.strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        stamp = job.id
    size = f"{job.width}²" if job.width == job.height else f"{job.width}×{job.height}"
    kind = job.preset if job.tool == "generate" else job.tool.replace("_", " ")
    if job.tool in ("variations", "edit"):
        kind = f"{job.tool} · {job.preset}"
    if job.tool == "import":
        kind = "imported"
    return f"{stamp} · {kind} · {size}"


class BackgroundTask:
    """A long operation started from the UI (model download, benchmark) with live status."""

    def __init__(self, kind: str, label: str) -> None:
        self.kind = kind
        self.label = label
        self.status = "running"
        self.done_bytes = 0
        self.total_bytes = 0
        self.current_file: str | None = None
        self.message = ""
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "progress": (self.done_bytes / self.total_bytes) if self.total_bytes else None,
            "current_file": self.current_file,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at),
        }


class Studio:
    def __init__(
        self,
        settings: Settings | None = None,
        engine: Engine | None = None,
        hardware: Hardware | None = None,
        start_worker: bool = True,
        background_engine=None,
        describe_engine=None,
        face_detector=None,
        search_engine=None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.settings.ensure_dirs()
        self.hardware = hardware or detect(self.settings.hardware_cache_file)
        self.models = ModelStore(self.settings)
        self.calibration = Calibration(self.settings.calibration_file, self.hardware)
        self.planner = Planner(
            self.hardware,
            self.models,
            self.calibration,
            allow_over_budget=self.settings.allow_over_budget,
            cap_bytes=self._cap_bytes(),
        )
        self.custom_model_problems = register_custom_models(
            list(self.settings.config().get("custom_models") or [])
        )
        self.engine: Engine = engine or SdCppEngine(self.settings)
        if background_engine is None:
            from imagestudio.engine.background import BackgroundEngine

            background_engine = BackgroundEngine()
        self.background_engine = background_engine
        if describe_engine is None:
            from imagestudio.engine.describe import DescribeEngine

            describe_engine = DescribeEngine(self.settings)
        self.describe_engine = describe_engine
        if face_detector is None:
            from imagestudio.engine.faces import FaceDetector

            face_detector = FaceDetector()
        self.face_detector = face_detector
        if search_engine is None:
            from imagestudio.engine.clip import ClipEngine

            search_engine = ClipEngine()
        self.search_engine = search_engine
        self._search_index = None
        self._index_lock = threading.Lock()
        self.jobs = JobStore(self.settings)
        self.worker = Worker(
            self.jobs,
            self.engine,
            on_finished=self._on_finished,
            background_engine=self.background_engine,
            tool_model_path=self._tool_model_path,
        )
        self.worker_wanted = start_worker
        self.tasks: dict[str, BackgroundTask] = {}
        if start_worker:
            self.worker.start()

    @property
    def worker_active(self) -> bool:
        """True when this process renders jobs; False when another process holds the lock
        and this one only submits."""
        return self.worker.active

    @property
    def worker_owner_pid(self) -> int | None:
        return self.worker.lock.owner()

    def close(self) -> None:
        self.worker.stop()

    # -- calibration feedback --------------------------------------------------------------

    def _on_finished(self, job: Job) -> None:
        if job.status == "done":
            self._write_metadata(job)
            self._maybe_auto_caption(job)
            self._index_later(job)
        if job.status != "done" or job.sampling_seconds is None or job.sampling_seconds <= 0:
            return
        if job.tool != "generate":
            return
        model = MODELS.get(job.model)
        if model is None:
            return
        try:
            variant = model.variant(job.variant)
        except KeyError:
            return
        self.calibration.record(
            model,
            variant.name,
            variant.download_bytes,
            job.width,
            job.height,
            job.steps,
            job.cfg_scale,
            job.load_seconds or 0.0,
            job.sampling_seconds,
        )

    # -- PNG metadata -----------------------------------------------------------------------

    def metadata_for(self, job: Job) -> dict[str, str]:
        """The text chunks written into a finished PNG: ``parameters`` in the widely read
        Automatic1111 layout, and ``imagestudio`` as JSON with the reproducible prompt."""
        generated = job.tool in ("generate", "variations", "edit")
        if generated:
            lines = [job.prompt]
            if job.negative_prompt:
                lines.append(f"Negative prompt: {job.negative_prompt}")
            lines.append(
                f"Steps: {job.steps}, Sampler: {job.sampler}, CFG scale: {job.cfg_scale:g}, "
                f"Seed: {job.seed}, Size: {job.width}x{job.height}, Model: {job.model} "
                f"[{job.variant}], Preset: {job.preset}, Tool: {job.tool}"
            )
            params = "\n".join(lines)
        else:
            params = (
                f"Tool: {job.tool}, Model: {job.tool_model or '-'}, Size: {job.width}x{job.height}"
            )
        project = self.project(job.project) if job.project else None
        info = {
            "app": f"Hivemind Image Studio {__version__}",
            "job": job.id,
            "tool": job.tool,
            "project": project["name"] if project else "",
            "style": job.style,
            "style_text": job.style_text,
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt,
            "reproducible_prompt": reproducible(job) if generated else "",
            "model": job.model,
            "variant": job.variant,
            "preset": job.preset,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "cfg_scale": job.cfg_scale,
            "seed": job.seed,
            "sampler": job.sampler,
            "source_job": job.source_job,
            "tool_model": job.tool_model,
            "caption": job.caption,
        }
        return {"parameters": params, "imagestudio": json.dumps(info, ensure_ascii=False)}

    def _write_metadata(self, job: Job) -> None:
        """Re-save the finished PNG with its text chunks (lossless). Never fails the job."""
        if not job.output_path or job.tool == "import":
            return
        try:
            from PIL import Image
            from PIL.PngImagePlugin import PngInfo

            info = PngInfo()
            for k, v in self.metadata_for(job).items():
                info.add_text(k, v)
            path = Path(job.output_path)
            tmp = path.with_suffix(".tmp.png")
            with Image.open(path) as im:
                im.load()
                im.save(tmp, "PNG", pnginfo=info)
            tmp.replace(path)
        except Exception as e:  # metadata is a convenience, never fail the job
            job.warnings = [*job.warnings, f"metadata not written: {e}"]
            self.jobs.put(job)

    @staticmethod
    def read_metadata(path: Path) -> dict[str, Any]:
        """Our own chunk if the file carries one, else whatever ``parameters`` text exists."""
        from PIL import Image

        try:
            with Image.open(path) as im:
                text = dict(getattr(im, "text", {}) or {})
        except Exception:
            return {}
        out: dict[str, Any] = {}
        if text.get("imagestudio"):
            try:
                out = json.loads(text["imagestudio"])
            except ValueError:
                out = {}
        if not out.get("prompt") and text.get("parameters"):
            out["prompt"] = text["parameters"].split("\n", 1)[0].strip()
            out["parameters"] = text["parameters"]
        return out

    # -- auto-captions ----------------------------------------------------------------------

    _caption_lock = threading.Lock()
    _caption_warned = False

    def auto_caption_ready(self) -> bool:
        vm = default_vision_model()
        return (
            bool(self.settings.config().get("auto_caption", False))
            and vm is not None
            and self.describe_engine.available()
            and self.models.vision_installed(vm)
        )

    def _maybe_auto_caption(self, job: Job) -> None:
        """When the user turned on auto-captions and the vision stack is installed, caption a
        finished image in a background thread (one at a time, never blocking the queue).
        Upscales and cutouts keep their source's content, so they inherit its caption."""
        if job.caption or job.tool == "remove_background":
            return
        if job.tool == "upscale" and job.source_job:
            src = self.jobs.get(job.source_job)
            if src is not None and src.caption:
                job.caption = src.caption
                self.jobs.put(job)
                return
        if not self.auto_caption_ready():
            return

        def run() -> None:
            with self._caption_lock:
                try:
                    self.describe(job.id, "describe")
                except StudioError as e:
                    if not Studio._caption_warned:
                        Studio._caption_warned = True
                        import sys

                        print(f"auto-caption skipped: {e}", file=sys.stderr)

        threading.Thread(target=run, name=f"caption-{job.id}", daemon=True).start()

    # -- generate -------------------------------------------------------------------------

    def plan(
        self,
        model: str | None,
        preset: str,
        width: int | None,
        height: int | None,
        variant: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
    ) -> Plan:
        self.planner.allow_over_budget = self.settings.allow_over_budget
        try:
            return self.planner.plan(
                model, preset, width, height, variant, steps, cfg_scale, sampler
            )
        except PlanError as e:
            raise StudioError(str(e)) from e

    def generate(
        self,
        prompt: str,
        preset: str = "best",
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str = "",
        model: str | None = None,
        variant: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
        threads: int | None = None,
        vae_tiling: bool = False,
        loras: list | None = None,
        style: str = "",
        style_text: str = "",
        _defer: bool = False,
    ) -> Job:
        if not prompt or not prompt.strip():
            raise StudioError("prompt must not be empty")
        parsed_text = prompt
        parsed = parse(prompt)
        if parsed.errors:
            raise StudioError("; ".join(parsed.errors))
        if not parsed.prompt:
            raise StudioError("the prompt has only parameters and no words to draw")
        p = parsed.params
        if "aspect" in p:
            base = self.plan(p.get("model", model), p.get("preset", preset), None, None)
            apply_aspect(p, base.width, base.height)
        prompt = parsed.prompt
        preset = p.get("preset", preset)
        width = p.get("width", width)
        height = p.get("height", height)
        seed = p.get("seed", seed)
        negative_prompt = p.get("negative_prompt", negative_prompt)
        model = p.get("model", model)
        variant = p.get("variant", variant)
        steps = p.get("steps", steps)
        cfg_scale = p.get("cfg_scale", cfg_scale)
        sampler = p.get("sampler", sampler)
        threads = p.get("threads", threads)
        vae_tiling = p.get("vae_tiling", vae_tiling)
        loras = self._resolve_loras(p.get("loras", loras))
        if p.get("count", 1) > 1 and not _defer:
            # --n: a batch of seeds for one prompt. The first job is returned; the others follow
            # it in the queue and all are listed by generate_many.
            return self.generate_many(
                parsed_text,
                count=p["count"],
                preset=preset,
                width=width,
                height=height,
                seed=seed,
                negative_prompt=negative_prompt,
                model=model,
                variant=variant,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                threads=threads,
                vae_tiling=vae_tiling,
                loras=loras,
            )[0]
        if "init" in p and not _defer:
            # --from in the prompt: hand over to image-to-image with the remaining settings
            return self._img2img(
                "variations",
                p["init"],
                prompt,
                p.get("strength", 0.4),
                None,
                preset=preset,
                seed=seed,
                negative_prompt=negative_prompt,
                model=model,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                loras=loras,
            )
        if not self.engine.available():
            raise StudioError(
                "the image engine (stable-diffusion.cpp) is not installed. Run "
                "`imagestudio install-engine` (downloads ~20-40 MB after confirmation)."
            )
        plan = self.plan(model, preset, width, height, variant, steps, cfg_scale, sampler)
        if not plan.installed:
            missing = self.models.missing_bytes(plan.variant)
            raise StudioError(
                f"{plan.model.display_name} [{plan.variant.name}] is not downloaded "
                f"({missing / GIB:.1f} GiB). Run `imagestudio download-model {plan.model.name}` "
                f"or call download_model(name={plan.model.name!r}, consent=true)."
            )
        if threads is not None and not 1 <= threads <= 256:
            raise StudioError(f"threads must be between 1 and 256, got {threads}")
        if seed is None or seed < 0:
            seed = secrets.randbelow(MAX_SEED)
        paths = {
            flag: str(self.models.path_for(f))
            for flag, prefix in plan.model.components.items()
            for f in plan.variant.files
            if f.filename.startswith(prefix)
        }
        if len(paths) != len(plan.model.components):
            raise StudioError(f"internal: component files for {plan.model.name} incomplete")
        job = Job(
            id=self.jobs.new_id(),
            project=self.current_project(),
            tool="generate",
            style=(style or "").strip()[:80],
            style_text=(style_text or "").strip()[:500],
            prompt=prompt.strip(),
            negative_prompt=negative_prompt or "",
            model=plan.model.name,
            variant=plan.variant.name,
            preset=plan.preset.name,
            width=plan.width,
            height=plan.height,
            steps=plan.steps,
            cfg_scale=plan.cfg_scale,
            seed=seed,
            sampler=plan.sampler,
            component_paths=paths,
            extra_args=list(plan.preset.extra_args)
            + (["--mmap"] if plan.low_memory else [])
            + (["--vae-tiling"] if vae_tiling else [])
            + (["--lora-model-dir", str(self.settings.loras_dir)] if loras else []),
            loras=loras,
            threads=threads,
            over_budget=plan.over_budget,
            live_preview=bool(self.settings.config().get("live_preview", True)),
            estimate_seconds=plan.estimate.seconds,
            estimate_basis=plan.estimate.basis,
            fit_note=plan.fit_note,
            warnings=list(plan.warnings),
        )
        if not _defer:
            self.worker.submit(job)
        return job

    def generate_many(
        self, prompt: str, count: int = 1, seed: int | None = None, **kw
    ) -> list[Job]:
        """``count`` jobs of one prompt with consecutive seeds (random start when no seed is
        given). Each is a normal job in the queue; the list is in queue order."""
        count = max(1, min(16, int(count)))
        parsed = parse(prompt)
        count = max(count, int(parsed.params.get("count", 1)))
        if parsed.params.get("seed") is not None:
            seed = parsed.params["seed"]
        if count == 1:
            return [self.generate(prompt, seed=seed, **kw)]
        if seed is None or seed < 0:
            seed = secrets.randbelow(MAX_SEED - count)
        text = without_batch_and_seed(prompt)
        return [self.generate(text, seed=seed + i, **kw) for i in range(count)]

    # -- image-to-image: variations and edit ----------------------------------------------

    def _img2img(
        self,
        tool: str,
        image: str,
        prompt: str,
        strength: float,
        mask: str | None = None,
        preset: str = "best",
        seed: int | None = None,
        negative_prompt: str = "",
        model: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
        loras: list | None = None,
    ) -> Job:
        if not 0.0 <= strength <= 1.0:
            raise StudioError("strength must be between 0 and 1")
        src, source_job = self._resolve_input(image)
        mask_path: Path | None = None
        if mask:
            mask_path, _ = self._resolve_input(mask)
        from PIL import Image

        with Image.open(src) as im:
            w, h = im.size
        # keep the source size, snapped to the engine's multiple of 16
        w, h = max(256, (w // 16) * 16), max(256, (h // 16) * 16)
        if source_job and not prompt.strip():
            prompt = self.jobs.get(source_job).prompt  # type: ignore[union-attr]
        if not prompt.strip():
            raise StudioError("edit and variations need a prompt (what the result should show)")
        # Instruction edits without a mask go to an installed edit model (Qwen-Image-Edit) unless
        # the caller named a model. Masks and variations stay on image-to-image.
        if tool == "edit" and mask_path is None and not (model and model != "auto"):
            chosen = self.edit_model()
            if chosen is not None:
                model = chosen.name
        edit_role = bool(model) and getattr(MODELS.get(model), "role", "") == "edit"
        if edit_role and mask_path is not None:
            raise StudioError(
                f"{MODELS[model].display_name} edits by instruction and does not take a mask; "
                "leave the model on auto for masked edits"
            )
        if edit_role and tool != "edit":
            raise StudioError(f"{MODELS[model].display_name} is an edit model; use edit()")
        job = self.generate(
            prompt,
            preset=preset,
            width=w,
            height=h,
            seed=seed,
            negative_prompt=negative_prompt,
            model=model,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            loras=loras,
            _defer=True,
        )
        job.tool = tool
        job.input_path = str(src)
        job.source_job = source_job
        job.mask_path = str(mask_path) if mask_path else None
        job.strength = 1.0 if edit_role else strength
        if not edit_role:
            job.estimate_seconds *= max(0.15, strength)  # fewer effective steps at low strength
        self.worker.submit(job)
        return job

    def _cap_bytes(self) -> int | None:
        gib = self.settings.max_memory_gib
        return int(gib * GIB) if gib else None

    def edit_model(self) -> ModelSpec | None:
        """The strongest installed edit model that fits this machine (or any installed one
        when over-budget runs are unlocked); None → image-to-image. A 20B editor that only runs
        while swapping is not offered as the default path: measured at an hour per step."""
        for m in edit_models():
            installed = self.models.installed_variants(m)
            if not installed:
                continue
            if self.planner.allow_over_budget:
                return m
            fits = fit_variants(m, self.hardware, 512, 512, self.planner.cap_bytes)
            names = {v.name for v in installed}
            if any(f.fits and f.variant.name in names for f in fits):
                return m
        return None

    def edit_model_note(self) -> str | None:
        """Why Edit uses image-to-image on this machine, if an edit model is installed but
        does not fit."""
        for m in edit_models():
            installed = self.models.installed_variants(m)
            if not installed or self.edit_model() is m:
                continue
            fits = fit_variants(m, self.hardware, 512, 512, self.planner.cap_bytes)
            smallest = min(
                f.required_bytes for f in fits if f.variant.name in {v.name for v in installed}
            )
            return (
                f"{m.display_name} is installed but needs about {smallest / GIB:.1f} GiB even at "
                f"512 px; this machine's budget is {fits[0].budget_bytes / GIB:.1f} GiB. Edit uses "
                "image-to-image here. Measured on a 16 GB laptop: about an hour per step while "
                "swapping."
            )
        return None

    def variations(self, image: str, strength: float = 0.4, prompt: str = "", **kw) -> Job:
        """Same picture, re-imagined: image-to-image at a moderate strength.

        On 4-step models (FLUX schnell) strength is a cliff, not a slope: up to about 0.45 the
        composition is kept, from about 0.5 it is re-invented from the prompt (measured)."""
        return self._img2img("variations", image, prompt, strength, None, **kw)

    def edit(
        self, image: str, prompt: str, strength: float = 0.45, mask: str | None = None, **kw
    ) -> Job:
        """Change a picture by instruction; with a mask only the white area is repainted."""
        return self._img2img("edit", image, prompt, strength, mask, **kw)

    # -- your own models ----------------------------------------------------------------------

    def custom_models(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.settings.config().get("custom_models") or []]

    def custom_families(self) -> dict[str, Any]:
        """The families a custom model can declare, with the files each needs and the
        studio's own copies of shared encoders (so a FLUX finetune can reuse them)."""
        out = {}
        for key, fam in CUSTOM_FAMILIES.items():
            shared: dict[str, str] = {}
            for flag, prefix in fam["defaults"].items():
                for m in MODELS.values():
                    if m.tier != "verified":
                        continue
                    for v in m.ranked_variants:
                        for f in v.files:
                            if f.filename.startswith(prefix) and self.models.file_installed(f):
                                shared.setdefault(flag, str(self.models.path_for(f)))
            out[key] = {"label": fam["label"], "components": fam["components"], "shared": shared}
        return out

    def add_custom_model(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Validate, store in config.json and register. Replaces an entry with the same name."""
        try:
            spec = build_custom_model(entry)
        except (ValueError, TypeError) as e:
            raise StudioError(str(e)) from e
        cfg = self.settings.config()
        entries = [e for e in cfg.get("custom_models") or [] if e.get("name") != spec.name]
        clean = {
            "name": spec.name,
            "display_name": spec.display_name,
            "family": spec.family,
            "licence": spec.licence,
            "files": {k: str(v) for k, v in dict(entry.get("files") or {}).items() if v},
            "note": spec.note,
        }
        entries.append(clean)
        cfg["custom_models"] = entries
        self.settings.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.custom_model_problems = register_custom_models(entries)
        return self.model_report(MODELS[spec.name])

    def remove_custom_model(self, name: str) -> None:
        """Forget the entry; the user's files are never deleted."""
        if any(j.model == name and j.status in ("queued", "running") for j in self.jobs.all()):
            raise StudioError(f"{name} is used by a queued or running job")
        cfg = self.settings.config()
        entries = cfg.get("custom_models") or []
        if not any(e.get("name") == name for e in entries):
            raise StudioError(f"no custom model named {name!r}")
        cfg["custom_models"] = [e for e in entries if e.get("name") != name]
        self.settings.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.custom_model_problems = register_custom_models(cfg["custom_models"])

    # -- LoRA: user-supplied adapters in <models>/loras ---------------------------------------

    LORA_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt")

    def loras(self) -> list[dict[str, Any]]:
        """The LoRA files available by name (file name without extension)."""
        d = self.settings.loras_dir
        out = []
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in self.LORA_EXTS:
                    out.append(
                        {
                            "name": p.stem,
                            "filename": p.name,
                            "bytes": p.stat().st_size,
                            "path": str(p),
                        }
                    )
        return out

    def _resolve_loras(self, loras: list | None) -> list[list]:
        if not loras:
            return []
        have = {x["name"]: x for x in self.loras()}
        out: list[list] = []
        for item in loras:
            if isinstance(item, str):
                name, _, w = item.rpartition(":") if ":" in item else (item, "", "")
                weight = float(w) if w else 1.0
            else:
                name, weight = item[0], float(item[1]) if len(item) > 1 else 1.0
            name = str(name).strip()
            for ext in self.LORA_EXTS:
                if name.lower().endswith(ext):
                    name = name[: -len(ext)]
            if name not in have:
                raise StudioError(
                    f"no LoRA named {name!r} in {self.settings.loras_dir}"
                    + (
                        f"; available: {', '.join(sorted(have))}"
                        if have
                        else "; add one on Settings"
                    )
                )
            if not -4.0 <= weight <= 4.0:
                raise StudioError(f"LoRA weight must be between -4 and 4, got {weight:g}")
            out.append([name, weight])
        return out

    def import_lora(self, data: bytes, filename: str) -> dict[str, Any]:
        """Copy an adapter file into the loras folder. Any LoRA trained for the active base
        model (FLUX.1 or Qwen-Image) works; the studio cannot check compatibility up front,
        a mismatch shows as an engine error on the first job."""
        name = Path(filename).name
        if Path(name).suffix.lower() not in self.LORA_EXTS:
            raise StudioError(f"a LoRA file ends with one of {', '.join(self.LORA_EXTS)}")
        if not data:
            raise StudioError("empty file")
        self.settings.loras_dir.mkdir(parents=True, exist_ok=True)
        dest = self.settings.loras_dir / name
        dest.write_bytes(data)
        return {"name": dest.stem, "filename": name, "bytes": len(data), "path": str(dest)}

    def remove_lora(self, name: str) -> int:
        active = [j for j in self.jobs.all() if j.status in ("queued", "running")]
        if any(any(n == name for n, _ in j.loras) for j in active):
            raise StudioError(f"LoRA {name!r} is used by a queued or running job")
        freed = 0
        for x in self.loras():
            if x["name"] == name:
                p = Path(x["path"])
                freed += p.stat().st_size
                p.unlink()
        if not freed:
            raise StudioError(f"no LoRA named {name!r}")
        return freed

    def save_upload(self, data: bytes, name: str = "upload.png") -> Path:
        """Store an uploaded image (sketch, photo) under the data dir and return its path."""
        import io

        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                fmt = (im.format or "PNG").lower()
        except Exception as e:
            raise StudioError(f"not an image: {e}") from e
        ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(fmt, ".png")
        folder = self.settings.data_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"up_{secrets.token_hex(6)}{ext}"
        p.write_bytes(data)
        return p

    # -- components: paths, remove, reveal --------------------------------------------------

    def components(self) -> list[dict[str, Any]]:
        """Every installable thing with where it lives and how big it is, for Settings."""
        out: list[dict[str, Any]] = []
        for m in MODELS.values():
            for v in m.variants:
                files = [self.models.file_info(f) for f in v.files]
                out.append(
                    {
                        "kind": "model",
                        "name": f"{m.name}:{v.name}",
                        "display_name": f"{m.display_name} [{v.name}]",
                        "licence": m.licence,
                        "installed": all(f["installed"] for f in files),
                        "bytes": sum(f["bytes"] for f in files),
                        "files": files,
                        "folder": str(self.settings.models_dir),
                    }
                )
        for t in TOOL_MODELS.values():
            fi = self.models.file_info(t.file)
            out.append(
                {
                    "kind": "tool",
                    "name": t.name,
                    "display_name": t.display_name,
                    "licence": t.licence,
                    "installed": fi["installed"],
                    "bytes": fi["bytes"],
                    "files": [fi],
                    "folder": str(self.settings.models_dir),
                }
            )
        for sm in SEARCH_MODELS.values():
            files = [self.models.file_info(f) for f in sm.files]
            out.append(
                {
                    "kind": "search",
                    "name": sm.name,
                    "display_name": sm.display_name,
                    "licence": sm.licence,
                    "installed": all(f["installed"] for f in files),
                    "bytes": sum(f["bytes"] for f in files),
                    "files": files,
                    "folder": str(self.settings.models_dir),
                }
            )
        for vm in VISION_MODELS.values():
            files = [self.models.file_info(f) for f in vm.files]
            out.append(
                {
                    "kind": "vision",
                    "name": vm.name,
                    "display_name": vm.display_name,
                    "licence": vm.licence,
                    "installed": all(f["installed"] for f in files),
                    "bytes": sum(f["bytes"] for f in files),
                    "files": files,
                    "folder": str(self.settings.models_dir),
                }
            )
        engines: list[tuple[str, Any, str, bool]] = []
        if isinstance(self.engine, SdCppEngine):
            for b in self.engine_backends():
                if not b["supported"]:
                    continue
                eng = SdCppEngine(self.settings, flavor=b["flavor"])
                engines.append(
                    (
                        f"image:{b['flavor']}",
                        eng,
                        f"stable-diffusion.cpp · {b['label']}",
                        b["active"],
                    )
                )
        else:
            engines.append(("image", self.engine, "stable-diffusion.cpp", True))
        engines.append(("vision", self.describe_engine, "llama.cpp", False))
        for which, eng, label, active in engines:
            d = getattr(eng, "install_dir", None)
            size = (
                sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                if d and d.exists()
                else 0
            )
            asset = getattr(eng, "asset", lambda: None)()
            installed = (
                (d / asset.exe).exists() if d and asset and which != "vision" else eng.available()
            )
            out.append(
                {
                    "kind": "engine",
                    "name": which,
                    "display_name": label,
                    "licence": "MIT",
                    "installed": installed,
                    "active": active,
                    "bytes": size,
                    "files": [],
                    "folder": str(d) if d else "",
                }
            )
        return out

    def _component_in_use(self, kind: str, name: str) -> bool:
        active = [j for j in self.jobs.all() if j.status in ("queued", "running")]
        if kind == "model":
            model, _, variant = name.partition(":")
            return any(j.model == model and j.variant == variant for j in active)
        if kind in ("tool", "vision"):
            return any(j.tool_model == name for j in active)
        if kind == "engine":
            if name == "vision":
                return False
            _, _, flavor = name.partition(":")
            is_active = not flavor or flavor == getattr(self.engine, "flavor", "cpu")
            return bool(active) and is_active
        return False

    def remove_component(self, kind: str, name: str) -> dict[str, Any]:
        """Delete a model, tool model, vision model or the studio's engine copy."""
        if self._component_in_use(kind, name):
            raise StudioError(f"{name} is used by a queued or running job; wait for it to finish")
        if kind == "model":
            model, _, variant = name.partition(":")
            m = MODELS.get(model)
            if m is None:
                raise StudioError(f"unknown model {model!r}")
            v = m.variant(variant)
            shared = {
                f.filename
                for other in m.variants
                if other is not v
                for f in other.files
                if self.models.variant_installed(other)
            }
            files = [
                f for f in v.files if f.filename not in shared
            ]  # keep files other sets still use
            freed = self.models.remove_files(files)
        elif kind == "tool":
            t = TOOL_MODELS.get(name)
            if t is None:
                raise StudioError(f"unknown tool model {name!r}")
            freed = self.models.remove_files([t.file])
        elif kind == "vision":
            vm = VISION_MODELS.get(name)
            if vm is None:
                raise StudioError(f"unknown vision model {name!r}")
            freed = self.models.remove_files(vm.files)
        elif kind == "search":
            sm = SEARCH_MODELS.get(name)
            if sm is None:
                raise StudioError(f"unknown search model {name!r}")
            freed = self.models.remove_files(sm.files)
            with contextlib.suppress(OSError):
                (self.settings.data_dir / "search-index.npz").unlink()
            self._search_index = None
        elif kind == "engine":
            eng = self._engine_for(name)
            if (
                eng is self.engine
                and isinstance(self.engine, SdCppEngine)
                and os.environ.get(ENV_SD_FLAVOR)
            ):
                raise StudioError(f"this build is pinned by {ENV_SD_FLAVOR}; cannot remove it")
            un = getattr(eng, "uninstall", None)
            if un is None:
                raise StudioError("this engine cannot be removed from here")
            freed = un()
        else:
            raise StudioError(f"unknown component kind {kind!r}")
        return {"removed": name, "freed_bytes": freed}

    def reveal(self, path: str) -> None:
        """Open the folder of ``path`` in the OS file manager (only inside the data dir)."""
        import subprocess
        import sys

        p = Path(path).resolve()
        root = self.settings.data_dir.resolve()
        if root not in p.parents and p != root:
            raise StudioError("only paths inside the studio data folder can be opened")
        target = p if p.is_dir() else p.parent
        if sys.platform == "win32":
            if p.is_file():
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["open", "-R" if p.is_file() else "", str(p)]
                if p.is_file()
                else ["open", str(target)]
            )
        else:
            subprocess.Popen(["xdg-open", str(target)])

    # -- images: delete, favourite, import, lineage -----------------------------------------

    def delete_job(self, job_id: str) -> dict[str, Any]:
        """Soft delete: the image moves to trash/ and the job is hidden. Running jobs are
        cancelled first."""
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        if job.status in ("queued", "running"):
            self.cancel(job_id)
        trash = self.settings.data_dir / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        if job.output_path and Path(job.output_path).exists():
            dest = trash / Path(job.output_path).name
            Path(job.output_path).replace(dest)
            job.output_path = str(dest)
        from datetime import UTC, datetime

        job.deleted_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.jobs.put(job)
        with contextlib.suppress(Exception):
            self.unindex_job(job_id)
        return {"deleted": job_id, "trash": str(trash)}

    def restore_job(self, job_id: str) -> dict[str, Any]:
        job = next((j for j in self.jobs.all(include_deleted=True) if j.id == job_id), None)
        if job is None or not job.deleted_at:
            raise StudioError(f"{job_id!r} is not in the trash")
        if job.output_path and Path(job.output_path).exists():
            dest = self.settings.out_dir / Path(job.output_path).name
            Path(job.output_path).replace(dest)
            job.output_path = str(dest)
        job.deleted_at = None
        self.jobs.put(job)
        return {"restored": job_id}

    def empty_trash(self, older_than_days: int = 0) -> int:
        from datetime import UTC, datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        n = 0
        for j in self.jobs.all(include_deleted=True):
            if j.deleted_at and datetime.fromisoformat(j.deleted_at) <= cutoff:
                if j.output_path and Path(j.output_path).exists():
                    Path(j.output_path).unlink()
                (self.settings.jobs_dir / f"{j.id}.json").unlink(missing_ok=True)
                with self.jobs._lock:
                    self.jobs._jobs.pop(j.id, None)
                n += 1
        return n

    # -- semantic search: CLIP embeddings of every finished image ---------------------------

    def search_model(self):
        from imagestudio.registry import default_search_model

        return default_search_model()

    def search_ready(self) -> bool:
        return self.search_engine.available() and self.models.vision_installed(self.search_model())

    def search_report(self) -> dict[str, Any]:
        sm = self.search_model()
        idx = self._index() if self.search_ready() else None
        done = [j for j in self.jobs.all() if j.status == "done" and j.output_path]
        return {
            "model": sm.name,
            "display_name": sm.display_name,
            "licence": sm.licence,
            "size_mb": round(sm.download_bytes / 1e6),
            "note": sm.note,
            "installed": self.models.vision_installed(sm),
            "engine_available": self.search_engine.available(),
            "ready": self.search_ready(),
            "indexed": sum(1 for j in done if idx is not None and j.id in idx),
            "total": len(done),
            "indexing": any(
                t.kind == "index" and t.status == "running" for t in self.tasks.values()
            ),
        }

    def _index(self):
        if self._search_index is None:
            from imagestudio.engine.clip import EmbeddingIndex

            self._search_index = EmbeddingIndex(self.settings.data_dir / "search-index.npz")
        return self._search_index

    def _paths(self, sm) -> dict[str, Path]:
        return {
            k: self.models.path_for(getattr(sm, k)) for k in ("vision", "text", "vocab", "merges")
        }

    def index_job(self, job: Job) -> bool:
        """Embed one finished image into the search index (no-op when not ready or done)."""
        if not self.search_ready() or job.status != "done" or not job.output_path:
            return False
        path = Path(job.output_path)
        if not path.exists():
            return False
        sm = self.search_model()
        with self._index_lock:
            idx = self._index()
            if job.id in idx:
                return False
            vec = self.search_engine.embed_image(path, self._paths(sm)["vision"])
            idx.add(job.id, vec)
            idx.save()
        return True

    def _index_later(self, job: Job) -> None:
        if not self.search_ready():
            return

        def run() -> None:
            with contextlib.suppress(Exception):  # never disturb the job that finished
                self.index_job(job)

        threading.Thread(target=run, name=f"index-{job.id}", daemon=True).start()

    def unindex_job(self, job_id: str) -> None:
        if (
            self._search_index is None
            and not (self.settings.data_dir / "search-index.npz").exists()
        ):
            return
        with self._index_lock:
            idx = self._index()
            if job_id in idx:
                idx.remove(job_id)
                idx.save()

    def reindex(self) -> BackgroundTask:
        """Embed every finished image that is not in the index yet, in the background."""
        if not self.search_ready():
            raise StudioError(self._search_missing())
        key = "index"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        task = BackgroundTask("index", "Indexing the gallery for search")
        self.tasks[key] = task

        def run() -> None:
            try:
                idx = self._index()
                todo = [
                    j
                    for j in self.jobs.all()
                    if j.status == "done" and j.output_path and j.id not in idx
                ]
                task.total_bytes = len(todo)
                for n, j in enumerate(todo, 1):
                    self.index_job(j)
                    task.done_bytes = n
                    task.message = f"{n} of {len(todo)} images"
                task.status, task.message = "done", f"{len(todo)} images indexed"
            except Exception as e:
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name="search-index", daemon=True).start()
        return task

    def _search_missing(self) -> str:
        sm = self.search_model()
        if not self.search_engine.available():
            return "semantic search needs onnxruntime: pip install 'hivemind-imagestudio[bg]'"
        return (
            f"semantic search needs {sm.display_name} ({sm.download_bytes / 1e6:.0f} MB, "
            f"{sm.licence}); install it from Settings → Models and engines or "
            f"`imagestudio download-model {sm.name}`"
        )

    def search(self, query: str, limit: int = 60) -> dict[str, Any]:
        """Rank the gallery by how well each image matches ``query``. Images not indexed yet
        are being indexed in the background; the reply says how many are still pending."""
        query = " ".join((query or "").split())
        if not query:
            raise StudioError("search needs a few words")
        if not self.search_ready():
            raise StudioError(self._search_missing())
        sm = self.search_model()
        paths = self._paths(sm)
        idx = self._index()
        done = [j for j in self.jobs.all() if j.status == "done" and j.output_path]
        pending = sum(1 for j in done if j.id not in idx)
        if pending:
            with contextlib.suppress(StudioError):
                self.reindex()
        vec = self.search_engine.embed_text(query, paths["text"], paths["vocab"], paths["merges"])
        live = {j.id for j in done}
        results = [
            {"id": jid, "score": round(score, 4)}
            for jid, score in idx.search(vec, limit=max(limit, 1) + pending)
            if jid in live
        ][:limit]
        return {
            "query": query,
            "results": results,
            "indexed": len(done) - pending,
            "total": len(done),
            "pending": pending,
        }

    def duplicates(self, threshold: float = 0.96) -> dict[str, Any]:
        """Groups of near-identical finished images, by embedding similarity."""
        if not self.search_ready():
            raise StudioError(self._search_missing())
        idx = self._index()
        live = {j.id for j in self.jobs.all() if j.status == "done" and j.output_path}
        groups = [[i for i in g if i in live] for g in idx.duplicates(threshold)]
        groups = [g for g in groups if len(g) > 1]
        return {"groups": groups, "threshold": threshold, "indexed": len(idx)}

    # -- styles: saved prompt wording ---------------------------------------------------------
    #
    # styles.json in the data dir: [{id, name, text, negative, created_at}]. The page appends a
    # style's text to the prompt; the job and its PNG record the name and the text, so an image
    # from another studio can offer its style for saving here.

    def styles(self) -> list[dict[str, Any]]:
        samples: dict[str, str] = {}
        for j in self.jobs.all():
            if j.style and j.status == "done" and j.style not in samples:
                samples[j.style] = j.id
        return [{**s, "sample_job": samples.get(s["name"])} for s in self._read_styles()]

    def add_style(self, name: str, text: str, negative: str = "") -> dict[str, Any]:
        name = " ".join((name or "").split())[:80]
        text = " ".join((text or "").split())[:500]
        if not name:
            raise StudioError("a style needs a name")
        if not text:
            raise StudioError("a style needs its wording (what gets added to the prompt)")
        items = self._read_styles()
        if any(s["name"].lower() == name.lower() for s in items):
            raise StudioError(f"a style called {name!r} already exists")
        entry = {
            "id": "sty_" + secrets.token_hex(4),
            "name": name,
            "text": text,
            "negative": " ".join((negative or "").split())[:300],
            "created_at": now_iso(),
        }
        items.append(entry)
        self._write_styles(items)
        return {**entry, "sample_job": None}

    def update_style(
        self,
        style_id: str,
        name: str | None = None,
        text: str | None = None,
        negative: str | None = None,
    ) -> dict[str, Any]:
        items = self._read_styles()
        entry = next((s for s in items if s["id"] == style_id), None)
        if entry is None:
            raise StudioError(f"no such style {style_id!r}")
        if name is not None:
            name = " ".join(name.split())[:80]
            if not name:
                raise StudioError("a style needs a name")
            if any(s["name"].lower() == name.lower() and s is not entry for s in items):
                raise StudioError(f"a style called {name!r} already exists")
            entry["name"] = name
        if text is not None:
            text = " ".join(text.split())[:500]
            if not text:
                raise StudioError("a style needs its wording")
            entry["text"] = text
        if negative is not None:
            entry["negative"] = " ".join(negative.split())[:300]
        self._write_styles(items)
        return entry

    def remove_style(self, style_id: str) -> dict[str, Any]:
        items = self._read_styles()
        if not any(s["id"] == style_id for s in items):
            raise StudioError(f"no such style {style_id!r}")
        self._write_styles([s for s in items if s["id"] != style_id])
        return {"removed": style_id}

    def export_styles(self) -> dict[str, Any]:
        return {
            "app": f"Hivemind Image Studio {__version__}",
            "kind": "imagestudio-styles",
            "styles": [
                {"name": s["name"], "text": s["text"], "negative": s.get("negative", "")}
                for s in self._read_styles()
            ],
        }

    def import_styles(self, data: Any) -> dict[str, Any]:
        """Merge a styles export (or a bare list) into the library; names already present
        are kept as they are. Returns how many were added and skipped."""
        rows = data.get("styles") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise StudioError('expected a styles export: {"styles": [{name, text}, ...]}')
        items = self._read_styles()
        have = {s["name"].lower() for s in items}
        added = skipped = 0
        for row in rows:
            if not isinstance(row, dict):
                skipped += 1
                continue
            name = " ".join(str(row.get("name") or "").split())[:80]
            text = " ".join(str(row.get("text") or "").split())[:500]
            if not name or not text or name.lower() in have:
                skipped += 1
                continue
            items.append(
                {
                    "id": "sty_" + secrets.token_hex(4),
                    "name": name,
                    "text": text,
                    "negative": " ".join(str(row.get("negative") or "").split())[:300],
                    "created_at": now_iso(),
                }
            )
            have.add(name.lower())
            added += 1
        if added:
            self._write_styles(items)
        return {"added": added, "skipped": skipped, "total": len(items)}

    def _read_styles(self) -> list[dict[str, Any]]:
        try:
            data = json.loads((self.settings.data_dir / "styles.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [
            s
            for s in data
            if isinstance(s, dict) and s.get("id") and s.get("name") and s.get("text")
        ]

    def _write_styles(self, items: list[dict[str, Any]]) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.data_dir / "styles.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        tmp.replace(path)

    # -- projects: named series of images ----------------------------------------------------
    #
    # projects.json in the data dir: [{id, name, description, created_at}]. A job carries the
    # id of the project it was made in (config "current_project" at the time), or None. The
    # gallery filters by it, PNG metadata names it, and a project exports as one zip.

    def projects(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for j in self.jobs.all():
            if j.project and j.status == "done":
                counts[j.project] = counts.get(j.project, 0) + 1
        return [{**p, "count": counts.get(p["id"], 0)} for p in self._read_projects()]

    def project(self, project_id: str) -> dict[str, Any] | None:
        return next((p for p in self._read_projects() if p["id"] == project_id), None)

    def add_project(self, name: str, description: str = "") -> dict[str, Any]:
        name = " ".join((name or "").split())
        if not name:
            raise StudioError("a project needs a name")
        if len(name) > 80:
            raise StudioError("project names are at most 80 characters")
        items = self._read_projects()
        if any(p["name"].lower() == name.lower() for p in items):
            raise StudioError(f"a project called {name!r} already exists")
        entry = {
            "id": "prj_" + secrets.token_hex(4),
            "name": name,
            "description": (description or "").strip()[:500],
            "created_at": now_iso(),
        }
        items.append(entry)
        self._write_projects(items)
        self.set_current_project(entry["id"])
        return {**entry, "count": 0}

    def update_project(
        self, project_id: str, name: str | None = None, description: str | None = None
    ) -> dict[str, Any]:
        items = self._read_projects()
        entry = next((p for p in items if p["id"] == project_id), None)
        if entry is None:
            raise StudioError(f"no such project {project_id!r}")
        if name is not None:
            name = " ".join(name.split())
            if not name:
                raise StudioError("a project needs a name")
            if any(p["name"].lower() == name.lower() and p is not entry for p in items):
                raise StudioError(f"a project called {name!r} already exists")
            entry["name"] = name[:80]
        if description is not None:
            entry["description"] = description.strip()[:500]
        self._write_projects(items)
        return entry

    def remove_project(self, project_id: str) -> dict[str, Any]:
        """Forget the project; its images stay in the gallery, unfiled."""
        items = self._read_projects()
        if not any(p["id"] == project_id for p in items):
            raise StudioError(f"no such project {project_id!r}")
        self._write_projects([p for p in items if p["id"] != project_id])
        moved = 0
        for j in self.jobs.all(include_deleted=True):
            if j.project == project_id:
                j.project = None
                self.jobs.put(j)
                moved += 1
        if self.current_project() == project_id:
            self.set_current_project(None)
        return {"removed": project_id, "unfiled": moved}

    def current_project(self) -> str | None:
        pid = self.settings.config().get("current_project")
        if not pid:
            return None
        return pid if any(p["id"] == pid for p in self._read_projects()) else None

    def set_current_project(self, project_id: str | None) -> dict[str, Any]:
        if project_id and not any(p["id"] == project_id for p in self._read_projects()):
            raise StudioError(f"no such project {project_id!r}")
        cfg = self.settings.config()
        if project_id:
            cfg["current_project"] = project_id
        else:
            cfg.pop("current_project", None)
        self.settings.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return {"current_project": project_id or None}

    def set_job_project(self, job_id: str, project_id: str | None) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        if project_id and not any(p["id"] == project_id for p in self._read_projects()):
            raise StudioError(f"no such project {project_id!r}")
        job.project = project_id or None
        self.jobs.put(job)
        return {"id": job_id, "project": job.project}

    def export_project(self, project_id: str) -> tuple[str, bytes]:
        """One zip: the finished images, prompts.txt with every image's reproducible prompt,
        and project.json. The file name is the project's name."""
        entry = self.project(project_id)
        if entry is None:
            raise StudioError(f"no such project {project_id!r}")
        jobs = [
            j
            for j in self.jobs.all()
            if j.project == project_id and j.status == "done" and j.output_path
        ]
        if not jobs:
            raise StudioError("this project has no finished images yet")
        return self._zip_of(jobs, entry["name"], entry.get("description") or "", entry)

    def export_images(self, job_ids: list[str], title: str = "selection") -> tuple[str, bytes]:
        """A zip of chosen finished images with prompts.txt, for a gallery selection."""
        jobs = []
        for jid in job_ids:
            j = self.jobs.get(str(jid))
            if j is not None and j.status == "done" and j.output_path:
                jobs.append(j)
        if not jobs:
            raise StudioError("none of the chosen images is finished")
        return self._zip_of(jobs, title, "", {"name": title})

    def _zip_of(
        self, jobs: list[Job], title: str, description: str, entry: dict[str, Any]
    ) -> tuple[str, bytes]:
        import re
        import zipfile

        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-") or "images"
        buf = io.BytesIO()
        lines = [f"# {title}", description, ""]
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, j in enumerate(jobs, 1):
                src = Path(j.output_path or "")
                if not src.exists():
                    continue
                name = f"{n:03d}-{j.id}{src.suffix.lower() or '.png'}"
                z.write(src, name)
                text = reproducible(j) if j.tool in ("generate", "variations", "edit") else ""
                lines.append(f"{name}  ({j.tool}, {j.model} [{j.variant}], seed {j.seed})")
                lines.append(text or j.prompt or j.caption or "")
                lines.append("")
            z.writestr("prompts.txt", "\n".join(lines))
            z.writestr(
                "project.json",
                json.dumps(
                    {
                        **entry,
                        "app": f"Hivemind Image Studio {__version__}",
                        "images": [j.id for j in jobs],
                    },
                    indent=2,
                ),
            )
        return f"{slug}.zip", buf.getvalue()

    def _read_projects(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(
                (self.settings.data_dir / "projects.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return []
        return [p for p in data if isinstance(p, dict) and p.get("id") and p.get("name")]

    def _write_projects(self, items: list[dict[str, Any]]) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.data_dir / "projects.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        tmp.replace(path)

    def set_favorite(self, job_id: str, value: bool) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        job.favorite = bool(value)
        self.jobs.put(job)
        return {"id": job_id, "favorite": job.favorite}

    def import_image(self, data: bytes, name: str = "import") -> Job:
        """Bring an outside image into the gallery as a finished job, so every tool can use it."""
        src = self.save_upload(data)
        from PIL import Image

        with Image.open(src) as im:
            w, h = im.size
        job = Job(
            id=self.jobs.new_id(),
            project=self.current_project(),
            tool="import",
            prompt=name,
            negative_prompt="",
            model="-",
            variant="-",
            preset="import",
            width=w,
            height=h,
            steps=0,
            cfg_scale=0.0,
            seed=0,
            sampler="-",
            component_paths={},
            extra_args=[],
            estimate_seconds=0,
            estimate_basis="",
            fit_note=None,
            warnings=[],
            status="done",
            phase="done",
            input_path=str(src),
            live_preview=False,
        )
        dest = self.settings.out_dir / f"{job.id}.png"
        meta = self.read_metadata(src)
        if meta.get("prompt"):
            job.prompt = str(meta["prompt"])
            job.negative_prompt = str(meta.get("negative_prompt") or "")
            job.style = str(meta.get("style") or "")[:80]
            job.style_text = str(meta.get("style_text") or "")[:500]
            if meta.get("caption"):
                job.caption = str(meta["caption"])
            for k in ("model", "variant", "preset", "sampler"):
                if meta.get(k):
                    setattr(job, k, str(meta[k]))
            for k in ("steps", "seed"):
                if isinstance(meta.get(k), int):
                    setattr(job, k, meta[k])
            if isinstance(meta.get("cfg_scale"), (int, float)):
                job.cfg_scale = float(meta["cfg_scale"])
        with Image.open(src) as im:
            im.save(dest, "PNG")
        from datetime import UTC, datetime

        job.output_path = str(dest)
        job.finished_at = job.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        job.total_seconds = 0.0
        self.jobs.put(job)
        self._index_later(job)
        self._maybe_auto_caption(job)
        return job

    def lineage(self, job_id: str) -> dict[str, Any]:
        jobs = self.jobs.all()
        job = next((j for j in jobs if j.id == job_id), None)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        parent = job.source_job
        children = [j.id for j in jobs if j.source_job == job_id]
        return {"id": job_id, "parent": parent, "children": children}

    # -- describe: image to text ----------------------------------------------------------

    def describe(
        self,
        image: str,
        task: str = "describe",
        question: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous: a few seconds on a CPU with the 3B model. Returns {text, seconds, model}."""
        from imagestudio.engine.describe import PROMPTS, DescribeSpec

        if task not in PROMPTS and task != "custom":
            raise StudioError(f"unknown task {task!r}; one of {', '.join(PROMPTS)}, custom")
        if task == "custom" and not (question or "").strip():
            raise StudioError("a custom task needs a question")
        vm = VISION_MODELS.get(model) if model else default_vision_model()
        if vm is None:
            raise StudioError(f"unknown vision model {model!r}; one of {', '.join(VISION_MODELS)}")
        if not self.describe_engine.available():
            raise StudioError(
                "the vision engine (llama.cpp) is not installed. Run `imagestudio install-engine "
                "--vision` (about 20 MB after confirmation)."
            )
        if not self.models.vision_installed(vm):
            raise StudioError(
                f"{vm.display_name} is not downloaded ({vm.download_bytes / GIB:.1f} GiB). Call "
                f"download_model(name={vm.name!r}, consent=true) or "
                f"`imagestudio download-model {vm.name}`."
            )
        src, source_job = self._resolve_input(image)
        from imagestudio.engine.base import EngineError

        try:
            res = self.describe_engine.describe(
                DescribeSpec(
                    image=src,
                    model_path=self.models.path_for(vm.model),
                    mmproj_path=self.models.path_for(vm.mmproj),
                    task=task,
                    question=question,
                )
            )
        except EngineError as e:
            msg = str(e)
            if "Access is denied" in msg or "Permission denied" in msg:
                msg += (
                    ". On Windows this is usually endpoint protection (Trend Micro, Defender) "
                    "blocking a newly downloaded program: allow it or trust the studio's tools "
                    "folder."
                )
            raise StudioError(msg) from e
        if source_job and task in ("describe", "alt"):
            job = self.jobs.get(source_job)
            if (job is not None and job.id in self.jobs.owned) or (
                job is not None and self.worker.active
            ):
                job.caption = res.text
                self.jobs.put(job)
        return {
            "text": res.text,
            "seconds": round(res.seconds, 1),
            "model": vm.name,
            "task": task,
            "source_job": source_job,
        }

    # -- prompt assistant: brief to prompt, with the same local language model --------------

    def assist_prompt(
        self,
        brief: str,
        current_prompt: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
    ) -> dict[str, Any]:
        """Turn a plain-words brief into a prompt and negative prompt for the model that will
        paint it (``model``: catalogue name or None for the planner's choice). With
        ``current_prompt`` the assistant improves that prompt in the direction of the brief
        instead of starting over. Synchronous. Nothing leaves the
        machine; the assistant is the local vision-language model running text-only. A minute
        or more on a CPU, longer when memory is tight (the model is 2.5 GiB)."""
        import json as _json
        import re as _re

        from imagestudio.engine.base import EngineError
        from imagestudio.engine.describe import TextSpec

        brief = (brief or "").strip()
        if not brief and not (current_prompt or "").strip():
            raise StudioError("tell the assistant what the image should show")
        vm = VISION_MODELS.get(vision_model) if vision_model else default_vision_model()
        if vm is None:
            raise StudioError(
                f"unknown vision model {vision_model!r}; one of {', '.join(VISION_MODELS)}"
            )
        if not self.describe_engine.available():
            raise StudioError(
                "the prompt assistant needs the vision engine (llama.cpp). Install it from "
                "Settings → Models and engines, or `imagestudio install-engine --vision` "
                "(about 20 MB)."
            )
        if not self.models.vision_installed(vm):
            raise StudioError(
                f"the prompt assistant uses {vm.display_name}, which is not downloaded "
                f"({vm.download_bytes / GIB:.1f} GiB). Install it from Settings → Models and "
                f"engines, or `imagestudio download-model {vm.name}`."
            )
        if self.describe_engine.text_binary() is None:
            raise StudioError(
                "this llama.cpp install has no text runner (llama-completion); reinstall the "
                "vision engine from Settings → Models and engines."
            )
        target = self._assist_target(model)
        system = _ASSIST_SYSTEM.format(**target)
        if (current_prompt or "").strip():
            user = f"Current prompt:\n{current_prompt.strip()}\n\n" + (
                f"Change requested: {brief}" if brief else "Improve this prompt."
            )
        else:
            user = f"Brief: {brief}"
        try:
            res = self.describe_engine.complete(
                TextSpec(
                    model_path=self.models.path_for(vm.model),
                    system=system,
                    user=user,
                    threads=self.settings.config().get("default_threads") or None,
                )
            )
        except EngineError as e:
            raise StudioError(str(e)) from e
        text = res.text.strip()
        prompt, negative, notes = text, "", ""
        m = _re.search(r"\{.*\}", text, _re.S)
        parsed = False
        if m:
            try:
                data = _json.loads(m.group(0))
                prompt = str(data.get("prompt") or "").strip() or prompt
                negative = str(data.get("negative") or "").strip()
                notes = str(data.get("notes") or "").strip()
                parsed = True
            except ValueError:
                parsed = False
        if not parsed:
            # the model stopped mid-JSON (token limit or a repetition loop): salvage the
            # fields that are complete, never show the user raw JSON
            pm = _re.search(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            nm = _re.search(r'"negative"\s*:\s*"((?:[^"\\]|\\.)*)', text)
            if pm:
                prompt = pm.group(1).strip()
                negative = nm.group(1).strip() if nm else ""
                notes = "the assistant ran out of words before finishing its notes"
            elif text.startswith("{"):
                pm2 = _re.search(r'"prompt"\s*:\s*"(.*)', text, _re.S)
                prompt = (pm2.group(1) if pm2 else text).rstrip('"').strip()
                negative = ""
                notes = "the assistant ran out of words; the prompt was cut short"
        # small models loop on list items: keep each negative term once, in order
        if negative:
            seen: list[str] = []
            for term in (t.strip() for t in negative.split(",")):
                if term and term.lower() not in (x.lower() for x in seen):
                    seen.append(term)
            negative = ", ".join(seen[:16])
        if target["negatives"] == "ignored":
            negative = ""
        return {
            "prompt": prompt,
            "negative_prompt": negative,
            "notes": notes,
            "target_model": target["display_name"],
            "seconds": round(res.seconds, 1),
            "model": vm.name,
        }

    def _assist_target(self, model: str | None) -> dict[str, str]:
        """What the assistant should know about the model that will paint the prompt."""
        spec = None
        try:
            if model and model != "auto":
                spec = MODELS.get(model)
                if spec is None:
                    e = next((c for c in self.custom_models() if c.get("name") == model), None)
                    if e is not None:
                        return self._assist_target_for(
                            str(e.get("family") or "flux"), str(e.get("display_name") or model)
                        )
            else:
                spec = self.plan(None, "best", None, None).model
        except (StudioError, AttributeError):
            spec = None
        fam = getattr(spec, "family", "flux") or "flux"
        name = getattr(spec, "display_name", None) or "FLUX.1-schnell"
        return self._assist_target_for(fam, name)

    @staticmethod
    def _assist_target_for(fam: str, name: str) -> dict[str, str]:
        if fam.startswith("qwen"):
            return {
                "display_name": name,
                "style": (
                    "full natural-language sentences; the model reads long descriptions well "
                    "and can render short text in quotes exactly, so if the brief wants words "
                    "in the picture, quote them"
                ),
                "negatives": "used",
            }
        if fam in ("sd15", "sdxl"):
            return {
                "display_name": name,
                "style": (
                    "comma-separated phrases, most important first, with quality words at the "
                    "end (highly detailed, sharp focus); keep it under 60 words"
                ),
                "negatives": "used",
            }
        return {
            "display_name": name,
            "style": (
                "one natural paragraph of plain descriptive sentences, 40 to 80 words; name the "
                "subject, composition, setting, lighting, medium or style, colours and mood; "
                "no weighting syntax, no artist names, no camera brands"
            ),
            "negatives": "ignored",
        }

    def vision_report(self) -> list[dict[str, Any]]:
        return [
            {
                "name": v.name,
                "display_name": v.display_name,
                "licence": v.licence,
                "size_gib": round(v.download_bytes / GIB, 2),
                "installed": self.models.vision_installed(v),
                "default": v.default,
                "note": v.note,
            }
            for v in VISION_MODELS.values()
        ]

    def start_vision_download(self, name: str) -> BackgroundTask:
        vm = VISION_MODELS.get(name) or SEARCH_MODELS.get(name)
        if vm is None:
            raise StudioError(f"unknown vision or search model {name!r}")
        key = f"download:{name}"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        task = BackgroundTask("download", vm.display_name)
        task.total_bytes = sum(f.size for f in vm.files if not self.models.file_installed(f))
        self.tasks[key] = task

        def progress(name_: str, done: int, total: int) -> None:
            task.current_file = name_
            finished = sum(f.size for f in vm.files if self.models.file_installed(f))
            task.done_bytes = min(task.total_bytes, finished + done) if task.total_bytes else done

        def run() -> None:
            try:
                self.models.install_vision(vm, lambda _d, _b: True, progress)
                task.done_bytes = task.total_bytes
                task.status, task.message = "done", "installed"
            except Exception as e:  # shown to the user
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    ENGINE_FLAVORS = ("cpu", "vulkan")

    def _engine_for(self, which: str):
        """``image`` (the active sd.cpp build), ``image:<flavor>`` (a specific build) or
        ``vision`` (llama.cpp)."""
        if which == "vision":
            return self.describe_engine
        name, _, flavor = which.partition(":")
        if name != "image":
            raise StudioError(f"unknown engine {which!r}")
        if not flavor or flavor == getattr(self.engine, "flavor", None):
            return self.engine
        if flavor not in self.ENGINE_FLAVORS:
            raise StudioError(
                f"unknown engine backend {flavor!r}; one of {', '.join(self.ENGINE_FLAVORS)}"
            )
        return SdCppEngine(self.settings, flavor=flavor)

    def engine_backends(self) -> list[dict[str, Any]]:
        """The sd.cpp builds this platform can run, which are installed, and which is active."""
        out = []
        for fl in self.ENGINE_FLAVORS:
            eng = SdCppEngine(self.settings, flavor=fl)
            asset = eng.asset()
            out.append(
                {
                    "flavor": fl,
                    "label": "CPU" if fl == "cpu" else "Vulkan (GPU)",
                    "supported": asset is not None,
                    "installed": asset is not None and (eng.install_dir / asset.exe).exists(),
                    "active": fl == getattr(self.engine, "flavor", "cpu"),
                    "bytes": asset.size if asset else 0,
                }
            )
        return out

    def engine_suggestion(self) -> dict[str, Any] | None:
        """A backend this machine would likely render faster with, or None. Never applied by
        itself: the user presses Use (quality is unchanged either way; only time differs)."""
        active = getattr(self.engine, "flavor", "cpu")
        if active != "cpu":
            return None
        backends = {b["flavor"]: b for b in self.engine_backends()}
        vk = backends.get("vulkan")
        if not vk or not vk["supported"] or not self.hardware.gpus:
            return None
        gpu = self.hardware.usable_gpu
        if gpu is not None:
            reason = (
                f"This machine has a discrete GPU ({gpu.name}). The Vulkan build renders on it; "
                "measured 2.4× faster than the CPU build even on an integrated GPU, and "
                "discrete GPUs gain more. Same model, same quality, only the time changes."
            )
            strength = "likely large"
        else:
            g = self.hardware.gpus[0]
            reason = (
                f"The Vulkan build can render on this machine's integrated GPU ({g.name}). "
                "Measured on a 16 GB laptop with Intel UHD graphics: 762 s instead of 1805 s "
                "for the same image, 2.4× faster. Same model, same quality, only the time "
                "changes; run the benchmark afterwards to recalibrate the estimates."
            )
            strength = "measured 2.4×"
        return {
            "flavor": "vulkan",
            "installed": bool(vk["installed"]),
            "action": "use" if vk["installed"] else "install",
            "reason": reason,
            "strength": strength,
            "bytes": vk["bytes"],
        }

    def try_start_worker(self) -> bool:
        """Start rendering in this process if the lock became free (after a handover)."""
        if self.worker_active:
            return True
        try:
            self.worker.start()
        except Exception:
            return False
        return self.worker_active

    def request_worker_handover(self) -> dict[str, Any]:
        """Ask the process holding the worker lock to stop once idle, so this one can take
        over (used when the rendering worker runs an older build). Refused mid-render."""
        if self.worker_active:
            return {"ok": False, "message": "the worker already runs in this process"}
        if any(j.status == "running" for j in self.jobs.all()):
            raise StudioError("a render is in progress; wait for it to finish before handing over")
        self.jobs.exit_marker().write_text("exit", encoding="utf-8")
        return {
            "ok": True,
            "message": "asked the other worker to stop; this studio takes over in a few seconds",
        }

    def set_engine_backend(self, flavor: str) -> None:
        """Switch the image engine build. Refused while jobs are queued or running."""
        flavor = (flavor or "cpu").lower()
        if flavor not in self.ENGINE_FLAVORS:
            raise StudioError(
                f"unknown engine backend {flavor!r}; one of {', '.join(self.ENGINE_FLAVORS)}"
            )
        if os.environ.get(ENV_SD_FLAVOR):
            raise StudioError(f"the engine backend is fixed by {ENV_SD_FLAVOR} in the environment")
        if not isinstance(self.engine, SdCppEngine):
            raise StudioError("the engine backend cannot be switched in this process")
        if any(j.status in ("queued", "running") for j in self.jobs.all()):
            raise StudioError("wait for the queue to finish before switching the engine backend")
        eng = SdCppEngine(self.settings, flavor=flavor)
        if eng.asset() is None:
            raise StudioError(f"no pinned {flavor} build of stable-diffusion.cpp for this platform")
        if not eng.available():
            raise StudioError(f"the {flavor} build is not installed yet; install it first")
        cfg = self.settings.config()
        cfg["engine_backend"] = flavor
        self.settings.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.engine = eng
        self.worker.engine = eng

    def start_engine_download(self, which: str) -> BackgroundTask:
        """Download the pinned sd.cpp or llama.cpp build in the background (UI consent given)."""
        engine = self._engine_for(which)
        install = getattr(engine, "install", None)
        if install is None:
            raise StudioError(f"engine {which!r} has no installer")
        key = f"engine:{which}"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        label = (
            "llama.cpp"
            if which == "vision"
            else f"stable-diffusion.cpp ({getattr(engine, 'flavor', 'cpu')})"
        )
        task = BackgroundTask("download", label)
        asset = engine.asset()
        task.total_bytes = asset.size if asset else 0
        self.tasks[key] = task

        def progress(done: int, total: int) -> None:
            task.done_bytes, task.total_bytes = done, total or task.total_bytes

        def run() -> None:
            try:
                install(lambda _d, _b: True, progress=progress)
                task.status, task.message = "done", "installed"
            except Exception as e:  # shown to the user
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    # -- task tools: upscale, remove_background -------------------------------------------

    def _tool_model_path(self, name: str | None) -> Path | None:
        t = TOOL_MODELS.get(name or "")
        if t is None or not self.models.tool_installed(t):
            return None
        return self.models.path_for(t.file)

    def _resolve_input(self, image: str) -> tuple[Path, str | None]:
        """``image`` is a job id or a path to a PNG/JPEG/WebP on this machine."""
        job = self.jobs.get(image)
        if job is not None:
            if job.status != "done" or not job.output_path:
                raise StudioError(f"job {image} has no finished image yet")
            return Path(job.output_path), job.id
        p = Path(image).expanduser()
        if not p.is_file():
            raise StudioError(f"{image!r} is neither a job id nor an existing image file")
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise StudioError("the input must be a PNG, JPEG or WebP image")
        return p, None

    def _tool_job(self, tool: str, image: str, model_name: str | None, repeats: int = 1) -> Job:
        kind = "upscaler" if tool == "upscale" else "background"
        tm = TOOL_MODELS.get(model_name) if model_name else default_tool_model(kind)
        if tm is None or tm.kind != kind:
            options = ", ".join(t.name for t in TOOL_MODELS.values() if t.kind == kind)
            raise StudioError(f"unknown {kind} model {model_name!r}; one of {options}")
        if not self.models.tool_installed(tm):
            raise StudioError(
                f"{tm.display_name} is not downloaded ({tm.file.size / GIB:.2f} GiB). Call "
                f"download_model(name={tm.name!r}, consent=true) or "
                f"`imagestudio download-model {tm.name}`."
            )
        if tool == "upscale" and not self.engine.available():
            raise StudioError("the image engine (stable-diffusion.cpp) is not installed")
        if tool == "remove_background" and not self.background_engine.available():
            raise StudioError(
                "background removal needs onnxruntime: pip install 'hivemind-imagestudio[bg]'"
            )
        src, source_job = self._resolve_input(image)
        from PIL import Image

        with Image.open(src) as im:
            w, h = im.size
        if tool == "upscale":
            w, h = w * (tm.scale**repeats), h * (tm.scale**repeats)
            estimate = 20.0 + 25.0 * (w * h) / 1_000_000  # ~25 s per output megapixel on CPU
            basis = "heuristic: about 25 s per output megapixel on a laptop CPU"
        else:
            estimate = 8.0
            basis = "heuristic: a few seconds"
        job = Job(
            id=self.jobs.new_id(),
            project=self.current_project(),
            tool=tool,
            prompt=f"{tool.replace('_', ' ')} of {source_job or src.name}",
            negative_prompt="",
            model=tm.name,
            variant="-",
            preset="tool",
            width=w,
            height=h,
            steps=repeats,
            cfg_scale=0.0,
            seed=0,
            sampler="-",
            component_paths={},
            extra_args=[],
            estimate_seconds=estimate,
            estimate_basis=basis,
            fit_note=None,
            warnings=[],
            input_path=str(src),
            source_job=source_job,
            tool_model=tm.name,
            live_preview=False,
        )
        self.worker.submit(job)
        return job

    def fix_faces(
        self,
        image: str,
        strength: float = 0.35,
        prompt: str = "",
        model: str | None = None,
        preset: str = "best",
        seed: int | None = None,
        detector: str | None = None,
        max_faces: int = 4,
    ) -> Job:
        """Re-render every face in a finished image at a higher resolution and blend it back.
        Faces are the most common reason a render is thrown away: at 1024 px a face is a few
        dozen pixels and the model cannot draw eyes there. Each face crop is scaled to the
        render size (768 px), passed through image-to-image at a low strength (0.35 keeps the
        likeness; 0.5 re-invents the face) with the source prompt plus face wording, then
        pasted back under a feathered mask. One engine run per face, so the estimate scales
        with the face count. The detector (UltraFace, MIT, 1.5 MB) is a tool model."""
        if not 0.1 <= strength <= 0.7:
            raise StudioError("strength must be between 0.1 and 0.7 (0.35 keeps the likeness)")
        kind = "face"
        tm = TOOL_MODELS.get(detector) if detector else default_tool_model(kind)
        if tm is None or tm.kind != kind:
            options = ", ".join(t.name for t in TOOL_MODELS.values() if t.kind == kind)
            raise StudioError(f"unknown face detector {detector!r}; one of {options}")
        if not self.models.tool_installed(tm):
            raise StudioError(
                f"{tm.display_name} is not downloaded ({tm.file.size / 1e6:.1f} MB). Call "
                f"download_model(name={tm.name!r}, consent=true) or "
                f"`imagestudio download-model {tm.name}`."
            )
        if not self.face_detector.available():
            raise StudioError(
                "face detection needs onnxruntime: pip install 'hivemind-imagestudio[bg]'"
            )
        if not self.engine.available():
            raise StudioError("the image engine (stable-diffusion.cpp) is not installed")
        src, source_job = self._resolve_input(image)
        from imagestudio.engine.base import EngineError
        from imagestudio.engine.faces import expand_box

        try:
            faces = self.face_detector.detect(src, self.models.path_for(tm.file))
        except EngineError as e:
            raise StudioError(str(e)) from e
        if not faces:
            raise StudioError("no faces found in this image")
        faces = faces[: max(1, min(max_faces, 8))]
        from PIL import Image

        with Image.open(src) as im:
            w, h = im.size
        regions = [list(expand_box(f, w, h)) for f in faces]
        if source_job and not prompt.strip():
            prompt = self.jobs.get(source_job).prompt  # type: ignore[union-attr]
        base_prompt = prompt.strip() or "a portrait"
        prompt = f"{base_prompt}, detailed face, sharp clear eyes, natural skin texture"
        if model and getattr(MODELS.get(model), "role", "") == "edit":
            raise StudioError(f"{MODELS[model].display_name} is an edit model; use edit()")
        size = 768 if preset == "best" else 512
        job = self.generate(
            prompt, preset=preset, width=size, height=size, seed=seed, model=model, _defer=True
        )
        job.tool = "fix_faces"
        job.input_path = str(src)
        job.source_job = source_job
        job.strength = strength
        job.regions = regions
        job.tool_model = tm.name
        job.live_preview = False
        job.render_size = size
        job.width, job.height = w, h  # the result keeps the source size
        job.estimate_seconds *= len(regions) * max(0.15, strength) * (size * size) / (1024 * 1024)
        job.estimate_basis += f"; {len(regions)} face(s) at {size} px, strength {strength:g}"
        self.worker.submit(job)
        return job

    def upscale(self, image: str, model: str | None = None, repeats: int = 1) -> Job:
        if not 1 <= repeats <= 2:
            raise StudioError("repeats must be 1 (4x) or 2 (16x)")
        return self._tool_job("upscale", image, model, repeats)

    def remove_background(self, image: str, model: str | None = None) -> Job:
        return self._tool_job("remove_background", image, model)

    def tools_report(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "kind": t.kind,
                "display_name": t.display_name,
                "licence": t.licence,
                "size_gib": round(t.file.size / GIB, 2),
                "installed": self.models.tool_installed(t),
                "default": t.default,
                "scale": t.scale,
                "note": t.note,
            }
            for t in TOOL_MODELS.values()
        ]

    def start_tool_download(self, name: str) -> BackgroundTask:
        tm = TOOL_MODELS.get(name)
        if tm is None:
            raise StudioError(f"unknown tool model {name!r}")
        key = f"download:{name}"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        task = BackgroundTask("download", tm.display_name)
        task.total_bytes = 0 if self.models.tool_installed(tm) else tm.file.size
        self.tasks[key] = task

        def progress(name_: str, done: int, total: int) -> None:
            task.current_file = name_
            task.done_bytes = done

        def run() -> None:
            try:
                self.models.install_tool(tm, lambda _d, _b: True, progress)
                task.done_bytes = task.total_bytes
                task.status, task.message = "done", "installed"
            except Exception as e:  # shown to the user
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    # -- job views ------------------------------------------------------------------------

    def job_status(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        ahead = self.worker.queue_seconds_ahead(job_id) if job.status == "queued" else 0.0
        eta = job.eta_seconds(ahead)
        position = None
        if job.status == "queued":
            queued_ids = [j.id for j in self.jobs.queued()]
            position = queued_ids.index(job_id) + 1 if job_id in queued_ids else None
        return {
            "id": job.id,
            "tool": job.tool,
            "source_job": job.source_job,
            "tool_model": job.tool_model,
            "strength": job.strength,
            "mask_path": job.mask_path,
            "caption": job.caption,
            "favorite": job.favorite,
            "project": job.project,
            "regions": job.regions,
            "style": job.style,
            "style_text": job.style_text,
            "deleted_at": job.deleted_at,
            "prompt": job.prompt,
            "reproducible_prompt": reproducible(job)
            if job.tool in ("generate", "variations", "edit")
            else "",
            "title": job_title(job),
            "status": job.status,
            "phase": job.phase,
            "progress": round(job.progress_fraction(), 3),
            "step": job.step,
            "total_steps": job.total_steps or job.steps,
            "seconds_per_step": job.seconds_per_step,
            "eta_seconds": None if eta is None else round(eta),
            "elapsed_seconds": round(job.elapsed_seconds()) if job.status == "running" else None,
            "summary": job.summary(queue_position=position, eta=eta, paused=self.jobs.is_paused()),
            "eta_human": None if eta is None else human_duration(eta),
            "estimate_basis": job.estimate_basis,
            "queue_position": position,
            "model": job.model,
            "variant": job.variant,
            "preset": job.preset,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "seed": job.seed,
            "fit_note": job.fit_note,
            "over_budget": job.over_budget,
            "warnings": job.warnings,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "output_path": job.output_path,
            "peak_rss_gib": round(job.peak_rss_bytes / GIB, 2) if job.peak_rss_bytes else None,
            "negative_prompt": job.negative_prompt,
            "cfg_scale": job.cfg_scale,
            "sampler": job.sampler,
            "threads": job.threads,
            "vae_tiling": "--vae-tiling" in job.extra_args,
            "total_seconds": job.total_seconds,
            "sampling_seconds": job.sampling_seconds,
            "has_preview": (self.settings.out_dir / job.preview_name).exists()
            if job.status == "running"
            else False,
        }

    def job_result(self, job_id: str, as_base64: bool = False) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        if job.status != "done":
            status = self.job_status(job_id)
            return {"id": job.id, "status": job.status, "ready": False, "status_detail": status}
        path = Path(job.output_path or "")
        if not path.exists():
            raise StudioError(f"job {job_id} finished but its image is missing at {path}")
        out: dict[str, Any] = {
            "id": job.id,
            "status": "done",
            "ready": True,
            "path": str(path),
            "mime_type": "image/png",
            "width": job.width,
            "height": job.height,
            "seed": job.seed,
            "model": job.model,
            "variant": job.variant,
            "preset": job.preset,
            "prompt": job.prompt,
            "total_seconds": job.total_seconds,
            "peak_rss_gib": round(job.peak_rss_bytes / GIB, 2) if job.peak_rss_bytes else None,
            "fit_note": job.fit_note,
        }
        if as_base64:
            out["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        return out

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": j.id,
                "status": j.status,
                "preset": j.preset,
                "model": j.model,
                "prompt": j.prompt[:80],
                "created_at": j.created_at,
                "finished_at": j.finished_at,
                "output_path": j.output_path,
            }
            for j in reversed(self.jobs.all())
        ]

    def cancel(self, job_id: str) -> bool:
        return self.worker.cancel(job_id)

    # -- images ---------------------------------------------------------------------------

    def preview_path(self, job_id: str) -> Path | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        p = self.settings.out_dir / job.preview_name
        return p if p.exists() else None

    EXPORT_FORMATS = {
        "png": ("PNG", "image/png", "png", False),
        "jpg": ("JPEG", "image/jpeg", "jpg", True),
        "webp": ("WEBP", "image/webp", "webp", True),
    }

    def export_image(
        self, job_id: str, fmt: str = "png", quality: int = 90, scale: int = 100
    ) -> tuple[bytes, str, str]:
        """The finished image as bytes.

        ``scale`` (10-100 percent) resizes the image and applies to every format; ``quality``
        (30-100) is the encoder quality for lossy formats and ignored for PNG. At 100/100 PNG
        the original file is returned byte for byte. Returns (data, mime_type, filename)."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "done" or not job.output_path:
            raise StudioError(f"job {job_id!r} has no finished image")
        if fmt not in self.EXPORT_FORMATS:
            raise StudioError(f"unknown format {fmt!r}; one of {', '.join(self.EXPORT_FORMATS)}")
        pil_name, mime, ext, lossy = self.EXPORT_FORMATS[fmt]
        scale = max(10, min(100, int(scale)))
        quality = max(30, min(100, int(quality)))
        path = Path(job.output_path)
        stem = f"{job.id}-{job.preset}-seed{job.seed}"
        if fmt == "png" and scale == 100:
            return path.read_bytes(), mime, f"{stem}-{job.width}x{job.height}.png"
        from PIL import Image

        buf = io.BytesIO()
        with Image.open(path) as im:
            if scale != 100:
                w = max(16, round(im.width * scale / 100))
                h = max(16, round(im.height * scale / 100))
                im = im.resize((w, h), Image.LANCZOS)
            size = im.size
            if lossy:
                im.convert("RGB").save(buf, pil_name, quality=quality, optimize=True)
            else:
                from PIL.PngImagePlugin import PngInfo

                info = PngInfo()
                for k, v in self.metadata_for(job).items():
                    info.add_text(k, v)
                im.save(buf, pil_name, optimize=True, pnginfo=info)
        suffix = f"-q{quality}" if lossy else ""
        return buf.getvalue(), mime, f"{stem}-{size[0]}x{size[1]}{suffix}.{ext}"

    def export_size(self, job_id: str, fmt: str, quality: int, scale: int) -> dict[str, Any]:
        """Bytes and pixel size an export would have, so the UI can show it first."""
        data, mime, name = self.export_image(job_id, fmt, quality, scale)
        job = self.jobs.get(job_id)
        assert job is not None
        s = max(10, min(100, int(scale)))
        return {
            "bytes": len(data),
            "filename": name,
            "mime": mime,
            "width": round(job.width * s / 100) if s != 100 else job.width,
            "height": round(job.height * s / 100) if s != 100 else job.height,
        }

    # -- diagnostics ------------------------------------------------------------------------

    def memory_users(self, limit: int = 8) -> dict[str, Any]:
        """Top processes by resident memory, grouped by name, so the user knows what to close.
        The studio cannot free another program's memory; it can only point at it."""
        import psutil

        by_name: dict[str, int] = {}
        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                info = proc.info
                name = (info["name"] or "?").lower().removesuffix(".exe")
                by_name[name] = by_name.get(name, 0) + info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        vm = psutil.virtual_memory()
        top = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return {
            "total_gib": round(vm.total / GIB, 1),
            "available_gib": round(vm.available / GIB, 1),
            "top": [{"name": n, "gib": round(b / GIB, 2)} for n, b in top],
        }

    def parse_prompt(self, text: str) -> dict[str, Any]:
        return parse(text).to_dict()

    # -- config -----------------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = self.settings.config()
        return {
            "allow_over_budget": self.settings.allow_over_budget,
            "allow_over_budget_from_env": bool(os.environ.get(ENV_ALLOW_OVER_BUDGET)),
            "default_threads": cfg.get("default_threads"),
            "live_preview": bool(cfg.get("live_preview", True)),
            "auto_caption": bool(cfg.get("auto_caption", False)),
            "auto_caption_ready": self.auto_caption_ready(),
            "pinned_prompts": list(cfg.get("pinned_prompts") or []),
            "current_project": self.current_project(),
            "remote_enabled": self.settings.remote_enabled,
            "paused": self.jobs.is_paused(),
            "max_memory_gib": self.settings.max_memory_gib,
            "memory_budget_gib": round(
                memory_budget_bytes(self.hardware, self.planner.cap_bytes) / GIB, 1
            ),
            "memory_budget_auto_gib": round(memory_budget_bytes(self.hardware) / GIB, 1),
            "engine_backend": getattr(self.engine, "flavor", "cpu"),
            "engine_backend_from_env": bool(os.environ.get(ENV_SD_FLAVOR)),
            "engine_backends": self.engine_backends()
            if isinstance(self.engine, SdCppEngine)
            else [],
            "data_dir": str(self.settings.data_dir),
            "config_file": str(self.settings.config_file),
        }

    def set_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Write user settings to config.json. Only known keys are accepted."""
        cfg = self.settings.config()
        if "allow_over_budget" in changes:
            cfg["allow_over_budget"] = bool(changes["allow_over_budget"])
        if "live_preview" in changes:
            cfg["live_preview"] = bool(changes["live_preview"])
        if "auto_caption" in changes:
            cfg["auto_caption"] = bool(changes["auto_caption"])
        if "remote_enabled" in changes:
            cfg["remote_enabled"] = bool(changes["remote_enabled"])
        if "paused" in changes:
            self.jobs.set_paused(bool(changes["paused"]))
        if "max_memory_gib" in changes:
            v = changes["max_memory_gib"]
            try:
                gib = float(v) if v not in (None, "", 0, "0") else None
            except (TypeError, ValueError) as e:
                raise StudioError("memory limit must be a number of GiB, or empty") from e
            if gib is not None and not 2 <= gib <= 1024:
                raise StudioError("memory limit must be between 2 and 1024 GiB")
            cfg["max_memory_gib"] = gib
            self.planner.cap_bytes = int(gib * GIB) if gib else None
        if "engine_backend" in changes:
            self.set_engine_backend(str(changes["engine_backend"]))
            cfg = self.settings.config()
        if "pinned_prompts" in changes:
            seen: list[str] = []
            for t in changes["pinned_prompts"] or []:
                t = str(t).strip()
                if t and t not in seen:
                    seen.append(t)
            cfg["pinned_prompts"] = seen[:50]
        if "default_threads" in changes:
            v = changes["default_threads"]
            cfg["default_threads"] = int(v) if v not in (None, "", 0) else None
        self.settings.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.planner.allow_over_budget = self.settings.allow_over_budget
        return self.get_config()

    def samples_by_preset(self) -> dict[str, dict[str, Any]]:
        """The newest finished job per preset: a visual comparison of what each preset
        produces on this machine."""
        out: dict[str, dict[str, Any]] = {}
        for j in reversed(self.jobs.all()):
            if j.status == "done" and j.preset not in out:
                out[j.preset] = self.job_status(j.id)
        return out

    # -- background tasks (UI) ------------------------------------------------------------

    def start_download(self, model_name: str, variant_name: str | None = None) -> BackgroundTask:
        """Download a model variant in a background thread after the UI has shown consent."""
        if model_name not in MODELS:
            raise StudioError(f"unknown model {model_name!r}")
        key = f"download:{model_name}"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        model = MODELS[model_name]
        plan = (
            self.plan(model_name, "best", None, None, variant_name)
            if variant_name is None
            else None
        )
        variant = plan.variant if plan else model.variant(variant_name)
        task = BackgroundTask("download", f"{model.display_name} [{variant.name}]")
        task.total_bytes = self.models.missing_bytes(variant)
        self.tasks[key] = task

        def progress(name: str, done: int, total: int) -> None:
            task.current_file = name
            # bytes finished in earlier files + this file's progress
            finished = sum(f.size for f in variant.files if self.models.file_installed(f))
            task.done_bytes = min(task.total_bytes, finished + done) if task.total_bytes else done

        def run() -> None:
            try:
                self.models.install_variant(model, variant, lambda _d, _b: True, progress)
                task.done_bytes = task.total_bytes
                task.status, task.message = "done", "installed"
            except Exception as e:
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    def start_benchmark(self, preset: str = "fast", target_minutes: float = 15.0) -> BackgroundTask:
        key = "benchmark"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        task = BackgroundTask("benchmark", f"benchmark ({preset})")
        self.tasks[key] = task

        def status(st: dict[str, Any]) -> None:
            task.message = f"{st['status']}/{st['phase']} step {st['step']}/{st['total_steps']}"
            task.result = {"job_id": st["id"], "status": st}

        def run() -> None:
            try:
                profile = self.benchmark(preset, target_minutes, wait=2.0, on_status=status)
                task.result = {
                    "profile": profile.to_dict(),
                    "job_id": (task.result or {}).get("job_id"),
                }
                task.status, task.message = "done", "profile written"
            except Exception as e:
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    def task_list(self) -> dict[str, dict[str, Any]]:
        return {k: t.to_dict() for k, t in self.tasks.items()}

    # -- benchmark ------------------------------------------------------------------------

    def benchmark(
        self,
        preset: str = "fast",
        target_minutes: float = 15.0,
        wait: float = 0.5,
        on_status=None,
    ) -> MachineProfile:
        """Run one real job and write the machine profile (detect-settings for this machine).

        The job is a normal queued job so the calibration learns from it too. Blocks until
        it finishes. The recommendation is the best preset whose calibrated estimate is
        within ``target_minutes``; the user may always choose a slower one.
        """
        import time

        free_start = self.hardware.ram_available_bytes
        job = self.generate(
            "benchmark: a lighthouse on a rocky coast at dusk", preset=preset, seed=7
        )
        while True:
            st = self.job_status(job.id)
            if on_status:
                on_status(st)
            if st["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(wait)
        if st["status"] != "done":
            raise StudioError(
                f"benchmark job {job.id} {st['status']}: {st['error'] or st['message']}"
            )
        j = self.jobs.get(job.id)
        assert j is not None
        model = MODELS[j.model]
        variant = model.variant(j.variant)
        coeffs = self.calibration.coefficients(model, variant.name, variant.download_bytes)
        estimates = {
            p.name: self.calibration.estimate(
                model, variant.name, variant.download_bytes, p.width, p.height, p.steps, p.cfg_scale
            ).seconds
            for p in model.presets.values()
        }
        needed = required_bytes(variant, j.width, j.height)
        swapped = free_start < needed
        gpu = self.hardware.usable_gpu
        notes = []
        if swapped:
            notes.append(
                f"Only {free_start / GIB:.1f} GiB was free at the start and the job needed about "
                f"{needed / GIB:.1f} GiB, so this run swapped; with more free memory the same "
                "settings run several times faster."
            )
        profile = MachineProfile(
            measured_at=now_iso(),
            cpu=self.hardware.cpu_name,
            physical_cores=self.hardware.physical_cores,
            ram_total_gib=round(self.hardware.ram_total_bytes / GIB, 1),
            gpu=gpu.name if gpu else None,
            budget_gib=round(memory_budget_bytes(self.hardware) / GIB, 1),
            model=model.name,
            variant=variant.name,
            benchmark_preset=preset,
            width=j.width,
            height=j.height,
            steps=j.steps,
            wall_seconds=round(j.total_seconds or 0.0, 1),
            sampling_seconds=round(j.sampling_seconds or 0.0, 1),
            seconds_per_step_megapixel=round(coeffs.seconds_per_step_megapixel, 2),
            peak_rss_gib=round(j.peak_rss_bytes / GIB, 2) if j.peak_rss_bytes else None,
            free_ram_gib_at_start=round(free_start / GIB, 1),
            swapped=swapped,
            target_minutes=target_minutes,
            recommended_preset=recommend(estimates, target_minutes * 60),
            preset_estimates={k: round(v) for k, v in estimates.items()},
            notes=notes,
        )
        profile.save(self.settings.profile_file)
        return profile

    # -- capabilities ---------------------------------------------------------------------

    def model_report(self, model: ModelSpec) -> dict[str, Any]:
        fits = fit_variants(model, self.hardware, 1024, 1024, self.planner.cap_bytes)
        installed = {v.name for v in self.models.installed_variants(model)}
        return {
            "name": model.name,
            "display_name": model.display_name,
            "licence": model.licence,
            "tier": model.tier,
            "role": model.role,
            "family": model.family,
            "quality_rank": model.quality_rank,
            "description": model.description,
            "homepage": model.homepage,
            "installed_variants": sorted(installed),
            "fits_any": any(f.fits for f in fits),
            "fit_note": None
            if any(f.fits for f in fits)
            else (
                f"needs at least {fits[-1].required_bytes / GIB:.1f} GiB at 1024 px for its "
                f"smallest set; this machine's budget is {fits[0].budget_bytes / GIB:.1f} GiB"
            ),
            "variants": [
                {
                    "name": f.variant.name,
                    "rank": f.variant.rank,
                    "download_gib": round(f.variant.download_bytes / GIB, 1),
                    "needs_gib_at_1024": round(f.required_bytes / GIB, 1),
                    "fits_this_machine": f.fits,
                    "installed": f.variant.name in installed,
                    "note": f.variant.note,
                }
                for f in fits
            ],
            "memory_budget_gib": round(fits[0].budget_bytes / GIB, 1),
            "presets": self.planner.preset_table(model),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "studio_version": __version__,
            "policy": {
                "default_preset": "best",
                "quality": "Quality is a setting, never a consequence: 'best' is full "
                "resolution and full steps on the strongest permissively licensed model; the "
                "machine decides how long it takes. Nothing downgrades silently; a memory fit, "
                "if any, is stated in fit_note.",
                "faster_only_with_consent": "'balanced' and 'fast' trade quality for time and "
                "must be chosen explicitly; each preset below states its trade-off.",
                "licences": "Permissive only (Apache-2.0, MIT, BSD). Non-commercial weights "
                "are excluded by policy.",
                "network": "Models and the engine download once, each after explicit consent; "
                "generation is fully offline.",
            },
            "hardware": self.hardware.to_dict(),
            "engine": {
                "name": self.engine.name,
                "available": self.engine.available(),
                "version": self.engine.version(),
                "pinned_release": SDCPP_TAG,
                "binary": str(getattr(self.engine, "binary", lambda: None)() or ""),
                "backend": getattr(self.engine, "flavor", "cpu"),
                "suggestion": self.engine_suggestion(),
            },
            "models": [self.model_report(m) for m in MODELS.values()],
            "tools": self.tools_report(),
            "vision_models": self.vision_report(),
            "search": self.search_report(),
            "loras": self.loras(),
            "loras_dir": str(self.settings.loras_dir),
            "edit_model": getattr(self.edit_model(), "name", None),
            "edit_model_note": self.edit_model_note(),
            "custom_families": self.custom_families(),
            "custom_model_problems": self.custom_model_problems,
            "vision_engine": {
                "name": self.describe_engine.name,
                "available": self.describe_engine.available(),
                "version": self.describe_engine.version(),
            },
            "background_engine": {
                "name": self.background_engine.name,
                "available": self.background_engine.available(),
                "version": self.background_engine.version(),
            },
            "excluded_by_policy": [{"model": m, "reason": r} for m, r in EXCLUDED_BY_POLICY],
            "licences": list(LICENCES),
            "paths": {
                "data_dir": str(self.settings.data_dir),
                "models_dir": str(self.settings.models_dir),
                "out_dir": str(self.settings.out_dir),
            },
            "profile": (
                p.to_dict() if (p := MachineProfile.load(self.settings.profile_file)) else None
            ),
            "queue": {
                "running": running.id if (running := self.jobs.running()) else None,
                "queued": len(self.jobs.queued()),
                "worker_in_this_process": self.worker_active,
                "paused": self.jobs.is_paused(),
                "worker_pid": self.worker_owner_pid,
                "worker_version": (self.worker.lock.info() or {}).get("version"),
                "worker_started_at": (self.worker.lock.info() or {}).get("started_at"),
                "worker_outdated": bool(
                    self.worker.lock.info()
                    and (self.worker.lock.info() or {}).get("version") != __version__
                ),
            },
        }
