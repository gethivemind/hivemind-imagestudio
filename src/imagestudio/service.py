"""The studio: wires hardware, models, calibration, planner, engine and the job queue.
Both the MCP server and the CLI talk to this one object."""

from __future__ import annotations

import base64
import io
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from imagestudio import __version__
from imagestudio.calibration import Calibration, human_duration
from imagestudio.config import Settings
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
from imagestudio.registry import EXCLUDED_BY_POLICY, LICENCES, MODELS, SDCPP_TAG, ModelSpec

MAX_SEED = 2**31 - 1


class StudioError(RuntimeError):
    pass


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
        )
        self.engine: Engine = engine or SdCppEngine(self.settings)
        self.jobs = JobStore(self.settings)
        self.worker = Worker(self.jobs, self.engine, on_finished=self._on_finished)
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
        if job.status != "done" or job.sampling_seconds is None or job.sampling_seconds <= 0:
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
    ) -> Job:
        if not prompt or not prompt.strip():
            raise StudioError("prompt must not be empty")
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
            tool="generate",
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
            + (["--vae-tiling"] if vae_tiling else []),
            threads=threads,
            over_budget=plan.over_budget,
            estimate_seconds=plan.estimate.seconds,
            estimate_basis=plan.estimate.basis,
            fit_note=plan.fit_note,
            warnings=list(plan.warnings),
        )
        self.worker.submit(job)
        return job

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
            "prompt": job.prompt,
            "status": job.status,
            "phase": job.phase,
            "progress": round(job.progress_fraction(), 3),
            "step": job.step,
            "total_steps": job.total_steps or job.steps,
            "seconds_per_step": job.seconds_per_step,
            "eta_seconds": None if eta is None else round(eta),
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

    def export_image(
        self, job_id: str, fmt: str = "png", quality: int = 90
    ) -> tuple[bytes, str, str]:
        """The finished image as bytes: original PNG, or a JPEG at the given quality.

        Returns (data, mime_type, filename)."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "done" or not job.output_path:
            raise StudioError(f"job {job_id!r} has no finished image")
        path = Path(job.output_path)
        stem = f"{job.id}-{job.preset}-{job.width}x{job.height}-seed{job.seed}"
        if fmt == "png":
            return path.read_bytes(), "image/png", f"{stem}.png"
        if fmt in ("jpg", "jpeg"):
            from PIL import Image

            q = max(30, min(100, int(quality)))
            buf = io.BytesIO()
            with Image.open(path) as im:
                im.convert("RGB").save(buf, "JPEG", quality=q, optimize=True)
            return buf.getvalue(), "image/jpeg", f"{stem}-q{q}.jpg"
        raise StudioError(f"unknown format {fmt!r}; png or jpg")

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
        fits = fit_variants(model, self.hardware, 1024, 1024)
        installed = {v.name for v in self.models.installed_variants(model)}
        return {
            "name": model.name,
            "display_name": model.display_name,
            "licence": model.licence,
            "quality_rank": model.quality_rank,
            "description": model.description,
            "homepage": model.homepage,
            "installed_variants": sorted(installed),
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
            },
            "models": [self.model_report(m) for m in MODELS.values()],
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
                "worker_pid": self.worker_owner_pid,
            },
        }
