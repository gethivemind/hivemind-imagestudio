"""The studio: wires hardware, models, calibration, planner, engine and the job queue.
Both the MCP server and the CLI talk to this one object."""

from __future__ import annotations

import base64
import secrets
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
from imagestudio.planner import GIB, Plan, PlanError, Planner, fit_variants
from imagestudio.registry import EXCLUDED_BY_POLICY, LICENCES, MODELS, SDCPP_TAG, ModelSpec

MAX_SEED = 2**31 - 1


class StudioError(RuntimeError):
    pass


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
        self.hardware = hardware or detect()
        self.models = ModelStore(self.settings)
        self.calibration = Calibration(self.settings.calibration_file, self.hardware)
        self.planner = Planner(self.hardware, self.models, self.calibration)
        self.engine: Engine = engine or SdCppEngine(self.settings)
        self.jobs = JobStore(self.settings)
        self.worker = Worker(self.jobs, self.engine, on_finished=self._on_finished)
        if start_worker:
            self.worker.start()

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
    ) -> Plan:
        try:
            return self.planner.plan(model, preset, width, height, variant)
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
    ) -> Job:
        if not prompt or not prompt.strip():
            raise StudioError("prompt must not be empty")
        if not self.engine.available():
            raise StudioError(
                "the image engine (stable-diffusion.cpp) is not installed. Run "
                "`imagestudio install-engine` (downloads ~20-40 MB after confirmation)."
            )
        plan = self.plan(model, preset, width, height, variant)
        if not plan.installed:
            missing = self.models.missing_bytes(plan.variant)
            raise StudioError(
                f"{plan.model.display_name} [{plan.variant.name}] is not downloaded "
                f"({missing / GIB:.1f} GiB). Run `imagestudio download-model {plan.model.name}` "
                f"or call download_model(name={plan.model.name!r}, consent=true)."
            )
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
            sampler=plan.model.sampler,
            component_paths=paths,
            extra_args=list(plan.preset.extra_args),
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
            position = [j.id for j in self.jobs.queued()].index(job_id) + 1
        return {
            "id": job.id,
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
            "warnings": job.warnings,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "output_path": job.output_path,
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
            "queue": {
                "running": running.id if (running := self.jobs.running()) else None,
                "queued": len(self.jobs.queued()),
            },
        }
