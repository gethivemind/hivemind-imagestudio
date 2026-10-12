"""The job queue. Long work runs in one worker thread; tools return a job id at once.

Every job is a JSON file under ``jobs/`` and every finished image a PNG under ``out/``, so
nothing is lost when the server stops. On start, jobs that were running are re-queued.
"""

from __future__ import annotations

import contextlib
import json
import queue
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imagestudio.config import Settings
from imagestudio.engine.base import Engine, EngineError, GenerateSpec, Progress

STATUSES = ("queued", "running", "done", "failed", "cancelled")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    tool: str
    prompt: str
    negative_prompt: str
    model: str
    variant: str
    preset: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    sampler: str
    component_paths: dict[str, str]
    extra_args: list[str]
    estimate_seconds: float
    estimate_basis: str
    fit_note: str | None
    warnings: list[str]
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    phase: str = "queued"
    step: int = 0
    total_steps: int = 0
    seconds_per_step: float | None = None
    message: str = ""
    output_path: str | None = None
    error: str | None = None
    load_seconds: float | None = None
    sampling_seconds: float | None = None
    total_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    # -- derived -------------------------------------------------------------------------

    def eta_seconds(self, queue_ahead_seconds: float = 0.0) -> float | None:
        """Remaining seconds, from live progress when sampling, else from the estimate."""
        if self.status in ("done", "failed", "cancelled"):
            return 0.0
        if self.status == "queued":
            return queue_ahead_seconds + self.estimate_seconds
        if self.phase == "sampling" and self.total_steps and self.seconds_per_step:
            remaining_steps = max(0, self.total_steps - self.step)
            # Decoding and saving take a small fraction of one step at most.
            return remaining_steps * self.seconds_per_step + self.seconds_per_step * 0.5
        elapsed = 0.0
        if self.started_at:
            elapsed = (datetime.now(UTC) - datetime.fromisoformat(self.started_at)).total_seconds()
        return max(0.0, self.estimate_seconds - elapsed)

    def progress_fraction(self) -> float:
        if self.status == "done":
            return 1.0
        if self.status != "running":
            return 0.0
        if self.phase == "sampling" and self.total_steps:
            return 0.1 + 0.85 * (self.step / self.total_steps)
        return {"loading": 0.02, "encoding": 0.08, "decoding": 0.96, "saving": 0.99}.get(
            self.phase, 0.0
        )


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.settings.jobs_dir.glob("*.json")):
            try:
                job = Job.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            self._jobs[job.id] = job

    def new_id(self) -> str:
        return "job_" + secrets.token_hex(6)

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            tmp = self.settings.jobs_dir / f"{job.id}.json.tmp"
            tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
            tmp.replace(self.settings.jobs_dir / f"{job.id}.json")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def queued(self) -> list[Job]:
        return [j for j in self.all() if j.status == "queued"]

    def running(self) -> Job | None:
        return next((j for j in self.all() if j.status == "running"), None)


class Worker:
    """One background thread draining the queue through the engine, in FIFO order."""

    def __init__(
        self,
        store: JobStore,
        engine: Engine,
        on_finished=None,
    ) -> None:
        self.store = store
        self.engine = engine
        self.on_finished = on_finished
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="imagestudio-worker", daemon=True)
        self._current: str | None = None
        # Recover: anything running when we last stopped goes back to the queue.
        for job in self.store.all():
            if job.status == "running":
                job.status, job.phase, job.message = "queued", "queued", "re-queued after restart"
                self.store.put(job)
        for job in self.store.queued():
            self._q.put(job.id)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._q.put("")  # wake
        if self._thread.is_alive():
            self._thread.join(timeout)

    def submit(self, job: Job) -> None:
        self.store.put(job)
        self._q.put(job.id)

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        if job.status == "running":
            cancel = getattr(self.engine, "cancel", None)
            if cancel:
                cancel()
            return True
        job.status, job.phase, job.finished_at = "cancelled", "cancelled", _now()
        self.store.put(job)
        return True

    def queue_seconds_ahead(self, job_id: str) -> float:
        """Estimated seconds of work queued before ``job_id`` (plus the running job's rest)."""
        total = 0.0
        running = self.store.running()
        if running and running.id != job_id:
            total += running.eta_seconds() or 0.0
        for j in self.store.queued():
            if j.id == job_id:
                break
            total += j.estimate_seconds
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._q.get()
            if not job_id or self._stop.is_set():
                continue
            job = self.store.get(job_id)
            if job is None or job.status != "queued":
                continue
            self._current = job_id
            self._execute(job)
            self._current = None

    def _execute(self, job: Job) -> None:
        job.status, job.phase, job.started_at = "running", "loading", _now()
        self.store.put(job)
        last_write = 0.0

        def on_progress(p: Progress) -> None:
            nonlocal last_write
            job.phase = p.phase
            if p.total_steps:
                job.step, job.total_steps = p.step, p.total_steps
            if p.seconds_per_step is not None:
                job.seconds_per_step = p.seconds_per_step
            if p.message:
                job.message = p.message
            now = time.monotonic()
            if now - last_write > 1.0 or p.phase != "sampling":
                last_write = now
                self.store.put(job)

        spec = GenerateSpec(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            width=job.width,
            height=job.height,
            steps=job.steps,
            cfg_scale=job.cfg_scale,
            seed=job.seed,
            sampler=job.sampler,
            output=self.store.settings.out_dir / f"{job.id}.png",
            component_paths={k: Path(v) for k, v in job.component_paths.items()},
            extra_args=tuple(job.extra_args),
        )
        try:
            result = self.engine.generate(spec, on_progress)
        except EngineError as e:
            if str(e) == "cancelled":
                job.status, job.phase, job.message = "cancelled", "cancelled", "cancelled by caller"
            else:
                job.status, job.phase, job.error = "failed", "failed", str(e)
        except Exception as e:
            job.status, job.phase, job.error = "failed", "failed", f"{type(e).__name__}: {e}"
        else:
            job.status, job.phase = "done", "done"
            job.step = job.total_steps = job.steps
            job.output_path = str(result.output)
            job.load_seconds = result.load_seconds
            job.sampling_seconds = result.sampling_seconds
            job.total_seconds = result.total_seconds
            job.message = "finished"
        job.finished_at = _now()
        self.store.put(job)
        if self.on_finished:
            with contextlib.suppress(Exception):
                self.on_finished(job)
