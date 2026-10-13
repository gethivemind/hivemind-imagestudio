"""The job queue. Long work runs in one worker thread; tools return a job id at once.

Every job is a JSON file under ``jobs/`` and every finished image a PNG under ``out/``, so
nothing is lost when the server stops. On start, jobs that were running are re-queued.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from imagestudio.config import Settings
from imagestudio.engine.base import Engine, EngineError, GenerateSpec, Progress, UpscaleSpec

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
    peak_rss_bytes: int | None = None
    threads: int | None = None
    over_budget: bool = False
    worker_pid: int | None = None  # process rendering this job; guards against duplicates
    cancel_requested: bool = False  # set by any process; honoured by the rendering one
    live_preview: bool = True  # write a per-step preview image while rendering
    input_path: str | None = None  # upscale / remove_background: the source image
    source_job: str | None = None  # the job the source image came from, if any
    tool_model: str | None = None  # upscale / remove_background: which tool model
    mask_path: str | None = None  # edit: inpainting mask
    caption: str | None = None  # describe: last caption written for this image
    favorite: bool = False
    deleted_at: str | None = None  # soft delete; the file moves to trash/
    strength: float = 0.6  # variations / edit: distance from the source image
    loras: list[list] = field(default_factory=list)  # [[name, weight], ...] from the loras folder

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
        """0-1. Before sampling reports steps, progress follows elapsed time against the
        estimate (capped at the point where sampling begins), so the bar keeps moving."""
        if self.status == "done":
            return 1.0
        if self.status != "running":
            return 0.0
        if self.phase == "sampling" and self.total_steps:
            return 0.1 + 0.85 * (self.step / self.total_steps)
        if self.phase in ("decoding", "saving"):
            return 0.97
        elapsed = 0.0
        if self.started_at:
            elapsed = (datetime.now(UTC) - datetime.fromisoformat(self.started_at)).total_seconds()
        if self.estimate_seconds > 0:
            return min(0.1, 0.1 * elapsed / (0.2 * self.estimate_seconds))
        return 0.02

    @property
    def preview_name(self) -> str:
        return f"{self.id}.preview.png"


class JobStore:
    """Jobs as JSON files. Several processes may share one directory: exactly one runs a
    worker, the others submit jobs and read status. ``refresh`` picks up files changed by
    another process; jobs this process is executing (``owned``) are never overwritten."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._mtimes: dict[str, float] = {}
        self.owned: set[str] = set()
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            for p in self.settings.jobs_dir.glob("*.json"):
                job_id = p.stem
                if job_id in self.owned:
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if self._mtimes.get(job_id) == mtime:
                    continue
                try:
                    job = Job.from_dict(json.loads(p.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError):
                    continue  # half-written by another process; next refresh gets it
                self._jobs[job.id] = job
                self._mtimes[job_id] = mtime

    def cancel_marker(self, job_id: str) -> Path:
        return self.settings.jobs_dir / f"{job_id}.cancel"

    def new_id(self) -> str:
        return "job_" + secrets.token_hex(6)

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            tmp = self.settings.jobs_dir / f"{job.id}.json.tmp"
            tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
            final = self.settings.jobs_dir / f"{job.id}.json"
            tmp.replace(final)
            with contextlib.suppress(OSError):
                self._mtimes[job.id] = final.stat().st_mtime

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self.refresh()
            return self._jobs.get(job_id)

    def all(self, include_deleted: bool = False) -> list[Job]:
        with self._lock:
            self.refresh()
            jobs = [j for j in self._jobs.values() if include_deleted or not j.deleted_at]
            return sorted(jobs, key=lambda j: j.created_at)

    def queued(self) -> list[Job]:
        return [j for j in self.all() if j.status == "queued"]

    def running(self) -> Job | None:
        return next((j for j in self.all() if j.status == "running"), None)


def _pid_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


_HELD_IN_PROCESS: set[Path] = set()  # several Studios in one process share one pid


class WorkerLock:
    """One worker per data directory. The lock file holds the owner's pid; a dead owner's
    lock is taken over."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.held = False

    def info(self) -> dict | None:
        """{pid, version, started_at} of the live owner, or None."""
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            d = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        except ValueError:
            return None
        pid = int(d.get("pid", 0))
        return d if pid and _pid_alive(pid) else None

    def owner(self) -> int | None:
        d = self.info()
        return int(d["pid"]) if d else None

    def acquire(self) -> bool:
        if self.path in _HELD_IN_PROCESS:
            return False
        owner = self.owner()
        if owner is not None and owner != os.getpid():
            return False
        _HELD_IN_PROCESS.add(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from imagestudio import __version__

        self.path.write_text(
            json.dumps({"pid": os.getpid(), "version": __version__, "started_at": _now()}),
            encoding="utf-8",
        )
        self.held = True
        return True

    def release(self) -> None:
        if self.held:
            _HELD_IN_PROCESS.discard(self.path)
            with contextlib.suppress(OSError):
                if self.owner() in (None, os.getpid()):
                    self.path.unlink()
            self.held = False


class Worker:
    """One background thread draining the queue through the engine, in FIFO order.

    Only the process holding the worker lock runs jobs; it also polls the store so jobs
    submitted by other processes (CLI, UI, another server) are picked up."""

    def __init__(
        self,
        store: JobStore,
        engine: Engine,
        on_finished=None,
        background_engine=None,
        tool_model_path=None,
    ) -> None:
        self.store = store
        self.engine = engine
        self.background_engine = background_engine
        self.tool_model_path = tool_model_path  # name -> Path
        self.on_finished = on_finished
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="imagestudio-worker", daemon=True)
        self._current: str | None = None
        self._seen: set[str] = set()
        self.lock = WorkerLock(store.settings.worker_lock_file)
        self.poll_seconds = 2.0

    def start(self) -> None:
        """Recover jobs left running by a previous worker, then start draining the queue.

        Recovery happens here, not in ``__init__``, so read-only users of the store (the CLI's
        doctor/plan/jobs commands) never touch a job another process is still running.
        """
        if self._thread.is_alive():
            return True
        if not self.lock.acquire():
            return False
        for job in self.store.all():
            if job.status == "running":
                if job.worker_pid and job.worker_pid != os.getpid() and _pid_alive(job.worker_pid):
                    continue  # another live process is rendering it; leave it alone
                job.status, job.phase, job.message = "queued", "queued", "re-queued after restart"
                job.worker_pid = None
                self.store.put(job)
        for job in self.store.queued():
            self._enqueue(job.id)
        self._thread.start()
        return True

    @property
    def active(self) -> bool:
        return self._thread.is_alive()

    def _enqueue(self, job_id: str) -> None:
        if job_id not in self._seen:
            self._seen.add(job_id)
            self._q.put(job_id)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._q.put("")  # wake
        if self._thread.is_alive():
            self._thread.join(timeout)
        self.lock.release()

    def submit(self, job: Job) -> None:
        """Persist the job. If this process runs the worker it is queued at once; otherwise
        the process holding the lock picks it up on its next poll."""
        self.store.put(job)
        if self.active:
            self._enqueue(job.id)

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        if job.status == "running":
            if job.id in self.store.owned:
                cancel = getattr(self.engine, "cancel", None)
                if cancel:
                    cancel()
            else:
                # Another process is rendering it: leave a marker file it polls for. A separate
                # file cannot be overwritten by that process's own progress writes.
                job.cancel_requested = True
                self.store.cancel_marker(job.id).write_text("cancel", encoding="utf-8")
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
            try:
                job_id = self._q.get(timeout=self.poll_seconds)
            except queue.Empty:
                for job in self.store.queued():  # submitted by another process
                    self._enqueue(job.id)
                continue
            if not job_id or self._stop.is_set():
                continue
            job = self.store.get(job_id)
            if job is None or job.status != "queued":
                continue
            self._current = job_id
            self._execute(job)
            self._current = None

    def _run_tool(self, job: Job, spec: GenerateSpec, on_progress):
        out = self.store.settings.out_dir / f"{job.id}.png"
        if job.tool in ("generate", "variations", "edit"):
            return self.engine.generate(spec, on_progress)
        if not job.input_path or not Path(job.input_path).exists():
            raise EngineError(f"input image not found: {job.input_path}")
        model_path = self.tool_model_path(job.tool_model) if self.tool_model_path else None
        if model_path is None:
            raise EngineError(f"tool model {job.tool_model!r} is not installed")
        if job.tool == "upscale":
            return self.engine.upscale(
                UpscaleSpec(
                    Path(job.input_path),
                    out,
                    model_path,
                    repeats=max(1, job.steps),
                    threads=job.threads,
                ),
                on_progress,
            )
        if job.tool == "remove_background":
            if self.background_engine is None:
                raise EngineError("no background-removal engine configured")
            from imagestudio.engine.background import BackgroundSpec

            return self.background_engine.remove_background(
                BackgroundSpec(Path(job.input_path), out, model_path, job.tool_model or ""),
                on_progress,
            )
        raise EngineError(f"unknown tool {job.tool!r}")

    def _execute(self, job: Job) -> None:
        self.store.owned.add(job.id)
        job.status, job.phase, job.started_at = "running", "loading", _now()
        job.worker_pid = os.getpid()
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

        stop_watch = threading.Event()

        def watch_cancel() -> None:
            marker = self.store.cancel_marker(job.id)
            while not stop_watch.wait(0.25):
                if marker.exists():
                    job.cancel_requested = True
                    cancel = getattr(self.engine, "cancel", None)
                    if cancel:
                        cancel()
                    return

        threading.Thread(target=watch_cancel, name=f"cancel-watch-{job.id}", daemon=True).start()

        from imagestudio.registry import MODELS

        edit_model = job.tool == "edit" and getattr(MODELS.get(job.model), "role", "") == "edit"
        prompt = job.prompt
        if job.loras:
            # stable-diffusion.cpp reads <lora:name:weight> from the prompt and loads the
            # file from --lora-model-dir (set in extra_args by the studio)
            prompt += " " + " ".join(f"<lora:{n}:{float(w):g}>" for n, w in job.loras)
        spec = GenerateSpec(
            prompt=prompt,
            negative_prompt=job.negative_prompt,
            width=job.width,
            height=job.height,
            steps=job.steps,
            cfg_scale=job.cfg_scale,
            seed=job.seed,
            sampler=job.sampler,
            output=self.store.settings.out_dir / f"{job.id}.png",
            preview_path=(self.store.settings.out_dir / job.preview_name)
            if job.live_preview
            else None,
            component_paths={k: Path(v) for k, v in job.component_paths.items()},
            extra_args=tuple(job.extra_args),
            threads=job.threads,
            init_image=(
                Path(job.input_path)
                if job.input_path and job.tool in ("variations", "edit") and not edit_model
                else None
            ),
            mask_image=Path(job.mask_path) if job.mask_path and not edit_model else None,
            strength=job.strength,
            ref_images=(Path(job.input_path),) if edit_model and job.input_path else (),
        )
        try:
            result = self._run_tool(job, spec, on_progress)
        except EngineError as e:
            stop_watch.set()
            if str(e) == "cancelled":
                job.status, job.phase, job.message = "cancelled", "cancelled", "cancelled by caller"
            else:
                job.status, job.phase, job.error = "failed", "failed", str(e)
        except Exception as e:
            stop_watch.set()
            job.status, job.phase, job.error = "failed", "failed", f"{type(e).__name__}: {e}"
        else:
            stop_watch.set()
            job.status, job.phase = "done", "done"
            job.step = job.total_steps = job.steps
            job.output_path = str(result.output)
            job.load_seconds = result.load_seconds
            job.sampling_seconds = result.sampling_seconds
            job.total_seconds = result.total_seconds
            job.peak_rss_bytes = result.peak_rss_bytes or None
            job.message = "finished"
        job.finished_at = _now()
        self.store.put(job)
        self.store.owned.discard(job.id)
        with contextlib.suppress(OSError):
            (self.store.settings.out_dir / job.preview_name).unlink()
        with contextlib.suppress(OSError):
            self.store.cancel_marker(job.id).unlink()
        if self.on_finished:
            with contextlib.suppress(Exception):
                self.on_finished(job)
