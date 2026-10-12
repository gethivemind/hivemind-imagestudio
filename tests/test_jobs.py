from __future__ import annotations

import base64
import json
import time

import pytest

from conftest import laptop
from imagestudio.config import Settings
from imagestudio.engine.fake import FakeEngine
from imagestudio.jobs import Job, JobStore, Worker
from imagestudio.service import Studio, StudioError


def wait_done(studio: Studio, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = studio.job_status(job_id)
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_generate_returns_immediately_with_estimate_and_finishes(studio: Studio):
    job = studio.generate("a lighthouse at dusk")
    assert job.status == "queued"
    assert job.estimate_seconds > 0
    assert job.preset == "best" and job.width == 1024 and job.steps == 4
    st = wait_done(studio, job.id)
    assert st["status"] == "done" and st["progress"] == 1.0
    result = studio.job_result(job.id)
    assert result["ready"] and result["path"].endswith(f"{job.id}.png")
    with_b64 = studio.job_result(job.id, as_base64=True)
    assert base64.b64decode(with_b64["base64"]).startswith(b"\x89PNG")
    # the image is kept on disk under the output folder
    assert (studio.settings.out_dir / f"{job.id}.png").exists()
    # the job record is persisted
    assert (studio.settings.jobs_dir / f"{job.id}.json").exists()


def test_result_before_done_does_not_block(settings: Settings):
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(step_delay=0.2), hardware=laptop())
    try:
        job = studio.generate("slow one")
        r = studio.job_result(job.id)
        assert r["ready"] is False and r["status"] in ("queued", "running")
        st = studio.job_status(job.id)
        assert st["eta_seconds"] is not None
        wait_done(studio, job.id)
    finally:
        studio.close()


def test_jobs_run_in_order_and_queue_position_is_reported(settings: Settings):
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(step_delay=0.05), hardware=laptop())
    try:
        a = studio.generate("first")
        b = studio.generate("second")
        st_b = studio.job_status(b.id)
        assert st_b["status"] == "queued" and st_b["queue_position"] == 1
        wait_done(studio, a.id)
        wait_done(studio, b.id)
        ids = [j["id"] for j in studio.list_jobs()]
        assert ids == [b.id, a.id]  # newest first
    finally:
        studio.close()


def test_engine_failure_marks_job_failed_and_worker_survives(settings: Settings):
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    engine = FakeEngine(fail=True)
    studio = Studio(settings=settings, engine=engine, hardware=laptop())
    try:
        job = studio.generate("doomed")
        st = wait_done(studio, job.id)
        assert st["status"] == "failed" and "fake engine failure" in st["error"]
        engine.fail = False
        job2 = studio.generate("fine")
        assert wait_done(studio, job2.id)["status"] == "done"
    finally:
        studio.close()


def test_cancel_queued_job(settings: Settings):
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(step_delay=0.1), hardware=laptop())
    try:
        a = studio.generate("running")
        b = studio.generate("queued")
        assert studio.cancel(b.id)
        assert studio.job_status(b.id)["status"] == "cancelled"
        wait_done(studio, a.id)
        assert not studio.cancel(a.id)
    finally:
        studio.close()


def test_running_jobs_are_requeued_after_restart(settings: Settings):
    store = JobStore(settings)
    job = Job(
        id="job_abc",
        tool="generate",
        prompt="p",
        negative_prompt="",
        model="flux-schnell",
        variant="q5",
        preset="best",
        width=512,
        height=512,
        steps=4,
        cfg_scale=1.0,
        seed=1,
        sampler="euler",
        component_paths={},
        extra_args=[],
        estimate_seconds=10,
        estimate_basis="x",
        fit_note=None,
        warnings=[],
        status="running",
    )
    store.put(job)
    store2 = JobStore(settings)
    engine = FakeEngine()
    worker = Worker(store2, engine)
    assert store2.get("job_abc").status == "running"  # not touched until the worker starts
    worker.start()
    assert store2.get("job_abc").status in ("queued", "running", "done")
    deadline = time.monotonic() + 5
    while store2.get("job_abc").status != "done" and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stop()
    assert store2.get("job_abc").status == "done"
    assert engine.calls and engine.calls[0].prompt == "p"


def test_generate_refuses_without_engine_or_model(settings: Settings):
    class NoEngine(FakeEngine):
        def available(self) -> bool:
            return False

    studio = Studio(settings=settings, engine=NoEngine(), hardware=laptop(), start_worker=False)
    with pytest.raises(StudioError, match="not installed"):
        studio.generate("x")
    studio2 = Studio(settings=settings, engine=FakeEngine(), hardware=laptop(), start_worker=False)
    with pytest.raises(StudioError, match="not downloaded"):
        studio2.generate("x")
    with pytest.raises(StudioError, match="empty"):
        studio2.generate("   ")


def test_finished_job_calibrates_estimates(studio: Studio):
    job = studio.generate("calibrate me", preset="fast")
    wait_done(studio, job.id)
    # the fake engine reports near-zero sampling time, which is skipped (<= 0 guard) unless
    # it is positive; force a positive record to check persistence through the hook path
    j = studio.jobs.get(job.id)
    j.sampling_seconds = 12.0
    j.load_seconds = 3.0
    studio._on_finished(j)
    data = json.loads(studio.settings.calibration_file.read_text())
    assert f"flux-schnell:{job.variant}" in data
    est = studio.plan("flux-schnell", "fast", None, None).estimate
    assert est.calibrated


def test_seed_is_recorded_and_reproducible(studio: Studio):
    a = studio.generate("same", seed=42)
    b = studio.generate("same", seed=42)
    assert a.seed == b.seed == 42
    c = studio.generate("random")
    assert 0 <= c.seed < 2**31


def test_low_memory_plan_passes_mmap_to_engine(settings: Settings):
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    hw = laptop()
    hw.ram_available_bytes = 1024**3
    engine = FakeEngine()
    studio = Studio(settings=settings, engine=engine, hardware=hw)
    try:
        job = studio.generate("tight")
        wait_done(studio, job.id)
        assert "--mmap" in engine.calls[0].extra_args
    finally:
        studio.close()


def test_threads_and_vae_tiling_reach_the_engine(studio: Studio):
    job = studio.generate("knobs", threads=4, vae_tiling=True)
    wait_done(studio, job.id)
    spec = studio.engine.calls[-1]
    assert spec.threads == 4 and "--vae-tiling" in spec.extra_args
    with pytest.raises(StudioError):
        studio.generate("x", threads=0)


def test_config_file_unlocks_over_budget(settings: Settings):
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q8"))
    settings.config_file.write_text('{"allow_over_budget": true}')
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop(), start_worker=False)
    plan = studio.plan("flux-schnell", "best", None, None, "q8")
    assert plan.over_budget


def test_benchmark_writes_profile_and_recommends(studio: Studio):
    seen = []
    profile = studio.benchmark("fast", target_minutes=15, wait=0.01, on_status=seen.append)
    assert seen and seen[-1]["status"] == "done"
    assert studio.settings.profile_file.exists()
    assert profile.model == "flux-schnell" and profile.benchmark_preset == "fast"
    assert set(profile.preset_estimates) == {"best", "balanced", "fast"}
    assert profile.recommended_preset in ("best", "balanced", "fast", None)
    caps = studio.capabilities()
    assert caps["profile"]["measured_at"] == profile.measured_at


def test_recommend_picks_highest_quality_within_target():
    from imagestudio.profile import recommend

    est = {"best": 900, "balanced": 400, "fast": 100}
    assert recommend(est, 1000) == "best"
    assert recommend(est, 500) == "balanced"
    assert recommend(est, 50) is None


def test_read_only_studio_does_not_requeue_running_jobs(settings: Settings):
    store = JobStore(settings)
    job = Job(
        id="job_live",
        tool="generate",
        prompt="p",
        negative_prompt="",
        model="flux-schnell",
        variant="q5",
        preset="best",
        width=512,
        height=512,
        steps=4,
        cfg_scale=1.0,
        seed=1,
        sampler="euler",
        component_paths={},
        extra_args=[],
        estimate_seconds=10,
        estimate_basis="x",
        fit_note=None,
        warnings=[],
        status="running",
    )
    store.put(job)
    Studio(settings=settings, engine=FakeEngine(), hardware=laptop(), start_worker=False)
    assert JobStore(settings).get("job_live").status == "running"


def test_second_process_submits_and_first_worker_runs_it(settings: Settings):
    """Two Studios on one data dir: the first holds the worker lock, the second only submits,
    and the first picks the job up by polling the directory."""
    from conftest import FLUX_SCHNELL, fake_install

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    first = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    first.worker.poll_seconds = 0.05
    second = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        assert first.worker_active and not second.worker_active
        assert second.worker_owner_pid == first.worker_owner_pid
        job = second.generate("from the second process")
        wait_done(second, job.id)
        assert first.engine.calls and first.engine.calls[0].prompt == "from the second process"
        assert not second.engine.calls
        assert second.job_status(job.id)["status"] == "done"
    finally:
        second.close()
        first.close()
    assert not settings.worker_lock_file.exists()


def test_stale_lock_from_dead_pid_is_taken_over(settings: Settings):
    settings.worker_lock_file.write_text("999999999")
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        assert studio.worker_active
    finally:
        studio.close()


def test_running_job_owned_by_a_live_process_is_not_requeued(settings: Settings):
    import os

    store = JobStore(settings)
    job = Job(
        id="job_theirs",
        tool="generate",
        prompt="p",
        negative_prompt="",
        model="flux-schnell",
        variant="q5",
        preset="best",
        width=512,
        height=512,
        steps=4,
        cfg_scale=1.0,
        seed=1,
        sampler="euler",
        component_paths={},
        extra_args=[],
        estimate_seconds=10,
        estimate_basis="x",
        fit_note=None,
        warnings=[],
        status="running",
        worker_pid=os.getppid(),  # a live process that is not us
    )
    store.put(job)
    engine = FakeEngine()
    worker = Worker(JobStore(settings), engine)
    worker.start()
    time.sleep(0.3)
    worker.stop()
    assert JobStore(settings).get("job_theirs").status == "running"
    assert not engine.calls
