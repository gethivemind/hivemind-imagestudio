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
    assert store2.get("job_abc").status == "queued"
    worker.start()
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
