"""upscale and remove_background as queued tools, with fake engines."""

from __future__ import annotations

import json
import time

import pytest

from conftest import FLUX_SCHNELL, fake_install, laptop
from imagestudio.config import Settings
from imagestudio.engine.fake import FakeBackgroundEngine, FakeEngine
from imagestudio.registry import TOOL_MODELS
from imagestudio.service import Studio, StudioError


def fake_tool(settings: Settings, name: str) -> None:
    t = TOOL_MODELS[name]
    (settings.models_dir / t.file.filename).write_bytes(b"stub")
    m = settings.manifest_file
    d = json.loads(m.read_text()) if m.exists() else {}
    d[t.file.filename] = t.file.sha256
    m.write_text(json.dumps(d))


def wait(studio: Studio, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = studio.job_status(job_id)
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


@pytest.fixture
def tools_studio(settings: Settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_tool(settings, "realesrgan-x4plus")
    fake_tool(settings, "isnet-general-use")
    st = Studio(
        settings=settings,
        engine=FakeEngine(),
        hardware=laptop(),
        background_engine=FakeBackgroundEngine(),
    )
    yield st
    st.close()


def test_upscale_from_a_finished_job(tools_studio: Studio):
    src = tools_studio.generate("a teapot", preset="fast")
    wait(tools_studio, src.id)
    job = tools_studio.upscale(src.id)
    assert job.tool == "upscale" and job.model == "realesrgan-x4plus" and job.source_job == src.id
    from PIL import Image

    with Image.open(tools_studio.job_result(src.id)["path"]) as im:
        sw, sh = im.size
    assert (job.width, job.height) == (sw * 4, sh * 4)
    st = wait(tools_studio, job.id)
    assert st["status"] == "done" and st["tool"] == "upscale"
    spec = tools_studio.engine.calls[-1]
    assert spec.input.name == f"{src.id}.png" and spec.repeats == 1
    assert tools_studio.job_result(job.id)["path"].endswith(f"{job.id}.png")


def test_upscale_from_a_file_and_repeats(tools_studio: Studio, tmp_path):
    from imagestudio.engine.fake import tiny_png

    p = tmp_path / "photo.png"
    p.write_bytes(tiny_png(16, 8))
    job = tools_studio.upscale(str(p), "realesrgan-x4plus", repeats=2)
    assert (job.width, job.height) == (16 * 16, 8 * 16) and job.source_job is None
    wait(tools_studio, job.id)
    with pytest.raises(StudioError, match="repeats"):
        tools_studio.upscale(str(p), repeats=3)
    with pytest.raises(StudioError, match="neither a job id"):
        tools_studio.upscale("nope.png")


def test_remove_background(tools_studio: Studio):
    src = tools_studio.generate("a cat", preset="fast")
    wait(tools_studio, src.id)
    job = tools_studio.remove_background(src.id)
    assert job.tool == "remove_background" and job.model == "isnet-general-use"
    st = wait(tools_studio, job.id)
    assert st["status"] == "done"
    spec = tools_studio.background_engine.calls[-1]
    assert spec.model_name == "isnet-general-use"


def test_missing_tool_model_is_explained(settings: Settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(
        settings=settings,
        engine=FakeEngine(),
        hardware=laptop(),
        background_engine=FakeBackgroundEngine(),
    )
    try:
        src = studio.generate("x", preset="fast")
        wait(studio, src.id)
        with pytest.raises(StudioError, match="not downloaded"):
            studio.upscale(src.id)
        with pytest.raises(StudioError, match="unknown upscaler model"):
            studio.upscale(src.id, "isnet-general-use")
        caps = studio.capabilities()
        assert {t["name"] for t in caps["tools"]} == set(TOOL_MODELS)
        assert all(t["installed"] is False for t in caps["tools"])
    finally:
        studio.close()


def test_tool_models_are_pinned_and_permissive():
    for t in TOOL_MODELS.values():
        assert len(t.file.sha256) == 64 and t.file.size > 0
        assert t.licence in ("Apache-2.0", "BSD-3-Clause", "MIT")
