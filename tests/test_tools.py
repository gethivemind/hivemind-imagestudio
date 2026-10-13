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


def test_variations_and_edit_are_image_to_image_jobs(tools_studio: Studio, tmp_path):
    src = tools_studio.generate("a teapot", preset="fast", seed=5)
    wait(tools_studio, src.id)
    var = tools_studio.variations(src.id, strength=0.4)
    assert var.tool == "variations" and var.prompt == "a teapot" and var.strength == 0.4
    assert var.source_job == src.id and var.input_path.endswith(f"{src.id}.png")
    st = wait(tools_studio, var.id)
    assert st["status"] == "done" and st["strength"] == 0.4
    spec = tools_studio.engine.calls[-1]
    assert spec.init_image is not None and spec.strength == 0.4 and spec.mask_image is None
    assert "--from" in st["reproducible_prompt"] and "--strength 0.4" in st["reproducible_prompt"]

    from imagestudio.engine.fake import tiny_png

    mask = tmp_path / "mask.png"
    mask.write_bytes(tiny_png(16, 16, (255, 255, 255)))
    ed = tools_studio.edit(src.id, "the same teapot but blue", strength=0.8, mask=str(mask))
    assert ed.tool == "edit" and ed.mask_path == str(mask)
    wait(tools_studio, ed.id)
    spec = tools_studio.engine.calls[-1]
    assert spec.mask_image == mask and spec.strength == 0.8
    with pytest.raises(StudioError, match="strength"):
        tools_studio.edit(src.id, "x", strength=1.5)
    with pytest.raises(StudioError, match="need a prompt"):
        tools_studio.edit(str(mask), "", strength=0.5)


def test_from_param_in_prompt_runs_variations(tools_studio: Studio):
    src = tools_studio.generate("a teapot", preset="fast")
    wait(tools_studio, src.id)
    job = tools_studio.generate(f"a copper teapot --from {src.id} --strength 0.3 --preset fast")
    assert job.tool == "variations" and job.prompt == "a copper teapot" and job.strength == 0.3
    wait(tools_studio, job.id)


def test_upload_roundtrip(tools_studio: Studio):
    from imagestudio.engine.fake import tiny_png

    p = tools_studio.save_upload(tiny_png(8, 8))
    assert p.exists() and p.suffix == ".png"
    with pytest.raises(StudioError, match="not an image"):
        tools_studio.save_upload(b"hello")


def fake_vision(settings: Settings, name: str = "qwen2.5-vl-3b") -> None:
    from imagestudio.registry import VISION_MODELS

    v = VISION_MODELS[name]
    m = settings.manifest_file
    d = json.loads(m.read_text()) if m.exists() else {}
    for fl in v.files:
        (settings.models_dir / fl.filename).write_bytes(b"stub")
        d[fl.filename] = fl.sha256
    m.write_text(json.dumps(d))


def test_describe_returns_text_and_captions_the_source_job(settings: Settings):
    from imagestudio.engine.fake import FakeDescribeEngine

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_vision(settings)
    studio = Studio(
        settings=settings,
        engine=FakeEngine(),
        hardware=laptop(),
        describe_engine=FakeDescribeEngine(),
    )
    try:
        src = studio.generate("a cat", preset="fast")
        wait(studio, src.id)
        r = studio.describe(src.id)
        assert (
            r["task"] == "describe"
            and "fake description" in r["text"]
            and r["model"] == "qwen2.5-vl-3b"
        )
        assert studio.job_status(src.id)["caption"] == r["text"]
        r2 = studio.describe(src.id, "prompt")
        assert "[prompt]" in r2["text"]
        spec = studio.describe_engine.calls[-1]
        assert spec.mmproj_path.name.startswith("mmproj-")
        with pytest.raises(StudioError, match="unknown task"):
            studio.describe(src.id, "poem")
        with pytest.raises(StudioError, match="needs a question"):
            studio.describe(src.id, "custom")
        caps = studio.capabilities()
        assert (
            caps["vision_models"][0]["installed"] is True
            and caps["vision_engine"]["available"] is True
        )
    finally:
        studio.close()


def test_describe_explains_missing_model(settings: Settings):
    from imagestudio.engine.fake import FakeDescribeEngine

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(
        settings=settings,
        engine=FakeEngine(),
        hardware=laptop(),
        describe_engine=FakeDescribeEngine(),
    )
    try:
        src = studio.generate("a cat", preset="fast")
        wait(studio, src.id)
        with pytest.raises(StudioError, match="not downloaded"):
            studio.describe(src.id)
    finally:
        studio.close()


def test_llama_assets_are_pinned():
    from imagestudio.registry import LLAMA_ASSETS, find_llama_asset

    assert find_llama_asset("win32", "AMD64").exe == "llama-mtmd-cli.exe"
    assert find_llama_asset("darwin", "arm64").url.startswith(
        "https://github.com/ggml-org/llama.cpp/"
    )
    assert all(len(a.sha256) == 64 for a in LLAMA_ASSETS)


def test_components_remove_and_in_use_guard(tools_studio: Studio):
    comps = {c["name"]: c for c in tools_studio.components()}
    assert comps["flux-schnell:q5"]["installed"] and comps["realesrgan-x4plus"]["installed"]
    assert comps["flux-schnell:q5"]["files"][0]["path"].endswith(".gguf")
    r = tools_studio.remove_component("tool", "realesrgan-x4plus")
    assert r["removed"] == "realesrgan-x4plus"
    assert not {c["name"]: c for c in tools_studio.components()}["realesrgan-x4plus"]["installed"]
    # a model behind a queued job cannot be removed
    tools_studio.engine.step_delay = 0.3
    job = tools_studio.generate("busy", preset="fast")
    with pytest.raises(StudioError, match="used by a queued or running job"):
        tools_studio.remove_component("model", "flux-schnell:q5")
    wait(tools_studio, job.id)
    r = tools_studio.remove_component("model", "flux-schnell:q5")
    assert r["removed"] == "flux-schnell:q5"
    assert not tools_studio.models.installed_variants(FLUX_SCHNELL)


def test_delete_restore_favorite_import_lineage(tools_studio: Studio):
    from imagestudio.engine.fake import tiny_png

    src = tools_studio.generate("a teapot", preset="fast")
    wait(tools_studio, src.id)
    var = tools_studio.variations(src.id, 0.5)
    wait(tools_studio, var.id)
    lin = tools_studio.lineage(src.id)
    assert lin["children"] == [var.id] and tools_studio.lineage(var.id)["parent"] == src.id
    assert tools_studio.set_favorite(src.id, True)["favorite"] is True
    assert tools_studio.job_status(src.id)["favorite"] is True
    d = tools_studio.delete_job(var.id)
    assert d["deleted"] == var.id
    assert var.id not in [j["id"] for j in tools_studio.list_jobs()]
    assert (tools_studio.settings.data_dir / "trash" / f"{var.id}.png").exists()
    tools_studio.restore_job(var.id)
    assert var.id in [j["id"] for j in tools_studio.list_jobs()]
    tools_studio.delete_job(var.id)
    assert tools_studio.empty_trash(0) == 1
    assert not (tools_studio.settings.data_dir / "trash" / f"{var.id}.png").exists()
    imp = tools_studio.import_image(tiny_png(12, 10), "photo.png")
    assert imp.tool == "import" and (imp.width, imp.height) == (12, 10) and imp.status == "done"
    assert tools_studio.job_status(imp.id)["title"].endswith("imported · 12×10")
    up = tools_studio.remove_background(imp.id)
    wait(tools_studio, up.id)
