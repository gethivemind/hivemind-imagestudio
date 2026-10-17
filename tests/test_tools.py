"""upscale and remove_background as queued tools, with fake engines."""

from __future__ import annotations

import json
import time
from pathlib import Path

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


def test_fix_faces_renders_each_face_and_keeps_the_source_size(settings: Settings):
    from imagestudio.engine.fake import FakeFaceDetector, tiny_png

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_tool(settings, "ultraface-rfb-640")
    det = FakeFaceDetector(faces=[(10, 10, 20, 20), (40, 8, 12, 12)])
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop(), face_detector=det)
    try:
        src = settings.data_dir / "portrait.png"
        src.write_bytes(tiny_png(64, 48))
        with pytest.raises(StudioError, match="strength"):
            studio.fix_faces(str(src), strength=0.9)
        job = studio.fix_faces(str(src), prompt="a sailor")
        assert job.tool == "fix_faces" and job.tool_model == "ultraface-rfb-640"
        assert (job.width, job.height) == (64, 48) and len(job.regions) == 2
        assert all(r[2] == r[3] and r[0] >= 0 and r[1] >= 0 for r in job.regions)
        assert job.prompt.startswith("a sailor, detailed face")
        assert "2 face(s)" in job.estimate_basis
        st = wait(studio, job.id)
        assert st["status"] == "done", st["error"]
        calls = [c for c in studio.engine.calls if c.init_image is not None]
        assert len(calls) == 2 and all(c.width == c.height == 768 for c in calls)
        assert calls[0].strength == 0.35 and calls[1].seed == calls[0].seed + 1
        from PIL import Image

        with Image.open(st["output_path"]) as im:
            assert im.size == (64, 48)
        assert studio.job_status(job.id)["regions"] == job.regions
        # a detector that sees nothing is an honest error, not an empty job
        studio.face_detector.faces = []
        with pytest.raises(StudioError, match="no faces found"):
            studio.fix_faces(str(src))
    finally:
        studio.close()


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


def test_projects_file_new_work_and_export_as_zip(settings: Settings):
    import io
    import zipfile

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        assert studio.projects() == [] and studio.current_project() is None
        loose = studio.generate("a loose image", preset="fast")
        wait(studio, loose.id)
        p = studio.add_project("Lighthouse series", "night studies")
        assert p["name"] == "Lighthouse series" and studio.current_project() == p["id"]
        with pytest.raises(StudioError, match="already exists"):
            studio.add_project("lighthouse SERIES")
        a = studio.generate("lighthouse one", preset="fast")
        b = studio.generate("lighthouse two", preset="fast")
        wait(studio, a.id)
        wait(studio, b.id)
        assert studio.job_status(a.id)["project"] == p["id"]
        assert studio.job_status(loose.id)["project"] is None
        assert studio.projects()[0]["count"] == 2
        # the PNG remembers the project by name
        out = Path(studio.job_status(a.id)["output_path"])
        assert Studio.read_metadata(out)["project"] == "Lighthouse series"
        # file the loose one in, then export
        studio.set_job_project(loose.id, p["id"])
        name, data = studio.export_project(p["id"])
        assert name == "Lighthouse-series.zip"
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            assert "prompts.txt" in names and "project.json" in names
            assert sum(n.endswith(".png") for n in names) == 3
            txt = z.read("prompts.txt").decode("utf-8")
            assert "lighthouse one" in txt and "night studies" in txt
        # rename, switch off, forget: images stay, unfiled
        studio.update_project(p["id"], name="Lighthouses")
        assert studio.project(p["id"])["name"] == "Lighthouses"
        studio.set_current_project(None)
        assert studio.current_project() is None
        c = studio.generate("after switching off", preset="fast")
        assert studio.job_status(c.id)["project"] is None
        r = studio.remove_project(p["id"])
        assert r["unfiled"] == 3 and studio.projects() == []
        assert studio.job_status(a.id)["project"] is None
        with pytest.raises(StudioError, match="no such project"):
            studio.set_current_project("prj_nope")
    finally:
        studio.close()


def test_assist_prompt_writes_for_the_target_model(settings: Settings):
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
        r = studio.assist_prompt("my cat on a spaceship")
        assert r["prompt"].startswith("a fake, detailed prompt")
        assert r["target_model"].startswith("FLUX.1") and r["model"] == "qwen2.5-vl-3b"
        # FLUX schnell has no guidance: the assistant's negative is dropped on purpose
        assert r["negative_prompt"] == "" and r["notes"] == "assumed daylight"
        spec = studio.describe_engine.calls[-1]
        assert "FLUX.1" in spec.system and "Brief: my cat on a spaceship" in spec.user
        r2 = studio.assist_prompt("make it night", current_prompt="a cat on a spaceship")
        spec2 = studio.describe_engine.calls[-1]
        assert "Current prompt:" in spec2.user and "Change requested: make it night" in spec2.user
        assert r2["prompt"]
        # a Qwen-Image target keeps negatives
        r3 = studio.assist_prompt("a poster that says HELLO", model="qwen-image")
        assert r3["negative_prompt"] == "blurry, text" and r3["target_model"].startswith("Qwen")
        with pytest.raises(StudioError, match="tell the assistant"):
            studio.assist_prompt("")
    finally:
        studio.close()


def test_assist_prompt_salvages_a_cut_off_answer(settings: Settings):
    from imagestudio.engine.describe import DescribeResult
    from imagestudio.engine.fake import FakeDescribeEngine

    class Cut(FakeDescribeEngine):
        def complete(self, spec):
            return DescribeResult(
                text=(
                    '{"prompt": "A cat on a ship.", '
                    '"negative": "blurry, dark, Blurry, abstract, abstr'
                ),
                seconds=0.1,
                model="fake",
            )

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_vision(settings)
    studio = Studio(
        settings=settings, engine=FakeEngine(), hardware=laptop(), describe_engine=Cut()
    )
    try:
        r = studio.assist_prompt("x", model="qwen-image")
        # never raw JSON; duplicates in the negative list collapse; the user is told
        assert r["prompt"] == "A cat on a ship."
        assert r["negative_prompt"] == "blurry, dark, abstract, abstr"
        assert "ran out of words" in r["notes"]
    finally:
        studio.close()


def test_assist_prompt_explains_what_to_install(settings: Settings):
    from imagestudio.engine.fake import FakeDescribeEngine

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(
        settings=settings,
        engine=FakeEngine(),
        hardware=laptop(),
        describe_engine=FakeDescribeEngine(),
    )
    try:
        with pytest.raises(StudioError, match="not downloaded"):
            studio.assist_prompt("a cat")
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


def test_auto_caption_runs_in_the_background_when_enabled(settings: Settings):
    from imagestudio.engine.fake import FakeDescribeEngine

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_vision(settings)
    fake_tool(settings, "realesrgan-x4plus")
    settings.config_file.parent.mkdir(parents=True, exist_ok=True)
    settings.config_file.write_text(json.dumps({"auto_caption": True}), encoding="utf-8")
    vision = FakeDescribeEngine()
    studio = Studio(
        settings=settings, engine=FakeEngine(), hardware=laptop(), describe_engine=vision
    )
    try:
        assert studio.get_config()["auto_caption"] is True
        assert studio.get_config()["auto_caption_ready"] is True
        src = studio.generate("a cat", preset="fast")
        wait(studio, src.id)
        for _ in range(100):
            if studio.jobs.get(src.id).caption:
                break
            time.sleep(0.05)
        assert studio.jobs.get(src.id).caption
        assert len(vision.calls) == 1
        # an upscale of a captioned image inherits the caption without a second model run
        up = studio.upscale(src.id)
        wait(studio, up.id)
        time.sleep(0.2)
        assert studio.jobs.get(up.id).caption == studio.jobs.get(src.id).caption
        assert len(vision.calls) == 1
        studio.set_config({"auto_caption": False})
        assert studio.get_config()["auto_caption"] is False
    finally:
        studio.close()


def test_loras_are_listed_validated_and_passed_to_the_engine(settings: Settings, tmp_path):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        assert studio.loras() == []
        with pytest.raises(StudioError, match="no LoRA named"):
            studio.generate("a fox --lora inkstyle:0.8", preset="fast")
        info = studio.import_lora(b"\x00" * 64, "inkstyle.safetensors")
        assert (
            info["name"] == "inkstyle" and studio.loras()[0]["filename"] == "inkstyle.safetensors"
        )
        with pytest.raises(StudioError, match="ends with"):
            studio.import_lora(b"x", "notes.txt")
        job = studio.generate(
            "a fox --lora inkstyle:0.8 --lora inkstyle.safetensors", preset="fast"
        )
        assert job.loras == [["inkstyle", 0.8], ["inkstyle", 1.0]]
        assert "--lora-model-dir" in job.extra_args and job.prompt == "a fox"
        wait(studio, job.id)
        spec = studio.engine.calls[-1]
        assert "<lora:inkstyle:0.8>" in spec.prompt and "--lora-model-dir" in spec.extra_args
        assert "--lora inkstyle:0.8" in studio.job_status(job.id)["reproducible_prompt"]
        # kwarg form, as the MCP tool and the UI send it
        job2 = studio.generate("a fox", preset="fast", loras=["inkstyle:0.5"])
        assert job2.loras == [["inkstyle", 0.5]]
        assert studio.capabilities()["loras"][0]["name"] == "inkstyle"
        wait(studio, job2.id)
        assert studio.remove_lora("inkstyle") == 64 and studio.loras() == []
    finally:
        studio.close()


def test_custom_models_register_plan_and_generate(settings: Settings, tmp_path):
    from imagestudio.registry import MODELS, build_custom_model

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    ck = tmp_path / "my-sd15.safetensors"
    ck.write_bytes(b"\x00" * 4096)
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        fams = studio.custom_families()
        assert set(fams) >= {"flux", "qwen-image", "sd15", "sdxl"}
        # the studio's own encoders are offered for reuse by a FLUX finetune
        assert fams["flux"]["shared"].get("--clip_l")
        with pytest.raises(StudioError, match="file not found"):
            studio.add_custom_model(
                {"name": "ghost", "family": "sd15", "files": {"-m": str(tmp_path / "nope")}}
            )
        with pytest.raises(StudioError, match="taken by a verified model"):
            studio.add_custom_model(
                {"name": "flux-schnell", "family": "sd15", "files": {"-m": str(ck)}}
            )
        rep = studio.add_custom_model(
            {
                "name": "My-SD15",
                "display_name": "My SD 1.5",
                "family": "sd15",
                "licence": "CreativeML Open RAIL-M",
                "files": {"-m": str(ck)},
            }
        )
        assert rep["tier"] == "custom" and rep["name"] == "my-sd15"
        assert MODELS["my-sd15"].presets["best"].steps == 30
        assert studio.settings.config()["custom_models"][0]["name"] == "my-sd15"
        # auto never picks it; by name it plans and runs with the user's file
        assert studio.plan(None, "fast", None, None).model.name == "flux-schnell"
        plan = studio.plan("my-sd15", "fast", None, None)
        assert plan.installed and plan.variant.name == "own"
        job = studio.generate("a cat --model my-sd15", preset="fast")
        assert job.model == "my-sd15" and job.component_paths["-m"] == str(ck)
        wait(studio, job.id)
        assert studio.engine.calls[-1].component_paths["-m"] == ck
        caps = studio.capabilities()
        assert any(m["name"] == "my-sd15" and m["tier"] == "custom" for m in caps["models"])
        # a second Studio on the same settings sees it again (persisted)
        studio2 = Studio(
            settings=settings, engine=FakeEngine(), hardware=laptop(), start_worker=False
        )
        assert "my-sd15" in MODELS and studio2.custom_models()[0]["name"] == "my-sd15"
        studio2.close()
        studio.remove_custom_model("my-sd15")
        assert "my-sd15" not in MODELS and ck.exists()
        with pytest.raises(StudioError, match="no custom model"):
            studio.remove_custom_model("my-sd15")
        # a broken entry in config.json is reported, not fatal
        settings.config_file.write_text(
            json.dumps({"custom_models": [{"name": "bad", "family": "sd15", "files": {}}]}),
            encoding="utf-8",
        )
        studio3 = Studio(
            settings=settings, engine=FakeEngine(), hardware=laptop(), start_worker=False
        )
        assert studio3.custom_model_problems and "bad" not in MODELS
        studio3.close()
        _ = build_custom_model
    finally:
        studio.close()


def test_edit_model_routes_instruction_edits_and_auto_never_picks_it(settings: Settings):
    from imagestudio.registry import MODELS, QWEN_IMAGE_EDIT

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        assert MODELS["qwen-image-edit"].role == "edit"
        src = studio.generate("a teapot", preset="fast")
        wait(studio, src.id)
        # without the edit model installed, edits are image-to-image with a strength
        e1 = studio.edit(src.id, "the same teapot but blue", preset="fast")
        assert e1.model == "flux-schnell" and e1.strength == 0.45
        wait(studio, e1.id)
        assert studio.engine.calls[-1].init_image is not None
        assert studio.engine.calls[-1].ref_images == ()
        # install the edit model (stubs); maskless edits now go to it with a reference image
        fake_install(settings, QWEN_IMAGE_EDIT, QWEN_IMAGE_EDIT.variant("q4"))
        studio.models._load()  # the store read the manifest at startup
        studio.settings.config_file.parent.mkdir(parents=True, exist_ok=True)
        studio.settings.config_file.write_text(
            json.dumps({"allow_over_budget": True}), encoding="utf-8"
        )
        studio.planner.allow_over_budget = True
        assert studio.edit_model() is QWEN_IMAGE_EDIT
        e2 = studio.edit(src.id, "the same teapot at night, lit by candles", preset="fast")
        assert e2.model == "qwen-image-edit" and e2.strength == 1.0 and e2.tool == "edit"
        assert e2.component_paths["--llm_vision"].endswith("mmproj-Q8_0.gguf")
        assert "--flow-shift" in e2.extra_args
        wait(studio, e2.id)
        spec = studio.engine.calls[-1]
        assert spec.ref_images and spec.init_image is None
        rp = studio.job_status(e2.id)["reproducible_prompt"]
        assert "--model qwen-image-edit" in rp and "--strength" not in rp
        # masked edits and variations stay on image-to-image; auto generation never picks it
        from pathlib import Path

        mask = studio.save_upload(Path(studio.job_result(src.id)["path"]).read_bytes())
        e3 = studio.edit(src.id, "blue", mask=str(mask), preset="fast")
        assert e3.model == "flux-schnell"
        v = studio.variations(src.id, preset="fast")
        assert v.model == "flux-schnell"
        assert studio.plan(None, "fast", None, None).model.name == "flux-schnell"
        with pytest.raises(StudioError, match="does not take a mask"):
            studio.edit(src.id, "blue", mask=str(mask), model="qwen-image-edit", preset="fast")
        assert studio.capabilities()["edit_model"] == "qwen-image-edit"
        for j in (e3, v):
            wait(studio, j.id)
    finally:
        studio.close()


def test_model_store_sees_downloads_made_by_another_process(settings: Settings):
    from imagestudio.models import ModelStore
    from imagestudio.registry import QWEN_IMAGE_EDIT

    store = ModelStore(settings)  # a running studio
    assert store.installed_variants(QWEN_IMAGE_EDIT) == []
    fake_install(settings, QWEN_IMAGE_EDIT, QWEN_IMAGE_EDIT.variant("q3"))  # the CLI, elsewhere
    assert [v.name for v in store.installed_variants(QWEN_IMAGE_EDIT)] == ["q3"]


def test_memory_limit_lowers_the_budget_and_edit_model_needs_a_fit(settings: Settings):
    from imagestudio.registry import QWEN_IMAGE_EDIT

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_install(settings, QWEN_IMAGE_EDIT, QWEN_IMAGE_EDIT.variant("q3"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        cfg = studio.get_config()
        auto = cfg["memory_budget_auto_gib"]
        assert cfg["max_memory_gib"] is None and cfg["memory_budget_gib"] == auto
        # the installed edit model does not fit a 16 GB laptop → Edit stays image-to-image
        assert studio.edit_model() is None
        assert "image-to-image" in (studio.edit_model_note() or "")
        rep = studio.model_report(QWEN_IMAGE_EDIT)
        assert rep["fits_any"] is False and "needs at least" in rep["fit_note"]
        # a lower cap shrinks the FLUX budget too and can make q5 stop fitting
        cfg = studio.set_config({"max_memory_gib": 6})
        assert cfg["memory_budget_gib"] == 6.0
        assert studio.model_report(FLUX_SCHNELL)["fits_any"] is False
        with pytest.raises(StudioError):
            studio.set_config({"max_memory_gib": 1})
        cfg = studio.set_config({"max_memory_gib": None})
        assert cfg["memory_budget_gib"] == auto
        # the unlock makes the installed edit model eligible again (explicit consent)
        studio.set_config({"allow_over_budget": True})
        assert studio.edit_model() is QWEN_IMAGE_EDIT
    finally:
        studio.close()
