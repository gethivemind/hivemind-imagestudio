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


def metadata_of(studio: Studio, job_id: str, key: str, timeout: float = 5.0) -> dict:
    """The PNG's imagestudio chunk once the worker has written it (it lands just after the
    job turns done)."""
    path = Path(studio.job_status(job_id)["output_path"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        meta = Studio.read_metadata(path)
        if key in meta:
            return meta
        time.sleep(0.05)
    raise AssertionError(f"metadata key {key!r} never appeared")


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


def fake_search(settings: Settings) -> None:
    from imagestudio.registry import default_search_model

    sm = default_search_model()
    m = settings.manifest_file
    d = json.loads(m.read_text()) if m.exists() else {}
    for fl in sm.files:
        (settings.models_dir / fl.filename).write_bytes(b"stub")
        d[fl.filename] = fl.sha256
    m.write_text(json.dumps(d))


def test_semantic_search_indexes_finished_images_and_finds_duplicates(settings: Settings):
    from imagestudio.engine.fake import FakeClipEngine

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(
        settings=settings, engine=FakeEngine(), hardware=laptop(), search_engine=FakeClipEngine()
    )
    try:
        with pytest.raises(StudioError, match="needs CLIP"):
            studio.search("a cat")
        assert studio.search_report()["installed"] is False
        fake_search(settings)
        assert studio.search_ready() and studio.search_report()["installed"] is True
        from imagestudio.engine.fake import tiny_png

        a = studio.generate("a cat", preset="fast", width=512, height=512)
        b = studio.generate("a cat again", preset="fast", width=512, height=512)
        for j in (a, b):
            wait(studio, j.id)
        # an import is finished on arrival and is indexed too; different pixels, no duplicate
        c = studio.import_image(tiny_png(8, 8, (200, 10, 10)), "red.png")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and studio.search_report()["indexed"] < 3:
            time.sleep(0.05)
        assert studio.search_report()["indexed"] == 3
        r = studio.search("a lighthouse", limit=10)
        assert r["total"] == 3 and r["pending"] == 0 and len(r["results"]) == 3
        assert all(-1.0 <= x["score"] <= 1.0 for x in r["results"])
        # the fake engine paints identical pixels for a and b: one duplicate group, c outside it
        d = studio.duplicates()
        assert len(d["groups"]) == 1 and sorted(d["groups"][0]) == sorted([a.id, b.id])
        assert c.id not in d["groups"][0]
        studio.delete_job(b.id)
        assert studio.search_report()["indexed"] == 2 and studio.duplicates()["groups"] == []
        with pytest.raises(StudioError, match="needs a few words"):
            studio.search("   ")
        # removing the search model also drops the index
        studio.remove_component("search", "clip-vit-b32")
        assert studio.search_report()["installed"] is False
        assert not (settings.data_dir / "search-index.npz").exists()
    finally:
        studio.close()


def test_series_and_web_page_export(settings: Settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    fake_tool(settings, "realesrgan-x4plus")
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        root = studio.generate("a red fox on a log", preset="fast", seed=3)
        wait(studio, root.id, 30)
        child = studio.variations(root.id, 0.4, "a red fox on a log at night", preset="fast")
        wait(studio, child.id, 30)
        grand = studio.upscale(child.id)
        wait(studio, grand.id, 30)
        s = studio.series(grand.id)
        assert s["root"] == root.id and [m["id"] for m in s["members"]] == [
            root.id,
            child.id,
            grand.id,
        ]
        assert s["members"][0]["changes"] == [] and s["members"][2]["current"] is True
        c1 = s["members"][1]["changes"]
        assert c1[0].startswith("variation at strength 0.4") and any(
            "added: at night" in x for x in c1
        )
        assert any(x.startswith("seed ") for x in c1)
        c2 = s["members"][2]["changes"]
        assert c2[0] == "upscaled" and any(x.startswith("size ") for x in c2)
        # the same series from the root's point of view
        assert studio.series(root.id)["members"][0]["current"] is True
        name, page = studio.export_html([root.id, "job_nope", child.id], "Fox studies", "two takes")
        html = page.decode("utf-8")
        assert name == "Fox-studies.html" and html.count("<figure>") == 2
        assert (
            "data:image/jpeg;base64," in html
            and "a red fox on a log" in html
            and "two takes" in html
        )
        p = studio.add_project("Foxes")
        studio.set_job_project(root.id, p["id"])
        name2, page2 = studio.export_project_html(p["id"])
        assert name2 == "Foxes.html" and page2.decode("utf-8").count("<figure>") == 1
        with pytest.raises(StudioError, match="none of the chosen"):
            studio.export_html(["job_nope"])
    finally:
        studio.close()


def test_outpaint_grows_the_canvas_and_masks_only_the_new_area(settings: Settings):
    from PIL import Image

    from imagestudio.engine.fake import tiny_png
    from imagestudio.outpaint import padding_for_fraction, padding_for_ratio

    assert (
        padding_for_ratio(512, 512, "16:9").left > 0
        and padding_for_ratio(512, 512, "16:9").top == 0
    )
    assert padding_for_ratio(512, 512, "9:16").top > 0
    assert not padding_for_ratio(1024, 576, "16:9").any
    p = padding_for_fraction(400, 200, 0.25, "horizontal")
    assert (p.left, p.right, p.top, p.bottom) == (100, 100, 0, 0)

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        src = settings.data_dir / "square.png"
        src.write_bytes(tiny_png(512, 512, (200, 40, 40)))
        job = studio.outpaint(str(src), ratio="16:9", prompt="more coastline")
        assert job.tool == "edit" and job.mask_path and job.input_path
        assert (job.width, job.height) == (896, 512)  # 512*16/9 = 910 → snapped to 16
        assert job.strength == 0.75 and "more coastline" in job.prompt and "seamless" in job.prompt
        with Image.open(job.mask_path) as m:
            assert m.size == (896, 512)
            px = m.convert("L").load()
            assert px[5, 256] == 255 and px[890, 256] == 255  # new area: paint
            assert px[448, 256] < 16  # the middle of the original: kept (band blur fades out)
        with Image.open(job.input_path) as c:
            assert c.size == (896, 512) and c.getpixel((448, 256))[0] > 150  # original pasted
        wait(studio, job.id)
        wide = settings.data_dir / "wide.png"
        wide.write_bytes(tiny_png(896, 504))
        with pytest.raises(StudioError, match="already"):
            studio.outpaint(str(wide), ratio="16:9")
        j2 = studio.outpaint(str(src), fraction=0.5, sides="bottom")
        assert (j2.width, j2.height) == (512, 768)
        with pytest.raises(StudioError, match="unknown ratio"):
            studio.outpaint(str(src), ratio="5:1")
        # huge canvases are scaled to the engine's 2048 limit instead of refused
        from imagestudio.outpaint import fit_to_engine

        cw, ch, scale = fit_to_engine(2048, 2048, padding_for_ratio(2048, 2048, "16:9"))
        assert cw == 2048 and ch % 16 == 0 and ch <= 1152 and scale < 1.0
    finally:
        studio.close()


def test_outpaint_finish_matches_colours_and_restores_the_original(tmp_path):
    import numpy as np
    from PIL import Image

    from imagestudio.outpaint import finish

    canvas = tmp_path / "canvas.png"
    result = tmp_path / "result.png"
    # canvas: warm original in the middle of a 300x100 strip
    c = Image.new("RGB", (300, 100), (120, 120, 120))
    c.paste(Image.new("RGB", (100, 100), (220, 140, 60)), (100, 0))
    c.save(canvas)
    # the model's output: everything cooler and flatter, including the kept area
    r = Image.new("RGB", (300, 100), (90, 100, 120))
    r.paste(Image.new("RGB", (100, 100), (160, 120, 90)), (100, 0))
    r.save(result)
    finish(result, canvas, (100, 0, 100, 100))
    with Image.open(result) as im:
        a = np.asarray(im.convert("RGB"), dtype=int)
    assert all(
        abs(int(c) - o) <= 3 for c, o in zip(a[50, 150], (220, 140, 60), strict=True)
    )  # original restored
    # the painted sides moved towards the original's warmth instead of staying cool
    assert a[50, 20][0] > 90 and a[50, 20][2] < 120
    assert im.size == (300, 100)


def test_transforms_apply_a_tuned_look_and_record_it(settings: Settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        ids = [t["id"] for t in studio.transforms()]
        assert (
            "sketch" in ids
            and "pixel" in ids
            and all(t["kind"] == "built-in" for t in studio.transforms())
        )
        src = studio.generate("a red fox on a log --seed 7", preset="fast")
        wait(studio, src.id)
        job = studio.transform(src.id, "sketch", preset="fast")
        assert job.tool == "variations" and job.strength == 0.55
        assert job.prompt.startswith("a red fox on a log, pencil sketch")
        assert "--seed" not in job.prompt and job.negative_prompt.startswith("colour")
        assert job.style == "Pencil sketch" and job.style_text.startswith("pencil sketch")
        wait(studio, job.id)
        # a look on a look replaces the earlier wording instead of stacking it
        job2 = studio.transform(job.id, "pixel", strength=0.4)
        assert (
            job2.prompt.startswith("a red fox on a log, pixel art") and "pencil" not in job2.prompt
        )
        assert job2.strength == 0.4
        # saved styles are transforms too
        s = studio.add_style("Gouache", "gouache painting, matte paper")
        assert any(
            t["id"] == "style:" + s["id"] and t["kind"] == "style" for t in studio.transforms()
        )
        job3 = studio.transform(src.id, "style:" + s["id"])
        assert job3.style == "Gouache" and job3.prompt.endswith("gouache painting, matte paper")
        with pytest.raises(StudioError, match="unknown transform"):
            studio.transform(src.id, "hologram")
    finally:
        studio.close()


def test_engine_suggestion_points_at_vulkan_only_when_it_applies(settings: Settings):
    import platform
    import sys

    from conftest import workstation
    from imagestudio.registry import find_engine_asset

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    vulkan_here = find_engine_asset(sys.platform, platform.machine(), "vulkan") is not None
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        sg = studio.engine_suggestion()
        if not vulkan_here:
            assert sg is None
        else:
            assert sg["flavor"] == "vulkan" and sg["action"] in ("install", "use")
            assert "integrated" in sg["reason"] and "2.4" in sg["strength"]
        assert studio.capabilities()["engine"]["suggestion"] == sg
    finally:
        studio.close()
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=workstation())
    try:
        sg = studio.engine_suggestion()
        if vulkan_here:
            assert sg["strength"] == "likely large" and "RTX 4090" in sg["reason"]
    finally:
        studio.close()
    # an active non-CPU backend means nothing to suggest
    engine = FakeEngine()
    engine.flavor = "vulkan"
    studio = Studio(settings=settings, engine=engine, hardware=laptop())
    try:
        assert studio.engine_suggestion() is None
    finally:
        studio.close()


def test_export_images_zips_a_selection(settings: Settings):
    import io
    import zipfile

    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        a = studio.generate("one", preset="fast")
        b = studio.generate("two", preset="fast")
        for j in (a, b):
            wait(studio, j.id)
        name, data = studio.export_images([a.id, "job_nope", b.id], "3 images")
        assert name == "3-images.zip"
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert sum(n.endswith(".png") for n in z.namelist()) == 2
            assert "one" in z.read("prompts.txt").decode("utf-8")
        with pytest.raises(StudioError, match="none of the chosen"):
            studio.export_images(["job_nope"])
    finally:
        studio.close()


def test_styles_library_records_into_jobs_and_round_trips(settings: Settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    try:
        assert studio.styles() == []
        s = studio.add_style("Gouache", "gouache painting, matte paper, flat colour", "glossy")
        assert s["id"].startswith("sty_") and s["sample_job"] is None
        with pytest.raises(StudioError, match="already exists"):
            studio.add_style("gouache", "x")
        with pytest.raises(StudioError, match="wording"):
            studio.add_style("Empty", "")
        job = studio.generate(
            "a harbour, gouache painting, matte paper, flat colour",
            preset="fast",
            style="Gouache",
            style_text="gouache painting, matte paper, flat colour",
        )
        wait(studio, job.id)
        st = studio.job_status(job.id)
        assert st["style"] == "Gouache" and st["style_text"].startswith("gouache painting")
        assert studio.styles()[0]["sample_job"] == job.id
        meta = metadata_of(studio, job.id, "style")
        assert meta["style"] == "Gouache" and meta["style_text"].startswith("gouache")
        # an exported PNG imported elsewhere carries the style along
        imported = studio.import_image(Path(st["output_path"]).read_bytes(), "shared.png")
        assert imported.style == "Gouache" and imported.style_text.startswith("gouache")
        # export / import merge: existing names are kept, new ones added
        exp = studio.export_styles()
        assert exp["kind"] == "imagestudio-styles" and exp["styles"][0]["name"] == "Gouache"
        exp["styles"].append({"name": "Linocut", "text": "linocut print, bold black lines"})
        exp["styles"].append({"name": "", "text": "nameless"})
        r = studio.import_styles(exp)
        assert r == {"added": 1, "skipped": 2, "total": 2}
        with pytest.raises(StudioError, match="expected a styles export"):
            studio.import_styles({"nope": 1})
        studio.update_style(s["id"], name="Gouache paint", negative="")
        assert studio.styles()[0]["name"] == "Gouache paint"
        studio.remove_style(s["id"])
        assert [x["name"] for x in studio.styles()] == ["Linocut"]
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
        assert metadata_of(studio, a.id, "project")["project"] == "Lighthouse series"
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
