"""The local web UI's JSON API, through Starlette's test client."""

from __future__ import annotations

import time

from starlette.testclient import TestClient

from imagestudio.service import Studio
from imagestudio.ui import build_app


def test_ui_page_plan_generate_and_image(studio: Studio):
    client = TestClient(build_app(studio))
    assert "Image Studio" in client.get("/").text
    caps = client.get("/api/capabilities").json()
    assert caps["policy"]["default_preset"] == "best"

    plan = client.get("/api/plan", params={"preset": "fast"}).json()
    assert plan["width"] == 512 and plan["load"]["label"]
    refused = client.get("/api/plan", params={"variant": "q8"})
    assert refused.status_code == 400 and "locked" in refused.json()["error"]

    r = client.post("/api/generate", json={"prompt": "a teapot", "preset": "fast", "seed": 3})
    assert r.status_code == 200
    job_id = r.json()["id"]
    for _ in range(200):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] == "done":
            break
        time.sleep(0.02)
    assert st["status"] == "done" and st["seed"] == 3
    img = client.get(f"/api/image/{job_id}")
    assert img.status_code == 200 and img.content.startswith(b"\x89PNG")
    listing = client.get("/api/jobs").json()
    assert listing["jobs"][0]["id"] == job_id and listing["worker_in_this_process"] is True
    bad = client.post("/api/generate", json={"prompt": ""})
    assert bad.status_code == 400


def test_downloads_exports_samples_and_tasks(studio: Studio, monkeypatch):
    client = TestClient(build_app(studio))
    r = client.post("/api/generate", json={"prompt": "a teapot", "preset": "fast", "seed": 3})
    job_id = r.json()["id"]
    for _ in range(200):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.02)
    png = client.get(f"/api/download/{job_id}?fmt=png")
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
    assert "attachment" in png.headers["content-disposition"]
    jpg = client.get(f"/api/download/{job_id}?fmt=jpg&quality=70")
    assert jpg.status_code == 200 and jpg.content[:2] == b"\xff\xd8"
    samples = client.get("/api/samples").json()
    assert samples["fast"]["id"] == job_id

    info = client.get(
        "/api/models/download", params={"model": "flux-schnell", "variant": "q4"}
    ).json()
    assert info["total_gib"] > 5 and not info["installed"] and info["licence"] == "Apache-2.0"
    assert client.post("/api/models/download", json={"model": "flux-schnell"}).status_code == 400

    def fake_install(model, variant, consent, progress=None):
        for fl in variant.files:
            if progress:
                progress(fl.filename, fl.size, fl.size)
        return True

    monkeypatch.setattr(studio.models, "install_variant", fake_install)
    t = client.post(
        "/api/models/download", json={"model": "flux-schnell", "variant": "q4", "consent": True}
    ).json()
    assert t["kind"] == "download"
    for _ in range(100):
        tasks = client.get("/api/tasks").json()
        if tasks["download:flux-schnell"]["status"] != "running":
            break
        time.sleep(0.02)
    assert tasks["download:flux-schnell"]["status"] == "done"

    b = client.post("/api/benchmark", json={"preset": "fast", "target_minutes": 15}).json()
    assert b["kind"] == "benchmark"
    for _ in range(300):
        tasks = client.get("/api/tasks").json()
        if tasks["benchmark"]["status"] != "running":
            break
        time.sleep(0.05)
    assert tasks["benchmark"]["status"] == "done"
    assert tasks["benchmark"]["result"]["profile"]["recommended_preset"] in (
        "best",
        "balanced",
        "fast",
        None,
    )


def test_export_formats_scale_and_config(studio: Studio):
    client = TestClient(build_app(studio))
    job_id = client.post(
        "/api/generate",
        json={"prompt": "x", "preset": "fast", "width": 256, "height": 256, "seed": 1},
    ).json()["id"]
    for _ in range(200):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "done":
            break
        time.sleep(0.02)
    for fmt, magic in (("png", b"\x89PNG"), ("jpg", b"\xff\xd8"), ("webp", b"RIFF")):
        r = client.get(f"/api/download/{job_id}", params={"fmt": fmt, "quality": 80, "scale": 50})
        assert r.status_code == 200 and r.content.startswith(magic), fmt
    size = client.get(
        f"/api/export-size/{job_id}", params={"fmt": "jpg", "quality": 60, "scale": 50}
    ).json()
    assert size["bytes"] > 0 and size["width"] == 128  # half of the 256 px job
    assert client.get(f"/api/download/{job_id}", params={"fmt": "bmp"}).status_code == 404

    cfg = client.get("/api/config").json()
    assert cfg["allow_over_budget"] is False
    cfg = client.post("/api/config", json={"allow_over_budget": True, "default_threads": 6}).json()
    assert cfg["allow_over_budget"] is True and cfg["default_threads"] == 6
    assert studio.settings.config_file.exists()
    assert client.get("/api/plan", params={"variant": "q8"}).status_code == 200  # unlocked now
    client.post("/api/config", json={"allow_over_budget": False})
    assert client.get("/api/plan", params={"variant": "q8"}).status_code == 400


def test_parse_and_memory_endpoints(studio: Studio):
    client = TestClient(build_app(studio))
    p = client.get("/api/parse", params={"prompt": "a dog --seed 5 --ar 4:3"}).json()
    assert (
        p["prompt"] == "a dog" and p["params"]["seed"] == 5 and p["params"]["aspect"] == [4.0, 3.0]
    )
    m = client.get("/api/memory").json()
    assert m["total_gib"] > 0 and isinstance(m["top"], list) and m["top"][0]["gib"] >= 0
