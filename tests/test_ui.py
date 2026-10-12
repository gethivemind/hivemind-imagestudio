"""The local web UI's JSON API, through Starlette's test client."""

from __future__ import annotations

import time

from starlette.testclient import TestClient

from imagestudio.service import Studio
from imagestudio.ui import build_app


def test_ui_page_plan_generate_and_image(studio: Studio):
    client = TestClient(build_app(studio))
    assert "Hivemind Image Studio" in client.get("/").text
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
