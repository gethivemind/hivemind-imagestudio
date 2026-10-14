"""Phone and remote access: devices, pairing codes, request authentication."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from starlette.testclient import TestClient

from imagestudio import remote
from imagestudio.service import Studio
from imagestudio.ui import build_app


def test_devices_pair_authenticate_revoke(tmp_path):
    d = remote.Devices(tmp_path / "devices.json")
    assert d.list() == [] and d.authenticate("nope") is None
    pc = d.start_pairing("Ratul's phone")
    assert d.pending_count() == 1
    assert d.complete_pairing("wrong") is None
    dev, token = d.complete_pairing(pc.code, "Mozilla/5.0 (Android)")
    assert d.complete_pairing(pc.code) is None  # one-time
    assert dev.name == "Ratul's phone" and len(token) > 30
    assert d.authenticate(token).id == dev.id
    stored = json.loads((tmp_path / "devices.json").read_text())
    assert token not in json.dumps(stored)  # only the hash is on disk
    assert remote.Devices(tmp_path / "devices.json").authenticate(token).id == dev.id
    assert d.rename(dev.id, "Pixel") and d.list()[0]["name"] == "Pixel"
    assert d.revoke(dev.id) and d.authenticate(token) is None
    assert d.revoke(dev.id) is False
    assert "<svg" in remote.qr_svg("http://192.168.1.2:8766/pair/abc")


def test_remote_requests_need_a_paired_device(studio: Studio):
    app = build_app(studio)
    owner = TestClient(app)  # loopback: the owner
    status = owner.get("/api/remote").json()
    assert status["enabled"] is False and status["devices"] == []
    # pairing needs remote access on
    assert owner.post("/api/pair/start", json={}).status_code == 400
    assert owner.post("/api/remote", json={"enabled": True}).json()["enabled"] is True
    pair = owner.post("/api/pair/start", json={"name": "phone"}).json()
    assert pair["code"] and pair["qr_svg"] and pair["urls"][0].endswith(pair["code"])

    phone = TestClient(app, client=("192.168.1.50", 40000))
    denied = phone.get("/api/capabilities")
    assert denied.status_code == 401 and "not paired" in denied.json()["error"]
    page = phone.get("/")
    assert page.status_code == 401 and "not paired" in page.text
    # completing the pairing sets the device cookie; the phone is in
    done = phone.get(f"/pair/{pair['code']}", follow_redirects=False)
    assert done.status_code == 303 and remote.COOKIE in done.cookies
    assert phone.get("/api/capabilities").status_code == 200
    # the owner-only endpoints still refuse a paired phone
    assert phone.post("/api/pair/start", json={}).status_code == 403
    assert phone.post("/api/remote", json={"enabled": False}).status_code == 403
    me = phone.get("/api/remote").json()
    assert me["this_device"]["name"] == "phone"
    # a second use of the code fails; the owner sees and revokes the device
    assert phone.get(f"/pair/{pair['code']}").status_code == 410
    devs = owner.get("/api/remote").json()["devices"]
    assert len(devs) == 1
    assert owner.post("/api/remote/devices/revoke", json={"id": devs[0]["id"]}).json()["ok"]
    assert phone.get("/api/capabilities").status_code == 401
    # the manifest and icon are public
    assert phone.get("/manifest.webmanifest").status_code == 200
    assert phone.get("/icon.svg").status_code == 200
