"""The MCP surface: tool names, descriptions written for an AI caller, and round trips
through the server's own call_tool."""

from __future__ import annotations

import json
import time

import pytest

from imagestudio.server import build_server
from imagestudio.service import Studio

EXPECTED_TOOLS = {
    "generate",
    "job_status",
    "job_result",
    "jobs",
    "cancel_job",
    "capabilities",
    "download_model",
    "plan",
    "upscale",
    "remove_background",
    "variations",
    "edit",
}


def _payload(result):
    """Normalise mcp 1.x/2.x call_tool return shapes into the dict the tool returned."""
    if isinstance(result, tuple):
        content, structured = result
        if isinstance(structured, dict):
            return structured.get("result", structured)
        result = content
    if isinstance(result, dict):
        return result.get("result", result)
    for attr in ("structured_content", "structuredContent"):
        sc = getattr(result, attr, None)
        if sc:
            return sc.get("result", sc)
    content = getattr(result, "content", result)
    if isinstance(content, list) and content and hasattr(content[0], "text"):
        return json.loads(content[0].text)
    if isinstance(result, list) and result and hasattr(result[0], "text"):
        return json.loads(result[0].text)
    raise AssertionError(f"unexpected call_tool result: {result!r}")


async def test_tools_are_listed_with_ai_facing_descriptions(studio: Studio):
    server = build_server(studio)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS
    by_name = {t.name: t for t in tools}
    gen = by_name["generate"].description
    assert "best" in gen and "balanced" in gen and "fast" in gen
    assert "job id" in gen.lower() and "poll" in gen.lower()
    assert "consent" in by_name["download_model"].description
    assert "hardware" in by_name["capabilities"].description.lower()
    schema = getattr(by_name["generate"], "input_schema", None) or by_name["generate"].inputSchema
    assert schema["properties"]["preset"]["default"] == "best"


async def test_generate_status_result_round_trip(studio: Studio):
    server = build_server(studio)
    r = _payload(await server.call_tool("generate", {"prompt": "a red bicycle"}))
    assert r["status"] in ("queued", "running") and r["preset"] == "best"
    assert r["estimate_seconds"] > 0 and r["estimate_human"]
    job_id = r["job_id"]
    for _ in range(200):
        st = _payload(await server.call_tool("job_status", {"job_id": job_id}))
        if st["status"] == "done":
            break
        time.sleep(0.02)
    assert st["status"] == "done"
    res = _payload(await server.call_tool("job_result", {"job_id": job_id, "as_base64": True}))
    assert res["ready"] and res["mime_type"] == "image/png" and res["base64"]
    listing = _payload(await server.call_tool("jobs", {}))
    assert listing["jobs"][0]["id"] == job_id


async def test_errors_come_back_as_error_field_not_exceptions(studio: Studio):
    server = build_server(studio)
    r = _payload(await server.call_tool("generate", {"prompt": "x", "preset": "ultra"}))
    assert "error" in r and "preset" in r["error"]
    r = _payload(await server.call_tool("job_status", {"job_id": "job_nope"}))
    assert "error" in r


async def test_capabilities_state_policy_hardware_models_and_licences(studio: Studio):
    server = build_server(studio)
    caps = _payload(await server.call_tool("capabilities", {}))
    assert caps["policy"]["default_preset"] == "best"
    assert caps["hardware"]["physical_cores"] == 10
    assert caps["hardware"]["ram_total_gib"] == pytest.approx(16.0)
    flux = next(m for m in caps["models"] if m["name"] == "flux-schnell")
    assert "q5" in flux["installed_variants"]
    best = next(p for p in flux["presets"] if p["preset"] == "best")
    assert best["width"] == 1024 and best["expected_time"]["human"]
    assert best["fit_note"]  # q8 does not fit 16 GB; this is said, not hidden
    assert any(li["licence"] == "MIT" for li in caps["licences"])
    assert any(e["model"] == "FLUX.1 [dev]" for e in caps["excluded_by_policy"])


async def test_download_model_requires_consent(studio: Studio):
    server = build_server(studio)
    r = _payload(
        await server.call_tool("download_model", {"name": "flux-schnell", "variant": "q4"})
    )
    assert r["downloaded"] is False and r["total_gib"] > 5 and "consent" in r["message"]
    assert all(f["filename"].endswith((".gguf", ".safetensors")) for f in r["files"])
    r = _payload(await server.call_tool("download_model", {"name": "nope"}))
    assert "error" in r
    r = _payload(
        await server.call_tool("download_model", {"name": "flux-schnell", "variant": "q5"})
    )
    assert r["already_installed"] is True


async def test_plan_tool_is_a_dry_run_with_load_meter(studio: Studio):
    server = build_server(studio)
    before = len(studio.list_jobs())
    p = _payload(await server.call_tool("plan", {"preset": "balanced"}))
    assert p["width"] == 768 and p["load"]["label"] in ("comfortable", "tight", "will swap")
    assert p["estimate"]["human"]
    assert len(studio.list_jobs()) == before
    r = _payload(await server.call_tool("plan", {"variant": "q8"}))
    assert "error" in r and "locked" in r["error"]


async def test_upscale_and_remove_background_tools_exist(studio: Studio):
    server = build_server(studio)
    names = {t.name for t in await server.list_tools()}
    assert {"upscale", "remove_background"} <= names
    r = _payload(await server.call_tool("upscale", {"image": "job_nope"}))
    assert "error" in r
    info = _payload(await server.call_tool("download_model", {"name": "realesrgan-anime-6b"}))
    assert (
        info["kind"] == "upscaler" and info["downloaded"] is False and "consent" in info["message"]
    )
