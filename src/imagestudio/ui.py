"""The studio's own local web UI: one HTML page plus a small JSON API over the same Studio
the MCP tools use. Settings panel with a live load meter, the queue, and a gallery."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from imagestudio.registry import GIB, MODELS
from imagestudio.service import Studio, StudioError

STATIC = Path(__file__).parent / "static"


def _int(v: str | None) -> int | None:
    return int(v) if v not in (None, "") else None


def _float(v: str | None) -> float | None:
    return float(v) if v not in (None, "") else None


def _str(v: str | None) -> str | None:
    return v if v not in (None, "", "auto") else None


def build_app(studio: Studio, on_remote_change=None) -> Starlette:
    """The local web app. ``on_remote_change(enabled)`` is called when the user toggles
    remote access, so a host process (the Windows app) can rebind the listener."""
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import RedirectResponse

    from imagestudio import remote as rmt

    devices = rmt.Devices(studio.settings.devices_file)
    public_prefixes = ("/pair/", "/manifest.webmanifest", "/icon.svg")

    class DeviceAuth(BaseHTTPMiddleware):
        """Loopback requests are the owner. Anything else needs a paired device token."""

        async def dispatch(self, request, call_next):
            host = request.client.host if request.client else None
            if rmt.is_loopback(host) or request.url.path.startswith(public_prefixes):
                return await call_next(request)
            token = request.cookies.get(rmt.COOKIE) or request.headers.get(rmt.HEADER)
            dev = devices.authenticate(token)
            if dev is None:
                if request.url.path.startswith("/api/"):
                    return JSONResponse(
                        {"error": "not paired with this studio; scan the QR code on Settings"},
                        status_code=401,
                    )
                return HTMLResponse(
                    (STATIC / "pair.html").read_text(encoding="utf-8"), status_code=401
                )
            request.state.device = dev
            return await call_next(request)

    async def index(_: Request) -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    async def manifest(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "name": "Image Studio by Hivemind",
                "short_name": "Image Studio",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0f1012",
                "theme_color": "#0f1012",
                "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
            },
            media_type="application/manifest+json",
        )

    async def icon(_: Request) -> Response:
        return Response(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" h'
            'eight="64" rx="14" fill="#0f1012"/><path d="M32 6 54 19v26L32 58 10 45V19z" fi'
            'll="#f0b429"/><circle cx="32" cy="32" r="12" fill="#1b1b19"/><circle cx="32" c'
            'y="32" r="5.5" fill="#f7f6f2"/></svg>',
            media_type="image/svg+xml",
        )

    # -- remote access and pairing (the owner's machine only, except completing a pairing) --

    def _owner_only(request: Request) -> JSONResponse | None:
        host = request.client.host if request.client else None
        if rmt.is_loopback(host):
            return None
        return JSONResponse({"error": "only from the studio's own machine"}, status_code=403)

    async def remote_status(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "enabled": studio.settings.remote_enabled,
                "addresses": rmt.lan_addresses(),
                "port": request.url.port or 8766,
                "devices": devices.list(),
                "pending": devices.pending_count(),
                "this_device": getattr(request.state, "device", None).public()
                if getattr(request.state, "device", None)
                else None,
            }
        )

    async def remote_set(request: Request) -> JSONResponse:
        if (deny := _owner_only(request)) is not None:
            return deny
        body = await request.json()
        enabled = bool(body.get("enabled"))
        studio.set_config({"remote_enabled": enabled})
        restarted = False
        if on_remote_change is not None:
            try:
                restarted = bool(on_remote_change(enabled))
            except Exception as e:  # the toggle is saved either way
                return JSONResponse({"enabled": enabled, "error": str(e)}, status_code=500)
        return JSONResponse(
            {
                "enabled": enabled,
                "restarted": restarted,
                "restart_required": not restarted and on_remote_change is None,
            }
        )

    async def pair_start(request: Request) -> JSONResponse:
        if (deny := _owner_only(request)) is not None:
            return deny
        if not studio.settings.remote_enabled:
            return JSONResponse({"error": "enable phone access first"}, status_code=400)
        body = (
            await request.json()
            if request.headers.get("content-length", "0") not in ("0", "")
            else {}
        )
        pc = devices.start_pairing(str(body.get("name") or "phone"))
        addrs = rmt.lan_addresses()
        port = request.url.port or 8766
        urls = [f"http://{a}:{port}/pair/{pc.code}" for a in addrs]
        return JSONResponse(
            {
                "code": pc.code,
                "expires_in": int(pc.expires - __import__("time").time()),
                "urls": urls,
                "qr_svg": rmt.qr_svg(urls[0]) if urls else None,
            }
        )

    async def pair_complete(request: Request):
        code = request.path_params["code"]
        res = devices.complete_pairing(code, request.headers.get("user-agent", ""))
        if res is None:
            return HTMLResponse(
                (STATIC / "pair.html")
                .read_text(encoding="utf-8")
                .replace(
                    "<!--MSG-->",
                    "This pairing code has expired or was already used. "
                    "Ask the studio for a new QR code.",
                ),
                status_code=410,
            )
        _dev, token = res
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            rmt.COOKIE, token, max_age=3650 * 24 * 3600, httponly=True, samesite="lax", path="/"
        )
        return resp

    async def device_rename(request: Request) -> JSONResponse:
        if (deny := _owner_only(request)) is not None:
            return deny
        body = await request.json()
        ok = devices.rename(str(body.get("id", "")), str(body.get("name", "")))
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)

    async def device_revoke(request: Request) -> JSONResponse:
        if (deny := _owner_only(request)) is not None:
            return deny
        body = await request.json()
        if body.get("all"):
            return JSONResponse({"revoked": devices.revoke_all()})
        ok = devices.revoke(str(body.get("id", "")))
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)

    async def capabilities(_: Request) -> JSONResponse:
        return JSONResponse(studio.capabilities())

    async def plan(request: Request) -> JSONResponse:
        q = request.query_params
        try:
            p = studio.plan(
                _str(q.get("model")),
                q.get("preset", "best"),
                _int(q.get("width")),
                _int(q.get("height")),
                _str(q.get("variant")),
                _int(q.get("steps")),
                _float(q.get("cfg_scale")),
                _str(q.get("sampler")),
            )
        except (StudioError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(p.to_dict())

    async def generate(request: Request) -> JSONResponse:
        body: dict[str, Any] = await request.json()
        try:
            jobs = studio.generate_many(
                body.get("prompt", ""),
                count=int(body.get("count") or 1),
                preset=body.get("preset", "best"),
                width=body.get("width") or None,
                height=body.get("height") or None,
                seed=body.get("seed") if body.get("seed") not in ("", None) else None,
                negative_prompt=body.get("negative_prompt", ""),
                model=_str(body.get("model")),
                variant=_str(body.get("variant")),
                steps=body.get("steps") or None,
                cfg_scale=body.get("cfg_scale")
                if body.get("cfg_scale") not in ("", None)
                else None,
                sampler=_str(body.get("sampler")),
                threads=body.get("threads") or None,
                vae_tiling=bool(body.get("vae_tiling", False)),
                loras=body.get("loras") or None,
            )
        except (StudioError, ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        out = studio.job_status(jobs[0].id)
        out["jobs"] = [j.id for j in jobs]
        return JSONResponse(out)

    async def jobs(_: Request) -> JSONResponse:
        rows = []
        for j in studio.jobs.all():
            rows.append(studio.job_status(j.id))
        rows.reverse()
        return JSONResponse(
            {
                "jobs": rows,
                "worker_in_this_process": studio.worker_active,
                "paused": studio.jobs.is_paused(),
            }
        )

    async def job(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.job_status(request.path_params["job_id"]))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def cancel(request: Request) -> JSONResponse:
        return JSONResponse({"cancelled": studio.cancel(request.path_params["job_id"])})

    async def image(request: Request):
        j = studio.jobs.get(request.path_params["job_id"])
        if j is None or not j.output_path or not Path(j.output_path).exists():
            return JSONResponse({"error": "no image"}, status_code=404)
        return FileResponse(j.output_path, media_type="image/png")

    async def preview(request: Request):
        p = studio.preview_path(request.path_params["job_id"])
        if p is None:
            return JSONResponse({"error": "no preview yet"}, status_code=404)
        headers = {"Cache-Control": "no-store"}
        return Response(p.read_bytes(), media_type="image/png", headers=headers)

    async def export_size(request: Request) -> JSONResponse:
        q = request.query_params
        try:
            return JSONResponse(
                studio.export_size(
                    request.path_params["job_id"],
                    q.get("fmt", "png"),
                    int(q.get("quality", "90")),
                    int(q.get("scale", "100")),
                )
            )
        except (StudioError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def run_tool(request: Request) -> JSONResponse:
        body = await request.json()
        tool = request.path_params["tool"]
        try:
            if tool == "variations":
                job = studio.variations(
                    body.get("image", ""),
                    float(body.get("strength", 0.55)),
                    body.get("prompt", ""),
                    preset=body.get("preset", "best"),
                    seed=body.get("seed") or None,
                    model=_str(body.get("model")),
                )
            elif tool == "edit":
                job = studio.edit(
                    body.get("image", ""),
                    body.get("prompt", ""),
                    float(body.get("strength", 0.75)),
                    _str(body.get("mask")),
                    preset=body.get("preset", "best"),
                    seed=body.get("seed") or None,
                    model=_str(body.get("model")),
                )
            elif tool == "upscale":
                job = studio.upscale(
                    body.get("image", ""), _str(body.get("model")), int(body.get("repeats", 1))
                )
            elif tool == "remove_background":
                job = studio.remove_background(body.get("image", ""), _str(body.get("model")))
            else:
                return JSONResponse({"error": f"unknown tool {tool}"}, status_code=404)
        except (StudioError, ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(studio.job_status(job.id))

    async def describe(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.describe(
                    body.get("image", ""),
                    body.get("task", "describe"),
                    _str(body.get("question")),
                    _str(body.get("model")),
                )
            )
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def vision_download(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return JSONResponse({"error": "consent required"}, status_code=400)
        try:
            return JSONResponse(studio.start_vision_download(body.get("name", "")).to_dict())
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def engine_download(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return JSONResponse({"error": "consent required"}, status_code=400)
        try:
            return JSONResponse(studio.start_engine_download(body.get("which", "image")).to_dict())
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def components(_: Request) -> JSONResponse:
        return JSONResponse({"components": studio.components()})

    async def remove_component(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(studio.remove_component(body.get("kind", ""), body.get("name", "")))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def reveal(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            studio.reveal(body.get("path", ""))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True})

    async def delete_job(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.delete_job(request.path_params["job_id"]))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def restore_job(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.restore_job(request.path_params["job_id"]))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def trash(_: Request) -> JSONResponse:
        rows = [
            studio.job_status(j.id)
            if False
            else {
                "id": j.id,
                "title": j.prompt,
                "deleted_at": j.deleted_at,
                "width": j.width,
                "height": j.height,
            }
            for j in studio.jobs.all(include_deleted=True)
            if j.deleted_at
        ]
        return JSONResponse({"trash": rows})

    async def empty_trash(_: Request) -> JSONResponse:
        return JSONResponse({"removed": studio.empty_trash(0)})

    async def favorite(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.set_favorite(request.path_params["job_id"], bool(body.get("favorite", True)))
            )
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def import_image(request: Request) -> JSONResponse:
        data = await request.body()
        if not data or len(data) > 64 * 1024 * 1024:
            return JSONResponse({"error": "empty or too large (64 MB max)"}, status_code=400)
        try:
            job = studio.import_image(data, request.query_params.get("name", "imported image"))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(studio.job_status(job.id))

    async def import_path(request: Request) -> JSONResponse:
        """Import an image from a path on this machine (the windowed app's file dialog)."""
        body = await request.json()
        p = Path(str(body.get("path", "")))
        if not p.is_file():
            return JSONResponse({"error": f"not a file: {p}"}, status_code=400)
        try:
            job = studio.import_image(p.read_bytes(), p.stem)
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(studio.job_status(job.id))

    async def lora_import_path(request: Request) -> JSONResponse:
        body = await request.json()
        p = Path(str(body.get("path", "")))
        if not p.is_file():
            return JSONResponse({"error": f"not a file: {p}"}, status_code=400)
        try:
            return JSONResponse(studio.import_lora(p.read_bytes(), p.name))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def lineage(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.lineage(request.path_params["job_id"]))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def upload(request: Request) -> JSONResponse:
        data = await request.body()
        if not data or len(data) > 64 * 1024 * 1024:
            return JSONResponse({"error": "empty or too large (64 MB max)"}, status_code=400)
        try:
            p = studio.save_upload(data)
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"path": str(p)})

    async def custom_models(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "models": studio.custom_models(),
                "families": studio.custom_families(),
                "problems": studio.custom_model_problems,
            }
        )

    async def custom_model_add(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(studio.add_custom_model(body))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def custom_model_remove(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            studio.remove_custom_model(str(body.get("name", "")))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True})

    async def worker_handover(request: Request) -> JSONResponse:
        if (deny := _owner_only(request)) is not None:
            return deny
        try:
            res = studio.request_worker_handover()
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(res)

    async def worker_start(request: Request) -> JSONResponse:
        if (deny := _owner_only(request)) is not None:
            return deny
        return JSONResponse({"worker_in_this_process": studio.try_start_worker()})

    async def loras(_: Request) -> JSONResponse:
        return JSONResponse({"loras": studio.loras(), "folder": str(studio.settings.loras_dir)})

    async def lora_upload(request: Request) -> JSONResponse:
        data = await request.body()
        if len(data) > 2 * 1024 * 1024 * 1024:
            return JSONResponse({"error": "too large (2 GB max)"}, status_code=400)
        try:
            return JSONResponse(
                studio.import_lora(data, request.query_params.get("name", "lora.safetensors"))
            )
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def lora_remove(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse({"freed_bytes": studio.remove_lora(str(body.get("name", "")))})
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def upload_view(request: Request):
        p = (studio.settings.data_dir / "uploads" / request.path_params["name"]).resolve()
        if (
            not str(p).startswith(str((studio.settings.data_dir / "uploads").resolve()))
            or not p.exists()
        ):
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(p))

    async def tool_download(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return JSONResponse({"error": "consent required"}, status_code=400)
        try:
            return JSONResponse(studio.start_tool_download(body.get("name", "")).to_dict())
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def parse_prompt(request: Request) -> JSONResponse:
        return JSONResponse(studio.parse_prompt(request.query_params.get("prompt", "")))

    async def memory(_: Request) -> JSONResponse:
        return JSONResponse(studio.memory_users())

    async def get_config(_: Request) -> JSONResponse:
        return JSONResponse(studio.get_config())

    async def set_config(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(studio.set_config(body))
        except (ValueError, TypeError, OSError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def download(request: Request):
        q = request.query_params
        try:
            data, mime, name = studio.export_image(
                request.path_params["job_id"],
                q.get("fmt", "png"),
                int(q.get("quality", "90")),
                int(q.get("scale", "100")),
            )
        except (StudioError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return Response(
            data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'}
        )

    async def samples(_: Request) -> JSONResponse:
        return JSONResponse(studio.samples_by_preset())

    async def download_info(request: Request) -> JSONResponse:
        q = request.query_params
        name = q.get("model", "")
        if name not in MODELS:
            return JSONResponse({"error": "unknown model"}, status_code=400)
        model = MODELS[name]
        try:
            plan = studio.plan(name, "best", None, None, _str(q.get("variant")))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        v = plan.variant
        missing = [f for f in v.files if not studio.models.file_installed(f)]
        return JSONResponse(
            {
                "model": model.name,
                "display_name": model.display_name,
                "variant": v.name,
                "licence": model.licence,
                "note": v.note,
                "files": [
                    {"filename": f.filename, "size_gib": round(f.size / GIB, 2), "source": f.source}
                    for f in missing
                ],
                "total_gib": round(sum(f.size for f in missing) / GIB, 2),
                "installed": not missing,
            }
        )

    async def start_download(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return JSONResponse({"error": "consent required"}, status_code=400)
        try:
            task = studio.start_download(body.get("model", ""), _str(body.get("variant")))
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(task.to_dict())

    async def start_benchmark(request: Request) -> JSONResponse:
        body = await request.json() if await request.body() else {}
        try:
            task = studio.start_benchmark(
                body.get("preset", "fast"), float(body.get("target_minutes", 15))
            )
        except StudioError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(task.to_dict())

    async def tasks(_: Request) -> JSONResponse:
        return JSONResponse(studio.task_list())

    return Starlette(
        middleware=[Middleware(DeviceAuth)],
        routes=[
            Route("/manifest.webmanifest", manifest),
            Route("/icon.svg", icon),
            Route("/api/remote", remote_status),
            Route("/api/remote", remote_set, methods=["POST"]),
            Route("/api/pair/start", pair_start, methods=["POST"]),
            Route("/pair/{code}", pair_complete),
            Route("/api/remote/devices/rename", device_rename, methods=["POST"]),
            Route("/api/remote/devices/revoke", device_revoke, methods=["POST"]),
            Route("/api/preview/{job_id}", preview),
            Route("/api/download/{job_id}", download),
            Route("/api/export-size/{job_id}", export_size),
            Route("/api/config", get_config),
            Route("/api/parse", parse_prompt),
            Route("/api/tools/{tool}", run_tool, methods=["POST"]),
            Route("/api/tools/download", tool_download, methods=["POST"]),
            Route("/api/upload", upload, methods=["POST"]),
            Route("/api/custom-models", custom_models),
            Route("/api/custom-models", custom_model_add, methods=["POST"]),
            Route("/api/custom-models/remove", custom_model_remove, methods=["POST"]),
            Route("/api/worker/handover", worker_handover, methods=["POST"]),
            Route("/api/worker/start", worker_start, methods=["POST"]),
            Route("/api/loras", loras),
            Route("/api/loras/upload", lora_upload, methods=["POST"]),
            Route("/api/loras/remove", lora_remove, methods=["POST"]),
            Route("/api/components", components),
            Route("/api/components/remove", remove_component, methods=["POST"]),
            Route("/api/reveal", reveal, methods=["POST"]),
            Route("/api/jobs/{job_id}", delete_job, methods=["DELETE"]),
            Route("/api/jobs/{job_id}/restore", restore_job, methods=["POST"]),
            Route("/api/jobs/{job_id}/favorite", favorite, methods=["POST"]),
            Route("/api/jobs/{job_id}/lineage", lineage),
            Route("/api/trash", trash),
            Route("/api/trash/empty", empty_trash, methods=["POST"]),
            Route("/api/import", import_image, methods=["POST"]),
            Route("/api/import-path", import_path, methods=["POST"]),
            Route("/api/loras/import-path", lora_import_path, methods=["POST"]),
            Route("/api/describe", describe, methods=["POST"]),
            Route("/api/vision/download", vision_download, methods=["POST"]),
            Route("/api/engines/download", engine_download, methods=["POST"]),
            Route("/api/uploads/{name}", upload_view),
            Route("/api/memory", memory),
            Route("/api/config", set_config, methods=["POST"]),
            Route("/api/samples", samples),
            Route("/api/models/download", download_info),
            Route("/api/models/download", start_download, methods=["POST"]),
            Route("/api/benchmark", start_benchmark, methods=["POST"]),
            Route("/api/tasks", tasks),
            Route("/", index),
            Route("/api/capabilities", capabilities),
            Route("/api/plan", plan),
            Route("/api/generate", generate, methods=["POST"]),
            Route("/api/jobs", jobs),
            Route("/api/jobs/{job_id}", job),
            Route("/api/jobs/{job_id}/cancel", cancel, methods=["POST"]),
            Route("/api/image/{job_id}", image),
        ],
    )


def serve_ui(host: str | None = None, port: int = 8766, start_worker: bool = True) -> None:
    """``host`` None: loopback, or every interface when remote access is enabled in config."""
    import uvicorn

    studio = Studio(start_worker=start_worker)
    if host is None:
        host = "0.0.0.0" if studio.settings.remote_enabled else "127.0.0.1"
    mode = "rendering jobs" if studio.worker_active else "submitting to the running worker"
    print(f"Hivemind Image Studio UI at http://{host}:{port}  ({mode})", flush=True)
    try:
        uvicorn.run(build_app(studio), host=host, port=port, log_level="warning")
    finally:
        studio.close()
