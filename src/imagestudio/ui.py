"""The studio's own local web UI: one HTML page plus a small JSON API over the same Studio
the MCP tools use. Settings panel with a live load meter, the queue, and a gallery."""

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


def build_app(studio: Studio) -> Starlette:
    async def index(_: Request) -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

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
        return JSONResponse({"jobs": rows, "worker_in_this_process": studio.worker_active})

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
        routes=[
            Route("/api/preview/{job_id}", preview),
            Route("/api/download/{job_id}", download),
            Route("/api/export-size/{job_id}", export_size),
            Route("/api/config", get_config),
            Route("/api/parse", parse_prompt),
            Route("/api/tools/{tool}", run_tool, methods=["POST"]),
            Route("/api/tools/download", tool_download, methods=["POST"]),
            Route("/api/upload", upload, methods=["POST"]),
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
        ]
    )


def serve_ui(host: str = "127.0.0.1", port: int = 8766, start_worker: bool = True) -> None:
    import uvicorn

    studio = Studio(start_worker=start_worker)
    mode = "rendering jobs" if studio.worker_active else "submitting to the running worker"
    print(f"Hivemind Image Studio UI at http://{host}:{port}  ({mode})", flush=True)
    try:
        uvicorn.run(build_app(studio), host=host, port=port, log_level="warning")
    finally:
        studio.close()
