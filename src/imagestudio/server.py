"""The MCP server. Tool descriptions are written for an AI caller: what it does, when to use
it, what comes back."""

from __future__ import annotations

import sys
from typing import Any

from imagestudio.registry import GIB, MODELS
from imagestudio.service import Studio, StudioError

try:  # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]


INSTRUCTIONS = """Hivemind Image Studio: free, local, offline image generation.

Policy: quality is a setting, never a consequence. generate() defaults to preset "best"
(full resolution, full steps, the strongest permissively licensed model that fits this
machine). The machine decides how long that takes, never how good it is; on a CPU-only laptop
expect several minutes per image. Faster presets ("balanced", "fast") exist only when the
caller chooses them, with the trade-off stated.

Workflow: call capabilities() once to learn hardware, installed models and expected times.
Call generate() to queue a job: it returns a job id and an ETA immediately. Poll job_status()
no more often than every 10-15 seconds until status is "done", then call job_result() for
the image path (or base64). Never block waiting inside a single tool call.
"""


def build_server(studio: Studio | None = None) -> FastMCP:
    studio = studio or Studio()
    mcp = FastMCP(
        name="hivemind-imagestudio",
        instructions=INSTRUCTIONS,
    )

    def _err(e: Exception) -> dict[str, Any]:
        return {"error": str(e)}

    @mcp.tool(
        name="generate",
        description=(
            "Queue a text-to-image job and return its job id immediately. Use this to make a "
            "new image from a prompt. The default preset 'best' renders at full resolution and "
            "full steps on the strongest permissively licensed model that fits this machine; "
            "it is slow on CPUs (minutes per image) by design and never lowers quality to be "
            "faster. Choose preset 'balanced' (smaller image or fewer steps, roughly half the "
            "time) or 'fast' (preview-grade, roughly a quarter of the time) only when the "
            "caller accepts the trade-off. width/height default to the preset's size (1024 for "
            "best); they must be 256-2048 and are rounded down to multiples of 16. seed: omit "
            "for random, repeat to reproduce. model: omit (auto) or one of "
            + ", ".join(f"'{m}'" for m in MODELS)
            + ". Advanced overrides, normally left unset because the presets hold the tested "
            "values: steps (1-100; FLUX schnell is distilled to 4 and gains nothing from more), "
            "cfg_scale (0-20 guidance strength; 1.0 for FLUX schnell, 2.5 for Qwen-Image; above "
            "1 doubles the work), sampler (euler default; euler_a, heun, dpm++2m, ...), "
            "variant (q8|q6|q5|q4 weight quantisation; default: highest fidelity that fits "
            "memory), threads (CPU threads; default: physical cores), vae_tiling (decode in "
            "tiles to save memory on large images, slightly slower). Settings the memory "
            "budget cannot hold are refused with the exact numbers unless the human has set "
            "allow_over_budget; use plan() first to see the load meter. "
            "Returns {job_id, status, estimate_seconds, estimate_human, estimate_basis, "
            "model, variant, preset, width, height, steps, seed, fit_note, warnings}. "
            "Then poll job_status(job_id) every 10-15 seconds and fetch job_result(job_id) when "
            "status is 'done'. Errors (engine or model not installed, bad preset) come back as "
            "{error}."
        ),
    )
    def generate(
        prompt: str,
        preset: str = "best",
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str = "",
        model: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
        variant: str | None = None,
        threads: int | None = None,
        vae_tiling: bool = False,
        count: int = 1,
    ) -> dict[str, Any]:
        try:
            jobs = studio.generate_many(
                prompt,
                count=count,
                preset=preset,
                width=width,
                height=height,
                seed=seed,
                negative_prompt=negative_prompt,
                model=model,
                variant=variant,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                threads=threads,
                vae_tiling=vae_tiling,
            )
        except StudioError as e:
            return _err(e)
        job = jobs[0]
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
            "job_ids": [j.id for j in jobs],
            "status": job.status,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_human": st["eta_human"],
            "estimate_basis": job.estimate_basis,
            "queue_position": st["queue_position"],
            "model": job.model,
            "variant": job.variant,
            "preset": job.preset,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "cfg_scale": job.cfg_scale,
            "sampler": job.sampler,
            "seed": job.seed,
            "fit_note": job.fit_note,
            "over_budget": job.over_budget,
            "warnings": job.warnings,
        }

    @mcp.tool(
        name="plan",
        description=(
            "Dry run of generate(): same parameters, nothing is queued. Returns the resolved "
            "plan {model, variant, preset, width, height, steps, cfg_scale, sampler, estimate "
            "{seconds, human, calibrated, basis}, load {score 0-100+ percent of this "
            "machine's memory budget, label comfortable|tight|will swap|over budget, "
            "required_gib, budget_gib, available_gib, meaning}, fit_note, warnings, installed, "
            "over_budget}. Use it as the performance meter: call it while choosing settings "
            "to see expected time and memory before committing, and to explain to the human "
            "why a setting is refused. Returns {error} for settings over the memory budget "
            "when allow_over_budget is not set."
        ),
    )
    def plan(
        preset: str = "best",
        width: int | None = None,
        height: int | None = None,
        model: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler: str | None = None,
        variant: str | None = None,
    ) -> dict[str, Any]:
        try:
            p = studio.plan(model, preset, width, height, variant, steps, cfg_scale, sampler)
        except StudioError as e:
            return _err(e)
        return p.to_dict()

    @mcp.tool(
        name="job_status",
        description=(
            "Report progress of a job started by generate(). Use it to decide whether to wait "
            "longer; poll every 10-15 seconds, not continuously. Returns {id, status "
            "(queued|running|done|failed|cancelled), phase (loading|encoding|sampling|decoding|"
            "saving), progress 0-1, step, total_steps, seconds_per_step, eta_seconds, eta_human, "
            "estimate_basis, queue_position, message, error, output_path when done}. The ETA "
            "comes from live step timing while sampling and from this machine's calibration "
            "otherwise."
        ),
    )
    def job_status(job_id: str) -> dict[str, Any]:
        try:
            return studio.job_status(job_id)
        except StudioError as e:
            return _err(e)

    @mcp.tool(
        name="job_result",
        description=(
            "Fetch the finished image of a job. Call it when job_status reports status 'done'. "
            "Returns {ready: true, path (PNG on this machine's disk, kept permanently under the "
            "studio's output folder), mime_type, width, height, seed, model, prompt, "
            "total_seconds} and, when as_base64 is true, 'base64' with the PNG bytes (use that "
            "when the caller cannot read local files; it is large). If the job is not finished "
            "it returns {ready: false, status, status_detail} instead of blocking."
        ),
    )
    def job_result(job_id: str, as_base64: bool = False) -> dict[str, Any]:
        try:
            return studio.job_result(job_id, as_base64)
        except StudioError as e:
            return _err(e)

    @mcp.tool(
        name="variations",
        description=(
            "Queue variations of an existing image (image-to-image) and return a job id. image is "
            "a job id or a local image path; prompt defaults to the source job's prompt; strength "
            "0-1 is how far to move from the source (0.3 subtle, 0.55 default, 0.8 loose). Same "
            "presets and settings as generate(); the output keeps the source size. Poll job_status."
        ),
    )
    def variations(
        image: str,
        strength: float = 0.55,
        prompt: str = "",
        preset: str = "best",
        seed: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        try:
            job = studio.variations(image, strength, prompt, preset=preset, seed=seed, model=model)
        except StudioError as e:
            return _err(e)
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
            "status": job.status,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_human": st["eta_human"],
            "width": job.width,
            "height": job.height,
            "seed": job.seed,
            "strength": job.strength,
        }

    @mcp.tool(
        name="edit",
        description=(
            "Queue an edit of an existing image by instruction and return a job id. image is a job "
            "id or a local image path; prompt describes the wanted result ('the same scene at "
            "night'); strength 0-1 is how much may change (0.75 default). mask: optional job id or "
            "path to a black/white image of the same size; only the white area is repainted "
            "(inpainting). Same presets and settings as generate(). Poll job_status."
        ),
    )
    def edit(
        image: str,
        prompt: str,
        strength: float = 0.75,
        mask: str | None = None,
        preset: str = "best",
        seed: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        try:
            job = studio.edit(image, prompt, strength, mask, preset=preset, seed=seed, model=model)
        except StudioError as e:
            return _err(e)
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
            "status": job.status,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_human": st["eta_human"],
            "width": job.width,
            "height": job.height,
            "seed": job.seed,
            "strength": job.strength,
        }

    @mcp.tool(
        name="describe",
        description=(
            "Image to text, locally, with llama.cpp and Qwen2.5-VL (Apache-2.0). Returns {text, "
            "seconds, model} synchronously (a few seconds on a CPU with the 3B model). image is a "
            "job id "
            "or a local image path. task: 'describe' (2-3 neutral sentences), 'prompt' (a text-to-"
            "image prompt that would recreate the picture), 'alt' (one sentence of alt text), "
            "'tags' (comma-separated), or 'custom' with your own question. Needs the vision engine "
            "and "
            "a vision model (download_model name='qwen2.5-vl-3b')."
        ),
    )
    def describe(
        image: str, task: str = "describe", question: str | None = None, model: str | None = None
    ) -> dict[str, Any]:
        try:
            return studio.describe(image, task, question, model)
        except StudioError as e:
            return _err(e)

    @mcp.tool(
        name="upscale",
        description=(
            "Queue a 4x upscale of a finished image with Real-ESRGAN (BSD licence) and return a "
            "job "
            "id. image is a job id from generate() or a path to a PNG/JPEG/WebP on this machine. "
            "model: 'realesrgan-x4plus' (default, photos and renders) or 'realesrgan-anime-6b' "
            "(flat colours, line art). repeats=2 runs it twice for 16x. Takes tens of seconds per "
            "output megapixel on a CPU. Poll job_status, then job_result for the PNG. Use it to "
            "turn "
            "a 1024 px render into a sharp 4096 px file instead of rendering large."
        ),
    )
    def upscale(image: str, model: str | None = None, repeats: int = 1) -> dict[str, Any]:
        try:
            job = studio.upscale(image, model, repeats)
        except StudioError as e:
            return _err(e)
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
            "status": job.status,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_human": st["eta_human"],
            "width": job.width,
            "height": job.height,
            "model": job.model,
        }

    @mcp.tool(
        name="remove_background",
        description=(
            "Queue background removal for an image and return a job id; the result is a PNG with a "
            "transparent background (alpha matte). image is a job id or a path to a PNG/JPEG/WebP. "
            "model: 'isnet-general-use' (default, cleaner edges) or 'u2net' (faster, softer). "
            "Takes a few seconds on a CPU. Poll job_status, then job_result."
        ),
    )
    def remove_background(image: str, model: str | None = None) -> dict[str, Any]:
        try:
            job = studio.remove_background(image, model)
        except StudioError as e:
            return _err(e)
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
            "status": job.status,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_human": st["eta_human"],
            "model": job.model,
        }

    @mcp.tool(
        name="jobs",
        description=(
            "List all jobs known to the studio, newest first, with id, status, preset, model, "
            "the first 80 characters of the prompt and the output path when done. Use it to "
            "recover a job id or to see what is queued or running."
        ),
    )
    def jobs() -> dict[str, Any]:
        return {"jobs": studio.list_jobs()}

    @mcp.tool(
        name="cancel_job",
        description=(
            "Cancel a queued or running job. Returns {cancelled: bool}. A running job stops "
            "within seconds; its partial work is discarded."
        ),
    )
    def cancel_job(job_id: str) -> dict[str, Any]:
        return {"cancelled": studio.cancel(job_id)}

    @mcp.tool(
        name="capabilities",
        description=(
            "Describe this studio: the hardware found (CPU cores, RAM, GPU and VRAM if any), "
            "the engine and whether it is installed, every catalogue model with its licence, "
            "quantisation variants, which fit this machine's memory and which are downloaded, "
            "each preset with its trade-off in words and the expected time on this machine, "
            "the quality policy, the licence list and the weights excluded by policy. Call it "
            "once before the first generate() to set expectations and to tell the caller how "
            "long an image will take. No parameters."
        ),
    )
    def capabilities() -> dict[str, Any]:
        return studio.capabilities()

    def download_model_impl(name: str, consent: bool, variant: str | None) -> dict[str, Any]:
        from imagestudio.registry import TOOL_MODELS, VISION_MODELS

        vm = VISION_MODELS.get(name)
        if vm is not None:
            missing = [f for f in vm.files if not studio.models.file_installed(f)]
            info = {
                "model": vm.name,
                "kind": "vision",
                "licence": vm.licence,
                "files": [
                    {"filename": f.filename, "size_gib": round(f.size / GIB, 2), "source": f.source}
                    for f in missing
                ],
                "total_gib": round(sum(f.size for f in missing) / GIB, 2),
                "already_installed": not missing,
            }
            if not missing:
                return info | {"downloaded": False, "message": "already installed"}
            if not consent:
                return info | {
                    "downloaded": False,
                    "message": "Call again with consent=true after the human approves.",
                }
            try:
                ok = studio.models.install_vision(vm, lambda _d, _b: True)
            except Exception as e:  # surfaced to the caller
                return info | {"downloaded": False, "error": str(e)}
            return info | {"downloaded": ok}
        tm = TOOL_MODELS.get(name)
        if tm is None:
            return _err(StudioError(f"unknown model {name!r}"))
        info = {
            "model": tm.name,
            "kind": tm.kind,
            "licence": tm.licence,
            "files": [
                {
                    "filename": tm.file.filename,
                    "size_gib": round(tm.file.size / GIB, 2),
                    "source": tm.file.source,
                }
            ],
            "total_gib": round(tm.file.size / GIB, 2),
            "already_installed": studio.models.tool_installed(tm),
        }
        if info["already_installed"]:
            return info | {"downloaded": False, "message": "already installed"}
        if not consent:
            return info | {
                "downloaded": False,
                "message": "Call again with consent=true after the human approves.",
            }
        try:
            ok = studio.models.install_tool(tm, lambda _d, _b: True)
        except Exception as e:  # surfaced to the caller
            return info | {"downloaded": False, "error": str(e)}
        return info | {"downloaded": ok}

    @mcp.tool(
        name="download_model",
        description=(
            "Download a catalogue model's weights into the studio's models folder so generate() "
            "can use it offline afterwards. This is the only tool that uses the network and it "
            "needs explicit consent: call it first with consent=false to get the exact files, "
            "total size in GiB and licence, then again with consent=true after the human "
            "agrees. name is one of "
            + ", ".join(f"'{m}'" for m in MODELS)
            + ", or a tool model: 'realesrgan-x4plus', 'realesrgan-anime-6b', "
            "'isnet-general-use', 'u2net'"
            "; variant is optional (q8|q6|q5|q4; default: the highest fidelity that fits this "
            "machine). Downloads are resumable and SHA-256 verified. This call blocks until the "
            "download finishes, which can take a long time for ~10 GB; prefer the CLI "
            "`imagestudio download-model <name> --yes` for very large sets."
        ),
    )
    def download_model(
        name: str, consent: bool = False, variant: str | None = None
    ) -> dict[str, Any]:
        if name not in MODELS:
            return download_model_impl(name, consent, variant)
        model = MODELS[name]
        try:
            plan = studio.plan(name, "best", None, None, variant)
        except StudioError as e:
            return _err(e)
        v = plan.variant
        missing = [f for f in v.files if not studio.models.file_installed(f)]
        info = {
            "model": model.name,
            "variant": v.name,
            "licence": model.licence,
            "files": [
                {"filename": f.filename, "size_gib": round(f.size / GIB, 2), "source": f.source}
                for f in missing
            ],
            "total_gib": round(sum(f.size for f in missing) / GIB, 2),
            "already_installed": not missing,
            "fit_note": plan.fit_note,
        }
        if not missing:
            return info | {"downloaded": False, "message": "already installed"}
        if not consent:
            return info | {
                "downloaded": False,
                "message": "Call again with consent=true after the human approves this download.",
            }
        try:
            ok = studio.models.install_variant(model, v, consent=lambda _d, _b: True)
        except Exception as e:
            return info | {"downloaded": False, "error": str(e)}
        return info | {"downloaded": ok}

    return mcp


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765) -> None:
    server = build_server()
    if transport == "stdio":
        server.run("stdio")
    else:
        server.settings.host = host  # type: ignore[attr-defined]
        server.settings.port = port  # type: ignore[attr-defined]
        server.run("streamable-http")


if __name__ == "__main__":  # pragma: no cover
    serve(sys.argv[1] if len(sys.argv) > 1 else "stdio")
