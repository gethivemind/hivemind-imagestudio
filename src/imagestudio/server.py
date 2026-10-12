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
            + ". Returns {job_id, status, estimate_seconds, estimate_human, estimate_basis, "
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
    ) -> dict[str, Any]:
        try:
            job = studio.generate(prompt, preset, width, height, seed, negative_prompt, model)
        except StudioError as e:
            return _err(e)
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
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
            "seed": job.seed,
            "fit_note": job.fit_note,
            "warnings": job.warnings,
        }

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

    @mcp.tool(
        name="download_model",
        description=(
            "Download a catalogue model's weights into the studio's models folder so generate() "
            "can use it offline afterwards. This is the only tool that uses the network and it "
            "needs explicit consent: call it first with consent=false to get the exact files, "
            "total size in GiB and licence, then again with consent=true after the human "
            "agrees. name is one of "
            + ", ".join(f"'{m}'" for m in MODELS)
            + "; variant is optional (q8|q6|q5|q4; default: the highest fidelity that fits this "
            "machine). Downloads are resumable and SHA-256 verified. This call blocks until the "
            "download finishes, which can take a long time for ~10 GB; prefer the CLI "
            "`imagestudio download-model <name> --yes` for very large sets."
        ),
    )
    def download_model(
        name: str, consent: bool = False, variant: str | None = None
    ) -> dict[str, Any]:
        if name not in MODELS:
            return _err(StudioError(f"unknown model {name!r}; known: {', '.join(MODELS)}"))
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
