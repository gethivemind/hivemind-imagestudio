# Hivemind Image Studio

A free, local, permissively licensed image MCP server. Text to image today; edits,
background removal and upscaling next. It runs on whatever machine you have, on the CPU
if that is all there is, and plugs into any MCP client (Hivemind, Claude Code, Cursor, ...).

Design: Hivemind ADR-051. Licence: Apache-2.0.

## The policy

**Quality is a setting, never a consequence.** The default preset is `best`: the strongest
permissively licensed model that fits your memory, at full resolution and full steps. The
machine decides how long that takes (threads, CPU or GPU, memory), never how good the result
is. Nothing downgrades silently to be faster: the studio estimates the time and tells you.

**Faster only with consent.** Presets `balanced` and `fast` exist and must be chosen
explicitly. Each states its trade-off in words, in the tool description and in
`capabilities()`:

| Preset (FLUX.1 schnell) | Size | Steps | Trade-off |
| --- | --- | --- | --- |
| `best` (default) | 1024 px | 4 | Full resolution, full steps. |
| `balanced` | 768 px | 4 | Roughly half the time, less fine detail, smaller image. |
| `fast` | 512 px | 4 | Roughly a quarter of the time, a preview-grade image. |

The one thing that can change *which weights* run is memory: weights that do not fit cannot
run at all. When a lower quantisation is used for that reason, the plan carries a `fit_note`
saying so, and resolution and steps stay unchanged.

**Reliable means a queue.** `generate` returns a job id at once. `job_status` reports phase,
step, a live ETA and the queue position. `job_result` returns the finished PNG as a path on
disk (kept permanently under the output folder) and as base64 on request. Jobs and outputs
are files, so nothing is lost when the server stops; running jobs are re-queued on restart.
Model downloads resume and are verified by SHA-256.

**Free means permissive licences only.** Generation: FLUX.1 [schnell] (Apache-2.0) and
Qwen-Image (Apache-2.0) in GGUF form, with their Apache-2.0 text encoders and VAEs. Engine:
stable-diffusion.cpp (MIT). Excluded by policy, whatever their quality: FLUX.1 [dev]
(non-commercial licence), RMBG-2.0 (CC BY-NC), Stable Diffusion 3.x (revenue-gated
community licence).

**Offline after setup.** The engine and the models download once, each after explicit
consent (`--yes` or `consent=true`), into the studio's own data folder. After that nothing
touches the network, and there are no keys or accounts.

## Hardware expectations, stated plainly

| Machine | 1024 px image at `best` |
| --- | --- |
| Laptop, 16 GB RAM, integrated graphics (no CUDA) | 5 to 10 minutes |
| Desktop CPU, 32 GB | 2 to 5 minutes |
| Mid-range discrete GPU (Vulkan or CUDA build) | seconds |

Memory decides which quantisation fits. FLUX.1 schnell needs about 18.6 GiB for the 8-bit
set, 13 GiB for 5-bit and 11 GiB for 4-bit at 1024 px (weights plus working memory); a 16 GB
machine runs the 5-bit set and `capabilities()` says so. Qwen-Image needs at least 17.5 GiB
even at 4-bit and does not fit a 16 GB machine; the studio reports that instead of trying.

Estimates start from a heuristic (labelled `uncalibrated`) and are replaced by timings from
real runs on your machine, kept per model variant in `calibration.json`.

## Install

Requires Python 3.11+.

```bash
pip install hivemind-imagestudio
```

or from a checkout:

```bash
uv venv && uv pip install -e ".[dev]"
```

Then:

```bash
imagestudio doctor
```

```bash
imagestudio install-engine --yes
```

```bash
imagestudio download-model flux-schnell --yes
```

`doctor` shows the hardware found, the engine, the models, which quantisations fit, and the
expected time per preset. `install-engine` fetches the pinned stable-diffusion.cpp release
for your OS (about 20 to 40 MB, hash-verified); set `IMAGESTUDIO_SD_BIN` to use a build of
your own instead, or put `sd-cli` on PATH. `download-model` picks the highest-fidelity set
that fits your memory (override with `--variant q8|q6|q5|q4`), about 10 GB for a 16 GB
machine, and asks before downloading unless `--yes` is given.

Generate from the command line:

```bash
imagestudio generate "a lighthouse at dusk, oil painting" --preset best
```

Run as an MCP server over stdio (the default) for an MCP client:

```json
{
  "mcpServers": {
    "imagestudio": { "command": "imagestudio", "args": ["serve"] }
  }
}
```

`imagestudio serve --transport http --port 8765` serves streamable HTTP instead.

### Where things live

Everything is under one data directory (`IMAGESTUDIO_DATA_DIR` to override):

| OS | Default |
| --- | --- |
| Windows | `%LOCALAPPDATA%\hivemind-imagestudio` |
| macOS | `~/Library/Application Support/hivemind-imagestudio` |
| Linux | `$XDG_DATA_HOME/hivemind-imagestudio` or `~/.local/share/hivemind-imagestudio` |

Inside: `models/` (weights and a manifest of verified hashes), `tools/` (the engine),
`out/` (every finished image, `<job_id>.png`), `jobs/` (one JSON per job),
`calibration.json`.

Other settings: `IMAGESTUDIO_THREADS` (default: physical cores), `IMAGESTUDIO_SDCPP_FLAVOR`
(`cpu`, the default, or `vulkan` for GPUs including integrated ones).

### Windows note

Endpoint protection (Trend Micro Apex One and similar) may flag the freshly downloaded
`sd-cli.exe` as a "newly encountered program" the first time it runs. It is the pinned
stable-diffusion.cpp release; its SHA-256 is in `registry.py`. Allow it, or add the
`tools/` folder under the data directory as a trusted path.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `generate(prompt, preset="best", width?, height?, seed?, negative_prompt?, model?)` | Queue a job; returns job id, estimate and the plan. |
| `job_status(job_id)` | Status, phase, progress, step, ETA, queue position. |
| `job_result(job_id, as_base64=false)` | Path to the PNG, plus base64 on request. |
| `jobs()` | All jobs, newest first. |
| `cancel_job(job_id)` | Cancel a queued or running job. |
| `capabilities()` | Hardware, engine, models and their fit, presets with expected times, licences, policy. |
| `download_model(name, consent=false, variant?)` | Report size and licence; download after consent. |

Tool descriptions are written for an AI caller: what the tool does, when to use it, what
comes back, and how often to poll.

## Licences

| Component | Licence |
| --- | --- |
| Hivemind Image Studio | Apache-2.0 |
| stable-diffusion.cpp, ggml | MIT |
| FLUX.1 [schnell] weights (GGUF by city96) | Apache-2.0 |
| T5-XXL v1.1 encoder (GGUF by city96), CLIP-L (comfyanonymous) | Apache-2.0 |
| FLUX VAE `ae.safetensors` | Apache-2.0 |
| Qwen-Image weights (GGUF by city96), Qwen-Image VAE | Apache-2.0 |
| Qwen2.5-VL-7B-Instruct text encoder (GGUF) | Apache-2.0 |

## Development

```bash
uv pip install -e ".[dev]" && ruff check src tests && pytest
```

Tests fake the engine and never need the models or the network. CI runs ruff and pytest on
Ubuntu and Windows.

## Roadmap

Increment 1 (this release): `generate` with the queue, hardware detection, hash-verified
model and engine downloads, calibration, CLI, MCP server.
Increment 2: `remove_background` (BiRefNet / rembg, MIT), `upscale` (Real-ESRGAN, BSD),
`edit` (inpaint), `variations`.
Later: optional ComfyUI engine, GPU builds (CUDA, ROCm), the Hivemind catalogue entry.
