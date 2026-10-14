# Hivemind Image Studio

**Website:** https://gethivemind.github.io/hivemind-imagestudio/ · **PyPI:** https://pypi.org/project/hivemind-imagestudio/ · **Releases:** https://github.com/gethivemind/hivemind-imagestudio/releases

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
| Laptop, 16 GB RAM, integrated graphics (no CUDA), other apps open | about 2 hours (measured, below) |
| Same laptop with 12+ GiB free | roughly 20 to 40 minutes (estimated; not swapping) |
| Desktop CPU, 32 GB | 10 to 20 minutes (estimated) |
| Mid-range discrete GPU (Vulkan or CUDA build) | seconds |

### What to expect on a 16 GB laptop with no GPU

Measured on an Intel i5-1235U (10 cores), 15.6 GiB RAM, Intel UHD integrated graphics,
Windows 11, FLUX.1 schnell `q5` weights memory-mapped, same prompt and seed, while the owner
kept working in a browser, an IDE and this app:

| Preset | Size | Free RAM at start | Wall time | Sampling | Peak engine RSS |
| --- | --- | --- | --- | --- | --- |
| `fast` | 512 px | 1.7 GiB | 22 min 31 s | 20 min 43 s | 7.8 GiB |
| `balanced` | 768 px | 7.3 GiB | 48 min 30 s | 44 min 48 s | 7.6 GiB |
| `best` | 1024 px | 6.6 GiB | 1 h 56 min | 1 h 50 min | 7.6 GiB |

The engine wants about 13 GiB for this set; with less free it pages weights from the model
files all the way through, and the wall time is dominated by that paging, not by arithmetic.
The same `fast` run without memory mapping took 37 min 32 s. Free memory matters more than
preset: an overnight or lunch-break `best` is the realistic way to use this class of machine,
and `imagestudio plan` tells you the expected time before you commit.

Memory decides which quantisation fits. FLUX.1 schnell needs about 18.6 GiB for the 8-bit
set, 13 GiB for 5-bit and 11 GiB for 4-bit at 1024 px (weights plus working memory); a 16 GB
machine runs the 5-bit set and `capabilities()` says so. Qwen-Image needs at least 17.5 GiB
even at 4-bit and does not fit a 16 GB machine; the studio reports that instead of trying.

Estimates start from a heuristic (labelled `uncalibrated`) and are replaced by timings from
real runs on your machine, kept per model variant in `calibration.json`.

## Install

Requires Python 3.11+.

```bash
pip install "hivemind-imagestudio[bg]"
```

or, kept out of every other environment:

```bash
uv tool install "hivemind-imagestudio[bg]"
```

The `bg` extra adds ONNX Runtime for background removal; without it everything else works.
The GitHub Releases of this repository and the `hivemind-imagestudio` package on PyPI are
the official builds (see NOTICE and TRADEMARK.md). Inside Hivemind, Settings → Tools
installs and connects it for you.

Open the studio in your browser (everything still runs on your machine):

```bash
imagestudio ui
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
machine, and asks before downloading unless `--yes` is given. It also takes tool models
(`realesrgan-x4plus`, `isnet-general-use`, ...) and vision models (`qwen2.5-vl-3b`). For
image to text, install the second engine with `imagestudio install-engine --vision`
(llama.cpp, MIT, about 20 MB).

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

### Engine backends: CPU or Vulkan

The pinned stable-diffusion.cpp release ships two builds the studio knows how to install:
`cpu` (the default) and `vulkan`, which runs the model on the GPU through the Vulkan API,
including integrated graphics. Install the second build with `imagestudio install-engine
--flavor vulkan` (add `--use` to switch to it), or from Settings → Models and engines, where
each build has Install, Use and Remove. The active build is stored as `engine_backend` in
`config.json`; the environment variable `IMAGESTUDIO_SDCPP_FLAVOR` overrides it. Switching
is refused while jobs are queued or running, and the benchmark should be run again
afterwards because the time estimates are calibrated per backend. Whether Vulkan is faster
depends entirely on the GPU and its memory. Measured on the 16 GB laptop (12th-gen Core
i5-1235U, Intel UHD integrated graphics, no discrete GPU), FLUX.1-schnell q5, 512², 4 steps,
seed 7, identical output:

| Backend | Wall time | To first step | Peak RSS |
| --- | --- | --- | --- |
| CPU | 30 min 5 s | 27 min 57 s | 6.99 GiB |
| Vulkan (integrated GPU) | 12 min 43 s | 7 min 57 s | 5.25 GiB |

Same image, 2.4× faster, less resident memory: the Vulkan build is worth installing even on
an integrated GPU. Both runs were on a loaded machine (browser, IDE), so absolute numbers vary;
the ratio is the point.

### Qwen-Image-Edit: editing by instruction

`qwen-image-edit` (Qwen-Image-Edit 2509, Apache-2.0, QuantStack GGUF with pinned hashes) is an
*edit model*: it takes the picture plus an instruction ("the same scene at night, lit windows",
"turn this sketch into a photo of a cabin") and keeps what you did not mention. Once a
variant is downloaded (`imagestudio download-model qwen-image-edit`, or Settings → Models and
engines), Edit without a mask and Sketch → Paint it use it automatically; masked edits,
Remove object and Variations stay on image-to-image with the base model, where the strength
slider applies. `auto` never picks an edit model for plain generation; name it with
`--model qwen-image-edit` only on edits. It shares the VAE and the 7B text encoder with
Qwen-Image and is just as memory-hungry: the q3 set is about 15 GB of weights, so on a 16 GB
machine it runs over budget (memory-mapped, slow) and needs the explicit unlock. With the
unlock set, the planner prefers a downloaded higher-fidelity set over a smaller one it would
have to fetch, and says so.

### LoRA

Drop adapter files (`.safetensors`, `.gguf`) into `<models>/loras`, or import them on
Settings → LoRA. Use them with `--lora name:0.8` in the prompt (repeatable; the LoRA picker
on Create writes the same text), with `loras=["name:0.8"]` on the MCP tool, or `--lora` on
the CLI. The weight is written into the reproducible prompt and the PNG metadata. The studio
cannot check that a LoRA was trained for the base model you generate with; a mismatch shows
as an engine error on the first job.

### Your own models

Settings → Your own models registers weights you already have: a FLUX or Qwen-Image finetune
(GGUF or safetensors, reusing the studio's CLIP-L, T5 and VAE downloads), or a Stable
Diffusion 1.5 / XL checkpoint. Entries live in `config.json` under `custom_models` and point
at your files; nothing is downloaded or hash-checked, the licence is what you declare, and
the model is labelled **your own**, never **verified**. `auto` never picks a custom model;
choose it by name (`--model my-flux`). Time estimates are rough until the benchmark has run
with it. This is the "bring your own model" tier from the roadmap; verified models stay the
pinned, hash-checked catalogue.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `generate(prompt, preset="best", width?, height?, seed?, negative_prompt?, model?, steps?, cfg_scale?, sampler?, variant?, threads?, vae_tiling?)` | Queue a text-to-image job; returns job id, estimate and the plan. Inline `--parameters` in the prompt are honoured. |
| `variations(image, strength=0.4, prompt?, preset?, seed?, model?)` | Image-to-image re-roll of a job id or local image. Strength is a cliff on 4-step models: up to about 0.45 keeps the composition, from about 0.5 it is re-invented (measured). |
| `edit(image, prompt, strength=0.45, mask?, preset?, seed?, model?)` | Change an image by instruction. Without a mask and with `qwen-image-edit` installed, the edit model does it (no strength); otherwise image-to-image on the base model, where strength applies; with a black/white mask only the white area is repainted. |
| `upscale(image, model?, repeats=1)` | 4× (or 16×) with Real-ESRGAN. |
| `remove_background(image, model?)` | Transparent PNG with IS-Net or U²-Net. |
| `describe(image, task="describe", question?, model?)` | Image to text with llama.cpp and Qwen2.5-VL: description, prompt, alt text, tags, or your own question. Synchronous. |
| `plan(...)` | Dry run of generate(): expected time and the load meter. |
| `job_status(job_id)` | Status, phase, progress, step, ETA, queue position. |
| `job_result(job_id, as_base64=false)` | Path to the PNG, plus base64 on request. |
| `jobs()` / `cancel_job(job_id)` | List, cancel. |
| `capabilities()` | Hardware, engine, models and their fit, tool models, presets with expected times, licences, policy. |
| `download_model(name, consent=false, variant?)` | Report size and licence; download after consent. Accepts generation and tool models. |

Tool descriptions are written for an AI caller: what the tool does, when to use it, what
comes back, and how often to poll.

### `generate` parameters

Every parameter is optional except the prompt. The same parameters are available on the
CLI (`imagestudio generate --help`) and will be mirrored in Hivemind's duty form.

| Parameter | Default | What it does |
| --- | --- | --- |
| `prompt` | required | What to draw. Be concrete: subject, setting, style, lighting. FLUX follows long prompts well. |
| `preset` | `best` | `best` = full resolution and full steps. `balanced` and `fast` trade quality for time; see the preset table. |
| `width`, `height` | from preset | 256 to 2048 px, rounded down to a multiple of 16. Any aspect ratio. More pixels means proportionally more time and memory. |
| `seed` | random | The random seed. Same prompt, settings and seed reproduce the same image; a different seed gives a different composition. The seed used is returned with every job. |
| `negative_prompt` | empty | What to avoid. Has little effect on FLUX schnell, which runs without guidance (`cfg_scale` 1). |
| `model` | auto | `flux-schnell`, `qwen-image`, or omit for the strongest installed model that fits memory. |
| `steps` | from preset | Advanced. Denoising steps, 1 to 100. FLUX schnell is distilled to 4 and gains nothing from more; Qwen-Image uses 50. The studio warns when extra steps only cost time. |
| `cfg_scale` | from preset | Advanced. Guidance strength, 0 to 20. 1.0 for FLUX schnell, 2.5 for Qwen-Image. Above 1 the model runs twice per step. |
| `sampler` | `euler` | Advanced. `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`. |
| `variant` | by memory fit | Weight quantisation `q8`, `q6`, `q5`, `q4`. Default: the highest fidelity that fits this machine's memory budget (reported in `fit_note`). Forcing one that does not fit is refused unless over-budget runs are unlocked. |
| `threads` | physical cores | CPU threads for the engine. Fewer threads leave the machine more responsive; more than the physical core count rarely helps. |
| `vae_tiling` | false | Decode the final image in tiles. Saves memory on large images at a small time cost; no quality change. |
| `loras` | none | Adapters from the loras folder as `name` or `name:weight` (weight -4 to 4, 1.0 default). In the prompt: `--lora name:0.8`, repeatable. |
| `count` | 1 | Queue this many images of the prompt with consecutive seeds (1 to 16). In the prompt: `--n 4`. Each is a normal job; the response lists all their ids. |

### The load meter: `plan`

`plan` takes the same settings as `generate` and queues nothing. It returns the resolved
plan, the expected time on this machine, and a **load** reading: the percent of the
machine's memory budget the job needs, with a label.

| Label | Meaning |
| --- | --- |
| `comfortable` | Fits in free memory with room to spare. |
| `tight` | Fits the budget, close to the machine's limit. |
| `will swap` | Fits total memory but not what is free right now. Runs, several times slower; weights are memory-mapped so the OS pages them from the model files. |
| `over budget` | Needs more memory than the machine has. Refused with the exact numbers. |

```bash
imagestudio plan --preset best --width 1536 --height 1536
```

A UI can call `plan` on every change of a knob and draw the meter live, like a game's
graphics settings screen. The memory budget is total RAM minus 2 GiB (plus VRAM of a
discrete GPU); time is never limited, only reported.

**Unlocking over-budget runs.** Time is never capped, but memory that does not exist cannot
be used, so settings over the budget are refused by default. To try anyway, set
`IMAGESTUDIO_ALLOW_OVER_BUDGET=1` or write `{"allow_over_budget": true}` into `config.json`
in the data directory. Every such job is marked `over_budget`; if the engine cannot allocate,
the job fails with the exact error and the memory it wanted, and nothing is downgraded.

### Benchmark: detect settings for this machine

```bash
imagestudio benchmark --target-minutes 15
```

Runs one real image (preset `fast` by default) through the normal queue, measures step time,
wall time, peak engine memory and whether the run swapped, then writes `profile.json` with
the expected time of every preset and the highest-quality preset that meets the target. It
is a recommendation, shown by `doctor` and `capabilities()`; the default stays `best`, and the
user may always pick a slower setting. Every later job refines the same calibration.

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

The code is Apache-2.0: use it, fork it, ship it. The names "Hivemind", "Hivemind Image
Studio" and the logos are trademarks and are not licensed with the code; forks need their own
name and may say "based on Hivemind Image Studio". The full policy is in [TRADEMARK.md](TRADEMARK.md).
Official builds are the GitHub Releases of this repository and the `hivemind-imagestudio`
package on PyPI.

Image Studio by Hivemind · made with ♥ by [Ratul Roy (royal-programmer)](https://github.com/royal-programmer).
Cite it with the `CITATION.cff` in the repository.

## Development

```bash
uv pip install -e ".[dev]" && ruff check src tests && pytest
```

Tests fake the engine and never need the models or the network. CI runs ruff and pytest on
Ubuntu and Windows.

## Roadmap

Increment 1 (this release): `generate` with the queue, hardware detection, hash-verified
model and engine downloads, calibration, CLI, MCP server.
Increment 2 (shipped): `upscale` (Real-ESRGAN, BSD-3-Clause), `remove_background` (IS-Net /
U²-Net, Apache-2.0, via ONNX Runtime), `variations` and `edit` with inpainting masks, a sketch
canvas in the UI, prompt parameters and reproducible prompts, the local web UI.
Also shipped: `describe` (image to text with llama.cpp and Qwen2.5-VL 3B/7B), a Home screen
of tasks with per-machine readiness and a getting-started checklist, optional auto-captions
(Settings → Preferences: every finished image is described in the background once the vision
stack is installed; captions show in the viewer and are searchable), a keyboard-shortcuts
overlay (`?`), and a purpose-built sketch canvas: brush with opacity, softness and pen
pressure, eraser, line, rectangle, ellipse (Shift constrains), fill, eyedropper, a faded
reference layer under the drawing, pan (Space, middle button, wheel) and zoom towards the
cursor. The sketch is deliberately not a paint program: at the strengths that make
sketch-to-image work, fine detail is re-invented by the model anyway. Better control will
come from conditioning (scribble or edge ControlNet) rather than from more brushes.
Also: every finished PNG carries its recipe as metadata (`parameters` in the common
Automatic1111 layout plus an `imagestudio` JSON chunk with the reproducible prompt; exports
keep it and re-importing such a file restores the prompt), `--n 4` queues a batch of
consecutive seeds (also `count` on the MCP tool and the ×N control in the UI), desktop
notifications when a job finishes while the tab is in the background, a one-click Remove
object (paint a mask, no prompt), Recreate (vision model writes the prompt, Create opens
with it at the same size), an A/B compare slider in the viewer, and prompt history with
pinned prompts stored in config.json. The viewer fits the image to the window, zooms
towards the cursor with the wheel, pans by dragging, and toggles fit / 100% on double-click.
Next: Qwen-Image-Edit as an instruction-edit model, LoRA, GPU builds (Vulkan, CUDA), optional
cloud engines behind the same interface, a marketing site.
Packaged app (noted 2026-10-14, after the first PyPI release): today the studio is a command-line
program plus a browser UI (`imagestudio ui`) and an MCP server (`imagestudio serve`); it needs
Python 3.11+ or uv, and something has to run those commands (inside Hivemind, the hub does; in
Claude Desktop or Cursor, the MCP config does). A double-click experience is a later step: a
packaged build with a tray icon that keeps `serve` and `ui` running, an installer per platform,
and no Python on the machine. Planned after the first external users arrive, since Hivemind
users already have the one-click path through Settings → Tools.
Later: optional ComfyUI engine, GPU builds (CUDA, ROCm), the Hivemind catalogue entry,
and a three-tier model list (Hivemind-verified, compatible-but-unverified, bring your own
model with a declared licence).

Video (placeholder, scope only): `generate_video` (text to video) and `animate` (image to
video) through the same engine and queue, with the same quality policy: the default is the
strongest permissively licensed model at full resolution and full steps, faster presets only
by explicit choice, and permissive licences only. Video needs a discrete GPU; on a CPU-only
machine the studio will say so rather than attempt it.
