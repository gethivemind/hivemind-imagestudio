# Changelog

All notable changes to Hivemind Image Studio. The format follows Keep a Changelog; versions
follow semantic versioning. Official builds: GitHub Releases and the `hivemind-imagestudio`
package on PyPI.

## 0.3.0 — 2026-10-14

First public release of the full studio.

### Added
- Local web UI (`imagestudio ui`): Home, Create, Sketch, Gallery, Settings; live preview,
  queue with ETA, load meter, prompt parameters (`--seed`, `--size`, `--preset`, `--n`,
  `--lora`, …), reproducible prompts, batches, favourites, trash, lineage, compare, prompt
  history, desktop notifications, keyboard shortcuts, first-run guide.
- Tools: variations, edit by instruction (image-to-image, or Qwen-Image-Edit 2509 when
  installed), remove object, sketch-to-image, upscale (Real-ESRGAN), remove background
  (IS-Net, U²-Net), describe / prompt from image / alt text / tags (llama.cpp + Qwen2.5-VL).
- Engine backends: CPU and Vulkan builds of stable-diffusion.cpp, switchable in Settings;
  measured 2.4× faster on an Intel integrated GPU.
- LoRA adapters from a local folder; your own models (FLUX, Qwen-Image, SD 1.5, SDXL
  families) registered from local files and labelled "your own", never "verified".
- PNG metadata (A1111 `parameters` + `imagestudio` JSON) on every finished image; kept on
  export, restored on import. Auto-captions with the local vision model.
- MCP tools: generate (with `count`, `loras`), plan, job_status, job_result, jobs,
  capabilities, download_model, variations, edit, upscale, remove_background, describe.
- Trademark policy, NOTICE, CITATION.cff, SPDX headers.

### Changed
- Image-to-image strength defaults lowered below the measured cliff on 4-step models
  (variations 0.4, edit 0.45, sketch 0.45) with honest help texts.
- Benchmark calibrates per backend; estimates include load time on memory-bound machines.

### Fixed
- Duplicate rendering when two processes ran workers (worker lock, cross-process cancel).
- Lightbox overflow, invisible brush cursor, memory refresh feedback, CI on Ubuntu/Windows.

## 0.2.0 — 2026-10-12

Increment 1: `generate` with the queue, hardware detection, hash-verified model and engine
downloads, calibration, CLI, MCP server. Published to PyPI on 2026-10-13.
