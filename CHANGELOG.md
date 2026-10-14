# Changelog

All notable changes to Hivemind Image Studio. The format follows Keep a Changelog; versions
follow semantic versioning. Official builds: GitHub Releases and the `hivemind-imagestudio`
package on PyPI.

## 0.3.3 — 2026-10-14

### Added
- Phone and remote access (Settings → Phone and remote): the studio listens on your local
  network, phones pair by scanning a QR code with a one-time code and get a revocable device
  key; the full studio at desktop quality on the phone, with a Camera button that imports a
  photo into the gallery. Device list with rename and revoke. Web app manifest and icon so the
  page can be added to the home screen. Nothing leaves your network.
- MSIX packaging for the Microsoft Store (`packaging/STORE.md`); every release builds it.

### Changed
- The window and taskbar show the studio icon when run from source too.

## 0.3.2 — 2026-10-14

### Changed
- The Windows app is a real window now (WebView2), not a browser tab: title bar and taskbar
  icon of its own, no console. Closing it during a render offers to keep rendering in the
  background with a tray icon (progress in its tooltip; click to come back). A second launch
  focuses the running window. Window size and position are remembered. Native file dialogs
  for importing images, LoRA files and your own model weights. Logs go to
  `<data>/logs/app.log` (Settings → About shows it). `--browser` keeps the old behaviour;
  machines without WebView2 are offered the download or the browser.

## 0.3.1 — 2026-10-14

### Added
- Windows app: `ImageStudio-<version>-windows-x64.zip` on GitHub Releases. Unzip, run
  `ImageStudio.exe`, the studio opens in your browser. Unsigned for now, so SmartScreen asks
  once per download; endpoint protection may ask about the engine the first time it runs.
  `ImageStudio.exe cli doctor` reaches the command line.

### Changed
- Background removal (ONNX Runtime) is part of the default install; the `bg` extra is kept
  as a no-op for compatibility. `pipx install hivemind-imagestudio` now gives the whole studio.
- Website: smooth scrolling, plain install command first.

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
