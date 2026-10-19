# Changelog

All notable changes to Hivemind Image Studio. The format follows Keep a Changelog; versions
follow semantic versioning. Official builds: GitHub Releases and the `hivemind-imagestudio`
package on PyPI.

## 0.3.9 — 2026-10-19

### Changed
- Renamed to **Presidium Image Studio** with the Presidium mark (six seats around one table,
  the picture on the table) and the Image Studio teal. The package, command, data folder and
  Store identity keep their names, so nothing installed breaks; only what you see changes.

### Added
- Series strip and "what changed" in the viewer; one-file web page export of a project or a
  selection.
- Extend the canvas (outpainting): grow an image to a new aspect ratio or add room on any side;
  the new area is painted to match with masked image-to-image. Viewer, CLI, MCP `outpaint`.
- Templates on Home and Create: eight starting points that turn a field or two into a full
  prompt, negative prompt and size, with honest notes about text rendering per model.
- "Turn this into…": fourteen one-click looks plus your saved styles, on one image or a
  selection, each at a strength tuned for the model's measured cliff; the PNG records the
  look. `imagestudio transform`, MCP `transform_image`.

## 0.3.8 — 2026-10-17

### Added
- GPU suggestion: when the Vulkan build would likely render faster on this machine's GPU,
  Create shows it with the measured number and an Install or Use button; never switched
  automatically. Home guide step for it.
- For agents: client configs for Claude Desktop, Claude Code, Cursor and VS Code under
  `examples/mcp`, a website page, and the MCP command (with the installed exe path) in About.
- Batch tools: select several gallery images and upscale, cut out, fix faces, file in a
  project, zip or trash them together.
- Search by meaning: CLIP ViT-B/32 (MIT, 150 MB) embeds every finished image in the
  background; the gallery search ranks by meaning and a Duplicates filter groups
  near-identical images. `imagestudio search`, MCP `search_images`.
- Styles library: saved prompt wording next to the built-in styles, recorded by name and text
  in every PNG, saveable from any image that carries one, exportable and importable as JSON.
- Fix faces: UltraFace (MIT) detection plus a per-face image-to-image pass at 768 px, blended
  back; viewer button, `imagestudio fix-faces`, MCP `fix_faces`.
- Projects: named series with a header chip, gallery filter, project name in PNG metadata,
  file-from-viewer, and a one-zip export with every image's reproducible prompt.
- Prompt assistant (✦ next to the prompt, `imagestudio assist`, MCP `write_prompt`): the local
  vision-language model runs text-only and turns a plain-words brief into a prompt and
  negative prompt tuned to the model that will paint it, or improves the prompt already typed.
- Microsoft Store listing (ID 9NJ8PVHV74KZ) linked from the website and README; raster
  home-screen icons and an "Add to Home Screen" hint for phones after pairing.
- MCP tools carry annotations (read-only, destructive, idempotent, open-world hints) so
  clients that gate on them can classify the server correctly.

### Changed
- Website links open in a new tab.

## 0.3.7 — 2026-10-16

### Added
- macOS (Apple silicon) build: `ImageStudio-<version>-macos-arm64.zip` with "Image Studio.app",
  a native window (system WebKit). Linux (x86_64) build: `ImageStudio-<version>-linux-x86_64.tar.gz`,
  opens the studio in the browser. Both unsigned, both with SHA-256 files; see
  `packaging/README-LINUX-MACOS.txt`. These builds are new; if one fails, the release ships without it.
- Pause switch for the queue (queue rail and tray menu): the running render finishes, nothing
  new starts, across every process that submits jobs. Memory limit in Settings → Preferences
  (GiB): every fit decision, estimate and refusal follows it.
- Edit uses an installed edit model only when it fits this machine's memory (or over-budget
  runs are unlocked); the viewer and model lists say which sets do not fit and why.
- One shared job summary (status, step, elapsed, ETA) in the capsule, queue card and gallery
  chip; the ETA says "measuring…" instead of guessing before sampling starts.

### Changed
- Closing the studio for real stops the running render; the tray is the way to keep it going.
- A worker that hands over to the window releases its lock, so a studio started by an MCP
  client no longer blocks the window from rendering.
- Themed dialogs everywhere (no native confirm/prompt); the side rails collapse fully.

### Fixed
- Progress bar stuck during long model loads; queue card overflow; missing live preview.

## 0.3.6 — 2026-10-15

### Added
- First Microsoft Store package: `ImageStudio-0.3.6-windows-x64-store.msix` built with the
  publisher's Partner Center identity (signed by Microsoft on submission). Privacy policy page
  on the website.

## 0.3.5 — 2026-10-14

Same as 0.3.4; the 0.3.4 Windows build failed in the installer step, so 0.3.4 exists on PyPI only.

## 0.3.4 — 2026-10-14

### Added
- `ImageStudio-<version>-Setup.exe`: a normal Windows installer (per user, no administrator
  rights). Start menu entry, search, Pin to Start, uninstall from Settings → Apps, optional
  desktop shortcut and start-at-sign-in. Models, images and settings are kept on uninstall.

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
