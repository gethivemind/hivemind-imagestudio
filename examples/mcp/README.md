# Image Studio as an MCP server

`imagestudio serve` speaks the Model Context Protocol over stdio. Any MCP client can make,
edit and search images on this machine, offline, with the same honesty rules as the window:
quality is a setting, never a consequence; estimates are measured; nothing is uploaded.

| Client | File | Where |
| --- | --- | --- |
| Claude Desktop | `claude-desktop.json` | `%APPDATA%\Claude\claude_desktop_config.json` (Windows), `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) |
| Claude Code | `claude-code.mcp.json` | `.mcp.json` in the project, or `claude mcp add imagestudio -- imagestudio serve` |
| Cursor | `cursor.mcp.json` | `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` |
| VS Code (Copilot agent mode) | `vscode.mcp.json` | `.vscode/mcp.json` |
| Windows app instead of pip | `windows-app.json` | the installed `ImageStudio.exe cli serve` is the same server |

The command must be on PATH: `pipx install hivemind-imagestudio` puts `imagestudio` there.

## A typical agent flow

1. `capabilities()` once: hardware, installed models, expected minutes per preset, what is
   missing. If no model is installed, `download_model(name, consent=true)` after telling the
   user the size and licence.
2. `write_prompt(brief)` turns a vague request into a prompt tuned to the model that will paint
   it (a minute or more on a CPU).
3. `generate(prompt)` returns a job id and an honest estimate immediately. Poll `job_status`
   every 10-15 s; never block inside one call. `job_result(job_id)` gives the PNG path (or
   base64 on request).
4. Follow-ups on the result: `fix_faces` when faces look wrong, `upscale` for a print-size file,
   `remove_background` for a cutout, `variations` for a re-roll that keeps the composition
   (strength ≤ 0.45), `edit` by instruction.
5. `search_images("the lighthouse one")` finds an earlier image by meaning instead of listing
   everything with `jobs()`.

## Who renders

Only one process renders at a time (a lock in the data folder). The first studio to start
takes the worker role: if the desktop window is open it renders; if a client launches
`imagestudio serve` first, that server renders and the window shows the queue. When the
window starts later it asks an idle worker to hand over, and both keep sharing one queue and
one gallery. Pause (queue rail or tray) holds every submitter, including agents.

## Timing to expect

Measured on a 16 GB laptop without a discrete GPU, FLUX.1-schnell q5 at 1024²: `fast` about
22 min, `balanced` 48 min, `best` just under 2 h on the CPU build; the Vulkan build was 2.4×
faster on the same integrated GPU. Describe takes seconds, the prompt assistant 20-40 s warm,
face repair one render per face. `capabilities()` reports the calibrated numbers for the
machine it runs on; quote those to the user, not these.
