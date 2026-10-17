"""Command line: doctor, install-engine, download-model, generate, jobs, serve."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
import time

from imagestudio import __version__
from imagestudio.calibration import human_duration
from imagestudio.engine.base import EngineError
from imagestudio.registry import GIB, MODELS, PRESET_NAMES
from imagestudio.service import Studio, StudioError


def _consent_prompt(yes: bool):
    def consent(description: str, size: int) -> bool:
        print(f"Download: {description}")
        print(f"Size: {size / GIB:.2f} GiB. Nothing else will be fetched.")
        if yes:
            print("Approved with --yes.")
            return True
        if not sys.stdin.isatty():
            print("No terminal to ask on; re-run with --yes to approve.")
            return False
        answer = input("Proceed? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    return consent


class _Bar:
    def __init__(self) -> None:
        self._last = 0.0
        self._name = ""

    def __call__(self, name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if name != self._name:
            self._name = name
            print()
        if now - self._last < 0.5 and done < total:
            return
        self._last = now
        pct = (done / total * 100) if total else 0
        if total >= GIB:
            amount = f"{done / GIB:6.2f}/{total / GIB:6.2f} GiB"
        else:
            amount = f"{done / 2**20:6.1f}/{total / 2**20:6.1f} MiB"
        print(f"\r  {name}: {amount} ({pct:5.1f}%)", end="", flush=True)
        if done >= total:
            print()


def cmd_doctor(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    caps = studio.capabilities()
    if args.json:
        print(json.dumps(caps, indent=2))
        return 0
    hw = caps["hardware"]
    print(f"Hivemind Image Studio {__version__}")
    print(f"Data dir: {caps['paths']['data_dir']}")
    print()
    print("Hardware")
    print(f"  OS/arch: {hw['os']} {hw['arch']}")
    print(f"  CPU: {hw['cpu_name']} ({hw['physical_cores']} cores, {hw['logical_cores']} threads)")
    print(f"  RAM: {hw['ram_total_gib']} GiB total, {hw['ram_available_gib']} GiB available")
    if hw["gpus"]:
        for g in hw["gpus"]:
            vram = f"{g['vram_gib']} GiB" if g["vram_gib"] else "unknown VRAM"
            shared = " (shared with RAM)" if g["shared_memory"] else ""
            print(f"  GPU: {g['name']} [{g['kind']}, {vram}{shared}]")
    else:
        print("  GPU: none found")
    print()
    eng = caps["engine"]
    state = (
        f"installed: {eng['version']} at {eng['binary']}" if eng["available"] else "NOT installed"
    )
    print(
        f"Engine: {eng['name']} ({state}); pinned release {eng['pinned_release']}; "
        f"backend {eng.get('backend', 'cpu')}"
    )
    print()
    print("Models")
    for m in caps["models"]:
        inst = ", ".join(m["installed_variants"]) or "none"
        print(f"  {m['display_name']} ({m['name']}, {m['licence']}): installed variants: {inst}")
        print(f"    memory budget on this machine: {m['memory_budget_gib']} GiB")
        for v in m["variants"]:
            fit = "fits" if v["fits_this_machine"] else "does NOT fit"
            mark = "*" if v["installed"] else " "
            print(
                f"    {mark} {v['name']}: download {v['download_gib']} GiB, needs "
                f"{v['needs_gib_at_1024']} GiB at 1024 px: {fit}"
            )
        for p in m["presets"]:
            if "error" in p:
                print(f"    preset {p['preset']}: {p['error']}")
                continue
            t = p["expected_time"]
            tag = "" if t["calibrated"] else " (uncalibrated)"
            print(
                f"    preset {p['preset']}: {p['width']}x{p['height']}, {p['steps']} steps, "
                f"variant {p['variant']}: about {t['human']}{tag}"
            )
            if p["fit_note"]:
                print(f"      fit: {p['fit_note']}")
    print()
    print("Policy: quality is a setting, never a consequence. Default preset: best.")
    print(
        "Licences: " + "; ".join(f"{li['component']}: {li['licence']}" for li in caps["licences"])
    )
    ex = ", ".join(f"{e['model']} ({e['reason']})" for e in caps["excluded_by_policy"])
    print(f"Excluded by policy: {ex}")
    if args.verify:
        print()
        print("Verifying installed model hashes (slow)...")
        for m in MODELS.values():
            for v in m.variants:
                for f in v.files:
                    if studio.models.path_for(f).exists():
                        ok = studio.models.verify_file(f)
                        print(f"  {f.filename}: {'ok' if ok else 'MISMATCH'}")
    return 0


def cmd_install_engine(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    if args.vision:
        engine = studio.describe_engine
    elif args.flavor:
        from imagestudio.engine.sdcpp import SdCppEngine

        engine = SdCppEngine(studio.settings, flavor=args.flavor)
    else:
        engine = studio.engine
    install = getattr(engine, "install", None)
    if install is None:
        print("This engine has no installer.")
        return 1
    bar = _Bar()
    try:
        path = install(_consent_prompt(args.yes), progress=lambda d, t: bar("engine", d, t))
    except EngineError as e:
        print(f"error: {e}")
        return 1
    print(f"Engine ready: {path}")
    print(f"Version: {engine.version()}")
    if getattr(args, "use", False) and not args.vision:
        try:
            studio.set_engine_backend(engine.flavor)
        except StudioError as e:
            print(f"not switched: {e}")
            return 1
        print(f"Active engine backend: {engine.flavor} (config.json)")
    return 0


def cmd_download_model(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    from imagestudio.registry import TOOL_MODELS, VISION_MODELS

    bar = _Bar()
    if args.name in TOOL_MODELS:
        t = TOOL_MODELS[args.name]
        ok = studio.models.install_tool(t, _consent_prompt(args.yes), progress=bar)
        print(f"{t.display_name} {'installed.' if ok else 'not downloaded.'}")
        return 0 if ok else 1
    if args.name in VISION_MODELS:
        v = VISION_MODELS[args.name]
        ok = studio.models.install_vision(v, _consent_prompt(args.yes), progress=bar)
        print(f"{v.display_name} {'installed.' if ok else 'not downloaded.'}")
        return 0 if ok else 1
    if args.name not in MODELS:
        print(f"unknown model {args.name!r}; known: {', '.join(MODELS)}")
        return 2
    model = MODELS[args.name]
    try:
        plan = studio.plan(args.name, "best", None, None, args.variant)
    except StudioError as e:
        print(f"error: {e}")
        return 1
    v = plan.variant
    if plan.fit_note:
        print(plan.fit_note)
    if studio.models.variant_installed(v):
        print(f"{model.display_name} [{v.name}] is already installed.")
        return 0
    print(f"{model.display_name} [{v.name}] ({model.licence}): {v.note}")
    for f in v.files:
        have = "have" if studio.models.file_installed(f) else "need"
        print(f"  {have}  {f.filename}  {f.size / GIB:.2f} GiB  from {f.source}")
    bar = _Bar()
    try:
        ok = studio.models.install_variant(model, v, _consent_prompt(args.yes), progress=bar)
    except Exception as e:
        print(f"\nerror: {e}")
        return 1
    if not ok:
        print("Not downloaded.")
        return 1
    print(f"{model.display_name} [{v.name}] installed.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    studio = Studio()
    try:
        job = studio.generate(
            args.prompt,
            preset=args.preset,
            width=args.width,
            height=args.height,
            seed=args.seed,
            negative_prompt=args.negative or "",
            model=args.model,
            variant=args.variant,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            threads=args.threads,
            vae_tiling=args.vae_tiling,
        )
    except StudioError as e:
        print(f"error: {e}")
        studio.close()
        return 1
    print(
        f"Job {job.id}: {job.model} [{job.variant}] preset {job.preset}, "
        f"{job.width}x{job.height}, {job.steps} steps, seed {job.seed}"
    )
    print(f"Estimate: {human_duration(job.estimate_seconds)} ({job.estimate_basis})")
    if job.fit_note:
        print(f"Fit: {job.fit_note}")
    for w in job.warnings:
        print(f"Warning: {w}")
    if args.no_wait:
        studio.close()
        return 0
    last = ""
    try:
        while True:
            st = studio.job_status(job.id)
            line = (
                f"{st['status']}/{st['phase']} {st['progress'] * 100:5.1f}%"
                f" step {st['step']}/{st['total_steps']} eta {st['eta_human']}"
            )
            if line != last:
                print(f"\r  {line:<70}", end="", flush=True)
                last = line
            if st["status"] in ("done", "failed", "cancelled"):
                print()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nCancelling...")
        studio.cancel(job.id)
        time.sleep(1)
    st = studio.job_status(job.id)
    if st["status"] == "done":
        print(f"Image: {st['output_path']}")
        j = studio.jobs.get(job.id)
        if j and j.total_seconds:
            print(
                f"Took {human_duration(j.total_seconds)} total "
                f"(load {human_duration(j.load_seconds or 0)}, "
                f"sampling {human_duration(j.sampling_seconds or 0)})"
            )
    else:
        print(f"{st['status']}: {st['error'] or st['message']}")
    studio.close()
    return 0 if st["status"] == "done" else 1


def cmd_plan(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    try:
        p = studio.plan(
            args.model,
            args.preset,
            args.width,
            args.height,
            args.variant,
            args.steps,
            args.cfg_scale,
            args.sampler,
        )
    except StudioError as e:
        print(f"refused: {e}")
        return 1
    d = p.to_dict()
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    load = d["load"]
    bar = "#" * min(40, load["score"] * 40 // 100) + "-" * max(0, 40 - load["score"] * 40 // 100)
    print(
        f"{d['model']} [{d['variant']}] preset {d['preset']}: {d['width']}x{d['height']}, "
        f"{d['steps']} steps, cfg {d['cfg_scale']}, sampler {d['sampler']}"
    )
    print(f"Expected time: {d['estimate']['human']} ({d['estimate']['basis']})")
    print(
        f"Load [{bar}] {load['score']}% of budget: {load['label']} "
        f"(needs {load['required_gib']} GiB, budget {load['budget_gib']} GiB, "
        f"free now {load['available_gib']} GiB)"
    )
    print(f"  {load['meaning']}")
    if d["fit_note"]:
        print(f"Fit: {d['fit_note']}")
    for w in d["warnings"]:
        print(f"Warning: {w}")
    if not d["installed"]:
        print("Model files not downloaded yet.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    studio = Studio()
    last = ""

    def show(st: dict) -> None:
        nonlocal last
        line = (
            f"{st['status']}/{st['phase']} step {st['step']}/{st['total_steps']} "
            f"eta {st['eta_human']}"
        )
        if line != last:
            print(f"\r  {line:<70}", end="", flush=True)
            last = line

    print(f"Benchmark: one {args.preset} image; target {args.target_minutes:g} min per image.")
    try:
        p = studio.benchmark(args.preset, args.target_minutes, wait=2.0, on_status=show)
    except StudioError as e:
        print(f"\nerror: {e}")
        studio.close()
        return 1
    studio.close()
    print()
    if args.json:
        print(json.dumps(p.to_dict(), indent=2))
        return 0
    print(
        f"Machine: {p.cpu}, {p.physical_cores} cores, {p.ram_total_gib} GiB RAM, "
        f"GPU {p.gpu or 'none usable'}; memory budget {p.budget_gib} GiB"
    )
    print(
        f"Measured: {p.model} [{p.variant}] {p.width}x{p.height} {p.steps} steps in "
        f"{human_duration(p.wall_seconds)} (sampling {human_duration(p.sampling_seconds)}), "
        f"peak engine memory {p.peak_rss_gib} GiB, free RAM at start {p.free_ram_gib_at_start} GiB"
    )
    for note in p.notes:
        print(f"  {note}")
    print("Expected per preset now:")
    for name, secs in p.preset_estimates.items():
        mark = " <- recommended" if name == p.recommended_preset else ""
        print(f"  {name:<9} {human_duration(secs)}{mark}")
    if p.recommended_preset is None:
        print(
            f"No preset meets {p.target_minutes:g} min on this machine; "
            "best still runs, just slower."
        )
    print(f"Profile written to {studio.settings.profile_file}")
    return 0


def cmd_fix_faces(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    try:
        job = studio.fix_faces(args.image, args.strength, args.prompt or "", args.model)
    except StudioError as e:
        print(f"error: {e}")
        return 1
    print(
        f"queued {job.id}: {len(job.regions)} face(s), strength {job.strength:g}, "
        f"about {human_duration(job.estimate_seconds)}. `imagestudio jobs` to follow."
    )
    return 0


def cmd_assist(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    try:
        r = studio.assist_prompt(" ".join(args.brief), args.improve, args.model)
    except StudioError as e:
        print(f"error: {e}")
        return 1
    print(r["prompt"])
    if r["negative_prompt"]:
        print(f"--no {r['negative_prompt']}")
    if r["notes"]:
        print(f"-- {r['notes']}", file=sys.stderr)
    print(f"-- for {r['target_model']} · {r['model']} · {r['seconds']} s", file=sys.stderr)
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    try:
        r = studio.describe(args.image, args.task, args.question, args.model)
    except StudioError as e:
        print(f"error: {e}")
        return 1
    print(r["text"])
    print(f"-- {r['model']} · {r['seconds']} s", file=sys.stderr)
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    studio = Studio(start_worker=False)
    rows = studio.list_jobs()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("No jobs.")
    for r in rows:
        print(f"{r['id']}  {r['status']:<9} {r['preset']:<8} {r['model']:<13} {r['prompt']}")
        if r["output_path"]:
            print(f"    {r['output_path']}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from imagestudio.ui import serve_ui

    serve_ui(args.host, args.port, start_worker=not args.no_worker)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from imagestudio.server import serve

    serve(args.transport, args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imagestudio",
        description="Hivemind Image Studio: free, local, permissively licensed image generation.",
    )
    p.add_argument("--version", action="version", version=f"imagestudio {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="show hardware, engine, models and expected times")
    d.add_argument("--json", action="store_true")
    d.add_argument("--verify", action="store_true", help="re-hash installed model files")
    d.set_defaults(fn=cmd_doctor)

    ie = sub.add_parser("install-engine", help="download the pinned stable-diffusion.cpp build")
    ie.add_argument("--yes", "-y", action="store_true", help="approve the download")
    ie.add_argument("--vision", action="store_true", help="install llama.cpp for describe instead")
    ie.add_argument(
        "--flavor",
        choices=["cpu", "vulkan"],
        help="which stable-diffusion.cpp build: cpu (default) or vulkan (GPU through Vulkan)",
    )
    ie.add_argument(
        "--use", action="store_true", help="after installing, make this build the active backend"
    )
    ie.set_defaults(fn=cmd_install_engine)

    dm = sub.add_parser("download-model", help="download a model's weights (asks for consent)")
    dm.add_argument("name", help="a generation model, tool model or vision model name")
    dm.add_argument("--variant", choices=["q8", "q6", "q5", "q4"])
    dm.add_argument("--yes", "-y", action="store_true", help="approve the download")
    dm.set_defaults(fn=cmd_download_model)

    g = sub.add_parser("generate", help="generate one image and wait for it")
    g.add_argument("prompt")
    g.add_argument("--preset", choices=PRESET_NAMES, default="best")
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--seed", type=int)
    g.add_argument("--negative")
    g.add_argument("--model", choices=list(MODELS))
    g.add_argument("--variant", choices=["q8", "q6", "q5", "q4"])
    g.add_argument("--steps", type=int, help="advanced: override the step count of the preset")
    g.add_argument("--cfg-scale", type=float, help="advanced: override guidance strength")
    g.add_argument("--sampler", help="advanced: euler (default), euler_a, heun, dpm++2m, ...")
    g.add_argument("--threads", type=int, help="CPU threads (default: physical cores)")
    g.add_argument("--vae-tiling", action="store_true", help="decode in tiles to save memory")
    g.add_argument("--no-wait", action="store_true", help="queue and exit (needs serve running)")
    g.set_defaults(fn=cmd_generate)

    pl = sub.add_parser("plan", help="dry run: expected time and load meter for settings")
    pl.add_argument("--preset", choices=PRESET_NAMES, default="best")
    pl.add_argument("--width", type=int)
    pl.add_argument("--height", type=int)
    pl.add_argument("--model", choices=list(MODELS))
    pl.add_argument("--variant", choices=["q8", "q6", "q5", "q4"])
    pl.add_argument("--steps", type=int)
    pl.add_argument("--cfg-scale", type=float)
    pl.add_argument("--sampler")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(fn=cmd_plan)

    b = sub.add_parser("benchmark", help="run one image and write the machine profile")
    b.add_argument("--preset", choices=PRESET_NAMES, default="fast")
    b.add_argument("--target-minutes", type=float, default=15.0)
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_benchmark)

    ff = sub.add_parser(
        "fix-faces", help="re-render every face in an image sharply and blend it back"
    )
    ff.add_argument("image", help="a job id or an image path")
    ff.add_argument("--strength", type=float, default=0.35, help="0.1-0.7; 0.35 keeps the likeness")
    ff.add_argument("--prompt", help="defaults to the source image's prompt")
    ff.add_argument("--model", help="which model repaints (default: the planner's choice)")
    ff.set_defaults(fn=cmd_fix_faces)

    asst = sub.add_parser("assist", help="write or improve a prompt with the local language model")
    asst.add_argument("brief", nargs="+", help="what the image should show, in plain words")
    asst.add_argument("--improve", help="an existing prompt to change in the brief's direction")
    asst.add_argument(
        "--model", help="the catalogue model that will paint (wording is tuned to it)"
    )
    asst.set_defaults(fn=cmd_assist)

    de = sub.add_parser("describe", help="image to text with the local vision model")
    de.add_argument("image", help="a job id or an image path")
    de.add_argument(
        "--task", choices=["describe", "prompt", "alt", "tags", "custom"], default="describe"
    )
    de.add_argument("--question", help="for --task custom: what to ask about the image")
    de.add_argument("--model", choices=["qwen2.5-vl-3b", "qwen2.5-vl-7b"])
    de.set_defaults(fn=cmd_describe)

    j = sub.add_parser("jobs", help="list jobs")
    j.add_argument("--json", action="store_true")
    j.set_defaults(fn=cmd_jobs)

    u = sub.add_parser("ui", help="run the local web UI")
    u.add_argument("--host", default=None)
    u.add_argument("--port", type=int, default=8766)
    u.add_argument(
        "--no-worker",
        action="store_true",
        help="only submit and view jobs; another process renders them",
    )
    u.set_defaults(fn=cmd_ui)

    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))
