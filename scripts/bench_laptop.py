"""Timing table for the README: run fast, balanced and best back to back on this machine and
record wall time, peak engine RSS and free RAM at start for each.

Usage: python scripts/bench_laptop.py [out.json]   (uses the normal data dir and models)
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import psutil

from imagestudio.service import Studio

PROMPT = "a lighthouse on a rocky coast at dusk, oil painting, warm light"
PRESETS = ("fast", "balanced", "best")
GIB = 1024**3


def engine_rss() -> int:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if p.info["name"] and p.info["name"].startswith("sd-cli"):
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def main(out_path: Path) -> int:
    studio = Studio()
    results: list[dict] = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
    done_presets = {r["preset"] for r in results if r.get("status") == "done"}
    try:
        for preset in PRESETS:
            if preset in done_presets:
                print(f"{preset}: already measured, skipping", flush=True)
                continue
            free_start = psutil.virtual_memory().available
            job = studio.generate(PROMPT, preset=preset, seed=7)
            print(
                f"{preset}: job {job.id} {job.width}x{job.height} steps {job.steps} "
                f"estimate {job.estimate_seconds:.0f}s free RAM {free_start / GIB:.1f} GiB "
                f"mmap={'--mmap' in job.extra_args}",
                flush=True,
            )
            peak = 0
            stop = threading.Event()

            def sample(stop_event: threading.Event = stop) -> None:
                nonlocal peak
                while not stop_event.is_set():
                    peak = max(peak, engine_rss())
                    stop_event.wait(2.0)

            t = threading.Thread(target=sample, daemon=True)
            t.start()
            t0 = time.monotonic()
            last = ""
            while True:
                st = studio.job_status(job.id)
                line = f"  {st['status']}/{st['phase']} step {st['step']}/{st['total_steps']}"
                if line != last:
                    print(line, flush=True)
                    last = line
                if st["status"] in ("done", "failed", "cancelled"):
                    break
                time.sleep(5)
            stop.set()
            t.join(5)
            wall = time.monotonic() - t0
            j = studio.jobs.get(job.id)
            rec = {
                "preset": preset,
                "job_id": job.id,
                "status": st["status"],
                "width": job.width,
                "height": job.height,
                "steps": job.steps,
                "variant": job.variant,
                "mmap": "--mmap" in job.extra_args,
                "estimate_seconds_before": round(job.estimate_seconds),
                "free_ram_gib_at_start": round(free_start / GIB, 1),
                "wall_seconds": round(wall),
                "load_seconds": round(j.load_seconds or 0) if j else None,
                "sampling_seconds": round(j.sampling_seconds or 0) if j else None,
                "peak_engine_rss_gib": round(peak / GIB, 2),
                "error": st.get("error"),
                "output_path": st.get("output_path"),
            }
            results = [r for r in results if r["preset"] != preset] + [rec]
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  RESULT {json.dumps(rec)}", flush=True)
            if st["status"] != "done":
                print(f"{preset} did not finish: {st.get('error')}", flush=True)
                return 1
    finally:
        studio.close()
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench.json")
    raise SystemExit(main(target))
