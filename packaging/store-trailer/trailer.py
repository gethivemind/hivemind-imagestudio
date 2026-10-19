"""Store trailer, recorded from the running studio with Playwright so it can be regenerated
for every release.

    python packaging/store-trailer/trailer.py [--url http://127.0.0.1:8766] [--out packaging/store-trailer/out]

Produces trailer.mp4 (1920x1080, H.264, no sound, 45-60 s), poster.png (1920x1080, the frame
the Store asks for next to a trailer) and the raw trailer.webm. Six scenes match the six Store
screenshots; captions are drawn by the page itself (an injected overlay in the Presidium type),
and the last frame is the mark with the lockup.

Reusable: ``record(url, out, brand, scenes)`` takes any product's brand (name, product, colour,
mark SVG) and any list of ``Scene`` objects, so Shape Studio and the desktop app can record
their own trailers with the same tool. Needs ffmpeg on PATH for the MP4; without it the WebM
is kept and the script says so."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (CSS and JS snippets read better on one line)

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SIZE = {"width": 1920, "height": 1080}


@dataclass(frozen=True)
class Brand:
    family: str
    product: str
    colour: str
    mark_svg: str  # full <svg> markup, light on transparent


@dataclass(frozen=True)
class Scene:
    caption: str
    action: Callable[[Recorder], None]
    seconds: float  # how long the scene stays on screen after the action


OVERLAY_CSS = """
#px-cap { position: fixed; left: 48px; bottom: 44px; z-index: 99999; pointer-events: none;
  font: 500 28px/1.3 "Segoe UI", system-ui, sans-serif; color: #F4F4F5; letter-spacing: -.2px;
  background: rgba(21,22,26,.78); padding: 14px 22px; border-radius: 14px; backdrop-filter: blur(6px);
  opacity: 0; transition: opacity .35s ease; max-width: 25vw; }
#px-cap.on { opacity: 1; }
#px-end { position: fixed; inset: 0; z-index: 99998; display: grid; place-items: center; pointer-events: none; background: var(--px-bg, #149A78);
  opacity: 0; transition: opacity .6s ease; }
#px-end.on { opacity: 1; }
#px-end svg { width: 200px; height: 200px; }
#px-end .lock { margin-top: 28px; font: 400 56px/1.1 "Segoe UI", system-ui, sans-serif; color: #F4F4F5; letter-spacing: -.5px; }
#px-end .lock b { font-weight: 600; }
#px-end .line { margin-top: 18px; font: 400 26px/1.3 "Segoe UI", system-ui, sans-serif; color: rgba(244,244,245,.85); }
"""


class Recorder:
    def __init__(self, page, brand: Brand) -> None:
        self.page = page
        self.brand = brand

    # ---- overlay
    def install_overlay(self) -> None:
        self.page.add_style_tag(content=OVERLAY_CSS)
        self.page.evaluate(
            """(bg) => { const c = document.createElement('div'); c.id = 'px-cap'; document.body.appendChild(c);
            const e = document.createElement('div'); e.id = 'px-end'; e.style.setProperty('--px-bg', bg); document.body.appendChild(e); }""",
            self.brand.colour,
        )

    def caption(self, text: str) -> None:
        # an open <dialog> lives in the browser's top layer, above any z-index: the caption
        # moves into it while it is open and back to <body> afterwards
        self.page.evaluate(
            """(t) => { const c = document.getElementById('px-cap'); const host = document.querySelector('dialog[open]') || document.body; if (c.parentElement !== host) host.appendChild(c); c.classList.remove('on'); setTimeout(() => { c.textContent = t; c.classList.add('on'); }, 200); }""",
            text,
        )

    def end_card(self, line: str) -> None:
        self.page.evaluate(
            """([svg, fam, prod, line]) => { const e = document.getElementById('px-end');
            e.innerHTML = `<div style="text-align:center">${svg}<div class="lock"><b>${fam}</b> ${prod}</div><div class="line">${line}</div></div>`;
            document.getElementById('px-cap').classList.remove('on'); e.classList.add('on'); }""",
            [self.brand.mark_svg, self.brand.family, self.brand.product, line],
        )

    # ---- helpers for scenes
    def tab(self, name: str) -> None:
        self.page.click(f'#tabs button[data-tab="{name}"]')
        time.sleep(0.6)

    def type_slowly(self, selector: str, text: str, delay_ms: int = 45) -> None:
        self.page.click(selector)
        self.page.fill(selector, "")
        self.page.type(selector, text, delay=delay_ms)
        self.page.dispatch_event(selector, "input")

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)


def record(url: str, out: Path, brand: Brand, scenes: list[Scene], end_line: str) -> Path:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport=SIZE,
            record_video_dir=str(raw_dir),
            record_video_size=SIZE,
            color_scheme="dark",
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(url)
        page.wait_for_selector("body[data-booted]", state="attached", timeout=60000)
        rec = Recorder(page, brand)
        rec.install_overlay()
        time.sleep(0.8)
        for scene in scenes:
            rec.caption(scene.caption)
            scene.action(rec)
            time.sleep(scene.seconds)
        rec.end_card(end_line)
        time.sleep(1.2)
        page.screenshot(path=str(out / "poster.png"))
        time.sleep(2.6)
        video = page.video
        context.close()
        webm_src = Path(video.path())
        browser.close()
    webm = out / "trailer.webm"
    shutil.move(str(webm_src), webm)
    shutil.rmtree(raw_dir, ignore_errors=True)
    mp4 = out / "trailer.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found: kept trailer.webm only; install ffmpeg to produce trailer.mp4")
        return webm
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(webm),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            ffmpeg.replace("ffmpeg", "ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"trailer.mp4 {probe.stdout.strip()} s, poster.png {SIZE['width']}x{SIZE['height']}")
    return mp4


# ---------------------------------------------------------------- Image Studio scenes


def imagestudio_scenes() -> list[Scene]:
    def create(r: Recorder) -> None:
        r.tab("create")
        r.type_slowly(
            "#prompt",
            "a red fox on a mossy log in a misty pine forest at sunrise, soft golden light",
            delay_ms=38,
        )
        r.wait(1.6)  # the plan and the measured estimate arrive

    def templates(r: Recorder) -> None:
        r.tab("home")
        r.wait(1.0)
        r.page.hover('#tpls [data-tpl="product"]')
        r.wait(0.5)
        r.page.hover('#tpls [data-tpl="thumbnail"]')
        r.wait(0.5)
        r.page.click('#tpls [data-tpl="thumbnail"]')
        r.wait(0.8)
        r.type_slowly('#tplFields input[data-k="subject"]', "a cat wearing sunglasses", 40)
        r.wait(0.6)
        r.page.click("#tplUse")
        r.wait(1.2)

    def search(r: Recorder) -> None:
        r.tab("gallery")
        r.wait(1.0)
        r.type_slowly("#search", "lighthouse at dusk", 55)
        r.wait(2.2)

    def versions(r: Recorder) -> None:
        r.page.fill("#search", "")
        r.page.dispatch_event("#search", "input")
        r.wait(0.8)
        cards = r.page.query_selector_all("#grid .card")
        # prefer an image that has a series behind it
        target = None
        for c in cards[:12]:
            img = c.query_selector("img")
            jid = (img.get_attribute("src") or "").split("/")[-1] if img else ""
            if jid:
                series = r.page.evaluate(
                    f"fetch('/api/jobs/{jid}/series').then(x => x.json()).then(s => s.members.length)"
                )
                if series and series > 1:
                    target = c
                    break
        (target or cards[0]).click()
        r.page.wait_for_selector("#lb[open]")
        r.caption("Every version of an image, and what changed.")  # re-homed into the dialog
        r.wait(1.4)
        strip = r.page.query_selector_all("#lbSeries img")
        for s in strip[:3]:
            s.hover()
            r.wait(0.5)
        r.page.hover("#xfTiles button")
        r.wait(0.8)

    def sketch(r: Recorder) -> None:
        r.page.keyboard.press("Escape")
        r.wait(0.4)
        r.tab("sketch")
        r.wait(0.8)
        box = r.page.eval_on_selector(
            "#skstage canvas",
            "c => { const b = c.getBoundingClientRect(); return [b.x, b.y, b.width, b.height]; }",
        )
        x0, y0, w, h = box
        pts = [(0.2, 0.7), (0.3, 0.45), (0.42, 0.6), (0.55, 0.35), (0.7, 0.55), (0.82, 0.4)]
        r.page.mouse.move(x0 + w * pts[0][0], y0 + h * pts[0][1])
        r.page.mouse.down()
        for fx, fy in pts[1:]:
            r.page.mouse.move(x0 + w * fx, y0 + h * fy, steps=18)
        r.page.mouse.up()
        r.wait(0.4)
        r.page.mouse.move(x0 + w * 0.25, y0 + h * 0.8)
        r.page.mouse.down()
        r.page.mouse.move(x0 + w * 0.8, y0 + h * 0.82, steps=20)
        r.page.mouse.up()

    def settings(r: Recorder) -> None:
        r.tab("machine")
        r.wait(1.0)
        r.page.mouse.wheel(0, 500)
        r.wait(1.0)
        r.page.mouse.wheel(0, 500)

    return [
        Scene("Type what you want. The time shown is measured on this machine.", create, 2.2),
        Scene("Or start from a template: a field or two, a full prompt.", templates, 1.4),
        Scene("Find your pictures by meaning. Offline.", search, 1.6),
        Scene("Every version of an image, and what changed.", versions, 1.4),
        Scene("Sketch the layout. Let the model paint.", sketch, 1.6),
        Scene("Benchmark once. Honest estimates ever after.", settings, 1.6),
    ]


def imagestudio_brand() -> Brand:
    from imagestudio.brand import LIGHT, PRODUCT, TEAL, mark_svg

    svg = mark_svg(colour=LIGHT).replace("<svg ", '<svg width="200" height="200" ', 1)
    return Brand("Presidium", PRODUCT, TEAL, svg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    path = record(
        args.url,
        out,
        imagestudio_brand(),
        imagestudio_scenes(),
        "Everything on your machine. Nothing uploaded.",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
