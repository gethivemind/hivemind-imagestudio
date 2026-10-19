"""Store and website screenshots of the running studio, 1440x900 PNG, taken with Playwright.

    python packaging/screenshots.py [http://127.0.0.1:8766]

Writes site/assets/shot-*.png (the website) and packaging/store-screenshots/*.png (Partner
Center). Needs a studio with a few finished images in its gallery; the gallery, viewer and
settings shots show real content, never mock-ups."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_SITE = ROOT / "site" / "assets"
OUT_STORE = ROOT / "packaging" / "store-screenshots"
SIZE = {"width": 1440, "height": 900}


def main(url: str) -> int:
    from playwright.sync_api import sync_playwright

    OUT_STORE.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=SIZE, color_scheme="dark")
        page.goto(url)
        page.wait_for_selector("body[data-booted]", state="attached", timeout=60000)

        def shot(name: str) -> None:
            time.sleep(0.6)
            for folder in (OUT_SITE, OUT_STORE):
                page.screenshot(path=str(folder / f"shot-{name}.png"))
            print(f"shot-{name}.png")

        page.click('#tabs button[data-tab="create"]')
        page.fill(
            "#prompt",
            "a red fox on a mossy log in a misty pine forest at sunrise, soft golden light",
        )
        page.dispatch_event("#prompt", "input")
        time.sleep(1.5)  # the plan and estimate arrive
        shot("create")
        page.click('#tabs button[data-tab="home"]')
        shot("home")
        page.click('#tabs button[data-tab="gallery"]')
        time.sleep(1.2)  # thumbnails
        shot("gallery")
        cards = page.query_selector_all("#grid .card")
        if cards:
            cards[0].click()
            page.wait_for_selector("#lb[open]")
            time.sleep(1.2)
            shot("viewer")
            page.keyboard.press("Escape")
        page.click('#tabs button[data-tab="sketch"]')
        shot("sketch")
        page.click('#tabs button[data-tab="machine"]')
        time.sleep(1.2)
        shot("settings")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"))
