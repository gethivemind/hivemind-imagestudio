"""Browser tests for the page, against the fake engine. They run only when Playwright and
its Chromium are installed (``pip install -e .[e2e] && playwright install chromium``); CI
has a job for them. They catch what unit tests cannot: a broken layout, a JavaScript error,
a button that does nothing.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from conftest import FLUX_SCHNELL, fake_install, laptop
from imagestudio.engine.fake import FakeEngine
from imagestudio.service import Studio
from imagestudio.ui import build_app

pw = pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def site(settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    studio = Studio(settings=settings, engine=FakeEngine(step_delay=0.05), hardware=laptop())
    port = _free_port()
    config = uvicorn.Config(build_app(studio), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", studio
    server.should_exit = True
    t.join(5)
    studio.close()


@pytest.fixture
def page(site):
    url, _ = site
    try:
        with pw.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                pytest.skip(f"chromium not available: {e}")
            pg = browser.new_page(viewport={"width": 1400, "height": 900})
            errors: list[str] = []
            pg.on("pageerror", lambda err: errors.append(str(err)))
            pg.goto(url)
            pg.wait_for_selector("#presetSeg button", state="attached")
            pg.click('#tabs button[data-tab="create"]')
            pg.wait_for_selector("#prompt", state="visible")
            pg.errors = errors  # type: ignore[attr-defined]
            yield pg
            browser.close()
    except pw.Error as e:  # pragma: no cover
        if "Executable doesn't exist" in str(e) or "browserType.launch" in str(e):
            pytest.skip(str(e))
        raise


def test_create_flow_renders_queues_and_shows_result(page):
    assert page.errors == []
    assert page.locator("#presetSeg button").count() == 3
    page.fill("#prompt", "a teapot --seed 9 --size 512x256")
    page.wait_for_selector(".pchip:has-text('seed 9')")
    page.wait_for_function("!document.getElementById('go').disabled")
    page.click("#go")
    page.wait_for_selector(".qcard", timeout=10000)
    page.wait_for_selector(".qcard img", timeout=15000)  # finished: thumbnail present
    assert "Done" in page.inner_text("#stwrap")
    # canvas shows the image, the status sits in the capsule stack, nothing overlaps the box
    assert page.locator(".canvas > img").count() == 1
    assert page.errors == []


def test_rails_toggle_without_breaking_the_canvas(page):
    width_before = page.evaluate("document.querySelector('.canvas').getBoundingClientRect().width")
    page.click("#railBtn")
    time.sleep(0.2)
    width_rail_off = page.evaluate(
        "document.querySelector('.canvas').getBoundingClientRect().width"
    )
    assert width_rail_off > width_before > 300
    queue_x = page.evaluate("document.querySelector('.qrail').getBoundingClientRect().left")
    canvas_x = page.evaluate("document.querySelector('.canvas').getBoundingClientRect().left")
    assert queue_x > canvas_x  # the queue stays on the right
    page.click("#railBtn")
    page.click("#focusBtn")
    time.sleep(0.2)
    assert (
        page.evaluate("document.querySelector('.canvas').getBoundingClientRect().width")
        > width_rail_off
    )
    assert page.errors == []


def test_gallery_lightbox_and_reproducible_prompt(page):
    page.fill("#prompt", "a lighthouse --seed 3 --preset fast")
    page.wait_for_function("!document.getElementById('go').disabled")
    page.click("#go")
    page.wait_for_selector(".qcard img", timeout=15000)
    page.click("#tabs button[data-tab='gallery']")
    page.wait_for_selector("#grid .card")
    page.click("#grid .card")
    page.wait_for_selector("#lb[open]")
    text = page.inner_text("#lbPrompt")
    assert "a lighthouse" in text and "--seed 3" in text and "--preset fast" in text
    assert page.locator("#lbTitle").inner_text().count("·") == 2  # "Oct 13, 00:24 · fast · 512²"
    # viewer: fit by default, wheel zooms, badge shows the percentage, double-click returns to fit
    assert page.inner_text("#lbZoom") == "fit"
    page.hover("#lbImg")
    page.mouse.wheel(0, -300)
    page.wait_for_function("document.getElementById('lbZoom').textContent !== 'fit'")
    assert page.inner_text("#lbZoom").endswith("%")
    page.dblclick("#lbImg")
    page.wait_for_function("document.getElementById('lbZoom').textContent === 'fit'")
    page.keyboard.press("Escape")
    page.wait_for_selector("#lb[open]", state="hidden")
    assert page.errors == []


def test_batch_history_and_compare(page):
    page.select_option("#count", "2")
    page.fill("#prompt", "two teapots --seed 11 --preset fast")
    page.wait_for_function("!document.getElementById('go').disabled")
    page.click("#go")
    page.wait_for_function("document.querySelectorAll('.qcard img').length >= 2", timeout=20000)
    page.click("#histBtn")
    page.wait_for_selector("#hist.open .hrow")
    assert "two teapots" in page.inner_text("#hist")
    page.click("#hist .hrow >> nth=0 >> .pin")
    page.wait_for_function("document.querySelector('#hist .pin.on') !== null")
    page.keyboard.press("Escape")
    page.click("#tabs button[data-tab='gallery']")
    page.wait_for_selector("#grid .card")
    assert page.locator("#grid .card").count() >= 2
    page.click("#grid .card >> nth=0")
    page.wait_for_selector("#lb[open]")
    page.click("#lbCompare")  # the other image in the list is a candidate
    page.wait_for_selector("#lbimg.compare")
    assert page.locator("#cmpSel option").count() >= 1
    page.click("#cmpX")
    page.wait_for_selector("#lbimg.compare", state="detached")
    assert page.errors == []


def test_machine_tab_and_settings(page):
    page.click("#tabs button[data-tab='machine']")
    page.wait_for_selector("#strip .stat")
    assert page.locator("#strip .stat").count() >= 5
    page.wait_for_selector("#modelsPanel table")
    page.check("#cfgOver")
    time.sleep(0.3)
    page.click("#tabs button[data-tab='create']")
    page.select_option("#variant", "q8")
    time.sleep(0.5)
    assert "refused" not in page.inner_text("#etaText")  # unlocked
    assert page.errors == []


def test_home_cards_reflect_readiness(page):
    page.click("#tabs button[data-tab='home']")
    page.wait_for_selector("#cards .tcard")
    cards = page.locator("#cards .tcard")
    assert cards.count() >= 6
    texts = page.inner_text("#cards")
    assert "Create" in texts and "Upscale" in texts and "Describe" in texts
    assert "ready on this machine" in texts  # Create: model + fake engine present
    page.click("#cards .tcard >> nth=0")  # Create card opens the Create tab
    page.wait_for_selector("#prompt", state="visible")
    page.click("#tabs button[data-tab='sketch']")
    page.wait_for_selector("#skC")
    assert page.is_visible("#brushbar")  # size, opacity, softness, pressure
    page.keyboard.press("o")
    assert page.get_attribute("#palette .tb.on", "data-tool") == "ellipse"
    page.keyboard.press("b")
    assert page.get_attribute("#palette .tb.on", "data-tool") == "brush"
    assert page.errors == []


def test_first_run_guide_and_shortcuts_overlay(page):
    page.click("#tabs button[data-tab='home']")
    # no benchmark and no finished image yet -> the guide shows with those steps open
    page.wait_for_selector("#guide .step")
    text = page.inner_text("#guide")
    assert "Getting started" in text and "Run the benchmark" in text
    assert page.locator("#guide .step.done").count() >= 2  # engine + model are installed
    page.click("#guide .step:not(.done) >> nth=0")  # jumps to the tab that completes it
    assert page.is_visible("#machine") or page.is_visible("#create")
    page.keyboard.press("?")
    page.wait_for_selector("#keysDlg[open]")
    assert "Keyboard shortcuts" in page.inner_text("#keysDlg")
    page.keyboard.press("Escape")
    page.wait_for_selector("#keysDlg", state="hidden")
    page.click("#tabs button[data-tab='home']")
    page.click("#guideHide")
    page.wait_for_selector("#guide .step", state="detached")
    assert page.errors == []
