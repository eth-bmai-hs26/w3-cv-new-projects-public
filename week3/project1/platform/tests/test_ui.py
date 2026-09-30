"""
End-to-end check of the built frontend with Playwright (Chromium).

Needs: `npm run build` done once, and a Python with playwright installed. The
server runs in its own process with PLATFORM_PYTHON (default: this Python), so
Playwright and the backend may live in different environments:

    PLATFORM_PYTHON=.venv/bin/python  python -m pytest tests/test_ui.py
    SCREENSHOTS=docs  ...             also writes the README screenshots
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

PLATFORM = Path(__file__).resolve().parent.parent
DIST = PLATFORM / "frontend" / "dist" / "index.html"
pytestmark = pytest.mark.skipif(not DIST.exists(), reason="frontend not built (cd frontend && npm run build)")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    py = os.environ.get("PLATFORM_PYTHON", sys.executable)
    data = tmp_path_factory.mktemp("ui-data")
    env = {**os.environ, "TILE_PLATFORM_DATA": str(data), "TILE_PLATFORM_CHECKPOINT": ""}
    imgs = data / "photos"
    subprocess.run([py, "-m", "backend", "samples", str(imgs), "--n", "24"], cwd=PLATFORM, env=env, check=True,
                   capture_output=True)
    subprocess.run([py, "-m", "backend", "seed", "--tiles", "150", "--days", "30", "--images", str(imgs / "original")],
                   cwd=PLATFORM, env=env, check=True, capture_output=True)
    port = _free_port()
    proc = subprocess.Popen([py, "-m", "backend", "--port", str(port), "--no-seed"], cwd=PLATFORM, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(proc.stdout.read().decode())
    yield {"url": url, "photos": sorted((imgs / "original").glob("*.jpg"))}
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("localStorage.setItem('inspector', 'M. Keller')")
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and pg.errors.append(m.text))
    yield pg
    ctx.close()


def _shot(pg, name, full=True):
    out = os.environ.get("SCREENSHOTS")
    if out:
        d = Path(out) if Path(out).is_absolute() else PLATFORM / out
        d.mkdir(parents=True, exist_ok=True)
        time.sleep(0.8)                                         # let charts finish drawing
        pg.screenshot(path=str(d / f"{name}.png"), full_page=full)


def test_dashboard_renders(server, page):
    page.goto(server["url"] + "/")
    page.wait_for_selector(".kpis .kpi-value")
    assert page.locator("h1").inner_text() == "Dashboard"
    assert page.locator(".hero-num").inner_text().startswith("€")
    assert page.locator(".recharts-bar-rectangle").count() > 10          # throughput + histogram bars
    assert page.locator(".supplier-table tbody tr").count() >= 3
    assert page.locator(".station-demo").is_visible()                    # demo mode clearly labelled
    _shot(page, "dashboard")
    assert not page.errors


def test_inspect_upload_flow(server, page):
    page.goto(server["url"] + "/inspect")
    page.wait_for_selector(".lot-picker")
    page.click("text=New lot")
    page.fill("input[list=suppliers]", "Rückbau Zürich AG")
    page.fill("input[list=types]", "Glazed ceramic 20×20")
    page.click("text=Create lot")
    page.wait_for_selector(".lot-line")
    files = [str(p) for p in server["photos"][:6]]
    page.set_input_files("input[type=file][multiple]", files)
    page.wait_for_function("document.querySelectorAll('.result').length >= 6", timeout=60000)
    assert page.locator(".conveyor-head b").inner_text() == "Batch finished"
    assert page.locator(".result .tag").count() == 6
    assert page.locator(".result-pct").first.inner_text().endswith("usable")
    page.evaluate("window.scrollTo(0, 0)")
    _shot(page, "inspect")
    assert not page.errors


def test_lot_pages(server, page):
    page.goto(server["url"] + "/lots")
    page.wait_for_selector("tbody tr")
    _shot(page, "lots")
    page.locator("tbody tr").first.click()
    page.wait_for_selector(".lot-sheet")
    assert page.locator("h1").inner_text().startswith("Lot L")
    _shot(page, "lot")
    href = page.locator("a:has-text('Printable report')").get_attribute("href")
    page.goto(server["url"] + href)
    page.wait_for_selector(".report-table tbody tr")
    _shot(page, "report")
    assert not page.errors


def test_review_decision(server, page):
    page.goto(server["url"] + "/review")
    page.wait_for_selector(".page-title")
    if page.locator(".queue-item").count() == 0:
        pytest.skip("no borderline tiles in this seed")
    _shot(page, "review")
    before = page.locator(".queue-item").count()
    tile = page.locator(".rd-side h2").inner_text()
    page.fill(".rd-side textarea", "crack only in the glaze")
    page.click("button:has-text('Approve')")
    page.wait_for_selector(".toast")
    assert "approved" in page.locator(".toast").first.inner_text()
    assert page.locator(".queue-item").count() == before - 1
    assert page.locator(".rd-side h2").count() == 0 or page.locator(".rd-side h2").inner_text() != tile
    assert not page.errors


def test_settings_preview_reacts(server, page):
    page.goto(server["url"] + "/settings")
    page.wait_for_selector(".preview-table")
    _shot(page, "settings")
    page.locator(".slider input").first.evaluate(
        """el => { const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                   set.call(el, '0.7'); el.dispatchEvent(new Event('input', { bubbles: true })); }""")
    page.wait_for_selector(".preview-table .changed", timeout=5000)
    assert page.locator(".pill").inner_text() == "Unsaved changes"
    assert not page.errors


def test_mobile_layout(server, browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
    ctx.add_init_script("localStorage.setItem('inspector', 'M. Keller')")
    pg = ctx.new_page()
    pg.goto(server["url"] + "/")
    pg.wait_for_selector(".kpis")
    assert pg.locator(".nav").bounding_box()["y"] > 700                # tab bar at the bottom
    assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    _shot(pg, "mobile-dashboard", full=False)
    pg.goto(server["url"] + "/inspect")
    pg.wait_for_selector(".drop")
    _shot(pg, "mobile-inspect", full=False)
    ctx.close()
