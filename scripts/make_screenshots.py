#!/usr/bin/env python
"""Capture README screenshots from the REAL running app (needs playwright + chromium).  python scripts/make_screenshots.py"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 5071
URL = f"http://127.0.0.1:{PORT}/"


def analyze(pg, rec, cond):
    pg.select_option("#exRecord", str(rec)); pg.select_option("#exCond", cond)
    pg.click("#btnExample"); pg.wait_for_selector("#inspector:not([hidden])", timeout=90000); pg.wait_for_timeout(900)


def main():
    srv = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py"), "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(URL + "api/overview", timeout=2).read(); break
            except Exception:
                time.sleep(0.5)
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1360, "height": 900}, device_scale_factor=2)
            pg.add_style_tag(content=":root{color-scheme:light}") if False else None
            pg.goto(URL); pg.wait_for_selector("#exRecord option", state="attached")
            # 1. clean example: summary + image + inspector (a beat in the middle)
            analyze(pg, 0, "clean")
            pg.click(".bchip[data-i='4']"); pg.wait_for_timeout(600)
            pg.locator("#result").screenshot(path=str(OUT / "01_analyze_clean.png"))
            # 2. occlusion: gaps are never filled in
            analyze(pg, 0, "occlusion_severe")
            pg.locator(".summary").screenshot(path=str(OUT / "02a_occlusion_summary.png"))
            pg.locator(".viewer").screenshot(path=str(OUT / "02_occlusion_gaps.png"))
            # 3. tilted image: everything rejected, measurements withheld, evaluation-only panel
            analyze(pg, 0, "perspective_severe")
            pg.locator("#result").screenshot(path=str(OUT / "03_perspective_rejected.png"))
            pg.locator("#inspector").screenshot(path=str(OUT / "03b_evidence_inspector.png"))
            # 4. evaluation tab
            pg.click(".tabs button[data-tab=evaluation]"); pg.wait_for_selector("#evalBody .panel", timeout=20000); pg.wait_for_timeout(800)
            panels = pg.locator("#evalBody .panel")
            panels.nth(0).screenshot(path=str(OUT / "04_evaluation_headline.png"))
            panels.nth(1).screenshot(path=str(OUT / "05_risk_coverage.png"))
            panels.nth(2).screenshot(path=str(OUT / "06_operating_points.png"))
            panels.nth(3).screenshot(path=str(OUT / "06b_prereg_negative_result.png"))
            # 5. reproduce the evaluation (real run)
            pg.click("#btnQuick")
            t = time.time()
            while pg.inner_text("#jobStage") not in ("finished", "failed") and time.time() - t < 300:
                pg.wait_for_timeout(1000)
            pg.wait_for_timeout(600)
            pg.locator("#reproduce").screenshot(path=str(OUT / "07_reproduce_evaluation.png"))
            # 6. about / limits
            pg.click(".tabs button[data-tab=about]"); pg.wait_for_timeout(500)
            pg.locator("#aboutBody").screenshot(path=str(OUT / "08_about_limits.png"))
            b.close()
    finally:
        srv.terminate()
    print("screenshots in", OUT)


if __name__ == "__main__":
    main()
