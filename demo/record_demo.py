#!/usr/bin/env python
"""Record the demo video from the REAL running app, with synthetic (Piper TTS) narration.

    python demo/record_demo.py            # needs: playwright+chromium, piper + voice model in demo/_voice, ffmpeg

Pipeline: narration.json -> Piper TTS per segment -> start the real Flask app -> Playwright performs the real UI
actions (analyze examples, open the evidence inspector, click "Reproduce the Evaluation" and wait for the real job)
while recording -> narration clips placed at their measured start offsets -> ffmpeg mux to MP4.
Outputs in demo/: voiceover.wav, screen_recording.mp4 (silent), ecg_trust_layer_demo.mp4, demo_timeline.md, thumbnail.png
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
WORK = DEMO / "_work"
VOICE = DEMO / "_voice" / "en_US-lessac-medium.onnx"
PORT = 5070
URL = f"http://127.0.0.1:{PORT}/"
SR = 22050


def synth(seg_id: str, text: str) -> Path:
    out = WORK / f"seg_{seg_id}.wav"
    subprocess.run(["piper", "-m", str(VOICE), "-c", str(VOICE) + ".json", "-f", str(out), "--length-scale", "1.04", "--sentence-silence", "0.25"],
                   input=text.encode(), check=True, capture_output=True)
    return out


def wav_info(p: Path):
    with wave.open(str(p)) as w:
        return w.getframerate(), w.getnframes() / w.getframerate()


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


# ------------------------------------------------------------------------------- UI actions (all real clicks)
def a_open(pg):
    pg.goto(URL); pg.wait_for_selector("#exRecord option", state="attached"); pg.wait_for_timeout(600)


def analyze(pg, rec: int, cond: str):
    pg.select_option("#exRecord", str(rec)); pg.wait_for_timeout(250)
    pg.select_option("#exCond", cond); pg.wait_for_timeout(250)
    pg.click("#btnExample")
    pg.wait_for_selector("#inspector:not([hidden])", timeout=90000)
    pg.wait_for_timeout(700)


def scroll_to(pg, sel, block="start"):
    pg.evaluate("([s,b]) => document.querySelector(s).scrollIntoView({behavior:'smooth', block:b})", [sel, block]); pg.wait_for_timeout(900)


def click_beat(pg, i):
    pg.click(f".bchip[data-i='{i}']"); pg.wait_for_timeout(700)


def tab(pg, name):
    pg.click(f".tabs button[data-tab='{name}']"); pg.wait_for_timeout(900)
    pg.evaluate("window.scrollTo({top:0,behavior:'smooth'})"); pg.wait_for_timeout(500)


def worst_beat(pg):
    """index of the lowest-trust beat on screen (read from the real DOM)"""
    return pg.evaluate("() => { const c=[...document.querySelectorAll('.bchip')]; let b=0,m=1e9; c.forEach((x,i)=>{const v=parseFloat(x.querySelector('b').textContent); if(v<m){m=v;b=i}}); return b }")


def first_with_label(pg, label):
    return pg.evaluate("(l) => { const c=[...document.querySelectorAll('.bchip')]; const i=c.findIndex(x=>x.classList.contains(l)); return i }", label)


def move_to(pg, sel):
    box = pg.locator(sel).first.bounding_box()
    if box:
        pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)


ACTIONS = {}


def action(name):
    def deco(fn):
        ACTIONS[name] = fn
        return fn
    return deco


@action("open")
def _(pg, ctx): a_open(pg)


@action("analyze_clean")
def _(pg, ctx):
    move_to(pg, "#btnExample"); analyze(pg, ctx.get("rec", 0), "clean"); scroll_to(pg, "#summary")


@action("show_image")
def _(pg, ctx): scroll_to(pg, ".viewer")


@action("inspect_accepted")
def _(pg, ctx):
    i = first_with_label(pg, "accepted"); i = 3 if i < 0 else min(i + 2, 8)
    move_to(pg, f".bchip[data-i='{i}']"); click_beat(pg, i); scroll_to(pg, "#inspector")


@action("open_evidence")
def _(pg, ctx):
    for k in range(2):
        els = pg.locator(".eg summary")
        if els.count() > k + 1:
            move_to(pg, f".eg:nth-child({k + 2}) summary"); pg.locator(".eg summary").nth(k + 1).click(); pg.wait_for_timeout(900)


@action("analyze_occlusion")
def _(pg, ctx):
    scroll_to(pg, ".controls"); analyze(pg, ctx.get("rec", 0), "occlusion_severe"); scroll_to(pg, ".viewer")


@action("analyze_blur")
def _(pg, ctx):
    scroll_to(pg, ".controls"); analyze(pg, ctx.get("rec", 0), ctx.get("hard_cond", "lowres_severe") if False else "blur_severe"); scroll_to(pg, "#summary")


@action("analyze_hard")
def _(pg, ctx):
    scroll_to(pg, ".controls"); analyze(pg, ctx.get("rec", 0), ctx.get("hard_cond", "perspective_severe")); scroll_to(pg, "#summary")


@action("inspect_rejected")
def _(pg, ctx):
    i = first_with_label(pg, "rejected")
    if i < 0:
        i = worst_beat(pg)
    move_to(pg, f".bchip[data-i='{i}']"); click_beat(pg, i); scroll_to(pg, "#inspector")
    pg.wait_for_timeout(500)


@action("tab_eval")
def _(pg, ctx): tab(pg, "evaluation")


def scroll_h2(pg, text):
    pg.evaluate("""(t) => { const h=[...document.querySelectorAll('#evalBody h2')].find(x=>x.textContent.includes(t)); if(h){ h.closest('.panel').scrollIntoView({behavior:'smooth', block:'start'}); } }""", text)
    pg.wait_for_timeout(1000)


@action("scroll_chart")
def _(pg, ctx): scroll_h2(pg, "Risk")


@action("scroll_op")
def _(pg, ctx): scroll_h2(pg, "Accept / uncertain")


@action("scroll_prereg")
def _(pg, ctx): scroll_h2(pg, "Pre-registered operating point")


@action("scroll_conditions")
def _(pg, ctx): scroll_h2(pg, "Where it works")


@action("scroll_measure")
def _(pg, ctx): scroll_h2(pg, "Do the reported measurements")


@action("reproduce")
def _(pg, ctx):
    scroll_to(pg, "#reproduce"); move_to(pg, "#btnQuick"); pg.wait_for_timeout(400)
    pg.click("#btnQuick"); ctx["job_started"] = True; pg.wait_for_timeout(500)


@action("tab_about")
def _(pg, ctx): tab(pg, "about")


@action("noop")
def _(pg, ctx): pass


INIT_JS = r"""
(() => {
  const mk = () => {
    if (document.getElementById('demo-badge')) return;
    const b = document.createElement('div'); b.id = 'demo-badge'; b.textContent = 'Narration: synthetic AI voice (Piper TTS)';
    b.style.cssText = 'position:fixed;right:12px;top:10px;background:rgba(15,20,24,.8);color:#cfe;font:600 12px system-ui,sans-serif;padding:4px 10px;border-radius:999px;z-index:99999;pointer-events:none';
    document.body.appendChild(b);
    const c = document.createElement('div'); c.id = 'demo-cursor';
    c.style.cssText = 'position:fixed;width:18px;height:18px;border-radius:50%;background:rgba(255,80,60,.55);border:2px solid #fff;box-shadow:0 0 6px #0008;z-index:100000;pointer-events:none;left:-40px;top:-40px;transform:translate(-50%,-50%)';
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  };
  if (document.body) mk(); else document.addEventListener('DOMContentLoaded', mk);
  window.__setCaption = () => {};  // burned-in subtitles removed on request; narration is audio only
})();
"""


def wait_server():
    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "api/overview", timeout=2).read(); return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def main() -> int:
    from playwright.sync_api import sync_playwright

    WORK.mkdir(parents=True, exist_ok=True)
    script = json.load(open(DEMO / "narration.json"))
    ctx_vars = script.get("vars", {})
    segs = script["segments"]
    durs = {}
    for s in segs:
        p = synth(s["id"], s["text"])
        sr, d = wav_info(p)
        assert sr == SR, sr
        durs[s["id"]] = d
        print(f"tts {s['id']:14s} {d:5.1f}s")

    srv = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py"), "--port", str(PORT)], stdout=open(WORK / "server.log", "w"), stderr=subprocess.STDOUT)
    timeline = []
    try:
        wait_server()
        with sync_playwright() as p:
            b = p.chromium.launch()
            vdir = WORK / "video"
            for f in vdir.glob("*.webm") if vdir.exists() else []:
                f.unlink()
            context = b.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(vdir), record_video_size={"width": 1440, "height": 900})
            context.add_init_script(INIT_JS)
            pg = context.new_page()
            t0 = time.monotonic()
            now = lambda: time.monotonic() - t0  # noqa: E731
            pg.wait_for_timeout(500)
            for s in segs:
                start = now()
                pg.evaluate("t => window.__setCaption && window.__setCaption(t)", s["text"])
                for name in s["actions"]:
                    ACTIONS[name](pg, ctx_vars)
                need = start + durs[s["id"]] + s.get("pad", 0.6)
                while now() < need:
                    pg.wait_for_timeout(100)
                if s.get("wait_job"):
                    t_wait = now()
                    while True:
                        st = pg.evaluate("() => document.querySelector('#jobStage').textContent")
                        if st in ("finished", "failed"):
                            break
                        if now() - t_wait > 600:
                            raise RuntimeError("job did not finish")
                        pg.wait_for_timeout(500)
                    pg.wait_for_timeout(1500); scroll_to(pg, "#jobResult", "center"); pg.wait_for_timeout(1200)
                timeline.append({"id": s["id"], "start": round(start, 2), "audio_s": round(durs[s["id"]], 2), "end": round(now(), 2), "text": s["text"]})
                print(f"seg {s['id']:14s} {start:6.1f} -> {now():6.1f}")
            pg.evaluate("t => window.__setCaption && window.__setCaption('')", "")
            pg.wait_for_timeout(800)
            total = now()
            vid = pg.video.path()
            context.close(); b.close()
    finally:
        srv.terminate()
    # ---- audio track at measured offsets
    n = int((total + 1.0) * SR)
    track = np.zeros(n, np.float32)
    for t in timeline:
        a = read_wav(WORK / f"seg_{t['id']}.wav")
        i0 = int(t["start"] * SR)
        track[i0:i0 + a.size] += a[: max(0, n - i0)]
    with wave.open(str(DEMO / "voiceover.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    # ---- video encode + mux
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-an", str(DEMO / "screen_recording.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(DEMO / "screen_recording.mp4"), "-i", str(DEMO / "voiceover.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
                    "-movflags", "+faststart", str(DEMO / "ecg_trust_layer_demo.mp4")], check=True)
    json.dump(timeline, open(WORK / "timeline.json", "w"), indent=1)
    print(f"done: total {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
