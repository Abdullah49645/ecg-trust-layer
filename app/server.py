#!/usr/bin/env python
"""ECG Trust Layer - local web app (Flask).

    python app/server.py                # http://127.0.0.1:5050
    python app/server.py --port 8000

Everything runs locally; uploaded images are decoded in memory and never written to disk.
The reference signal is only ever used to *evaluate* the bundled examples after the fact - it is not an input to
the trust score, and uploads have no reference at all.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgtrust import analysis, data, degrade, evaluation, extract, metrics, pipeline, render, trust  # noqa: E402

RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "trust_model.json"
APP_RUNS = RESULTS / "app_runs"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 8000

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), template_folder=str(Path(__file__).parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

EXAMPLE_CONDITIONS = {
    "clean": "Clean render",
    "lowres_mild": "Low resolution (mild)",
    "blur_severe": "Blur (severe)",
    "perspective_severe": "Perspective / tilt (severe)",
    "occlusion_severe": "Occluded trace (severe)",
    "ink_overlay_severe": "Ink strokes over trace (severe)",
    "thick_mild": "Thick trace (mild)",
    "grid_severe": "Dark grid (severe)",
}

_lock = threading.Lock()
_model = {"obj": None}
_store: "OrderedDict[str, dict]" = OrderedDict()
_signals: dict = {}
STORE_MAX = 24


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------
def get_model() -> pipeline.TrustModel:
    if _model["obj"] is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("results/trust_model.json is missing - run `python scripts/run_evaluation.py --preset full` first")
        _model["obj"] = pipeline.TrustModel.load(MODEL_PATH)
    return _model["obj"]


def example_rows() -> list:
    return data.load_selection(ROOT / "configs" / "quick_selection.json")["test"]


def load_example_signal(idx: int):
    rows = example_rows()
    if not 0 <= idx < len(rows):
        raise KeyError(idx)
    if idx not in _signals:
        row = rows[idx]
        _signals[idx] = data.load_record(ROOT / "data" / "quick", row["filename_hr"], "II", row)
    return _signals[idx]


def put(entry: dict) -> str:
    tok = uuid.uuid4().hex[:12]
    with _lock:
        _store[tok] = entry
        while len(_store) > STORE_MAX:
            _store.popitem(last=False)
    return tok


def get(tok: str) -> dict:
    with _lock:
        if tok not in _store:
            raise KeyError(tok)
        _store.move_to_end(tok)
        return _store[tok]


def png_bytes(img_bgr: np.ndarray) -> io.BytesIO:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return io.BytesIO(buf.tobytes())


def _nan_to_none(a, nd=1):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def build_payload(image_rgb: np.ndarray, source: dict, sig=None) -> tuple[dict, dict]:
    """Run the reference-free pipeline; optionally attach evaluation-only information for bundled examples."""
    model = get_model()
    t0 = time.time()
    out = pipeline.analyze_image(image_rgb, model)
    ext = out.get("ext")
    payload: dict = {
        "status": out["status"], "reason": out.get("reason", ""), "source": source,
        "image": {"height": int(image_rgb.shape[0]), "width": int(image_rgb.shape[1])},
        "thresholds": {"t_accept": model.t_accept, "t_reject": model.t_reject, "score": model.operational},
        "runtime_s": None, "beats": [], "evaluation_only": None,
    }
    if ext is None or not ext.ok:
        payload["runtime_s"] = round(time.time() - t0, 2)
        payload["message"] = ("The trace could not be digitized from this image, so no measurement is reported. "
                              + (out.get("reason") or ""))
        return payload, {"image": image_rgb, "out": out}
    cal = out.get("calibration", {})
    payload["calibration"] = {k: cal.get(k) for k in ("px_per_mm_x", "px_per_mm_y", "scale_resolution", "pitch_dispersion_x", "pitch_dispersion_y", "tilt_mm") if k in cal}
    payload["valid_coverage"] = round(out["valid_coverage"], 4)
    xs = ext.x_px.astype(int)
    payload["trace"] = {"x0": int(xs[0]), "y": _nan_to_none(ext.y_px, 1)}
    # missing runs (px) - shown as explicit gaps, never filled
    gaps, start = [], None
    for i, v in enumerate(ext.valid):
        if not v and start is None:
            start = i
        if v and start is not None:
            gaps.append([int(xs[start]), int(xs[i - 1])]); start = None
    if start is not None:
        gaps.append([int(xs[start]), int(xs[-1])])
    payload["gaps"] = gaps
    beats = out["beats"]
    ev_rows = None
    if sig is not None:
        ev = analysis.evaluate_record(ext, out["beatset"], sig.signal_mv, sig.fs)
        ev_rows = ev["beats"] if ev.get("ok") else None
        t_ref = np.arange(len(sig.signal_mv)) / sig.fs
        al = ev.get("alignment") or {}
        payload["evaluation_only"] = {
            "note": "Reference-based. Computed AFTER scoring, only for bundled examples. Never an input to the trust score; not available for uploads.",
            "alignment": {k: al.get(k) for k in ("lag_s", "offset_mv")},
            "waveform_nrmse_dc_lag": (al.get("dc_lag") or {}).get("nrmse"),
            "n_reference_beats": ev.get("n_ref_beats"), "n_extracted_beats": ev.get("n_ext_beats"),
            "hr_ref_bpm": ev.get("hr_ref_bpm"),
        }
        ref_at_cols = np.interp(ext.t_s + al.get("lag_s", 0.0), t_ref, sig.signal_mv) if al.get("ok") else None
    for i, b in enumerate(beats):
        row = {k: b[k] for k in ("beat", "t_r_s", "trust", "trust_heuristic", "trust_learned", "label", "qrs_core_observed", "evidence", "measurements", "rr_eligible")}
        row["window_x"] = [int(xs[max(0, min(b["window_cols"][0], xs.size - 1))]), int(xs[max(0, min(b["window_cols"][1] - 1, xs.size - 1))])]
        row["features"] = {k: round(v, 4) for k, v in b["features"].items()}
        if ev_rows is not None and i < len(ev_rows):
            e = ev_rows[i]
            nr = e.get("nrmse")
            matched = bool(e.get("matched")) and nr is not None and np.isfinite(nr)
            row["evaluation_only"] = {
                "matched_reference_beat": matched, "nrmse": round(float(nr), 4) if matched else None,
                "bad": (not matched) or float(nr) > pipeline.BAD_NRMSE_TOLERANCE,
                "r_amp_err_mv": e.get("r_amp_err_mv"), "rr_err_ms": e.get("rr_err_ms"),
            }
            if ref_at_cols is not None:
                i0, i1 = b["window_cols"]
                sl = slice(i0, i1, 2)
                row["evaluation_only"]["curve"] = {
                    "t": _nan_to_none(ext.t_s[sl] - ext.t_s[i0], 3),
                    "extracted_mv": _nan_to_none(ext.mv[sl] - al["offset_mv"], 3),
                    "reference_mv": _nan_to_none(ref_at_cols[sl], 3),
                }
        payload["beats"].append(row)
    counts = {k: sum(1 for b in beats if b["label"] == k) for k in ("accepted", "uncertain", "rejected")}
    payload["counts"] = counts
    hr = out.get("heart_rate", {})
    payload["heart_rate"] = hr
    payload["runtime_s"] = round(time.time() - t0, 2)
    if not beats:
        payload["message"] = "The trace was digitized but no beats were detected, so nothing can be scored or measured."
    return pipeline._to_py(payload), {"image": image_rgb, "out": out}


def render_example(idx: int, cond: str) -> np.ndarray:
    sig = load_example_signal(idx)
    cfg = degrade.condition_render_config(render.RenderConfig(), cond)
    r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
    img, _ = degrade.apply_condition(r, cond, degrade.condition_seed(sig.record_id, cond))
    return img


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------------------------------
# pages + read-only API
# ---------------------------------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/api/overview")
def api_overview():
    info = {"app": "ECG Trust Layer", "version": "0.1.0", "pipeline_version": pipeline.PIPELINE_VERSION,
            "model_ready": MODEL_PATH.exists(), "results_full": (RESULTS / "results_full.json").exists(),
            "results_quick": (RESULTS / "results_quick.json").exists(),
            "conditions": EXAMPLE_CONDITIONS, "labels": trust.LABEL_COLORS,
            "raw_cache_complete": all((RESULTS / "raw" / f"{s}.json.gz").exists() for s in ("tune", "val", "test"))}
    if MODEL_PATH.exists():
        m = get_model()
        info["model"] = {"operational_score": m.operational, "t_accept": m.t_accept, "t_reject": m.t_reject}
    return jsonify(info)


@app.get("/api/examples")
def api_examples():
    rows = example_rows()
    return jsonify({"records": [{"index": i, "record": r.get("filename_hr", "").split("/")[-1], "ecg_id": r.get("ecg_id")} for i, r in enumerate(rows)],
                    "conditions": [{"id": k, "label": v} for k, v in EXAMPLE_CONDITIONS.items()]})


@app.get("/api/example/<int:idx>/<cond>/analyze")
def api_example_analyze(idx: int, cond: str):
    if cond not in EXAMPLE_CONDITIONS:
        return jsonify({"error": f"unknown condition '{cond}'"}), 404
    try:
        img = render_example(idx, cond)
        payload, entry = build_payload(img, {"kind": "example", "index": idx, "condition": cond,
                                             "label": f"Bundled PTB-XL test-fold record #{idx + 1} · {EXAMPLE_CONDITIONS[cond]} · SYNTHETIC rendering"},
                                       load_example_signal(idx))
    except KeyError:
        return jsonify({"error": "unknown example"}), 404
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    payload["token"] = put(entry)
    return jsonify(payload)


@app.post("/api/analyze")
def api_upload():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "no file field named 'image' in the request"}), 400
    raw = f.read()
    if not raw:
        return jsonify({"error": "the uploaded file is empty"}), 400
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return jsonify({"error": "the file could not be decoded as an image (use PNG or JPEG)"}), 400
    h, w = arr.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        return jsonify({"error": f"image size {w}x{h} is outside the supported range ({MIN_SIDE}-{MAX_SIDE} px per side)"}), 400
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    try:
        payload, entry = build_payload(rgb, {"kind": "upload", "label": f"Uploaded image ({w}x{h}px) - no reference available"})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    payload["token"] = put(entry)
    return jsonify(payload)


@app.get("/api/image/<tok>/original.png")
def api_original(tok):
    try:
        e = get(tok)
    except KeyError:
        return jsonify({"error": "unknown or expired analysis token"}), 404
    return send_file(png_bytes(to_bgr(e["image"])), mimetype="image/png")


@app.get("/api/image/<tok>/beat/<int:k>.png")
def api_beat_crop(tok, k):
    try:
        e = get(tok)
    except KeyError:
        return jsonify({"error": "unknown or expired analysis token"}), 404
    out = e["out"]
    if k < 0 or k >= len(out.get("beats", [])):
        return jsonify({"error": "no such beat"}), 404
    ext, beat = out["ext"], out["beats"][k]
    img = to_bgr(e["image"]).copy()
    xs = ext.x_px.astype(int)
    i0, i1 = beat["window_cols"]
    x0, x1 = int(xs[max(0, i0)]), int(xs[min(xs.size - 1, i1 - 1)])
    ys = ext.y_px[max(0, i0):i1]
    ys = ys[np.isfinite(ys)]
    if ys.size == 0:
        ylo, yhi = 0, img.shape[0]
    else:
        ylo, yhi = int(max(0, ys.min() - 50)), int(min(img.shape[0], ys.max() + 50))
    col = {"accepted": (76, 127, 27), "uncertain": (37, 121, 183), "rejected": (32, 35, 180), "unable": (128, 114, 102)}[beat["label"]]
    for j in range(max(0, i0), min(xs.size, i1)):
        if np.isfinite(ext.y_px[j]):
            cv2.circle(img, (int(xs[j]), int(round(ext.y_px[j]))), 1, col, -1, cv2.LINE_AA)
    crop = img[ylo:yhi, max(0, x0 - 12):min(img.shape[1], x1 + 12)]
    scale = 2 if crop.shape[1] < 700 else 1
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return send_file(png_bytes(crop), mimetype="image/png")


@app.get("/api/results/<which>")
def api_results(which):
    if which not in ("quick", "full"):
        return jsonify({"error": "unknown result set"}), 404
    p = RESULTS / f"results_{which}.json"
    if not p.exists():
        return jsonify({"error": f"{p.name} not found - run `python scripts/run_evaluation.py --preset {which}`"}), 404
    return send_file(p, mimetype="application/json")


# ---------------------------------------------------------------------------------------------------
# "Reproduce the evaluation": a real run in a background thread, with real progress
# ---------------------------------------------------------------------------------------------------
JOB: dict = {"state": "idle"}
_job_lock = threading.Lock()


def _numeric_leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _numeric_leaves(v, f"{prefix}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _numeric_leaves(v, f"{prefix}[{i}]")
    elif isinstance(x, bool) or x is None or isinstance(x, str):
        yield prefix, x
    elif isinstance(x, (int, float)):
        yield prefix, float(x)


def compare_results(new: dict, shipped: dict) -> dict:
    a = dict(_numeric_leaves(new["test"]))
    b = dict(_numeric_leaves(shipped["test"]))
    keys = set(a) & set(b)
    worst, worst_key, mism = 0.0, None, 0
    for k in keys:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if np.isnan(x) and np.isnan(y):
                continue
            d = abs(x - y)
            if d > worst:
                worst, worst_key = d, k
        elif x != y:
            mism += 1
    return {"n_values_compared": len(keys), "keys_only_in_one": len(set(a) ^ set(b)), "max_abs_difference": worst,
            "max_difference_at": worst_key, "non_numeric_mismatches": mism,
            "identical": bool(worst < 1e-9 and mism == 0 and len(set(a) ^ set(b)) == 0)}


def _run_job(mode: str):
    t0 = time.time()

    def cb(stage, done, total, msg):
        JOB.update(stage=stage, done=int(done), total=int(total), message=str(msg), elapsed_s=round(time.time() - t0, 1))

    try:
        run_dir = APP_RUNS / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        JOB["run_dir"] = str(run_dir.relative_to(ROOT))
        if mode == "quick":
            shutil.copy(MODEL_PATH, run_dir / "trust_model.json")
            res = evaluation.run_quick(workers=2, progress=cb, out_dir=run_dir)
            shipped_p = RESULTS / "results_quick.json"
        else:  # re-derive the full statistics from cached per-image rows (fit + evaluate); images are not re-processed
            (run_dir / "raw").symlink_to((RESULTS / "raw").resolve(), target_is_directory=True)
            if (RESULTS / "feasibility_gate.json").exists():
                shutil.copy(RESULTS / "feasibility_gate.json", run_dir / "feasibility_gate.json")
            res = evaluation.run_full(workers=2, progress=cb, use_cache=True, out_dir=run_dir)
            shipped_p = RESULTS / "results_full.json"
        cmp = compare_results(res, json.load(open(shipped_p))) if shipped_p.exists() else None
        h = res["test"]["hypotheses"]
        JOB.update(state="done", stage="done", elapsed_s=round(time.time() - t0, 1), comparison=cmp,
                   summary={"mode": mode, "n_beats": res["test"]["n_beats"], "n_records": res["test"]["n_records"],
                            "aurc_accept_all": h["aurc_accept_all"], "aurc_trust": h["aurc_trust_operational"],
                            "aurc_best_baseline": h["aurc_best_baseline"], "best_baseline": h["best_simple_baseline"],
                            "H1": h["H1_trust_beats_accept_all"], "H2": h["H2_trust_beats_best_baseline"],
                            "operational": res["test"]["operational"]["classes"], "runtime_s": res["runtime_s"]})
    except Exception as exc:  # surfaced to the user, not swallowed
        JOB.update(state="error", error=f"{type(exc).__name__}: {exc}", elapsed_s=round(time.time() - t0, 1))


@app.post("/api/eval/start")
def api_eval_start():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "quick")
    if mode not in ("quick", "full_cached"):
        return jsonify({"error": "mode must be 'quick' or 'full_cached'"}), 400
    if not MODEL_PATH.exists():
        return jsonify({"error": "results/trust_model.json is missing - run the full evaluation first"}), 503
    if mode == "full_cached" and not all((RESULTS / "raw" / f"{s}.json.gz").exists() for s in ("tune", "val", "test")):
        return jsonify({"error": "no cached per-image results found (results/raw). Run `python scripts/run_evaluation.py --preset full` from the command line."}), 409
    with _job_lock:
        if JOB.get("state") == "running":
            return jsonify({"error": "an evaluation is already running"}), 409
        JOB.clear()
        JOB.update(state="running", mode=mode, stage="starting", done=0, total=1, message="starting", elapsed_s=0.0, started=time.time())
        threading.Thread(target=_run_job, args=(mode,), daemon=True).start()
    return jsonify({"started": True, "mode": mode})


@app.get("/api/eval/status")
def api_eval_status():
    return jsonify({k: v for k, v in JOB.items() if k != "started"} or {"state": "idle"})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"file too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}), 413


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()
    if not MODEL_PATH.exists():
        print("WARNING: results/trust_model.json not found - run `python scripts/run_evaluation.py --preset full` first.", file=sys.stderr)
    print(f"ECG Trust Layer running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
