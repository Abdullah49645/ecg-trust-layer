"""End-to-end processing used by the CLI evaluation, the tests and the web app (one implementation).

* :func:`process_record_condition` -- controlled experiment unit: render a real PTB-XL lead, degrade the image,
  extract, compute reference-free trust features, then (evaluation side only) score against the reference.
* :func:`analyze_image` -- application unit: an arbitrary image in, extraction + trust + labelled measurements out.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import analysis, data, degrade, extract, render, trust

PIPELINE_VERSION = "1.0"
BAD_NRMSE_TOLERANCE = 0.10  # beat is "bad" when its waveform NRMSE exceeds this (fraction of record amplitude)
UNMATCHED_PENALTY = 1.0  # error assigned to detected beats with no reference counterpart (false detection)


def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_py(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def process_record_condition(sig: data.LoadedSignal, condition: str, split: str,
                             render_base: Optional[render.RenderConfig] = None,
                             ext_cfg: Optional[extract.ExtractionConfig] = None) -> dict:
    """Run one controlled experiment unit. The returned dict is JSON-serialisable and carries full provenance."""
    t0 = time.time()
    render_base = render_base or render.RenderConfig()
    ext_cfg = ext_cfg or extract.ExtractionConfig()
    cfg = degrade.condition_render_config(render_base, condition)
    rendered = render.render_ecg(sig.signal_mv, sig.fs, cfg)
    seed = degrade.condition_seed(sig.record_id, condition)
    image, deg_rec = degrade.apply_condition(rendered, condition, seed)
    row: dict = {
        "pipeline_version": PIPELINE_VERSION,
        "split": split,
        "condition": condition,
        "condition_family": degrade.CONDITION_FAMILIES[condition],
        "source": sig.provenance(),
        "rendering": {**cfg.to_dict(), "center_mv": rendered.center_mv, "image_size_px": list(image.shape[1::-1]), **rendered.meta},
        "degradation": deg_rec,
        "extraction_config": ext_cfg.to_dict(),
        "synthetic_rendering": True,
    }
    try:
        ext = extract.extract_trace(image, ext_cfg)
    except extract.ExtractionError as exc:
        row.update(status="unable", reason=str(exc), beats=[], runtime_s=time.time() - t0)
        return _to_py(row)
    row["status"] = ext.status
    row["reason"] = ext.reason
    row["calibration"] = ext.calibration
    row["image_stats"] = ext.stats
    row["valid_coverage"] = ext.coverage
    if not ext.ok:
        row.update(beats=[], runtime_s=time.time() - t0)
        return _to_py(row)
    bs = analysis.find_beats(ext)
    feats = trust.beat_features(ext, bs)
    ev = analysis.evaluate_record(ext, bs, sig.signal_mv, sig.fs)  # reference used HERE only, never for features
    row["alignment"] = ev.get("alignment")
    for k in ("n_ref_beats", "n_ext_beats", "n_matched", "hr_ext_bpm", "hr_ref_bpm", "hr_abs_err_bpm", "ref_beat_recall"):
        row[k] = ev.get(k)
    beats_out = []
    for i, f in enumerate(feats):
        e = ev["beats"][i] if ev.get("ok") else {}
        matched = bool(e.get("matched")) and np.isfinite(e.get("nrmse", np.nan))
        nrmse = float(e["nrmse"]) if matched else float("nan")
        beats_out.append({
            **{k: f[k] for k in trust.FEATURES},
            "qrs_core_observed": f["qrs_core_observed"],
            "beat": i,
            "t_r_s": float(bs.t_r[i]),
            "r_amp_mv_ext": float(bs.amp_mv[i]),
            "rr_ms_ext": float(bs.rr_prev_ms[i]),
            "rr_eligible": bool(i > 0 and bs.rr_gap_valid[i] >= analysis.RR_MIN_GAP_VALID),
            "matched": matched,
            "nrmse": nrmse,
            "err": nrmse if matched else UNMATCHED_PENALTY,
            "bad": (not matched) or nrmse > BAD_NRMSE_TOLERANCE,
            "r_amp_err_mv": e.get("r_amp_err_mv"),
            "r_time_err_ms": e.get("r_time_err_ms"),
            "rr_err_ms": e.get("rr_err_ms"),
        })
    row["beats"] = beats_out
    row["runtime_s"] = time.time() - t0
    return _to_py(row)


# ------------------------------------------------------------------------------------------------
# application: analyse any image (no reference available)
# ------------------------------------------------------------------------------------------------
@dataclass
class TrustModel:
    anchors: Dict[str, dict]
    learned: Optional[trust.LearnedModel]
    operational: str  # "heuristic" | "learned"
    t_accept: float
    t_reject: float
    meta: dict

    @classmethod
    def load(cls, path) -> "TrustModel":
        d = json.load(open(path))
        lm = trust.LearnedModel.from_dict(d["learned"]) if d.get("learned") else None
        return cls(d["anchors"], lm, d["operational_score"], float(d["thresholds"]["t_accept"]), float(d["thresholds"]["t_reject"]), d)

    def score(self, feats: List[dict]) -> List[dict]:
        """Both scores for each beat feature dict, plus the operational one."""
        out = []
        X = np.array([[f[n] for n in trust.FEATURES] for f in feats]) if feats else np.zeros((0, len(trust.FEATURES)))
        learned = trust.learned_trust(self.learned, X) if (self.learned is not None and len(feats)) else [float("nan")] * len(feats)
        for i, f in enumerate(feats):
            h = trust.heuristic_trust(f, self.anchors)
            lr = float(learned[i])
            out.append({"heuristic": h, "learned": lr, "operational": h if self.operational == "heuristic" else lr})
        return out


def analyze_image(image, model: TrustModel, ext_cfg: Optional[extract.ExtractionConfig] = None) -> dict:
    """Extraction + beat-level trust + gated measurements for one image. Reference-free."""
    ext_cfg = ext_cfg or extract.ExtractionConfig()
    out: dict = {"pipeline_version": PIPELINE_VERSION}
    try:
        ext = extract.extract_trace(image, ext_cfg)
    except extract.ExtractionError as exc:
        return {**out, "status": "unable", "reason": str(exc), "beats": [], "ext": None}
    out["status"] = ext.status
    out["reason"] = ext.reason
    out["ext"] = ext
    out["calibration"] = _to_py(ext.calibration)
    out["image_stats"] = _to_py(ext.stats)
    out["valid_coverage"] = ext.coverage
    if not ext.ok:
        out["beats"] = []
        return out
    bs = analysis.find_beats(ext)
    ev = trust.column_evidence(ext)
    feats = trust.beat_features(ext, bs, ev)
    scores = model.score(feats)
    beats = []
    for i, f in enumerate(feats):
        label = trust.classify(scores[i]["operational"], f["qrs_core_observed"] > 0.5, model.t_accept, model.t_reject)
        beats.append({
            "beat": i,
            "t_r_s": float(bs.t_r[i]),
            "window_cols": [int(bs.win[i][0]), int(bs.win[i][1])],
            "trust": scores[i]["operational"],
            "trust_heuristic": scores[i]["heuristic"],
            "trust_learned": scores[i]["learned"],
            "label": label,
            "qrs_core_observed": bool(f["qrs_core_observed"] > 0.5),
            "features": {k: float(f[k]) for k in trust.FEATURES},
            "evidence": _to_py(trust.explain(f, model.anchors)),
            "r_amp_mv": float(bs.amp_mv[i]),
            "rr_ms": float(bs.rr_prev_ms[i]),
            "rr_eligible": bool(i > 0 and bs.rr_gap_valid[i] >= analysis.RR_MIN_GAP_VALID),
        })
    # Measurements are reported ONLY for accepted beats (RR additionally needs the previous beat accepted).
    for i, b in enumerate(beats):
        b["measurements"] = {
            "r_amplitude_mv": b["r_amp_mv"] if b["label"] == "accepted" else None,
            "rr_ms": b["rr_ms"] if (b["label"] == "accepted" and i > 0 and beats[i - 1]["label"] == "accepted" and b["rr_eligible"] and np.isfinite(b["rr_ms"])) else None,
        }
        b["measurements"]["status"] = "reported" if b["label"] == "accepted" else ("withheld: " + b["label"])
    rr = [b["measurements"]["rr_ms"] for b in beats if b["measurements"]["rr_ms"] is not None]
    out["heart_rate"] = {
        "bpm": float(60000.0 / np.mean(rr)) if rr else None,
        "n_rr_intervals": len(rr),
        "method": "mean of RR intervals whose both bounding beats were accepted and whose span was fully observed",
    }
    out["beats"] = beats
    out["beatset"] = bs
    out["column_evidence"] = ev
    out["polarity"] = bs.polarity
    return out
