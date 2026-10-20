"""Trust layer: image-conditioned, beat-level evidence about whether an extracted beat can be verified.

Nothing in this module reads a reference signal. Every feature is computed from

* the original image (through the ink map / mask produced by the extractor), and
* the extraction itself (re-rendered back into image coordinates and compared with the image).

Evidence groups (each reported separately so a human can see *why* a beat was accepted or rejected):

``rerender``    re-render the extracted trace into the image and compare with observed ink:
                *unsupported* = fraction of drawn trace pixels not on ink; *unexplained* = ink in the
                trace band that the re-rendered trace does not account for (competing/unclaimed ink).
``ambiguity``   columns with competing ink runs or run height that slope+thickness cannot explain.
``continuity``  roughness (RMS second difference, mm) of the extracted waveform in the beat window.
``missing``     fraction of the beat window with no valid extracted sample.
``calibration`` grid-pitch disagreement between strips, tilt drift, line-fit residual, anisotropy.
``image``       trace SNR (local & global), edge sharpness, resolution (px/mm).

Two aggregators are provided and compared in the evaluation:

1. ``heuristic``  -- transparent. Each feature maps to a 0-1 subscore by a piecewise-linear function between
   a *good* and a *bad* anchor (anchors come from the feature distribution on the tune split: good = 90th
   percentile of clean-image beats, bad = 98th percentile over all tune beats; no error labels involved).
   Group score = mean of its subscores. Trust = 100 * geometric mean of the six group scores (equal group
   weights: no evidence favours any group a priori).
2. ``learned``    -- L2 logistic regression on the standardised raw features predicting "beat error exceeds
   tolerance", trained on the tune split only. Trust = 100 * (1 - P(bad)). The reference is used only to make
   *training labels*, never as an input feature.

A trust value is a rank-ordered evidence index. It is NOT a probability of correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .analysis import BeatSet
from .extract import ExtractionResult

GROUPS: Dict[str, List[str]] = {
    "rerender": ["f_unsupported", "f_unexplained"],
    "ambiguity": ["f_ambiguity"],
    "continuity": ["f_roughness"],
    "missing": ["f_missing"],
    "calibration": ["f_cal_disp", "f_tilt", "f_cal_resid", "f_aniso", "f_scale_ambig"],
    "image": ["f_snr_local", "f_sharp", "f_res", "f_snr"],
}
FEATURES: List[str] = [f for g in GROUPS.values() for f in g]
HIGHER_IS_BETTER = {"f_snr_local", "f_sharp", "f_res", "f_snr"}
FEATURE_DOC = {
    "f_unsupported": "fraction of re-rendered trace pixels that do not lie on observed ink (0 = fully supported)",
    "f_unexplained": "observed trace-band ink not accounted for by the re-rendered trace, per drawn pixel (0 = none)",
    "f_ambiguity": "fraction of window columns with competing ink runs or excess run height",
    "f_roughness": "RMS second difference of the extracted waveform in the window (mm)",
    "f_missing": "fraction of window columns with no valid extracted sample",
    "f_cal_disp": "largest relative disagreement of grid pitch between image strips",
    "f_tilt": "drift of grid lines top-to-bottom (mm): rotation/perspective evidence",
    "f_cal_resid": "residual of grid-line fit (mm)",
    "f_aniso": "|px/mm horizontal / px/mm vertical - 1|",
    "f_scale_ambig": "0 = scale read from major/minor grid lines, 0.5 = resolved only by heart-rate plausibility, 1 = unresolved",
    "f_snr_local": "trace ink level in the window / background noise",
    "f_sharp": "ink mass per run pixel on flat trace parts (1 = crisp, lower = blurred)",
    "f_res": "image resolution (pixels per mm)",
    "f_snr": "global trace ink level / background noise",
}

LABELS = ("accepted", "uncertain", "rejected", "unable")
LABEL_COLORS = {"accepted": "#1b7f4c", "uncertain": "#b7791f", "rejected": "#b42318", "unable": "#667085"}


@dataclass
class ColumnEvidence:
    precision: np.ndarray  # fraction of re-rendered pixels in the column that sit on ink
    unexplained: np.ndarray  # unexplained ink per drawn pixel (capped)
    ambiguous: np.ndarray  # bool
    jerk_mm: np.ndarray  # |second difference| of y (mm)
    sharp: np.ndarray  # run mass per run pixel
    local_level: np.ndarray  # ink level at the chosen run centre
    rerender_mask: np.ndarray  # uint8 HxW re-rendered trace (for visualisation)


def rerender(ext: ExtractionResult) -> np.ndarray:
    """Draw the extracted trace back into image coordinates (valid columns only, broken at gaps)."""
    h, w = ext.image_shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    th = max(1, int(round(ext.stats.get("trace_thickness_px", 2.0))))
    x, y, v = ext.x_px, ext.y_px, ext.valid
    i = 0
    n = x.size
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and v[j + 1]:
            j += 1
        pts = np.stack([x[i : j + 1], y[i : j + 1]], axis=1)
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        if pts.shape[0] >= 2:
            cv2.polylines(canvas, [pts], False, 1, th, cv2.LINE_8)
        else:
            cv2.circle(canvas, tuple(int(c) for c in pts[0, 0]), max(1, th // 2), 1, -1)
        i = j + 1
    return canvas


def column_evidence(ext: ExtractionResult) -> ColumnEvidence:
    """Per-column evidence arrays (aligned with ``ext.x_px``)."""
    canvas = rerender(ext)
    mask = ext.mask
    xs = ext.x_px.astype(int)
    drawn = canvas[:, xs].sum(axis=0).astype(np.float64)
    hit = (canvas[:, xs] & mask[:, xs]).sum(axis=0).astype(np.float64)
    precision = np.where(drawn > 0, hit / np.maximum(drawn, 1.0), 0.0)
    band = cv2.dilate(canvas, np.ones((5, 5), np.uint8))
    unexpl_cnt = (mask[:, xs] & (1 - band[:, xs])).sum(axis=0).astype(np.float64)
    unexplained = np.where(ext.valid, np.minimum(unexpl_cnt / np.maximum(drawn, 1.0), 3.0), 3.0)
    heights = ext.bot_px - ext.top_px + 1
    ambiguous = ext.valid & ((ext.other_mass > 1.5) | (ext.height_excess_px > 2.0))
    # roughness: second difference in mm using the local mm calibration
    y_mm = ext.y_mm.copy()
    jerk = np.full(xs.size, np.nan)
    ok = ext.valid[:-2] & ext.valid[1:-1] & ext.valid[2:]
    jerk_mid = np.abs(y_mm[2:] - 2 * y_mm[1:-1] + y_mm[:-2])
    jerk[1:-1] = np.where(ok, jerk_mid, np.nan)
    sharp = np.where(ext.valid, ext.run_mass / np.maximum(heights, 1.0), np.nan)
    ink = ext.ink
    cy = np.clip(np.round(np.nan_to_num(ext.y_px, nan=0.0)).astype(int), 0, ink.shape[0] - 1)
    level = np.where(ext.valid, ink[cy, xs], np.nan)
    return ColumnEvidence(precision, unexplained, ambiguous, jerk, sharp, level, canvas)


def image_level_features(ext: ExtractionResult) -> Dict[str, float]:
    c = ext.calibration
    st = ext.stats
    return {
        "f_cal_disp": float(max(c.get("pitch_dispersion_x", 1.0), c.get("pitch_dispersion_y", 1.0))),
        "f_tilt": float(c.get("tilt_drift_mm", 0.0)),
        "f_cal_resid": float(max(c.get("fit_resid_mm_x", 9.0), c.get("fit_resid_mm_y", 9.0))),
        "f_aniso": float(c.get("anisotropy", 0.0)),
        "f_scale_ambig": {"major_lines": 0.0, "minor_lines_unreadable": 0.0}.get(c.get("scale_resolution", ""), 1.0 if c.get("scale_resolution") in ("ambiguous", "unresolved", None, "") else 0.5),
        "f_res": float(c.get("px_per_mm_x", 0.0)),
        "f_snr": float(st.get("snr", 0.0)),
    }


def beat_features(ext: ExtractionResult, beats: BeatSet, ev: Optional[ColumnEvidence] = None) -> List[Dict[str, float]]:
    """Reference-free feature vector for every candidate beat."""
    if not ext.ok or beats.n == 0:
        return []
    ev = ev or column_evidence(ext)
    img = image_level_features(ext)
    noise = max(ext.stats.get("noise_sigma", 0.0), 1.0)
    out = []
    flat = np.zeros(ext.n_columns, dtype=bool)
    slope = np.abs(np.gradient(np.nan_to_num(ext.y_px, nan=0.0)))
    flat = ext.valid & (slope < 0.6)
    for i in range(beats.n):
        i0, i1 = [int(k) for k in beats.win[i]]
        i1 = max(i1, i0 + 1)
        v = ext.valid[i0:i1]
        n = max(v.size, 1)
        f = dict(img)
        f["f_missing"] = float(1.0 - v.mean()) if v.size else 1.0
        if v.any():
            f["f_unsupported"] = float(np.mean(1.0 - ev.precision[i0:i1][v]))
            f["f_unexplained"] = float(np.mean(ev.unexplained[i0:i1][v]))
            f["f_ambiguity"] = float(np.mean(ev.ambiguous[i0:i1]))
            j = ev.jerk_mm[i0:i1]
            f["f_roughness"] = float(np.sqrt(np.nanmean(j**2))) if np.isfinite(j).any() else 5.0
            lv = ev.local_level[i0:i1]
            f["f_snr_local"] = float(np.nanmedian(lv) / noise) if np.isfinite(lv).any() else 0.0
            fl = flat[i0:i1] & np.isfinite(ev.sharp[i0:i1])
            f["f_sharp"] = float(np.median(ev.sharp[i0:i1][fl])) if fl.sum() >= 3 else float(np.nanmedian(ev.sharp[i0:i1]))
        else:
            f.update(f_unsupported=1.0, f_unexplained=3.0, f_ambiguity=1.0, f_roughness=5.0, f_snr_local=0.0, f_sharp=0.0)
        # QRS core observed? (R time +- 60 ms must be fully valid for any amplitude measurement)
        t = ext.t_s
        core = (t >= beats.t_r[i] - 0.06) & (t <= beats.t_r[i] + 0.06)
        f["qrs_core_observed"] = float(ext.valid[core].all()) if core.any() else 0.0
        f["beat"] = i
        out.append(f)
    return out


# ----------------------------------------------------------------------------------------------
# aggregators
# ----------------------------------------------------------------------------------------------
def fit_anchors(feats_clean: np.ndarray, feats_all: np.ndarray, names: Sequence[str] = FEATURES) -> Dict[str, dict]:
    """Anchors for the heuristic score from tune-split feature distributions (no error labels)."""
    anchors = {}
    for j, n in enumerate(names):
        c, a = feats_clean[:, j], feats_all[:, j]
        c, a = c[np.isfinite(c)], a[np.isfinite(a)]
        if c.size == 0 or a.size == 0:
            anchors[n] = {"good": 0.0, "bad": 1.0, "active": False}
            continue
        if n in HIGHER_IS_BETTER:
            good, bad = float(np.percentile(c, 10)), float(np.percentile(a, 2))
            active = good > bad + 1e-9
        else:
            good, bad = float(np.percentile(c, 90)), float(np.percentile(a, 98))
            active = bad > good + 1e-9
        anchors[n] = {"good": good, "bad": bad, "active": bool(active)}
    return anchors


def subscore(value: float, good: float, bad: float) -> float:
    """1 at/before ``good``, 0 at/after ``bad``, linear in between (works for either direction)."""
    if not np.isfinite(value):
        return 0.0
    if good == bad:
        return 1.0
    s = (value - bad) / (good - bad)
    return float(min(1.0, max(0.0, s)))


def group_scores(feat: Dict[str, float], anchors: Dict[str, dict]) -> Dict[str, float]:
    out = {}
    for g, names in GROUPS.items():
        subs = [subscore(feat[n], anchors[n]["good"], anchors[n]["bad"]) for n in names if anchors[n].get("active", True)]
        out[g] = float(np.mean(subs)) if subs else 1.0
    return out


def heuristic_trust(feat: Dict[str, float], anchors: Dict[str, dict], groups: Optional[Sequence[str]] = None) -> float:
    gs = group_scores(feat, anchors)
    use = list(groups) if groups is not None else list(GROUPS)
    vals = np.array([max(gs[g], 1e-6) for g in use])
    return float(100.0 * np.exp(np.mean(np.log(vals))))


@dataclass
class LearnedModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    names: List[str] = field(default_factory=lambda: list(FEATURES))

    def p_bad(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mean) / self.scale
        s = z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-s))

    def trust(self, X: np.ndarray) -> np.ndarray:
        return 100.0 * (1.0 - self.p_bad(X))

    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": float(self.intercept)}

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedModel":
        return cls(np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]), float(d["intercept"]), list(d["names"]))


def _prep(X: np.ndarray) -> np.ndarray:
    """Signed-log compression of heavy-tailed features before standardisation."""
    return np.sign(X) * np.log1p(np.abs(X))


def fit_learned(X: np.ndarray, y_bad: np.ndarray, C: float = 0.5, names: Sequence[str] = FEATURES) -> LearnedModel:
    from sklearn.linear_model import LogisticRegression

    Xp = _prep(np.nan_to_num(X, nan=0.0))
    mean, scale = Xp.mean(axis=0), Xp.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit((Xp - mean) / scale, y_bad.astype(int))
    return LearnedModel(mean, scale, clf.coef_[0], float(clf.intercept_[0]), list(names))


def learned_trust(model: LearnedModel, X: np.ndarray) -> np.ndarray:
    Xp = _prep(np.nan_to_num(X, nan=0.0))
    return model.trust(Xp)


def explain(feat: Dict[str, float], anchors: Dict[str, dict]) -> List[dict]:
    """Human-readable evidence list for the beat inspector (worst group first)."""
    gs = group_scores(feat, anchors)
    rows = []
    for g, names in GROUPS.items():
        rows.append({
            "group": g,
            "score": gs[g],
            "features": [{"name": n, "value": float(feat[n]), "doc": FEATURE_DOC[n],
                          "good": anchors[n]["good"], "bad": anchors[n]["bad"], "subscore": subscore(feat[n], anchors[n]["good"], anchors[n]["bad"]) if anchors[n].get("active", True) else None}
                         for n in names],
        })
    rows.sort(key=lambda r: r["score"])
    return rows


def classify(trust: float, qrs_core_observed: bool, t_accept: float, t_reject: float) -> str:
    """Decision rule. A beat whose QRS core is not fully observed can never be accepted for measurement."""
    if trust < t_reject:
        return "rejected"
    if trust >= t_accept and qrs_core_observed:
        return "accepted"
    return "uncertain"
