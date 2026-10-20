"""Image -> signal extraction for single-lead ECG-paper images.

Pipeline (all steps use only the image and the nominal paper standard 25 mm/s, 10 mm/mV):

1. **Grid calibration.** The minor/major pitch of the millimetre grid is found with a
   fold-and-contrast search (a true minor pitch shows one distinct "major" line per five
   lines; 5x the true pitch does not). Line positions are then fitted *per strip* (row bands
   for the time axis, column bands for the amplitude axis) so that mild rotation / perspective
   is rectified locally, and the disagreement between strips is kept as calibration-uncertainty
   evidence.
2. **Grid suppression.** Coloured grids are removed by using ``255 - max(R,G,B)`` as the ink
   channel; residual (e.g. dark-gray) grid lines are subtracted using median line profiles.
3. **Trace/background separation.** Adaptive threshold at a fraction of the trace ink level
   (median of per-column maxima); text and small blobs are removed by connected components.
4. **Trace tracking.** Per column, candidate ink runs are linked by Viterbi dynamic programming
   with an explicit *missing* state (no silent filling). Tall runs are handled with local
   extremum correction.
5. **Physical calibration.** Pixel coordinates map to paper millimetres through the local grid,
   then to seconds (25 mm/s) and millivolts (10 mm/mV). The vertical origin (DC level) is not
   observable from an image without a calibration pulse, so the output is median-centred.

The reference signal is never used here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks


class ExtractionError(ValueError):
    """Raised for input that cannot be processed at all (empty/unreadable/wrong shape)."""


@dataclass
class ExtractionConfig:
    mm_per_s: float = 25.0
    mm_per_mv: float = 10.0
    px_per_mm_hint: Optional[float] = None  # used only if the grid cannot be read
    minor_pitch_px_range: Tuple[float, float] = (1.5, 24.0)
    n_strips_x: int = 6  # row bands used to calibrate the time axis
    n_strips_y: int = 8  # column bands used to calibrate the amplitude axis
    thr_frac: float = 0.45  # threshold as a fraction of the trace ink level
    min_component_width_mm: float = 3.0
    dp_missing_cost_px: float = 7.0
    dp_main_component_penalty_px: float = 4.0
    dp_reentry_scale: float = 0.35
    max_candidates_per_column: int = 6
    min_image_px: int = 64

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["minor_pitch_px_range"] = list(self.minor_pitch_px_range)
        return d


@dataclass
class ExtractionResult:
    status: str  # "ok" | "unable"
    reason: str
    config: dict
    # per image column inside the trace x-extent -------------------------------------------------
    x_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y_px: np.ndarray = field(default_factory=lambda: np.zeros(0))  # chosen sample position (px), NaN if missing
    top_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bot_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    valid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    t_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mv: np.ndarray = field(default_factory=lambda: np.zeros(0))  # NaN where not valid
    x_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_runs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # competing ink runs in the column
    run_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))  # ink mass of chosen run (units of trace level * px)
    other_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))  # ink mass of non-chosen runs in the column
    height_excess_px: np.ndarray = field(default_factory=lambda: np.zeros(0))  # run height beyond what slope+thickness explain
    # image level -------------------------------------------------------------------------------------
    calibration: Dict[str, object] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)
    # intermediate rasters (kept for the trust layer / visualisation) ----------------------------
    ink: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    image_shape: Tuple[int, int] = (0, 0)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def n_columns(self) -> int:
        return int(self.x_px.size)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def summary(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "n_columns": self.n_columns,
            "valid_coverage": self.coverage,
            "calibration": self.calibration,
            "stats": self.stats,
        }


# ----------------------------------------------------------------------------------------------
# input handling
# ----------------------------------------------------------------------------------------------
def _validate_image(img) -> np.ndarray:
    if img is None:
        raise ExtractionError("image is None (unreadable file?)")
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ExtractionError(f"expected HxWx3 image, got shape {img.shape}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.size == 0 or min(img.shape[:2]) < 8:
        raise ExtractionError("image is empty or too small")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------------------------
# grid calibration
# ----------------------------------------------------------------------------------------------
def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    prof = np.asarray(profile, dtype=np.float64)
    prof = prof - np.median(prof)
    return np.convolve(prof, np.array([0.25, 0.5, 0.25]), mode="same")


def _detect_peaks(sm: np.ndarray):
    """Sub-pixel positions and heights of grid-line peaks in a median profile."""
    # 97th percentile: dominated by grid lines (which occupy many samples), not by a few dark trace rows
    hi = float(np.percentile(sm, 97))
    if hi < 2.0:
        return None
    peaks, _ = find_peaks(sm, prominence=0.3 * hi)
    if peaks.size < 8:
        return None
    # A flat stretch of the ECG trace can fill most of a row/column band and appear as a very strong
    # "line". Grid lines have similar heights to each other, ink is much darker: drop height outliers.
    hts_all = sm[peaks]
    keep_pk = hts_all <= 1.8 * np.percentile(hts_all, 85)  # relative to the *major*-line level, so majors survive
    peaks = peaks[keep_pk]
    if peaks.size < 8:
        return None
    pos, hts = [], []
    for pk in peaks:
        lo, up = max(0, pk - 2), min(sm.size, pk + 3)
        w = np.clip(sm[lo:up], 0, None)
        pos.append(float((np.arange(lo, up) * w).sum() / max(w.sum(), 1e-9)))
        hts.append(float(sm[pk]))
    return np.array(pos), np.array(hts)


def _assign_indices(pos: np.ndarray, hts: np.ndarray, p_seed: float):
    """Give each detected line an integer index (1 index = 1 minor pitch) while refining the pitch."""
    kp, kh, ks = [pos[0]], [hts[0]], [0]
    p_cur = p_seed
    for x, hh in zip(pos[1:], hts[1:]):
        d = x - kp[-1]
        if d < 0.5 * p_cur:  # spurious double peak: keep the stronger
            if hh > kh[-1]:
                kp[-1], kh[-1] = x, hh
            continue
        ks.append(ks[-1] + int(round(d / p_cur)))
        kp.append(x)
        kh.append(hh)
        if len(ks) >= 6:
            p_cur = float(np.polyfit(ks, kp, 1)[0])
    return np.array(ks, dtype=np.float64), np.array(kp), np.array(kh)


def _analyse_profile(profile: np.ndarray, pmin: float, pmax: float) -> Optional[dict]:
    """Find grid lines in a 1-D median darkness profile."""
    sm = _smooth_profile(profile)
    det = _detect_peaks(sm)
    if det is None:
        return None
    pos, hts = det
    d = np.diff(pos)
    d = d[d > 0.5]
    if d.size < 6:
        return None
    p_seed = float(np.percentile(d, 40))
    ks, kp, kh = _assign_indices(pos, hts, p_seed)
    keep = np.ones(ks.size, dtype=bool)
    for _ in range(4):
        if keep.sum() < 8:
            return None
        co = np.polyfit(ks[keep], kp[keep], 1)
        res = kp - np.polyval(co, ks)
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        newkeep = np.abs(res) < max(3.0 * sd, 0.8)
        if newkeep.sum() == keep.sum():
            break
        keep = newkeep
    if keep.sum() < 8:
        return None
    p_minor = float(np.polyfit(ks[keep], kp[keep], 1)[0])
    ksk, kpk, khk = ks[keep], kp[keep], kh[keep]
    minor_missing = False
    if p_minor > pmax * 1.15 and pmin * 0.85 <= p_minor / 5.0 <= pmax * 1.15:
        # only the 5 mm lines were readable (minor lines too faint): treat detected lines as 5 mm apart
        minor_missing = True
        lidx = ksk - ksk[0]
        co = np.polyfit(lidx, kpk, 1)
        return {
            "p_minor_px": p_minor / 5.0, "px_per_mm": p_minor / 5.0, "major_distinct": True, "unit_mm": 5.0,
            "lines_px": kpk, "lines_idx": lidx, "a": float(co[1]), "b": float(co[0]),
            "resid_px": float(np.std(kpk - np.polyval(co, lidx))), "n_lines": int(keep.sum()),
            "height": float(np.median(khk)), "major_ratio": float("inf"), "minor_missing": True,
        }
    if not (pmin * 0.85 <= p_minor <= pmax * 1.15):
        return None
    # which index residue (mod 5) holds the strong "major" lines?
    means = np.array([khk[(ksk.astype(int) % 5) == r].mean() if np.any((ksk.astype(int) % 5) == r) else 0.0 for r in range(5)])
    order = np.argsort(means)[::-1]
    r_major = int(order[0])
    distinct = bool(means[order[0]] >= 1.25 * max(means[order[1]], 1e-6))
    if distinct:
        sel = (ksk.astype(int) % 5) == r_major
        unit_mm = 5.0
        lines = kpk[sel]
        lidx = (ksk[sel] - r_major) / 5.0
        # absolute index relative to first major line
        lidx = lidx - lidx[0]
    else:
        unit_mm = 1.0
        lines = kpk
        lidx = ksk - ksk[0]
    if lines.size < 4 or float(np.ptp(lidx)) < 3:
        return None  # too few distinct lines to define a pitch
    resid_px = float(np.std(kpk - np.polyval(np.polyfit(ksk, kpk, 1), ksk)))
    if resid_px > 0.15 * p_minor:
        return None  # lines are not regularly spaced: this is not a ruled grid
    if keep.sum() * p_minor < 0.35 * sm.size:
        return None  # lines cover too little of the image for a trustworthy pitch
    co = np.polyfit(lidx, lines, 1)
    return {
        "p_minor_px": p_minor,
        "px_per_mm": p_minor,
        "major_distinct": distinct,
        "unit_mm": unit_mm,
        "lines_px": lines,
        "lines_idx": lidx,
        "a": float(co[1]),
        "b": float(co[0]),  # px per unit_mm
        "resid_px": float(np.std(kpk - np.polyval(np.polyfit(ksk, kpk, 1), ksk))),
        "n_lines": int(keep.sum()),
        "height": float(np.median(khk)),
        "major_ratio": float(means[order[0]] / max(means[order[1]], 1e-6)),
        "minor_missing": False,
    }


def _strip_map(fit: dict, ref: dict) -> Optional[dict]:
    """Polynomial map px -> mm for one strip, with line indices made consistent with ``ref``."""
    lines = fit["lines_px"]
    if fit["unit_mm"] != ref["unit_mm"]:
        return None
    m = np.round((lines - ref["a"]) / ref["b"])
    # drop duplicates
    uniq, ui = np.unique(m, return_index=True)
    lines, m = lines[ui], uniq
    if lines.size < 4:
        return None
    mm = m * ref["unit_mm"]
    deg = 2 if lines.size >= 8 else 1
    co = np.polyfit(lines, mm, deg)
    resid = float(np.std(mm - np.polyval(co, lines)))
    lo, hi = float(lines.min()), float(lines.max())
    return {"co": co, "lo": lo, "hi": hi, "resid_mm": resid, "n": int(lines.size)}


def _eval_map(mp: dict, px) -> np.ndarray:
    px = np.asarray(px, dtype=np.float64)
    c = np.clip(px, mp["lo"], mp["hi"])
    v = np.polyval(mp["co"], c)
    der = np.polyval(np.polyder(mp["co"]), c)
    return v + der * (px - c)


def calibrate_grid(darkness: np.ndarray, cfg: ExtractionConfig) -> dict:
    """Estimate local grid geometry. ``darkness`` is a float image where grid lines are bright."""
    h, w = darkness.shape
    pmin, pmax = cfg.minor_pitch_px_range
    out: dict = {"ok": False, "reason": ""}
    per_axis = {}
    for axis in ("x", "y"):
        n_strips = cfg.n_strips_x if axis == "x" else cfg.n_strips_y
        length = h if axis == "x" else w
        edges = np.linspace(0, length, n_strips + 1).astype(int)
        strips = []
        for a, b in zip(edges[:-1], edges[1:]):
            if axis == "x":  # row band -> vertical lines (x positions): calibrates TIME
                prof = np.median(darkness[a:b], axis=0)
            else:  # column band -> horizontal lines (y positions): calibrates AMPLITUDE
                prof = np.median(darkness[:, a:b], axis=1)
            strips.append((0.5 * (a + b), prof))
        glob = np.median(darkness, axis=0) if axis == "x" else np.median(darkness, axis=1)
        cand = []
        g = _analyse_profile(glob, pmin, pmax)
        if g is not None:
            cand.append(g)
        fits = [(c, _analyse_profile(p, pmin, pmax)) for c, p in strips]
        good = [(c, f) for c, f in fits if f is not None]
        if not good:
            out["reason"] = f"no grid lines found along {axis}"
            return out
        # reference: global if it has about as many lines as the best strip, else the best strip
        best_strip = max(good, key=lambda cf: cf[1]["n_lines"] * cf[1]["height"])[1]
        ref = g if (g is not None and g["n_lines"] >= 0.8 * best_strip["n_lines"] and g["resid_px"] <= 1.5 * best_strip["resid_px"] + 0.3) else best_strip
        maps = []
        for c, f in good:
            mp = _strip_map(f, ref)
            if mp is not None and abs(f["px_per_mm"] / ref["px_per_mm"] - 1.0) < 0.12:
                mp["center"] = float(c)
                mp["px_per_mm"] = f["px_per_mm"]
                mp["resid_px"] = f["resid_px"]
                mp["height"] = f["height"]
                mp["major_distinct"] = f["major_distinct"]
                maps.append(mp)
        if not maps:
            out["reason"] = f"grid maps inconsistent along {axis}"
            return out
        per_axis[axis] = {"maps": maps, "ref": ref, "n_strips": n_strips}

    ppm_x = np.array([m["px_per_mm"] for m in per_axis["x"]["maps"]])
    ppm_y = np.array([m["px_per_mm"] for m in per_axis["y"]["maps"]])
    # tilt: change of the same vertical line's x position between the top and bottom strips (in mm)
    mx = sorted(per_axis["x"]["maps"], key=lambda m: m["center"])
    ref_px = 0.5 * (per_axis["x"]["ref"]["lines_px"][0] + per_axis["x"]["ref"]["lines_px"][-1])
    drift = abs(float(_eval_map(mx[-1], ref_px) - _eval_map(mx[0], ref_px))) if len(mx) > 1 else 0.0
    def _line_positions(ref: dict, length: int) -> np.ndarray:
        step = ref["b"] / ref["unit_mm"]  # px per mm
        if not np.isfinite(step) or step <= 1.0:
            return np.zeros(0)
        j = np.arange(np.floor((0 - ref["a"]) / step) - 1, np.ceil((length - 1 - ref["a"]) / step) + 2)
        pos = ref["a"] + step * j
        return pos[(pos >= 0) & (pos <= length - 1)]

    out.update(
        ok=True,
        grid_cols_px=_line_positions(per_axis["x"]["ref"], w),
        grid_rows_px=_line_positions(per_axis["y"]["ref"], h),
        px_per_mm_x=float(np.median(ppm_x)),
        px_per_mm_y=float(np.median(ppm_y)),
        pitch_dispersion_x=float((ppm_x.max() - ppm_x.min()) / np.median(ppm_x)),
        pitch_dispersion_y=float((ppm_y.max() - ppm_y.min()) / np.median(ppm_y)),
        anisotropy=float(abs(np.median(ppm_x) / np.median(ppm_y) - 1.0)),
        fit_resid_px_x=float(np.median([m["resid_px"] for m in per_axis["x"]["maps"]])),
        fit_resid_px_y=float(np.median([m["resid_px"] for m in per_axis["y"]["maps"]])),
        fit_resid_mm_x=float(np.median([m["resid_mm"] for m in per_axis["x"]["maps"]])),
        fit_resid_mm_y=float(np.median([m["resid_mm"] for m in per_axis["y"]["maps"]])),
        n_strips_ok_x=len(per_axis["x"]["maps"]),
        n_strips_ok_y=len(per_axis["y"]["maps"]),
        n_strips_x=per_axis["x"]["n_strips"],
        n_strips_y=per_axis["y"]["n_strips"],
        grid_contrast=float(np.median([m["height"] for m in per_axis["x"]["maps"] + per_axis["y"]["maps"]])),
        major_distinct=bool(per_axis["x"]["ref"]["major_distinct"] and per_axis["y"]["ref"]["major_distinct"]),
        tilt_drift_mm=drift,
        maps_x=per_axis["x"]["maps"],
        maps_y=per_axis["y"]["maps"],
    )
    return out


# ----------------------------------------------------------------------------------------------
# main extractor
# ----------------------------------------------------------------------------------------------
def _cap_profile(p: np.ndarray, pos: Optional[np.ndarray]) -> np.ndarray:
    """Grid-line ink model along one axis: profile values at calibrated line positions, capped at 1.5x the
    typical line level so a flat trace dominating a row/column is never mistaken for a grid line."""
    q = np.zeros_like(p)
    if pos is None or len(pos) < 4:
        return q
    idx = np.clip(np.round(pos).astype(int), 0, p.size - 1)
    typ = float(np.median(p[idx]))
    cap = 1.5 * max(typ, 0.0) + 2.0
    for d in (-1, 0, 1):
        near = np.clip(idx + d, 0, p.size - 1)
        q[near] = np.minimum(np.maximum(p[near], 0.0), cap)
    return q


def _grid_model_subtract(ink: np.ndarray, grid_cols: Optional[np.ndarray], grid_rows: Optional[np.ndarray]) -> np.ndarray:
    """Subtract residual (e.g. dark-gray) grid lines using median line profiles at the calibrated line positions."""
    base = float(np.median(ink))
    gc = _cap_profile(np.median(ink, axis=0) - base, grid_cols)
    gr = _cap_profile(np.median(ink, axis=1) - base, grid_rows)
    g = np.maximum(gc[None, :], gr[:, None])
    return np.clip(ink - 0.95 * g, 0, None)


def _runs_in_column(col: np.ndarray, ink_col: np.ndarray, level: float):
    ys = np.flatnonzero(col)
    if ys.size == 0:
        return []
    splits = np.flatnonzero(np.diff(ys) > 2) + 1  # bridge 1-px gaps
    runs = []
    for seg in np.split(ys, splits):
        top, bot = int(seg[0]), int(seg[-1])
        w = ink_col[top : bot + 1]
        mass = float(w.sum() / max(level, 1e-6))
        cen = float((np.arange(top, bot + 1) * w).sum() / max(w.sum(), 1e-6))
        runs.append((top, bot, cen, mass))
    return runs


def _gap(a_top, a_bot, b_top, b_bot) -> float:
    if b_top > a_bot:
        return b_top - a_bot
    if a_top > b_bot:
        return a_top - b_bot
    return 0.0


def extract_trace(image, cfg: Optional[ExtractionConfig] = None) -> ExtractionResult:
    """Extract the ECG trace from ``image`` (HxWx3 uint8). Never uses a reference signal.

    Returns an :class:`ExtractionResult` with ``status == "unable"`` (and a reason) when the grid
    or the trace cannot be read; raises :class:`ExtractionError` for unusable input.
    """
    cfg = cfg or ExtractionConfig()
    img = _validate_image(image)
    h, w = img.shape[:2]
    if min(h, w) < cfg.min_image_px:
        raise ExtractionError(f"image too small ({w}x{h})")
    res = ExtractionResult(status="unable", reason="", config=cfg.to_dict(), image_shape=(h, w))

    imgf = img.astype(np.float32)
    mx = imgf.max(axis=2)
    lum = imgf @ np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR/RGB agnostic enough for darkness
    ink0 = 255.0 - mx  # coloured grid -> ~0, black trace -> ~235
    ink0 = cv2.GaussianBlur(ink0, (0, 0), 0.6)

    # ---- grid calibration (uses luminance darkness: pink lines are visible there) ----
    dark = 255.0 - lum
    try:
        cal = calibrate_grid(dark, cfg)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:  # degenerate geometry -> "unable", never a crash
        cal = {"ok": False, "reason": f"grid fit numerically degenerate ({type(exc).__name__})"}
    if not cal["ok"]:
        if cfg.px_per_mm_hint:
            cal = {"ok": False, "reason": cal["reason"], "px_per_mm_x": cfg.px_per_mm_hint, "px_per_mm_y": cfg.px_per_mm_hint,
                   "pitch_dispersion_x": 1.0, "pitch_dispersion_y": 1.0, "anisotropy": 0.0, "grid_contrast": 0.0,
                   "tilt_drift_mm": 0.0, "fit_resid_px_x": 9.0, "fit_resid_px_y": 9.0, "fit_resid_mm_x": 9.0, "fit_resid_mm_y": 9.0, "maps_x": [], "maps_y": [], "major_distinct": False,
                   "n_strips_ok_x": 0, "n_strips_ok_y": 0, "n_strips_x": cfg.n_strips_x, "n_strips_y": cfg.n_strips_y,
                   "used_hint": True}
        else:
            res.reason = f"grid calibration failed: {cal['reason']}"
            res.calibration = {k: v for k, v in cal.items() if k not in ("maps_x", "maps_y", "grid_cols_px", "grid_rows_px")}
            return res
    ppm_x, ppm_y = cal["px_per_mm_x"], cal["px_per_mm_y"]

    # ---- grid suppression and trace level ----
    ink = _grid_model_subtract(ink0, cal.get("grid_cols_px"), cal.get("grid_rows_px"))
    colmax = ink.max(axis=0)
    level = float(np.median(colmax))
    bg = ink[ink < 0.5 * max(level, 1.0)]
    noise = float(1.4826 * np.median(np.abs(bg - np.median(bg)))) if bg.size else 0.0
    if level < 12.0 or level < 4.0 * noise:
        res.reason = f"no distinguishable trace ink (level={level:.1f}, noise={noise:.1f})"
        res.calibration = {k: v for k, v in cal.items() if k not in ("maps_x", "maps_y", "grid_cols_px", "grid_rows_px")}
        res.stats = {"trace_level": level, "noise_sigma": noise}
        return res
    thr = cfg.thr_frac * level
    mask = (ink > thr).astype(np.uint8)

    # ---- remove text / small blobs via connected components ----
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_w = cfg.min_component_width_mm * ppm_x
    keep = np.zeros(n, dtype=bool)
    masses = np.zeros(n)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_WIDTH] >= min_w:
            keep[i] = True
            masses[i] = st[i, cv2.CC_STAT_AREA]
    if not keep.any():
        res.reason = "no trace-like connected component wide enough"
        res.calibration = {k: v for k, v in cal.items() if k not in ("maps_x", "maps_y", "grid_cols_px", "grid_rows_px")}
        res.stats = {"trace_level": level, "noise_sigma": noise}
        return res
    main_label = int(np.argmax(masses))
    # Only components that span a long x range count as trace; short ones are strokes/blobs (still kept as candidates)
    mask_k = keep[lab].astype(np.uint8)
    cols = np.flatnonzero(mask_k.any(axis=0))
    x0, x1 = int(cols[0]), int(cols[-1])
    xs = np.arange(x0, x1 + 1)

    # ---- candidate runs per column ----
    cand: List[list] = []
    for x in xs:
        runs = _runs_in_column(mask_k[:, x], ink[:, x], level)
        # tag runs by main component membership
        tagged = []
        for (top, bot, cen, mass) in runs:
            lbl = lab[int(round(cen)), x] if mask_k[int(round(cen)), x] else lab[top, x]
            tagged.append((top, bot, cen, mass, int(lbl) == main_label))
        tagged.sort(key=lambda r: -r[3])
        cand.append(tagged[: cfg.max_candidates_per_column])

    # ---- Viterbi tracking with explicit missing state ----
    ncol = xs.size
    INF = 1e18
    cost: List[np.ndarray] = []
    back: List[np.ndarray] = []  # index of previous state (len = n_cands+1; last = missing)
    miss_c = cfg.dp_missing_cost_px
    # state arrays per column: candidates 0..m-1, missing = m
    prev_cost = None
    prev_cands = None
    carry_iv: List[Tuple[float, float]] = []  # carry interval for missing state per column
    for ci in range(ncol):
        cs = cand[ci]
        m = len(cs)
        c = np.full(m + 1, INF)
        b = np.zeros(m + 1, dtype=np.int64)
        em = np.array([(0.0 if r[4] else cfg.dp_main_component_penalty_px) for r in cs] + [miss_c])
        if ci == 0:
            c[:] = em
            b[:] = -1
            carry = (0.0, 0.0)
            carry_iv.append(carry)
        else:
            pm = len(prev_cands)
            # run -> run, missing -> run
            for j, r in enumerate(cs):
                best, bi = INF, 0
                for i, pr in enumerate(prev_cands):
                    t = prev_cost[i] + _gap(pr[0], pr[1], r[0], r[1])
                    if t < best:
                        best, bi = t, i
                # from missing
                civ = carry_iv[ci - 1]
                if civ != (0.0, 0.0) or True:
                    t = prev_cost[pm] + cfg.dp_reentry_scale * min(_gap(civ[0], civ[1], r[0], r[1]), 60.0) if civ[1] > 0 else prev_cost[pm]
                    if t < best:
                        best, bi = t, pm
                c[j] = best + em[j]
                b[j] = bi
            # into missing
            best, bi = INF, 0
            for i in range(pm + 1):
                t = prev_cost[i]
                if t < best:
                    best, bi = t, i
            c[m] = best + em[m]
            b[m] = bi
            # carry interval for the missing state
            if bi < pm:
                pr = prev_cands[bi]
                carry_iv.append((pr[0], pr[1]))
            else:
                carry_iv.append(carry_iv[ci - 1])
        if ci == 0:
            pass
        cost.append(c)
        back.append(b)
        prev_cost, prev_cands = c, cs
    # backtrack
    state = int(np.argmin(cost[-1]))
    path = [0] * ncol
    for ci in range(ncol - 1, -1, -1):
        path[ci] = state
        state = int(back[ci][state]) if ci > 0 else state

    top = np.full(ncol, np.nan)
    bot = np.full(ncol, np.nan)
    cen = np.full(ncol, np.nan)
    mass = np.zeros(ncol)
    other = np.zeros(ncol)
    nrun = np.zeros(ncol, dtype=int)
    valid = np.zeros(ncol, dtype=bool)
    for ci in range(ncol):
        cs = cand[ci]
        nrun[ci] = len(cs)
        s = path[ci]
        if s < len(cs):
            r = cs[s]
            top[ci], bot[ci], cen[ci], mass[ci] = r[0], r[1], r[2], r[3]
            valid[ci] = True
            other[ci] = sum(rr[3] for k, rr in enumerate(cs) if k != s)
        else:
            other[ci] = sum(rr[3] for rr in cs)

    # nominal trace thickness (flat columns)
    hts = (bot - top + 1)
    slope = np.abs(np.gradient(np.where(valid, cen, np.nan)))
    flat = valid & (slope < 0.6)
    th = float(np.nanmedian(hts[flat])) if flat.sum() > 20 else float(np.nanmedian(hts[valid])) if valid.any() else 2.0
    th = max(th, 1.0)

    # expected height from slope; excess is ambiguity evidence
    cen_f = _interp_nan(cen)
    sl = np.abs(np.gradient(cen_f)) if ncol > 2 else np.zeros(ncol)
    sl = np.maximum(sl, np.abs(np.roll(cen_f, -1) - cen_f) * 0.5 + np.abs(cen_f - np.roll(cen_f, 1)) * 0.5)
    excess = np.where(valid, np.maximum(0.0, hts - (th + sl)), 0.0)

    # sample position with extremum correction (peaks use the top edge, valleys the bottom edge)
    y_est = cen.copy()
    k = 2
    for ci in range(k, ncol - k):
        if not valid[ci] or hts[ci] <= 2.5 * th:
            continue
        nb = slice(ci - k, ci + k + 1)
        if not valid[nb].all():
            continue
        if top[ci] <= np.min(top[nb]):
            y_est[ci] = top[ci] + th / 2.0
        elif bot[ci] >= np.max(bot[nb]):
            y_est[ci] = bot[ci] - th / 2.0

    # ---- physical calibration through the local grid ----
    x_mm, y_mm = _to_mm(xs.astype(np.float64), y_est, valid, cal, ppm_x, ppm_y, h, w)
    first = int(np.flatnonzero(valid)[0]) if valid.any() else 0
    t_s = (x_mm - x_mm[first]) / cfg.mm_per_s
    mv_raw = -(y_mm) / cfg.mm_per_mv
    mv = np.where(valid, mv_raw - np.nanmedian(mv_raw[valid]), np.nan) if valid.any() else mv_raw

    scale_res = "major_lines" if cal.get("major_distinct") else "unresolved"
    if cal.get("minor_missing"):
        scale_res = "minor_lines_unreadable"
    if scale_res == "unresolved" and valid.sum() >= 20 and not cal.get("used_hint"):
        # Only a uniform line lattice is readable: it could be 1 mm or 5 mm lines. Decide by whether the
        # implied heart rate and QRS width are physiologically plausible (image-only evidence + physiological prior).
        ok1 = _scale_plausible(*_implied_hr_qrs(t_s, mv, valid))
        ok5 = _scale_plausible(*_implied_hr_qrs(t_s * 5.0, mv * 5.0, valid))
        if ok5 and not ok1:
            t_s, mv, x_mm, y_mm = t_s * 5.0, mv * 5.0, x_mm * 5.0, y_mm * 5.0
            cal = dict(cal, px_per_mm_x=cal["px_per_mm_x"] / 5.0, px_per_mm_y=cal["px_per_mm_y"] / 5.0)
            scale_res = "heart_rate_plausibility_5mm_lattice"
        elif ok1 and not ok5:
            scale_res = "heart_rate_plausibility_1mm_lattice"
        else:
            scale_res = "ambiguous"
    cal = dict(cal, scale_resolution=scale_res)
    res.status = "ok" if valid.sum() >= 20 else "unable"
    res.reason = "" if res.status == "ok" else "too few valid trace columns"
    res.x_px, res.y_px = xs.astype(np.float64), np.where(valid, y_est, np.nan)
    res.top_px, res.bot_px, res.valid = top, bot, valid
    res.t_s, res.mv, res.x_mm, res.y_mm = t_s, mv, x_mm, y_mm
    res.n_runs, res.run_mass, res.other_mass, res.height_excess_px = nrun, mass, other, excess
    cal_pub = {k: v for k, v in cal.items() if k not in ("maps_x", "maps_y", "grid_cols_px", "grid_rows_px")}
    res.calibration = cal_pub
    res.stats = {
        "trace_level": level,
        "noise_sigma": noise,
        "snr": level / max(noise, 1.0),
        "threshold": thr,
        "trace_thickness_px": th,
        "x_extent_px": [x0, x1],
        "n_columns": int(ncol),
        "n_components_kept": int(keep.sum()),
    }
    res.ink, res.mask = ink, mask_k
    return res


def _implied_hr_qrs(t_s: np.ndarray, mv: np.ndarray, valid: np.ndarray):
    """(heart rate in bpm, median QRS half-height width in s) implied by a candidate time scale, or (None, None)."""
    from . import beats as _beats

    us = _beats.uniformize(t_s, mv, valid)
    if us.x.size < int(1.2 * us.fs) or us.fs < 20:
        return None, None
    idx, pol = _beats.detect_r_peaks(us.x, us.fs)
    if idx.size < 2:
        return None, None
    hr = float(60.0 / np.median(np.diff(us.t[idx])))
    widths = []
    lim = int(0.25 * us.fs)
    for i in idx:
        a, b = int(round(i - 0.12 * us.fs)), int(round(i - 0.06 * us.fs))
        if a < 0 or b <= a or i + lim >= us.x.size or i - lim < 0:
            continue
        y = pol * (us.x - np.median(us.x[a:b]))
        half = y[i] / 2.0
        if half <= 0:
            continue
        l = i
        while l > i - lim and y[l] > half:
            l -= 1
        r = i
        while r < i + lim and y[r] > half:
            r += 1
        widths.append((r - l) / us.fs)
    return hr, (float(np.median(widths)) if widths else None)


def _scale_plausible(hr, width) -> bool:
    """Physiological plausibility of a candidate time scale: HR 30-240 bpm and QRS half-height width 20-160 ms."""
    if hr is None or width is None:
        return False
    return 30.0 <= hr <= 240.0 and 0.020 <= width <= 0.160


def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64).copy()
    bad = ~np.isfinite(a)
    if bad.all():
        return np.zeros_like(a)
    if bad.any():
        idx = np.arange(a.size)
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def _to_mm(x_px, y_px, valid, cal, ppm_x, ppm_y, h, w):
    """Map pixel coordinates to paper millimetres using the locally fitted grid (bilinear across strips)."""
    mxs = sorted(cal.get("maps_x") or [], key=lambda m: m["center"])
    mys = sorted(cal.get("maps_y") or [], key=lambda m: m["center"])
    y_safe = np.where(np.isfinite(y_px), y_px, np.nanmedian(y_px[valid]) if valid.any() else h / 2.0)
    if mxs:
        centers = np.array([m["center"] for m in mxs])
        mm_all = np.stack([_eval_map(m, x_px) for m in mxs], axis=0)  # strips x cols
        if len(mxs) == 1:
            x_mm = mm_all[0]
        else:
            x_mm = np.empty_like(x_px)
            for i in range(x_px.size):
                x_mm[i] = np.interp(y_safe[i], centers, mm_all[:, i])
    else:
        x_mm = x_px / ppm_x
    if mys:
        centers = np.array([m["center"] for m in mys])
        mm_all = np.stack([_eval_map(m, y_safe) for m in mys], axis=0)
        if len(mys) == 1:
            y_mm = mm_all[0]
        else:
            y_mm = np.empty_like(x_px)
            for i in range(x_px.size):
                y_mm[i] = np.interp(x_px[i], centers, mm_all[:, i])
    else:
        y_mm = y_safe / ppm_y
    return x_mm, y_mm
