"""Beat-level analysis of an extraction (reference-free) and its offline evaluation (uses the reference).

``find_beats``      -- reference-free: candidate beats, R amplitude, RR intervals, beat windows.
``evaluate_record`` -- evaluation only: alignment, waveform metrics, beat matching, per-beat errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import beats as B
from . import metrics as M
from .extract import ExtractionResult

RR_MIN_GAP_VALID = 0.99  # an RR interval is reportable only if the stretch between the two R peaks was observed


@dataclass
class BeatSet:
    t_r: np.ndarray  # R time in the extracted time frame (s)
    amp_mv: np.ndarray  # R amplitude above pre-QRS baseline (mV)
    rr_prev_ms: np.ndarray  # RR to previous detected beat (ms), NaN for the first beat
    win: np.ndarray  # (n, 2) column index range of the beat window
    polarity: int
    fs_u: float
    # fraction of observed (valid) columns between the previous R peak and this one; an RR interval is only
    # trustworthy if nothing in between was hidden (a missed beat inside a gap would silently double the RR)
    rr_gap_valid: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return int(self.t_r.size)


def find_beats(ext: ExtractionResult) -> BeatSet:
    """Detect candidate beats on the extracted signal. Never touches a reference."""
    empty = BeatSet(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 2), dtype=int), 1, 0.0)
    if not ext.ok:
        return empty
    us = B.uniformize(ext.t_s, ext.mv, ext.valid)
    if us.x.size < int(1.5 * us.fs):
        return empty
    idx, pol = B.detect_r_peaks(us.x, us.fs)
    if idx.size == 0:
        return BeatSet(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 2), dtype=int), pol, us.fs)
    t_r = us.t[idx]
    amps = np.array([B.r_amplitude(us.x, us.fs, int(i), pol) for i in idx])
    rr = np.full(idx.size, np.nan)
    rr[1:] = np.diff(t_r) * 1000.0
    win = np.array([B.beat_window(ext.t_s, tr) for tr in t_r], dtype=int)
    gap = np.full(idx.size, np.nan)
    for i in range(1, idx.size):
        seg = (ext.t_s >= t_r[i - 1]) & (ext.t_s <= t_r[i])
        gap[i] = float(ext.valid[seg].mean()) if seg.any() else 0.0
    return BeatSet(t_r, amps, rr, win, pol, us.fs, gap)


def reference_beats(t_ref: np.ndarray, x_ref: np.ndarray, fs: float):
    idx, pol = B.detect_r_peaks(x_ref, fs)
    amps = np.array([B.r_amplitude(x_ref, fs, int(i), pol) for i in idx]) if idx.size else np.zeros(0)
    return t_ref[idx] if idx.size else np.zeros(0), amps, pol


def evaluate_record(ext: ExtractionResult, beatset: BeatSet, x_ref: np.ndarray, fs_ref: float) -> dict:
    """Offline evaluation of one extraction against its known reference signal (never used by the trust score)."""
    t_ref = np.arange(len(x_ref)) / fs_ref
    out: dict = {"status": ext.status}
    if not ext.ok:
        out.update(ok=False, beats=[], reason=ext.reason)
        return out
    al = M.align_and_score(ext.t_s, ext.mv, ext.valid, t_ref, x_ref)
    out["alignment"] = al
    out["valid_coverage"] = ext.coverage
    if not al.get("ok"):
        out.update(ok=False, beats=[])
        return out
    lag, off, amp = al["lag_s"], al["offset_mv"], al["amp_ref_mv"]
    r_t, r_amp, r_pol = reference_beats(t_ref, x_ref, fs_ref)
    # extracted R times expressed in the reference time frame
    match = B.match_beats(beatset.t_r + lag, r_t, tol_s=0.1) if beatset.n else []
    rows: List[dict] = []
    ref_at_cols = np.interp(ext.t_s + lag, t_ref, x_ref)
    err_cols = ext.mv - off - ref_at_cols
    for i in range(beatset.n):
        i0, i1 = beatset.win[i]
        v = ext.valid[i0:i1] & np.isfinite(ext.mv[i0:i1])
        e = err_cols[i0:i1][v]
        row = {
            "beat": i,
            "t_r_s": float(beatset.t_r[i]),
            "matched": match[i] is not None,
            "window_valid_fraction": float(v.mean()) if v.size else 0.0,
            "r_amp_mv_ext": float(beatset.amp_mv[i]),
            "rr_ms_ext": float(beatset.rr_prev_ms[i]),
            "rr_eligible": bool(i > 0 and beatset.rr_gap_valid.size > i and beatset.rr_gap_valid[i] >= RR_MIN_GAP_VALID),
        }
        if match[i] is not None and e.size >= 5:
            j = match[i]
            row.update(
                ref_t_r_s=float(r_t[j]),
                rmse_mv=float(np.sqrt(np.mean(e**2))),
                nrmse=float(np.sqrt(np.mean(e**2)) / amp),
                max_abs_err_mv=float(np.max(np.abs(e))),
                r_amp_mv_ref=float(r_amp[j]),
                r_amp_err_mv=float(beatset.amp_mv[i] - r_amp[j]),
                r_time_err_ms=float((beatset.t_r[i] + lag - r_t[j]) * 1000.0),
            )
        else:
            row.update(nrmse=float("nan"), rmse_mv=float("nan"), r_amp_err_mv=float("nan"), r_time_err_ms=float("nan"))
        rows.append(row)
    # RR error: the extracted RR is the interval to the previous *detected* beat; the reference RR is the interval
    # to the true preceding beat. If a beat was missed in between, the extracted RR is ~2x and the error is large
    # (it must never be scored against the interval between the same two detections, which would hide the miss).
    for i in range(1, beatset.n):
        j = match[i]
        if j is not None and j >= 1:
            rr_ref = (r_t[j] - r_t[j - 1]) * 1000.0
            rows[i]["rr_ms_ref"] = float(rr_ref)
            rows[i]["rr_err_ms"] = float(rows[i]["rr_ms_ext"] - rr_ref)
    hr_ext = float(60000.0 / np.mean(beatset.rr_prev_ms[1:])) if beatset.n > 2 else float("nan")
    hr_ref = float(60.0 / np.mean(np.diff(r_t))) if r_t.size > 2 else float("nan")
    out.update(
        ok=True,
        beats=rows,
        n_ref_beats=int(r_t.size),
        n_ext_beats=int(beatset.n),
        n_matched=int(sum(m is not None for m in match)),
        ref_polarity=int(r_pol),
        hr_ext_bpm=hr_ext,
        hr_ref_bpm=hr_ref,
        hr_abs_err_bpm=float(abs(hr_ext - hr_ref)) if np.isfinite(hr_ext) and np.isfinite(hr_ref) else float("nan"),
        ref_beat_recall=float(sum(m is not None for m in match) / r_t.size) if r_t.size else float("nan"),
    )
    return out
