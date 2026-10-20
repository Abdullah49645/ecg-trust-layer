"""Beat detection and the (deliberately small) set of beat-level measurements.

Detector: a simplified Pan-Tompkins pipeline (band-pass 5-15 Hz -> derivative -> squaring -> 120 ms
moving integration -> adaptive threshold with a 250 ms refractory period), followed by refinement of
each fiducial to the local extremum of the signal. Polarity (upright vs inverted QRS) is estimated
from the detected candidates. The same detector runs on the extracted signal and on the reference.

Measurements (the only ones this project reports):

* **RR interval (ms)** between consecutive detected R peaks, and instantaneous heart rate = 60000 / RR.
* **R amplitude (mV)**: signed peak value minus the median of the PR-region baseline window
  ``[R-120 ms, R-60 ms]`` (polarity-corrected).

No clinical interpretation (normal / abnormal) is made anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

BEAT_PRE_S = 0.20
BEAT_POST_S = 0.40


@dataclass
class UniformSignal:
    t: np.ndarray
    x: np.ndarray  # linearly bridged where needed for detection
    observed: np.ndarray  # True where the sample is backed by a valid image column
    fs: float


def uniformize(t: np.ndarray, x: np.ndarray, valid: np.ndarray, fs: Optional[float] = None) -> UniformSignal:
    """Resample a (possibly non-uniform) per-column signal to a uniform grid.

    Invalid columns are *bridged by linear interpolation only for beat detection*; ``observed`` records
    which output samples are close (<= 1.5 column spacings) to a valid column so that bridged regions are
    never mistaken for verified measurements.
    """
    v = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if v.sum() < 4:
        return UniformSignal(np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), fs or 1.0)
    tv, xv = t[v], x[v]
    dt = float(np.median(np.diff(tv)))
    fs = fs or (1.0 / dt if dt > 0 else 100.0)
    tu = np.arange(tv[0], tv[-1], 1.0 / fs)
    xu = np.interp(tu, tv, xv)
    idx = np.searchsorted(tv, tu)
    idx = np.clip(idx, 1, tv.size - 1)
    near = np.minimum(np.abs(tu - tv[idx - 1]), np.abs(tu - tv[idx]))
    observed = near <= 1.5 * max(dt, 1.0 / fs)
    return UniformSignal(tu, xu, observed, fs)


def _bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    hi = min(15.0, 0.4 * fs)
    lo = 5.0
    if hi <= lo + 1:
        return x - np.median(x)
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    if x.size < 3 * max(len(a), len(b)) + 1:
        return x - np.median(x)
    return filtfilt(b, a, x)


def detect_r_peaks(x: np.ndarray, fs: float) -> Tuple[np.ndarray, int]:
    """Return ``(sample indices of R peaks, polarity)`` with polarity = +1 (upright) or -1 (inverted)."""
    if x.size < int(1.0 * fs):
        return np.zeros(0, dtype=int), 1
    f = _bandpass(x - np.median(x), fs)
    d = np.gradient(f)
    sq = d**2
    w = max(1, int(round(0.12 * fs)))
    integ = np.convolve(sq, np.ones(w) / w, mode="same")
    ref_level = np.percentile(integ, 98)
    if ref_level <= 0:
        return np.zeros(0, dtype=int), 1
    pk, _ = find_peaks(integ, height=0.25 * ref_level, distance=max(1, int(0.25 * fs)))
    if pk.size == 0:
        return np.zeros(0, dtype=int), 1
    # refine to the extremum of the (lightly smoothed) signal around each candidate
    half = max(1, int(round(0.06 * fs)))
    base = np.median(x)
    pos_votes = neg_votes = 0
    cands = []
    for p in pk:
        lo, hi = max(0, p - half), min(x.size, p + half + 1)
        seg = x[lo:hi] - base
        i_max, i_min = int(np.argmax(seg)), int(np.argmin(seg))
        cands.append((lo, seg, i_max, i_min))
        if abs(seg[i_max]) >= abs(seg[i_min]):
            pos_votes += 1
        else:
            neg_votes += 1
    pol = 1 if pos_votes >= neg_votes else -1
    out = []
    for lo, seg, i_max, i_min in cands:
        out.append(lo + (i_max if pol > 0 else i_min))
    out = np.unique(np.array(out, dtype=int))
    return out, pol


def r_amplitude(x: np.ndarray, fs: float, r_idx: int, polarity: int) -> float:
    """R amplitude above the pre-QRS baseline window [R-120 ms, R-60 ms] (mV, polarity-corrected)."""
    a, b = int(round(r_idx - 0.12 * fs)), int(round(r_idx - 0.06 * fs))
    if a < 0 or b <= a or r_idx >= x.size:
        return float("nan")
    base = float(np.median(x[a:b]))
    return float(polarity * (x[r_idx] - base))


def beat_window(t: np.ndarray, t_r: float) -> Tuple[int, int]:
    """Column index range ``[i0, i1)`` of the beat window around an R time on a (non-uniform) time axis."""
    i0 = int(np.searchsorted(t, t_r - BEAT_PRE_S, side="left"))
    i1 = int(np.searchsorted(t, t_r + BEAT_POST_S, side="right"))
    return i0, i1


def match_beats(t_ext_r: np.ndarray, t_ref_r: np.ndarray, tol_s: float = 0.1) -> List[Optional[int]]:
    """Greedy nearest-neighbour one-to-one matching of extracted R times to reference R times."""
    used = set()
    out: List[Optional[int]] = []
    for t in t_ext_r:
        if t_ref_r.size == 0:
            out.append(None)
            continue
        d = np.abs(t_ref_r - t)
        order = np.argsort(d)
        pick = None
        for j in order:
            if d[j] > tol_s:
                break
            if int(j) not in used:
                pick = int(j)
                used.add(pick)
                break
        out.append(pick)
    return out
