"""Alignment and waveform metrics (evaluation side only; uses the reference signal).

Definitions (all on valid extracted samples; ``x`` = extracted, ``r`` = reference, units mV):

* **Pearson r**  = cov(x, r) / (std(x) * std(r)).  Dimensionless. Insensitive to gain and offset,
  therefore never used alone.
* **RMSE**       = sqrt(mean((x - r)^2)).  mV.
* **NRMSE**      = RMSE / (P99(r_record) - P1(r_record)).  Dimensionless; the denominator is the robust
  peak-to-peak amplitude of the *whole reference record*, so errors are comparable across records.
* **Max |err|**  = max |x - r|.  mV.
* **Valid coverage** = valid samples / columns inside the trace x-extent.  Dimensionless.
* **Gain**       = OLS slope of x on r.  Dimensionless; 1.0 means correct amplitude calibration.

Alignment procedure (three stages, all reported):

1. ``raw``      : extracted millivolts (median-centred by the extractor) vs the reference at the same
                  nominal times. No offset or lag compensation.
2. ``dc``       : + remove the DC offset  (median of x - r). An image has no absolute zero.
3. ``dc_lag``   : + a single global time lag in [-100, +100] ms (0.5 ms steps) minimising RMSE after DC
                  removal. The image has no absolute time origin either (the stroke end-cap alone shifts the
                  first column by ~1 px = 5 ms). The reference is evaluated at ``t_extracted + lag``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def robust_range(ref: np.ndarray) -> float:
    r = float(np.percentile(ref, 99) - np.percentile(ref, 1))
    return r if r > 1e-6 else 1.0


def basic_metrics(x: np.ndarray, r: np.ndarray, amp: float) -> Dict[str, float]:
    """Metrics for already-paired, finite samples."""
    if x.size < 3:
        return {"pearson": float("nan"), "rmse_mv": float("nan"), "nrmse": float("nan"), "max_abs_err_mv": float("nan"), "gain": float("nan"), "n": int(x.size)}
    e = x - r
    rmse = float(np.sqrt(np.mean(e**2)))
    sx, sr = np.std(x), np.std(r)
    pear = float(np.corrcoef(x, r)[0, 1]) if sx > 1e-9 and sr > 1e-9 else float("nan")
    gain = float(np.polyfit(r, x, 1)[0]) if sr > 1e-9 else float("nan")
    return {"pearson": pear, "rmse_mv": rmse, "nrmse": rmse / amp, "max_abs_err_mv": float(np.max(np.abs(e))), "gain": gain, "n": int(x.size)}


def align_and_score(t_ext: np.ndarray, x_ext: np.ndarray, valid: np.ndarray, t_ref: np.ndarray, x_ref: np.ndarray,
                    max_lag_s: float = 0.1, lag_step_s: float = 0.0005) -> dict:
    """Align extracted waveform to the reference (evaluation only) and report metrics at each stage."""
    amp = robust_range(x_ref)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x_ext)
    out: dict = {"amp_ref_mv": amp, "valid_coverage": float(v.mean()) if v.size else 0.0}
    if v.sum() < 20:
        out.update(ok=False, lag_s=0.0, offset_mv=0.0)
        return out
    tv, xv = t_ext[v], x_ext[v]

    def ref_at(lag):
        return np.interp(tv + lag, t_ref, x_ref)

    r0 = ref_at(0.0)
    out["raw"] = basic_metrics(xv, r0, amp)
    off0 = float(np.median(xv - r0))
    out["dc"] = basic_metrics(xv - off0, r0, amp)
    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, lag_step_s)
    best, best_lag, best_off = np.inf, 0.0, off0
    for lag in lags:
        rr = ref_at(lag)
        off = float(np.median(xv - rr))
        e = float(np.mean((xv - off - rr) ** 2))
        if e < best:
            best, best_lag, best_off = e, float(lag), off
    out["lag_s"] = best_lag
    out["offset_mv"] = best_off
    out["dc_lag"] = basic_metrics(xv - best_off, ref_at(best_lag), amp)
    out["ok"] = True
    return out
