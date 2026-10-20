"""Tiny analytic ECG generator used ONLY as a unit-test fixture (no dataset needed).

It is never used in the scientific evaluation, which uses real PTB-XL signals.
"""
from __future__ import annotations

import numpy as np

from .data import LoadedSignal


def synthetic_ecg(duration_s: float = 10.0, fs: float = 500.0, hr_bpm: float = 72.0, seed: int = 0, noise_mv: float = 0.004) -> np.ndarray:
    """Sum-of-Gaussians P-QRS-T beats (millivolts). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr = 60.0 / hr_bpm
    beat_t = 0.4
    comps = [(-0.2, 0.12, 0.025), (-0.035, -0.18, 0.012), (0.0, 1.1, 0.010), (0.03, -0.25, 0.012), (0.25, 0.30, 0.045)]
    while beat_t < duration_s - 0.3:
        for dt, amp, w in comps:
            x += amp * np.exp(-0.5 * ((t - (beat_t + dt)) / w) ** 2)
        beat_t += rr
    return x + rng.normal(0, noise_mv, n)


def synthetic_signal(record_id: str = "synthetic_00", **kw) -> LoadedSignal:
    fs = kw.pop("fs", 500.0)
    sig = synthetic_ecg(fs=fs, **kw)
    return LoadedSignal(signal_mv=sig, fs=fs, lead="II", units="mV", record_id=record_id, dataset="SYNTHETIC TEST FIXTURE (not PTB-XL)")
