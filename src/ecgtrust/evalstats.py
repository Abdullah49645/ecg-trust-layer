"""Selective-measurement statistics: risk-coverage, AURC, record-level bootstrap.

Convention: a *score* is higher = more trusted. A beat is *accepted* when its score is at or above a
threshold. ``err`` is the beat error (NRMSE or 1.0 for a false detection); ``bad`` is the binary bad-beat flag.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def _tie_groups(score_sorted: np.ndarray) -> np.ndarray:
    """Index of the last element of every tie group in a descending-sorted score array."""
    n = score_sorted.size
    if n == 0:
        return np.zeros(0, dtype=int)
    last = np.flatnonzero(np.diff(score_sorted) != 0)
    return np.concatenate([last, [n - 1]])


def risk_coverage(score: np.ndarray, err: np.ndarray, bad: np.ndarray) -> Dict[str, np.ndarray]:
    """Risk-coverage curve at every distinct threshold (ties are accepted together)."""
    score, err, bad = np.asarray(score, float), np.asarray(err, float), np.asarray(bad, float)
    order = np.argsort(-score, kind="stable")
    s, e, b = score[order], err[order], bad[order]
    ce, cb = np.cumsum(e), np.cumsum(b)
    g = _tie_groups(s)
    k = g + 1
    return {
        "threshold": s[g],
        "coverage": k / score.size,
        "n_accepted": k,
        "mean_err": ce[g] / k,
        "bad_rate": cb[g] / k,
    }


def aurc(score: np.ndarray, err: np.ndarray) -> float:
    """Area under the risk-coverage curve = average over ranks of the mean error of the top-k beats.

    Ties are handled by averaging over all orderings inside a tie group (expected value), so a score with
    few distinct values is not given an arbitrary advantage or disadvantage.
    """
    score, err = np.asarray(score, float), np.asarray(err, float)
    n = score.size
    if n == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, e = score[order], err[order]
    csum = np.concatenate([[0.0], np.cumsum(e)])
    g = _tie_groups(s)
    starts = np.concatenate([[0], g[:-1] + 1])
    total = 0.0
    for a, b in zip(starts, g):
        m = b - a + 1
        mean_in = (csum[b + 1] - csum[a]) / m
        # expected cumulative mean after taking j of m tied items: (csum[a] + j * mean_in) / (a + j)
        j = np.arange(1, m + 1)
        total += float(np.sum((csum[a] + j * mean_in) / (a + j)))
    return total / n


def at_coverage(score: np.ndarray, err: np.ndarray, bad: np.ndarray, coverage: float) -> Dict[str, float]:
    n = score.size
    k = max(1, int(round(coverage * n)))
    order = np.argsort(-np.asarray(score, float), kind="stable")[:k]
    return {"coverage": k / n, "mean_err": float(np.mean(err[order])), "bad_rate": float(np.mean(bad[order]))}


def bootstrap_records(rec_ids: np.ndarray, funcs: Dict[str, callable], n_boot: int = 1000, seed: int = 1234) -> Dict[str, np.ndarray]:
    """Resample RECORDS (all their beats across all conditions) with replacement and evaluate each function.

    ``funcs[name](idx)`` receives the concatenated beat indices of the resampled records and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(rec_ids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    out = {k: np.zeros(n_boot) for k in funcs}
    for bi in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([groups[p] for p in pick])
        for k, f in funcs.items():
            out[k][bi] = f(idx)
    return out


def ci95(x: np.ndarray) -> List[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def choose_thresholds(score: np.ndarray, bad: np.ndarray, accept_max_bad: float = 0.05, reject_min_bad: float = 0.5) -> Dict[str, float]:
    """Pre-declared rule (configs/analysis_plan.json): thresholds are chosen on the validation split."""
    score, bad = np.asarray(score, float), np.asarray(bad, float)
    order = np.argsort(-score, kind="stable")
    s, b = score[order], bad[order]
    g = _tie_groups(s)
    k = g + 1
    bad_rate_top = np.cumsum(b)[g] / k
    ok = np.flatnonzero(bad_rate_top <= accept_max_bad)
    t_accept = float(s[g][ok.max()]) if ok.size else float(s.max() + 1e-6)
    # rejected set = scores strictly below t; evaluate for each candidate t = s[g] (items after the group)
    n = s.size
    cum_bad_total = np.cumsum(b)
    rej_rate = []
    t_cands = []
    for gi in range(len(g) - 1):
        n_rej = n - (g[gi] + 1)
        bad_rej = cum_bad_total[-1] - cum_bad_total[g[gi]]
        rej_rate.append(bad_rej / n_rej)
        t_cands.append(float(s[g[gi]]))  # threshold: score >= this stays; strictly lower are rejected
    rej_rate = np.array(rej_rate)
    okr = np.flatnonzero(rej_rate >= reject_min_bad) if rej_rate.size else np.array([], dtype=int)
    t_reject = float(t_cands[okr.min()]) if okr.size else float(s.min() - 1e-6)
    t_reject = min(t_reject, t_accept)
    return {"t_accept": t_accept, "t_reject": t_reject}
