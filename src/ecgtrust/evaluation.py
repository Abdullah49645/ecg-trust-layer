"""Reproducible evaluation orchestrator (full and quick presets).

Stage 1  process every (record, condition) -> raw beat rows (cached in results/raw/*.json.gz)
Stage 2  fit the heuristic anchors and the learned model on TUNE; choose the operational score and the
         accept/reject thresholds on VAL (pre-declared rules in configs/analysis_plan.json); freeze them
Stage 3  evaluate on TEST (untouched until here): risk-coverage, AURC with record-level bootstrap, baselines,
         ablations, operational classes, measurement errors, per-condition breakdown
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import __version__, data, degrade, evalstats, extract, pipeline, render, trust

ROOT = Path(__file__).resolve().parents[2]
Progress = Callable[[str, int, int, str], None]


def _noop(stage: str, done: int, total: int, msg: str) -> None:
    pass


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# stage 1: processing
# ----------------------------------------------------------------------------------------------
def _work(args) -> List[dict]:
    data_root, row, split, conditions, render_d, ext_d = args
    sig = data.load_record(data_root, row["filename_hr"], "II", row)
    rb = render.RenderConfig.from_dict(render_d)
    ec = extract.ExtractionConfig(**{k: (tuple(v) if k == "minor_pitch_px_range" else v) for k, v in ext_d.items()})
    out = []
    for c in conditions:
        out.append(pipeline.process_record_condition(sig, c, split, rb, ec))
    return out


def process_split(split: str, rows: List[dict], data_root, conditions: Sequence[str], workers: int = 2,
                  progress: Progress = _noop, cache: Optional[Path] = None, use_cache: bool = True) -> List[dict]:
    if cache is not None and use_cache and Path(cache).exists():
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("conditions") == list(conditions) and len(cached["rows"]) == len(rows) * len(conditions):
            progress(f"{split} (cached)", len(rows), len(rows), "loaded cached raw results")
            return cached["rows"]
    rb, ec = render.RenderConfig().to_dict(), extract.ExtractionConfig().to_dict()
    jobs = [(str(data_root), r, split, list(conditions), rb, ec) for r in rows]
    results: List[dict] = []
    done = 0
    progress(split, 0, len(jobs), f"{len(jobs)} records x {len(conditions)} conditions")
    if workers <= 1:
        for j in jobs:
            results.extend(_work(j))
            done += 1
            progress(split, done, len(jobs), f"record {done}/{len(jobs)}")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            for out in ex.map(_work, jobs):
                results.extend(out)
                done += 1
                progress(split, done, len(jobs), f"record {done}/{len(jobs)}")
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt", encoding="utf-8") as fh:
            json.dump({"conditions": list(conditions), "rows": results}, fh)
    return results


# ----------------------------------------------------------------------------------------------
# beat tables
# ----------------------------------------------------------------------------------------------
class BeatTable:
    """Flat arrays over all beats of a set of image rows."""

    def __init__(self, rows: List[dict]):
        self.rows = rows
        recs, conds, fams, imgs, prevs = [], [], [], [], []
        X, err, bad, matched, core, rr_err, ramp_err, rr_elig = [], [], [], [], [], [], [], []
        for ii, r in enumerate(rows):
            base = len(recs)
            for k, b in enumerate(r.get("beats", [])):
                recs.append(r["source"]["record_id"])
                conds.append(r["condition"])
                fams.append(r["condition_family"])
                imgs.append(ii)
                prevs.append(base + k - 1 if k > 0 else -1)
                X.append([b[n] for n in trust.FEATURES])
                err.append(b["err"])
                bad.append(1.0 if b["bad"] else 0.0)
                matched.append(bool(b["matched"]))
                core.append(b["qrs_core_observed"] > 0.5)
                rr_elig.append(bool(b.get("rr_eligible", False)))
                rr_err.append(b["rr_err_ms"] if b.get("rr_err_ms") is not None else np.nan)
                ramp_err.append(b["r_amp_err_mv"] if b.get("r_amp_err_mv") is not None else np.nan)
        self.rec = np.array(recs)
        self.cond = np.array(conds)
        self.family = np.array(fams)
        self.img = np.array(imgs, dtype=int)
        self.prev = np.array(prevs, dtype=int)
        self.X = np.array(X, dtype=float).reshape(-1, len(trust.FEATURES))
        self.err = np.array(err, dtype=float)
        self.bad = np.array(bad, dtype=float)
        self.matched = np.array(matched, dtype=bool)
        self.core = np.array(core, dtype=bool)
        self.rr_err = np.array(rr_err, dtype=float)
        self.rr_elig = np.array(rr_elig, dtype=bool)
        self.ramp_err = np.array(ramp_err, dtype=float)

    @property
    def n(self) -> int:
        return int(self.err.size)

    def col(self, name: str) -> np.ndarray:
        return self.X[:, trust.FEATURES.index(name)]


# ----------------------------------------------------------------------------------------------
# stage 2: model fitting
# ----------------------------------------------------------------------------------------------
def fit_model(tune: BeatTable, val: BeatTable, plan: dict) -> dict:
    clean = tune.cond == "clean"
    anchors = trust.fit_anchors(tune.X[clean], tune.X)
    learned = trust.fit_learned(tune.X, tune.bad)

    def heur_scores(t: BeatTable) -> np.ndarray:
        return np.array([trust.heuristic_trust(dict(zip(trust.FEATURES, x)), anchors) for x in t.X])

    sc_val = {"heuristic": heur_scores(val), "learned": trust.learned_trust(learned, val.X)}
    aurc_val = {k: evalstats.aurc(v, val.err) for k, v in sc_val.items()}
    operational = min(aurc_val, key=aurc_val.get)
    thr_pre = evalstats.choose_thresholds(sc_val[operational], val.bad, 0.05, 0.5)  # pre-registered rule (vacuous here, kept and reported)
    add = load_json(ROOT / "configs" / "analysis_addendum.json")["posthoc_operating_point"]
    thr = evalstats.choose_thresholds(sc_val[operational], val.bad, add["accept_max_bad"], add["reject_min_bad"])  # POST-HOC, see addendum
    return {
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "package_version": __version__,
        "anchors": anchors,
        "learned": learned.to_dict(),
        "operational_score": operational,
        "operational_choice": {"rule": plan["trust_scores"]["operational_choice_rule"], "val_aurc": aurc_val},
        "thresholds": {**thr, "rule": f"POST-HOC (configs/analysis_addendum.json): lowest val threshold with accepted bad-beat rate <= {add['accept_max_bad']:.0%}; reject: highest val threshold with rejected bad-beat rate >= {add['reject_min_bad']:.0%}"},
        "thresholds_prereg": {**thr_pre, "rule": plan["trust_scores"]["threshold_rule"]},
        "fit_on": {"tune_beats": tune.n, "val_beats": val.n, "tune_bad_rate": float(tune.bad.mean()), "val_bad_rate": float(val.bad.mean())},
        "feature_doc": trust.FEATURE_DOC,
        "groups": trust.GROUPS,
    }


def model_scores(t: BeatTable, model: dict) -> Dict[str, np.ndarray]:
    anchors = model["anchors"]
    learned = trust.LearnedModel.from_dict(model["learned"])
    feats = [dict(zip(trust.FEATURES, x)) for x in t.X]
    out = {
        "heuristic": np.array([trust.heuristic_trust(f, anchors) for f in feats]),
        "learned": trust.learned_trust(learned, t.X),
    }
    out["operational"] = out[model["operational_score"]]
    return out


# ----------------------------------------------------------------------------------------------
# stage 3: statistics
# ----------------------------------------------------------------------------------------------
def _downsample_curve(c: Dict[str, np.ndarray], n: int = 160) -> Dict[str, list]:
    m = c["coverage"].size
    idx = np.unique(np.linspace(0, m - 1, min(n, m)).astype(int))
    return {k: np.asarray(v)[idx].round(5).tolist() for k, v in c.items() if k in ("coverage", "mean_err", "bad_rate", "threshold")}


def _class_summary(label: np.ndarray, t: BeatTable) -> Dict[str, dict]:
    out = {}
    for lab in trust.LABELS[:3]:
        m = label == lab
        out[lab] = {
            "n": int(m.sum()), "fraction": float(m.mean()) if label.size else float("nan"),
            "mean_err": float(t.err[m].mean()) if m.any() else None,
            "bad_rate": float(t.bad[m].mean()) if m.any() else None,
            "median_nrmse_matched": float(np.median(t.err[m & t.matched])) if (m & t.matched).any() else None,
        }
    out["all"] = {"n": int(t.n), "fraction": 1.0, "mean_err": float(t.err.mean()), "bad_rate": float(t.bad.mean()),
                  "median_nrmse_matched": float(np.median(t.err[t.matched]))}
    return out


def evaluate_beats(t: BeatTable, rows: List[dict], model: dict, n_boot: int = 1000, with_ablations: bool = True) -> dict:
    """Statistics for a beat table using a frozen model."""
    sc = model_scores(t, model)
    thr = model["thresholds"]
    methods: Dict[str, np.ndarray] = {
        "trust_operational": sc["operational"],
        "trust_heuristic": sc["heuristic"],
        "trust_learned": sc["learned"],
        "accept_all": np.zeros(t.n),
        "image_quality_snr": t.col("f_snr"),
        "continuity": -t.col("f_roughness"),
        "valid_fraction": 1.0 - t.col("f_missing"),
        "oracle_error_ranking": -t.err,
    }
    anchors = model["anchors"]
    if with_ablations:
        feats = [dict(zip(trust.FEATURES, x)) for x in t.X]
        for g in trust.GROUPS:
            methods[f"ablation_only_{g}"] = np.array([trust.heuristic_trust(f, anchors, [g]) for f in feats])
            methods[f"ablation_drop_{g}"] = np.array([trust.heuristic_trust(f, anchors, [h for h in trust.GROUPS if h != g]) for f in feats])
    res: dict = {"n_beats": t.n, "n_records": int(np.unique(t.rec).size), "n_images": int(len(rows)),
                 "overall": {"mean_err_all": float(t.err.mean()), "bad_rate_all": float(t.bad.mean()),
                             "median_nrmse_matched": float(np.median(t.err[t.matched])) if t.matched.any() else None,
                             "unmatched_fraction": float(1 - t.matched.mean())}}
    # ---- point estimates
    mres = {}
    for name, s in methods.items():
        entry = {"aurc": evalstats.aurc(s, t.err)}
        entry["at_coverage"] = {f"{c:.1f}": evalstats.at_coverage(s, t.err, t.bad, c) for c in evalstats.COVERAGE_LEVELS}
        mres[name] = entry
    res["methods"] = mres
    res["curves"] = {k: _downsample_curve(evalstats.risk_coverage(methods[k], t.err, t.bad))
                     for k in ("trust_operational", "trust_heuristic", "trust_learned", "image_quality_snr", "continuity", "valid_fraction", "oracle_error_ranking")}
    res["curves"]["accept_all"] = {"coverage": [0.0, 1.0], "mean_err": [res["overall"]["mean_err_all"]] * 2, "bad_rate": [res["overall"]["bad_rate_all"]] * 2}

    # ---- paired record-level bootstrap on AURC
    baselines = ["image_quality_snr", "continuity", "valid_fraction"]
    best_base = min(baselines, key=lambda k: mres[k]["aurc"])
    funcs = {"accept_all": lambda i: float(t.err[i].mean()),
             "trust_operational": lambda i: evalstats.aurc(methods["trust_operational"][i], t.err[i]),
             best_base: lambda i: evalstats.aurc(methods[best_base][i], t.err[i])}
    boot = evalstats.bootstrap_records(t.rec, funcs, n_boot=n_boot)
    d1 = boot["accept_all"] - boot["trust_operational"]
    d2 = boot[best_base] - boot["trust_operational"]
    res["hypotheses"] = {
        "primary_metric": "AURC (mean beat error averaged over coverage; lower is better)",
        "aurc_trust_operational": {"point": mres["trust_operational"]["aurc"], "ci95": evalstats.ci95(boot["trust_operational"])},
        "aurc_accept_all": {"point": mres["accept_all"]["aurc"], "ci95": evalstats.ci95(boot["accept_all"])},
        "best_simple_baseline": best_base,
        "aurc_best_baseline": {"point": mres[best_base]["aurc"], "ci95": evalstats.ci95(boot[best_base])},
        "H1_trust_beats_accept_all": {"delta_point": mres["accept_all"]["aurc"] - mres["trust_operational"]["aurc"], "delta_ci95": evalstats.ci95(d1),
                                       "relative_reduction_point": 1 - mres["trust_operational"]["aurc"] / mres["accept_all"]["aurc"],
                                       "supported": bool(np.percentile(d1, 2.5) > 0)},
        "H2_trust_beats_best_baseline": {"delta_point": mres[best_base]["aurc"] - mres["trust_operational"]["aurc"], "delta_ci95": evalstats.ci95(d2),
                                          "supported": bool(np.percentile(d2, 2.5) > 0)},
        "n_bootstrap": n_boot, "bootstrap_unit": "record (all conditions of a record resampled together)",
    }
    # ---- operational classes
    label = np.array([trust.classify(s, c, thr["t_accept"], thr["t_reject"]) for s, c in zip(sc["operational"], t.core)])
    res["operational"] = {"thresholds": {"t_accept": thr["t_accept"], "t_reject": thr["t_reject"]}, "score": model["operational_score"],
                          "kind": "post-hoc operating point (configs/analysis_addendum.json), thresholds chosen on val", "classes": _class_summary(label, t)}
    if "thresholds_prereg" in model:
        tp = model["thresholds_prereg"]
        lab_pre = np.array([trust.classify(s, c, tp["t_accept"], tp["t_reject"]) for s, c in zip(sc["operational"], t.core)])
        res["operational_prereg"] = {"thresholds": {"t_accept": tp["t_accept"], "t_reject": tp["t_reject"]}, "score": model["operational_score"],
                                     "kind": "pre-registered rule (accepted bad-rate <= 5%); vacuous because the base bad-beat rate is below 5%", "classes": _class_summary(lab_pre, t)}
    # ---- measurements among accepted beats
    acc = label == "accepted"
    prev_acc = np.array([(p >= 0 and acc[p]) for p in t.prev])
    rr_ok = np.isfinite(t.rr_err)
    ra_ok = np.isfinite(t.ramp_err)

    def stats(x):
        return {"n": int(x.size), "mean_abs": float(np.mean(np.abs(x))) if x.size else None, "median_abs": float(np.median(np.abs(x))) if x.size else None,
                "p90_abs": float(np.percentile(np.abs(x), 90)) if x.size else None}

    res["measurements"] = {
        "r_amplitude_error_mv": {"all_matched_beats": stats(t.ramp_err[ra_ok]), "accepted_beats": stats(t.ramp_err[ra_ok & acc]),
                                  "n_reported_fraction": float(acc.mean())},
        "rr_interval_error_ms": {"all_matched_pairs": stats(t.rr_err[rr_ok]), "accepted_pairs": stats(t.rr_err[rr_ok & acc & prev_acc & t.rr_elig])},
    }
    hr_all, hr_acc = [], []
    for ii, r in enumerate(rows):
        if not r.get("beats") or r.get("hr_ref_bpm") is None:
            continue
        m = t.img == ii
        idx = np.flatnonzero(m)
        rr = [r["beats"][k]["rr_ms_ext"] for k, gi in enumerate(idx) if acc[gi] and prev_acc[gi] and r["beats"][k].get("rr_eligible") and r["beats"][k]["rr_ms_ext"] is not None]
        if r.get("hr_abs_err_bpm") is not None:
            hr_all.append(r["hr_abs_err_bpm"])
        if rr:
            hr_acc.append(abs(60000.0 / np.mean(rr) - r["hr_ref_bpm"]))
    res["measurements"]["heart_rate_abs_error_bpm"] = {
        "all_images": {"n": len(hr_all), "mean": float(np.mean(hr_all)) if hr_all else None, "median": float(np.median(hr_all)) if hr_all else None,
                       "p90": float(np.percentile(hr_all, 90)) if hr_all else None},
        "accepted_pairs_only": {"n": len(hr_acc), "mean": float(np.mean(hr_acc)) if hr_acc else None, "median": float(np.median(hr_acc)) if hr_acc else None,
                                "p90": float(np.percentile(hr_acc, 90)) if hr_acc else None},
        "images_without_a_reportable_hr": int(len(hr_all) - len(hr_acc)),
    }
    # ---- per condition
    by = {}
    for c in degrade.CONDITIONS:
        rimg = [r for r in rows if r["condition"] == c]
        if not rimg:
            continue
        m = t.cond == c
        n_unable = int(sum(1 for r in rimg if r["status"] != "ok" or not r.get("beats")))
        entry = {"n_images": len(rimg), "n_unable": n_unable, "n_beats": int(m.sum())}
        if m.any():
            entry.update(mean_err_all=float(t.err[m].mean()), bad_rate_all=float(t.bad[m].mean()),
                         accepted_fraction=float((acc & m).sum() / m.sum()),
                         accepted_bad_rate=float(t.bad[acc & m].mean()) if (acc & m).any() else None,
                         accepted_mean_err=float(t.err[acc & m].mean()) if (acc & m).any() else None,
                         rejected_fraction=float(((label == "rejected") & m).sum() / m.sum()),
                         mean_trust=float(sc["operational"][m].mean()))
        by[c] = entry
    res["by_condition"] = by
    res["status_counts"] = {"images_total": len(rows), "images_unable": int(sum(1 for r in rows if r["status"] != "ok" or not r.get("beats")))}
    return res


# ----------------------------------------------------------------------------------------------
# presets / runners
# ----------------------------------------------------------------------------------------------
def environment_info() -> dict:
    import cv2
    import scipy
    import sklearn

    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "opencv": cv2.__version__, "scikit_learn": sklearn.__version__}


def run_full(workers: int = 2, progress: Progress = _noop, use_cache: bool = True, data_root=None, n_boot: int = 1000,
             out_dir=None, limit: Optional[Dict[str, int]] = None) -> dict:
    t0 = time.time()
    data_root = Path(data_root or ROOT / "data" / "ptbxl")
    out_dir = Path(out_dir or ROOT / "results")
    plan = load_json(ROOT / "configs" / "analysis_plan.json")
    sel = data.load_selection(ROOT / "configs" / "record_selection.json")
    conds = list(degrade.CONDITIONS)
    raw = {}
    for split in ("tune", "val", "test"):
        rows = sel[split][: (limit or {}).get(split, None)] if limit else sel[split]
        raw[split] = process_split(split, rows, data_root, conds, workers, progress, out_dir / "raw" / f"{split}.json.gz", use_cache and not limit)
    progress("fit", 0, 1, "fitting trust model on tune, choosing operational score/thresholds on val")
    tune, val, test = BeatTable(raw["tune"]), BeatTable(raw["val"]), BeatTable(raw["test"])
    model = fit_model(tune, val, plan)
    json.dump(model, open(out_dir / "trust_model.json", "w"), indent=2)
    progress("evaluate", 0, 1, "evaluating on the untouched test split (bootstrap)")
    test_res = evaluate_beats(test, raw["test"], model, n_boot=n_boot)
    val_res = evaluate_beats(val, raw["val"], model, n_boot=200, with_ablations=False)
    tune_res = {"n_beats": tune.n, "mean_err_all": float(tune.err.mean()), "bad_rate_all": float(tune.bad.mean())}
    gate = load_json(out_dir / "feasibility_gate.json") if (out_dir / "feasibility_gate.json").exists() else None
    result = {
        "preset": "full",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": time.time() - t0,
        "environment": environment_info(),
        "dataset": {"id": data.DATASET_ID, "lead": "II", "fs_hz": 500, "units": "mV", "license": "CC BY 4.0",
                    "images": "SYNTHETIC renderings of digital ECG signals - not photographs of real paper ECGs",
                    "split_sizes_records": {k: len(v) for k, v in sel.items()} if not limit else {k: min(len(v), limit.get(k, 10**9)) for k, v in sel.items()},
                    "n_conditions": len(conds), "conditions": conds},
        "config_hashes": {"render": sha256_of(render.RenderConfig().to_dict()), "extract": sha256_of(extract.ExtractionConfig().to_dict()),
                          "analysis_plan": sha256_of(plan), "conditions": sha256_of(degrade.CONDITIONS)},
        "protocol": {"analysis_plan": plan},
        "feasibility_gate": {"passed": gate["passed"], "summary": gate["summary"]} if gate else None,
        "trust_model": {k: model[k] for k in ("operational_score", "operational_choice", "thresholds", "thresholds_prereg", "fit_on")},
        "tune": tune_res,
        "val": {"hypotheses": val_res["hypotheses"], "overall": val_res["overall"]},
        "test": test_res,
    }
    json.dump(result, open(out_dir / ("results_full.json" if not limit else "results_full_limited.json"), "w"), indent=2)
    progress("done", 1, 1, "finished")
    return result


def run_quick(workers: int = 2, progress: Progress = _noop, out_dir=None, n_boot: int = 300) -> dict:
    """Quick preset: frozen trust model, bundled held-out records, a subset of conditions. Same code path as the full run."""
    t0 = time.time()
    out_dir = Path(out_dir or ROOT / "results")
    q = load_json(ROOT / "configs" / "quick_preset.json")
    model = load_json(out_dir / "trust_model.json")
    sel = data.load_selection(ROOT / "configs" / "quick_selection.json")["test"]
    rows = process_split("quick-test", sel, ROOT / "data" / "quick", q["conditions"], workers, progress, None, False)
    t = BeatTable(rows)
    res = evaluate_beats(t, rows, model, n_boot=n_boot, with_ablations=False)
    result = {
        "preset": "quick",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": time.time() - t0,
        "environment": environment_info(),
        "dataset": {"id": data.DATASET_ID, "lead": "II", "images": "SYNTHETIC renderings of digital ECG signals",
                    "n_records": len(sel), "conditions": q["conditions"], "split": "test (held-out PTB-XL fold 10), frozen trust model from the full run"},
        "trust_model": {k: model[k] for k in ("operational_score", "thresholds", "thresholds_prereg") if k in model},
        "test": res,
    }
    json.dump(result, open(out_dir / "results_quick.json", "w"), indent=2)
    progress("done", 1, 1, "finished")
    return result
