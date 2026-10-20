#!/usr/bin/env python
"""Feasibility gate: can the extractor recover clean synthetic renderings? (criteria in configs/feasibility_gate.json)

Runs on the *tune* split only (PTB-XL folds 1-8), clean condition, lead II.  Writes results/feasibility_gate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgtrust import analysis, data, degrade, extract, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "ptbxl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "feasibility_gate.json"))
    args = ap.parse_args()

    gate = json.load(open(ROOT / "configs" / "feasibility_gate.json"))
    crit = gate["criteria"]
    sel = data.load_selection(ROOT / "configs" / "record_selection.json")[gate["split"]]
    if args.limit:
        sel = sel[: args.limit]
    rows = []
    t0 = time.time()
    for row in sel:
        sig = data.load_record(args.data_dir, row["filename_hr"], "II", row)
        cfg = degrade.condition_render_config(render.RenderConfig(), gate["condition"])
        r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
        ext = extract.extract_trace(r.image)
        bs = analysis.find_beats(ext)
        ev = analysis.evaluate_record(ext, bs, sig.signal_mv, sig.fs)
        if not ev.get("ok"):
            rows.append({"record": sig.record_id, "ok": False, "reason": ext.reason})
            continue
        al = ev["alignment"]
        amp_errs = [abs(b["r_amp_err_mv"]) for b in ev["beats"] if np.isfinite(b.get("r_amp_err_mv", np.nan))]
        rows.append({
            "record": sig.record_id, "ok": True,
            "nrmse_raw": al["raw"]["nrmse"], "nrmse_dc": al["dc"]["nrmse"], "nrmse": al["dc_lag"]["nrmse"],
            "pearson": al["dc_lag"]["pearson"], "rmse_mv": al["dc_lag"]["rmse_mv"],
            "max_abs_err_mv": al["dc_lag"]["max_abs_err_mv"], "gain": al["dc_lag"]["gain"],
            "lag_ms": al["lag_s"] * 1000, "valid_coverage": ev["valid_coverage"],
            "hr_abs_err_bpm": ev["hr_abs_err_bpm"], "median_abs_r_amp_err_mv": float(np.median(amp_errs)) if amp_errs else float("nan"),
            "beat_recall": ev["ref_beat_recall"],
        })
    ok = [r for r in rows if r["ok"]]

    def med(k):
        return float(np.nanmedian([r[k] for r in ok])) if ok else float("nan")

    summary = {
        "n_records": len(rows), "n_ok": len(ok),
        "median_nrmse": med("nrmse"), "p90_nrmse": float(np.nanpercentile([r["nrmse"] for r in ok], 90)) if ok else float("nan"),
        "median_pearson": med("pearson"), "median_valid_coverage": med("valid_coverage"),
        "median_abs_heart_rate_error_bpm": med("hr_abs_err_bpm"),
        "median_abs_r_amplitude_error_mv": med("median_abs_r_amp_err_mv"),
        "median_nrmse_raw": med("nrmse_raw"), "median_nrmse_dc_only": med("nrmse_dc"),
        "median_gain": med("gain"), "median_lag_ms": med("lag_ms"), "median_beat_recall": med("beat_recall"),
    }
    checks = {
        "median_nrmse_max": summary["median_nrmse"] <= crit["median_nrmse_max"],
        "p90_nrmse_max": summary["p90_nrmse"] <= crit["p90_nrmse_max"],
        "median_pearson_min": summary["median_pearson"] >= crit["median_pearson_min"],
        "median_valid_coverage_min": summary["median_valid_coverage"] >= crit["median_valid_coverage_min"],
        "median_abs_heart_rate_error_bpm_max": summary["median_abs_heart_rate_error_bpm"] <= crit["median_abs_heart_rate_error_bpm_max"],
        "median_abs_r_amplitude_error_mv_max": summary["median_abs_r_amplitude_error_mv"] <= crit["median_abs_r_amplitude_error_mv_max"],
    }
    passed = all(checks.values()) and len(ok) == len(rows)
    out = {"gate": gate, "summary": summary, "checks": checks, "passed": bool(passed), "per_record": rows, "runtime_s": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("FEASIBILITY GATE:", "PASSED" if passed else "NOT PASSED", f"({len(ok)}/{len(rows)} extracted)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
