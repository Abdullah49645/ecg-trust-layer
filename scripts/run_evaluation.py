#!/usr/bin/env python
"""One-command evaluation.

    python scripts/run_evaluation.py --preset full    # tune/val/test, 21 conditions (~25-40 min on 2 cores; cached afterwards)
    python scripts/run_evaluation.py --preset quick   # frozen model, 10 bundled held-out records, 8 conditions (~1-2 min)

Outputs (results/): results_full.json | results_quick.json, trust_model.json, raw/*.json.gz (cache).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgtrust import evaluation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["full", "quick"], default="quick")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-cache", action="store_true", help="recompute raw per-image results instead of reusing results/raw")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--limit", default="", help="debug: e.g. tune=6,val=4,test=6 (writes results_full_limited.json)")
    args = ap.parse_args()

    last = {"s": None}
    t0 = time.time()

    def progress(stage, done, total, msg):
        key = (stage, done)
        if last["s"] != key and (done == total or done % 5 == 0 or done == 0):
            print(f"[{time.time() - t0:7.1f}s] {stage:18s} {done}/{total}  {msg}", flush=True)
            last["s"] = key

    if args.preset == "full":
        limit = dict(kv.split("=") for kv in args.limit.split(",") if kv) if args.limit else None
        if limit:
            limit = {k: int(v) for k, v in limit.items()}
        res = evaluation.run_full(workers=args.workers, progress=progress, use_cache=not args.no_cache, n_boot=args.n_boot or 1000, limit=limit)
    else:
        res = evaluation.run_quick(workers=args.workers, progress=progress, n_boot=args.n_boot or 300)
    h = res["test"]["hypotheses"]
    print("\n=== summary ===")
    print(f"preset={res['preset']}  runtime={res['runtime_s']:.0f}s  beats={res['test']['n_beats']}  records={res['test']['n_records']}")
    print(f"AURC accept-all   : {h['aurc_accept_all']['point']:.4f}")
    print(f"AURC trust ({res['trust_model']['operational_score']}): {h['aurc_trust_operational']['point']:.4f}  ci95={[round(x, 4) for x in h['aurc_trust_operational']['ci95']]}")
    print(f"AURC best baseline ({h['best_simple_baseline']}): {h['aurc_best_baseline']['point']:.4f}")
    print(f"H1 (trust < accept-all)     : supported={h['H1_trust_beats_accept_all']['supported']}  delta={h['H1_trust_beats_accept_all']['delta_point']:.4f} ci95={[round(x, 4) for x in h['H1_trust_beats_accept_all']['delta_ci95']]}")
    print(f"H2 (trust < best baseline)  : supported={h['H2_trust_beats_best_baseline']['supported']}  delta={h['H2_trust_beats_best_baseline']['delta_point']:.4f} ci95={[round(x, 4) for x in h['H2_trust_beats_best_baseline']['delta_ci95']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
