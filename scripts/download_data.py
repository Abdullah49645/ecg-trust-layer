#!/usr/bin/env python
"""Download the PTB-XL subset used by the evaluation (500 Hz records, ~20 MB total).

Selection is deterministic (seeded, one record per patient, official folds), and is
written to ``configs/record_selection.json``. Re-running skips files already present.

    python scripts/download_data.py                 # full selection (tune 60 / val 30 / test 80)
    python scripts/download_data.py --n-tune 10 --n-val 5 --n-test 10   # smaller
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgtrust import data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "ptbxl"))
    ap.add_argument("--n-tune", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    root = Path(args.data_dir)
    print(f"Downloading PTB-XL metadata into {root} ...")
    data.download_metadata(root)
    meta = data.load_metadata(root)
    sel = data.select_records(meta, {"tune": args.n_tune, "val": args.n_val, "test": args.n_test}, seed=args.seed)
    data.save_selection(ROOT / "configs" / "record_selection.json", sel, args.seed)
    total = sum(len(v) for v in sel.values())
    k = 0
    for split, rows in sel.items():
        for row in rows:
            data.download_record(root, row["filename_hr"])
            k += 1
            if k % 10 == 0 or k == total:
                print(f"  {k}/{total} records")
    print("Done. Dataset: PTB-XL v1.0.3, CC BY 4.0 - see NOTICE.md for attribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
