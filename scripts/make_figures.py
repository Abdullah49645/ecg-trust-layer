#!/usr/bin/env python
"""Figures for the report, generated from results/results_full.json (no numbers are typed by hand)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results" / "results_full.json"))
T = R["test"]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1) risk-coverage
fig, ax = plt.subplots(figsize=(7.2, 4.2))
style = {"trust_operational": ("#0e5a64", 2.6, "-", "Trust score"), "accept_all": ("#444", 1.6, "--", "Accept every beat"),
         "image_quality_snr": ("#4a7fb5", 1.4, "-", "Global image SNR"), "continuity": ("#7a5c99", 1.4, "-", "Continuity"),
         "oracle_error_ranking": ("#9aa0a6", 1.4, ":", "Oracle (uses reference; upper bound)")}
for k, (c, w, ls, lab) in style.items():
    cv = T["curves"][k]
    ax.plot(cv["coverage"], cv["mean_err"], color=c, lw=w, ls=ls, label=lab)
ax.set_xlabel("coverage: share of beats accepted (most trusted first)")
ax.set_ylabel("mean beat error (NRMSE; unmatched = 1)")
ax.set_ylim(0, 0.12); ax.set_xlim(0, 1)
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"Held-out test: {T['n_records']} records, {T['n_beats']} beats, 21 image conditions", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "risk_coverage.png", dpi=170); plt.close(fig)

# 2) per-condition: bad-beat rate all vs accepted
conds = list(T["by_condition"])
all_r = [T["by_condition"][c]["bad_rate_all"] * 100 for c in conds]
acc_r = [(T["by_condition"][c]["accepted_bad_rate"] or 0) * 100 for c in conds]
acc_s = [T["by_condition"][c]["accepted_fraction"] * 100 for c in conds]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = range(len(conds))
ax.bar([i - 0.2 for i in x], all_r, 0.4, color="#b0b7bd", label="bad-beat rate, all beats")
ax.bar([i + 0.2 for i in x], acc_r, 0.4, color="#0e5a64", label="bad-beat rate, accepted beats")
for i, s in enumerate(acc_s):
    ax.text(i + 0.2, acc_r[i] + 0.8, f"{s:.0f}%", ha="center", fontsize=7, color="#0e5a64")
ax.set_xticks(list(x)); ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("% of beats that are bad"); ax.legend(frameon=False)
ax.set_title("Per condition (labels above teal bars: share of beats accepted)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "per_condition.png", dpi=170); plt.close(fig)
print("figures written to", OUT)
