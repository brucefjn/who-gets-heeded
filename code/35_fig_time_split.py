"""
35_fig_time_split.py - Appendix figure: Finding 1 revision rates by period.

Same visual conventions as code/30_fig_f1_bars.py (paper Figure 2, left):
DejaVu Sans, light blue for less-addressed obligations, dark navy for
high-engagement obligations, bold value labels, dashed horizontal grid.

Reads the per-period numbers written by code/32_time_split.py
(data/processed/robustness_time_split_summary.json), so rerun code/32 first
if the data change.

Usage:
    python3 code/35_fig_time_split.py [output.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SUMMARY = "data/processed/robustness_time_split_summary.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fig_time_split_f1.png"
LIGHT, DARK = "#bcd0ea", "#0f3b6f"

with open(SUMMARY, encoding="utf-8") as f:
    splits = json.load(f)["splits"]

panels = [
    ("year_cluster", "By year of the proposed rule",
     ["2010-2012", "2013-2015", "2016-2018", "2019-2022"],
     ["2010–12", "2013–15", "2016–18", "2019–22"]),
    ("admin_final", "By administration at the final rule",
     ["Obama", "Trump", "Biden"], ["Obama", "Trump", "Biden"]),
]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4), dpi=200,
                         gridspec_kw={"width_ratios": [4, 3]})
width = 0.36
for ax, (key, title, periods, labels) in zip(axes, panels):
    per = splits[key]["periods"]
    low = [per[p]["f1"]["rate_low"] for p in periods]
    high = [per[p]["f1"]["rate_high"] for p in periods]
    n_high = [per[p]["f1"]["n_high"] for p in periods]
    n_dk = [per[p]["f1"]["dockets"] for p in periods]
    xs = list(range(len(periods)))
    ax.bar([x - width / 2 for x in xs], low, width, color=LIGHT, zorder=2)
    ax.bar([x + width / 2 for x in xs], high, width, color=DARK, zorder=2)
    for x, lo_v, hi_v in zip(xs, low, high):
        ax.text(x - width / 2, lo_v + 1.5, f"{lo_v:.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=DARK)
        ax.text(x + width / 2, hi_v + 1.5, f"{hi_v:.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\n{d} rulemakings\nn high = {n}"
                        for lab, d, n in zip(labels, n_dk, n_high)], fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=12, pad=8)
    ax.yaxis.grid(True, ls="--", lw=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    p = splits[key]["f1_interaction"].get("wald_p")
    if p is not None:
        ax.text(0.5, 0.97, f"heterogeneity test: p = {p:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, color="#333333")
axes[0].set_ylabel("Revision rate (%)", fontsize=12)

fig.suptitle("Engagement and rule-text revision by period (exploratory)",
             fontsize=15, fontweight="bold", y=0.99)
fig.legend(handles=[Patch(color=LIGHT, label="Less-addressed (< 5 addressing commenters)"),
                    Patch(color=DARK, label="High-engagement (≥ 5 addressing commenters)")],
           loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False,
           fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.925))
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
