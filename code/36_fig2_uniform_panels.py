"""
36_fig2_uniform_panels.py - Paper Figure 2 (Findings 1 and 2) as two matching panels.

Supersedes the two separate styles used for Figure 2:
    left  panel: code/30_fig_f1_bars.py
    right panel: build_fig_f2_directional() in code/28_generate_paper_figures.py
Both panels now share figure size, fonts, colors (light blue / navy, as in
code/30 and code/35), axis scale, title and subtitle layout, and the gap
annotation. The canvas size is fixed (no tight bounding box), so the two PNGs
have identical pixel dimensions and render at the same scale when included
side by side at 0.47\\textwidth.

Numbers are the paper's (Section 6.1). If data/processed/responsiveness_analysis.csv
is present, the script recomputes them and stops if anything disagrees.

Usage (repo root):
    python3 code/36_fig2_uniform_panels.py [output_dir]
Writes fig4_f1_engagement_revision.png and fig_f2_directional.png.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "paper/camera_ready_figures"
CSV = "data/processed/responsiveness_analysis.csv"

LIGHT, DARK = "#bcd0ea", "#0f3b6f"
FIGSIZE, DPI = (5.2, 3.9), 200   # same aspect as the previous left panel

F1 = dict(low=51.5, high=69.0, n_low=12075, n_high=168)
F2 = dict(opp=62.3, sup=68.3, n_opp=212, n_sup=312, z=-1.42, p=0.155)


def verify() -> None:
    """Recompute the plotted numbers from the per-obligation table."""
    if not os.path.exists(CSV):
        print(f"note: {CSV} not found; plotting the paper's numbers unchecked")
        return
    import pandas as pd
    d = pd.read_csv(CSV)
    p = d[d["outcome_state"] != "NEW"].copy()
    p["rev"] = p["outcome_state"].isin(["SURVIVED-edited", "MODIFIED", "DROPPED"])
    hi = p["n_comments_addressed"] >= 5
    got1 = dict(low=round(100 * p.loc[~hi, "rev"].mean(), 1),
                high=round(100 * p.loc[hi, "rev"].mean(), 1),
                n_low=int((~hi).sum()), n_high=int(hi.sum()))
    op, su = p["has_opposing"] == True, p["has_supporting"] == True  # noqa: E712
    x1, n1 = int(p.loc[op, "rev"].sum()), int(op.sum())
    x2, n2 = int(p.loc[su, "rev"].sum()), int(su.sum())
    pool = (x1 + x2) / (n1 + n2)
    z = (x1 / n1 - x2 / n2) / math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    got2 = dict(opp=round(100 * x1 / n1, 1), sup=round(100 * x2 / n2, 1),
                n_opp=n1, n_sup=n2, z=round(z, 2),
                p=round(math.erfc(abs(z) / math.sqrt(2)), 3))
    for want, got in ((F1, got1), (F2, got2)):
        for key, val in want.items():
            if got[key] != val:
                sys.exit(f"mismatch in {key}: paper {val}, data {got[key]}")
    print(f"numbers verified against {CSV}")


def panel(path: str, title: str, subtitle: str, vals, labels) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes([0.135, 0.255, 0.835, 0.525])
    x = [0, 1]
    lo, hi = vals
    ax.bar(x, vals, width=0.42, color=[LIGHT, DARK], zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=DARK)

    # Gap annotation: dashed guides from each bar top, double arrow, and the
    # label just below the arrow (clear of the bar-top value labels).
    ax.plot([0.21, 0.5], [lo, lo], ls="--", lw=1, color=DARK, zorder=3)
    ax.plot([0.5, 0.79], [hi, hi], ls="--", lw=1, color=DARK, zorder=3)
    ax.annotate("", xy=(0.5, hi), xytext=(0.5, lo),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    ax.text(0.5, lo - 2.5, f"+{hi - lo:.1f} pp", ha="center", va="top",
            fontsize=12, fontweight="bold", color=DARK)

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylabel("Revision rate (%)", fontsize=11.5)
    ax.yaxis.grid(True, ls="--", lw=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    cx = 0.135 + 0.835 / 2
    fig.text(cx, 0.975, title, ha="center", va="top",
             fontsize=13.5, fontweight="bold")
    fig.text(cx, 0.895, subtitle, ha="center", va="top",
             fontsize=10.5, linespacing=1.4, color="#222222")
    fig.savefig(path, dpi=DPI)   # fixed canvas: identical size for both panels
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    verify()
    os.makedirs(OUT_DIR, exist_ok=True)
    panel(os.path.join(OUT_DIR, "fig4_f1_engagement_revision.png"),
          "Engagement and rule-text revision",
          "Pooled: $\\chi^2$ = 19.71, $p$ < 0.001\n"
          "Within rulemaking: OR = 1.54 [0.83, 2.87], $p$ = 0.174",
          (F1["low"], F1["high"]),
          [f"Less-addressed\n(< 5 addressing\ncommenters)\nn = {F1['n_low']:,}",
           f"High-engagement\n(≥ 5 addressing\ncommenters)\nn = {F1['n_high']:,}"])
    panel(os.path.join(OUT_DIR, "fig_f2_directional.png"),
          "Direction and rule-text revision",
          f"Two-proportion $z$ = −{abs(F2['z']):.2f}, $p$ = {F2['p']:.3f}\n"
          "Difference within sampling noise",
          (F2["opp"], F2["sup"]),
          [f"Opposing\n(≥ 1 opposing\ncommenter)\nn = {F2['n_opp']:,}",
           f"Supporting\n(≥ 1 supporting\ncommenter)\nn = {F2['n_sup']:,}"])


if __name__ == "__main__":
    main()
