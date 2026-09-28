"""
30_fig_f1_bars.py — Finding 1 bar chart (paper figure fig4_f1_engagement_revision.png)
with the corrected, proposed-side numbers from code/29_f1_excluding_new.py.

The 487 final-only NEW obligations are excluded (they have no proposed version
to revise). Numbers below are copied from the code/29 output:
    high engagement (>= 5 addressers): 69.0% revised, n = 168
    less engaged (< 5 addressers):     51.5% revised, n = 12,075
    chi2 = 19.71, p < 0.001; docket-FE logit OR = 1.54 [0.83, 2.87], p = 0.174

Usage:
    python3 code/30_fig_f1_bars.py [output.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fig4_f1_engagement_revision.png"

LOW, HIGH = 51.5, 69.0
N_LOW, N_HIGH = 12075, 168
LIGHT, DARK = "#bcd0ea", "#0f3b6f"

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)

x = [0, 1]
ax.bar(x, [LOW, HIGH], width=0.42, color=[LIGHT, DARK], zorder=2)

for xi, val in zip(x, [LOW, HIGH]):
    ax.text(xi, val + 2.2, f"{val:.1f}%", ha="center", va="bottom",
            fontsize=17, fontweight="bold", color=DARK)

# Gap annotation between the bars.
ax.plot([0.21, 0.55], [LOW, LOW], ls="--", lw=1, color=DARK)
ax.plot([0.55, 0.79], [HIGH, HIGH], ls="--", lw=1, color=DARK)
ax.annotate("", xy=(0.5, HIGH), xytext=(0.5, LOW),
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.3))
ax.text(0.5, (LOW + HIGH) / 2, f"+{HIGH - LOW:.1f} pp", ha="center",
        va="center", fontsize=13, fontweight="bold", color=DARK,
        bbox=dict(facecolor="white", edgecolor="none", pad=2))

ax.set_xticks(x)
ax.set_xticklabels([f"Less-addressed\n(< 5 addressing commenters)\nn = {N_LOW:,}",
                    f"High-engagement\n(≥ 5 addressing commenters)\nn = {N_HIGH:,}"],
                   fontsize=11)
ax.set_ylim(0, 100)
ax.set_ylabel("Revision rate (%)", fontsize=12)
ax.yaxis.grid(True, ls="--", lw=0.6, color="#cccccc", zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

ax.set_title("Engagement and rule-text revision", fontsize=15,
             fontweight="bold", pad=44)
ax.text(0.5, 1.02,
        "$\\chi^2$ = 19.71, $p$ < 0.001; within-rulemaking OR (docket FE) = 1.54,\n"
        "95% CI [0.83, 2.87], $p$ = 0.174, $n$ = 12,243 proposed-side obligations",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
