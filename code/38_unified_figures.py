"""
38_unified_figures.py - Figures 1, 3, 4, and 5 of the camera-ready paper in the
same visual theme as Figures 2 and 6 (code/36, code/35).

Theme: DejaVu Sans; navy #0f3b6f for the emphasized group, light blue #bcd0ea
for the comparison group; bold navy value labels; dashed light-gray grid;
bold chart titles with a one- or two-line statistical subtitle.

  fig1_pipeline.png                 Figure 1: the auditing framework, showing
                                    each measurement component's audit status
                                    and which findings it feeds
  fig2_f3_org_majority.png          Figure 3: audit-corrected Finding 3 composition
  fig3_classifier_per_class.png     Figure 4: classifier vs. the human audit
  figure_5_f3_audit_comparison.png  Figure 5: Finding 3 before and after the audit

Numbers are the paper's. Figure 4 is computed from
data/processed/outcome_audit_reconciled.csv and checked against the paper.

Usage (repo root):
    python3 code/38_unified_figures.py [output_dir]
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/camera_ready_figures"
DPI = 200
LIGHT, DARK = "#bcd0ea", "#0f3b6f"
# One muted colour family per finding, plus one for the human audits. The light/dark
# pair inside each family keeps every chart readable in greyscale and for colour-blind readers.
F1_DARK, F1_LIGHT = DARK, LIGHT                 # Finding 1: engagement (blue)
F2_DARK, F2_LIGHT = "#0f6b5f", "#b3ddd6"        # Finding 2: direction (teal)
F3_DARK, F3_LIGHT = "#b5541c", "#f2c9a5"        # Finding 3: commenter type (amber)
AU_DARK, AU_LIGHT = "#4a3a8c", "#cdc6e8"        # blind human audits (purple)
TEXT, MUTED, ARROW = "#222222", "#6b7686", "#7a8594"
PANEL, GRID = "#f4f6f9", "#cccccc"
SE, MO = "SURVIVED-edited", "MODIFIED"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def style_axes(ax) -> None:
    ax.yaxis.grid(True, ls="--", lw=0.6, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- Figure 1
W1, H1 = 11.0, 5.3          # inches; printed at full text width
EDGE = "#8a94a6"


def figure_1(path: str) -> None:
    fig = plt.figure(figsize=(W1, H1), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fy = lambda inches: inches / H1          # noqa: E731  vertical inches -> fraction
    dash = {"proxy": (0, (5, 3)), "descriptive": (0, (1.2, 2.2))}

    def pill(cx, cy, label, kind):
        w = (len(label) * 0.068 + 0.26) / W1
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - fy(0.1)), w, fy(0.2),
                                    boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=AU_DARK if kind == "audited" else "white",
                                    ec=AU_DARK if kind == "audited" else EDGE,
                                    lw=1.0, ls=dash.get(kind, "solid"), zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.4, fontweight="bold",
                color="white" if kind == "audited" else "#4b5563", zorder=4)

    def box(x0, y0, x1, y1, title, lines, kind, badge=None, fill=DARK):
        face = {"input": PANEL, "audited": "#f3f1fa", "proxy": "white",
                "descriptive": "white", "finding": fill}[kind]
        edge = {"input": "#d5dbe3", "audited": AU_DARK, "finding": fill}.get(kind, EDGE)
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.012", fc=face, ec=edge,
                                    lw=1.6 if kind in ("audited", "finding") else 1.2,
                                    ls=dash.get(kind, "solid"), zorder=2))
        cx = (x0 + x1) / 2
        y = y1 - fy(0.15)
        if not badge:   # centre the text block vertically when there is no badge
            block = fy(0.19) + fy(0.27) - fy(0.19) + fy(0.18) * len(lines)
            y = (y0 + y1) / 2 + block / 2
        ax.text(cx, y, title, ha="center", va="top", fontsize=11, fontweight="bold",
                color="white" if kind == "finding" else "#1f2a37", zorder=3)
        y -= fy(0.27)
        for ln in lines:
            ax.text(cx, y, ln, ha="center", va="top", fontsize=9.3,
                    color="white" if kind == "finding" else TEXT, zorder=3)
            y -= fy(0.18)
        if badge:
            pill(cx, y0 + fy(0.2), badge, kind)

    def arrow(points):
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=ARROW, lw=1.3, zorder=1, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=11, color=ARROW, lw=1.3, zorder=1,
                                     shrinkA=0, shrinkB=0))

    S0, S1 = 0.012, 0.150          # sample
    I0, I1 = 0.182, 0.345          # inputs
    M0, M1 = 0.378, 0.735          # measurement
    Ma1, Mb0 = 0.551, 0.562        # measurement sub-columns
    F0, F1 = 0.782, 0.988          # findings
    BUS = 0.759
    R1 = (0.675, 0.910)
    R2 = (0.385, 0.620)
    R3 = (0.095, 0.330)

    for x, label in (((S0 + S1) / 2, "SAMPLE"), ((I0 + I1) / 2, "DATA"),
                     ((M0 + M1) / 2, "MEASUREMENT"), ((F0 + F1) / 2, "FINDINGS")):
        ax.text(x, 0.958, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=MUTED)

    box(S0, *R1[:1], S1, R1[1], "EPA dockets", ["2010–2022", "6,145 dockets", "786,197 comments"], "input")
    box(S0, R2[0], S1, R2[1], "36 anchors", ["stratified random", "+ extreme-case"], "input")
    box(I0, R1[0], I1, R1[1], "Federal Register", ["proposed + final rules", "72 documents"], "input")
    box(I0, R2[0], I1, R2[1], "Public comments", ["70,075 analyzable", "incl. 1,634 attachments"], "input")
    box(I0, R3[0], I1, R3[1], "Rhetoric schema", ["23 indicators, GPT-5", "not used for findings"], "descriptive", "Descriptive only")

    box(M0, R1[0], M1, R1[1], "Obligation extraction",
        ["deontic parser + GPT-5 verifier", "12,730 obligations · precision 0.95"],
        "audited", "Audited · n = 66")
    box(M0, R2[0], Ma1, R2[1], "Comment matching",
        ["GPT-5 judge", "matcher–human κ = 0.82"], "audited", "Audited · n = 100")
    box(Mb0, R2[0], M1, R2[1], "Outcome states",
        ["proposed → final", "F3 uses corrected labels"], "audited", "Audited · n = 150")
    box(M0, R3[0], Ma1, R3[1], "Commenter type",
        ["Title-field patterns", "org. vs individual"], "proxy", "Proxy · not audited")

    box(F0, R1[0], F1, R1[1], "Finding 1", ["Engagement tracks revision", "across rulemakings"], "finding", fill=F1_DARK)
    box(F0, R2[0], F1, R2[1], "Finding 2", ["Direction does not", "differentiate outcomes"], "finding", fill=F2_DARK)
    box(F0, R3[0], F1, R3[1], "Finding 3", ["Org-majority engagement", "→ editorial refinements"], "finding", fill=F3_DARK)

    mid = lambda r: (r[0] + r[1]) / 2   # noqa: E731
    arrow([((S0 + S1) / 2, R1[0]), ((S0 + S1) / 2, R2[1])])                  # dockets -> anchors
    arrow([(S1, mid(R2) - 0.03), (I0, mid(R2) - 0.03)])                      # anchors -> comments
    arrow([(S1, mid(R2) + 0.05), (0.166, mid(R2) + 0.05), (0.166, mid(R1)), (I0, mid(R1))])  # -> FR
    arrow([(I1, mid(R1)), (M0, mid(R1))])                                     # FR -> extraction
    arrow([(I1, mid(R2)), (M0, mid(R2))])                                     # comments -> matching
    arrow([((I0 + I1) / 2, R2[0]), ((I0 + I1) / 2, R3[1])])                  # comments -> rhetoric
    arrow([(I1, R2[0] + 0.035), (0.362, R2[0] + 0.035), (0.362, mid(R3)), (M0, mid(R3))])  # -> type
    arrow([((M0 + Ma1) / 2, R1[0]), ((M0 + Ma1) / 2, R2[1])])                # extraction -> matching
    arrow([((Mb0 + M1) / 2, R1[0]), ((Mb0 + M1) / 2, R2[1])])                # extraction -> outcomes
    # matching and outcome states feed all three findings through one bus
    ax.plot([(M0 + Ma1) / 2, (M0 + Ma1) / 2, BUS], [R2[0], R2[0] - 0.03, R2[0] - 0.03],
            color=ARROW, lw=1.3, zorder=1)
    ax.plot([M1, BUS], [mid(R2), mid(R2)], color=ARROW, lw=1.3, zorder=1)
    ax.plot([BUS, BUS], [mid(R3) + 0.04, mid(R1)], color=ARROW, lw=1.3, zorder=1)
    for y in (mid(R1), mid(R2), mid(R3) + 0.04):
        arrow([(BUS, y), (F0, y)])
    arrow([(Ma1, mid(R3) - 0.05), (F0, mid(R3) - 0.05)])                     # type -> Finding 3

    # legend
    items = [("Audited by blind dual-annotator coding", "audited"),
             ("Proxy, not yet audited", "proxy"), ("Descriptive only", "descriptive")]
    widths = [(len(t) * 0.066 + 0.55) / W1 for t, _ in items]
    x = 0.5 - (sum(widths) + 0.03 * (len(items) - 1)) / 2
    for (label, kind), w in zip(items, widths):
        ax.add_patch(FancyBboxPatch((x, 0.030), 0.024, fy(0.16),
                                    boxstyle="round,pad=0,rounding_size=0.008",
                                    fc=AU_DARK if kind == "audited" else "white",
                                    ec=AU_DARK if kind == "audited" else EDGE, lw=1.0,
                                    ls=dash.get(kind, "solid")))
        ax.text(x + 0.031, 0.030 + fy(0.08), label, va="center", fontsize=9, color=TEXT)
        x += w + 0.03
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------ Figures 3 and 5
def stacked_panel(ax, cells: dict, title: str, subtitle: str, title_size=12.0) -> None:
    """cells: {'SE': (org, not), 'MO': (org, not)}."""
    xs = [0, 1]
    names = [SE, MO]
    orgs = [100 * cells[k][0] / sum(cells[k]) for k in ("SE", "MO")]
    rest = [100 - o for o in orgs]
    ns = [sum(cells[k]) for k in ("SE", "MO")]
    ax.bar(xs, orgs, width=0.5, color=F3_DARK, zorder=2)
    ax.bar(xs, rest, width=0.5, bottom=orgs, color=F3_LIGHT, zorder=2)
    for x, o, r in zip(xs, orgs, rest):
        ax.text(x, o / 2, f"{o:.0f}%", ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", zorder=3)
        ax.text(x, o + r / 2, f"{r:.0f}%", ha="center", va="center", fontsize=13,
                fontweight="bold", color=F3_DARK, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\nn = {k}" for n, k in zip(names, ns)], fontsize=10.5)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylabel("Obligations (%)", fontsize=11.5)
    style_axes(ax)
    ax.set_title(f"{title}\n", fontsize=title_size, pad=6)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, color=TEXT)


def legend_org(fig, y) -> None:
    fig.legend(handles=[Patch(color=F3_DARK, label="Organizational-majority"),
                        Patch(color=F3_LIGHT, label="Not organizational-majority")],
               loc="center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False, fontsize=10.5)


PRE = {"SE": (43, 15), "MO": (28, 30)}
POST = {"SE": (13, 2), "MO": (57, 43)}


def figure_3(path: str) -> None:
    fig = plt.figure(figsize=(5.2, 3.9), dpi=DPI)
    ax = fig.add_axes([0.135, 0.19, 0.835, 0.50])
    stacked_panel(ax, POST, "", "")
    ax.set_title("")
    ax.texts[-1].remove()
    cx = 0.135 + 0.835 / 2
    fig.text(cx, 0.975, "Commenter composition by outcome", ha="center", va="top",
             fontsize=13.5, fontweight="bold")
    fig.text(cx, 0.895, "Audit-corrected: Fisher OR = 4.90, p = 0.044, n = 115",
             ha="center", va="top", fontsize=10.5, color=TEXT)
    legend_org(fig, 0.775)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")


def figure_5(path: str) -> None:
    fig = plt.figure(figsize=(10.8, 4.6), dpi=DPI)
    a1 = fig.add_axes([0.075, 0.17, 0.40, 0.52])
    a2 = fig.add_axes([0.575, 0.17, 0.40, 0.52])
    stacked_panel(a1, PRE, "Pre-audit (classifier labels)",
                  "Fisher OR = 3.07, p = 0.007, n = 116")
    stacked_panel(a2, POST, "Audit-corrected (adjudicated labels)",
                  "Fisher OR = 4.90, p = 0.044, n = 115")
    fig.suptitle("Finding 3 before and after the blind audit", fontsize=15,
                 fontweight="bold", y=0.985)
    legend_org(fig, 0.885)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- Figure 4
CLASSES = ["SURVIVED-unchanged", SE, MO, "DROPPED", "NEW"]
SHORT = ["SU", "SE", "MO", "DR", "NEW"]


def figure_4(path: str) -> None:
    import pandas as pd
    r = pd.read_csv("data/processed/outcome_audit_reconciled.csv")
    cm = [[int(((r.classifier_label == c) & (r.adjudicated_label == g)).sum())
           for g in CLASSES] for c in CLASSES]
    prec = [cm[k][k] / max(1, sum(cm[k])) for k in range(5)]
    rec = [cm[k][k] / max(1, sum(row[k] for row in cm)) for k in range(5)]
    assert round(prec[2], 2) == 0.94 and round(rec[2], 2) == 0.53, "MODIFIED metrics changed"
    assert round(prec[1], 2) == 0.20 and round(rec[1], 2) == 0.65, "SURVIVED-edited metrics changed"

    fig = plt.figure(figsize=(10.8, 4.6), dpi=DPI)
    a1 = fig.add_axes([0.085, 0.13, 0.33, 0.62])
    a2 = fig.add_axes([0.575, 0.13, 0.38, 0.62])

    cmap = LinearSegmentedColormap.from_list("theme", ["#ffffff", AU_LIGHT, AU_DARK])
    vmax = max(max(row) for row in cm) ** 0.5
    a1.imshow([[v ** 0.5 for v in row] for row in cm], cmap=cmap, vmin=0, vmax=vmax,
              aspect="equal")
    for i in range(5):
        a1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=AU_DARK, lw=1.6))
        for j in range(5):
            v = cm[i][j]
            a1.text(j, i, str(v), ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if v ** 0.5 > 0.55 * vmax else AU_DARK)
    a1.set_xticks(range(5))
    a1.set_yticks(range(5))
    a1.set_xticklabels(SHORT, fontsize=10.5)
    a1.set_yticklabels(SHORT, fontsize=10.5)
    a1.set_xlabel("Adjudicated human label", fontsize=11)
    a1.set_ylabel("Classifier label", fontsize=11)
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_title("Classifier vs. human labels (n = 150)", fontsize=12, pad=8)

    ys = list(range(5))
    h = 0.36
    a2.barh([y - h / 2 for y in ys], prec, height=h, color=AU_DARK, zorder=2, label="Precision")
    a2.barh([y + h / 2 for y in ys], rec, height=h, color=AU_LIGHT, zorder=2, label="Recall")
    for y, p, q in zip(ys, prec, rec):
        a2.text(p + 0.015, y - h / 2, f"{p:.2f}", va="center", fontsize=9.5,
                fontweight="bold", color=AU_DARK)
        a2.text(q + 0.015, y + h / 2, f"{q:.2f}", va="center", fontsize=9.5,
                fontweight="bold", color=AU_DARK)
    a2.axvline(0.70, color=ARROW, ls="--", lw=1.1, zorder=3)
    a2.set_yticks(ys)
    a2.set_yticklabels(SHORT, fontsize=10.5)
    a2.invert_yaxis()
    a2.set_xlim(0, 1.12)
    a2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.tick_params(axis="x", labelsize=10.5)
    a2.tick_params(axis="y", length=0)
    a2.set_xlabel("Score", fontsize=11)
    a2.xaxis.grid(True, ls="--", lw=0.6, color=GRID, zorder=0)
    a2.set_axisbelow(True)
    for side in ("top", "right"):
        a2.spines[side].set_visible(False)
    a2.set_title("Per-class precision and recall", fontsize=12, pad=8)

    fig.suptitle("Outcome-state classifier against the human audit", fontsize=15,
                 fontweight="bold", y=0.985)
    fig.legend(handles=[Patch(fc="white", ec=AU_DARK, lw=1.6, label="Agreement (outlined)"),
                        Patch(color=AU_DARK, label="Precision"), Patch(color=AU_LIGHT, label="Recall"),
                        Line2D([0], [0], color=ARROW, ls="--", lw=1.1, label="κ ≥ 0.70 threshold")],
               loc="center", bbox_to_anchor=(0.5, 0.885), ncol=4, frameon=False, fontsize=10.5)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_1(os.path.join(OUT, "fig1_pipeline.png"))
    figure_3(os.path.join(OUT, "fig2_f3_org_majority.png"))
    figure_4(os.path.join(OUT, "fig3_classifier_per_class.png"))
    figure_5(os.path.join(OUT, "figure_5_f3_audit_comparison.png"))
