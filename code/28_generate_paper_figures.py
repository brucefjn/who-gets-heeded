"""
28_generate_paper_figures.py — Regenerate the four paper figures with
publication-quality polish for the EAAMO 2026 submission.

Filename → paper-position mapping (paper.tex include order):
    fig1_pipeline.pdf              — paper Figure 1   (pipeline diagram)
    fig4_f1_engagement_revision.pdf — paper Figure 2   (Finding 1 LOO forest)
    fig2_f3_org_majority.pdf       — paper Figure 4   (Finding 3 mosaic, pre/post audit)
    fig3_classifier_per_class.pdf  — paper Figure 3   (classifier confusion matrix)

The PDFs are emitted to paper/figures/. The paper.tex \\includegraphics
calls already reference these filenames; this script does NOT touch
paper.tex.

DATA SOURCES (real audit data, do not invent):
  - Figure 1: static design + paper-reported kappa annotations
  - Figure 2: requires per-docket LOO chi-square. If
    data/processed/finding1_loo_sensitivity.csv exists, reads it; else
    requires data/processed/stage4/*.parquet + outcomes parquet to
    re-compute. Surfaces a clear error if neither is available
    (the existing fig4_*.pdf is preserved in that case).
  - Figure 3: data/processed/outcome_audit_classifier_labels.csv +
    data/processed/outcome_audit_yue.csv (rater_1_label is the gold;
    rater_2 is currently NaN — Yue hasn't coded yet — so the
    "adjudicated gold" is Bruce-only and the figure caption flags it).
  - Figure 4: pre-audit cells (43, 15, 28, 30) from §6.4 baseline;
    audit-corrected cells (13, 2, 57, 43) per the task spec.

Coherent palette (used across all 4):
    AUDITED:        ColorBrewer Set2 green        '#66c2a5'
    DESCRIPTIVE:    ColorBrewer Set2 yellow       '#ffd92f'
    PROXY:          ColorBrewer Set2 gray         '#b3b3b3'
    ORG-MAJORITY:   ColorBrewer Set2 salmon       '#fc8d62'
    NON-ORG:        ColorBrewer Set2 blue         '#8da0cb'
    REFERENCE LINE: dark gray                     '#404040'

Usage:
    python3 code/28_generate_paper_figures.py
    python3 code/28_generate_paper_figures.py --only fig1,fig3,fig4
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
except ImportError as e:
    sys.exit(f"ERROR: {e}. Required: pip install matplotlib pandas pyarrow")


# ---------------------------------------------------------------------------
# Paths + palette + typography
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = _REPO_ROOT / "paper" / "figures"
DPI = 200

AUDITED = "#66c2a5"     # Set2 green
DESCRIPTIVE = "#ffd92f" # Set2 yellow
PROXY = "#b3b3b3"       # Set2 gray
ORG = "#fc8d62"         # Set2 salmon (org-majority)
NONORG = "#8da0cb"      # Set2 blue
REF = "#404040"         # dark gray reference line
EDGE = "#444444"        # box edge

# Diagonal/off-diagonal heatmap colors (Fig 3): green-good, red-bad
DIAG_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "diag_green", ["#f0f7f4", "#1b7837"],
)
OFF_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "off_red", ["#fdecea", "#c0392b"],
)


def _set_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def _save(fig, basename: str) -> None:
    """Save both .pdf and .png next to paper/figures/. PDFs are what
    paper.tex includes; PNGs are convenience for quick visual review."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{basename}.pdf"
    png_path = FIG_DIR / f"{basename}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"  wrote {pdf_path}", file=sys.stderr)
    print(f"  wrote {png_path}", file=sys.stderr)


# ===========================================================================
# Figure 1 — Pipeline diagram (paper Figure 1)
# ===========================================================================
def build_fig1_pipeline() -> None:
    """Four-stage pipeline left-to-right with audit-kappa annotations.
    Coding by color: green = audited, yellow = descriptive-only,
    gray = proxy/unaudited, red = failure surfaced.

    Layout uses a wide figure (designed for figure* / two-column-spanning)
    with explicit y-bands so title, body text, and component pills do
    not overlap.  No aspect=equal — text is sized in points so we want
    the data coords to stretch with the figure size."""
    print("\n[fig1] pipeline diagram", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Four cards laid out left-to-right.  Each card occupies the full
    # height band (y = 8 .. 92).  Within a card we use three y-bands:
    #   title:      y in [82, 90]
    #   body lines: y in [42, 78]   (line spacing = 5 y-units)
    #   pills:      y in [11, 38]   (pill height 4.4, gap 1)
    # Card widths are equal (22 wide); gaps between cards = 2.
    CARD_W = 22
    CARD_GAP = 2
    LEFT_PAD = 2
    card_xs = [LEFT_PAD + i * (CARD_W + CARD_GAP) for i in range(4)]

    stages = [
        {
            "x": card_xs[0],
            "title": "1. Corpus",
            "lines": [
                "786,197 EPA comments",
                "(2010–2022)",
                "",
                "1,634 attachments",
                "OCR-recovered",
                "(92.4% within",
                "1,769 sample)",
            ],
            "components": [
                ("Comment ingest", PROXY),
                ("Submitter-type reconstruction", PROXY),
            ],
        },
        {
            "x": card_xs[1],
            "title": "2. Anchor selection",
            "lines": [
                "36 anchor rulemakings",
                "(stratified +",
                "extreme-case)",
                "",
                "29 productive",
                "after fielding",
                "",
            ],
            "components": [
                ("Stratified random", DESCRIPTIVE),
                ("Extreme-case sampling", DESCRIPTIVE),
            ],
        },
        {
            "x": card_xs[2],
            "title": "3. Extract & match",
            "lines": [
                "Obligations:",
                "12,243 proposed",
                "+ 487 NEW",
                "",
                "Comment-obligation",
                "pairs: 398,509",
                "(addressed: 5.4%)",
            ],
            "components": [
                ("Obligation extraction (P=0.95)", AUDITED),
                ("Stage 4 matching (κ=1.0 / 0.95)", AUDITED),
                ("23-indicator rhetoric schema", DESCRIPTIVE),
            ],
        },
        {
            "x": card_xs[3],
            "title": "4. Outcomes & audit",
            "lines": [
                "Outcome states:",
                "SE / MO / DR /",
                "SU / NEW",
                "",
                "Per-class audit",
                "(n=150)",
                "",
            ],
            "components": [
                ("Outcome classifier", AUDITED),
                ("κ=0.898 inter-rater", AUDITED),
                ("κ=0.137 clf vs human", "#dd6363"),
                ("Findings 1 / 2 / 3", AUDITED),
            ],
        },
    ]

    # ---- y-band constants (data coordinates 0..100) --------------------
    CARD_Y_BOTTOM = 8
    CARD_Y_TOP = 92
    CARD_H = CARD_Y_TOP - CARD_Y_BOTTOM  # 84
    TITLE_Y = 87
    BODY_Y_TOP = 78
    BODY_LINE_SPACING = 4.6
    PILL_Y_TOP = 36   # top edge of topmost pill
    PILL_H = 4.6
    PILL_GAP = 1.3

    # Draw stages
    for s in stages:
        x = s["x"]
        # Card background
        bg = FancyBboxPatch(
            (x, CARD_Y_BOTTOM), CARD_W, CARD_H,
            boxstyle="round,pad=0.4,rounding_size=1.8",
            linewidth=0.9, edgecolor=EDGE, facecolor="white",
        )
        ax.add_patch(bg)
        # Title bar (light fill behind title)
        title_bar = FancyBboxPatch(
            (x + 0.5, CARD_Y_TOP - 9.5), CARD_W - 1, 8,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=0, facecolor="#f2efe8", alpha=0.85,
        )
        ax.add_patch(title_bar)
        # Title text
        ax.text(
            x + CARD_W / 2, TITLE_Y, s["title"],
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1f1f1f",
        )
        # Body text lines
        for i, line in enumerate(s["lines"]):
            ax.text(
                x + CARD_W / 2,
                BODY_Y_TOP - i * BODY_LINE_SPACING,
                line,
                ha="center", va="top",
                fontsize=8.5, color="#333",
            )
        # Component pills
        for j, (label, color) in enumerate(s["components"]):
            pill_top = PILL_Y_TOP - j * (PILL_H + PILL_GAP)
            pill = FancyBboxPatch(
                (x + 1, pill_top - PILL_H), CARD_W - 2, PILL_H,
                boxstyle="round,pad=0,rounding_size=1.2",
                linewidth=0.5, edgecolor=EDGE, facecolor=color, alpha=0.85,
            )
            ax.add_patch(pill)
            ax.text(
                x + CARD_W / 2, pill_top - PILL_H / 2, label,
                ha="center", va="center",
                fontsize=7.8, color="#111",
            )

    # Connector arrows between cards (at card mid-height)
    arrow_y = (CARD_Y_BOTTOM + CARD_Y_TOP) / 2
    for i in range(len(stages) - 1):
        x0 = stages[i]["x"] + CARD_W
        x1 = stages[i + 1]["x"]
        arrow = FancyArrowPatch(
            (x0 + 0.1, arrow_y), (x1 - 0.3, arrow_y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.1, color=EDGE,
        )
        ax.add_patch(arrow)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=AUDITED, edgecolor=EDGE, label="Audited"),
        mpatches.Patch(facecolor=DESCRIPTIVE, edgecolor=EDGE,
                       label="Descriptive-only"),
        mpatches.Patch(facecolor=PROXY, edgecolor=EDGE,
                       label="Proxy / unaudited"),
        mpatches.Patch(facecolor="#dd6363", edgecolor=EDGE,
                       label="Failure surfaced"),
    ]
    ax.legend(
        handles=legend_items, loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=False, fontsize=9,
    )

    _save(fig, "fig1_pipeline")
    plt.close(fig)


# ===========================================================================
# Figure 2 — Finding 1 LOO forest plot (paper Figure 2, filename fig4_*)
# ===========================================================================
def _load_or_compute_f1_loo() -> pd.DataFrame | None:
    """Return DataFrame with columns [docket_id, chi2, p, V, n_dropped]
    for each of the 29 LOO conditions, or None if data unavailable.

    Tries in order:
      1. Precomputed CSV at data/processed/finding1_loo_sensitivity.csv
      2. Recompute from data/processed/stage4/*.parquet +
         data/processed/path_a_obligation_outcomes.parquet
    """
    precomputed = Path("data/processed/finding1_loo_sensitivity.csv")
    if precomputed.exists():
        return pd.read_csv(precomputed)

    stage4_files = sorted(glob.glob("data/processed/stage4/*.parquet"))
    outcomes_path = Path("data/processed/path_a_obligation_outcomes.parquet")
    if not stage4_files or not outcomes_path.exists():
        return None

    from scipy import stats
    import re
    print("  [fig2] re-computing LOO across 29 dockets …", file=sys.stderr)
    df_matches = pd.concat([pd.read_parquet(p) for p in stage4_files])
    df_outcomes = pd.read_parquet(outcomes_path)
    addr = df_matches[df_matches["addressed"] == True]   # noqa: E712
    n_per = addr.groupby("obligation_id")["comment_id"].nunique()
    out = df_outcomes.merge(
        n_per.rename("n_addressing").reset_index(),
        on="obligation_id", how="left",
    ).fillna({"n_addressing": 0})
    out["high_engagement"] = out["n_addressing"] >= 5
    out["was_revised"] = out["outcome_state"].isin(
        ["SURVIVED-edited", "MODIFIED", "DROPPED"]
    )
    _DOCKET_RE = re.compile(r"^(EPA-[A-Z0-9-]+?)__(?:proposed|final)__")
    out["docket_id"] = out["obligation_id"].map(
        lambda x: (_DOCKET_RE.match(x).group(1)
                   if isinstance(x, str) and _DOCKET_RE.match(x) else None)
    )
    out = out.dropna(subset=["docket_id"])
    rows = []
    for docket in sorted(out["docket_id"].unique()):
        sub = out[out["docket_id"] != docket]
        ct = pd.crosstab(sub["high_engagement"], sub["was_revised"])
        if ct.size < 4:
            continue
        chi2, p, _dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = (chi2 / n) ** 0.5
        n_dropped = (out["docket_id"] == docket).sum()
        rows.append({"docket_id": docket, "chi2": chi2, "p": p,
                     "V": v, "n_dropped": int(n_dropped)})
    df = pd.DataFrame(rows)
    df.to_csv(precomputed, index=False)
    return df


def build_fig2_finding1_loo() -> bool:
    """Forest plot of LOO chi-square across 29 dockets, with the
    headline chi-square at 24.53 as a reference line. Annotates the
    two most-influential dockets (largest |Δchi-square|)."""
    print("\n[fig2] Finding 1 LOO forest plot "
          "(filename fig4_f1_engagement_revision)", file=sys.stderr)
    loo = _load_or_compute_f1_loo()
    if loo is None:
        print("  [fig2] BLOCKED — neither precomputed LOO CSV "
              "(data/processed/finding1_loo_sensitivity.csv) nor "
              "stage4 parquets + outcomes parquet are available in "
              "this worktree.\n  [fig2] Preserving existing "
              "fig4_f1_engagement_revision.pdf — re-run on a "
              "workstation with the production data.",
              file=sys.stderr)
        return False

    # Sort by chi2 ascending so the highest-influence drop (the docket
    # whose removal moves the test stat MOST) is visually prominent.
    loo = loo.sort_values("chi2").reset_index(drop=True)
    n = len(loo)
    fig, ax = plt.subplots(figsize=(3.3, 5.2))
    headline_chi2 = 24.53

    # Identify the two most-influential dockets (largest abs Δ from headline)
    loo["abs_delta"] = (loo["chi2"] - headline_chi2).abs()
    most_influential = set(loo.nlargest(2, "abs_delta")["docket_id"].tolist())
    annotate_targets = {
        "EPA-HQ-OAR-2009-0491": "EPA-HQ-OAR-2009-0491\n(n=1,638)",
        "EPA-HQ-OAR-2010-0505": "EPA-HQ-OAR-2010-0505\n(n=954)",
    }

    # Highlight bars whose docket is in `annotate_targets` OR in `most_influential`.
    y = np.arange(n)
    bar_colors = []
    for d in loo["docket_id"]:
        if d in annotate_targets or d in most_influential:
            bar_colors.append(AUDITED)
        else:
            bar_colors.append("#cccccc")

    ax.barh(y, loo["chi2"], color=bar_colors, edgecolor="none", height=0.7)
    ax.axvline(headline_chi2, color=REF, linestyle="--", linewidth=0.9,
               label=f"Headline χ² = {headline_chi2}")

    # Annotate the two most-influential dockets explicitly.
    for d, label in annotate_targets.items():
        if d not in set(loo["docket_id"]):
            continue
        i = loo.index[loo["docket_id"] == d][0]
        ax.annotate(
            label, xy=(loo["chi2"].iloc[i], i),
            xytext=(loo["chi2"].iloc[i] + 1.0, i + 0.5),
            fontsize=6.5, color="#222",
            arrowprops=dict(arrowstyle="-", color="#888", linewidth=0.5),
        )

    # Compact y-tick labels — last 4 chars of docket
    short_labels = [d[-7:] for d in loo["docket_id"]]
    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=5.6)
    ax.set_xlabel("Chi-square (with one docket removed)", fontsize=8.5)
    ax.set_ylabel("Docket dropped", fontsize=8.5)
    ax.set_title(
        "Finding 1 robustness: χ² under leave-one-docket-out\n"
        f"(n = {n} dockets; chi-square range "
        f"[{loo['chi2'].min():.2f}, {loo['chi2'].max():.2f}])",
        fontsize=8.5, pad=6,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_xlim(0, max(loo["chi2"].max() * 1.18, headline_chi2 * 1.18))
    _save(fig, "fig4_f1_engagement_revision")
    plt.close(fig)
    return True


# ===========================================================================
# Figure 3 — Confusion matrix + per-class precision/recall sidebar
# (paper Figure 3, filename fig3_classifier_per_class)
# ===========================================================================
_OUTCOME_ORDER = ["SURVIVED-unchanged", "SURVIVED-edited",
                  "MODIFIED", "DROPPED", "NEW"]
_OUTCOME_SHORT = {
    "SURVIVED-unchanged": "SU",
    "SURVIVED-edited":    "SE",
    "MODIFIED":           "MO",
    "DROPPED":            "DR",
    "NEW":                "NEW",
}


def _load_confusion_matrix() -> tuple[np.ndarray, list[str], dict] | None:
    """Return (counts 5x5, labels short, per_class_stats dict) using
    rater_1 as the gold (rater_2 currently empty — Yue hasn't coded)."""
    clf_path = Path("data/processed/outcome_audit_classifier_labels.csv")
    yue_path = Path("data/processed/outcome_audit_yue.csv")
    if not clf_path.exists() or not yue_path.exists():
        return None
    clf = pd.read_csv(clf_path)
    gold = pd.read_csv(yue_path, encoding="utf-8-sig")
    merged = clf.merge(gold[["obligation_id", "rater_1_label", "rater_2_label"]],
                       on="obligation_id", how="inner")
    # rater_2 is currently all-NaN; fall back to rater_1 as the gold.
    if merged["rater_2_label"].notna().any():
        # If reconciliation lands later, prefer rater_2 where present.
        merged["gold"] = merged["rater_2_label"].fillna(merged["rater_1_label"])
    else:
        merged["gold"] = merged["rater_1_label"]
    n = len(merged)
    cm = np.zeros((len(_OUTCOME_ORDER), len(_OUTCOME_ORDER)), dtype=int)
    for _, row in merged.iterrows():
        c = row["classifier_label"]
        g = row["gold"]
        if c in _OUTCOME_ORDER and g in _OUTCOME_ORDER:
            i = _OUTCOME_ORDER.index(c)
            j = _OUTCOME_ORDER.index(g)
            cm[i, j] += 1
    # Per-class precision (column-wise: TP / sum of predictions in that row?)
    # Standard: precision = TP / (TP + FP) for the class.
    # If rows = classifier, cols = gold: precision_class = cm[k,k] / sum(cm[k,:])
    # recall_class = cm[k,k] / sum(cm[:,k])
    stats = {}
    for k, label in enumerate(_OUTCOME_ORDER):
        tp = cm[k, k]
        pred_total = cm[k, :].sum()
        gold_total = cm[:, k].sum()
        prec = tp / pred_total if pred_total > 0 else float("nan")
        rec = tp / gold_total if gold_total > 0 else float("nan")
        stats[label] = {"precision": prec, "recall": rec,
                        "n_gold": int(gold_total), "n_pred": int(pred_total)}
    return cm, [_OUTCOME_SHORT[x] for x in _OUTCOME_ORDER], stats


def build_fig3_confusion() -> bool:
    """Heatmap of classifier (rows) vs adjudicated gold (cols) with a
    per-class precision/recall sidebar on the right."""
    print("\n[fig3] confusion matrix + precision/recall sidebar",
          file=sys.stderr)
    loaded = _load_confusion_matrix()
    if loaded is None:
        print("  [fig3] BLOCKED — audit CSVs missing.", file=sys.stderr)
        return False
    cm, labels, stats = loaded
    n_total = int(cm.sum())

    fig = plt.figure(figsize=(6.4, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0],
                          wspace=0.45, left=0.10, right=0.97,
                          top=0.86, bottom=0.18)
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])

    # ---- Confusion matrix heatmap ----
    K = len(labels)
    # Two-color overlay: diagonal cells use green colormap, off-diagonal use red.
    diag_vals = np.where(np.eye(K, dtype=bool), cm, np.nan)
    off_vals = np.where(~np.eye(K, dtype=bool), cm, np.nan)
    # Independent normalization so the diagonal greens are visible alongside
    # high off-diagonals.
    vmax_diag = max(1, np.nanmax(diag_vals)) if np.any(~np.isnan(diag_vals)) else 1
    vmax_off = max(1, np.nanmax(off_vals)) if np.any(~np.isnan(off_vals)) else 1

    ax_cm.imshow(
        np.ma.masked_invalid(diag_vals), cmap=DIAG_CMAP,
        vmin=0, vmax=vmax_diag, aspect="equal",
    )
    ax_cm.imshow(
        np.ma.masked_invalid(off_vals), cmap=OFF_CMAP,
        vmin=0, vmax=vmax_off, aspect="equal",
    )
    # Cell annotations: black text where the cell color is light, white where dark
    for i in range(K):
        for j in range(K):
            v = cm[i, j]
            if i == j:
                txt_color = "#0a0a0a" if v / vmax_diag < 0.55 else "white"
            else:
                txt_color = "#0a0a0a" if v / vmax_off < 0.55 else "white"
            ax_cm.text(j, i, str(int(v)), ha="center", va="center",
                       color=txt_color, fontsize=9, fontweight="bold")

    ax_cm.set_xticks(np.arange(K))
    ax_cm.set_yticks(np.arange(K))
    ax_cm.set_xticklabels(labels, fontsize=8)
    ax_cm.set_yticklabels(labels, fontsize=8)
    ax_cm.set_xlabel("Adjudicated gold (rater)", fontsize=8.5)
    ax_cm.set_ylabel("Classifier prediction", fontsize=8.5)
    ax_cm.set_title("Classifier × gold confusion (n=" + f"{n_total})",
                    fontsize=9.5, pad=4)
    # Show diagonal grid for legibility
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    ax_cm.tick_params(length=0)

    # ---- Right sidebar: per-class precision + recall horizontal bars ----
    y = np.arange(K)
    bar_h = 0.36
    prec = [stats[c]["precision"] for c in _OUTCOME_ORDER]
    rec = [stats[c]["recall"] for c in _OUTCOME_ORDER]
    ax_pr.barh(y + bar_h / 2, prec, height=bar_h,
               color=ORG, edgecolor="none", label="Precision")
    ax_pr.barh(y - bar_h / 2, rec, height=bar_h,
               color=NONORG, edgecolor="none", label="Recall")
    ax_pr.set_yticks(y)
    ax_pr.set_yticklabels(labels, fontsize=8)
    ax_pr.invert_yaxis()
    ax_pr.set_xlim(0, 1.0)
    ax_pr.set_xlabel("Score", fontsize=8.5)
    ax_pr.set_title("Per-class metrics", fontsize=9.5, pad=4)
    ax_pr.axvline(0.70, color=REF, linestyle="--", linewidth=0.7,
                  label="κ ≥ 0.70")
    # Place legend below the panel so it cannot overlap the bars (NEW + DR
    # rows both extend to 1.0 and were colliding with "lower right").
    ax_pr.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=False, fontsize=7.5, handlelength=1.6,
        columnspacing=1.2, borderaxespad=0.2,
    )

    # Annotate scores at bar tips (each label on its own bar: precision
    # bars sit at y + bar_h/2, recall bars at y - bar_h/2)
    for i, (pv, rv) in enumerate(zip(prec, rec)):
        if not np.isnan(pv):
            ax_pr.text(pv + 0.01, i + bar_h / 2,
                       f"{pv:.2f}",
                       va="center", fontsize=6.4, color="#222")
        if not np.isnan(rv):
            ax_pr.text(rv + 0.01, i - bar_h / 2,
                       f"{rv:.2f}",
                       va="center", fontsize=6.4, color="#222")

    _save(fig, "fig3_classifier_per_class")
    plt.close(fig)
    return True


# ===========================================================================
# Figure 4 — Mosaic plot: pre-audit vs audit-corrected F3 contingency
# (paper Figure 4, filename fig2_f3_org_majority)
# ===========================================================================
def _draw_mosaic(ax, cells: tuple[int, int, int, int],
                 title: str, panel_or: float, panel_p: float,
                 show_legend: bool) -> None:
    """cells = (SE-org, SE-noOrg, MO-org, MO-noOrg). Mosaic: column
    widths proportional to outcome-state column totals; within each
    column, row heights proportional to (org / noOrg).

    Color encodes org-majority status (consistent across panels)."""
    se_org, se_noo, mo_org, mo_noo = cells
    se_total = se_org + se_noo
    mo_total = mo_org + mo_noo
    total = se_total + mo_total
    # Column widths (proportional to outcome marginal)
    se_w = se_total / total
    mo_w = mo_total / total
    # Within-column heights
    def _h_org(col_total, org_count):
        return (org_count / col_total) if col_total > 0 else 0.0
    se_h_org = _h_org(se_total, se_org)
    mo_h_org = _h_org(mo_total, mo_org)

    # Coords: x is in (0, 1), y in (0, 1)
    x0 = 0.0
    # SE column
    ax.add_patch(Rectangle(
        (x0, 0.0), se_w, se_h_org,
        facecolor=ORG, edgecolor="white", linewidth=1.2,
    ))
    ax.add_patch(Rectangle(
        (x0, se_h_org), se_w, 1.0 - se_h_org,
        facecolor=NONORG, edgecolor="white", linewidth=1.2,
    ))
    # MO column
    x1 = x0 + se_w
    ax.add_patch(Rectangle(
        (x1, 0.0), mo_w, mo_h_org,
        facecolor=ORG, edgecolor="white", linewidth=1.2,
    ))
    ax.add_patch(Rectangle(
        (x1, mo_h_org), mo_w, 1.0 - mo_h_org,
        facecolor=NONORG, edgecolor="white", linewidth=1.2,
    ))

    # Cell count labels (centered inside each cell when room allows)
    def _cell_label(cx, cy, w, h, count, dark_bg):
        if w * h < 0.018:
            return
        ax.text(cx, cy, str(count), ha="center", va="center",
                fontsize=8.5,
                color="white" if dark_bg else "#101010",
                fontweight="bold")

    _cell_label(x0 + se_w / 2, se_h_org / 2,
                se_w, se_h_org, se_org, dark_bg=True)
    _cell_label(x0 + se_w / 2, se_h_org + (1 - se_h_org) / 2,
                se_w, 1 - se_h_org, se_noo, dark_bg=False)
    _cell_label(x1 + mo_w / 2, mo_h_org / 2,
                mo_w, mo_h_org, mo_org, dark_bg=True)
    _cell_label(x1 + mo_w / 2, mo_h_org + (1 - mo_h_org) / 2,
                mo_w, 1 - mo_h_org, mo_noo, dark_bg=False)

    # X-axis labels: outcome states
    ax.set_xticks([x0 + se_w / 2, x1 + mo_w / 2])
    ax.set_xticklabels(["SURVIVED-edited", "MODIFIED"], fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Subtle vertical separator at the SE/MO column boundary
    ax.axvline(x1, color="white", linewidth=2.2)
    ax.set_title(f"{title}\nOR = {panel_or:.2f}, p = {panel_p:.3f}",
                 fontsize=9.5, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_legend:
        ax.legend(handles=[
            mpatches.Patch(facecolor=ORG, edgecolor="white",
                           label="Org-majority addressing"),
            mpatches.Patch(facecolor=NONORG, edgecolor="white",
                           label="Non-org-majority"),
        ], loc="lower center", bbox_to_anchor=(0.5, -0.25),
                  ncol=2, frameon=False, fontsize=7.5)


def build_fig4_finding3_mosaic() -> None:
    """Two-panel mosaic showing the §6.4 cells pre-audit vs after
    classifier-error correction. Width ∝ outcome marginal; within-column
    height ∝ org-majority share. Color = org-majority status."""
    print("\n[fig4] Finding 3 pre/post-audit mosaic "
          "(filename fig2_f3_org_majority)", file=sys.stderr)
    # Pre-audit cells from the §6.4 baseline.
    pre_cells = (43, 15, 28, 30)
    # Audit-corrected cells per task spec.
    post_cells = (13, 2, 57, 43)

    # Recompute OR + Fisher p for each panel so the title carries the
    # right numbers even if the spec values drift.
    from scipy.stats import fisher_exact
    def _or_p(cells):
        a, b, c, d = cells
        ad = (a + 0.5) * (d + 0.5) if min(a, b, c, d) == 0 else a * d
        bc = (b + 0.5) * (c + 0.5) if min(a, b, c, d) == 0 else b * c
        OR = ad / bc
        _, p = fisher_exact([[a, b], [c, d]])
        return OR, float(p)

    pre_or, pre_p = _or_p(pre_cells)
    post_or, post_p = _or_p(post_cells)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3),
                             gridspec_kw=dict(wspace=0.18))
    _draw_mosaic(axes[0], pre_cells, "Pre-audit", pre_or, pre_p,
                 show_legend=False)
    _draw_mosaic(axes[1], post_cells, "Audit-corrected", post_or, post_p,
                 show_legend=True)

    fig.suptitle(
        "Finding 3: outcome × org-majority before and after classifier "
        "audit (column width ∝ outcome n; row height ∝ org share)",
        fontsize=9.5, y=1.01,
    )
    _save(fig, "fig2_f3_org_majority")
    plt.close(fig)


# ===========================================================================
# Figure F2 — Finding 2 directional null (oppose vs support revision rates)
# ===========================================================================
def build_fig_f2_directional() -> None:
    """Two-bar comparison of revision rates among addressed obligations
    with ≥1 opposing vs ≥1 supporting commenter. The story is the null:
    62.3% vs 68.3% revised, two-prop z = -1.42, p = 0.155 (n_op=212, n_sp=312).

    Visual style matches Yue's F1 engagement bar (single-color, percentage
    labels on top, statistical subtitle, light grid). Stylistic choice:
    use a single muted blue across both bars to underscore that the
    finding is the equality, not a contrast."""
    print("\n[fig_f2] directional null", file=sys.stderr)

    # Hard-coded from §6.3 / Finding 2 in paper. Numbers verified against
    # the body text: "62.3% revised; 68.3% revised; z = -1.42, p = 0.155".
    rate_oppose = 0.623
    rate_support = 0.683
    n_oppose = 212
    n_support = 312
    z_stat = -1.42
    p_val = 0.155

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    # Single neutral color since story is "no difference"
    bar_color = "#5b8db5"  # muted blue
    x = np.arange(2)
    rates = [rate_oppose * 100, rate_support * 100]
    bars = ax.bar(x, rates, width=0.55, color=bar_color,
                  edgecolor="white", linewidth=1.5)

    # Percentage labels above bars
    for xi, rate in zip(x, rates):
        ax.text(xi, rate + 1.5, f"{rate:.1f}%",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#1f3a5f")

    # X-tick labels (two-line)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"≥ 1 opposing\ncommenter\n(n = {n_oppose})",
         f"≥ 1 supporting\ncommenter\n(n = {n_support})"],
        fontsize=10,
    )
    ax.set_ylabel("Revision rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.6)
    ax.set_axisbelow(True)

    # Title + statistical subtitle
    fig.suptitle("Finding 2: support vs opposition does not differentiate "
                 "revision outcomes",
                 fontsize=11, y=0.99, fontweight="bold")
    ax.set_title(
        f"Two-proportion $z$ = {z_stat:.2f}, $p$ = {p_val:.3f}; "
        f"6.0-pp difference within sampling noise",
        fontsize=9.5, style="italic", color="#444", pad=8,
    )

    _save(fig, "fig_f2_directional")
    plt.close(fig)


# ===========================================================================
# Figure F3-PRE — F3 pre-audit vs audit-corrected side-by-side comparison
# ===========================================================================
def build_fig_f3_audit_comparison() -> None:
    """Side-by-side stacked-bar comparison of F3 contingency before and
    after the §6.5 audit. Demonstrates that the audit *corroborated* the
    directional finding and *increased* the effect size, rather than
    weakening it.

    Pre-audit cells (43, 15, 28, 30) → OR=3.07, p=0.007, n=116
    Audit-corrected (13, 2, 57, 43)   → OR=4.90, p=0.044, n=115

    Visual style matches Yue's Image 1 (vertical stacked bars with
    org-majority share, percentage labels in white, light blue / dark
    blue palette)."""
    print("\n[fig_f3_audit_compare] pre vs audit-corrected", file=sys.stderr)

    # Cells: (SE-org, SE-noOrg, MO-org, MO-noOrg)
    pre = (43, 15, 28, 30)
    post = (13, 2, 57, 43)

    def _pct_org(cells):
        se_org, se_no, mo_org, mo_no = cells
        se_total = se_org + se_no
        mo_total = mo_org + mo_no
        return (se_org / se_total, se_total, mo_org / mo_total, mo_total)

    pre_se_org, pre_se_n, pre_mo_org, pre_mo_n = _pct_org(pre)
    post_se_org, post_se_n, post_mo_org, post_mo_n = _pct_org(post)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4),
                             gridspec_kw=dict(wspace=0.32))
    plt.subplots_adjust(bottom=0.18, top=0.85)

    ORG_COLOR = "#1f4e79"      # dark blue
    NONORG_COLOR = "#a8c8e4"   # light blue

    def _draw_panel(ax, se_org_frac, se_n, mo_org_frac, mo_n,
                    title, or_val, p_val):
        x = np.arange(2)
        org_pct = [se_org_frac * 100, mo_org_frac * 100]
        nonorg_pct = [(1 - se_org_frac) * 100, (1 - mo_org_frac) * 100]

        ax.bar(x, org_pct, width=0.55, color=ORG_COLOR,
               edgecolor="white", linewidth=1.5,
               label="Organizational-majority")
        ax.bar(x, nonorg_pct, width=0.55, bottom=org_pct, color=NONORG_COLOR,
               edgecolor="white", linewidth=1.5,
               label="Not organizational-majority")

        # Org-majority percentage labels (centered inside dark band)
        for xi, (op, np_) in enumerate(zip(org_pct, nonorg_pct)):
            ax.text(xi, op / 2, f"{op:.0f}%",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")
            # Non-org label inside light band
            ax.text(xi, op + np_ / 2, f"{np_:.0f}%",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#1f3a5f")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"SURVIVED-edited\n(n = {se_n})",
             f"MODIFIED\n(n = {mo_n})"],
            fontsize=10,
        )
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Share of engaged obligations (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc",
                alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{title}\nFisher's exact OR = {or_val:.2f}, $p$ = {p_val:.3f}",
            fontsize=11, pad=10, fontweight="bold",
        )

    # Pre-audit panel
    _draw_panel(axes[0], pre_se_org, pre_se_n, pre_mo_org, pre_mo_n,
                f"Pre-audit (n = {pre_se_n + pre_mo_n})", 3.07, 0.007)
    # Audit-corrected panel
    _draw_panel(axes[1], post_se_org, post_se_n, post_mo_org, post_mo_n,
                f"Audit-corrected (n = {post_se_n + post_mo_n})", 4.90, 0.044)

    # Single legend at bottom — pulled below the x-tick labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, fontsize=10.5)

    fig.suptitle(
        "Finding 3: blind dual-annotator audit corroborates the "
        "directional finding (OR increased from 3.07 to 4.90)",
        fontsize=11.5, y=0.97, fontweight="bold",
    )

    _save(fig, "fig_f3_audit_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--only", type=str, default="all",
        help="Comma-separated subset to regenerate: fig1, fig2, fig3, "
             "fig4, fig_f2, fig_f3_compare (or 'all'). Useful when one "
             "figure is blocked by data availability and you want to "
             "refresh the others.",
    )
    args = ap.parse_args()
    _set_rcparams()

    all_figs = {"fig1", "fig2", "fig3", "fig4", "fig_f2", "fig_f3_compare"}
    which = (set(args.only.split(",")) if args.only != "all" else all_figs)
    print(f"[paper-figures] regenerating: {sorted(which)}", file=sys.stderr)
    print(f"[paper-figures] output dir: {FIG_DIR}", file=sys.stderr)

    fig2_ok = True
    fig3_ok = True
    if "fig1" in which:
        build_fig1_pipeline()
    if "fig2" in which:
        fig2_ok = build_fig2_finding1_loo()
    if "fig3" in which:
        fig3_ok = build_fig3_confusion()
    if "fig4" in which:
        build_fig4_finding3_mosaic()
    if "fig_f2" in which:
        build_fig_f2_directional()
    if "fig_f3_compare" in which:
        build_fig_f3_audit_comparison()

    print("\n[paper-figures] DONE", file=sys.stderr)
    if not fig2_ok:
        print("  WARNING: fig2 (Finding 1 LOO) was not regenerated; the "
              "existing PDF on disk is preserved.", file=sys.stderr)
    if not fig3_ok:
        print("  WARNING: fig3 (confusion matrix) was not regenerated; "
              "the existing PDF on disk is preserved.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
