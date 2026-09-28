"""
21_finding3_robustness_battery.py — Multi-operationalization sensitivity
battery for Finding 3 of the EAAMO paper (§6.4 robustness check).

Finding 3 (paper §6.4) reports, on the engaged-obligation subset
(n=116, ≥5 addressers, restricted to SURVIVED-edited vs MODIFIED):

    Fisher's exact: OR = 3.07, 95% CI [1.41, 6.71], p = 0.007
    2x2 cells: SE-org=43, SE-not-org=15, MO-org=28, MO-not-org=30

The baseline relies on a single permissive title-pattern classifier
that converts a comment title into a binary `organizational` / `individual`
label, then aggregates to `org_majority` per obligation at the >50%
threshold. Reviewers will hammer on the classifier choice and the
hard >50% knife-edge. This battery shows the finding is not an
artifact of either:

  Test 0 (baseline)   — replicates the canonical title-pattern
                        classifier and confirms it produces the
                        paper-reported cells. ABORTS if the
                        reproduction fails.
  Test 1 (attachment) — re-runs with a fully INDEPENDENT signal:
                        `is_attachment_only` (uploaded a PDF letterhead
                        with no inline text). No title text used at all.
  Test 2 (combined)   — `organizational` if EITHER signal fires (most
                        inclusive defensible operationalization).
  Test 3 (threshold)  — baseline classifier, but vary the org-share
                        threshold t ∈ {0.33, 0.40, 0.50, 0.60, 0.67}.
                        Tests whether the finding hinges on the
                        >50% knife-edge.
  Test 4 (CMH)        — Cochran-Mantel-Haenszel stratified by docket
                        (extracted from obligation_id). Controls
                        docket clustering directly while testing
                        whether the org-majority effect holds within
                        dockets. Reports common OR + p-value of
                        equal-odds (homogeneity) test.

Why this matters for §6.4: convergence across Tests 0/1/2 means two
independent operationalizations of "submitter is organizational" reach
the same directional + significance conclusion. Test 3 confirms the
finding is not knife-edge. Test 4 is the most principled response to
the docket-clustering concern Yue raised
(notes/2026-05-15_yue_paper_spot_review.md).

CLASSIFIER CONVENTIONS:
- Baseline title-pattern classifier (Test 0, Test 3, Test 4) is:
      title contains "submitted by" | "on behalf" | "written by" → individual
      otherwise                                                  → organizational
  Identical to code/20_finding3_analysis.py and code/19_finding3_loo_sensitivity.py.
- Attachment-only proxy (Test 1) is:
      is_attachment_only == True  → organizational
      else                        → individual
- Combined inclusive (Test 2) is:
      (title-pattern says organizational) OR (is_attachment_only) → organizational
      else                                                        → individual

The combined classifier in code/lib/submitter_classification.py is NOT used here
(it's calibrated for a different pipeline step and disagrees with the §6.4
convention on ~14% of comments — see §8 of the paper for the limitation framing).

REPRODUCTION RULE (team policy):
This script must be committed before its numbers are cited in §6.4 / Table 1.
Per the team's "commit analysis scripts" rule (the same convention surfaced
when code/20_finding3_analysis.py was discovered uncommitted).

Inputs (mirror code/20_finding3_analysis.py exactly):
  data/processed/stage4/*.parquet                  (Stage 4 obligation-comment pairs)
  data/processed/path_a_obligation_outcomes.parquet (Path A outcome states)
  data/processed/comments_augmented/EPA_*.parquet  (per-comment titles + is_attachment_only)

Usage:
  python3 code/21_finding3_robustness_battery.py

Output: printed comparison table. No file outputs.

Required packages: pandas, pyarrow, scipy (≥1.10 for stats.contingency.StratifiedTable)
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from math import exp, log, sqrt

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats.contingency import odds_ratio as scipy_odds_ratio  # noqa: F401  (sanity import)
except ImportError as e:
    sys.exit(f"ERROR: {e}. Required: pip install pandas pyarrow 'scipy>=1.10' statsmodels")

# Test 4 (CMH stratified by docket) uses `StratifiedTable` for the
# Mantel-Haenszel common OR + Wald CI + Tarone homogeneity test. This
# class lives in **statsmodels**, not in scipy. (Some early drafts of
# the task spec wrote `scipy.stats.contingency.StratifiedTable`, but
# scipy.stats.contingency only exposes `odds_ratio` for 2x2 tables;
# the stratified version has always been a statsmodels class.)
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
except ImportError:
    sys.exit("ERROR: statsmodels.stats.contingency_tables.StratifiedTable "
             "not available. Task M's Test 4 (CMH stratified by docket) "
             "requires statsmodels. Run: pip install statsmodels.")


DEFAULT_OUTCOMES = "data/processed/path_a_obligation_outcomes.parquet"
STAGE4_GLOB = "data/processed/stage4/*.parquet"
COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"


# ---------------------------------------------------------------------------
# Inline 2x2 helpers — single-file, no imports from sibling scripts
# (mirrors code/19_finding3_loo_sensitivity.py:fisher_exact_2x2 conventions
# so the Wald CI matches whatever the LOO sensitivity script reports).
# ---------------------------------------------------------------------------
def fisher_exact_2x2(a: int, b: int, c: int, d: int
                    ) -> tuple[float, float, float, float]:
    """Return (OR, Wald 95% CI low, Wald 95% CI high, two-sided Fisher p).

    Cell convention: a=SE-org, b=SE-noOrg, c=MO-org, d=MO-noOrg
    matching the §6.4 paper-reported (43, 15, 28, 30) layout.
    """
    # Haldane–Anscombe continuity correction when any cell is zero.
    if min(a, b, c, d) <= 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = float(a), float(b), float(c), float(d)
    odds_ratio = (a_ * d_) / (b_ * c_)
    se_log = sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    log_or = log(odds_ratio)
    ci_lo = exp(log_or - 1.96 * se_log)
    ci_hi = exp(log_or + 1.96 * se_log)
    # scipy.stats.fisher_exact handles the exact p-value (uses lgamma
    # internally); reuse it to avoid duplicating the hypergeometric sum.
    _, p_two = stats.fisher_exact([[a, b], [c, d]])
    return odds_ratio, ci_lo, ci_hi, float(p_two)


def docket_of_obligation(obl_id: str) -> str | None:
    """Reverse stage4_embedding_prefilter.obligation_id_of:
    'EPA-…-NNNN__{proposed|final}__cidx_split' → docket."""
    if not isinstance(obl_id, str):
        return None
    m = re.match(r"^(EPA-[A-Z0-9-]+?)__(?:proposed|final)__", obl_id)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_data(outcomes_path: str):
    """Return (addressed, df_outcomes, comments) where:
      - addressed: Stage 4 pairs filtered to addressed==True
      - df_outcomes: per-obligation outcome_state
      - comments: per-comment ['document_id', 'title', 'is_attachment_only']
    Mirrors code/20_finding3_analysis.py's loading + adds is_attachment_only
    (needed for Test 1)."""
    stage4_files = sorted(glob.glob(STAGE4_GLOB))
    if not stage4_files:
        sys.exit(f"ERROR: no parquets at {STAGE4_GLOB} — Stage 4 outputs are "
                 "gitignored; this script must be run on a workstation that "
                 "has the production data.")
    if not glob.glob(outcomes_path) and not glob.glob(outcomes_path + ".cfr_restricted.bak"):
        sys.exit(f"ERROR: outcomes parquet not found at {outcomes_path} — "
                 "run code/17_compute_obligation_outcomes.py --all first.")

    print(f"Loading Stage 4 matches ({len(stage4_files)} parquets) + "
          f"outcomes from {outcomes_path} ...", file=sys.stderr)
    df_matches = pd.concat([pd.read_parquet(p) for p in stage4_files])
    df_outcomes = pd.read_parquet(outcomes_path)
    addressed = df_matches[df_matches["addressed"] == True]  # noqa: E712

    comments_dfs = []
    for year_pq in sorted(glob.glob(COMMENTS_GLOB)):
        cdf = pd.read_parquet(
            year_pq, columns=["document_id", "title", "is_attachment_only"],
        )
        comments_dfs.append(cdf)
    if not comments_dfs:
        sys.exit(f"ERROR: no parquets at {COMMENTS_GLOB}")
    comments = pd.concat(comments_dfs, ignore_index=True)
    print(f"Loaded: {len(addressed):,} addressed pairs; "
          f"{len(df_outcomes):,} outcome rows; "
          f"{len(comments):,} comments", file=sys.stderr)
    return addressed, df_outcomes, comments


def _apply_baseline_title_classifier(comments: pd.DataFrame) -> pd.DataFrame:
    """Add `submitter_kind` column via the §6.4 title-pattern classifier.
    Convention identical to code/20_finding3_analysis.py."""
    is_individual = comments["title"].fillna("").str.contains(
        "submitted by|on behalf|written by", case=False, regex=True,
    )
    comments = comments.copy()
    comments["submitter_kind_title"] = is_individual.map(
        {True: "individual", False: "organizational"}
    )
    return comments


def _build_engaged_subset(
    addressed: pd.DataFrame,
    df_outcomes: pd.DataFrame,
    comments: pd.DataFrame,
    submitter_kind_col: str,
) -> pd.DataFrame:
    """Aggregate addressed pairs to per-obligation org/ind counts using
    the chosen `submitter_kind_col`, merge with outcomes, restrict to
    n_addr >= 5 AND outcome_state ∈ {SURVIVED-edited, MODIFIED}."""
    addr_with_kind = addressed.merge(
        comments[["document_id", submitter_kind_col]],
        left_on="comment_id", right_on="document_id", how="left",
    )
    agg = addr_with_kind.groupby("obligation_id").agg(
        n_addr=("comment_id", "nunique"),
        n_org=(submitter_kind_col,
               lambda s: (s == "organizational").sum()),
        n_ind=(submitter_kind_col,
               lambda s: (s == "individual").sum()),
    ).reset_index()
    agg["org_share"] = agg["n_org"] / (
        (agg["n_org"] + agg["n_ind"]).replace(0, 1)
    )
    merged = df_outcomes.merge(agg, on="obligation_id", how="left")
    high = merged[merged["n_addr"] >= 5].copy()
    subset = high[high["outcome_state"].isin(
        ["MODIFIED", "SURVIVED-edited"])].copy()
    return subset


def _cells_or_ci_p(subset: pd.DataFrame, threshold: float = 0.5,
                   *, rule: str = "share_gt"
                  ) -> tuple[int, int, int, int, float, float, float, float]:
    """Compute the 2x2 cells + Fisher OR/CI/p with one of two
    aggregation rules. Returns (a=SE-org, b=SE-noOrg, c=MO-org,
    d=MO-noOrg, OR, ci_lo, ci_hi, fisher_p).

    Aggregation rules:
      rule="share_gt"  (default) — org_majority := org_share > threshold.
                       Used for Tests 0/2/3.
      rule="any"       — org_majority := n_org > 0 (any-org-presence).
                       Used for Test 1 — attachment-only is a sparse
                       signal (~10-15% of comments) so the strict >50%
                       rule never fires for it. The any-presence rule is
                       the conservative-classifier sensitivity already
                       used elsewhere in §6.4.
    """
    if rule == "share_gt":
        org_indicator = subset["org_share"] > threshold
    elif rule == "any":
        org_indicator = subset["n_org"] > 0
    else:
        raise ValueError(f"unknown rule: {rule!r}")
    se = subset["outcome_state"] == "SURVIVED-edited"
    mo = subset["outcome_state"] == "MODIFIED"
    a = int((se & org_indicator).sum())
    b = int((se & ~org_indicator).sum())
    c = int((mo & org_indicator).sum())
    d = int((mo & ~org_indicator).sum())
    OR, ci_lo, ci_hi, p_two = fisher_exact_2x2(a, b, c, d)
    return a, b, c, d, OR, ci_lo, ci_hi, p_two


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_test_0_baseline(addressed, df_outcomes, comments):
    """Replicate the canonical §6.4 result. Aborts if it doesn't match."""
    comments = _apply_baseline_title_classifier(comments)
    subset = _build_engaged_subset(
        addressed, df_outcomes, comments, "submitter_kind_title",
    )
    n = len(subset)
    a, b, c, d, OR, ci_lo, ci_hi, p = _cells_or_ci_p(subset, 0.5)
    expected = (43, 15, 28, 30)
    cells_match = (a, b, c, d) == expected
    or_match = abs(OR - 3.07) < 0.05
    p_match = abs(p - 0.007) < 0.005
    print(f"[Test 0 baseline] n={n}, cells=({a}, {b}, {c}, {d}), "
          f"OR={OR:.3f}, 95% CI=[{ci_lo:.2f}, {ci_hi:.2f}], p={p:.4f}",
          file=sys.stderr)
    if not (cells_match and or_match and p_match):
        print(f"\n  DIAGNOSTIC: baseline reproduction failed.", file=sys.stderr)
        print(f"  Expected cells: {expected}; got: ({a}, {b}, {c}, {d})",
              file=sys.stderr)
        print(f"  Expected OR≈3.07, p≈0.007; got OR={OR:.3f}, p={p:.4f}",
              file=sys.stderr)
        print(f"  Subset n: {n} (paper: 116)", file=sys.stderr)
        print(f"  Check that you're running against the same outcomes "
              f"parquet code/20_finding3_analysis.py uses (the canonical "
              f"text-only classification, not the cfr_restricted.bak).",
              file=sys.stderr)
        sys.exit(2)
    return subset, comments, ("Baseline (title patterns)", n, a, b, c, d,
                              OR, ci_lo, ci_hi, p)


def run_test_1_attachment(addressed, df_outcomes, comments):
    """Submitter_kind from is_attachment_only ALONE — fully independent
    of any title text. Uses the ANY-org-presence aggregation rule
    (`n_org > 0`) rather than `org_share > 0.5` because attachment-only
    is sparse (~10-15% of comments); the strict majority rule never
    fires under this classifier and yields degenerate cells
    (paper-side run produced (0, 58, 0, 58) before the fix). The
    any-presence rule matches the conservative-classifier sensitivity
    elsewhere in §6.4."""
    comments = comments.copy()
    comments["submitter_kind_att"] = comments["is_attachment_only"].map(
        {True: "organizational", False: "individual"}
    ).fillna("individual")
    subset = _build_engaged_subset(
        addressed, df_outcomes, comments, "submitter_kind_att",
    )
    n = len(subset)
    a, b, c, d, OR, ci_lo, ci_hi, p = _cells_or_ci_p(subset, rule="any")
    return ("Test 1: attachment-only (any-org rule)", n, a, b, c, d,
            OR, ci_lo, ci_hi, p)


def run_test_2_combined(addressed, df_outcomes, comments):
    """Submitter_kind = organizational if (title-pattern says org) OR
    is_attachment_only. Most inclusive defensible operationalization."""
    is_individual_title = comments["title"].fillna("").str.contains(
        "submitted by|on behalf|written by", case=False, regex=True,
    )
    title_org = ~is_individual_title
    is_att = comments["is_attachment_only"].fillna(False).astype(bool)
    comments = comments.copy()
    comments["submitter_kind_combined"] = (title_org | is_att).map(
        {True: "organizational", False: "individual"}
    )
    subset = _build_engaged_subset(
        addressed, df_outcomes, comments, "submitter_kind_combined",
    )
    n = len(subset)
    a, b, c, d, OR, ci_lo, ci_hi, p = _cells_or_ci_p(subset, 0.5)
    return "Test 2: combined inclusive", n, a, b, c, d, OR, ci_lo, ci_hi, p


def run_test_3_threshold(subset_baseline: pd.DataFrame, threshold: float):
    """Baseline title-pattern classifier; vary the org_share threshold."""
    n = len(subset_baseline)
    a, b, c, d, OR, ci_lo, ci_hi, p = _cells_or_ci_p(subset_baseline, threshold)
    return (f"Test 3: threshold = {threshold:.2f}", n, a, b, c, d,
            OR, ci_lo, ci_hi, p)


def run_test_4_cmh(subset_baseline: pd.DataFrame):
    """Cochran-Mantel-Haenszel stratified by docket via
    `scipy.stats.contingency.StratifiedTable`. Drops strata with
    < 2 obs or any all-zero marginal (CMH is undefined for those).

    Returns (row_tuple, footer_dict) where:
      row_tuple matches the other tests' shape so it can be appended to
        the comparison table:
            (label, n, -1, -1, -1, -1, common_OR, ci_lo, ci_hi, p_common)
        Cells columns are sentinels (-1) — per-stratum cells don't fit
        a single 4-tuple.
      footer_dict carries the CMH-only extras printed below the table:
            {strata_used, strata_dropped, p_equal_odds, n_total}
    """
    subset = subset_baseline.copy()
    subset["org_majority"] = subset["org_share"] > 0.5
    subset["docket_id"] = subset["obligation_id"].map(docket_of_obligation)
    subset = subset.dropna(subset=["docket_id"])
    tables: list[np.ndarray] = []
    strata_used = 0
    strata_dropped = 0
    for _docket, g in subset.groupby("docket_id"):
        se_org = int(((g["outcome_state"] == "SURVIVED-edited")
                      & g["org_majority"]).sum())
        se_noo = int(((g["outcome_state"] == "SURVIVED-edited")
                      & ~g["org_majority"]).sum())
        mo_org = int(((g["outcome_state"] == "MODIFIED")
                      & g["org_majority"]).sum())
        mo_noo = int(((g["outcome_state"] == "MODIFIED")
                      & ~g["org_majority"]).sum())
        total = se_org + se_noo + mo_org + mo_noo
        if total < 2:
            strata_dropped += 1
            continue
        # CMH undefined when a marginal is zero in any stratum direction.
        if ((se_org + se_noo) == 0 or (mo_org + mo_noo) == 0
                or (se_org + mo_org) == 0 or (se_noo + mo_noo) == 0):
            strata_dropped += 1
            continue
        tables.append(np.array([[se_org, se_noo], [mo_org, mo_noo]]))
        strata_used += 1

    if not tables:
        row = ("Test 4: CMH (docket strata)", 0, -1, -1, -1, -1,
               float("nan"), float("nan"), float("nan"), float("nan"))
        footer = {"strata_used": 0, "strata_dropped": strata_dropped,
                  "p_equal_odds": float("nan"), "n_total": 0}
        return row, footer

    stratified = StratifiedTable(tables)
    # statsmodels.stats.contingency_tables.StratifiedTable surfaces:
    #   - oddsratio_pooled  (Mantel-Haenszel common OR)
    #   - logodds_pooled, logodds_pooled_se  (used to derive Wald CI)
    #   - test_null_odds()  (test of common OR == 1)
    #   - test_equal_odds() (Tarone homogeneity test across strata)
    common_or = float(stratified.oddsratio_pooled)
    log_or = float(stratified.logodds_pooled)
    log_or_se = float(stratified.logodds_pooled_se)
    ci_lo = float(np.exp(log_or - 1.96 * log_or_se))
    ci_hi = float(np.exp(log_or + 1.96 * log_or_se))
    p_common = float(stratified.test_null_odds().pvalue)
    p_equal = float(stratified.test_equal_odds().pvalue)
    n_total = int(sum(t.sum() for t in tables))
    row = ("Test 4: CMH (docket strata)", n_total,
           -1, -1, -1, -1,
           common_or, ci_lo, ci_hi, p_common)
    footer = {
        "strata_used": strata_used,
        "strata_dropped": strata_dropped,
        "p_equal_odds": p_equal,
        "n_total": n_total,
    }
    return row, footer


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _print_table(rows: list[tuple]) -> None:
    """rows[i] is a 10-tuple (label, n, a, b, c, d, OR, ci_lo, ci_hi, p).
    Standard tests fill a/b/c/d with the 2x2 cells. The CMH row uses
    sentinel cells (-1) and fills OR + ci_lo + ci_hi + p with the
    common-OR estimate; the equal-odds p-value and strata-used /
    strata-dropped breakdown are printed below the table via the
    footer dict returned by run_test_4_cmh."""
    headers = ("Test", "n",
               "Cells (SE-org/SE-noOrg/MO-org/MO-noOrg)",
               "OR", "95% CI", "p")
    fmt = "{:<38} {:>5} {:>40}  {:>5}  {:>14}  {:>7}"
    print("\n" + fmt.format(*headers))
    print("-" * 120)
    for r in rows:
        label, n, a, b, c, d, OR, ci_lo, ci_hi, p = r
        if label.startswith("Test 4:"):
            cells_str = "(see CMH footer below)"
            or_str = f"{OR:.2f}" if not np.isnan(OR) else "n/a"
            if np.isnan(ci_lo) or np.isnan(ci_hi):
                ci_str = "n/a"
            else:
                ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
            p_str = f"{p:.3f}" if not np.isnan(p) else "n/a"
            if not np.isnan(p) and p < 0.0005:
                p_str = f"{p:.2e}"
            print(fmt.format(label, n, cells_str, or_str, ci_str, p_str))
            continue
        cells_str = f"{a}/{b}/{c}/{d}"
        or_str = f"{OR:.2f}"
        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        p_str = f"{p:.3f}" if p >= 0.0005 else f"{p:.2e}"
        print(fmt.format(label, n, cells_str, or_str, ci_str, p_str))


def _print_cmh_footer(footer: dict) -> None:
    """Print the CMH-only extras: equal-odds (homogeneity) p-value +
    strata used/dropped breakdown. Lives below the comparison table
    because these don't fit the 4-column-cells row layout."""
    print("\nCMH (Test 4) details:")
    if footer.get("n_total", 0) == 0:
        print("  No usable strata — CMH not computed (every stratum had "
              "<2 obs or a zero marginal).")
        if footer.get("strata_dropped", 0):
            print(f"  Strata dropped: {footer['strata_dropped']}")
        return
    p_eq = footer.get("p_equal_odds", float("nan"))
    if np.isnan(p_eq):
        eq_str = "n/a (degenerate)"
    elif p_eq < 0.0005:
        eq_str = f"{p_eq:.2e}"
    else:
        eq_str = f"{p_eq:.3f}"
    print(f"  Strata used:        {footer.get('strata_used', 0)}")
    print(f"  Strata dropped:     {footer.get('strata_dropped', 0)}"
          "  (each <2 obs or a zero-marginal cell)")
    print(f"  Equal-odds p-value: {eq_str}"
          "  (test of OR homogeneity across dockets; high p ⇒"
          " no evidence ORs differ across strata)")
    print(f"  Total n (used):     {footer['n_total']}"
          "  (sum of contributions from non-degenerate strata)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--outcomes-path", default=DEFAULT_OUTCOMES,
        help=f"Path A outcomes parquet (default: {DEFAULT_OUTCOMES}). "
             "Mirrors code/20_finding3_analysis.py.",
    )
    args = ap.parse_args()

    addressed, df_outcomes, comments = _load_data(args.outcomes_path)

    rows: list[tuple] = []

    # --- Test 0 ---
    subset_baseline, comments_with_kind, baseline_row = run_test_0_baseline(
        addressed, df_outcomes, comments,
    )
    rows.append(baseline_row)

    # --- Test 1 ---
    rows.append(run_test_1_attachment(addressed, df_outcomes, comments))

    # --- Test 2 ---
    rows.append(run_test_2_combined(addressed, df_outcomes, comments))

    # --- Test 3 (5 threshold values) ---
    for t in (0.33, 0.40, 0.50, 0.60, 0.67):
        rows.append(run_test_3_threshold(subset_baseline, t))

    # --- Test 4 (CMH) ---
    cmh_row, cmh_footer = run_test_4_cmh(subset_baseline)
    rows.append(cmh_row)

    _print_table(rows)
    _print_cmh_footer(cmh_footer)

    # Interpretive footer.
    print("\nInterpretive notes:")
    print("  - Test 1 uses the ANY-org-presence rule (n_org > 0), not")
    print("    org_share > 0.5, because attachment-only is too sparse")
    print("    to majoritize (the strict rule yields degenerate cells).")
    print("  - Test 3 t=0.50 must equal Test 0 baseline by construction.")
    print("  - Tests 0/1/2 converging in direction + significance = the")
    print("    finding holds across two independent operationalizations.")
    print("  - Test 4 common OR is the docket-clustering-adjusted estimate;")
    print("    a non-significant equal-odds p suggests no statistical")
    print("    evidence of OR heterogeneity across dockets (i.e., the")
    print("    common OR is a reasonable summary).")


if __name__ == "__main__":
    main()
