"""
19_finding3_loo_sensitivity.py — Leave-one-docket-out sensitivity for Finding 3.

Finding 3 (paper §6.4): on the engaged-obligation subset, organizational-majority
commenter composition co-occurs with editorial outcomes (SURVIVED-edited) more
often than with substantive modification (MODIFIED). Baseline test:

    Fisher's exact: OR = 3.07, 95% CI [1.41, 6.71], p = 0.007, n = 116
    2x2 cells: SE-org=43, SE-not-org=15, MO-org=28, MO-not-org=30

This script tests whether the OR + significance + CI are robust to dropping any
single docket. Addresses Yue's docket-clustering concern from
notes/2026-05-15_yue_paper_spot_review.md, specifically the E15 fuel rule
(EPA-HQ-OAR-2018-0775).

REQUIRES pyarrow (cannot run in the Claude sandbox; run locally on your Mac).

Inputs:
  data/processed/stage4/*__matches.parquet     (per-anchor obligation-comment pairs)
  data/processed/comments/EPA_YYYY.parquet     (per-comment titles → submitter_kind)
  data/processed/responsiveness_analysis.csv   (per-obligation outcome_state)

Output: prints baseline + all-LOO summary + most-influential dockets + E15 check.
        Optional --csv-out writes data/processed/finding3_loo_sensitivity.csv

Usage:
  python3 code/19_finding3_loo_sensitivity.py
  python3 code/19_finding3_loo_sensitivity.py --csv-out
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from math import exp, log, lgamma, sqrt
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas required. pip install pandas pyarrow")

# NOTE: this script uses the SIMPLE title-pattern classifier defined inline
# below (matches code/20_finding3_analysis.py), NOT code/lib/submitter_classification.py.
# The simple classifier is what produces the §6.4 paper-reported cells.

STAGE4_GLOB = "data/processed/stage4/*.parquet"
COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"
RESPONSIVENESS_CSV = Path("data/processed/responsiveness_analysis.csv")
OUTPUT_CSV = Path("data/processed/finding3_loo_sensitivity.csv")


def docket_of_obligation(obl_id: str) -> str | None:
    m = re.match(r"^(EPA-[A-Z0-9-]+?)__(?:proposed|final)__", obl_id)
    return m.group(1) if m else None


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    """(OR, Wald 95% CI low, high, two-sided Fisher exact p)."""
    if min(a, b, c, d) <= 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d
    OR = (a_ * d_) / (b_ * c_)
    se = sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    log_or = log(OR)
    ci_lo, ci_hi = exp(log_or - 1.96 * se), exp(log_or + 1.96 * se)

    def lc(n, k):
        if k < 0 or k > n: return float("-inf")
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    n1, n2, k = a + b, c + d, a + c
    log_p_obs = lc(n1, a) + lc(n2, k - a) - lc(n1 + n2, k)
    a_min, a_max = max(0, k - n2), min(n1, k)
    p_two = sum(exp(lc(n1, ai) + lc(n2, k - ai) - lc(n1 + n2, k))
                for ai in range(a_min, a_max + 1)
                if lc(n1, ai) + lc(n2, k - ai) - lc(n1 + n2, k) <= log_p_obs + 1e-10)
    return OR, ci_lo, ci_hi, p_two


def compute_finding3(engaged: pd.DataFrame) -> dict:
    """2x2: {SURVIVED-edited, MODIFIED} × {org-majority, not-org-majority}."""
    se = engaged[engaged["outcome_state"] == "SURVIVED-edited"]
    mo = engaged[engaged["outcome_state"] == "MODIFIED"]
    a = int((se["org_majority"] == 1).sum())
    b = int((se["org_majority"] == 0).sum())
    c = int((mo["org_majority"] == 1).sum())
    d = int((mo["org_majority"] == 0).sum())
    n = a + b + c + d
    if n == 0:
        return dict(a=a, b=b, c=c, d=d, n=n,
                    OR=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p=float("nan"))
    OR, lo, hi, p = fisher_exact_2x2(a, b, c, d)
    return dict(a=a, b=b, c=c, d=d, n=n, OR=OR, ci_lo=lo, ci_hi=hi, p=p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", action="store_true")
    args = ap.parse_args()

    print("=== Step 1: load per-anchor matches parquets ===")
    matches_files = sorted(glob.glob(STAGE4_GLOB))
    if not matches_files:
        sys.exit(f"ERROR: no files match {STAGE4_GLOB}")
    print(f"  Found {len(matches_files)} matches parquets")

    dfs = []
    for f in matches_files:
        d = pd.read_parquet(f, columns=["comment_id", "obligation_id", "addressed", "stance"])
        dfs.append(d)
    matches = pd.concat(dfs, ignore_index=True)
    addressed = matches[matches["addressed"] == True].copy()  # noqa: E712
    print(f"  Total pairs: {len(matches):,}; addressed=True: {len(addressed):,}")

    print("\n=== Step 2: build comment_id → submitter_kind (SIMPLE classifier, paper-canonical) ===")
    # Uses the SAME classifier as code/20_finding3_analysis.py, which reproduces
    # the §6.4 paper-reported cells (43, 15, 28, 30) and PROJECT_FACTS distribution
    # 71.4% individual / 28.6% organizational. This is a permissive title-pattern
    # heuristic distinct from code/lib/submitter_classification.py (the canonical
    # lib classifier built for the stratified-sampling step).
    comments_files = sorted(glob.glob(COMMENTS_GLOB))
    if not comments_files:
        sys.exit(f"ERROR: no files match {COMMENTS_GLOB}")
    title_dfs = []
    for f in comments_files:
        d = pd.read_parquet(f, columns=["document_id", "title"])
        title_dfs.append(d)
    titles = pd.concat(title_dfs, ignore_index=True).drop_duplicates("document_id")
    titles = titles.rename(columns={"document_id": "comment_id"})
    # SIMPLE three-pattern classifier (this is what produced §6.4's cells in the
    # original analysis): individual if title contains any of those three phrases;
    # organizational otherwise. No "other" category.
    titles["submitter_kind"] = (
        titles["title"].fillna("").str.contains(
            "submitted by|on behalf|written by", case=False, regex=True
        ).map({True: "individual", False: "organizational"})
    )
    kind_counts = titles["submitter_kind"].value_counts(dropna=False)
    print(f"  Loaded {len(titles):,} comments")
    print(f"  Submitter kind distribution: {kind_counts.to_dict()}")
    total = kind_counts.sum()
    if total > 0:
        print(f"  Org rate: {100*kind_counts.get('organizational', 0)/total:.1f}% "
              f"(PROJECT_FACTS §2 reports 28.6%)")

    print("\n=== Step 3: aggregate per obligation ===")
    addr = addressed.merge(titles[["comment_id", "submitter_kind"]], on="comment_id", how="left")
    addr["org_flag"] = (addr["submitter_kind"] == "organizational").astype(int)
    addr["ind_flag"] = (addr["submitter_kind"] == "individual").astype(int)
    # drop pairs with 'other' submitter_kind (matches the original analysis convention)
    addr_clean = addr[addr["submitter_kind"].isin(["organizational", "individual"])].copy()

    per_obl = (
        addr_clean.groupby("obligation_id")
        .agg(n_addr=("comment_id", "size"),
             n_org=("org_flag", "sum"),
             n_ind=("ind_flag", "sum"))
        .reset_index()
    )
    per_obl["org_share"] = per_obl["n_org"] / per_obl["n_addr"]
    # Test multiple definitions of "org-majority" to find which reproduces paper §6.4 cells
    per_obl["org_majority_strict"] = (per_obl["org_share"] > 0.5).astype(int)  # >50% org
    per_obl["org_majority_any"] = (per_obl["n_org"] >= 1).astype(int)          # ≥1 org addresser
    per_obl["org_majority_third"] = (per_obl["org_share"] >= 1/3).astype(int)  # ≥1/3 org
    per_obl["org_majority"] = per_obl["org_majority_strict"]  # default for downstream
    per_obl["docket"] = per_obl["obligation_id"].map(docket_of_obligation)

    resp = pd.read_csv(RESPONSIVENESS_CSV)
    per_obl = per_obl.merge(resp[["obligation_id", "outcome_state"]], on="obligation_id", how="left")

    engaged = per_obl[per_obl["n_addr"] >= 5].copy()
    print(f"  Engaged obligations (n_addr ≥ 5, org/ind only): {len(engaged):,}")
    print(f"  By outcome: {engaged['outcome_state'].value_counts().to_dict()}")

    print(f"\n=== Step 4: probe THREE definitions of 'org-majority' to find paper cells ===")
    print(f"  Paper §6.4 reports cells (43, 15, 28, 30) on the SE/MO × org-maj contrast.")
    for col, label in [
        ("org_majority_strict", "STRICT: org_share > 0.5"),
        ("org_majority_any", "ANY:    n_org >= 1"),
        ("org_majority_third", "THIRD:  org_share >= 1/3"),
    ]:
        engaged["org_majority"] = engaged[col]
        b = compute_finding3(engaged)
        cells = (b["a"], b["b"], b["c"], b["d"])
        match = cells == (43, 15, 28, 30)
        marker = "  <<<<<< MATCHES PAPER CELLS" if match else ""
        print(f"  [{label}]")
        print(f"    Cells: SE-org={b['a']}, SE-not-org={b['b']}, MO-org={b['c']}, MO-not-org={b['d']} (n={b['n']}){marker}")
        print(f"    OR = {b['OR']:.3f}, 95% CI [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}], Fisher p = {b['p']:.4g}")

    # Pick whichever definition matches the paper cells (or default to strict if none match)
    matched_col = None
    for col in ["org_majority_strict", "org_majority_any", "org_majority_third"]:
        engaged["org_majority"] = engaged[col]
        b = compute_finding3(engaged)
        if (b["a"], b["b"], b["c"], b["d"]) == (43, 15, 28, 30):
            matched_col = col
            break
    if matched_col:
        print(f"\n  Using {matched_col!r} for LOO sensitivity (matches paper cells)")
        engaged["org_majority"] = engaged[matched_col]
    else:
        print(f"\n  No definition reproduces paper cells exactly. Using STRICT for LOO; result is illustrative only.")
        engaged["org_majority"] = engaged["org_majority_strict"]
    base = compute_finding3(engaged)

    print(f"\n=== Step 5: LOO sensitivity ===")
    dockets = sorted(engaged["docket"].dropna().unique())
    print(f"  Dockets contributing to engaged subset: {len(dockets)}")

    rows = []
    for d in dockets:
        rest = engaged[engaged["docket"] != d]
        rows.append({"dropped_docket": d,
                     "n_dropped": len(engaged) - len(rest),
                     **compute_finding3(rest)})
    out = pd.DataFrame(rows)

    print(f"  OR:    min={out['OR'].min():.3f}, median={out['OR'].median():.3f}, max={out['OR'].max():.3f}")
    print(f"  CI lo: min={out['ci_lo'].min():.3f}, max={out['ci_lo'].max():.3f}")
    print(f"  CI hi: min={out['ci_hi'].min():.3f}, max={out['ci_hi'].max():.3f}")
    print(f"  p:     min={out['p'].min():.4g}, max={out['p'].max():.4g}")
    print(f"  Conditions with p < 0.05: {(out['p'] < 0.05).sum()}/{len(out)}")
    print(f"  Conditions with CI lower bound > 1.0: {(out['ci_lo'] > 1.0).sum()}/{len(out)}")

    sorted_out = out.copy()
    sorted_out["abs_swing"] = (sorted_out["OR"] - base["OR"]).abs()
    sorted_out = sorted_out.sort_values("abs_swing", ascending=False).head(8)
    print(f"\n=== Most-influential dockets (largest OR swing when removed) ===")
    print(f"{'Docket':<32} {'n_drop':>7} {'OR':>7} {'95% CI':>20} {'p':>9}")
    for _, r in sorted_out.iterrows():
        ci_str = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
        print(f"  {r['dropped_docket']:<30} {int(r['n_dropped']):>7} "
              f"{r['OR']:>7.3f} {ci_str:>20} {r['p']:>9.4g}")

    e15 = out[out["dropped_docket"] == "EPA-HQ-OAR-2018-0775"]
    if len(e15):
        e = e15.iloc[0]
        print(f"\n=== E15 fuel rule check (Yue's specific concern) ===")
        print(f"  Drop EPA-HQ-OAR-2018-0775 (n_obl_dropped = {int(e['n_dropped'])}):")
        print(f"    OR = {e['OR']:.3f}, 95% CI [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}], p = {e['p']:.4g}")
        survives = e["p"] < 0.05 and e["ci_lo"] > 1.0
        print(f"  -> Finding 3 {'SURVIVES' if survives else 'FAILS'} the E15 leave-out")
    else:
        print("\n=== E15 fuel rule note ===")
        print("  E15 (EPA-HQ-OAR-2018-0775) does NOT contribute to the engaged "
              "subset; no LOO needed.")

    if args.csv_out:
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
