"""
29_f1_excluding_new.py — Finding 1 rerun without the final-only NEW obligations.

Why: NEW obligations exist only in the final rule, so "revised between
proposed and final" is undefined for them. code/22 counts them as
"not revised" (was_revised = SE/MOD/DROPPED), inside n = 12,730.
This script reports Finding 1 both ways so the paper can use the
proposed-side denominator (12,243) if the numbers move.

Reuses the data loading and FE-logit code from
code/22_clustered_logistic_f1_f3.py, so inputs and conventions are identical.

Usage (from the repo root, on the machine with the production data):
    python3 code/29_f1_excluding_new.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "c22", HERE / "22_clustered_logistic_f1_f3.py")
c22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c22)

pd = c22.pd
stats = c22.stats

REVISED = ["SURVIVED-edited", "MODIFIED", "DROPPED"]


def f1_summary(df: "pd.DataFrame", label: str) -> None:
    d = df.copy()
    d["high"] = d["n_addressing"] >= 5
    d["rev"] = d["outcome_state"].isin(REVISED)
    ct = pd.crosstab(d["high"], d["rev"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = int(ct.values.sum())
    v = (chi2 / n) ** 0.5
    rate_hi = d.loc[d["high"], "rev"].mean() * 100
    rate_lo = d.loc[~d["high"], "rev"].mean() * 100
    t1 = c22.test_1_f1_clustered(df)
    print(f"\n[{label}]  n = {n:,}  (high-engagement n = {int(d['high'].sum()):,})")
    print(f"  revised: high {rate_hi:.1f}%  vs  low {rate_lo:.1f}%  "
          f"(gap {rate_hi - rate_lo:.1f} pp)")
    print(f"  chi2 = {chi2:.2f}, p = {p:.2e}, Cramer's V = {v:.3f}")
    print(f"  docket-FE logit, cluster-robust SE: OR = {t1['OR']:.2f} "
          f"[{t1['ci_lo']:.2f}, {t1['ci_hi']:.2f}], p = {t1['p']:.3f}, "
          f"dockets = {t1['k_dockets']}")


AUDIT_OUTCOMES = "data/processed/path_a_obligation_outcomes_audit_corrected.parquet"


def loo_summary(df: "pd.DataFrame", label: str) -> None:
    """Leave-one-docket-out chi-square, same test as the headline."""
    d = df.copy()
    d["high"] = d["n_addressing"] >= 5
    d["rev"] = d["outcome_state"].isin(REVISED)
    rows = []
    for dk in sorted(d["docket_id"].dropna().unique()):
        sub = d[d["docket_id"] != dk]
        ct = pd.crosstab(sub["high"], sub["rev"])
        if ct.shape != (2, 2):
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append((dk, chi2, p))
    chis = [r[1] for r in rows]
    ps = [r[2] for r in rows]
    print(f"\n[{label}] leave-one-docket-out: "
          f"{sum(p < 0.05 for p in ps)}/{len(rows)} with p < 0.05; "
          f"chi2 range [{min(chis):.2f}, {max(chis):.2f}]; max p = {max(ps):.2e}")


def main() -> None:
    addressed, df_outcomes, comments = c22._load_data(c22.DEFAULT_OUTCOMES)
    out = c22._build_per_obligation_aggregate(addressed, df_outcomes, comments)

    new = out[out["outcome_state"] == "NEW"]
    print(f"NEW obligations: {len(new):,}")
    print(f"  with >=1 addressing commenter: {int((new['n_addressing'] >= 1).sum()):,}")
    print(f"  with >=5 addressing commenters (high-engagement): "
          f"{int((new['n_addressing'] >= 5).sum()):,}")

    f1_summary(out, "AS PUBLISHED: all obligations, NEW counted as not revised")
    f1_summary(out[out["outcome_state"] != "NEW"],
               "PROPOSED-SIDE ONLY: NEW excluded")

    # Leave-one-docket-out check (paper: "29/29"), both definitions.
    loo_summary(out, "AS PUBLISHED")
    loo_summary(out[out["outcome_state"] != "NEW"], "PROPOSED-SIDE ONLY")

    # Audit-corrected labels (paper: F1 chi2 = 22.93), both definitions.
    if pathlib.Path(AUDIT_OUTCOMES).exists():
        df_audit = pd.read_parquet(AUDIT_OUTCOMES)
        out_a = c22._build_per_obligation_aggregate(addressed, df_audit, comments)
        f1_summary(out_a, "AUDIT-CORRECTED, as published")
        f1_summary(out_a[out_a["outcome_state"] != "NEW"],
                   "AUDIT-CORRECTED, NEW excluded")
    else:
        print(f"\n(skipped audit-corrected check: {AUDIT_OUTCOMES} not found)")


if __name__ == "__main__":
    sys.exit(main())
