"""
32_time_split.py - Do Findings 1-3 differ across time periods?

Responds to EAAMO 2026 reviewer EiG2 ("the analysis is entirely atemporal;
it's unclear whether the findings vary across time and between
administrations"). Descriptive only: each period has few rulemakings, so
per-period p-values are exploratory.

Two splits of the anchor rulemakings:
  YEAR CLUSTER   the paper's own sampling strata (2010-2012, 2013-2015,
                 2016-2018, 2019-2022), from data/processed/anchor_rules_locked.csv.
  ADMINISTRATION in office when the analyzed final rule was published
                 (Obama before 2017-01-20; Trump 2017-01-20 to 2021-01-19;
                 Biden from 2021-01-20). The analyzed final rule is the
                 Federal Register document the extraction pipeline used,
                 i.e. the largest-by-n_chars 'final' entry for the docket in
                 data/processed/federal_register_index.csv (same rule as
                 code/12's _pick_canonical_fr).

For each period it reports:
  Finding 1  proposed-side obligations (NEW excluded), high-engagement
             (>= 5 addressing commenters) vs less-addressed revision rates,
             the gap, and Fisher's exact p.
  Finding 2  revision rates for obligations with >= 1 opposing vs >= 1
             supporting commenter, and the two-proportion z-test p.
  Finding 3  2x2 cells (SE-org, SE-not, MO-org, MO-not) and Fisher's exact,
             with classifier-assigned and audit-corrected outcome labels.
It also tests whether the Finding 1 gap differs across periods: a
docket-fixed-effects logit with high-engagement x period interactions and
docket-clustered standard errors (joint Wald test of the interactions).

Reuses code/22's data loading so inputs and conventions match the paper.

Usage (repo root, on the machine with the production data):
    python3 code/32_time_split.py

Writes data/processed/robustness_time_split_summary.json.
Required packages: pandas, pyarrow, numpy, scipy, statsmodels.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ANCHORS_CSV = "data/processed/anchor_rules_locked.csv"
FR_INDEX_CSV = "data/processed/federal_register_index.csv"
AUDIT_OUTCOMES = "data/processed/path_a_obligation_outcomes_audit_corrected.parquet"
SUMMARY_JSON = "data/processed/robustness_time_split_summary.json"

REVISED = ["SURVIVED-edited", "MODIFIED", "DROPPED"]
HIGH = 5
YEAR_CLUSTERS = ["2010-2012", "2013-2015", "2016-2018", "2019-2022"]
ADMINS = ["Obama", "Trump", "Biden"]


# ---------------------------------------------------------------------------
# Period assignment (pure functions)
# ---------------------------------------------------------------------------
def administration(date_str: str | None) -> str | None:
    if not date_str:
        return None
    d = str(date_str)[:10]
    if d < "2017-01-20":
        return "Obama"
    if d < "2021-01-20":
        return "Trump"
    return "Biden"


def canonical_fr(fr_rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(docket_id, rule_type) -> largest-by-n_chars FR row (code/12 rule)."""
    best: dict[tuple[str, str], dict] = {}
    for r in fr_rows:
        key = (r.get("docket_id"), r.get("rule_type"))
        try:
            n = int(r.get("n_chars") or 0)
        except (TypeError, ValueError):
            n = 0
        if key not in best or n > best[key]["_n"]:
            best[key] = {**r, "_n": n}
    return best


def docket_periods(anchors_csv: str = ANCHORS_CSV,
                   fr_index_csv: str = FR_INDEX_CSV) -> dict[str, dict]:
    with open(anchors_csv, encoding="utf-8", newline="") as f:
        anchors = list(csv.DictReader(f))
    with open(fr_index_csv, encoding="utf-8", newline="") as f:
        fr = canonical_fr(list(csv.DictReader(f)))
    out = {}
    for a in anchors:
        d = a["docket_id"]
        prop = fr.get((d, "proposed"), {})
        fin = fr.get((d, "final"), {})
        out[d] = {
            "year_cluster": a.get("year_cluster"),
            "proposed_doc": prop.get("fr_doc_number"),
            "proposed_pub": prop.get("publication_date"),
            "final_doc": fin.get("fr_doc_number"),
            "final_pub": fin.get("publication_date"),
            "admin_final": administration(fin.get("publication_date")),
        }
    return out


# ---------------------------------------------------------------------------
# Statistics per period
# ---------------------------------------------------------------------------
def _fisher_p(a, b, c, d, stats) -> float:
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def f1_period(d, stats) -> dict:
    x = d[d["outcome_state"] != "NEW"]
    high = x["n_addressing"] >= HIGH
    rev = x["outcome_state"].isin(REVISED)
    a, b = int((high & rev).sum()), int((high & ~rev).sum())
    c, e = int((~high & rev).sum()), int((~high & ~rev).sum())
    rh = 100 * a / (a + b) if a + b else float("nan")
    rl = 100 * c / (c + e) if c + e else float("nan")
    return {"dockets": int(x["docket_id"].nunique()), "n": int(len(x)),
            "n_high": a + b, "rate_high": rh, "rate_low": rl,
            "gap_pp": rh - rl if a + b and c + e else float("nan"),
            "fisher_p": _fisher_p(a, b, c, e, stats) if a + b and c + e else float("nan")}


def f2_period(d, ztest) -> dict:
    rev = d["outcome_state"].isin(REVISED)
    opp, sup = d["n_opposing"] > 0, d["n_supporting"] > 0
    n_o, n_s = int(opp.sum()), int(sup.sum())
    r_o = 100 * rev[opp].mean() if n_o else float("nan")
    r_s = 100 * rev[sup].mean() if n_s else float("nan")
    p = float("nan")
    if n_o and n_s:
        _, p = ztest([int(rev[opp].sum()), int(rev[sup].sum())], [n_o, n_s])
    return {"n_opposing": n_o, "rate_opposing": r_o,
            "n_supporting": n_s, "rate_supporting": r_s, "z_p": float(p)}


def f3_period(d, c22) -> dict:
    h = d[(d["n_addressing"] >= HIGH)
          & d["outcome_state"].isin(["SURVIVED-edited", "MODIFIED"])]
    se = h["outcome_state"] == "SURVIVED-edited"
    om = h["org_share"] > 0.5
    a, b = int((se & om).sum()), int((se & ~om).sum())
    c, e = int((~se & om).sum()), int((~se & ~om).sum())
    if a + b + c + e == 0:
        return {"cells": [0, 0, 0, 0], "n": 0, "OR": float("nan"), "p": float("nan")}
    OR, _, _, p = c22.fisher_exact_2x2(a, b, c, e)
    return {"cells": [a, b, c, e], "n": a + b + c + e, "OR": OR, "p": p}


def interaction_test(d, col, smf) -> dict:
    """Docket-FE logit: revised ~ high + high x period + C(docket), clustered SE."""
    x = d[(d["outcome_state"] != "NEW") & d[col].notna()].copy()
    x["he"] = (x["n_addressing"] >= HIGH).astype(int)
    x["rev"] = x["outcome_state"].isin(REVISED).astype(int)
    levels = [p for p in sorted(x[col].unique()) if x.loc[x[col] == p, "he"].sum() > 0]
    if len(levels) < 2:
        return {"note": "fewer than two periods with high-engagement obligations"}
    ref, others = levels[0], levels[1:]
    terms = []
    for i, p in enumerate(others):
        name = f"he_x_{i}"
        x[name] = x["he"] * (x[col] == p).astype(int)
        terms.append(name)
    formula = "rev ~ he + " + " + ".join(terms) + " + C(docket_id)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.logit(formula, data=x).fit(
                cov_type="cluster", cov_kwds={"groups": x["docket_id"]},
                disp=False, maxiter=300)
            wt = m.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
        except Exception as e:  # noqa: BLE001
            return {"note": f"fit failed: {type(e).__name__}: {e}"}
    per = {ref: float(np.exp(m.params["he"]))}
    for i, p in enumerate(others):
        per[p] = float(np.exp(m.params["he"] + m.params[terms[i]]))
    return {"reference": ref, "within_OR_by_period": per,
            "wald_stat": float(wt.statistic), "wald_df": int(len(terms)),
            "wald_p": float(wt.pvalue)}


def _f(x, nd=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _p(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.2e}" if x < 0.001 else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "c22", HERE / "22_clustered_logistic_f1_f3.py")
    c22 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c22)
    pd, stats, smf = c22.pd, c22.stats, c22.smf
    from statsmodels.stats.proportion import proportions_ztest

    addressed, df_outcomes, kinds = c22._load_data(c22.DEFAULT_OUTCOMES)
    out = c22._build_per_obligation_aggregate(addressed, df_outcomes, kinds)
    stance = addressed.groupby("obligation_id").agg(
        n_opposing=("stance", lambda s: int((s == "OPPOSING").sum())),
        n_supporting=("stance", lambda s: int((s == "SUPPORTING").sum())),
    ).reset_index()
    out = out.merge(stance, on="obligation_id", how="left").fillna(
        {"n_opposing": 0, "n_supporting": 0})

    # Baseline check: pooled Finding 1 must match the paper.
    pooled = f1_period(out, stats)
    if pooled["n"] != 12243 or pooled["n_high"] != 168:
        sys.exit(f"ABORT: pooled Finding 1 drifted (n={pooled['n']}, "
                 f"n_high={pooled['n_high']}; expected 12243, 168)")

    periods = docket_periods()
    out["year_cluster"] = out["docket_id"].map(lambda d: periods.get(d, {}).get("year_cluster"))
    out["admin_final"] = out["docket_id"].map(lambda d: periods.get(d, {}).get("admin_final"))

    audit = None
    if pathlib.Path(AUDIT_OUTCOMES).exists():
        audit = c22._build_per_obligation_aggregate(
            addressed, pd.read_parquet(AUDIT_OUTCOMES), kinds)
        audit["year_cluster"] = audit["docket_id"].map(lambda d: periods.get(d, {}).get("year_cluster"))
        audit["admin_final"] = audit["docket_id"].map(lambda d: periods.get(d, {}).get("admin_final"))

    # --- Docket table (what each docket contributes, and which documents were analyzed).
    print("\n=== Anchor dockets with verified obligations ===")
    print(f"{'docket':<24}{'year cluster':>13}{'proposed FR (date)':>28}"
          f"{'final FR (date)':>28}{'admin':>7}{'oblig.':>8}{'high':>6}")
    x = out[out["outcome_state"] != "NEW"]
    for d, g in x.groupby("docket_id"):
        p = periods.get(d, {})
        prop = f"{p.get('proposed_doc')} ({p.get('proposed_pub')})"
        fin = f"{p.get('final_doc')} ({p.get('final_pub')})"
        print(f"{d:<24}{str(p.get('year_cluster')):>13}{prop:>28}{fin:>28}"
              f"{str(p.get('admin_final')):>7}{len(g):>8}{int((g['n_addressing'] >= HIGH).sum()):>6}")

    results = {"pooled_f1": pooled, "splits": {}}
    for col, order, title in (("year_cluster", YEAR_CLUSTERS, "YEAR CLUSTER (proposed rule)"),
                              ("admin_final", ADMINS, "ADMINISTRATION (final rule published)")):
        res = {"periods": {}}
        print(f"\n\n##### Split by {title} #####")
        print("\nFinding 1 (proposed-side obligations):")
        print(f"{'period':<12}{'dockets':>8}{'oblig.':>8}{'high':>6}{'high %':>8}"
              f"{'low %':>8}{'gap pp':>8}{'Fisher p':>10}")
        for per in order:
            s = f1_period(out[out[col] == per], stats)
            res["periods"].setdefault(per, {})["f1"] = s
            print(f"{per:<12}{s['dockets']:>8}{s['n']:>8}{s['n_high']:>6}"
                  f"{_f(s['rate_high']):>8}{_f(s['rate_low']):>8}{_f(s['gap_pp']):>8}"
                  f"{_p(s['fisher_p']):>10}")

        print("\nFinding 2 (obligations with >= 1 opposing vs >= 1 supporting commenter):")
        print(f"{'period':<12}{'n opp':>7}{'opp %':>8}{'n sup':>7}{'sup %':>8}{'z-test p':>10}")
        for per in order:
            s = f2_period(out[out[col] == per], proportions_ztest)
            res["periods"][per]["f2"] = s
            print(f"{per:<12}{s['n_opposing']:>7}{_f(s['rate_opposing']):>8}"
                  f"{s['n_supporting']:>7}{_f(s['rate_supporting']):>8}{_p(s['z_p']):>10}")

        for label, frame in (("classifier-assigned", out), ("audit-corrected", audit)):
            if frame is None:
                continue
            print(f"\nFinding 3 ({label} outcomes; cells SE-org/SE-not/MO-org/MO-not):")
            print(f"{'period':<12}{'cells':>16}{'n':>5}{'Fisher OR':>11}{'p':>8}")
            for per in order:
                s = f3_period(frame[frame[col] == per], c22)
                res["periods"][per][f"f3_{label}"] = s
                cells = "/".join(str(v) for v in s["cells"])
                print(f"{per:<12}{cells:>16}{s['n']:>5}{_f(s['OR'], 2):>11}{_p(s['p']):>8}")

        it = interaction_test(out, col, smf)
        res["f1_interaction"] = it
        print("\nDoes the Finding 1 gap differ across periods? "
              "(docket-FE logit, high x period, clustered SE)")
        if "note" in it:
            print(f"  {it['note']}")
        else:
            ors = ", ".join(f"{k}: {v:.2f}" for k, v in it["within_OR_by_period"].items())
            print(f"  within-rulemaking OR by period: {ors}")
            print(f"  joint Wald test of the interactions: chi2({it['wald_df']}) = "
                  f"{it['wald_stat']:.2f}, p = {_p(it['wald_p'])}")
        results["splits"][col] = res

    pathlib.Path(SUMMARY_JSON).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {SUMMARY_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
