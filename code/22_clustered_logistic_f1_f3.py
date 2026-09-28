"""
22_clustered_logistic_f1_f3.py — Docket-clustered logistic regressions
for Findings 1 and 3 of the EAAMO paper.

Backs §8.1 "Tests do not account for docket-level clustering" — the
paragraph currently flags clustered / FE specifications as "the right
journal-version specification" deferred to follow-up. This script does
it now. Produces three regression rows for §6.3 / §6.4 / §8.1 Table 1:
  T1: F1 high_engagement → was_revised, docket FE + cluster-robust SE
  T2: F3 org_majority → was_edited, cluster-robust SE only (cross-docket)
  T3: F3 org_majority → was_edited, docket FE (will probably hit
      perfect separation on the small engaged subset; fallback +
      diagnostic emit per the script spec)

Baseline replication checks (T0a, T0b) reproduce the §6.3 chi-square
and §6.4 Fisher's exact numbers and ABORT if either drifts — they
guard the data-loading path against silent regressions.

CLASSIFIER CONVENTION (matches code/20_finding3_analysis.py inline rule;
NOT the lib classifier at code/lib/submitter_classification.py):
    title contains "submitted by" | "on behalf" | "written by" → individual
    otherwise                                                   → organizational

DOCKET ID extraction: uses df_outcomes['docket_id'] if present;
otherwise parses from obligation_id with the canonical
`obligation_id_of` namespace `{docket}__{rule_type}__{cidx}_{sidx}`.

COMMIT-BEFORE-CITE: per the team policy surfaced when
code/20_finding3_analysis.py was discovered uncommitted, this script
must be committed before its numbers are cited in §8.1 / Table 1.

Inputs (mirror code/20_finding3_analysis.py exactly):
  data/processed/stage4/*.parquet                  (Stage 4 obligation-comment pairs)
  data/processed/path_a_obligation_outcomes.parquet (Path A outcome states)
  data/processed/comments_augmented/EPA_*.parquet  (per-comment titles)

Usage:
  python3 code/22_clustered_logistic_f1_f3.py

Output: printed comparison table + per-test diagnostic notes. No file outputs.

Required packages: pandas, numpy, scipy, statsmodels (≥0.14 for
`cov_type='cluster'`); optional sklearn for T3's L2 fallback.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import warnings
from math import exp, log, sqrt

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
except ImportError as e:
    sys.exit(f"ERROR: {e}. Required: pip install pandas pyarrow scipy 'statsmodels>=0.14'")


DEFAULT_OUTCOMES = "data/processed/path_a_obligation_outcomes.parquet"
STAGE4_GLOB = "data/processed/stage4/*.parquet"
COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"

# Same individual-phrase set as code/20_finding3_analysis.py.
INDIV_PHRASE_RE = r"submitted by|on behalf|written by"


# ---------------------------------------------------------------------------
# Inline 2x2 fisher helper (matches code/19_finding3_loo_sensitivity.py /
# code/21_finding3_robustness_battery.py conventions so Wald CI numbers
# are interchangeable across scripts).
# ---------------------------------------------------------------------------
def fisher_exact_2x2(a: int, b: int, c: int, d: int
                    ) -> tuple[float, float, float, float]:
    if min(a, b, c, d) <= 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = float(a), float(b), float(c), float(d)
    OR = (a_ * d_) / (b_ * c_)
    se_log = sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    log_or = log(OR)
    ci_lo = exp(log_or - 1.96 * se_log)
    ci_hi = exp(log_or + 1.96 * se_log)
    _, p_two = stats.fisher_exact([[a, b], [c, d]])
    return OR, ci_lo, ci_hi, float(p_two)


# ---------------------------------------------------------------------------
# Docket-id extraction
# ---------------------------------------------------------------------------
_DOCKET_RE = re.compile(r"^(EPA-[A-Z0-9-]+?)__(?:proposed|final)__")


def docket_of_obligation(obl_id: str) -> str | None:
    if not isinstance(obl_id, str):
        return None
    m = _DOCKET_RE.match(obl_id)
    if m:
        return m.group(1)
    # Fallback per the task spec: strip the trailing '-NNNN' provision
    # identifier if the obligation_id uses dash-only namespacing.
    parts = obl_id.split("-")
    if len(parts) >= 5 and parts[0] == "EPA":
        return "-".join(parts[:5])
    return None


def _attach_docket_id(df: pd.DataFrame) -> pd.DataFrame:
    if "docket_id" in df.columns:
        return df
    df = df.copy()
    df["docket_id"] = df["obligation_id"].map(docket_of_obligation)
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_data(outcomes_path: str):
    stage4_files = sorted(glob.glob(STAGE4_GLOB))
    if not stage4_files:
        sys.exit(f"ERROR: no parquets at {STAGE4_GLOB} — Stage 4 outputs "
                 "are gitignored; this script must be run on a "
                 "workstation that has the production data.")
    if not glob.glob(outcomes_path):
        sys.exit(f"ERROR: outcomes parquet not found at {outcomes_path}.")
    print(f"Loading Stage 4 matches ({len(stage4_files)} parquets) + "
          f"outcomes from {outcomes_path} ...", file=sys.stderr)
    df_matches = pd.concat([pd.read_parquet(p) for p in stage4_files])
    df_outcomes = pd.read_parquet(outcomes_path)
    addressed = df_matches[df_matches["addressed"] == True]  # noqa: E712

    comments_dfs = []
    for year_pq in sorted(glob.glob(COMMENTS_GLOB)):
        cdf = pd.read_parquet(year_pq, columns=["document_id", "title"])
        comments_dfs.append(cdf)
    if not comments_dfs:
        sys.exit(f"ERROR: no parquets at {COMMENTS_GLOB}")
    comments = pd.concat(comments_dfs, ignore_index=True)

    # Inline title-pattern classifier (matches code/20_finding3_analysis.py).
    is_indiv = comments["title"].fillna("").str.contains(
        INDIV_PHRASE_RE, case=False, regex=True,
    )
    comments["submitter_kind"] = is_indiv.map(
        {True: "individual", False: "organizational"}
    )
    print(f"  comments: {len(comments):,}; "
          f"org_share = {(comments['submitter_kind'] == 'organizational').mean():.3f}",
          file=sys.stderr)

    return addressed, df_outcomes, comments


def _build_per_obligation_aggregate(addressed, df_outcomes, comments):
    """Build the per-obligation aggregate used by Tests 1, 2, 3:
       outcome_state + n_addressing + n_org + n_ind + docket_id."""
    addr_with_kind = addressed.merge(
        comments[["document_id", "submitter_kind"]],
        left_on="comment_id", right_on="document_id", how="left",
    )
    agg = addr_with_kind.groupby("obligation_id").agg(
        n_addressing=("comment_id", "nunique"),
        n_org=("submitter_kind",
               lambda s: (s == "organizational").sum()),
        n_ind=("submitter_kind",
               lambda s: (s == "individual").sum()),
    ).reset_index()
    agg["org_share"] = agg["n_org"] / (
        (agg["n_org"] + agg["n_ind"]).replace(0, 1)
    )
    out = df_outcomes.merge(agg, on="obligation_id", how="left").fillna(
        {"n_addressing": 0, "n_org": 0, "n_ind": 0, "org_share": 0.0}
    )
    out = _attach_docket_id(out)
    return out


# ---------------------------------------------------------------------------
# Test 0a — Finding 1 baseline chi-square
# ---------------------------------------------------------------------------
def test_0a_baseline(out: pd.DataFrame) -> tuple[float, float, float, int]:
    out = out.copy()
    out["high_engagement"] = out["n_addressing"] >= 5
    # §6.3 baseline: was_revised is SE/MOD/DROPPED (NEW treated as NOT
    # revised — matches code/20_finding3_analysis.py:95).
    out["was_revised"] = out["outcome_state"].isin(
        ["SURVIVED-edited", "MODIFIED", "DROPPED"]
    )
    ct = pd.crosstab(out["high_engagement"], out["was_revised"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = int(ct.values.sum())
    v = (chi2 / n) ** 0.5
    expected_n = 12730
    expected_chi2 = 24.53
    expected_v = 0.044
    if (n != expected_n or abs(chi2 - expected_chi2) > 0.5
            or abs(v - expected_v) > 0.005):
        print(f"\n[T0a] BASELINE MISMATCH — aborting.", file=sys.stderr)
        print(f"  expected: n={expected_n}, chi2={expected_chi2}, V={expected_v}",
              file=sys.stderr)
        print(f"  got:      n={n}, chi2={chi2:.2f}, V={v:.3f}",
              file=sys.stderr)
        print(f"  crosstab:\n{ct}", file=sys.stderr)
        sys.exit(2)
    return chi2, p, v, n


# ---------------------------------------------------------------------------
# Test 0b — Finding 3 baseline Fisher's exact
# ---------------------------------------------------------------------------
def test_0b_baseline(out: pd.DataFrame
                    ) -> tuple[int, int, int, int, float, float, float, float, int]:
    high = out[out["n_addressing"] >= 5].copy()
    high["org_majority"] = high["org_share"] > 0.5
    subset = high[high["outcome_state"].isin(
        ["SURVIVED-edited", "MODIFIED"])].copy()
    se = subset["outcome_state"] == "SURVIVED-edited"
    mo = subset["outcome_state"] == "MODIFIED"
    om = subset["org_majority"]
    a = int((se & om).sum())
    b = int((se & ~om).sum())
    c = int((mo & om).sum())
    d = int((mo & ~om).sum())
    OR, ci_lo, ci_hi, p = fisher_exact_2x2(a, b, c, d)
    n = len(subset)
    expected = (43, 15, 28, 30)
    if (a, b, c, d) != expected or abs(OR - 3.07) > 0.05 or abs(p - 0.007) > 0.005:
        print(f"\n[T0b] BASELINE MISMATCH — aborting.", file=sys.stderr)
        print(f"  expected: cells={expected}, OR=3.07, p=0.007, n=116",
              file=sys.stderr)
        print(f"  got:      cells=({a}, {b}, {c}, {d}), OR={OR:.3f}, "
              f"p={p:.4f}, n={n}", file=sys.stderr)
        sys.exit(2)
    return a, b, c, d, OR, ci_lo, ci_hi, p, n


# ---------------------------------------------------------------------------
# Test 1 — Finding 1 docket-FE logistic + cluster-robust SE
# ---------------------------------------------------------------------------
def test_1_f1_clustered(out: pd.DataFrame) -> dict:
    df = out.copy()
    df["high_engagement"] = (df["n_addressing"] >= 5).astype(int)
    df["was_revised"] = df["outcome_state"].isin(
        ["SURVIVED-edited", "MODIFIED", "DROPPED"]
    ).astype(int)
    df = df.dropna(subset=["docket_id"])
    n_dockets = df["docket_id"].nunique()

    # Logit with docket FE + cluster-robust SE clustered on docket_id.
    # Cluster + FE on the same variable is unusual but defensible: the
    # FE absorbs the docket-level mean of `was_revised`, and the cluster
    # accounts for residual within-docket correlation. This is the
    # "right journal-version specification" §8.1 flags.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.logit(
                "was_revised ~ high_engagement + C(docket_id)", data=df,
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": df["docket_id"]},
                disp=False, maxiter=200,
            )
        except (PerfectSeparationError, np.linalg.LinAlgError) as e:
            return {
                "label": "T1 F1 docket-FE + cluster-SE",
                "n": len(df), "k_dockets": n_dockets,
                "OR": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"),
                "notes": (f"fit failed: {type(e).__name__}: {e}"),
            }
    coef = float(model.params.get("high_engagement", float("nan")))
    se = float(model.bse.get("high_engagement", float("nan")))
    p = float(model.pvalues.get("high_engagement", float("nan")))
    if np.isnan(coef) or np.isnan(se):
        return {
            "label": "T1 F1 docket-FE + cluster-SE",
            "n": len(df), "k_dockets": n_dockets,
            "OR": float("nan"), "ci_lo": float("nan"),
            "ci_hi": float("nan"), "p": float("nan"),
            "notes": "high_engagement coef not in fit output",
        }
    OR = exp(coef)
    ci_lo = exp(coef - 1.96 * se)
    ci_hi = exp(coef + 1.96 * se)
    return {
        "label": "T1 F1 docket-FE + cluster-SE",
        "n": int(len(df)), "k_dockets": n_dockets,
        "OR": OR, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p,
        "notes": f"high_engagement coef={coef:.3f}, SE={se:.3f}",
    }


# ---------------------------------------------------------------------------
# Test 2 — Finding 3 cross-docket logistic + cluster-robust SE (no FE)
# ---------------------------------------------------------------------------
def test_2_f3_clustered(out: pd.DataFrame) -> dict:
    high = out[out["n_addressing"] >= 5].copy()
    high["org_majority"] = (high["org_share"] > 0.5).astype(int)
    subset = high[high["outcome_state"].isin(
        ["SURVIVED-edited", "MODIFIED"])].copy()
    subset["was_edited"] = (
        subset["outcome_state"] == "SURVIVED-edited"
    ).astype(int)
    subset = subset.dropna(subset=["docket_id"])
    n_dockets = subset["docket_id"].nunique()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.logit(
                "was_edited ~ org_majority", data=subset,
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": subset["docket_id"]},
                disp=False, maxiter=200,
            )
        except (PerfectSeparationError, np.linalg.LinAlgError) as e:
            return {
                "label": "T2 F3 cross-docket + cluster-SE",
                "n": len(subset), "k_dockets": n_dockets,
                "OR": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"),
                "notes": f"fit failed: {type(e).__name__}: {e}",
            }
    coef = float(model.params.get("org_majority", float("nan")))
    se = float(model.bse.get("org_majority", float("nan")))
    p = float(model.pvalues.get("org_majority", float("nan")))
    OR = exp(coef)
    ci_lo = exp(coef - 1.96 * se)
    ci_hi = exp(coef + 1.96 * se)
    return {
        "label": "T2 F3 cross-docket + cluster-SE",
        "n": int(len(subset)), "k_dockets": n_dockets,
        "OR": OR, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p,
        "notes": f"org_majority coef={coef:.3f}, SE={se:.3f}",
    }


# ---------------------------------------------------------------------------
# Test 3 — Finding 3 docket-fixed-effects logistic
# (Likely perfect separation given n=116 across ~20 dockets.)
# ---------------------------------------------------------------------------
def _diagnose_perfect_separation(subset: pd.DataFrame) -> dict:
    """Per-docket within-stratum diagnostics: which dockets have all-
    SURVIVED-edited or all-MODIFIED observations (the canonical
    perfect-separation flavor for FE logistic on a binary outcome)."""
    diag = {
        "n_dockets_total": 0, "n_dockets_perfect_sep": 0,
        "n_dockets_singleton": 0,
        "n_obs_in_perfect_sep_dockets": 0,
        "perfect_sep_dockets": [],
    }
    if subset.empty:
        return diag
    for docket, g in subset.groupby("docket_id"):
        diag["n_dockets_total"] += 1
        n_in = len(g)
        if n_in < 2:
            diag["n_dockets_singleton"] += 1
            diag["n_obs_in_perfect_sep_dockets"] += n_in
            diag["perfect_sep_dockets"].append((docket, n_in, "singleton"))
            continue
        unique_outcomes = g["was_edited"].nunique()
        if unique_outcomes == 1:
            label = "all-edited" if g["was_edited"].iloc[0] == 1 else "all-modified"
            diag["n_dockets_perfect_sep"] += 1
            diag["n_obs_in_perfect_sep_dockets"] += n_in
            diag["perfect_sep_dockets"].append((docket, n_in, label))
    return diag


def test_3_f3_docket_fe(out: pd.DataFrame) -> dict:
    high = out[out["n_addressing"] >= 5].copy()
    high["org_majority"] = (high["org_share"] > 0.5).astype(int)
    subset = high[high["outcome_state"].isin(
        ["SURVIVED-edited", "MODIFIED"])].copy()
    subset["was_edited"] = (
        subset["outcome_state"] == "SURVIVED-edited"
    ).astype(int)
    subset = subset.dropna(subset=["docket_id"])
    n_dockets = subset["docket_id"].nunique()
    diag = _diagnose_perfect_separation(subset)

    result = {
        "label": "T3 F3 docket-FE logistic",
        "n": int(len(subset)), "k_dockets": n_dockets,
        "OR": float("nan"), "ci_lo": float("nan"),
        "ci_hi": float("nan"), "p": float("nan"),
        "notes": "",
        "diag": diag,
    }

    # First try standard MLE.
    fitted = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.logit(
                "was_edited ~ org_majority + C(docket_id)", data=subset,
            ).fit(disp=False, maxiter=200)
        except (PerfectSeparationError, np.linalg.LinAlgError, Exception) as e:
            fit_err = f"{type(e).__name__}: {e}"
            result["notes"] = f"MLE failed: {fit_err}"
        else:
            # statsmodels often returns infinite SE for separated
            # coefficients without raising; check for that.
            coef = float(model.params.get("org_majority", float("nan")))
            se = float(model.bse.get("org_majority", float("nan")))
            if np.isfinite(coef) and np.isfinite(se) and se > 0:
                p = float(model.pvalues.get("org_majority", float("nan")))
                result["OR"] = exp(coef)
                result["ci_lo"] = exp(coef - 1.96 * se)
                result["ci_hi"] = exp(coef + 1.96 * se)
                result["p"] = p
                result["notes"] = (f"MLE converged: coef={coef:.3f}, "
                                   f"SE={se:.3f}")
                fitted = True
            else:
                result["notes"] = ("MLE returned non-finite coef/SE — "
                                   "perfect separation in at least one stratum")

    if fitted:
        return result

    # Fallback: L2-penalized via sklearn. Reports a biased but defensible
    # point estimate for the OR; no SE / CI / p (penalized inference is
    # ill-defined under perfect-separation).
    try:
        from sklearn.linear_model import LogisticRegression
        X = pd.get_dummies(
            subset[["org_majority", "docket_id"]],
            columns=["docket_id"], drop_first=True,
        ).astype(float)
        y = subset["was_edited"].astype(int).values
        clf = LogisticRegression(
            penalty="l2", C=10.0, solver="lbfgs",
            max_iter=1000, fit_intercept=True,
        )
        clf.fit(X, y)
        cols = list(X.columns)
        idx = cols.index("org_majority")
        coef_pen = float(clf.coef_[0, idx])
        result["OR"] = exp(coef_pen)
        result["notes"] += (
            f" | L2-penalized fallback (sklearn, C=10): coef={coef_pen:.3f}, "
            f"OR={exp(coef_pen):.2f}; CI / p not reported (penalized "
            f"inference under separation is ill-defined)"
        )
    except ImportError:
        result["notes"] += " | sklearn not installed; no penalized fallback"
    except Exception as e:  # noqa: BLE001
        result["notes"] += f" | L2 fallback failed: {type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _fmt_ci(ci_lo: float, ci_hi: float) -> str:
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "n/a"
    return f"[{ci_lo:.2f}, {ci_hi:.2f}]"


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.0005:
        return f"{p:.2e}"
    return f"{p:.3f}"


def _fmt_or(OR: float) -> str:
    return "n/a" if np.isnan(OR) else f"{OR:.2f}"


def _print_table(t0a, t0b, t1, t2, t3) -> None:
    chi2, chi_p, v, n0a = t0a
    a, b, c, d, OR0, ci_lo0, ci_hi0, p0, n0b = t0b

    headers = ("Test", "n", "OR", "95% CI", "p", "Notes")
    fmt = "{:<46} {:>5}  {:>5}  {:>14}  {:>9}  {:<55}"
    print("\n" + fmt.format(*headers))
    print("-" * 150)
    # T0a — chi-square (no OR/CI; report V instead in the OR column slot
    # and chi-square p in the p column)
    print(fmt.format(
        "T0a Finding 1 baseline (chi-square)", n0a, "----",
        f"V={v:.3f}", _fmt_p(chi_p),
        f"chi2={chi2:.2f}, replicates §6.3 (V=0.044)",
    ))
    print(fmt.format(
        "T0b Finding 3 baseline (Fisher's exact)", n0b, _fmt_or(OR0),
        _fmt_ci(ci_lo0, ci_hi0), _fmt_p(p0),
        f"cells=({a},{b},{c},{d}), replicates §6.4",
    ))
    for tr in (t1, t2, t3):
        notes = tr["notes"]
        if "k_dockets" in tr and tr["k_dockets"] > 0:
            notes = f"k_dockets={tr['k_dockets']}; " + notes
        if len(notes) > 55:
            notes = notes[:52] + "..."
        print(fmt.format(
            tr["label"], tr["n"], _fmt_or(tr["OR"]),
            _fmt_ci(tr["ci_lo"], tr["ci_hi"]), _fmt_p(tr["p"]), notes,
        ))


def _print_interpretation(t1, t2, t3) -> None:
    print("\nInterpretation:")
    # T1
    if not np.isnan(t1["OR"]):
        print(f"  T1: Docket FE + cluster-SE leaves the F1 engagement→revision")
        print(f"      estimate at OR={t1['OR']:.2f} (CI={_fmt_ci(t1['ci_lo'], t1['ci_hi'])}); ")
        print(f"      p={_fmt_p(t1['p'])}. Confirms or rebuts §6.3 chi-square "
              f"interpretation.")
    else:
        print(f"  T1: Docket FE + cluster-SE fit failed — see notes column.")
    # T2
    if not np.isnan(t2["OR"]):
        print(f"  T2: Cross-docket clustered SE yields OR={t2['OR']:.2f} on org_majority")
        print(f"      (CI={_fmt_ci(t2['ci_lo'], t2['ci_hi'])}, p={_fmt_p(t2['p'])}).")
        print(f"      Compare to T0b Fisher's OR ≈ 3.07; point estimate ought to")
        print(f"      be similar, CI may be wider due to within-docket correlation.")
    else:
        print(f"  T2: Cross-docket clustered fit failed — see notes column.")
    # T3
    diag = t3.get("diag", {})
    print(f"  T3: Docket FE logistic on the engaged subset is the most demanding "
          f"specification.")
    if diag:
        print(f"      Dockets with perfect separation (all SE or all MO within "
              f"docket): {diag.get('n_dockets_perfect_sep', 0)} of "
              f"{diag.get('n_dockets_total', 0)}.")
        print(f"      Singleton-docket strata: {diag.get('n_dockets_singleton', 0)} "
              f"({diag.get('n_obs_in_perfect_sep_dockets', 0)} obs in dropped or "
              f"degenerate dockets).")
        if diag.get("perfect_sep_dockets"):
            ex = diag["perfect_sep_dockets"][:5]
            ex_str = "; ".join(f"{d[0]} (n={d[1]}, {d[2]})" for d in ex)
            tail = " …" if len(diag["perfect_sep_dockets"]) > 5 else ""
            print(f"      First few: {ex_str}{tail}")
    if not np.isnan(t3["OR"]):
        print(f"      OR estimate: {t3['OR']:.2f}; see notes for whether MLE or "
              f"penalized fallback produced it.")
    else:
        print(f"      No OR estimate available; see notes.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--outcomes-path", default=DEFAULT_OUTCOMES,
        help=f"Path A outcomes parquet (default: {DEFAULT_OUTCOMES}).",
    )
    args = ap.parse_args()

    addressed, df_outcomes, comments = _load_data(args.outcomes_path)
    out = _build_per_obligation_aggregate(addressed, df_outcomes, comments)

    print("\n=== Baseline replication checks (T0a, T0b) ===", file=sys.stderr)
    t0a = test_0a_baseline(out)
    print(f"  T0a chi-square: chi2={t0a[0]:.2f}, p={t0a[1]:.2e}, V={t0a[2]:.3f}, "
          f"n={t0a[3]}  ✓", file=sys.stderr)
    t0b = test_0b_baseline(out)
    print(f"  T0b Fisher: cells=({t0b[0]}, {t0b[1]}, {t0b[2]}, {t0b[3]}), "
          f"OR={t0b[4]:.3f}, p={t0b[7]:.4f}, n={t0b[8]}  ✓", file=sys.stderr)

    print("\n=== Clustered / FE regressions (T1, T2, T3) ===", file=sys.stderr)
    t1 = test_1_f1_clustered(out)
    print(f"  T1 done: OR={_fmt_or(t1['OR'])}, p={_fmt_p(t1['p'])}, "
          f"k_dockets={t1.get('k_dockets', 0)}", file=sys.stderr)
    t2 = test_2_f3_clustered(out)
    print(f"  T2 done: OR={_fmt_or(t2['OR'])}, p={_fmt_p(t2['p'])}, "
          f"k_dockets={t2.get('k_dockets', 0)}", file=sys.stderr)
    t3 = test_3_f3_docket_fe(out)
    print(f"  T3 done: OR={_fmt_or(t3['OR'])}, p={_fmt_p(t3['p'])}",
          file=sys.stderr)

    _print_table(t0a, t0b, t1, t2, t3)
    _print_interpretation(t1, t2, t3)


if __name__ == "__main__":
    main()
