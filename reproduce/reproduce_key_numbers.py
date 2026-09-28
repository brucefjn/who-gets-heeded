"""
reproduce_key_numbers.py - Recompute the paper's headline statistics from the
released tables in data/processed/, with no LLM calls and no raw comment data.

Reproduces:
  Finding 1   revision rates, Yates chi-square, Cramer's V, and the
              leave-one-docket-out check (Appendix C.1)
  Finding 2   opposing vs supporting revision rates and the two-proportion z-test
  Finding 3   pre-audit and audit-corrected contingency cells, Fisher odds ratios
              and p-values (needs data/processed/finding3_obligations.csv)
  Audits      extraction precision (Section 5), matcher agreement (Appendix B.1),
              outcome-state classifier agreement (Section 6.3)

Writes:
  data/processed/finding1_loo_proposed_side.csv
  docs/matcher_audit_results.md   (inter-rater statistics and disagreement list)

Usage (from the repository root):
    python3 reproduce/reproduce_key_numbers.py
Requires only pandas.
"""
from __future__ import annotations

import math
import os

import pandas as pd

DATA = "data/processed"
REVISED = {"SURVIVED-edited", "MODIFIED", "DROPPED"}
SE, MO = "SURVIVED-edited", "MODIFIED"


# ---------------------------------------------------------------- statistics
def kappa(a, b) -> float:
    a = pd.Series(list(a)).astype(str).str.strip().str.upper()
    b = pd.Series(list(b)).astype(str).str.strip().str.upper()
    po = (a.values == b.values).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def yates_chi2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """2x2 chi-square with continuity correction; returns (chi2, p)."""
    n = a + b + c + d
    num = n * max(0.0, abs(a * d - b * c) - n / 2) ** 2
    chi2 = num / ((a + b) * (c + d) * (a + c) * (b + d))
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    pool = (x1 + x2) / (n1 + n2)
    z = (x1 / n1 - x2 / n2) / math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    return z, math.erfc(abs(z) / math.sqrt(2))


def fisher_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Sample odds ratio ad/bc and two-sided Fisher exact p-value."""
    r1, c1, n = a + b, a + c, a + b + c + d

    def logp(x: int) -> float:
        return (math.lgamma(c1 + 1) - math.lgamma(x + 1) - math.lgamma(c1 - x + 1)
                + math.lgamma(n - c1 + 1) - math.lgamma(r1 - x + 1)
                - math.lgamma(n - c1 - r1 + x + 1)
                - (math.lgamma(n + 1) - math.lgamma(r1 + 1) - math.lgamma(n - r1 + 1)))

    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    p_obs = logp(a)
    p = sum(math.exp(logp(x)) for x in range(lo, hi + 1) if logp(x) <= p_obs + 1e-7)
    odds = (a * d) / (b * c) if b * c else float("inf")
    return odds, min(1.0, p)


def show(label: str, value: str, paper: str) -> None:
    print(f"  {label:<46} {value:<28} paper: {paper}")


# ---------------------------------------------------------------- findings
def finding_1_and_2() -> None:
    d = pd.read_csv(os.path.join(DATA, "responsiveness_analysis.csv"))
    d["docket"] = d["obligation_id"].str.split("__").str[0]
    p = d[d["outcome_state"] != "NEW"].copy()          # proposed-side obligations
    p["rev"] = p["outcome_state"].isin(REVISED)
    p["high"] = p["n_comments_addressed"] >= 5

    def cells(df):
        return (int((df.high & df.rev).sum()), int((df.high & ~df.rev).sum()),
                int((~df.high & df.rev).sum()), int((~df.high & ~df.rev).sum()))

    a, b, c, e = cells(p)
    chi2, pv = yates_chi2(a, b, c, e)
    print("\nFinding 1 (12,243 proposed-side obligations)")
    show("high-engagement revised", f"{100 * a / (a + b):.1f}% of {a + b}", "69.0% of 168")
    show("less-addressed revised", f"{100 * c / (c + e):.1f}% of {c + e:,}", "51.5% of 12,075")
    show("chi-square (Yates), p", f"{chi2:.2f}, {pv:.1e}", "19.71, p < 0.001")
    show("Cramer's V", f"{math.sqrt(chi2 / len(p)):.3f}", "0.040")

    rows = []
    for dk in sorted(p["docket"].unique()):
        sub = p[p["docket"] != dk]
        x2, px = yates_chi2(*cells(sub))
        rows.append(dict(dropped_docket=dk, n_obligations_dropped=int((p.docket == dk).sum()),
                         chi2_yates=round(x2, 4), p=px, cramers_v=round(math.sqrt(x2 / len(sub)), 4)))
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(DATA, "finding1_loo_proposed_side.csv"), index=False)
    show("leave-one-docket-out chi-square range",
         f"[{loo.chi2_yates.min():.2f}, {loo.chi2_yates.max():.2f}], max p {loo.p.max():.1e}",
         "[11.44, 23.77], all p < 0.001")

    op, su = p[p["has_opposing"] == True], p[p["has_supporting"] == True]  # noqa: E712
    z, pz = two_prop_z(int(op.rev.sum()), len(op), int(su.rev.sum()), len(su))
    print("\nFinding 2")
    show(">= 1 opposing commenter revised", f"{100 * op.rev.mean():.1f}% of {len(op)}", "62.3% of 212")
    show(">= 1 supporting commenter revised", f"{100 * su.rev.mean():.1f}% of {len(su)}", "68.3% of 312")
    show("two-proportion z, p", f"{z:.2f}, {pz:.3f}", "-1.42, 0.155")


def finding_3() -> None:
    path = os.path.join(DATA, "finding3_obligations.csv")
    print("\nFinding 3")
    if not os.path.exists(path):
        print("  finding3_obligations.csv not present; skipped")
        return
    f = pd.read_csv(path)
    f["org_majority"] = f["org_majority"].astype(str).str.lower().eq("true")
    for col, paper in (("outcome_state", "43/15/28/30, OR 3.07, p 0.007, n 116"),
                       ("outcome_state_audit_corrected", "13/2/57/43, OR 4.90, p 0.044, n 115")):
        s = f[f[col].isin([SE, MO])]
        a = int(((s[col] == SE) & s.org_majority).sum())
        b = int(((s[col] == SE) & ~s.org_majority).sum())
        c = int(((s[col] == MO) & s.org_majority).sum())
        d = int(((s[col] == MO) & ~s.org_majority).sum())
        odds, pv = fisher_2x2(a, b, c, d)
        show(col, f"{a}/{b}/{c}/{d}, OR {odds:.2f}, p {pv:.3f}, n {a + b + c + d}", paper)


# ---------------------------------------------------------------- audits
def extraction_audit() -> None:
    a = pd.read_csv(os.path.join(DATA, "2026-05-11_path_a_audit_2018-0775.csv"))
    lab = a["RECONCILED_LABEL"].str.strip().str.lower()
    k = int(lab.isin(["good", "borderline"]).sum())
    lo, hi = wilson(k, len(a))
    print("\nExtraction precision audit (Section 5)")
    show("exact inter-rater agreement",
         f"{int((a.BRUCE_LABEL.str.strip() == a.YUE_LABEL.str.strip()).sum())}/{len(a)}", "66/66")
    show("Cohen's kappa", f"{kappa(a.BRUCE_LABEL, a.YUE_LABEL):.3f}", "1.0")
    show("precision (good + borderline)", f"{k / len(a):.4f} [{lo:.3f}, {hi:.3f}]",
         "0.9545 [0.875, 0.984]")


def matcher_audit() -> None:
    b = pd.read_csv(os.path.join(DATA, "stage4_validation_audit_bruce.csv"))
    y = pd.read_csv(os.path.join(DATA, "stage4_validation_audit_yue.csv"))
    key = pd.read_csv(os.path.join(DATA, "stage4_validation_audit_KEY.csv"))
    m = (b[["pair_id", "comment_id", "obligation_id", "BRUCE_ADDRESSED", "BRUCE_STANCE"]]
         .merge(y[["pair_id", "YUE_ADDRESSED", "YUE_STANCE"]], on="pair_id")
         .merge(key[["pair_id", "llm_addressed", "llm_stance"]], on="pair_id"))
    up = lambda s: s.astype(str).str.strip().str.upper()  # noqa: E731
    addressed = up(m.BRUCE_ADDRESSED).isin(["TRUE", "1", "YES"])
    stats = [
        ("inter-rater kappa, addressing (n = 100)", kappa(m.BRUCE_ADDRESSED, m.YUE_ADDRESSED), "1.000"),
        ("inter-rater kappa, stance (n = 100)", kappa(m.BRUCE_STANCE, m.YUE_STANCE), "0.953"),
        (f"inter-rater kappa, stance, addressed pairs (n = {int(addressed.sum())})",
         kappa(m.BRUCE_STANCE[addressed], m.YUE_STANCE[addressed]), "0.897 (n = 49)"),
        ("rater 1 vs matcher kappa, addressing", kappa(m.BRUCE_ADDRESSED, m.llm_addressed), "0.820"),
        ("rater 2 vs matcher kappa, addressing", kappa(m.YUE_ADDRESSED, m.llm_addressed), "0.820"),
        ("rater 1 vs matcher kappa, stance", kappa(m.BRUCE_STANCE, m.llm_stance), "about 0.70"),
        ("rater 2 vs matcher kappa, stance", kappa(m.YUE_STANCE, m.llm_stance), "about 0.70"),
    ]
    print("\nObligation-comment matcher audit (Appendix B.1)")
    for label, val, paper in stats:
        show(label, f"{val:.3f}", paper)

    stance_dis = m[up(m.BRUCE_STANCE) != up(m.YUE_STANCE)]
    addr_dis = m[up(m.BRUCE_ADDRESSED) != up(m.YUE_ADDRESSED)]
    llm_dis = m[(up(m.BRUCE_ADDRESSED) != up(m.llm_addressed)) | (up(m.YUE_ADDRESSED) != up(m.llm_addressed))]
    lines = ["# Obligation-comment matcher audit: results",
             "",
             "Generated by `reproduce/reproduce_key_numbers.py` from the released coding sheets "
             "(`data/processed/stage4_validation_audit_*.csv`). Rater 1 is Jianing Fan (columns `BRUCE_*`); "
             "rater 2 is Yue Yao (columns `YUE_*`). Comment text is not included; look up any `comment_id` "
             "on regulations.gov.",
             "",
             "## Agreement (100 stratified comment-obligation pairs)",
             "",
             "| Statistic | Cohen's kappa |",
             "|---|---|"]
    lines += [f"| {label} | {val:.3f} |" for label, val, _ in stats]
    lines += ["", f"## Inter-rater disagreements on addressing ({len(addr_dis)})", ""]
    if addr_dis.empty:
        lines += ["None."]
    else:
        lines += ["| pair_id | comment_id | obligation_id | rater 1 | rater 2 |", "|---|---|---|---|---|"]
        lines += [f"| {r.pair_id} | {r.comment_id} | {r.obligation_id} | {r.BRUCE_ADDRESSED} | "
                  f"{r.YUE_ADDRESSED} |" for r in addr_dis.itertuples()]
    lines += ["", f"## Inter-rater disagreements on stance ({len(stance_dis)})", "",
              "| pair_id | comment_id | obligation_id | rater 1 | rater 2 |", "|---|---|---|---|---|"]
    lines += [f"| {r.pair_id} | {r.comment_id} | {r.obligation_id} | {r.BRUCE_STANCE} | {r.YUE_STANCE} |"
              for r in stance_dis.itertuples()]
    lines += ["", f"## Pairs where the matcher's addressing label differs from a rater ({len(llm_dis)})", "",
              "| pair_id | comment_id | obligation_id | rater 1 | rater 2 | matcher |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {r.pair_id} | {r.comment_id} | {r.obligation_id} | {r.BRUCE_ADDRESSED} | "
              f"{r.YUE_ADDRESSED} | {r.llm_addressed} |" for r in llm_dis.itertuples()]
    with open("docs/matcher_audit_results.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def outcome_audit() -> None:
    r = pd.read_csv(os.path.join(DATA, "outcome_audit_reconciled.csv"))
    t1 = r[r["audit_tier"].astype(str).str.contains("1")]
    t2 = r[~r.index.isin(t1.index)]

    def semo(df, x, y):
        s = df[df[x].isin([SE, MO]) & df[y].isin([SE, MO])]
        return kappa(s[x], s[y]), len(s)

    print("\nOutcome-state classifier audit (Section 6.3)")
    show("inter-rater kappa, 5-class (n = 150)", f"{kappa(r.rater_1_label, r.rater_2_label):.3f}", "0.898")
    k, n = semo(r, "rater_1_label", "rater_2_label")
    show(f"inter-rater kappa, SE vs MO (n = {n})", f"{k:.3f}", "0.816 (n = 134)")
    show("Tier 1 inter-rater kappa, 5-class", f"{kappa(t1.rater_1_label, t1.rater_2_label):.3f}", "0.816")
    k, n = semo(t1, "rater_1_label", "rater_2_label")
    show(f"Tier 1 inter-rater kappa, SE vs MO (n = {n})", f"{k:.3f}", "0.803")
    k, n = semo(r, "classifier_label", "adjudicated_label")
    show(f"classifier vs adjudicated, SE vs MO (n = {n})", f"{k:.3f}", "0.137 (n = 129)")
    show("classifier vs adjudicated, 5-class",
         f"{kappa(r.classifier_label, r.adjudicated_label):.3f}", "0.324")
    show("Tier 2 classifier vs adjudicated, 5-class",
         f"{kappa(t2.classifier_label, t2.adjudicated_label):.2f}", "0.59")
    for cls, paper in ((MO, "precision 0.94, recall 0.52-0.53"), (SE, "precision 0.17-0.20, recall 0.61-0.65")):
        vals = []
        for rater in ("rater_1_label", "rater_2_label"):
            tp = int(((r.classifier_label == cls) & (r[rater] == cls)).sum())
            vals.append(f"{tp / (r.classifier_label == cls).sum():.2f}/{tp / (r[rater] == cls).sum():.2f}")
        show(f"{cls} precision/recall vs rater 1, rater 2", ", ".join(vals), paper)


if __name__ == "__main__":
    finding_1_and_2()
    finding_3()
    extraction_audit()
    matcher_audit()
    outcome_audit()
