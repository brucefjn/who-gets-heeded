"""
33_pairing_window_check.py - Are Findings 1 and 3 sensitive to how the
proposed/final Federal Register documents were paired, and to comments from
other comment periods in the same docket?

Why this exists: code/12 picks, for each (docket, rule_type), the largest
Federal Register document listed in data/processed/federal_register_index.csv.
Several anchor dockets host more than one rulemaking, so that rule can pair
a proposal and a final rule from different rulemakings, and it can attach
comments from other comment periods in the same docket. While building
code/32 we found, for example:
  - EPA-HQ-OW-2009-0819: the analyzed proposal is the 2013 NPRM, but the
    analyzed final rule is the 2024 supplemental rule (its verified
    obligations cite compliance dates in 2028-2035).
  - EPA-HQ-OAR-2002-0058: the analyzed final rule (2011-03-21) predates the
    analyzed proposal (2011-12-23).
  - EPA-HQ-OAR-2015-0072: the analyzed proposal is dated 2023-01-27, after
    the 2010-2022 comment corpus ends.
  - EPA-HQ-OAR-2009-0234 and EPA-HQ-OAR-2011-0044 use the same Federal
    Register documents, so their obligations are duplicated.

This script prints the pairing diagnostics for every anchor and re-estimates
Findings 1 and 3 under three sensitivity variants:
  A  EXCLUDE FLAGGED   drop dockets flagged below.
  B  COMMENT WINDOW    count only addressing comments received between the
                       analyzed proposal's publication date and the analyzed
                       final rule's publication date (inclusive).
  C  A + B.

Flags (printed with the reason for each docket):
  ORDER        analyzed final rule published before the analyzed proposal.
  POST_CORPUS  analyzed proposal published after 2022-12-31, the end of the
               comment corpus, so no analyzed comment can respond to it.
  SHARED       analyzed proposal or final document also used by another
               anchor; the anchor with more comments is kept.
  CHECKED      mismatch confirmed by reading the verified obligations
               (EPA-HQ-OW-2009-0819, see above). Use --no-checked to skip.

Reuses code/22 (data loading, regressions), code/31 (Finding 1 and 3
statistics), and code/32 (canonical document per docket).

Usage (repo root, on the machine with the production data):
    python3 code/33_pairing_window_check.py

Writes data/processed/robustness_pairing_window_summary.json.
Required packages: pandas, pyarrow, numpy, scipy, statsmodels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AUDIT_OUTCOMES = "data/processed/path_a_obligation_outcomes_audit_corrected.parquet"
SUMMARY_JSON = "data/processed/robustness_pairing_window_summary.json"
CORPUS_END = "2022-12-31"
CHECKED_MISMATCH = {
    "EPA-HQ-OW-2009-0819": ("analyzed proposal is the 2013 NPRM; analyzed final is the "
                            "2024 supplemental rule (final-side obligations cite 2028-2035 dates)"),
}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flag_dockets(periods: dict, n_comments: dict, use_checked: bool = True) -> dict:
    """docket -> list of (flag, reason)."""
    flags: dict[str, list[tuple[str, str]]] = {d: [] for d in periods}
    for d, p in periods.items():
        pp, fp = p.get("proposed_pub"), p.get("final_pub")
        if pp and fp and fp < pp:
            flags[d].append(("ORDER", f"final {fp} precedes proposal {pp}"))
        if pp and pp > CORPUS_END:
            flags[d].append(("POST_CORPUS", f"proposal {pp} is after {CORPUS_END}"))
        if use_checked and d in CHECKED_MISMATCH:
            flags[d].append(("CHECKED", CHECKED_MISMATCH[d]))
    # Shared documents: keep the anchor with the most comments in each group.
    by_doc: dict[str, list[str]] = {}
    for d, p in periods.items():
        for key in ("proposed_doc", "final_doc"):
            if p.get(key):
                by_doc.setdefault(p[key], []).append(d)
    for doc, ds in by_doc.items():
        ds = sorted(set(ds), key=lambda x: -n_comments.get(x, 0))
        for d in ds[1:]:
            reason = f"FR document {doc} also analyzed for {ds[0]}"
            if ("SHARED", reason) not in flags[d]:
                flags[d].append(("SHARED", reason))
    return {d: f for d, f in flags.items() if f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-checked", action="store_true",
                    help="do not exclude the manually checked mismatch list")
    args = ap.parse_args()

    c22 = _load("c22", "22_clustered_logistic_f1_f3.py")
    m31 = _load("m31", "31_dedup_mass_comments.py")
    m32 = _load("m32", "32_time_split.py")
    pd = c22.pd

    addressed, df_outcomes, kinds = c22._load_data(c22.DEFAULT_OUTCOMES)
    out = c22._build_per_obligation_aggregate(addressed, df_outcomes, kinds)
    df_audit = (pd.read_parquet(AUDIT_OUTCOMES)
                if pathlib.Path(AUDIT_OUTCOMES).exists() else None)

    base = m31.f1_stats(out, c22)
    if base["n"] != 12243 or abs(base["chi2"] - 19.71) > 0.05:
        sys.exit(f"ABORT: Finding 1 baseline drifted (n={base['n']}, chi2={base['chi2']:.2f})")
    c22.test_0b_baseline(out)

    periods = m32.docket_periods()
    with open(m32.ANCHORS_CSV, encoding="utf-8", newline="") as f:
        n_comments = {r["docket_id"]: int(float(r.get("n_comments") or 0))
                      for r in csv.DictReader(f)}
    analyzed = set(out.loc[out["outcome_state"] != "NEW", "docket_id"].dropna())
    flags = {d: f for d, f in flag_dockets(periods, n_comments, not args.no_checked).items()
             if d in analyzed}

    # --- Pairing diagnostics.
    print("\n=== Analyzed Federal Register pairs (code/12 rule: largest document) ===")
    print(f"{'docket':<24}{'proposed FR (date)':>28}{'final FR (date)':>28}"
          f"{'oblig.':>8}{'high':>6}{'F3 set':>8}  flags")
    x = out[out["outcome_state"] != "NEW"]
    f3set = x[(x["n_addressing"] >= 5) & x["outcome_state"].isin(["SURVIVED-edited", "MODIFIED"])]
    for d in sorted(analyzed):
        p = periods.get(d, {})
        g = x[x["docket_id"] == d]
        fl = ", ".join(k for k, _ in flags.get(d, [])) or "-"
        print(f"{d:<24}{p.get('proposed_doc', '')} ({p.get('proposed_pub', '')})".ljust(52)
              + f"{p.get('final_doc', '')} ({p.get('final_pub', '')})".rjust(28)
              + f"{len(g):>8}{int((g['n_addressing'] >= 5).sum()):>6}"
              + f"{int((f3set['docket_id'] == d).sum()):>8}  {fl}")
    print("\nFlag reasons:")
    for d, fl in sorted(flags.items()):
        for k, why in fl:
            print(f"  {d}: {k} - {why}")
    flagged = set(flags)

    # --- Comment-window filter.
    def windowed(addr):
        need = set(addr["comment_id"].astype(str))
        date_cols = None
        parts = []
        for pq in sorted(glob.glob(c22.COMMENTS_GLOB)):
            if date_cols is None:
                import pyarrow.parquet as pqm
                names = set(pqm.ParquetFile(pq).schema.names)
                date_cols = [c for c in ("received_date", "posted_date") if c in names]
                if not date_cols:
                    return None, "no received_date / posted_date column in comments parquet"
            df = pd.read_parquet(pq, columns=["document_id"] + date_cols)
            df = df[df["document_id"].astype(str).isin(need)]
            if not df.empty:
                parts.append(df)
        dates = pd.concat(parts, ignore_index=True).drop_duplicates("document_id")
        when = None
        for c in date_cols:   # received_date first, posted_date as fallback
            v = pd.to_datetime(dates[c], errors="coerce", utc=True).dt.tz_localize(None)
            when = v if when is None else when.fillna(v)
        when_by_id = dict(zip(dates["document_id"].astype(str), when))
        w = pd.to_datetime(addr["comment_id"].astype(str).map(when_by_id))
        dk = addr["obligation_id"].map(c22.docket_of_obligation)
        lo = pd.to_datetime(dk.map(lambda d: periods.get(d, {}).get("proposed_pub")))
        hi = pd.to_datetime(dk.map(lambda d: periods.get(d, {}).get("final_pub")))
        keep = w.notna() & (w >= lo) & (w <= hi + pd.Timedelta(days=1))
        note = (f"kept {int(keep.sum()):,} of {len(addr):,} addressing pairs; "
                f"{int(w.isna().sum()):,} pairs had no comment date")
        return addr[keep.values], note

    variants = {"paper": (out, None)}
    variants["A exclude flagged"] = (out[~out["docket_id"].isin(flagged)], None)
    addr_w, note = windowed(addressed)
    if addr_w is not None:
        out_w = c22._build_per_obligation_aggregate(addr_w, df_outcomes, kinds)
        variants["B comment window"] = (out_w, note)
        variants["C = A + B"] = (out_w[~out_w["docket_id"].isin(flagged)], note)
    else:
        print(f"\nComment-window variants skipped: {note}")

    results = {"flags": {d: fl for d, fl in flags.items()}, "variants": {}}
    print("\n=== Finding 1 (proposed-side obligations) ===")
    print(f"{'variant':<20}{'n':>7}{'n_high':>7}{'high %':>8}{'low %':>8}{'gap pp':>8}"
          f"{'chi2':>8}{'p':>10}{'FE OR [95% CI]':>22}{'FE p':>8}")
    for name, (frame, _) in variants.items():
        s = m31.f1_stats(frame, c22)
        results["variants"].setdefault(name, {})["f1"] = s
        ci = f"{m31._f(s['fe_OR'])} [{m31._f(s['fe_ci'][0])}, {m31._f(s['fe_ci'][1])}]"
        print(f"{name:<20}{s['n']:>7}{s['n_high']:>7}{s['rate_high']:>8.1f}{s['rate_low']:>8.1f}"
              f"{s['rate_high'] - s['rate_low']:>8.1f}{s['chi2']:>8.2f}{m31._p(s['p']):>10}"
              f"{ci:>22}{m31._p(s['fe_p']):>8}")

    for label in ("classifier-assigned", "audit-corrected"):
        if label == "audit-corrected" and df_audit is None:
            continue
        print(f"\n=== Finding 3 ({label} outcomes) ===")
        print(f"{'variant':<20}{'cells SEo/SEn/MOo/MOn':>24}{'n':>5}{'%org SE':>9}{'%org MO':>9}"
              f"{'Fisher OR':>11}{'p':>8}{'CMH OR':>9}{'p':>8}")
        for name, (frame, _) in variants.items():
            if label == "audit-corrected":
                src = addressed if name in ("paper", "A exclude flagged") else addr_w
                fa = c22._build_per_obligation_aggregate(src, df_audit, kinds)
                if name in ("A exclude flagged", "C = A + B"):
                    fa = fa[~fa["docket_id"].isin(flagged)]
                frame = fa
            s = m31.f3_stats(frame, c22)
            results["variants"][name][f"f3_{label}"] = s
            cells = "/".join(str(v) for v in s["cells"])
            print(f"{name:<20}{cells:>24}{s['n']:>5}{s['pct_org_SE']:>9.1f}{s['pct_org_MO']:>9.1f}"
                  f"{m31._f(s['fisher_OR']):>11}{m31._p(s['fisher_p']):>8}"
                  f"{m31._f(s['cmh']['OR']):>9}{m31._p(s['cmh']['p']):>8}")
    for name, (_, note) in variants.items():
        if note:
            print(f"\n{name}: {note}")
            results["variants"][name]["note"] = note

    pathlib.Path(SUMMARY_JSON).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {SUMMARY_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
