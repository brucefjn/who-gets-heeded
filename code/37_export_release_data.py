"""
37_export_release_data.py - Write the parquet-derived tables for the public
release repository (who-gets-heeded) without comment text or submitter
metadata.

Run from the repo root on a machine with pyarrow:
    python3 code/37_export_release_data.py ~/Downloads/EAAMO_2026/who-gets-heeded

Writes into <release>/data/processed/:
  stage4_matches.csv.gz          comment_id, obligation_id, addressed, stance for all
                                 candidate pairs. The matcher's free-text justifications
                                 are dropped because they can quote comment text.
  obligation_outcomes.csv        outcome state per obligation (text-similarity classifier)
  obligation_outcomes_audit_corrected.csv
  obligation_outcomes_section_gated.csv
                                 section-gated sensitivity artifact (Appendix B.2)
  addressing_comment_types.csv   comment_id, docket_id, and commenter type under the
                                 permissive (headline) and ternary classifiers, for every
                                 comment that addresses at least one obligation. Titles
                                 are not written.
  finding3_obligations.csv       per high-engagement obligation: addressing counts by
                                 commenter type, org-majority flag, and outcome before and
                                 after the blind audit (same construction as code/20, code/27)
  finding3_loo_sensitivity.csv   per-docket Finding 3 leave-one-out (runs code/19 --csv-out)
  layer_c_indicator_prevalence.csv
                                 per-indicator prevalence of the 23-indicator schema, only if
                                 data/processed/layer_c_full.csv is present
and checks the Finding 3 cells against the paper (43/15/28/30 and 13/2/57/43).
"""
from __future__ import annotations

import glob
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
from submitter_classification import classify_title  # noqa: E402

PROC = Path("data/processed")
SE, MO = "SURVIVED-edited", "MODIFIED"
PERMISSIVE = "submitted by|on behalf|written by"   # headline rule in code/20
LAYER_C = [
    "names_targeted_provision", "states_requested_change", "provides_example",
    "provides_legal_or_empirical_background",
    "expresses_explicit_support", "expresses_explicit_opposition",
    "burdensome", "lacks_flexibility", "not_sufficient_time", "conflicting_interests",
    "disputed_information", "legal_challenge", "overreach", "requests_clarification",
    "lacks_clarity", "seeks_exclusion", "too_broad", "too_narrow",
    "technical_scientific_frame_present", "legal_statutory_frame_present",
    "justice_equity_frame_present", "economic_cost_benefit_frame_present",
    "lived_experience_frame_present",
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1]).expanduser() / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Matcher output: IDs and labels only.
    files = sorted(glob.glob(str(PROC / "stage4" / "*.parquet")))
    m = pd.concat([pd.read_parquet(f, columns=["comment_id", "obligation_id", "addressed", "stance"])
                   for f in files], ignore_index=True)
    m.to_csv(out / "stage4_matches.csv.gz", index=False, compression="gzip")
    print(f"stage4_matches.csv.gz: {len(m):,} pairs from {len(files)} dockets")

    # 2. Outcome tables (no text columns are written by code/17 or code/27).
    variants = {
        "obligation_outcomes.csv": PROC / "path_a_obligation_outcomes.parquet",
        "obligation_outcomes_audit_corrected.csv": PROC / "path_a_obligation_outcomes_audit_corrected.parquet",
        "obligation_outcomes_section_gated.csv": PROC / "path_a_obligation_outcomes.parquet.cfr_restricted.bak",
    }
    tables = {}
    for name, path in variants.items():
        df = pd.read_parquet(path)
        df.to_csv(out / name, index=False)
        tables[name] = df
        print(f"{name}: {len(df):,} rows; columns {list(df.columns)}")

    # 3. Commenter types for addressing comments. Titles are read but never written.
    addressed = m[m["addressed"] == True]  # noqa: E712
    ids = set(addressed["comment_id"])
    parts = []
    for f in sorted(glob.glob(str(PROC / "comments_augmented" / "EPA_*.parquet"))):
        c = pd.read_parquet(f, columns=["document_id", "title"])
        parts.append(c[c["document_id"].isin(ids)])
    raw = pd.concat(parts, ignore_index=True)
    title = raw["title"].fillna("")
    raw["kind"] = title.str.contains(PERMISSIVE, case=False, regex=True).map(
        {True: "individual", False: "organizational"})
    types = pd.DataFrame({
        "comment_id": raw["document_id"],
        "docket_id": raw["document_id"].str.rsplit("-", n=1).str[0],
        "commenter_type_permissive": raw["kind"],
        "commenter_type_ternary": title.map(classify_title),
    }).drop_duplicates("comment_id")
    types.to_csv(out / "addressing_comment_types.csv", index=False)
    print(f"addressing_comment_types.csv: {len(types):,} comments "
          f"({len(ids - set(types['comment_id']))} addressing IDs not found)")

    # 4. Finding 3 per-obligation table, built exactly as in code/20 and code/27.
    joined = addressed.merge(raw[["document_id", "kind"]], left_on="comment_id",
                             right_on="document_id", how="left")
    agg = joined.groupby("obligation_id").agg(
        n_addressing=("comment_id", "nunique"),
        n_org=("kind", lambda s: (s == "organizational").sum()),
        n_ind=("kind", lambda s: (s == "individual").sum()),
    ).reset_index()
    agg["pct_org"] = agg["n_org"] / (agg["n_org"] + agg["n_ind"]).replace(0, 1)
    agg["org_majority"] = agg["pct_org"] > 0.5
    base = tables["obligation_outcomes.csv"][["obligation_id", "docket_id", "outcome_state"]]
    corrected = tables["obligation_outcomes_audit_corrected.csv"][["obligation_id", "outcome_state"]].rename(
        columns={"outcome_state": "outcome_state_audit_corrected"})
    f3 = base.merge(corrected, on="obligation_id", how="left").merge(agg, on="obligation_id", how="inner")
    f3 = f3[f3["n_addressing"] >= 5]
    f3.to_csv(out / "finding3_obligations.csv", index=False)
    for col, paper in (("outcome_state", (43, 15, 28, 30)),
                       ("outcome_state_audit_corrected", (13, 2, 57, 43))):
        s = f3[f3[col].isin([SE, MO])]
        got = tuple(int(((s[col] == o) & (s["org_majority"] == g)).sum())
                    for o in (SE, MO) for g in (True, False))
        print(f"Finding 3 cells, {col}: {got} " + ("OK" if got == paper else f"MISMATCH (paper {paper})"))

    # 5. Finding 3 leave-one-docket-out, per docket.
    run = subprocess.run([sys.executable, str(HERE / "19_finding3_loo_sensitivity.py"), "--csv-out"],
                         capture_output=True, text=True)
    loo = PROC / "finding3_loo_sensitivity.csv"
    if run.returncode == 0 and loo.exists():
        shutil.copy(loo, out / loo.name)
        print("finding3_loo_sensitivity.csv: copied")
    else:
        print("code/19 did not finish; finding3_loo_sensitivity.csv not written\n" + run.stderr[-800:])

    # 6. Per-indicator prevalence of the rhetoric schema (Appendix A.2).
    lc = PROC / "layer_c_full.csv"
    if lc.exists():
        d = pd.read_csv(lc, usecols=lambda c: c in LAYER_C)
        prev = pd.DataFrame({"indicator": LAYER_C, "n_coded": len(d),
                             "prevalence_pct": [round(100 * pd.to_numeric(d[c]).mean(), 1) for c in LAYER_C]})
        prev.to_csv(out / "layer_c_indicator_prevalence.csv", index=False)
        opp = prev.set_index("indicator").loc["expresses_explicit_opposition", "prevalence_pct"]
        print(f"layer_c_indicator_prevalence.csv: {len(d):,} comments; explicit opposition {opp}% (paper 59.5%)")
    else:
        print("layer_c_full.csv not found here; the per-indicator table needs the Layer C output file")


if __name__ == "__main__":
    main()
