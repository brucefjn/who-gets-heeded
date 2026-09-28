"""Audit 1 sampler — Outcome-state classifier validation, F3-aligned design.

DESIGN (revised per reviewer feedback):

  Tier 1 — F3-determining obligations. ALL 116 obligations that drive the §6.4
  Finding 3 Fisher's exact test (i.e., obligations with >=5 addressing commenters
  AND outcome_state in {SURVIVED-edited, MODIFIED}). Auditing these directly
  validates the load-bearing claim of the paper.

  Tier 2 — Supplementary stratified sample (~34 obligations) covering the
  remaining outcome classes (SURVIVED-unchanged, DROPPED, NEW) plus additional
  SURVIVED-edited/MODIFIED rows from outside the F3 engaged subset. This gives
  general 5-class classifier metrics.

  Total: ~150 obligations. The exact Tier 2 composition is configurable.

BLINDNESS FIX: the blind coding sheet does NOT include cosine_similarity. The
classifier is itself a cosine-threshold function, so exposing the cosine to
human coders would leak the classifier's decision boundary and invalidate
the audit. Coders see only proposed_text, final_text, docket_id, and (for
DROPPED/NEW) the CFR section reference for absence-checking context.

DROPPED/NEW caveat: these classes can only be audited at the "plausibility
of stated absence" level — the human coder verifies that the row is presented
consistently with the classifier's absence claim, not that the obligation is
truly absent from the full rule text. The rubric reflects this.

Inputs (must exist on local disk; gitignored from repo):
  - data/processed/path_a_obligation_outcomes.parquet
  - data/processed/stage4/*.parquet  (the obligation-comment matcher output)
  - data/processed/path_a_obligations_verified_*.csv (per-anchor verified obls)

Outputs:
  - data/processed/outcome_audit_BLIND.csv         — canonical blind template
        (do NOT edit this; each rater copies it to a per-rater file)
  - data/processed/outcome_audit_classifier_labels.csv — truth file (separate)
  - data/processed/outcome_audit_sampling_log.json — record of what was sampled

Each rater then copies the BLIND template to a per-rater file:
  cp data/processed/outcome_audit_BLIND.csv data/processed/outcome_audit_yue.csv
  cp data/processed/outcome_audit_BLIND.csv data/processed/outcome_audit_bruce.csv
and fills in rater_1_label (in the yue file) or rater_2_label (in the bruce
file) only — leaving the other rater's column empty. This matches the
Stage 4 audit pattern (per-rater files joined at compute time) and avoids
the merge conflicts that would happen if both raters edited a shared sheet.

Usage:
    python3 code/23_outcome_audit_sampler.py
    python3 code/23_outcome_audit_sampler.py --tier2-size 50  # bigger Tier 2
    python3 code/23_outcome_audit_sampler.py --seed 99
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTCOMES_PARQUET = REPO_ROOT / "data" / "processed" / "path_a_obligation_outcomes.parquet"
STAGE4_GLOB = REPO_ROOT / "data" / "processed" / "stage4" / "*.parquet"
VERIFIED_DIR = REPO_ROOT / "data" / "processed"
BLIND_OUT = REPO_ROOT / "data" / "processed" / "outcome_audit_BLIND.csv"
TRUTH_OUT = REPO_ROOT / "data" / "processed" / "outcome_audit_classifier_labels.csv"
LOG_OUT = REPO_ROOT / "data" / "processed" / "outcome_audit_sampling_log.json"

OUTCOME_CLASSES = ["SURVIVED-unchanged", "SURVIVED-edited", "MODIFIED", "DROPPED", "NEW"]
DEFAULT_SEED = 20260516
F3_ENGAGEMENT_THRESHOLD = 5  # matches §6.4 / code/20_finding3_analysis.py


def load_obligation_texts() -> dict[str, dict]:
    """Load obligation_text + CFR section for every verified obligation across
    all anchor dockets. Returns dict obligation_id -> {text, cfr_section, docket}.

    obligation_id format matches code/lib/stage4_embedding_prefilter.py::
    obligation_id_of, which writes: f"{docket}__{rule}__{cidx}_{sidx}"
    """
    out: dict[str, dict] = {}
    # Filename pattern: path_a_obligations_verified_<DOCKET>_<RULE>.csv
    # where DOCKET typically contains hyphens (e.g., EPA-HQ-OAR-2018-0775)
    # and RULE is "proposed" or "final".
    import re
    fname_pattern = re.compile(
        r"^path_a_obligations_verified_(?P<docket>.+?)_(?P<rule>proposed|final)$"
    )

    for csv_path in sorted(VERIFIED_DIR.glob("path_a_obligations_verified_*.csv")):
        stem = csv_path.stem
        m = fname_pattern.match(stem)
        if not m:
            print(f"WARN: could not parse {stem}", file=sys.stderr)
            continue
        docket_id = m.group("docket")
        rule = m.group("rule")

        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            if not row.get("verified", False):
                continue
            text = row.get("obligation_text", "")
            if pd.isna(text) or not text:
                continue
            cidx = row.get("candidate_idx")
            sidx = row.get("split_idx", 0)
            if pd.isna(cidx):
                continue
            # Match obligation_id_of() format exactly:
            # f"{docket}__{rule}__{cidx}_{sidx}"
            try:
                cidx_int = int(cidx)
                sidx_int = int(sidx) if not pd.isna(sidx) else 0
            except (ValueError, TypeError):
                # fall back to string form if not parseable as int
                cidx_int = cidx
                sidx_int = sidx if not pd.isna(sidx) else 0
            obl_id = f"{docket_id}__{rule}__{cidx_int}_{sidx_int}"
            out[obl_id] = {
                "text": text,
                "cfr_section": row.get("cfr_section", ""),
                "cfr_part": row.get("cfr_part", ""),
                "docket_id": docket_id,
            }
    return out


def reconstruct_f3_engaged_subset(outcomes: pd.DataFrame) -> set[str]:
    """Replicate the §6.4 engaged-subset construction from code/20_finding3_analysis.py.

    Returns the set of obligation_ids in the n=116 Finding-3 contingency:
    obligations with >= F3_ENGAGEMENT_THRESHOLD addressing commenters and
    outcome_state in {SURVIVED-edited, MODIFIED}.
    """
    match_files = sorted(REPO_ROOT.glob("data/processed/stage4/*.parquet"))
    if not match_files:
        raise FileNotFoundError(
            f"No Stage 4 matcher parquets at {REPO_ROOT}/data/processed/stage4/*.parquet. "
            f"This script needs them to reconstruct the F3 engaged subset."
        )
    df_matches = pd.concat([pd.read_parquet(p) for p in match_files])
    addressed = df_matches[df_matches["addressed"] == True]  # noqa: E712
    agg = addressed.groupby("obligation_id").agg(
        n_addressing=("comment_id", "nunique")
    ).reset_index()
    merged = outcomes.merge(agg, on="obligation_id", how="left").fillna({"n_addressing": 0})
    engaged = merged[
        (merged["n_addressing"] >= F3_ENGAGEMENT_THRESHOLD)
        & (merged["outcome_state"].isin(["SURVIVED-edited", "MODIFIED"]))
    ]
    return set(engaged["obligation_id"].tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier2-size", type=int, default=34,
                        help="Number of supplementary non-F3 obligations to sample "
                             "for general 5-class classifier metrics (default 34, "
                             "for ~150 total assuming F3 subset is ~116).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED}, do NOT change "
                             f"unless re-sampling from scratch).")
    args = parser.parse_args()

    if not OUTCOMES_PARQUET.exists():
        print(f"ERROR: {OUTCOMES_PARQUET} not found. "
              f"Run code/17_compute_obligation_outcomes.py first.", file=sys.stderr)
        return 1

    print(f"Loading outcomes parquet: {OUTCOMES_PARQUET}")
    df = pd.read_parquet(OUTCOMES_PARQUET)
    print(f"  {len(df):,} obligations across {df['docket_id'].nunique()} dockets")
    print(f"\nOutcome state distribution:")
    print(df["outcome_state"].value_counts().to_string())

    print(f"\n=== Tier 1: Reconstructing F3 engaged subset ===")
    f3_ids = reconstruct_f3_engaged_subset(df)
    print(f"  F3-determining obligations: n = {len(f3_ids)}")
    df["in_f3"] = df["obligation_id"].isin(f3_ids)
    tier1 = df[df["in_f3"]].copy()
    tier1["audit_tier"] = "T1_F3"
    print(f"  Tier 1 audit subset: {len(tier1)} obligations")
    print(f"  Per-outcome: {tier1['outcome_state'].value_counts().to_dict()}")

    print(f"\n=== Tier 2: Supplementary stratified sample (target n = {args.tier2_size}) ===")
    rng = np.random.default_rng(args.seed)
    non_f3 = df[~df["in_f3"]].copy()
    # Allocate Tier 2 proportional-ish: emphasize outcome classes underrepresented in F3
    # F3 is by definition SURVIVED-edited + MODIFIED only. Tier 2 should rebalance.
    tier2_allocation = {
        "SURVIVED-unchanged": 12,  # general SU validation
        "SURVIVED-edited": 8,      # cross-validate F3-internal vs non-F3 SE
        "MODIFIED": 6,             # cross-validate F3-internal vs non-F3 MO
        "DROPPED": 4,              # plausibility-of-absence audit
        "NEW": 4,                  # plausibility-of-absence audit
    }
    total_t2 = sum(tier2_allocation.values())
    if total_t2 != args.tier2_size:
        # rescale proportionally to hit the target
        scale = args.tier2_size / total_t2
        tier2_allocation = {k: max(1, round(v * scale)) for k, v in tier2_allocation.items()}

    tier2_samples = []
    for outcome, n in tier2_allocation.items():
        class_df = non_f3[non_f3["outcome_state"] == outcome]
        n_target = min(n, len(class_df))
        if n_target == 0:
            print(f"  WARN: no non-F3 rows for {outcome}", file=sys.stderr)
            continue
        if n_target < n:
            print(f"  WARN: requested {n} for {outcome}, only {n_target} available")
        sample = class_df.sample(n=n_target, random_state=rng.integers(0, 2**32))
        tier2_samples.append(sample)
    tier2 = pd.concat(tier2_samples, ignore_index=True)
    tier2["audit_tier"] = "T2_general"
    print(f"  Tier 2 supplementary: {len(tier2)} obligations")
    print(f"  Per-outcome: {tier2['outcome_state'].value_counts().to_dict()}")

    audit_set = pd.concat([tier1, tier2], ignore_index=True)
    audit_set = audit_set.sample(frac=1, random_state=rng.integers(0, 2**32))  # shuffle order
    audit_set = audit_set.reset_index(drop=True)

    print(f"\n=== Total audit set ===")
    print(f"  Total: {len(audit_set)} obligations across {audit_set['docket_id'].nunique()} dockets")
    print(f"  Tier 1 (F3-determining): {(audit_set['audit_tier'] == 'T1_F3').sum()}")
    print(f"  Tier 2 (general validation): {(audit_set['audit_tier'] == 'T2_general').sum()}")
    print(f"  Per-outcome: {audit_set['outcome_state'].value_counts().to_dict()}")

    # === Load obligation texts ===
    print(f"\nLoading obligation_text from verified CSVs...")
    texts = load_obligation_texts()
    print(f"  Loaded {len(texts):,} obligation_text entries")

    def get_proposed_text(row):
        if row["outcome_state"] == "NEW":
            return ""  # NEW has no proposed counterpart
        return texts.get(row["obligation_id"], {}).get("text", "")

    def get_final_text(row):
        if row["outcome_state"] == "DROPPED":
            return ""  # DROPPED has no final counterpart
        if row["outcome_state"] == "NEW":
            return texts.get(row["obligation_id"], {}).get("text", "")
        return texts.get(row.get("matched_final_obligation_id", ""), {}).get("text", "")

    def get_cfr_ref(row):
        # For DROPPED/NEW provide the CFR section reference so the human coder
        # can spot-check the FR text for the plausibility-of-absence check.
        info = texts.get(row["obligation_id"], {})
        cfr_part = info.get("cfr_part", "")
        cfr_section = info.get("cfr_section", "")
        if cfr_part and cfr_section:
            return f"40 CFR {cfr_part} § {cfr_section}"
        return ""

    audit_set["proposed_obligation_text"] = audit_set.apply(get_proposed_text, axis=1)
    audit_set["final_obligation_text"] = audit_set.apply(get_final_text, axis=1)
    audit_set["cfr_reference"] = audit_set.apply(get_cfr_ref, axis=1)

    # === Build BLIND coding sheet ===
    # NB: NO cosine_similarity column. Classifier is a cosine-threshold function,
    # so exposing cosine to human coders would invalidate blindness.
    blind_cols = [
        "obligation_id",
        "docket_id",
        "cfr_reference",
        "proposed_obligation_text",
        "final_obligation_text",
    ]
    blind = audit_set[blind_cols].copy()
    blind["rater_1_label"] = ""  # Yue
    blind["rater_2_label"] = ""  # Bruce
    blind["rater_1_notes"] = ""
    blind["rater_2_notes"] = ""

    blind.to_csv(BLIND_OUT, index=False)
    print(f"\nWrote blind coding sheet:    {BLIND_OUT}")
    print(f"  Columns: {list(blind.columns)}")
    print(f"  (NOTE: cosine_similarity intentionally excluded to preserve blindness)")

    truth = audit_set[["obligation_id", "outcome_state", "audit_tier"]].copy()
    truth = truth.rename(columns={"outcome_state": "classifier_label"})
    truth.to_csv(TRUTH_OUT, index=False)
    print(f"Wrote classifier-truth file: {TRUTH_OUT}")

    # Sampling log
    log = {
        "seed": args.seed,
        "tier2_size_requested": args.tier2_size,
        "n_total": int(len(audit_set)),
        "n_tier1_f3": int((audit_set["audit_tier"] == "T1_F3").sum()),
        "n_tier2_general": int((audit_set["audit_tier"] == "T2_general").sum()),
        "per_outcome_total": {k: int(v) for k, v in
                              audit_set["outcome_state"].value_counts().to_dict().items()},
        "per_outcome_tier1": {k: int(v) for k, v in
                              tier1["outcome_state"].value_counts().to_dict().items()},
        "per_outcome_tier2": {k: int(v) for k, v in
                              tier2["outcome_state"].value_counts().to_dict().items()},
    }
    LOG_OUT.write_text(json.dumps(log, indent=2))
    print(f"Wrote sampling log:          {LOG_OUT}")

    print(f"\nNext steps (per-rater file pattern, matches Stage 4 audit):")
    print(f"  1. Commit {BLIND_OUT.name} + {TRUTH_OUT.name} + sampling log; push.")
    print(f"  2. Each author copies the BLIND template to their per-rater file:")
    print(f"       cp data/processed/outcome_audit_BLIND.csv data/processed/outcome_audit_yue.csv")
    print(f"       cp data/processed/outcome_audit_BLIND.csv data/processed/outcome_audit_bruce.csv")
    print(f"  3. Both authors blind-code SEPARATE per-rater files (no peeking).")
    print(f"     - Yue fills rater_1_label in outcome_audit_yue.csv")
    print(f"     - Bruce fills rater_2_label in outcome_audit_bruce.csv")
    print(f"     - Each author leaves the OTHER rater's column empty")
    print(f"     - Use the rubric at notes/2026-05-16_outcome_audit_rubric.md")
    print(f"     - For DROPPED/NEW: use cfr_reference to spot-check FR text")
    print(f"  4. Each author commits their own per-rater file independently")
    print(f"     (no merge conflict; the two files are written by different authors)")
    print(f"  5. Run code/24_outcome_audit_compute.py when BOTH per-rater files committed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
