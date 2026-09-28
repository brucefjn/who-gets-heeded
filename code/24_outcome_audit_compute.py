"""Audit 1 compute — Outcome-state classifier validation (per-rater file pattern).

Reads:
  - data/processed/outcome_audit_yue.csv        (Yue's per-rater file with
                                                  rater_1_label filled in)
  - data/processed/outcome_audit_bruce.csv      (Bruce's per-rater file with
                                                  rater_2_label filled in)
  - data/processed/outcome_audit_classifier_labels.csv (truth file)
  - data/processed/outcome_audit_reconciled.csv (optional, for the second
                                                  --with-adjudicated pass)

Joins the two per-rater files on obligation_id and computes:

  HEADLINE (most relevant for §6.x of paper)
    - Inter-rater Cohen's κ on the load-bearing SURVIVED-edited vs MODIFIED
      binary (the boundary that drives §6.4 Finding 3 equity claim)
    - Exact-agreement % on the SE/MO binary

  GENERAL
    - Inter-rater Cohen's κ on the full 5-class label
    - Exact-agreement % on 5-class
    - 5x5 confusion matrix (rater 1 × rater 2)

  CLASSIFIER EVAL (against EACH rater individually, NOT consensus-only —
  restricting to consensus inflates classifier agreement by excluding the
  hardest examples)
    - Classifier-vs-rater-1 κ (full audit set)
    - Classifier-vs-rater-2 κ (full audit set)
    - Classifier-vs-adjudicated-gold κ (after manual reconciliation, run with
      --with-adjudicated flag)
    - Per-class classifier precision/recall against each rater

  TIER-SPECIFIC METRICS
    - Above metrics broken out by Tier 1 (F3-determining, n≈116) vs
      Tier 2 (general validation supplement, n≈34)
    - Directly addresses "is the classifier reliable on the exact subset
      that supports the main claim?" — the §6.4 equity finding

Usage (after both per-rater files are committed):
    python3 code/24_outcome_audit_compute.py
        # initial pass — reports inter-rater κ, classifier-vs-each-rater κ
        # flags disagreements for joint adjudication

    python3 code/24_outcome_audit_compute.py --with-adjudicated
        # after manual reconciliation: adds classifier-vs-adjudicated-gold κ
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
YUE_PATH = REPO_ROOT / "data" / "processed" / "outcome_audit_yue.csv"
BRUCE_PATH = REPO_ROOT / "data" / "processed" / "outcome_audit_bruce.csv"
TRUTH_PATH = REPO_ROOT / "data" / "processed" / "outcome_audit_classifier_labels.csv"
RESULTS_PATH = REPO_ROOT / "data" / "processed" / "outcome_audit_results.json"
RECONCILED_PATH = REPO_ROOT / "data" / "processed" / "outcome_audit_reconciled.csv"

OUTCOME_CLASSES = ["SURVIVED-unchanged", "SURVIVED-edited", "MODIFIED", "DROPPED", "NEW"]
HEADLINE_BINARY = ("SURVIVED-edited", "MODIFIED")


def load_rater_file(path: Path, rater_label_col: str) -> pd.DataFrame:
    """Load a per-rater file and return a slim DataFrame with obligation_id
    and the rater's label.

    rater_label_col: 'rater_1_label' for Yue's file, 'rater_2_label' for Bruce's.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Each rater must copy outcome_audit_BLIND.csv "
            f"to their per-rater file (outcome_audit_yue.csv or "
            f"outcome_audit_bruce.csv), code their labels, and commit."
        )
    df = pd.read_csv(path)
    if "obligation_id" not in df.columns:
        raise ValueError(f"{path} missing obligation_id column.")
    if rater_label_col not in df.columns:
        raise ValueError(f"{path} missing {rater_label_col} column.")
    notes_col = "rater_1_notes" if rater_label_col == "rater_1_label" else "rater_2_notes"
    cols = ["obligation_id", rater_label_col]
    if notes_col in df.columns:
        cols.append(notes_col)
    return df[cols].copy()


def compute_metrics(r1: pd.Series, r2: pd.Series, clf: pd.Series,
                    adjudicated: pd.Series | None = None,
                    label: str = "") -> dict:
    """Compute the inter-rater + classifier-vs-each-rater metrics for a subset."""
    out: dict = {"label": label, "n": int(len(r1))}

    # Inter-rater (5-class)
    out["kappa_5class_r1_r2"] = float(cohen_kappa_score(r1, r2))
    out["exact_agreement_5class"] = float((r1 == r2).mean())

    # Inter-rater on load-bearing binary (SE/MO)
    binary_mask = r1.isin(HEADLINE_BINARY) & r2.isin(HEADLINE_BINARY)
    n_binary = int(binary_mask.sum())
    out["n_se_vs_mo_pairs"] = n_binary
    if n_binary >= 2:
        out["kappa_se_vs_mo_r1_r2"] = float(cohen_kappa_score(r1[binary_mask], r2[binary_mask]))
        out["exact_se_vs_mo"] = float((r1[binary_mask] == r2[binary_mask]).mean())
    else:
        out["kappa_se_vs_mo_r1_r2"] = None
        out["exact_se_vs_mo"] = None

    # Classifier vs each rater individually (5-class, full audit set NOT consensus-only)
    out["kappa_classifier_vs_r1"] = float(cohen_kappa_score(clf, r1))
    out["kappa_classifier_vs_r2"] = float(cohen_kappa_score(clf, r2))
    # Classifier vs each rater on the SE/MO binary
    bin_r1_mask = r1.isin(HEADLINE_BINARY) & clf.isin(HEADLINE_BINARY)
    bin_r2_mask = r2.isin(HEADLINE_BINARY) & clf.isin(HEADLINE_BINARY)
    if bin_r1_mask.sum() >= 2:
        out["kappa_classifier_vs_r1_se_mo"] = float(
            cohen_kappa_score(clf[bin_r1_mask], r1[bin_r1_mask]))
    else:
        out["kappa_classifier_vs_r1_se_mo"] = None
    if bin_r2_mask.sum() >= 2:
        out["kappa_classifier_vs_r2_se_mo"] = float(
            cohen_kappa_score(clf[bin_r2_mask], r2[bin_r2_mask]))
    else:
        out["kappa_classifier_vs_r2_se_mo"] = None

    # Classifier vs adjudicated gold (if provided)
    if adjudicated is not None:
        gold_mask = adjudicated.notna() & (adjudicated != "") & (adjudicated != "DISAGREE")
        n_gold = int(gold_mask.sum())
        out["n_adjudicated_gold"] = n_gold
        if n_gold >= 2:
            out["kappa_classifier_vs_adjudicated"] = float(
                cohen_kappa_score(clf[gold_mask], adjudicated[gold_mask]))
            bin_gold_mask = (
                adjudicated[gold_mask].isin(HEADLINE_BINARY)
                & clf[gold_mask].isin(HEADLINE_BINARY)
            )
            if bin_gold_mask.sum() >= 2:
                out["kappa_classifier_vs_adjudicated_se_mo"] = float(cohen_kappa_score(
                    clf[gold_mask][bin_gold_mask],
                    adjudicated[gold_mask][bin_gold_mask]
                ))
            else:
                out["kappa_classifier_vs_adjudicated_se_mo"] = None
        else:
            out["kappa_classifier_vs_adjudicated"] = None
            out["kappa_classifier_vs_adjudicated_se_mo"] = None

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-adjudicated", action="store_true",
                        help="Include classifier-vs-adjudicated-gold metrics. Requires "
                             "the reconciled CSV to have 'adjudicated_label' column "
                             "filled in for previously-DISAGREE rows.")
    args = parser.parse_args()

    # === Load per-rater files ===
    print(f"Loading Yue's labels:   {YUE_PATH}")
    yue_df = load_rater_file(YUE_PATH, "rater_1_label")
    print(f"  {len(yue_df)} rows")

    print(f"Loading Bruce's labels: {BRUCE_PATH}")
    bruce_df = load_rater_file(BRUCE_PATH, "rater_2_label")
    print(f"  {len(bruce_df)} rows")

    # Sanity check: same obligation_ids
    yue_ids = set(yue_df["obligation_id"])
    bruce_ids = set(bruce_df["obligation_id"])
    if yue_ids != bruce_ids:
        only_yue = yue_ids - bruce_ids
        only_bruce = bruce_ids - yue_ids
        print(f"ERROR: per-rater files have different obligation_id sets.", file=sys.stderr)
        if only_yue:
            print(f"  Only in Yue's file ({len(only_yue)}): {list(only_yue)[:5]}...",
                  file=sys.stderr)
        if only_bruce:
            print(f"  Only in Bruce's file ({len(only_bruce)}): {list(only_bruce)[:5]}...",
                  file=sys.stderr)
        print(f"  Both raters should have started from outcome_audit_BLIND.csv.",
              file=sys.stderr)
        return 2

    print(f"Loading classifier truth: {TRUTH_PATH}")
    if not TRUTH_PATH.exists():
        print(f"ERROR: {TRUTH_PATH} not found.", file=sys.stderr)
        return 1
    truth = pd.read_csv(TRUTH_PATH)
    print(f"  {len(truth)} rows")

    # === Join ===
    merged = yue_df.merge(bruce_df, on="obligation_id", how="inner")
    merged = merged.merge(truth, on="obligation_id", how="left")
    print(f"\nJoined: {len(merged)} rows with both raters' labels + classifier truth")

    # Clean label columns
    merged["rater_1_label"] = merged["rater_1_label"].fillna("").astype(str).str.strip()
    merged["rater_2_label"] = merged["rater_2_label"].fillna("").astype(str).str.strip()
    merged["classifier_label"] = merged["classifier_label"].fillna("").astype(str).str.strip()

    n_missing_r1 = (merged["rater_1_label"] == "").sum()
    n_missing_r2 = (merged["rater_2_label"] == "").sum()
    if n_missing_r1 or n_missing_r2:
        print(f"WARN: missing labels — rater 1 (Yue): {n_missing_r1}, "
              f"rater 2 (Bruce): {n_missing_r2}", file=sys.stderr)

    coded = merged[(merged["rater_1_label"] != "") & (merged["rater_2_label"] != "")].copy()
    print(f"Analyzing {len(coded)} of {len(merged)} pairs with both labels present")

    r1 = coded["rater_1_label"]
    r2 = coded["rater_2_label"]
    clf = coded["classifier_label"]

    # Validate label values
    for name, s in [("rater 1 (Yue)", r1), ("rater 2 (Bruce)", r2)]:
        invalid = set(s.unique()) - set(OUTCOME_CLASSES)
        if invalid:
            print(f"ERROR: {name} has invalid labels: {invalid}", file=sys.stderr)
            return 3

    # === Load adjudicated labels if requested ===
    adjudicated = None
    if args.with_adjudicated:
        if not RECONCILED_PATH.exists():
            print(f"ERROR: --with-adjudicated set but {RECONCILED_PATH} not found.",
                  file=sys.stderr)
            print(f"       Run this script without --with-adjudicated first to produce "
                  f"the reconciliation file, then fill in adjudicated_label for DISAGREE rows.",
                  file=sys.stderr)
            return 4
        recon = pd.read_csv(RECONCILED_PATH)
        if "adjudicated_label" not in recon.columns:
            print(f"ERROR: reconciled CSV missing 'adjudicated_label' column.",
                  file=sys.stderr)
            return 4
        recon = recon[["obligation_id", "adjudicated_label"]]
        coded = coded.merge(recon, on="obligation_id", how="left")
        # Where raters agreed and no manual adjudication was needed, the adjudicated
        # label is whatever r1 = r2 says
        coded.loc[coded["adjudicated_label"].isna() | (coded["adjudicated_label"] == ""),
                  "adjudicated_label"] = coded["rater_1_label"]
        # After this step, if a row still has "DISAGREE" in adjudicated_label, it means
        # the rater needs to actually fill it in
        n_unresolved = (coded["adjudicated_label"] == "DISAGREE").sum()
        if n_unresolved > 0:
            print(f"ERROR: {n_unresolved} rows still flagged 'DISAGREE' in adjudicated_label.",
                  file=sys.stderr)
            print(f"       Open {RECONCILED_PATH.name} and replace 'DISAGREE' with the",
                  file=sys.stderr)
            print(f"       jointly-agreed label for each such row.", file=sys.stderr)
            return 5
        adjudicated = coded["adjudicated_label"]

    # === Headline metrics on full audit set ===
    print(f"\n{'=' * 60}")
    print(f"HEADLINE: full audit set (n = {len(coded)})")
    print(f"{'=' * 60}")
    full_metrics = compute_metrics(r1, r2, clf, adjudicated, label="full_audit_set")
    print(f"  Inter-rater κ (5-class):           {full_metrics['kappa_5class_r1_r2']:.4f}")
    if full_metrics['kappa_se_vs_mo_r1_r2'] is not None:
        print(f"  Inter-rater κ (SE vs MO binary):   "
              f"{full_metrics['kappa_se_vs_mo_r1_r2']:.4f} "
              f"(n={full_metrics['n_se_vs_mo_pairs']})")
    else:
        print(f"  Inter-rater κ (SE vs MO): insufficient pairs")
    print(f"  Exact agreement (5-class):         {full_metrics['exact_agreement_5class']:.1%}")
    print(f"  Classifier vs rater 1 (Yue) κ:     {full_metrics['kappa_classifier_vs_r1']:.4f}")
    print(f"  Classifier vs rater 2 (Bruce) κ:   {full_metrics['kappa_classifier_vs_r2']:.4f}")
    if full_metrics.get("kappa_classifier_vs_r1_se_mo") is not None:
        print(f"  Classifier vs Yue κ (SE/MO):       "
              f"{full_metrics['kappa_classifier_vs_r1_se_mo']:.4f}")
        print(f"  Classifier vs Bruce κ (SE/MO):     "
              f"{full_metrics['kappa_classifier_vs_r2_se_mo']:.4f}")
    if adjudicated is not None:
        if full_metrics.get("kappa_classifier_vs_adjudicated") is not None:
            print(f"  Classifier vs adjudicated κ:       "
                  f"{full_metrics['kappa_classifier_vs_adjudicated']:.4f}")
        if full_metrics.get("kappa_classifier_vs_adjudicated_se_mo") is not None:
            print(f"  Classifier vs adjudicated SE/MO κ: "
                  f"{full_metrics['kappa_classifier_vs_adjudicated_se_mo']:.4f}")

    # === Tier-specific: Tier 1 (F3-determining) vs Tier 2 (general) ===
    tier_results = {}
    for tier_name in ["T1_F3", "T2_general"]:
        tier_mask = coded["audit_tier"] == tier_name
        if tier_mask.sum() < 2:
            continue
        print(f"\n{'=' * 60}")
        print(f"TIER: {tier_name}  (n = {tier_mask.sum()})")
        print(f"{'=' * 60}")
        tier_r1, tier_r2, tier_clf = r1[tier_mask], r2[tier_mask], clf[tier_mask]
        tier_adj = adjudicated[tier_mask] if adjudicated is not None else None
        tier_m = compute_metrics(tier_r1, tier_r2, tier_clf, tier_adj, label=tier_name)
        tier_results[tier_name] = tier_m
        print(f"  Inter-rater κ (5-class):           {tier_m['kappa_5class_r1_r2']:.4f}")
        if tier_m.get('kappa_se_vs_mo_r1_r2') is not None:
            print(f"  Inter-rater κ (SE vs MO):          "
                  f"{tier_m['kappa_se_vs_mo_r1_r2']:.4f} "
                  f"(n={tier_m['n_se_vs_mo_pairs']})")
        print(f"  Classifier vs Yue κ:               {tier_m['kappa_classifier_vs_r1']:.4f}")
        print(f"  Classifier vs Bruce κ:             {tier_m['kappa_classifier_vs_r2']:.4f}")

    # === Confusion matrix (Yue × Bruce) ===
    print(f"\n{'=' * 60}")
    print(f"Confusion matrix (Yue rows × Bruce cols, full audit set)")
    print(f"{'=' * 60}")
    cm = confusion_matrix(r1, r2, labels=OUTCOME_CLASSES)
    cm_df = pd.DataFrame(cm, index=OUTCOME_CLASSES, columns=OUTCOME_CLASSES)
    print(cm_df.to_string())

    # === Per-class classifier precision/recall against each rater ===
    print(f"\n{'=' * 60}")
    print(f"Per-class classifier precision/recall (vs each rater, full audit set)")
    print(f"{'=' * 60}")
    per_class = {}
    for rater_name, rater in [("yue", r1), ("bruce", r2)]:
        p, r, f, support = precision_recall_fscore_support(
            rater, clf, labels=OUTCOME_CLASSES, zero_division=0.0,
        )
        per_class[rater_name] = []
        print(f"\n  Against rater {rater_name}:")
        for cls, prec, rec, supp in zip(OUTCOME_CLASSES, p, r, support):
            print(f"    {cls:24s}  precision = {prec:.3f}  recall = {rec:.3f}  n = {supp}")
            per_class[rater_name].append({
                "class": cls, "precision": float(prec), "recall": float(rec), "n": int(supp)
            })

    # === Save results ===
    results = {
        "full_audit_set": full_metrics,
        "tier_results": tier_results,
        "confusion_matrix_yue_x_bruce": cm.tolist(),
        "confusion_matrix_labels": OUTCOME_CLASSES,
        "classifier_per_class": per_class,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results: {RESULTS_PATH}")

    # === Reconciliation file ===
    disagree_mask = (r1 != r2)
    n_disagree = int(disagree_mask.sum())
    print(f"\n{n_disagree} disagreements flagged for joint adjudication.")

    if not args.with_adjudicated:
        print(f"Next steps:")
        print(f"  1. Open {RECONCILED_PATH.name}; for each row with adjudicated_label='DISAGREE',")
        print(f"     discuss with co-author and replace 'DISAGREE' with the agreed label.")
        print(f"  2. Re-run: python3 code/24_outcome_audit_compute.py --with-adjudicated")
        print(f"  3. The reported inter-rater κ above (from blind labels) does NOT change.")
        print(f"     Adjudication is for classifier evaluation only.")

        reconciled = coded[[
            "obligation_id", "audit_tier", "rater_1_label", "rater_2_label", "classifier_label"
        ]].copy()
        # Carry over notes if present
        for src_df, col in [(yue_df, "rater_1_notes"), (bruce_df, "rater_2_notes")]:
            if col in src_df.columns:
                reconciled = reconciled.merge(src_df[["obligation_id", col]],
                                              on="obligation_id", how="left")
        reconciled["adjudicated_label"] = ""
        reconciled.loc[r1 == r2, "adjudicated_label"] = r1[r1 == r2].values
        reconciled.loc[r1 != r2, "adjudicated_label"] = "DISAGREE"
        reconciled.to_csv(RECONCILED_PATH, index=False)
        print(f"Wrote reconciliation file: {RECONCILED_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
