"""
18_stage4_audit_reconciliation.py — Stage 4 blind audit reconciliation.

Reads:
  data/processed/stage4_validation_audit_bruce.csv  (Bruce's 100 codes; BRUCE_* filled)
  data/processed/stage4_validation_audit_yue.csv    (Yue's 100 codes; YUE_* filled)

Computes:
  - Pair_id alignment check (both files must have same 100 pairs in same order)
  - Marginal distribution comparison (ADDRESSED + STANCE)
  - Cohen's kappa on ADDRESSED (binary)
  - Cohen's kappa on STANCE (multi-class, on full sample + on addressed-by-both subset)
  - Confusion matrices
  - Disagreement list (for Friday reconciliation discussion)

Optional:
  --with-llm-key  : also load stage4_validation_audit_KEY.csv and compare
                    each human to the LLM matcher's labels.

Usage:
  python3 code/18_stage4_audit_reconciliation.py
  python3 code/18_stage4_audit_reconciliation.py --with-llm-key

No external dependencies — pure Python stdlib (no scipy needed).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

BRUCE_PATH = Path("data/processed/stage4_validation_audit_bruce.csv")
YUE_PATH = Path("data/processed/stage4_validation_audit_yue.csv")
KEY_PATH = Path("data/processed/stage4_validation_audit_KEY.csv")


def load_codes(path: Path, prefix: str) -> List[Dict[str, str]]:
    """Load audit CSV, return list of dicts with keys: pair_id, addressed, stance, notes."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Cannot run reconciliation.")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "pair_id": r["pair_id"],
                "addressed": r[f"{prefix}_ADDRESSED"].strip().upper(),
                "stance": r[f"{prefix}_STANCE"].strip().upper(),
                "notes": r.get(f"{prefix}_NOTES", "").strip(),
            })
    return rows


def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> Tuple[float, float, float]:
    """Compute Cohen's kappa. Returns (kappa, observed_agreement, expected_agreement).
    Pure stdlib; works for any categorical label set."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    categories = sorted(set(labels_a) | set(labels_b))
    # Observed agreement
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # Expected agreement under marginals independence
    p_e = 0.0
    for cat in categories:
        p_a = labels_a.count(cat) / n
        p_b = labels_b.count(cat) / n
        p_e += p_a * p_b
    if p_e == 1.0:
        kappa = 1.0 if p_o == 1.0 else float("nan")
    else:
        kappa = (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e


def confusion_matrix(labels_a: List[str], labels_b: List[str], cats: List[str]) -> List[List[int]]:
    """Build a confusion matrix: rows = rater A categories, cols = rater B categories."""
    mat = [[0] * len(cats) for _ in cats]
    idx = {c: i for i, c in enumerate(cats)}
    for a, b in zip(labels_a, labels_b):
        if a in idx and b in idx:
            mat[idx[a]][idx[b]] += 1
    return mat


def print_confusion(cats: List[str], mat: List[List[int]], a_label: str, b_label: str) -> None:
    col_w = max(max(len(c) for c in cats), 4)
    header = " " * (col_w + 2) + " | ".join(f"{c:>{col_w}}" for c in cats) + "  |  total"
    print(f"  {a_label} (rows) x {b_label} (cols):")
    print("  " + header)
    print("  " + "-" * len(header))
    for i, c in enumerate(cats):
        row = mat[i]
        total = sum(row)
        print(f"  {c:<{col_w}}  " + " | ".join(f"{v:>{col_w}}" for v in row) + f"  |  {total}")
    col_totals = [sum(mat[i][j] for i in range(len(cats))) for j in range(len(cats))]
    grand = sum(col_totals)
    print("  " + "-" * len(header))
    print(f"  {'total':<{col_w}}  " + " | ".join(f"{v:>{col_w}}" for v in col_totals) + f"  |  {grand}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm-key", action="store_true",
                        help="Also compare both humans to LLM matcher KEY.csv")
    args = parser.parse_args()

    print("=" * 72)
    print("Stage 4 blind audit reconciliation")
    print("=" * 72)

    bruce = load_codes(BRUCE_PATH, "BRUCE")
    yue = load_codes(YUE_PATH, "YUE")

    if len(bruce) != len(yue):
        sys.exit(f"ERROR: row count mismatch — bruce={len(bruce)}, yue={len(yue)}")
    for i, (b, y) in enumerate(zip(bruce, yue)):
        if b["pair_id"] != y["pair_id"]:
            sys.exit(f"ERROR: pair_id mismatch at row {i}: bruce={b['pair_id']!r} yue={y['pair_id']!r}")
    print(f"\n[OK] Pair alignment: 100/100 pair_ids match in order. n = {len(bruce)}.")

    # --- ADDRESSED analysis ---
    print("\n" + "-" * 72)
    print("ADDRESSED (binary TRUE/FALSE)")
    print("-" * 72)
    b_addr = [r["addressed"] for r in bruce]
    y_addr = [r["addressed"] for r in yue]

    b_dist = Counter(b_addr)
    y_dist = Counter(y_addr)
    print(f"\n  Bruce marginal: TRUE={b_dist['TRUE']}, FALSE={b_dist['FALSE']}")
    print(f"  Yue   marginal: TRUE={y_dist['TRUE']}, FALSE={y_dist['FALSE']}")

    kappa_a, p_o_a, p_e_a = cohens_kappa(b_addr, y_addr)
    print(f"\n  Observed agreement: {p_o_a:.3f} ({int(p_o_a * len(bruce))}/{len(bruce)} pairs)")
    print(f"  Expected agreement (chance): {p_e_a:.3f}")
    print(f"  Cohen's kappa: {kappa_a:.3f}")
    print(f"\n  Interpretation (Landis & Koch 1977 conventions):")
    print(f"    > 0.80 = almost perfect    0.61-0.80 = substantial")
    print(f"    0.41-0.60 = moderate       0.21-0.40 = fair")

    print()
    print_confusion(["TRUE", "FALSE"],
                    confusion_matrix(b_addr, y_addr, ["TRUE", "FALSE"]),
                    "Bruce", "Yue")

    # --- STANCE analysis ---
    print("\n" + "-" * 72)
    print("STANCE (5-class on full sample)")
    print("-" * 72)
    stance_cats = ["SUPPORTING", "OPPOSING", "SUGGESTING_MODIFICATION", "NONE"]
    b_st = [r["stance"] for r in bruce]
    y_st = [r["stance"] for r in yue]

    print(f"\n  Bruce marginal: " + ", ".join(f"{c}={Counter(b_st).get(c, 0)}" for c in stance_cats))
    print(f"  Yue   marginal: " + ", ".join(f"{c}={Counter(y_st).get(c, 0)}" for c in stance_cats))

    kappa_s, p_o_s, p_e_s = cohens_kappa(b_st, y_st)
    print(f"\n  Observed agreement: {p_o_s:.3f}")
    print(f"  Expected agreement: {p_e_s:.3f}")
    print(f"  Cohen's kappa (full): {kappa_s:.3f}")

    print()
    print_confusion(stance_cats, confusion_matrix(b_st, y_st, stance_cats), "Bruce", "Yue")

    # --- STANCE on addressed-by-both subset ---
    both_addr_idx = [i for i in range(len(bruce))
                     if bruce[i]["addressed"] == "TRUE" and yue[i]["addressed"] == "TRUE"]
    if both_addr_idx:
        print(f"\n--- STANCE on pairs addressed=TRUE by both (n={len(both_addr_idx)}) ---")
        b_st_sub = [bruce[i]["stance"] for i in both_addr_idx]
        y_st_sub = [yue[i]["stance"] for i in both_addr_idx]
        addressed_stance_cats = ["SUPPORTING", "OPPOSING", "SUGGESTING_MODIFICATION"]
        kappa_ss, p_o_ss, p_e_ss = cohens_kappa(b_st_sub, y_st_sub)
        print(f"  Observed agreement: {p_o_ss:.3f}")
        print(f"  Expected agreement: {p_e_ss:.3f}")
        print(f"  Cohen's kappa (addressed-by-both): {kappa_ss:.3f}")
        print()
        print_confusion(addressed_stance_cats,
                        confusion_matrix(b_st_sub, y_st_sub, addressed_stance_cats),
                        "Bruce", "Yue")

    # --- Disagreement list ---
    print("\n" + "-" * 72)
    print("DISAGREEMENT LIST (for Friday reconciliation)")
    print("-" * 72)
    disagreements = []
    for b, y in zip(bruce, yue):
        if b["addressed"] != y["addressed"] or b["stance"] != y["stance"]:
            disagreements.append((b, y))
    print(f"\n  Total disagreements: {len(disagreements)}/{len(bruce)}")
    addr_disagree = sum(1 for b, y in disagreements if b["addressed"] != y["addressed"])
    stance_only = len(disagreements) - addr_disagree
    print(f"    ADDRESSED disagreements: {addr_disagree}")
    print(f"    STANCE-only disagreements (both agree on addressed): {stance_only}")

    print(f"\n  Disagreement details (first 30):")
    print(f"  {'pair_id':<45} {'Bruce':<32} {'Yue':<32}")
    print("  " + "-" * 110)
    for b, y in disagreements[:30]:
        bc = f"{b['addressed']}/{b['stance']}"
        yc = f"{y['addressed']}/{y['stance']}"
        print(f"  {b['pair_id']:<45} {bc:<32} {yc:<32}")
    if len(disagreements) > 30:
        print(f"  ... ({len(disagreements) - 30} more disagreements not shown)")

    # --- Optional: compare to LLM matcher key ---
    if args.with_llm_key:
        print("\n" + "-" * 72)
        print("HUMAN vs LLM MATCHER (comparison from KEY.csv)")
        print("-" * 72)
        if not KEY_PATH.exists():
            print(f"  KEY.csv not found at {KEY_PATH}; skipping.")
        else:
            with open(KEY_PATH) as f:
                key = list(csv.DictReader(f))
            if len(key) != len(bruce):
                print(f"  WARN: KEY row count {len(key)} != audit count {len(bruce)}")
            else:
                key_map = {r["pair_id"]: r for r in key}
                llm_addr = []
                llm_st = []
                for b in bruce:
                    k = key_map.get(b["pair_id"], {})
                    # KEY.csv uses llm_addressed (True/False) and llm_stance (UPPER)
                    llm_addr.append(str(k.get("llm_addressed", "")).strip().upper())
                    llm_st.append(str(k.get("llm_stance", "")).strip().upper())

                k_ba, _, _ = cohens_kappa(b_addr, llm_addr)
                k_ya, _, _ = cohens_kappa(y_addr, llm_addr)
                print(f"\n  Cohen's kappa Bruce-vs-LLM (ADDRESSED): {k_ba:.3f}")
                print(f"  Cohen's kappa Yue-vs-LLM   (ADDRESSED): {k_ya:.3f}")

                k_bs, _, _ = cohens_kappa(b_st, llm_st)
                k_ys, _, _ = cohens_kappa(y_st, llm_st)
                print(f"  Cohen's kappa Bruce-vs-LLM (STANCE):    {k_bs:.3f}")
                print(f"  Cohen's kappa Yue-vs-LLM   (STANCE):    {k_ys:.3f}")

    print("\n" + "=" * 72)
    print("Reconciliation summary complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
