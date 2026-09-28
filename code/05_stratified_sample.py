"""
05_stratified_sample.py
Step 3 of Option 2 — stratified within-rule sampling.

For each anchor rule, sample 50 attachment-only comments stratified by
submitter type inferred from the Title field:
  - 40% organizational ('Comment from <Org>' / 'Comments of <Org>' / titled
                        with role+organization phrases)
  - 40% individual     ('Comment submitted by <person>' / 'Anonymous public
                        comment')
  - 20% other          (unmatched)

If a rule has fewer than 50 attachment-only comments, take all and document
the actual mix. Random seed is fixed (42) for reproducibility.

Inputs:
  data/processed/anchor_rules_locked.csv
  data/processed/comments/EPA_*.{parquet,pkl}

Output:
  data/processed/attachment_sample.csv
    columns: docket_id, document_id, title, attachment_files,
             submitter_kind, sample_index
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "comments"
ANCHOR_LOCKED = PROJECT_ROOT / "data" / "processed" / "anchor_rules_locked.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_sample.csv"

# Make code/lib importable without requiring a package install.
sys.path.insert(0, str(PROJECT_ROOT / "code" / "lib"))
from submitter_classification import classify_title  # noqa: E402

SEED = 42
TARGET_N = 50
MIX = {"organizational": 0.40, "individual": 0.40, "other": 0.20}


def stratified_sample(df: pd.DataFrame, n: int, mix: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows from df stratified by 'submitter_kind' per `mix`.

    For each stratum, take min(quota, available). If the strata together
    fall short of n (some strata exhausted), redistribute the deficit by
    drawing additional rows from rows not yet taken (any kind).
    """
    if len(df) <= n:
        return df.assign(sample_index=range(len(df))).copy()

    # Integer quotas that sum to exactly n
    quotas = {k: int(round(v * n)) for k, v in mix.items()}
    while sum(quotas.values()) > n:
        quotas[max(quotas, key=quotas.get)] -= 1
    while sum(quotas.values()) < n:
        quotas[max(quotas, key=quotas.get)] += 1

    base_seed = int(rng.integers(0, 2**31 - 1))
    sampled_parts: list[pd.DataFrame] = []

    # First pass: take from each stratum, capped at availability
    for i, (kind, q) in enumerate(quotas.items()):
        pool = df[df["submitter_kind"] == kind]
        take = min(q, len(pool))
        if take == 0:
            continue
        if take == len(pool):
            sampled_parts.append(pool)
        else:
            sampled_parts.append(pool.sample(take, random_state=(base_seed + i) % (2**31 - 1)))

    # Second pass: redistribute any deficit by sampling from rows not yet taken
    taken_idx = pd.concat(sampled_parts).index if sampled_parts else pd.Index([])
    deficit = n - len(taken_idx)
    if deficit > 0:
        leftover = df.drop(index=taken_idx, errors="ignore")
        if len(leftover) > 0:
            extra = leftover.sample(min(deficit, len(leftover)),
                                    random_state=(base_seed + 1000) % (2**31 - 1))
            sampled_parts.append(extra)

    if not sampled_parts:
        return df.head(n).assign(sample_index=range(min(n, len(df)))).copy()

    out = pd.concat(sampled_parts).drop_duplicates("document_id").reset_index(drop=True)
    out = out.head(n)
    out["sample_index"] = range(len(out))
    return out


def main() -> int:
    if not ANCHOR_LOCKED.exists():
        print(f"ERROR: anchor list not found at {ANCHOR_LOCKED}", file=sys.stderr)
        return 2
    anchors = pd.read_csv(ANCHOR_LOCKED)
    docket_ids = set(anchors["docket_id"])
    print(f"[anchors] {len(docket_ids)} dockets")

    paths = sorted(COMMENTS_DIR.glob("EPA_*.parquet")) + sorted(COMMENTS_DIR.glob("EPA_*.pkl"))
    seen = set(); parts = []
    for p in paths:
        if p.stem in seen: continue
        seen.add(p.stem)
        parts.append(pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_pickle(p))
    comments = pd.concat(parts, ignore_index=True)
    print(f"[load] comments: {len(comments):,}")

    pool = comments[
        (comments["docket_id"].isin(docket_ids))
        & (comments["is_attachment_only"])
    ].copy()
    pool["submitter_kind"] = pool["title"].apply(classify_title)
    print(f"[pool] attachment-only in anchor dockets: {len(pool):,}")
    print("        submitter-kind mix:")
    print(pool["submitter_kind"].value_counts().to_string())

    rng = np.random.default_rng(SEED)
    out_parts: list[pd.DataFrame] = []
    for docket in sorted(docket_ids):
        sub = pool[pool["docket_id"] == docket]
        if sub.empty:
            print(f"  WARN: docket {docket} has 0 attachment-only comments")
            continue
        sampled = stratified_sample(sub, TARGET_N, MIX, rng)
        sampled["docket_id"] = docket
        out_parts.append(sampled)
        print(f"  {docket}: pool={len(sub):>6,}  sampled={len(sampled):>3} "
              f"(org/ind/oth = "
              f"{int((sampled['submitter_kind']=='organizational').sum())}/"
              f"{int((sampled['submitter_kind']=='individual').sum())}/"
              f"{int((sampled['submitter_kind']=='other').sum())})")

    out = pd.concat(out_parts, ignore_index=True)
    out = out[["docket_id", "document_id", "title", "attachment_files",
               "submitter_kind", "sample_index"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\n[write] {OUT_PATH}  ({len(out):,} sampled attachments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
