#!/usr/bin/env python3
"""
tier_a_sample.py — generate the Tier A stratified sample for Layer C coding.

Pre-registration: notes/2026-05-13_tier_a_strata_preregistration.md
Random seed: 42 (locked)

Outputs (all under data/processed/):
  - tier_a_sample/EPA_sample.parquet     — sampled comments, Layer C orchestrator-ready
  - tier_a_anchors.csv                    — unique docket_ids in the sample (used as --anchors-csv)
  - tier_a_strata_assignments.csv         — full audit trail: per-comment stratum assignment
                                            + tercile boundaries record

Strata: (program_office × year_bucket × tercile), 9 × 4 × 3 = 108 cells.
Allocation: equal per cell at 926 target, full-population cap.

Run:
    python3 code/tier_a_sample.py

After this script lands, fire Layer C with:
    nohup python3 code/16_layer_c_run_full.py \\
        --all --no-batch --concurrency 80 --max-cost 400 \\
        --anchors-csv data/processed/tier_a_anchors.csv \\
        --comments-glob "data/processed/tier_a_sample/EPA_sample.parquet" \\
        --output data/processed/tier_a_layer_c_full.csv \\
        > /tmp/tier_a_fire.log 2>&1 &
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import pandas as pd

SEED = 42
TARGET_N = 100_000
PER_CELL_TARGET = 926  # ≈ 100,000 / 108
COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"
ANCHOR_CSV = "data/processed/anchor_rules_locked.csv"

OUT_DIR = Path("data/processed/tier_a_sample")
OUT_PARQUET = OUT_DIR / "EPA_sample.parquet"
OUT_ANCHORS = Path("data/processed/tier_a_anchors.csv")
OUT_STRATA = Path("data/processed/tier_a_strata_assignments.csv")

KNOWN_PROGRAM_OFFICES = {
    "OAR", "OW", "OPP", "OPPT", "OLEM",
    "OA", "OPA", "RCRA", "R08-OAR",
}

YEAR_BUCKETS = [
    ("Y1", 2002, 2008),  # Bush II
    ("Y2", 2009, 2016),  # Obama
    ("Y3", 2017, 2020),  # Trump
    ("Y4", 2021, 2023),  # Biden
]


def log(msg: str) -> None:
    print(f"[tier-a-sample] {msg}", flush=True)


def parse_program_office(docket_id: str) -> str:
    """Pre-registration §3.1 parsing rule."""
    parts = docket_id.split("-")
    if len(parts) < 3:
        return "OTHER"
    if parts[1] == "HQ":
        if len(parts) < 4:
            return "OTHER"
        po = parts[2]
    else:
        po = parts[1] + "-" + parts[2]
    return po if po in KNOWN_PROGRAM_OFFICES else "OTHER"


def parse_year(docket_id: str) -> int | None:
    """Year from the docket_id field (4-digit segment between program office and sequence)."""
    parts = docket_id.split("-")
    for p in parts:
        if len(p) == 4 and p.isdigit():
            year = int(p)
            if 1990 <= year <= 2030:
                return year
    return None


def year_bucket(year: int | None) -> str | None:
    if year is None:
        return None
    for label, lo, hi in YEAR_BUCKETS:
        if lo <= year <= hi:
            return label
    return None  # outside all buckets


def load_anchor_dockets() -> set[str]:
    with open(ANCHOR_CSV, "r", encoding="utf-8") as f:
        return {row["docket_id"] for row in csv.DictReader(f)}


def load_tier_a_frame(anchors: set[str]) -> pd.DataFrame:
    """Load all parquets, filter to NON-anchor analyzable comments."""
    cols = ["document_id", "docket_id", "comment", "text_source",
            "is_attachment_only", "title"]
    parts = []
    for path in sorted(glob.glob(COMMENTS_GLOB)):
        df = pd.read_parquet(path, columns=cols)
        # Tier A = non-anchor + same analyzability filters as Tier B
        m = df[~df["docket_id"].isin(anchors)
               & (df["text_source"] != "attachment_failed")]
        # Drop empty-text rows (mirrors orchestrator's _prepare_comment_rows)
        m = m[m["comment"].apply(
            lambda x: isinstance(x, str) and bool(x.strip())
        )]
        if not m.empty:
            parts.append(m)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ANCHORS.parent.mkdir(parents=True, exist_ok=True)

    anchors = load_anchor_dockets()
    log(f"loaded {len(anchors)} Tier B anchor dockets to exclude")

    log("loading Tier A frame (non-anchor analyzable comments)...")
    frame = load_tier_a_frame(anchors)
    log(f"Tier A frame size: {len(frame):,} comments "
        f"across {frame['docket_id'].nunique():,} non-anchor dockets")

    # Pre-registration §3.1: program office
    frame["program_office"] = frame["docket_id"].apply(parse_program_office)
    # Pre-registration §3.2: year bucket
    frame["year"] = frame["docket_id"].apply(parse_year)
    frame["year_bucket"] = frame["year"].apply(year_bucket)

    # Drop rows without a parseable year_bucket (outside 2002-2023 or unparseable docket_id)
    pre_drop = len(frame)
    frame = frame[frame["year_bucket"].notna()].reset_index(drop=True)
    log(f"dropped {pre_drop - len(frame):,} rows with unparseable year_bucket")

    # Pre-registration §3.3: per-docket comment-count tercile
    docket_sizes = frame.groupby("docket_id").size().rename("docket_size")
    tercile_boundaries = docket_sizes.quantile([1/3, 2/3]).values
    t1_max, t2_max = float(tercile_boundaries[0]), float(tercile_boundaries[1])
    log(f"tercile boundaries (docket comment-count): "
        f"T1 < {t1_max:.0f}, T2 = [{t1_max:.0f}, {t2_max:.0f}), T3 >= {t2_max:.0f}")

    def to_tercile(size: int) -> str:
        if size < t1_max:
            return "T1"
        if size < t2_max:
            return "T2"
        return "T3"

    frame = frame.merge(
        docket_sizes.reset_index(),
        on="docket_id", how="left",
    )
    frame["tercile"] = frame["docket_size"].apply(to_tercile)
    frame["cell"] = (frame["program_office"] + "|"
                     + frame["year_bucket"] + "|"
                     + frame["tercile"])

    cell_sizes = frame.groupby("cell").size().sort_values(ascending=False)
    log(f"populated cells: {len(cell_sizes)} of 108 maximum")
    log(f"cell size distribution: min={cell_sizes.min()}, "
        f"median={int(cell_sizes.median())}, "
        f"max={cell_sizes.max():,}")

    # Pre-registration §4: equal allocation per cell with full-population cap
    sampled_indices = []
    cells_at_capacity = 0
    cells_full_population = 0
    for cell, group in frame.groupby("cell"):
        n_in_cell = len(group)
        n_take = min(n_in_cell, PER_CELL_TARGET)
        if n_take == n_in_cell:
            cells_full_population += 1
        else:
            cells_at_capacity += 1
        s = group.sample(n=n_take, random_state=SEED)
        sampled_indices.extend(s.index.tolist())

    initial_n = len(sampled_indices)
    log(f"after initial pass: {initial_n:,} sampled "
        f"({cells_full_population} cells at full population, "
        f"{cells_at_capacity} cells at 926 cap)")

    # If under target, redistribute shortfall to cells with remaining population
    shortfall = TARGET_N - initial_n
    if shortfall > 0 and cells_at_capacity > 0:
        log(f"shortfall: {shortfall:,} comments to redistribute to "
            f"{cells_at_capacity} oversampled cells")
        # Cells that hit the cap (have additional population available)
        cap_cells = [cell for cell, group in frame.groupby("cell")
                     if len(group) > PER_CELL_TARGET]
        per_cap_cell_extra = shortfall // len(cap_cells)
        log(f"adding ~{per_cap_cell_extra} extra per capped cell")
        for cell in cap_cells:
            group = frame[frame["cell"] == cell]
            already_sampled = set(s for s in sampled_indices if s in group.index)
            remaining = group[~group.index.isin(already_sampled)]
            n_take = min(len(remaining), per_cap_cell_extra)
            if n_take > 0:
                extra = remaining.sample(n=n_take, random_state=SEED + 1)
                sampled_indices.extend(extra.index.tolist())

    # Final cap at TARGET_N
    if len(sampled_indices) > TARGET_N:
        log(f"trimming {len(sampled_indices) - TARGET_N:,} to hit target {TARGET_N:,}")
        # Random trim with fixed seed for reproducibility
        import random
        rng = random.Random(SEED + 2)
        rng.shuffle(sampled_indices)
        sampled_indices = sampled_indices[:TARGET_N]

    log(f"final sample size: {len(sampled_indices):,}")

    # Build sample DataFrame
    frame["sampled"] = frame.index.isin(set(sampled_indices))
    sample_df = frame.loc[sampled_indices].reset_index(drop=True)

    # ---- Artifact 1: tier_a_sample/EPA_sample.parquet ----
    # Subset to the columns the Layer C orchestrator expects, in expected order
    orch_cols = ["document_id", "docket_id", "comment", "text_source",
                 "is_attachment_only", "title"]
    sample_df[orch_cols].to_parquet(OUT_PARQUET, index=False)
    log(f"wrote sample parquet ({len(sample_df):,} rows) -> {OUT_PARQUET}")

    # ---- Artifact 2: tier_a_anchors.csv ----
    sample_dockets = sample_df["docket_id"].unique()
    anchors_df = pd.DataFrame({"docket_id": sorted(sample_dockets)})
    anchors_df.to_csv(OUT_ANCHORS, index=False)
    log(f"wrote anchors CSV ({len(sample_dockets):,} unique dockets) -> {OUT_ANCHORS}")

    # ---- Artifact 3: tier_a_strata_assignments.csv ----
    # Full audit trail: every Tier A frame comment + cell + sampled indicator
    audit_cols = ["document_id", "docket_id", "program_office", "year",
                  "year_bucket", "docket_size", "tercile", "cell", "sampled"]
    frame[audit_cols].to_csv(OUT_STRATA, index=False)
    log(f"wrote strata audit ({len(frame):,} rows) -> {OUT_STRATA}")

    # ---- Per-cell summary for sanity ----
    log("")
    log("=== per-cell sample size (post-allocation) ===")
    summary = (sample_df.groupby("cell").size()
               .reset_index(name="sampled_n")
               .merge(cell_sizes.reset_index(name="frame_n"), on="cell")
               .sort_values("sampled_n", ascending=False))
    log(f"total cells with at least one sample: {len(summary)}")
    log(f"summary head:\n{summary.head(10).to_string(index=False)}")
    log(f"summary tail:\n{summary.tail(5).to_string(index=False)}")

    # Sanity: program_office distribution
    log("")
    log("=== sampled program_office distribution ===")
    log(sample_df["program_office"].value_counts().to_string())

    log("")
    log("=== sampled year_bucket distribution ===")
    log(sample_df["year_bucket"].value_counts().sort_index().to_string())

    log("")
    log("=== sampled tercile distribution ===")
    log(sample_df["tercile"].value_counts().sort_index().to_string())

    log("")
    log("done. next: commit artifacts, then fire Layer C orchestrator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
