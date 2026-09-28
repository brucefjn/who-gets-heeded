"""
14_stage4_aggregate.py — obligation-level aggregation for Path A Stage 4.

Joins per-anchor Stage 4 match parquets (output of 13_stage4_run_per_anchor)
into a single obligation-level table that the downstream responsiveness
regression consumes. Per Stage 4 sub-spec §549–580 + Path A §5.3 of the
paper draft.

Computed per obligation (across the comments that addressed it):
  n_comments_addressed       — count of distinct comments with addressed=True
  n_supporting               — addressed=True AND stance=SUPPORTING
  n_opposing                 — addressed=True AND stance=OPPOSING
  n_modifying                — addressed=True AND stance=SUGGESTING_MODIFICATION
  pct_organizational         — share of addressing comments from organizational
                               submitters (Title-field reconstructed)
  pct_individual             — share of addressing comments from individual
                               submitters
  mean_cosine                — mean prefilter cosine among addressing comments
  addressing_comment_ids     — semicolon-joined list (auditability)

Left-joined to the Path A outcome state (seven-state taxonomy) per
obligation. Obligations with no matched comments are retained with null
comment-side fields.

Output schema (parquet, one row per obligation):
  [obligation_id, docket_id, rule_type, candidate_idx, split_idx,
   cfr_section, obligation_text,
   outcome_state,                       # null if outcomes file missing
   n_comments_addressed, n_supporting, n_opposing, n_modifying,
   pct_organizational, pct_individual, mean_cosine,
   addressing_comment_ids]

Usage:
    python code/14_stage4_aggregate.py \\
        --input-dir data/processed/stage4/ \\
        --path-a-outcomes data/processed/path_a_obligation_outcomes.parquet \\
        --output data/processed/stage4_obligation_level.parquet
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/processed/stage4/")
DEFAULT_OUTPUT = Path("data/processed/stage4_obligation_level.parquet")
DEFAULT_COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"


# Make code/lib importable without requiring a package install — matches
# the convention used in 13_stage4_run_per_anchor.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))
from submitter_classification import classify_title  # noqa: E402


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_all_matches(input_dir: Path):
    import pandas as pd
    paths = sorted(glob.glob(str(input_dir / "*__matches.parquet")))
    if not paths:
        return None
    parts = [pd.read_parquet(p) for p in paths]
    out = pd.concat(parts, ignore_index=True)
    print(f"[aggregate] loaded {len(out):,} match rows from {len(paths)} "
          "per-anchor parquets", file=sys.stderr)
    return out


def _load_comments_index(comments_glob: str, comment_ids: set[str]):
    """Return DataFrame with [comment_id, docket_id, title, submitter_kind]
    for every comment_id that appears in the match set."""
    import pandas as pd
    parts = []
    for path in sorted(glob.glob(comments_glob)):
        df = pd.read_parquet(path, columns=["document_id", "docket_id", "title"])
        match = df[df["document_id"].isin(comment_ids)]
        if not match.empty:
            parts.append(match)
    if not parts:
        return pd.DataFrame(columns=["comment_id", "docket_id", "title",
                                     "submitter_kind"])
    out = pd.concat(parts, ignore_index=True).rename(
        columns={"document_id": "comment_id"})
    out["submitter_kind"] = out["title"].map(classify_title)
    return out


def _load_obligations_universe(matches_df) -> "pandas.DataFrame":
    """Reload Stage 1b verified CSVs for every (docket_id, rule_type) seen
    in the match set so obligations with zero addressing comments still
    appear in the output."""
    import pandas as pd
    pairs = (matches_df[["obligation_id"]]
             .drop_duplicates()
             ["obligation_id"].tolist())
    docket_rule = set()
    for oid in pairs:
        try:
            docket, rule, _ = oid.split("__", 2)
            docket_rule.add((docket, rule))
        except ValueError:
            continue
    universe_rows: list[dict] = []
    for docket, rule in sorted(docket_rule):
        path = (Path("data/processed")
                / f"path_a_obligations_verified_{docket}_{rule}.csv")
        if not path.exists():
            print(f"  [aggregate] WARN: obligations CSV missing for "
                  f"{docket}/{rule}; obligations with zero matches in this "
                  "docket will not appear.", file=sys.stderr)
            continue
        d = pd.read_csv(path)
        d["docket_id"] = docket
        d["rule_type"] = rule
        for _, row in d.iterrows():
            cidx = row.get("candidate_idx")
            sidx = row.get("split_idx", 0)
            if cidx is None or (isinstance(cidx, float) and pd.isna(cidx)):
                continue
            oid = f"{docket}__{rule}__{int(cidx)}_{int(sidx)}"
            universe_rows.append({
                "obligation_id": oid,
                "docket_id": docket,
                "rule_type": rule,
                "candidate_idx": int(cidx),
                "split_idx": int(sidx),
                "cfr_section": row.get("cfr_section"),
                "obligation_text": row.get("obligation_text"),
            })
    return pd.DataFrame(universe_rows).drop_duplicates(subset="obligation_id")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(
    matches_df,
    comments_idx,
    obligations_universe,
    outcomes_df=None,
):
    """Compute obligation-level features. Returns a DataFrame keyed by
    obligation_id with the schema described in the module docstring."""
    import pandas as pd

    # Join submitter_kind onto matches via comment_id
    if not comments_idx.empty:
        m = matches_df.merge(
            comments_idx[["comment_id", "submitter_kind"]],
            on="comment_id", how="left",
        )
    else:
        m = matches_df.copy()
        m["submitter_kind"] = "unknown"

    # Restrict to addressed=True rows for the count-based aggregates;
    # `mean_cosine` is over addressing comments only (consistent with the
    # definition of "addressed an obligation").
    addressed = m[m["addressed"] == True]   # noqa: E712 — explicit for clarity

    def _agg(group):
        n = len(group)
        kinds = group["submitter_kind"].fillna("unknown")
        n_org = int((kinds == "organizational").sum())
        n_ind = int((kinds == "individual").sum())
        # Denominator excludes unknowns so the percentages mean "of the
        # comments whose kind we could identify". Document this in paper.
        n_known = n_org + n_ind
        return pd.Series({
            "n_comments_addressed": int((group["addressed"] == True).sum()),
            "n_supporting": int((group["stance"] == "SUPPORTING").sum()),
            "n_opposing": int((group["stance"] == "OPPOSING").sum()),
            "n_modifying": int((group["stance"] == "SUGGESTING_MODIFICATION").sum()),
            "pct_organizational": (n_org / n_known) if n_known else float("nan"),
            "pct_individual": (n_ind / n_known) if n_known else float("nan"),
            "mean_cosine": float(group["cosine_similarity"].mean()),
            "addressing_comment_ids": ";".join(sorted(set(group["comment_id"].astype(str)))),
        })

    if addressed.empty:
        feats = pd.DataFrame(columns=[
            "obligation_id", "n_comments_addressed", "n_supporting",
            "n_opposing", "n_modifying", "pct_organizational",
            "pct_individual", "mean_cosine", "addressing_comment_ids",
        ])
    else:
        feats = (addressed.groupby("obligation_id", as_index=False)
                 .apply(_agg, include_groups=False).reset_index(drop=True))

    out = obligations_universe.merge(feats, on="obligation_id", how="left")

    # Fill zero-count fields for obligations with no addressing comments.
    for col in ("n_comments_addressed", "n_supporting", "n_opposing",
                "n_modifying"):
        out[col] = out[col].fillna(0).astype("int64")
    out["addressing_comment_ids"] = out["addressing_comment_ids"].fillna("")

    if outcomes_df is not None and not outcomes_df.empty:
        out = out.merge(outcomes_df, on="obligation_id", how="left")
    else:
        out["outcome_state"] = None

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--path-a-outcomes", type=Path, default=None,
                    help="Optional parquet with [obligation_id, outcome_state] "
                         "for the seven-state taxonomy. If missing, the join "
                         "is skipped and outcome_state is null.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--comments-glob", type=str, default=DEFAULT_COMMENTS_GLOB)
    args = ap.parse_args()

    matches = _load_all_matches(args.input_dir)
    if matches is None:
        print(f"[aggregate] no match parquets found in {args.input_dir}; "
              "nothing to aggregate.", file=sys.stderr)
        return 1

    comment_ids = set(matches["comment_id"].astype(str).unique())
    comments_idx = _load_comments_index(args.comments_glob, comment_ids)
    obligations_universe = _load_obligations_universe(matches)

    outcomes_df = None
    if args.path_a_outcomes is not None:
        if args.path_a_outcomes.exists():
            import pandas as pd
            outcomes_df = pd.read_parquet(args.path_a_outcomes)
            print(f"[aggregate] joined outcomes from {args.path_a_outcomes} "
                  f"({len(outcomes_df):,} rows)", file=sys.stderr)
        else:
            print(f"[aggregate] WARN: --path-a-outcomes={args.path_a_outcomes} "
                  "not found; output will have null outcome_state. This file "
                  "is a downstream Path A §5.3 dependency that has not yet "
                  "been produced.", file=sys.stderr)

    out = aggregate(matches, comments_idx, obligations_universe, outcomes_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[aggregate] wrote {len(out):,} obligation rows to {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
