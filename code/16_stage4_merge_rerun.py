"""
16_stage4_merge_rerun.py — Merge Task K rerun results into per-anchor parquets.

Operational sequence after rerun batches have been polled and folded
to per-anchor parquets in `data/processed/stage4_rerun/` via
`14_stage4_retrieve_batches.py --inflight-csv .../stage4_rerun_inflight.csv
--output-dir data/processed/stage4_rerun/`:

For each rerun parquet at `data/processed/stage4_rerun/<docket>__matches.parquet`:
  1. Load the corresponding original at `data/processed/stage4/<docket>__matches.parquet`.
  2. For each (comment_id, obligation_id) pair in the rerun parquet,
     REPLACE the matching row in the original. Untouched rows stay.
  3. Add (or update) `truncation_cap_used` column:
       - 1400 for original-cap rows that weren't replaced
       - 5000 for rerun rows
  4. Write the upserted DataFrame back to `data/processed/stage4/<docket>__matches.parquet`.

Idempotency: a second invocation finds rerun rows already merged
(their truncation_cap_used == 5000 already) and produces a byte-
identical output.

Usage:
    python3 code/16_stage4_merge_rerun.py
    python3 code/16_stage4_merge_rerun.py --dry-run   # show counts only
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from stage4_llm_match import (  # noqa: E402
    RERUN_TRUNCATION_HEAD_CHARS,
    RERUN_TRUNCATION_TAIL_CHARS,
    TRUNCATION_HEAD_CHARS,
    TRUNCATION_TAIL_CHARS,
)


DEFAULT_STAGE4_DIR = Path("data/processed/stage4")
DEFAULT_RERUN_DIR = Path("data/processed/stage4_rerun")

# Cap accounting (head + tail). 1400 = v2.3 default, 5000 = Task K rerun.
ORIGINAL_TRUNCATION_CAP = TRUNCATION_HEAD_CHARS + TRUNCATION_TAIL_CHARS
RERUN_TRUNCATION_CAP = (RERUN_TRUNCATION_HEAD_CHARS
                        + RERUN_TRUNCATION_TAIL_CHARS)
JOIN_KEYS = ("comment_id", "obligation_id")


def _merge_one_parquet(
    original_path: Path,
    rerun_path: Path,
    *,
    rerun_cap: int = RERUN_TRUNCATION_CAP,
    original_cap: int = ORIGINAL_TRUNCATION_CAP,
    dry_run: bool = False,
) -> dict:
    """Upsert one rerun parquet's rows into the matching original.
    Returns a summary dict with row counts."""
    import pandas as pd

    summary = {
        "docket_id": original_path.stem.replace("__matches", ""),
        "rerun_path": str(rerun_path),
        "original_path": str(original_path),
        "n_rerun_rows": 0,
        "n_original_rows": 0,
        "n_replaced": 0,
        "n_kept_original": 0,
        "n_new_appended": 0,
        "status": "skipped",
    }

    if not rerun_path.exists():
        summary["reason"] = "no rerun parquet for this docket"
        return summary
    rerun_df = pd.read_parquet(rerun_path)
    summary["n_rerun_rows"] = len(rerun_df)
    if rerun_df.empty:
        summary["reason"] = "rerun parquet is empty"
        return summary

    if not original_path.exists():
        # No original to merge into — write the rerun as the new
        # canonical parquet with truncation_cap_used populated.
        new_df = rerun_df.copy()
        new_df["truncation_cap_used"] = rerun_cap
        summary["n_new_appended"] = len(new_df)
        summary["status"] = "no_original_wrote_rerun"
        if not dry_run:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_parquet(original_path, index=False)
        return summary

    original_df = pd.read_parquet(original_path)
    summary["n_original_rows"] = len(original_df)

    # Ensure the join keys are stringly-typed on both sides so the
    # set-comparison + indexing path is predictable.
    for k in JOIN_KEYS:
        if k in original_df.columns:
            original_df[k] = original_df[k].astype(str)
        if k in rerun_df.columns:
            rerun_df[k] = rerun_df[k].astype(str)

    # Identify rows to replace via the (comment_id, obligation_id) key.
    rerun_keys = set(
        zip(rerun_df["comment_id"].tolist(),
            rerun_df["obligation_id"].tolist())
    )
    original_keys = list(zip(
        original_df["comment_id"].tolist(),
        original_df["obligation_id"].tolist(),
    ))
    keep_mask = [k not in rerun_keys for k in original_keys]

    kept = original_df[keep_mask].copy()
    summary["n_kept_original"] = len(kept)
    summary["n_replaced"] = len(original_df) - len(kept)
    # Rerun rows that don't correspond to any original row — usually 0,
    # but report it so any leakage is visible.
    original_key_set = set(original_keys)
    rerun_keys_only_in_rerun = [
        k for k in zip(rerun_df["comment_id"].tolist(),
                       rerun_df["obligation_id"].tolist())
        if k not in original_key_set
    ]
    summary["n_new_appended"] = len(rerun_keys_only_in_rerun)

    # Add / refresh truncation_cap_used. Pre-existing column values are
    # preserved on rows we keep; new-rerun rows get RERUN_TRUNCATION_CAP.
    if "truncation_cap_used" not in kept.columns:
        kept["truncation_cap_used"] = original_cap
    rerun_with_cap = rerun_df.copy()
    rerun_with_cap["truncation_cap_used"] = rerun_cap

    # Align columns — pandas concat raises if the column sets differ;
    # fill missing columns with sensible defaults.
    all_cols = list(dict.fromkeys(
        list(kept.columns) + list(rerun_with_cap.columns)
    ))
    for col in all_cols:
        if col not in kept.columns:
            kept[col] = None
        if col not in rerun_with_cap.columns:
            rerun_with_cap[col] = None
    kept = kept[all_cols]
    rerun_with_cap = rerun_with_cap[all_cols]

    merged = pd.concat([kept, rerun_with_cap], ignore_index=True)

    # Idempotency anchor: if every rerun (comment_id, obligation_id)
    # was ALREADY upserted at rerun_cap previously, the merge is a no-op
    # because rerun_with_cap has the same key set and `truncation_cap_used`
    # already equals rerun_cap. Output is byte-identical apart from row
    # ordering; we keep concat order stable.

    summary["status"] = "merged"
    if not dry_run:
        merged.to_parquet(original_path, index=False)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4_DIR)
    ap.add_argument("--rerun-dir", type=Path, default=DEFAULT_RERUN_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be merged; don't write parquets.")
    args = ap.parse_args(argv)

    if not args.rerun_dir.exists():
        print(f"[merge-rerun] {args.rerun_dir} not found; nothing to merge.",
              file=sys.stderr)
        return 0

    rerun_parquets = sorted(
        Path(p) for p in glob.glob(str(args.rerun_dir / "*__matches.parquet"))
    )
    if not rerun_parquets:
        print(f"[merge-rerun] no *__matches.parquet under {args.rerun_dir}",
              file=sys.stderr)
        return 0

    print(f"[merge-rerun] scanning {len(rerun_parquets)} rerun parquets",
          file=sys.stderr)
    summaries: list[dict] = []
    total_replaced = 0
    total_kept = 0
    for rerun_path in rerun_parquets:
        docket = rerun_path.stem.replace("__matches", "")
        original_path = args.stage4_dir / f"{docket}__matches.parquet"
        s = _merge_one_parquet(
            original_path, rerun_path, dry_run=args.dry_run,
        )
        summaries.append(s)
        total_replaced += s["n_replaced"]
        total_kept += s["n_kept_original"]
        print(f"  {docket}: original={s['n_original_rows']:,}  "
              f"rerun={s['n_rerun_rows']:,}  replaced={s['n_replaced']:,}  "
              f"kept={s['n_kept_original']:,}  "
              f"new={s['n_new_appended']}  status={s['status']}",
              file=sys.stderr)

    print(f"\n[merge-rerun] totals: replaced={total_replaced:,}  "
          f"kept={total_kept:,}  parquets={len(summaries)}", file=sys.stderr)
    if args.dry_run:
        print("[merge-rerun] dry-run: no parquets written.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
