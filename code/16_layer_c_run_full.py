"""
16_layer_c_run_full.py — Layer C primary production orchestrator.

Codes the 70,075-comment Tier B analytic sample (PROJECT_FACTS §3 + §5)
with the 23 binary rhetoric features locked in
notes/2026-05-08_prompt_template_v1.md. Reuses `_call_openai_batch` from
path_a_obligation_llm.py — same retry / recovery / cost-cap semantics
as Path A Stage 2 and Stage 4.

PROJECT_FACTS §7 prices Layer C primary at ~$0.0044 per call batched
(3K input + 500 output @ gpt-5 batched rates). §8 budgets ~$155 batched
for 70K comments. The orchestrator defaults --max-cost to $200 to leave
~30% headroom for recovery-path retries.

Usage:
    # Smoke (1-anchor dry-run)
    python code/16_layer_c_run_full.py \\
        --docket EPA-HQ-OAR-2018-0775 --sample-size 5 --dry-run

    # Production run
    python code/16_layer_c_run_full.py --all --batch

The orchestrator mirrors code/12_path_a_extract_all_anchors.py's CLI +
summary block — same dispatch flag (`--batch`/`--no-batch`), same
mutually-exclusive scope flag (`--all` vs `--docket`), same
CostCapExceeded raise on overrun.
"""
from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from layer_c_judge import (  # noqa: E402
    LAYER_C_BINARIES,
    estimate_cost,
    estimate_cost_realtime,
    run_layer_c,
)
from path_a_obligation_llm import (  # noqa: E402
    CostCapExceeded,
    DEFAULT_OPENAI_MODEL,
)


DEFAULT_ANCHORS_CSV = Path("data/processed/anchor_rules_locked.csv")
DEFAULT_COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"
DEFAULT_OUTPUT = Path("data/processed/layer_c_full.csv")
DEFAULT_MAX_COST = 200.0
SAMPLE_SEED = 42


# ---------------------------------------------------------------------------
# Tier B loader
# ---------------------------------------------------------------------------
def _load_anchor_dockets(anchors_csv: Path) -> set[str]:
    with anchors_csv.open("r", encoding="utf-8") as f:
        return {row["docket_id"] for row in csv.DictReader(f)}


def _load_tier_b(
    anchors: set[str], comments_glob: str,
    docket_filter: Optional[str] = None,
):
    """Read all sharded parquets, filter to:
      - docket_id ∈ anchors (Tier B is the 36-anchor sub-corpus)
      - text_source != 'attachment_failed' (drop unrecoverable rows)
    Returns a single DataFrame with [document_id, docket_id, comment,
    text_source, is_attachment_only, title] columns."""
    import pandas as pd
    parts = []
    cols = ["document_id", "docket_id", "comment", "text_source",
            "is_attachment_only", "title"]
    for path in sorted(glob.glob(comments_glob)):
        df = pd.read_parquet(path, columns=cols)
        match = df[df["docket_id"].isin(anchors)
                   & (df["text_source"] != "attachment_failed")]
        if docket_filter is not None:
            match = match[match["docket_id"] == docket_filter]
        if not match.empty:
            parts.append(match)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Skip-existing — read already-coded comment_ids from a previous run
# ---------------------------------------------------------------------------
def _existing_comment_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    out: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "comment_id" not in reader.fieldnames:
            return set()
        for row in reader:
            cid = row.get("comment_id")
            if cid:
                out.add(str(cid))
    return out


# ---------------------------------------------------------------------------
# Output writer (append-safe so --skip-existing can resume)
# ---------------------------------------------------------------------------
def _write_layer_c_csv(df, output_path: Path, *, append: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and output_path.exists():
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(df, elapsed_s: float) -> None:
    if df is None or len(df) == 0:
        print("\n[layer-c] summary: zero rows coded.", file=sys.stderr)
        return
    n = len(df)
    cost = float(df["cost_usd"].sum())
    in_tok = int(df["in_tokens"].sum())
    out_tok = int(df["out_tokens"].sum())
    n_trunc = int(df["truncation_flag"].sum())
    print("\n[layer-c] summary", file=sys.stderr)
    print(f"  coded: {n:,}  cost=${cost:,.4f}  "
          f"in_tok={in_tok:,} out_tok={out_tok:,}  "
          f"truncated={n_trunc}  elapsed={elapsed_s:.1f}s",
          file=sys.stderr)
    # Per-binary positive rate.
    print("  per-binary positive rate (sorted, ≥10% bolded with *):",
          file=sys.stderr)
    rates = []
    for b in LAYER_C_BINARIES:
        if b in df.columns:
            rate = float((df[b] == 1).mean())
            rates.append((b, rate))
    rates.sort(key=lambda x: -x[1])
    for name, rate in rates:
        marker = "*" if rate >= 0.10 else " "
        print(f"   {marker} {name:<40} {rate * 100:>5.1f}%", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", dest="all_comments", action="store_true",
                       help="Run across the full Tier B 70K corpus.")
    scope.add_argument("--docket", type=str,
                       help="Run a single docket's comments (smoke tests).")
    bg = ap.add_mutually_exclusive_group()
    bg.add_argument("--batch", dest="batch", action="store_true", default=True,
                    help="Use OpenAI Batch API (default).")
    bg.add_argument("--no-batch", dest="batch", action="store_false",
                    help="Per-comment real-time path (smoke / regression).")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="Concurrency for --no-batch real-time mode. "
                         "Default 20 keeps us at ~80K TPM (well under "
                         "gpt-5 Tier-4 ceiling). Set to 1 for sequential. "
                         "Ignored in --batch mode.")
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST,
                    help=f"Hard cost cap (USD). Default ${DEFAULT_MAX_COST:.0f} "
                         "(PROJECT_FACTS §8 budgets ~$155 batched; 30% "
                         "headroom for recovery). Real-time at ~$0.0089/call "
                         "needs ~$700 for the full 70K corpus — set --max-cost "
                         "accordingly when using --no-batch.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip comment_ids already present in --output.")
    ap.add_argument("--sample-size", type=int, default=0,
                    help="Random-sample N comments after filtering "
                         f"(seed={SAMPLE_SEED}). 0 = no sampling.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print eligible count + estimated cost; no API call.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS_CSV)
    ap.add_argument("--comments-glob", type=str, default=DEFAULT_COMMENTS_GLOB)
    ap.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    anchors = _load_anchor_dockets(args.anchors_csv)
    print(f"[layer-c] loaded {len(anchors)} anchor dockets from "
          f"{args.anchors_csv}", file=sys.stderr)

    docket_filter = args.docket if not args.all_comments else None
    if docket_filter and docket_filter not in anchors:
        print(f"[layer-c] WARN: --docket {docket_filter} is not in "
              "anchor_rules_locked.csv; running anyway but the result "
              "is not part of Tier B.", file=sys.stderr)

    df = _load_tier_b(anchors, args.comments_glob, docket_filter)
    n_loaded = len(df)
    print(f"[layer-c] Tier B candidates loaded: {n_loaded:,}", file=sys.stderr)

    if args.skip_existing:
        already = _existing_comment_ids(args.output)
        if already:
            df = df[~df["document_id"].astype(str).isin(already)]
            print(f"[layer-c] --skip-existing dropped "
                  f"{n_loaded - len(df):,} comments already in "
                  f"{args.output}", file=sys.stderr)

    if args.sample_size and args.sample_size > 0:
        if len(df) > args.sample_size:
            df = df.sample(n=args.sample_size, random_state=SAMPLE_SEED) \
                   .reset_index(drop=True)
            print(f"[layer-c] random-sample applied: {len(df):,} of "
                  f"{n_loaded:,} (seed={SAMPLE_SEED})", file=sys.stderr)

    n_eligible = len(df)
    # Use the right cost anchor for the active dispatch mode. Real-time
    # is ~2× the batched rate (3K in × $1.25/M + 500 out × $10/M vs the
    # batched anchor of $0.0044), so the dry-run estimate has to track
    # the mode the user is actually about to fire.
    if args.batch:
        est_cost = estimate_cost(n_eligible)
        cost_anchor_str = "$0.0044/call, gpt-5 batched"
    else:
        est_cost = estimate_cost_realtime(n_eligible)
        cost_anchor_str = "$0.0089/call, gpt-5 standard (real-time)"
    if args.dry_run:
        print(f"\n[layer-c dry-run] eligible comments: {n_eligible:,}",
              file=sys.stderr)
        print(f"[layer-c dry-run] estimated cost: ${est_cost:,.2f} "
              f"(anchor {cost_anchor_str} per PROJECT_FACTS §7)",
              file=sys.stderr)
        # Surface mode + cap so the user can compare to the cap before
        # firing the real run.
        print(f"[layer-c dry-run] mode: batch={args.batch}  "
              f"concurrency={args.concurrency}  "
              f"max_cost=${args.max_cost:.2f}", file=sys.stderr)
        return 0

    if n_eligible == 0:
        print("[layer-c] no eligible comments; nothing to do.", file=sys.stderr)
        return 0

    df = df.rename(columns={"document_id": "comment_id"})
    t_start = time.time()
    try:
        coded = run_layer_c(
            df, model=args.model, batch=args.batch,
            max_cost_usd=args.max_cost,
            concurrency=args.concurrency,
        )
    except CostCapExceeded as e:
        print(f"\n[layer-c] ABORTED — {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    elapsed_s = time.time() - t_start
    n_written = _write_layer_c_csv(
        coded, args.output, append=args.skip_existing and args.output.exists(),
    )
    print(f"[layer-c] wrote {n_written:,} rows to {args.output}",
          file=sys.stderr)
    _print_summary(coded, elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
