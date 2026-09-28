"""
13_stage4_run_per_anchor.py — Stage 4 orchestrator (Path A
comment-to-obligation matching).

Per notes/2026-05-07_stage1_rubric_design.md §549–580. For each anchor:
  1. Load NPRM obligations (Stage 1b output for that docket / rule type).
  2. Load the anchor's analytic-corpus comments.
  3. Run the embedding prefilter (top-K=10, cosine >= 0.5).
  4. (Optional) call gpt-5 on each surviving pair via stage4_llm_match.
  5. Write per-anchor parquet to data/processed/stage4/.

Usage:
    # Single-anchor smoke (no LLM call)
    python code/13_stage4_run_per_anchor.py \\
        --anchor EPA-HQ-OAR-2018-0775 --dry-run

    # Single-anchor production run
    python code/13_stage4_run_per_anchor.py \\
        --anchor EPA-HQ-OAR-2018-0775 --max-cost 50

    # All anchors (explicit opt-in)
    python code/13_stage4_run_per_anchor.py --all --max-cost 1500

Default behavior requires either --anchor or --all so the user explicitly
opts into the full 36-anchor production scope.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make code/lib importable without requiring a package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from stage4_embedding_prefilter import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    filter_eligible_comments,
    obligation_id_of,
    prefilter_top_k,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/stage4")
DEFAULT_ANCHORS_CSV = Path("data/processed/anchor_rules_locked.csv")
DEFAULT_COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"
DEFAULT_OBLIGATIONS_DIR = Path("data/processed")

# Submit-only mode (Task I, 2026-05-14) writes one row per submitted
# batch to this CSV; the retrieval script reads it to poll batches in
# parallel without holding the embedding model in RAM. Schema:
#   docket_id, rule_type, batch_id, job_id, input_jsonl,
#   manifest_json, n_pairs, submitted_at_utc, model, status
DEFAULT_INFLIGHT_CSV = Path("data/processed/stage4_inflight_batches.csv")
INFLIGHT_CSV_FIELDS = (
    "docket_id", "rule_type", "batch_id", "job_id", "input_jsonl",
    "manifest_json", "n_pairs", "submitted_at_utc", "model", "status",
)


@dataclass
class AnchorResult:
    docket_id: str
    status: str   # 'ok' | 'skipped' | 'failed' | 'dry-run' | 'submitted'
    reason: str = ""
    n_comments: int = 0
    n_obligations: int = 0
    n_pairs: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    out_path: Optional[Path] = None
    # Submit-only mode (Task I): populated when status='submitted'.
    batch_id: Optional[str] = None
    job_id: Optional[str] = None


def _append_inflight_row(inflight_csv: Path, row: dict) -> None:
    """Append a row to the in-flight tracking CSV (create + write header
    if it doesn't yet exist). Concurrent appends from sequential CLI
    invocations are safe because each writes exactly one row and the
    user runs invocations serially per the task spec."""
    inflight_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not inflight_csv.exists()
    with inflight_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INFLIGHT_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INFLIGHT_CSV_FIELDS})


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _load_anchor_ids(anchors_csv: Path) -> list[str]:
    with anchors_csv.open("r", encoding="utf-8") as f:
        return [row["docket_id"] for row in csv.DictReader(f)]


def _obligation_csv_path(docket_id: str, rule_type: str) -> Path:
    return (DEFAULT_OBLIGATIONS_DIR
            / f"path_a_obligations_verified_{docket_id}_{rule_type}.csv")


def _load_obligations(docket_id: str, rule_type: str) -> Optional[list[dict]]:
    path = _obligation_csv_path(docket_id, rule_type)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["obligation_id"] = obligation_id_of(r)
    return rows


def _load_comments_for_docket(docket_id: str, comments_glob: str):
    """Return a pandas DataFrame of comments scoped to docket_id. We scan
    each year's parquet because the analytic corpus is sharded by year.
    Eligibility filtering (attachment-failed / too-short) is applied
    separately by the caller via `filter_eligible_comments` so the drop
    counts can be logged."""
    import pandas as pd
    parts: list[pd.DataFrame] = []
    for path in sorted(glob.glob(comments_glob)):
        df = pd.read_parquet(path, columns=[
            "document_id", "docket_id", "comment", "text_source",
            "is_attachment_only", "title",
        ])
        match = df[df["docket_id"] == docket_id]
        if not match.empty:
            parts.append(match)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    # Stage 4 requires non-empty comment text. Length-based eligibility
    # (≥50 chars after strip) is handled by filter_eligible_comments.
    out = out[out["comment"].fillna("").str.len() > 0].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Per-anchor pipeline
# ---------------------------------------------------------------------------
def run_anchor(
    docket_id: str,
    *,
    rule_type: str,
    output_dir: Path,
    comments_glob: str,
    k: int,
    similarity_threshold: float,
    dry_run: bool,
    max_cost_usd: float,
    model: str,
    batch: bool,
    cumulative_cost_before: float = 0.0,
    submit_only: bool = False,
    inflight_csv: Path = DEFAULT_INFLIGHT_CSV,
    head_chars: Optional[int] = None,
    tail_chars: Optional[int] = None,
) -> AnchorResult:
    res = AnchorResult(docket_id=docket_id, status="ok")
    print(f"[stage4] anchor={docket_id} rule_type={rule_type}", file=sys.stderr)

    obligations = _load_obligations(docket_id, rule_type)
    if obligations is None:
        res.status = "skipped"
        res.reason = (f"no obligations CSV at "
                      f"{_obligation_csv_path(docket_id, rule_type)} "
                      "(Stage 1b has not yet produced this anchor / rule_type)")
        print(f"  [stage4] SKIP — {res.reason}", file=sys.stderr)
        return res
    res.n_obligations = len(obligations)

    comments_df = _load_comments_for_docket(docket_id, comments_glob)
    if comments_df is None or comments_df.empty:
        res.status = "skipped"
        res.reason = (f"no comments matched docket_id={docket_id} in "
                      f"{comments_glob}")
        print(f"  [stage4] SKIP — {res.reason}", file=sys.stderr)
        return res
    n_loaded = len(comments_df)

    # Eligibility filter (attachment-failed + too-short). Runs BEFORE
    # the embedding step so dropped rows don't burn embedding compute.
    comments_df, drop_counts = filter_eligible_comments(comments_df)
    print(f"  [stage4] eligibility filter: loaded {n_loaded:,} → kept "
          f"{drop_counts['n_kept']:,}  "
          f"(attachment_failed={drop_counts['attachment_failed']}, "
          f"too_short={drop_counts['too_short']}, "
          f"total_dropped={drop_counts['total_dropped']})",
          file=sys.stderr)
    if comments_df.empty:
        res.status = "skipped"
        res.reason = ("no eligible comments after applying the Stage 4 "
                      "filter (all dropped as attachment-failed or <50 chars)")
        print(f"  [stage4] SKIP — {res.reason}", file=sys.stderr)
        return res
    res.n_comments = len(comments_df)
    print(f"  [stage4] loaded {res.n_comments:,} comments × "
          f"{res.n_obligations} obligations", file=sys.stderr)

    # Prefilter
    t0 = time.time()
    pairs = prefilter_top_k(
        comments_df.to_dict("records"),
        obligations,
        k=k,
        similarity_threshold=similarity_threshold,
        docket_id=docket_id,
    )
    dt = time.time() - t0
    res.n_pairs = len(pairs)
    from stage4_llm_match import estimate_cost  # late import
    res.estimated_cost_usd = estimate_cost(res.n_pairs)
    print(f"  [stage4] prefilter: {res.n_pairs:,} pairs "
          f"({res.n_pairs / max(1, res.n_comments):.2f}/comment); "
          f"prefilter elapsed={dt:.1f}s; "
          f"estimated LLM cost=${res.estimated_cost_usd:.2f}", file=sys.stderr)

    if dry_run:
        res.status = "dry-run"
        return res

    # ---- Submit-only mode (Task I) ----
    # Stage the batch, write tracking row, exit. The retrieval script
    # (14_stage4_retrieve_batches.py) polls + folds results later. This
    # frees the embedding model from RAM after ~60 s of work per anchor
    # so 26 sequential submits stay under the 16 GB workstation ceiling.
    #
    # Sharding (Task J, 2026-05-14): anchors whose prefilter pair count
    # exceeds 50K (the OpenAI Batch API hard cap, e.g. EPA-HQ-OAR-
    # 2017-0355 at 157,883 pairs) get split into N shards by
    # `submit_batches_sharded` — N tracking rows for one anchor. The
    # retrieve script groups by (docket_id, rule_type) and concatenates
    # the per-shard DataFrames before writing the per-anchor parquet.
    if submit_only:
        from stage4_llm_match import (
            CostCapExceeded, prepare_batch_submission, submit_batches_sharded,
        )
        cdf = comments_df.rename(columns={"document_id": "comment_id"})
        odf = _obligations_df(obligations)
        submission = prepare_batch_submission(
            pairs, cdf, odf, model=model,
            head_chars=head_chars, tail_chars=tail_chars,
        )
        if submission["n_pairs"] == 0:
            res.status = "skipped"
            res.reason = "prefilter produced 0 submittable pairs"
            print(f"  [stage4] SKIP — {res.reason}", file=sys.stderr)
            return res
        try:
            manifests = submit_batches_sharded(
                submission,
                model=model,
                batch_id_prefix=f"stage4_{docket_id}_{rule_type}",
                max_cost_usd=max_cost_usd - cumulative_cost_before,
            )
        except CostCapExceeded as e:
            res.status = "failed"
            res.reason = f"cost cap exceeded: {e}"
            print(f"  [stage4] FAIL — {res.reason}", file=sys.stderr)
            return res
        except Exception as e:
            res.status = "failed"
            res.reason = f"{type(e).__name__}: {e}"
            print(f"  [stage4] FAIL — {res.reason}", file=sys.stderr)
            return res
        for mf in manifests:
            _append_inflight_row(inflight_csv, {
                "docket_id": docket_id, "rule_type": rule_type,
                "batch_id": mf["batch_id"], "job_id": mf["job_id"],
                "input_jsonl": mf["input_jsonl"],
                "manifest_json": mf["manifest_json"],
                "n_pairs": mf["n_pairs"],
                "submitted_at_utc": mf["submitted_at_utc"],
                "model": mf["model"],
                "status": "submitted",
            })
        res.status = "submitted"
        # When sharded, report the first batch_id + shard count; the
        # full per-shard inventory is in the tracking CSV.
        res.batch_id = manifests[0]["batch_id"]
        res.job_id = manifests[0]["job_id"]
        res.estimated_cost_usd = float(res.n_pairs) * 0.0019
        if len(manifests) == 1:
            print(f"  [stage4] submitted batch_id={manifests[0]['batch_id']} "
                  f"(job_id={manifests[0]['job_id']}) — tracking row "
                  f"appended to {inflight_csv}", file=sys.stderr)
        else:
            print(f"  [stage4] submitted {len(manifests)} shards "
                  f"(job_id={manifests[0]['job_id']}, "
                  f"first batch_id={manifests[0]['batch_id']}) — "
                  f"{len(manifests)} tracking rows appended to {inflight_csv}",
                  file=sys.stderr)
        return res

    # ---- Backward-compat submit-then-poll mode (existing) ----
    from stage4_llm_match import match_pairs, CostCapExceeded
    remaining_budget = max_cost_usd - cumulative_cost_before
    if remaining_budget <= 0:
        res.status = "skipped"
        res.reason = (f"cumulative cost cap reached before this anchor "
                      f"(cap=${max_cost_usd:.2f})")
        print(f"  [stage4] SKIP — {res.reason}", file=sys.stderr)
        return res

    # Comments DataFrame for match_pairs needs the `comment_id` join key.
    cdf = comments_df.rename(columns={"document_id": "comment_id"})
    odf = _obligations_df(obligations)

    try:
        match_df = match_pairs(
            pairs, cdf, odf,
            model=model, batch=batch, max_cost_usd=remaining_budget,
        )
    except CostCapExceeded as e:
        res.status = "failed"
        res.reason = f"cost cap exceeded: {e}"
        print(f"  [stage4] FAIL — {res.reason}", file=sys.stderr)
        return res
    except Exception as e:
        res.status = "failed"
        res.reason = f"{type(e).__name__}: {e}"
        print(f"  [stage4] FAIL — {res.reason}", file=sys.stderr)
        return res

    res.actual_cost_usd = float(match_df["cost_usd"].sum()) if not match_df.empty else 0.0

    out_path = output_dir / f"{docket_id}__matches.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_parquet(out_path, index=False)
    res.out_path = out_path
    print(f"  [stage4] wrote {len(match_df)} matched rows to {out_path}; "
          f"actual cost=${res.actual_cost_usd:.4f}", file=sys.stderr)
    return res


def _obligations_df(obligations: list[dict]):
    import pandas as pd
    df = pd.DataFrame(obligations)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--anchor", type=str,
                   help="Run a single anchor (docket_id).")
    g.add_argument("--all", action="store_true",
                   help="Run all 36 anchors. Requires explicit opt-in.")
    ap.add_argument("--rule-type", choices=("proposed", "final"),
                    default="proposed",
                    help="Stage 1b output to load. Spec default is 'proposed' "
                         "(NPRM); 'final' is offered for smoke tests on the "
                         "2018-0775 final-rule output.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run prefilter only; print estimated cost and exit.")
    ap.add_argument("--max-cost", type=float, default=1500.0,
                    help="Hard cumulative cost cap (USD) across anchors.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--similarity-threshold", type=float,
                    default=DEFAULT_SIMILARITY_THRESHOLD)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS_CSV)
    ap.add_argument("--comments-glob", type=str, default=DEFAULT_COMMENTS_GLOB)
    ap.add_argument("--model", type=str, default="gpt-5")
    ap.add_argument("--no-batch", action="store_true",
                    help="Disable batched-API path. (Scaffold note: even with "
                         "batch enabled the OpenAI Batch API wrapper is not "
                         "yet implemented; calls run sequentially.)")
    ap.add_argument("--submit-only", action="store_true",
                    help="Stage the batch (prefilter + JSONL + Batch API "
                         "submit) and exit without polling. Writes a tracking "
                         "row to data/processed/stage4_inflight_batches.csv "
                         "and a manifest sidecar next to the input JSONL. "
                         "Use 14_stage4_retrieve_batches.py to poll + fold "
                         "results later. Holds RAM only for ~60 s per "
                         "invocation (prefilter + submit), freeing the "
                         "embedding model so sequential anchors can run "
                         "on a 16 GB workstation.")
    ap.add_argument("--inflight-csv", type=Path, default=DEFAULT_INFLIGHT_CSV,
                    help="Tracking CSV for --submit-only mode.")
    ap.add_argument("--truncation-cap-head", type=int, default=None,
                    help="Override the comment-text head-char cap in the "
                         "prompt builder. Defaults to TRUNCATION_HEAD_CHARS "
                         "(1200). Task K rerun pipeline uses 4000.")
    ap.add_argument("--truncation-cap-tail", type=int, default=None,
                    help="Override the comment-text tail-char cap. Defaults "
                         "to TRUNCATION_TAIL_CHARS (200). Task K rerun "
                         "pipeline uses 1000.")
    args = ap.parse_args()

    if args.all:
        anchor_ids = _load_anchor_ids(args.anchors_csv)
        print(f"[stage4] --all: {len(anchor_ids)} anchors from "
              f"{args.anchors_csv}", file=sys.stderr)
    else:
        anchor_ids = [args.anchor]

    results: list[AnchorResult] = []
    cumulative_cost = 0.0
    for docket_id in anchor_ids:
        res = run_anchor(
            docket_id,
            rule_type=args.rule_type,
            output_dir=args.output_dir,
            comments_glob=args.comments_glob,
            k=args.top_k,
            similarity_threshold=args.similarity_threshold,
            dry_run=args.dry_run,
            max_cost_usd=args.max_cost,
            model=args.model,
            batch=not args.no_batch,
            cumulative_cost_before=cumulative_cost,
            submit_only=args.submit_only,
            inflight_csv=args.inflight_csv,
            head_chars=args.truncation_cap_head,
            tail_chars=args.truncation_cap_tail,
        )
        results.append(res)
        cumulative_cost += res.actual_cost_usd

    _print_summary(results, args.dry_run, cumulative_cost, args.max_cost)
    return 0


def _print_summary(
    results: list[AnchorResult],
    dry_run: bool,
    cumulative_cost: float,
    max_cost: float,
) -> None:
    n_ok = sum(1 for r in results
               if r.status in ("ok", "dry-run", "submitted"))
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_fail = sum(1 for r in results if r.status == "failed")
    total_pairs = sum(r.n_pairs for r in results)
    total_est = sum(r.estimated_cost_usd for r in results)

    print("\n[stage4] summary", file=sys.stderr)
    print(f"  anchors processed: {len(results)} "
          f"(ok/dry={n_ok}, skipped={n_skip}, failed={n_fail})",
          file=sys.stderr)
    print(f"  prefilter pairs total: {total_pairs:,}", file=sys.stderr)
    print(f"  estimated LLM cost: ${total_est:,.2f} "
          f"(spec anchor: $0.0019 / pair, gpt-5 batched)", file=sys.stderr)
    if not dry_run:
        print(f"  actual LLM cost: ${cumulative_cost:,.4f} of cap "
              f"${max_cost:,.2f}", file=sys.stderr)
    print("\n  per-anchor:", file=sys.stderr)
    for r in results:
        line = (f"    {r.docket_id:<28} {r.status:<10} "
                f"comments={r.n_comments:>7,} "
                f"obligations={r.n_obligations:>4,} "
                f"pairs={r.n_pairs:>7,} "
                f"est=${r.estimated_cost_usd:>7.2f}")
        if r.status == "ok":
            line += f"  actual=${r.actual_cost_usd:.4f}  out={r.out_path}"
        elif r.status == "submitted":
            line += f"  batch_id={r.batch_id}"
        elif r.status in ("skipped", "failed"):
            line += f"  reason={r.reason}"
        print(line, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
