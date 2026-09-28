"""
15_stage4_retrun_truncated.py — Stage 4 truncation rerun submitter (Task K).

Scans the per-anchor parquets at `data/processed/stage4/*.parquet`,
identifies rows where `uncertainty_truncated == True`, reconstructs the
input messages with a higher truncation cap (default head=4000,
tail=1000 vs v2.3's 1200+200), and submits the surgical rerun via the
existing `submit_batches_sharded` pipeline.

Why: the v2.3 cap (1,400 chars total) truncated 74,810 of 398,509
pairs (18.77%) — concentrated in long OPPOSING comments (37%
truncation rate) and the top input-token quartile (55%). Surgical
rerun at the higher cap preserves the 323,699 untruncated rows
unchanged and replaces only the 74,810 truncated rows after Task K's
merge step.

Output: each (docket_id, rule_type) group with ≥1 truncated row is
re-submitted via `submit_batches_sharded` (so anchors > 50K truncated
pairs auto-shard, e.g. EPA-HQ-OAR-2017-0355). Tracking rows append to
`data/processed/stage4_rerun_inflight.csv` (deliberately SEPARATE from
the main inflight CSV so the existing retrieve script can be pointed
at the rerun batches without touching the original Stage 4 outputs).

Operational sequence:
    python3 code/15_stage4_retrun_truncated.py
    python3 code/14_stage4_retrieve_batches.py \\
        --inflight-csv data/processed/stage4_rerun_inflight.csv \\
        --output-dir data/processed/stage4_rerun/
    python3 code/16_stage4_merge_rerun.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from stage4_embedding_prefilter import obligation_id_of  # noqa: E402
from stage4_llm_match import (  # noqa: E402
    RERUN_TRUNCATION_HEAD_CHARS,
    RERUN_TRUNCATION_TAIL_CHARS,
    CostCapExceeded,
    prepare_batch_submission,
    submit_batches_sharded,
)


DEFAULT_STAGE4_DIR = Path("data/processed/stage4")
DEFAULT_RERUN_INFLIGHT_CSV = Path("data/processed/stage4_rerun_inflight.csv")
DEFAULT_COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"
DEFAULT_OBLIGATIONS_DIR = Path("data/processed")
INFLIGHT_CSV_FIELDS = (
    "docket_id", "rule_type", "batch_id", "job_id", "input_jsonl",
    "manifest_json", "n_pairs", "submitted_at_utc", "model", "status",
)


# ---------------------------------------------------------------------------
# Truncated-row identification
# ---------------------------------------------------------------------------
def _parse_obligation_id(oid: str) -> tuple[str, str]:
    """Reverse `stage4_embedding_prefilter.obligation_id_of`:
    'docket__rule_type__candidate_idx_split_idx' → (docket, rule_type)."""
    parts = oid.split("__", 2)
    if len(parts) < 3:
        raise ValueError(f"obligation_id has unexpected shape: {oid!r}")
    docket = parts[0]
    rule = parts[1]
    return docket, rule


def _find_truncated_pairs(stage4_dir: Path):
    """Walk every parquet in stage4_dir, return:
        truncated_by_group: dict[(docket, rule_type)] → list of dicts
            with keys {comment_id, obligation_id, cosine_similarity}
        scan_summary: dict with total row + truncated-row counts."""
    import pandas as pd
    truncated_by_group: dict[tuple[str, str], list[dict]] = {}
    total_rows = 0
    total_truncated = 0
    n_parquets = 0
    for path in sorted(glob.glob(str(stage4_dir / "*__matches.parquet"))):
        df = pd.read_parquet(path)
        n_parquets += 1
        total_rows += len(df)
        if "uncertainty_truncated" not in df.columns:
            print(f"  [rerun-scan] WARN: {Path(path).name} has no "
                  "`uncertainty_truncated` column; skipping.",
                  file=sys.stderr)
            continue
        mask = df["uncertainty_truncated"].astype(bool)
        # If a previous Task K merge added `truncation_cap_used`, only
        # re-run rows that were generated under the old cap (idempotency:
        # a row already processed at the new cap shouldn't be touched).
        if "truncation_cap_used" in df.columns:
            mask = mask & (df["truncation_cap_used"]
                           < (RERUN_TRUNCATION_HEAD_CHARS
                              + RERUN_TRUNCATION_TAIL_CHARS))
        rerun_df = df[mask]
        total_truncated += len(rerun_df)
        if len(rerun_df) == 0:
            continue
        for _, row in rerun_df.iterrows():
            oid = str(row["obligation_id"])
            try:
                docket, rule = _parse_obligation_id(oid)
            except ValueError as e:
                print(f"  [rerun-scan] {Path(path).name}: skipping row "
                      f"with bad obligation_id ({e})", file=sys.stderr)
                continue
            truncated_by_group.setdefault((docket, rule), []).append({
                "comment_id": str(row["comment_id"]),
                "obligation_id": oid,
                "cosine_similarity": float(row["cosine_similarity"]),
            })
    summary = {
        "n_parquets_scanned": n_parquets,
        "total_rows": total_rows,
        "total_truncated": total_truncated,
        "n_groups": len(truncated_by_group),
    }
    return truncated_by_group, summary


# ---------------------------------------------------------------------------
# Per-group data loading
# ---------------------------------------------------------------------------
def _load_obligations(docket_id: str, rule_type: str,
                      obligations_dir: Path = DEFAULT_OBLIGATIONS_DIR
                      ) -> list[dict]:
    path = (obligations_dir
            / f"path_a_obligations_verified_{docket_id}_{rule_type}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 1b verified CSV not found at {path}; the rerun script "
            "cannot rebuild prompts without the obligation metadata.")
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["obligation_id"] = obligation_id_of(r)
    return rows


def _load_comments_for_docket(docket_id: str,
                              comments_glob: str = DEFAULT_COMMENTS_GLOB):
    """Mirror the existing 13_stage4_run_per_anchor `_load_comments_for_docket`
    loader so the rerun re-uses identical comment text."""
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
    out = out[out["comment"].fillna("").str.len() > 0].reset_index(drop=True)
    return out.rename(columns={"document_id": "comment_id"})


# ---------------------------------------------------------------------------
# Tracking-CSV append
# ---------------------------------------------------------------------------
def _append_inflight_row(inflight_csv: Path, row: dict) -> None:
    inflight_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not inflight_csv.exists()
    with inflight_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INFLIGHT_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INFLIGHT_CSV_FIELDS})


# ---------------------------------------------------------------------------
# Per-group submission
# ---------------------------------------------------------------------------
def _submit_one_group(
    docket_id: str,
    rule_type: str,
    truncated_pairs: list[dict],
    *,
    inflight_csv: Path,
    model: str,
    head_chars: int,
    tail_chars: int,
    max_cost_usd: float,
    cumulative_cost_before: float,
    dry_run: bool,
) -> dict:
    """Submit one (docket, rule_type) group's truncated subset as a
    sharded batch and append tracking rows. Returns a summary dict."""
    import pandas as pd

    summary = {
        "docket_id": docket_id, "rule_type": rule_type,
        "n_truncated_pairs": len(truncated_pairs),
        "n_shards": 0, "status": "pending",
    }

    obligations = _load_obligations(docket_id, rule_type)
    comments_df = _load_comments_for_docket(docket_id)
    if comments_df is None or comments_df.empty:
        summary["status"] = "skipped"
        summary["reason"] = (f"no comments matched docket_id={docket_id} in "
                             f"{DEFAULT_COMMENTS_GLOB}")
        print(f"  [rerun] SKIP {docket_id}/{rule_type}: {summary['reason']}",
              file=sys.stderr)
        return summary
    odf = pd.DataFrame(obligations)

    # Restrict to only the truncated subset's comment_id + obligation_id
    # values so the prompt builder doesn't burn cycles on rows we already
    # have good results for.
    needed_cids = {p["comment_id"] for p in truncated_pairs}
    needed_oids = {p["obligation_id"] for p in truncated_pairs}
    comments_df = comments_df[comments_df["comment_id"].isin(needed_cids)]
    odf = odf[odf["obligation_id"].isin(needed_oids)]

    pairs = [
        (p["comment_id"], p["obligation_id"], p["cosine_similarity"])
        for p in truncated_pairs
    ]

    submission = prepare_batch_submission(
        pairs, comments_df, odf, model=model,
        head_chars=head_chars, tail_chars=tail_chars,
    )
    if submission["n_pairs"] == 0:
        summary["status"] = "skipped"
        summary["reason"] = "no submittable pairs after comment + obligation joins"
        print(f"  [rerun] SKIP {docket_id}/{rule_type}: {summary['reason']}",
              file=sys.stderr)
        return summary

    if dry_run:
        # Compute shard count without submitting.
        from stage4_llm_match import BATCH_API_MAX_REQUESTS_PER_BATCH
        shard_size = BATCH_API_MAX_REQUESTS_PER_BATCH
        n_shards = (submission["n_pairs"] + shard_size - 1) // shard_size
        summary["n_shards"] = n_shards
        summary["status"] = "dry-run"
        print(f"  [rerun] {docket_id}/{rule_type}: would submit "
              f"{submission['n_pairs']:,} pairs in {n_shards} shard(s) "
              f"(head={head_chars}, tail={tail_chars})", file=sys.stderr)
        return summary

    try:
        manifests = submit_batches_sharded(
            submission,
            model=model,
            batch_id_prefix=f"stage4_rerun_{docket_id}_{rule_type}",
            max_cost_usd=max_cost_usd - cumulative_cost_before,
        )
    except CostCapExceeded as e:
        summary["status"] = "failed"
        summary["reason"] = f"cost cap exceeded: {e}"
        print(f"  [rerun] FAIL {docket_id}/{rule_type}: {summary['reason']}",
              file=sys.stderr)
        return summary
    except Exception as e:  # noqa: BLE001
        summary["status"] = "failed"
        summary["reason"] = f"{type(e).__name__}: {e}"
        print(f"  [rerun] FAIL {docket_id}/{rule_type}: {summary['reason']}",
              file=sys.stderr)
        return summary

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
    summary["n_shards"] = len(manifests)
    summary["status"] = "submitted"
    summary["first_batch_id"] = manifests[0]["batch_id"]
    summary["job_id"] = manifests[0]["job_id"]
    print(f"  [rerun] {docket_id}/{rule_type}: submitted "
          f"{submission['n_pairs']:,} pairs in {len(manifests)} shard(s) "
          f"(first batch_id={manifests[0]['batch_id']})", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4_DIR,
                    help="Directory holding per-anchor matches parquets.")
    ap.add_argument("--rerun-inflight-csv", type=Path,
                    default=DEFAULT_RERUN_INFLIGHT_CSV,
                    help="Separate tracking CSV for the rerun batches. The "
                         "retrieve script reads this and writes its parquets "
                         "to --output-dir (suggested: "
                         "data/processed/stage4_rerun/).")
    ap.add_argument("--head-chars", type=int,
                    default=RERUN_TRUNCATION_HEAD_CHARS,
                    help="Comment-text head-char cap for the rerun prompt "
                         "(default 4000).")
    ap.add_argument("--tail-chars", type=int,
                    default=RERUN_TRUNCATION_TAIL_CHARS,
                    help="Comment-text tail-char cap for the rerun prompt "
                         "(default 1000).")
    ap.add_argument("--model", type=str, default="gpt-5")
    ap.add_argument("--max-cost", type=float, default=400.0,
                    help="Hard cumulative cost cap across all rerun groups. "
                         "Spec-estimated total: $200-250 (74K pairs × "
                         "$0.0019 × ~1.5x for longer prompts).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan + report shard plan; do not submit any batch.")
    args = ap.parse_args(argv)

    truncated_by_group, scan_summary = _find_truncated_pairs(args.stage4_dir)

    print(f"[rerun] scanned {scan_summary['n_parquets_scanned']} parquets: "
          f"{scan_summary['total_rows']:,} total rows, "
          f"{scan_summary['total_truncated']:,} truncated "
          f"({100.0 * scan_summary['total_truncated'] / max(1, scan_summary['total_rows']):.2f}%) "
          f"across {scan_summary['n_groups']} (docket, rule_type) groups",
          file=sys.stderr)

    if not truncated_by_group:
        print("[rerun] nothing to do — no truncated rows found.",
              file=sys.stderr)
        return 0

    summaries: list[dict] = []
    cumulative_cost = 0.0
    for (docket_id, rule_type), pairs in sorted(truncated_by_group.items()):
        s = _submit_one_group(
            docket_id, rule_type, pairs,
            inflight_csv=args.rerun_inflight_csv,
            model=args.model,
            head_chars=args.head_chars,
            tail_chars=args.tail_chars,
            max_cost_usd=args.max_cost,
            cumulative_cost_before=cumulative_cost,
            dry_run=args.dry_run,
        )
        summaries.append(s)
        if s.get("status") == "submitted":
            cumulative_cost += s["n_truncated_pairs"] * 0.0019 * 1.5

    n_submitted = sum(1 for s in summaries if s["status"] == "submitted")
    n_skipped = sum(1 for s in summaries if s["status"] == "skipped")
    n_failed = sum(1 for s in summaries if s["status"] == "failed")
    n_dry = sum(1 for s in summaries if s["status"] == "dry-run")
    total_pairs = sum(s["n_truncated_pairs"] for s in summaries
                      if s["status"] in ("submitted", "dry-run"))
    total_shards = sum(s["n_shards"] for s in summaries)

    print("\n[rerun] summary", file=sys.stderr)
    print(f"  groups: {len(summaries)} "
          f"(submitted={n_submitted}, dry-run={n_dry}, "
          f"skipped={n_skipped}, failed={n_failed})", file=sys.stderr)
    print(f"  total truncated pairs: {total_pairs:,}", file=sys.stderr)
    print(f"  total shards: {total_shards}", file=sys.stderr)
    if not args.dry_run and n_submitted:
        print(f"\n  Next steps:", file=sys.stderr)
        print(f"    python3 code/14_stage4_retrieve_batches.py \\\\",
              file=sys.stderr)
        print(f"        --inflight-csv {args.rerun_inflight_csv} \\\\",
              file=sys.stderr)
        print(f"        --output-dir data/processed/stage4_rerun/",
              file=sys.stderr)
        print(f"    python3 code/16_stage4_merge_rerun.py",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
