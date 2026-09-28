"""
14_stage4_retrieve_batches.py — poll + fold Stage 4 batches submitted in
`--submit-only` mode.

Reads `data/processed/stage4_inflight_batches.csv` (written by
`code/13_stage4_run_per_anchor.py --submit-only`), polls every batch in
parallel via asyncio + thread-dispatched OpenAI calls, and as each one
completes writes the canonical per-anchor parquet at
`data/processed/stage4/<docket_id>__matches.parquet`.

The intent: keep this process lightweight (no sentence-transformer
model load, no embedding compute) so polling 26+ in-flight batches
costs roughly the same RAM as a tiny long-running asyncio program.

Usage:
    python3 code/14_stage4_retrieve_batches.py
    python3 code/14_stage4_retrieve_batches.py --inflight-csv path
    python3 code/14_stage4_retrieve_batches.py --seed-orphans

`--seed-orphans` writes the three 2026-05-14 in-flight rows (the batches
that were already submitted by the old submit-then-poll path before
Task I landed) into the tracking CSV. Those rows have no manifest
sidecar, so the retrieve script will log a clear instruction pointing
to the orphan-reconstruction workflow (mirroring Yue's
`parse_orphan_layer_c_outputs.py` pattern).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))


DEFAULT_INFLIGHT_CSV = Path("data/processed/stage4_inflight_batches.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/stage4")
INFLIGHT_CSV_FIELDS = (
    "docket_id", "rule_type", "batch_id", "job_id", "input_jsonl",
    "manifest_json", "n_pairs", "submitted_at_utc", "model", "status",
)
TERMINAL_NON_SUCCESS = {"failed", "expired", "cancelled"}

# Polling defaults: short start + cap at 5 min so we don't hammer the
# Batch API endpoint, while still picking up completions reasonably
# promptly. The poll loop is async + per-batch — independent retries.
INITIAL_POLL_SECONDS = 30.0
MAX_POLL_SECONDS = 300.0
DEFAULT_TIMEOUT_HOURS = 24.0

# Pre-seeded orphan batches from 2026-05-14. These were submitted by
# the old submit-then-poll path before Task I landed; they have no
# manifest sidecar. The seed helper writes them with `manifest_json=""`
# so the retrieve loop logs a clear orphan-recovery instruction
# (mirrors Yue's `parse_orphan_layer_c_outputs.py` pattern).
ORPHAN_SEED_ROWS = [
    {
        "docket_id": "EPA-HQ-OAR-2009-0491", "rule_type": "proposed",
        "batch_id": "batch_6a062f67561881908d5e18acde15dcc9",
        "job_id": "", "input_jsonl": "", "manifest_json": "",
        "n_pairs": "", "submitted_at_utc": "2026-05-14T07:00:00Z",
        "model": "gpt-5", "status": "orphan",
    },
    {
        "docket_id": "EPA-HQ-OAR-2009-0926", "rule_type": "proposed",
        "batch_id": "batch_6a062f64cfb08190a498887d0d4d587b",
        "job_id": "", "input_jsonl": "", "manifest_json": "",
        "n_pairs": "", "submitted_at_utc": "2026-05-14T07:00:00Z",
        "model": "gpt-5", "status": "orphan",
    },
    {
        "docket_id": "EPA-HQ-OAR-2011-0817", "rule_type": "proposed",
        "batch_id": "batch_6a062f631ffc8190bd070a2d4f3f5673",
        "job_id": "", "input_jsonl": "", "manifest_json": "",
        "n_pairs": "", "submitted_at_utc": "2026-05-14T07:00:00Z",
        "model": "gpt-5", "status": "orphan",
    },
]


@dataclass
class TrackingRow:
    docket_id: str
    rule_type: str
    batch_id: str
    job_id: str
    input_jsonl: str
    manifest_json: str
    n_pairs: int
    submitted_at_utc: str
    model: str
    status: str


@dataclass
class RetrieveResult:
    row: TrackingRow
    final_status: str
    n_written: int = 0
    out_path: Optional[Path] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    # Task J: each per-shard worker attaches its DataFrame here.
    # `_merge_and_write` groups by (docket_id, rule_type) and writes one
    # parquet per anchor after all shards complete. For unsharded
    # anchors the group has size 1 and behavior is identical to the
    # pre-Task-J path.
    df: object = None


# ---------------------------------------------------------------------------
# Tracking-CSV I/O
# ---------------------------------------------------------------------------
def _read_inflight(path: Path) -> list[TrackingRow]:
    if not path.exists():
        return []
    out: list[TrackingRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                n = int(row.get("n_pairs") or 0)
            except (TypeError, ValueError):
                n = 0
            out.append(TrackingRow(
                docket_id=row.get("docket_id", ""),
                rule_type=row.get("rule_type", ""),
                batch_id=row.get("batch_id", ""),
                job_id=row.get("job_id", ""),
                input_jsonl=row.get("input_jsonl", ""),
                manifest_json=row.get("manifest_json", ""),
                n_pairs=n,
                submitted_at_utc=row.get("submitted_at_utc", ""),
                model=row.get("model", "gpt-5"),
                status=row.get("status", "submitted"),
            ))
    return out


def _write_inflight(path: Path, rows: list[TrackingRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INFLIGHT_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "docket_id": r.docket_id, "rule_type": r.rule_type,
                "batch_id": r.batch_id, "job_id": r.job_id,
                "input_jsonl": r.input_jsonl,
                "manifest_json": r.manifest_json,
                "n_pairs": r.n_pairs,
                "submitted_at_utc": r.submitted_at_utc,
                "model": r.model, "status": r.status,
            })


def _seed_orphans(path: Path) -> int:
    """Append the 3 pre-Task-I orphan rows if not already present.
    Returns the number of rows added."""
    existing = _read_inflight(path)
    existing_ids = {r.batch_id for r in existing}
    to_add = [r for r in ORPHAN_SEED_ROWS
              if r["batch_id"] not in existing_ids]
    if not to_add:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INFLIGHT_CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in to_add:
            w.writerow({k: r.get(k, "") for k in INFLIGHT_CSV_FIELDS})
    return len(to_add)


# ---------------------------------------------------------------------------
# Per-batch async worker
# ---------------------------------------------------------------------------
async def _retrieve_one(
    row: TrackingRow,
    output_dir: Path,
    *,
    client,
    timeout_hours: float,
    initial_poll_s: float,
    max_poll_s: float,
    log_prefix: str = "",
    sleep_fn=None,
    time_fn=None,
) -> RetrieveResult:
    """Poll one batch with exponential backoff; on completion download +
    fold via the manifest; write per-anchor parquet."""
    from stage4_llm_match import collect_batch_once

    if sleep_fn is None:
        sleep_fn = asyncio.sleep
    if time_fn is None:
        time_fn = time.time

    res = RetrieveResult(row=row, final_status="pending")
    if not row.batch_id:
        res.final_status = "skipped"
        res.error = "no batch_id"
        return res

    # Orphans (no manifest_json) — surface the clear path forward.
    if not row.manifest_json or row.manifest_json == "":
        res.final_status = "orphan_no_manifest"
        res.error = (
            f"orphan batch {row.batch_id} has no manifest sidecar "
            "(submitted before --submit-only landed). To recover, "
            "re-run the prefilter for this docket and rebuild the "
            "pair-meta mapping (see notes — orphan-reconstruction "
            "workflow mirrors `parse_orphan_layer_c_outputs.py`).")
        print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
              f"ORPHAN {row.batch_id} — manifest missing; logging + skipping",
              file=sys.stderr)
        return res

    delay = float(initial_poll_s)
    t_start = time_fn()
    while True:
        elapsed_hr = (time_fn() - t_start) / 3600.0
        if elapsed_hr > timeout_hours:
            res.final_status = "timeout"
            res.error = (f"batch {row.batch_id} did not complete within "
                         f"{timeout_hours}h")
            print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
                  f"TIMEOUT {row.batch_id}", file=sys.stderr)
            return res

        try:
            # `collect_batch_once` is sync; run on a thread so the
            # asyncio loop can multiplex many in-flight polls.
            status, df = await asyncio.to_thread(
                collect_batch_once,
                row.batch_id, Path(row.manifest_json), client=client,
                output_dir=Path(row.input_jsonl).parent if row.input_jsonl else None,
            )
        except Exception as e:  # noqa: BLE001
            res.final_status = "error"
            res.error = f"{type(e).__name__}: {e}"
            print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
                  f"ERROR polling {row.batch_id}: {e}", file=sys.stderr)
            return res

        print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
              f"{row.batch_id} status={status} elapsed={elapsed_hr:.2f}h",
              file=sys.stderr)

        if status == "completed":
            res.final_status = "completed"
            res.elapsed_s = time_fn() - t_start
            if df is not None and len(df) > 0:
                # Task J: defer the parquet write to `_merge_and_write`
                # so sharded anchors get concatenated per (docket_id,
                # rule_type) before emission. For unsharded anchors
                # the group has one shard and behavior is identical.
                res.df = df
                print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
                      f"shard ready ({len(df):,} rows) — pending merge",
                      file=sys.stderr)
            else:
                res.error = "completed batch produced 0 rows"
                print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
                      f"WARN — completed batch had 0 rows", file=sys.stderr)
            return res

        if status in TERMINAL_NON_SUCCESS:
            res.final_status = status
            res.error = f"batch {row.batch_id} ended with status={status}"
            print(f"{log_prefix}[retrieve] {row.docket_id}/{row.rule_type}: "
                  f"FAIL {row.batch_id} status={status}", file=sys.stderr)
            return res

        await sleep_fn(delay)
        delay = min(delay * 2.0, max_poll_s)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
async def _run_all(
    rows: list[TrackingRow],
    output_dir: Path,
    *,
    client,
    timeout_hours: float,
    initial_poll_s: float,
    max_poll_s: float,
) -> list[RetrieveResult]:
    """Fire one async worker per tracking row; await all of them in
    parallel. asyncio.gather preserves submission order."""
    tasks = [
        asyncio.create_task(_retrieve_one(
            row, output_dir,
            client=client,
            timeout_hours=timeout_hours,
            initial_poll_s=initial_poll_s,
            max_poll_s=max_poll_s,
        ))
        for row in rows
    ]
    return await asyncio.gather(*tasks)


def _merge_and_write(
    results: list[RetrieveResult], output_dir: Path,
) -> None:
    """Group completed shards by (docket_id, rule_type) and write one
    parquet per anchor. Task J (2026-05-14): replaces the per-shard
    parquet emission so anchors that were sharded at submit time (e.g.
    EPA-HQ-OAR-2017-0355 at 157,883 pairs → 4 shards) get their
    per-shard DataFrames concatenated before disk emission.

    For unsharded anchors (1 row per (docket, rule_type)), this writes
    the single shard's DataFrame as-is — observable behavior matches
    the pre-Task-J path exactly. Mutates each per-shard
    `RetrieveResult` to set `out_path` and `n_written` (the latter
    reports the TOTAL merged row count, not just the shard's
    contribution — useful for the summary log).
    """
    from collections import defaultdict
    import pandas as pd

    groups: dict[tuple[str, str], list[RetrieveResult]] = defaultdict(list)
    for res in results:
        if (res.final_status == "completed"
                and res.df is not None and len(res.df) > 0):
            groups[(res.row.docket_id, res.row.rule_type)].append(res)

    for (docket_id, rule_type), shard_results in groups.items():
        dfs = [r.df for r in shard_results]
        if len(dfs) == 1:
            combined = dfs[0]
        else:
            combined = pd.concat(dfs, ignore_index=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{docket_id}__matches.parquet"
        combined.to_parquet(out_path, index=False)
        total = len(combined)
        for r in shard_results:
            r.out_path = out_path
            r.n_written = total
        if len(shard_results) == 1:
            print(f"[retrieve] {docket_id}/{rule_type}: wrote "
                  f"{total:,} rows -> {out_path}", file=sys.stderr)
        else:
            print(f"[retrieve] {docket_id}/{rule_type}: merged "
                  f"{len(shard_results)} shards, wrote {total:,} rows "
                  f"-> {out_path}", file=sys.stderr)


def _update_status_in_csv(
    inflight_csv: Path, results: list[RetrieveResult],
) -> None:
    """After all workers complete, update each tracking row's status
    column. Status transitions: submitted -> completed/failed/expired/
    cancelled/timeout/orphan_no_manifest/error."""
    rows = _read_inflight(inflight_csv)
    by_id = {r.batch_id: r for r in rows}
    for res in results:
        if res.row.batch_id in by_id:
            by_id[res.row.batch_id].status = res.final_status
    _write_inflight(inflight_csv, rows)


def _print_summary(results: list[RetrieveResult]) -> None:
    n_completed = sum(1 for r in results if r.final_status == "completed")
    n_orphan = sum(1 for r in results if r.final_status == "orphan_no_manifest")
    n_failed = sum(1 for r in results
                   if r.final_status in TERMINAL_NON_SUCCESS)
    n_timeout = sum(1 for r in results if r.final_status == "timeout")
    n_error = sum(1 for r in results if r.final_status == "error")
    total_rows = sum(r.n_written for r in results)

    print("\n[retrieve] summary", file=sys.stderr)
    print(f"  batches: {len(results)} "
          f"(completed={n_completed}, failed={n_failed}, "
          f"timeout={n_timeout}, orphan={n_orphan}, error={n_error})",
          file=sys.stderr)
    print(f"  total rows written: {total_rows:,}", file=sys.stderr)
    if n_orphan or n_failed or n_timeout or n_error:
        print("\n  attention required:", file=sys.stderr)
        for r in results:
            if r.final_status in ("completed",):
                continue
            print(f"    {r.row.docket_id}/{r.row.rule_type} "
                  f"({r.row.batch_id}): {r.final_status} — {r.error}",
                  file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inflight-csv", type=Path, default=DEFAULT_INFLIGHT_CSV)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed-orphans", action="store_true",
                    help="Append the three 2026-05-14 pre-Task-I orphan "
                         "rows to the tracking CSV (idempotent) and exit.")
    ap.add_argument("--timeout-hours", type=float,
                    default=DEFAULT_TIMEOUT_HOURS)
    ap.add_argument("--initial-poll-seconds", type=float,
                    default=INITIAL_POLL_SECONDS)
    ap.add_argument("--max-poll-seconds", type=float,
                    default=MAX_POLL_SECONDS)
    args = ap.parse_args(argv)

    if args.seed_orphans:
        n = _seed_orphans(args.inflight_csv)
        print(f"[retrieve] seeded {n} orphan row(s) into {args.inflight_csv}",
              file=sys.stderr)
        return 0

    rows = _read_inflight(args.inflight_csv)
    rows = [r for r in rows if r.status not in ("completed",)
            + tuple(TERMINAL_NON_SUCCESS)]
    if not rows:
        print(f"[retrieve] no in-flight rows in {args.inflight_csv}; "
              "nothing to do.", file=sys.stderr)
        return 0

    print(f"[retrieve] polling {len(rows)} batches from {args.inflight_csv}",
          file=sys.stderr)

    import openai
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[retrieve] ERROR: OPENAI_API_KEY env var not set",
              file=sys.stderr)
        return 1
    client = openai.OpenAI(api_key=api_key)

    results = asyncio.run(_run_all(
        rows, args.output_dir,
        client=client,
        timeout_hours=args.timeout_hours,
        initial_poll_s=args.initial_poll_seconds,
        max_poll_s=args.max_poll_seconds,
    ))
    # Task J: merge per-shard DataFrames into one parquet per anchor
    # (no-op for unsharded anchors — group size 1).
    _merge_and_write(results, args.output_dir)
    _update_status_in_csv(args.inflight_csv, results)
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
