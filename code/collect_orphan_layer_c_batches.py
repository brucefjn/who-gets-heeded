#!/usr/bin/env python3
"""
Collector for orphan Layer C batches submitted 2026-05-13 03:05 UTC.

Background
----------
4 healthy batches were submitted in parallel — the ones that escaped the
org-wide 1.5M enqueued-token cap. Pollers were later pkill'd. By that
point, 2 small batches (83 + 146 pairs) had completed quickly on OpenAI's
side and their results were already collected by the orchestrator. The
other 2 (875 + 522 pairs) were still in_progress and got orphaned when
the polling pythons died.

This script polls all 4 batch_ids defensively. For the 2 that completed
before the pkill, OpenAI still has the output file for ~30 days, so
re-downloading is harmless insurance. For the 2 truly orphaned ones,
this is the primary collection path.

Output
------
data/intermediate/openai_batches/{job_id}__output.jsonl
data/intermediate/openai_batches/{job_id}__errors.jsonl   (if any errors)

Parsing the JSONL into per-docket CSVs is intentionally left to the
existing orchestrator's parser (called separately after this collector
returns); this script's only job is to get the bytes off OpenAI's
servers before they expire.

Run
---
    nohup python3 code/collect_orphan_layer_c_batches.py \
        > /tmp/orphan_collector.log 2>&1 &
    tail -f /tmp/orphan_collector.log
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    import openai
except ImportError:
    sys.stderr.write(
        "openai SDK not installed. "
        "Run: pip install openai --break-system-packages\n"
    )
    sys.exit(1)


# (docket, batch_id, job_id, expected_pairs)
ORPHAN_BATCHES = [
    ("EPA-HQ-OAR-2009-0234",
     "batch_6a03ea6b0d388190ab3ec0723c493dd3",
     "20260513T030513_f0725ee8", 875),
    ("EPA-HQ-OAR-2009-0517",
     "batch_6a03ea6cb2408190abd8dd70a0cb6eb9",
     "20260513T030515_b7cef45a", 83),
    ("EPA-HQ-OAR-2015-0531",
     "batch_6a03eaa4d8ac8190804521223fb7e43b",
     "20260513T030611_fd09d1bf", 146),
    ("EPA-HQ-OW-2017-0644",
     "batch_6a03eaa354a08190a765f40ced13e6fc",
     "20260513T030609_8b6d4bc7", 522),
]

OUTPUT_DIR = Path("data/intermediate/openai_batches")
POLL_INTERVAL_S = 60
TIMEOUT_HR = 14
TERMINAL_NON_SUCCESS = {"failed", "expired", "cancelled"}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log("ERROR: OPENAI_API_KEY not set in environment")
        return 1

    client = openai.OpenAI(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pending = {bid: (docket, job_id, expected)
               for docket, bid, job_id, expected in ORPHAN_BATCHES}
    completed: dict = {}
    failed: dict = {}
    start = time.time()

    log(f"Starting collector for {len(pending)} batches:")
    for docket, bid, job_id, expected in ORPHAN_BATCHES:
        log(f"  - {docket} | {bid} | expected {expected} pairs")

    while pending:
        elapsed_hr = (time.time() - start) / 3600.0
        if elapsed_hr > TIMEOUT_HR:
            log(f"Timeout {TIMEOUT_HR}hr reached. "
                f"{len(pending)} batches still pending.")
            break

        for bid in list(pending.keys()):
            docket, job_id, expected = pending[bid]
            try:
                b = client.batches.retrieve(bid)
            except Exception as e:
                log(f"  retrieve error for {bid}: {e}")
                continue

            counts = dict(b.request_counts) if b.request_counts else None
            log(f"  {docket}: status={b.status} counts={counts}")

            if b.status == "completed":
                if b.output_file_id:
                    content = client.files.content(b.output_file_id).read()
                    out_path = OUTPUT_DIR / f"{job_id}__output.jsonl"
                    out_path.write_bytes(content)
                    n_lines = content.count(b"\n")
                    log(f"  OK  {docket}: saved {len(content)} bytes "
                        f"({n_lines} lines) -> {out_path}")
                    completed[bid] = (docket, out_path, n_lines)
                else:
                    log(f"  WARN {docket}: completed but no output_file_id")
                if b.error_file_id:
                    err = client.files.content(b.error_file_id).read()
                    err_path = OUTPUT_DIR / f"{job_id}__errors.jsonl"
                    err_path.write_bytes(err)
                    log(f"  WARN {docket}: also saved errors -> {err_path}")
                del pending[bid]

            elif b.status in TERMINAL_NON_SUCCESS:
                log(f"  FAIL {docket}: status={b.status}; errors={b.errors}")
                failed[bid] = (docket, b.status)
                del pending[bid]

        if pending:
            time.sleep(POLL_INTERVAL_S)

    log("=" * 60)
    log(f"DONE. completed={len(completed)} failed={len(failed)} "
        f"still_pending={len(pending)}")
    for bid, (docket, path, n) in completed.items():
        log(f"  OK   {docket}: {n} lines -> {path}")
    for bid, (docket, status) in failed.items():
        log(f"  FAIL {docket}: {status}")
    for bid, (docket, _, _) in pending.items():
        log(f"  PEND {docket}: still pending at script timeout")

    return 0 if not failed and not pending else 2


if __name__ == "__main__":
    sys.exit(main())
