"""
parse_orphan_layer_c_outputs.py — Reconstruct the canonical Layer C CSV
from orphaned OpenAI Batch API output JSONLs.

The 4 batches below completed cleanly on OpenAI's side but their polling
pythons were pkill'd before the orchestrator could fold them into
data/processed/layer_c_full.csv. A separate orphan collector retrieved
the raw output JSONLs; this script reconstructs the canonical CSV by:

  1. For each (docket, job_id), re-load Tier B for that docket using the
     exact filtering chain the orchestrator used (_load_tier_b +
     _prepare_comment_rows). This reproduces the submission-order index
     that custom_id="pair_{idx}" refers to.
  2. Parse the output JSONL, look up rows[idx] for comment_id, validate
     the model's JSON with the existing _validated_layer_c_dict, and
     build the canonical record via _result_to_record (via="batch", so
     batched pricing applies — these were genuine batch calls).

Writes a fresh CSV (overwrites). Per-row failures are skipped + logged;
the script never aborts on a single bad row.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from layer_c_judge import (  # noqa: E402
    _canonical_columns,
    _prepare_comment_rows,
    _result_to_record,
    _validated_layer_c_dict,
    truncate_for_layer_c,
)
from path_a_obligation_llm import _strip_codefence  # noqa: E402


# Locked at the orphan event (2026-05-13 03:05 UTC). Order doesn't
# matter; per-docket counts are reported in the summary.
ORPHAN_BATCHES = [
    {"docket_id": "EPA-HQ-OAR-2009-0234",
     "job_id": "20260513T030513_f0725ee8", "pairs": 875},
    {"docket_id": "EPA-HQ-OAR-2009-0517",
     "job_id": "20260513T030515_b7cef45a", "pairs": 83},
    {"docket_id": "EPA-HQ-OAR-2015-0531",
     "job_id": "20260513T030611_fd09d1bf", "pairs": 146},
    {"docket_id": "EPA-HQ-OW-2017-0644",
     "job_id": "20260513T030609_8b6d4bc7", "pairs": 522},
]

DEFAULT_ANCHORS_CSV = Path("data/processed/anchor_rules_locked.csv")
DEFAULT_COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"
DEFAULT_BATCHES_DIR = Path("data/intermediate/openai_batches")
DEFAULT_OUTPUT = Path("data/processed/layer_c_full.csv")


def _load_orchestrator_helpers():
    """`16_layer_c_run_full.py` starts with a digit so we can't import it
    by name; use importlib to lift its private _load_anchor_dockets and
    _load_tier_b helpers. Reusing keeps the parquet glob + filter chain
    in lock-step with what the orchestrator originally submitted."""
    spec = importlib.util.spec_from_file_location(
        "layer_c_orchestrator",
        _REPO_ROOT / "code" / "16_layer_c_run_full.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_anchor_dockets, mod._load_tier_b


def _parse_one_batch(
    docket_id: str,
    job_id: str,
    rows: list[dict],
    batches_dir: Path,
) -> tuple[list[dict], dict]:
    """Parse one orphan output JSONL. Returns (records, stats). Stats
    keys: n_parsed, n_validation_fail, n_missing_response, n_bad_json,
    n_skipped_oor, n_bad_lines."""
    stats = {
        "n_parsed": 0,
        "n_validation_fail": 0,
        "n_missing_response": 0,
        "n_bad_json": 0,
        "n_skipped_oor": 0,
        "n_bad_lines": 0,
    }
    output_path = batches_dir / f"{job_id}__output.jsonl"
    if not output_path.exists():
        print(f"  [orphan-parse] WARN: {output_path} missing — skipping "
              f"docket {docket_id}", file=sys.stderr)
        return [], stats

    records: list[dict] = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                stats["n_bad_lines"] += 1
                print(f"  [orphan-parse] {job_id}: bad JSONL line: {e}",
                      file=sys.stderr)
                continue

            custom_id = row.get("custom_id", "")
            if not isinstance(custom_id, str) or not custom_id.startswith("pair_"):
                stats["n_bad_lines"] += 1
                continue
            try:
                idx = int(custom_id.split("_", 1)[1])
            except ValueError:
                stats["n_bad_lines"] += 1
                continue
            if idx < 0 or idx >= len(rows):
                stats["n_skipped_oor"] += 1
                print(f"  [orphan-parse] {job_id}: pair_{idx} out of range "
                      f"(have {len(rows)} rows)", file=sys.stderr)
                continue

            err = row.get("error")
            response = row.get("response")
            if err is not None or response is None:
                stats["n_missing_response"] += 1
                print(f"  [orphan-parse] {job_id} pair_{idx}: err={err!r}, "
                      f"no response", file=sys.stderr)
                continue

            body = response.get("body") or {}
            usage = body.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or 0)

            try:
                choices = body.get("choices") or []
                content = choices[0]["message"]["content"]
            except (IndexError, KeyError, TypeError) as e:
                stats["n_missing_response"] += 1
                print(f"  [orphan-parse] {job_id} pair_{idx}: missing "
                      f"content: {e}", file=sys.stderr)
                continue

            try:
                parsed = json.loads(_strip_codefence(content))
            except json.JSONDecodeError as e:
                stats["n_bad_json"] += 1
                print(f"  [orphan-parse] {job_id} pair_{idx}: JSON decode: "
                      f"{e}", file=sys.stderr)
                continue

            try:
                validated = _validated_layer_c_dict(parsed)
            except Exception as e:  # noqa: BLE001
                stats["n_validation_fail"] += 1
                print(f"  [orphan-parse] {job_id} pair_{idx}: validation: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                continue

            # _result_to_record reads `in_tokens`, `out_tokens`, `via`
            # off the result dict; via="batch" routes through batched
            # pricing (these WERE batch calls).
            result_dict = dict(validated)
            result_dict["in_tokens"] = in_tok
            result_dict["out_tokens"] = out_tok
            result_dict["via"] = "batch"

            comment_row = rows[idx]
            # truncation_flag isn't stored in the JSONL — recompute from
            # the original comment text. The policy is deterministic so
            # the recomputed flag matches what the orchestrator wrote.
            _, was_truncated = truncate_for_layer_c(comment_row["comment"])

            record = _result_to_record(comment_row, result_dict, was_truncated)
            records.append(record)
            stats["n_parsed"] += 1

    return records, stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS_CSV)
    ap.add_argument("--comments-glob", type=str, default=DEFAULT_COMMENTS_GLOB)
    ap.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES_DIR)
    args = ap.parse_args(argv)

    import pandas as pd  # local import: keeps --help cheap

    load_anchor_dockets, load_tier_b = _load_orchestrator_helpers()
    anchors = load_anchor_dockets(args.anchors_csv)
    print(f"[orphan-parse] loaded {len(anchors)} anchor dockets from "
          f"{args.anchors_csv}", file=sys.stderr)

    all_records: list[dict] = []
    per_docket: dict[str, dict] = {}
    grand_stats = {
        "n_parsed": 0, "n_validation_fail": 0, "n_missing_response": 0,
        "n_bad_json": 0, "n_skipped_oor": 0, "n_bad_lines": 0,
    }

    for batch in ORPHAN_BATCHES:
        docket_id = batch["docket_id"]
        job_id = batch["job_id"]
        expected_pairs = batch["pairs"]

        df = load_tier_b(anchors, args.comments_glob, docket_id)
        rows = _prepare_comment_rows(df)
        print(f"[orphan-parse] {docket_id}: reloaded {len(rows)} rows "
              f"(batch had {expected_pairs} pairs)", file=sys.stderr)
        if len(rows) != expected_pairs:
            print(f"  [orphan-parse] WARN: row-count mismatch "
                  f"({len(rows)} reloaded vs {expected_pairs} submitted). "
                  f"Submission-order recovery may be wrong.",
                  file=sys.stderr)

        records, stats = _parse_one_batch(
            docket_id, job_id, rows, args.batches_dir,
        )
        all_records.extend(records)
        per_docket[docket_id] = {
            "count": len(records), "expected": expected_pairs, "stats": stats,
        }
        for k in grand_stats:
            grand_stats[k] += stats[k]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_records, columns=_canonical_columns())
    out_df.to_csv(args.output, index=False)
    print(f"\n[orphan-parse] wrote {len(out_df):,} rows to {args.output}",
          file=sys.stderr)

    print("\n[orphan-parse] per-docket row counts:", file=sys.stderr)
    for docket_id, info in per_docket.items():
        delta = info["expected"] - info["count"]
        marker = "" if delta == 0 else f"  (Δ -{delta})"
        anomalies = []
        for k, v in info["stats"].items():
            if k == "n_parsed":
                continue
            if v:
                anomalies.append(f"{k}={v}")
        anomaly_str = f"  [{', '.join(anomalies)}]" if anomalies else ""
        print(f"  {docket_id}: {info['count']:,} / {info['expected']:,}"
              f"{marker}{anomaly_str}", file=sys.stderr)

    total_cost = float(out_df["cost_usd"].sum()) if len(out_df) else 0.0
    print(f"\n[orphan-parse] total cost recovered: ${total_cost:,.4f}",
          file=sys.stderr)
    print(f"[orphan-parse] grand totals: "
          f"parsed={grand_stats['n_parsed']:,}  "
          f"validation_fail={grand_stats['n_validation_fail']}  "
          f"missing_response={grand_stats['n_missing_response']}  "
          f"bad_json={grand_stats['n_bad_json']}  "
          f"bad_lines={grand_stats['n_bad_lines']}  "
          f"skipped_oor={grand_stats['n_skipped_oor']}",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
