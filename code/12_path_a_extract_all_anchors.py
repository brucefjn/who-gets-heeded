"""
12_path_a_extract_all_anchors.py — Path A Stage 2 production orchestrator.

Iterates Stage 1a (heuristic candidate extraction) + Stage 1b (LLM
verification) over every anchor in data/processed/anchor_rules_locked.csv,
for proposed and/or final FR documents, and writes per-anchor CSVs at the
canonical paths Stage 4 consumes:

  data/processed/path_a_obligations_candidates_<docket>_<rule_type>.csv
  data/processed/path_a_obligations_verified_<docket>_<rule_type>.csv

The canonical FR document per (docket_id, rule_type) is resolved through
data/processed/federal_register_index.csv. When the index has multiple
documents for a single (docket, rule_type) — e.g. an extension notice
alongside the actual NPRM — we pick the one with the largest n_chars.
This is the same document the in-sample 2018-0775 final extraction used
(2019-11653 at 372k chars).

Per PROJECT_FACTS §8 the Stage 2 production budget is ~$60 batched; the
default --max-cost is 80.0 to leave headroom. The orchestrator tracks
cumulative cost across anchors and raises CostCapExceeded before
overrunning the cap.

Usage:
    # Single anchor smoke (the in-sample docket)
    python code/12_path_a_extract_all_anchors.py \\
        --docket EPA-HQ-OAR-2018-0775 --rule-type final --dry-run

    # Production run
    python code/12_path_a_extract_all_anchors.py --all --skip-existing

Per-anchor failures do not kill the run: the failure is logged with
traceback, the next anchor proceeds, and the end-of-run summary lists
every failed (docket, rule_type) with its error category.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make code/lib importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from path_a_obligation_heuristic import (  # noqa: E402
    extract_candidates_from_file,
    write_candidates_csv,
)
from path_a_obligation_llm import (  # noqa: E402
    CostCapExceeded,
    DEFAULT_OPENAI_MODEL,
    GPT5_BATCH_INPUT_PRICE_PER_M,
    GPT5_BATCH_OUTPUT_PRICE_PER_M,
    GPT5_STANDARD_INPUT_PRICE_PER_M,
    GPT5_STANDARD_OUTPUT_PRICE_PER_M,
    run_primary_verification,
)


DEFAULT_ANCHORS_CSV = Path("data/processed/anchor_rules_locked.csv")
DEFAULT_FR_INDEX = Path("data/processed/federal_register_index.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_MAX_COST = 80.0   # ~$20 headroom over the $60 PROJECT_FACTS §8 budget


@dataclass
class AnchorTarget:
    docket_id: str
    rule_type: str
    fr_text_path: Path
    fr_doc_number: str
    n_chars: int


@dataclass
class AnchorResult:
    docket_id: str
    rule_type: str
    status: str   # 'ok' | 'skipped' | 'failed' | 'dry-run'
    reason: str = ""
    n_candidates: int = 0
    n_verified: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    candidates_csv: Optional[Path] = None
    verified_csv: Optional[Path] = None
    fr_doc_number: str = ""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _load_anchors(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [row["docket_id"] for row in csv.DictReader(f)]


def _load_fr_index(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_canonical_fr(
    fr_index: list[dict], docket_id: str, rule_type: str,
) -> Optional[dict]:
    """Return the largest-by-n_chars FR document for this (docket, rule_type),
    or None if the index has no entry. Largest-by-n_chars discriminates
    the actual NPRM from short ancillary documents (extension notices,
    correction notices) that share the same (docket, rule_type) pair."""
    matches = [r for r in fr_index
               if r.get("docket_id") == docket_id
               and r.get("rule_type") == rule_type]
    if not matches:
        return None
    def _nchars(r):
        try:
            return int(r.get("n_chars") or 0)
        except (TypeError, ValueError):
            return 0
    matches.sort(key=_nchars, reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------
def plan_targets(
    anchor_dockets: list[str],
    fr_index: list[dict],
    rule_types: list[str],
    output_dir: Path,
    *,
    skip_existing: bool,
) -> tuple[list[AnchorTarget], list[AnchorResult]]:
    """Resolve each (docket, rule_type) to either a runnable target or
    a pre-canned skipped/ok result. Skip reasons are surfaced verbatim
    in the final run summary."""
    targets: list[AnchorTarget] = []
    pre_results: list[AnchorResult] = []
    for docket in anchor_dockets:
        for rt in rule_types:
            verified_csv = (output_dir
                            / f"path_a_obligations_verified_{docket}_{rt}.csv")
            if skip_existing and verified_csv.exists():
                pre_results.append(AnchorResult(
                    docket_id=docket, rule_type=rt, status="skipped",
                    reason=f"--skip-existing and {verified_csv.name} already present",
                    verified_csv=verified_csv,
                ))
                continue
            fr_row = _pick_canonical_fr(fr_index, docket, rt)
            if fr_row is None:
                pre_results.append(AnchorResult(
                    docket_id=docket, rule_type=rt, status="skipped",
                    reason="no entry in federal_register_index for this (docket, rule_type)",
                ))
                continue
            text_path = Path(fr_row.get("text_path") or "")
            if not text_path.is_absolute():
                text_path = _REPO_ROOT / text_path
            if not text_path.exists():
                pre_results.append(AnchorResult(
                    docket_id=docket, rule_type=rt, status="skipped",
                    reason=f"FR text file not found at {text_path}",
                ))
                continue
            try:
                n_chars = int(fr_row.get("n_chars") or 0)
            except (TypeError, ValueError):
                n_chars = 0
            targets.append(AnchorTarget(
                docket_id=docket, rule_type=rt,
                fr_text_path=text_path,
                fr_doc_number=str(fr_row.get("fr_doc_number") or ""),
                n_chars=n_chars,
            ))
    return targets, pre_results


# ---------------------------------------------------------------------------
# Per-target execution
# ---------------------------------------------------------------------------
def _cost_from_tokens(in_tok: int, out_tok: int, *, batch: bool) -> float:
    if batch:
        return (in_tok * GPT5_BATCH_INPUT_PRICE_PER_M / 1_000_000.0
                + out_tok * GPT5_BATCH_OUTPUT_PRICE_PER_M / 1_000_000.0)
    return (in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
            + out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0)


def run_one_target(
    target: AnchorTarget,
    *,
    output_dir: Path,
    model: str,
    batch: bool,
    cumulative_cost_before: float,
    max_cost_usd: float,
    concurrency: int = 1,
    extract_fn=extract_candidates_from_file,
    write_candidates_fn=write_candidates_csv,
    verify_fn=run_primary_verification,
) -> AnchorResult:
    """Run Stage 1a + Stage 1b for one (docket, rule_type). The Stage 1a
    + 1b entry points are injectable via the *_fn kwargs so tests can
    swap in stubs and exercise the orchestrator's error-handling and
    cost-accounting paths without making real LLM calls."""
    t_start = time.time()
    result = AnchorResult(
        docket_id=target.docket_id, rule_type=target.rule_type,
        status="ok", fr_doc_number=target.fr_doc_number,
    )

    # Pre-check cost cap (before Stage 1b incurs any cost).
    remaining_budget = max_cost_usd - cumulative_cost_before
    if remaining_budget <= 0:
        raise CostCapExceeded(
            f"Cost cap ${max_cost_usd:,.2f} already reached "
            f"(cumulative=${cumulative_cost_before:.4f}) before "
            f"{target.docket_id}/{target.rule_type}."
        )

    candidates_csv = (output_dir
                      / f"path_a_obligations_candidates_{target.docket_id}_{target.rule_type}.csv")
    verified_csv = (output_dir
                    / f"path_a_obligations_verified_{target.docket_id}_{target.rule_type}.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1a ----
    print(f"  [stage1a] {target.docket_id}/{target.rule_type}  "
          f"fr={target.fr_doc_number} text={target.fr_text_path}",
          file=sys.stderr)
    cands = extract_fn(target.fr_text_path, target.docket_id, target.rule_type)
    cands_list = list(cands)
    n_cands = write_candidates_fn(cands_list, candidates_csv)
    result.n_candidates = n_cands
    result.candidates_csv = candidates_csv
    print(f"  [stage1a] wrote {n_cands} candidates to {candidates_csv.name}",
          file=sys.stderr)

    if n_cands == 0:
        result.status = "ok"
        result.reason = "Stage 1a produced zero candidates; no LLM call made."
        result.verified_csv = None
        result.elapsed_s = time.time() - t_start
        return result

    # ---- Stage 1b ----
    print(f"  [stage1b] verifying {n_cands} candidates "
          f"(batch={batch})", file=sys.stderr)
    summary = verify_fn(
        candidates_csv=candidates_csv,
        binding_text_path=target.fr_text_path,
        out_csv=verified_csv,
        provider="openai",
        model=model,
        batch=batch,
        max_cost_usd=remaining_budget,
        concurrency=concurrency,
    )
    in_tok = int(summary.get("total_in_tokens", 0) or 0)
    out_tok = int(summary.get("total_out_tokens", 0) or 0)
    cost = _cost_from_tokens(in_tok, out_tok, batch=batch)

    result.in_tokens = in_tok
    result.out_tokens = out_tok
    result.cost_usd = cost
    result.n_verified = int(summary.get("verified_rows_written", 0) or 0)
    result.verified_csv = verified_csv
    result.elapsed_s = time.time() - t_start
    print(f"  [stage1b] verified={result.n_verified}  "
          f"in_tok={in_tok:,} out_tok={out_tok:,}  "
          f"cost=${cost:.4f}  elapsed={result.elapsed_s:.1f}s",
          file=sys.stderr)

    # Hard post-check (in case the per-call cost overshot the budget).
    if cumulative_cost_before + cost > max_cost_usd:
        raise CostCapExceeded(
            f"Cost cap ${max_cost_usd:,.2f} exceeded after "
            f"{target.docket_id}/{target.rule_type} "
            f"(cumulative=${cumulative_cost_before + cost:.4f}). "
            "Per-anchor CSV was written; remaining anchors will not run."
        )
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_orchestrator(args, *, stage1a_fn=None, stage1b_fn=None,
                     write_csv_fn=None) -> list[AnchorResult]:
    """Top-level runner. The Stage 1a / 1b entry points can be injected
    for unit tests; defaults use the real lib implementations.

    `args` is an argparse.Namespace (or attribute-compatible object) with
    the CLI fields documented in `_parse_args`."""
    extract_fn = stage1a_fn or extract_candidates_from_file
    write_fn = write_csv_fn or write_candidates_csv
    verify_fn = stage1b_fn or run_primary_verification

    anchor_dockets = (
        [args.docket] if getattr(args, "docket", None)
        else _load_anchors(args.anchors_csv)
    )
    fr_index = _load_fr_index(args.fr_index)
    rule_types = (["proposed", "final"] if args.rule_type == "both"
                  else [args.rule_type])
    targets, pre_results = plan_targets(
        anchor_dockets, fr_index, rule_types, args.output_dir,
        skip_existing=args.skip_existing,
    )

    if args.batch:
        print("[stage2] batch mode: Stage 1b verification submits "
              "candidates through OpenAI's Batch API end-to-end (primary "
              "+ safety-net), ~50% gpt-5 pricing, 24h async per anchor.",
              file=sys.stderr)
    elif args.concurrency > 1:
        # Conservative throughput math: 1,847 tokens/call × concurrency × ~5s
        # round-trip ≈ throughput floor; gpt-5 Tier-4 ceiling is ~800K TPM.
        cpm = args.concurrency * 60 // 5   # rough calls/min if each call ~5s
        print(f"[stage2] concurrent real-time: --concurrency={args.concurrency} "
              f"→ ~{cpm} calls/min throughput target (TPM-limited)",
              file=sys.stderr)
    else:
        print("[stage2] sequential real-time: --concurrency=1 (legacy "
              "loop). Pass --concurrency N for parallel dispatch.",
              file=sys.stderr)

    if args.dry_run:
        return _dry_run_report(targets, pre_results, args)

    results: list[AnchorResult] = list(pre_results)
    cumulative_cost = 0.0
    for target in targets:
        try:
            res = run_one_target(
                target,
                output_dir=args.output_dir, model=args.model, batch=args.batch,
                cumulative_cost_before=cumulative_cost,
                max_cost_usd=args.max_cost,
                concurrency=args.concurrency,
                extract_fn=extract_fn, write_candidates_fn=write_fn,
                verify_fn=verify_fn,
            )
        except CostCapExceeded:
            # Bubble up — remaining anchors will not run.
            raise
        except Exception as e:
            print(f"  [stage2] FAIL {target.docket_id}/{target.rule_type}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results.append(AnchorResult(
                docket_id=target.docket_id, rule_type=target.rule_type,
                status="failed", reason=f"{type(e).__name__}: {e}",
                fr_doc_number=target.fr_doc_number,
            ))
            continue
        cumulative_cost += res.cost_usd
        results.append(res)

    _print_summary(results, cumulative_cost, args.max_cost)
    return results


def _dry_run_report(targets: list[AnchorTarget],
                    pre_results: list[AnchorResult],
                    args) -> list[AnchorResult]:
    """Print the planned invocation list + a cost estimate, return one
    AnchorResult per (docket, rule_type) so tests can assert on the plan."""
    # Rough per-pair anchor for the estimate. Stage 1b cost depends on
    # candidate count, which we don't know until Stage 1a runs. Use the
    # 2018-0775 final benchmark (~38 candidates → ~$0.30 batched real
    # cost per the v2.3 run notes) as a per-anchor anchor. This is a
    # ballpark; treat the printed estimate accordingly.
    PER_TARGET_ESTIMATE_USD = 0.50
    estimated_cost = PER_TARGET_ESTIMATE_USD * len(targets)

    print(f"\n[stage2 dry-run] planned invocations: {len(targets)}", file=sys.stderr)
    print(f"[stage2 dry-run] skipped before invocation: {len(pre_results)}",
          file=sys.stderr)
    print(f"[stage2 dry-run] estimated cost: ${estimated_cost:.2f} "
          f"(anchor ${PER_TARGET_ESTIMATE_USD:.2f}/target — Stage 1b cost "
          "scales with candidate count, which is post-Stage-1a)",
          file=sys.stderr)
    print("\n  planned (would run):", file=sys.stderr)
    for t in targets:
        print(f"    {t.docket_id:<28} {t.rule_type:<8} fr={t.fr_doc_number:<14} "
              f"chars={t.n_chars:>7,}", file=sys.stderr)
    if pre_results:
        print("\n  pre-skipped (would not run):", file=sys.stderr)
        for r in pre_results:
            print(f"    {r.docket_id:<28} {r.rule_type:<8} reason={r.reason}",
                  file=sys.stderr)

    results: list[AnchorResult] = list(pre_results)
    for t in targets:
        results.append(AnchorResult(
            docket_id=t.docket_id, rule_type=t.rule_type, status="dry-run",
            fr_doc_number=t.fr_doc_number,
            reason=f"planned (would extract from {t.fr_text_path})",
        ))
    return results


def _print_summary(results: list[AnchorResult],
                   cumulative_cost: float, max_cost: float) -> None:
    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_fail = sum(1 for r in results if r.status == "failed")
    n_dry = sum(1 for r in results if r.status == "dry-run")

    print("\n[stage2] summary", file=sys.stderr)
    print(f"  targets: {len(results)} "
          f"(ok={n_ok}, skipped={n_skip}, failed={n_fail}, dry-run={n_dry})",
          file=sys.stderr)
    print(f"  total cost: ${cumulative_cost:,.4f} of cap "
          f"${max_cost:,.2f}", file=sys.stderr)
    if n_fail:
        print("  failures:", file=sys.stderr)
        for r in results:
            if r.status == "failed":
                print(f"    {r.docket_id}/{r.rule_type}: {r.reason}",
                      file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", dest="all_anchors", action="store_true",
                   help="Iterate every anchor in anchor_rules_locked.csv.")
    g.add_argument("--docket", type=str,
                   help="Run a single docket (for smoke tests).")
    ap.add_argument("--rule-type", choices=("proposed", "final", "both"),
                    default="both")
    bg = ap.add_mutually_exclusive_group()
    bg.add_argument("--batch", dest="batch", action="store_true", default=True,
                    help="Use batched gpt-5 pricing for cost accounting "
                         "(default). See module docstring for the Stage 1b "
                         "wiring caveat.")
    bg.add_argument("--no-batch", dest="batch", action="store_false",
                    help="Account at standard (real-time) gpt-5 rates.")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="When --no-batch is set, fire up to N parallel "
                         "real-time calls per anchor (asyncio + thread pool). "
                         "Default 20 → ~90K TPM at 1,847 tokens/call (well "
                         "under the gpt-5 Tier-4 800K TPM ceiling). Set to 1 "
                         "for the legacy sequential loop. Ignored when "
                         "--batch is in effect.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip any (docket, rule_type) whose verified CSV "
                         "already exists at the canonical path.")
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST,
                    help="Hard cumulative cost cap in USD (default "
                         f"${DEFAULT_MAX_COST:.0f}; PROJECT_FACTS §8 budget "
                         "is ~$60 batched).")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS_CSV)
    ap.add_argument("--fr-index", type=Path, default=DEFAULT_FR_INDEX)
    ap.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    run_orchestrator(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
