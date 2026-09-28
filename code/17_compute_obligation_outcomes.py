"""
17_compute_obligation_outcomes.py — Path A proposed-to-final outcome matcher.

Per PROJECT_FACTS §12 #5 (7-state taxonomy), each Stage 1b obligation
should be tagged with one of:

  SURVIVED-unchanged    — cosine ≥ 0.95 AND same cfr_section
  SURVIVED-edited       — cosine ≥ 0.85 AND same cfr_section
  MODIFIED              — cosine in [0.55, 0.85) (or ≥ 0.85 with a different
                          cfr_section — the spec's SURVIVED gates on same cfr,
                          so a strong text match in a different section
                          is treated as MODIFIED)
  DROPPED               — cosine < 0.55 OR no final-side obligations exist
                          (DROPPED-explicit vs DROPPED-silent deferred to v2,
                          which needs preamble parsing)
  NEW                   — a final obligation whose max cosine to ANY proposed
                          obligation is < 0.55

Per-docket logic:
  1. Load proposed verified CSV and final verified CSV.
  2. Skip if either is missing or empty (no outcomes computed for that docket).
  3. Embed obligation_text on both sides with sentence-transformers
     (reusing `data/intermediate/stage4_embeddings/<docket>__obligations.parquet`
     when available — these were built during Stage 4 prefilter).
  4. Compute the cosine matrix (proposed × final), pick the best final per
     proposed, classify; pick the worst-best per final to flag NEW.

Downstream: `code/14_stage4_aggregate.py` consumes the output parquet at
`data/processed/path_a_obligation_outcomes.parquet` via the
`--path-a-outcomes` flag.

Output columns:
    obligation_id                  — proposed-side ID (or final-side ID for NEW)
    outcome_state                  — one of the 5 states above
    matched_final_obligation_id    — final-side ID for SURVIVED-*/MODIFIED;
                                     None for DROPPED and NEW
    best_cosine                    — best cosine for the row's perspective
                                     (max over final for proposed rows;
                                      max over proposed for NEW rows)
    docket_id                      — added for traceability

CLI:
    python3 code/17_compute_obligation_outcomes.py --all
    python3 code/17_compute_obligation_outcomes.py --anchor EPA-HQ-OAR-2018-0775
    python3 code/17_compute_obligation_outcomes.py --all --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from stage4_embedding_prefilter import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    EMBED_DIM,
    embed_obligations,
    obligation_id_of,
)


DEFAULT_ANCHORS_CSV = Path("data/processed/anchor_rules_locked.csv")
DEFAULT_OBLIGATIONS_DIR = Path("data/processed")
DEFAULT_OUTPUT = Path("data/processed/path_a_obligation_outcomes.parquet")

# Spec-locked thresholds (PROJECT_FACTS §12 #5).
SURVIVED_UNCHANGED_THRESHOLD = 0.95
SURVIVED_EDITED_THRESHOLD = 0.85
MODIFIED_FLOOR = 0.55
# A final obligation is NEW iff its max cosine to ANY proposed obligation
# is strictly below this floor. Same constant as the proposed-side
# DROPPED floor — symmetric.
NEW_FLOOR = MODIFIED_FLOOR


# ---------------------------------------------------------------------------
# Outcome state values (export-friendly tuple — downstream code can
# import this and validate without re-listing the states).
# ---------------------------------------------------------------------------
OUTCOME_STATES = (
    "SURVIVED-unchanged",
    "SURVIVED-edited",
    "MODIFIED",
    "DROPPED",
    "NEW",
)


@dataclass
class DocketOutcome:
    docket_id: str
    status: str   # 'ok' | 'skipped'
    reason: str = ""
    n_proposed: int = 0
    n_final: int = 0
    n_outcomes: int = 0
    state_counts: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _verified_csv_path(docket_id: str, rule_type: str,
                      obligations_dir: Path = DEFAULT_OBLIGATIONS_DIR
                      ) -> Path:
    return (obligations_dir
            / f"path_a_obligations_verified_{docket_id}_{rule_type}.csv")


def _load_obligations(docket_id: str, rule_type: str,
                      obligations_dir: Path = DEFAULT_OBLIGATIONS_DIR
                      ) -> Optional[list[dict]]:
    """Return list[dict] with obligation_id synthesized, or None if the
    file is missing OR has zero rows."""
    path = _verified_csv_path(docket_id, rule_type, obligations_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    for r in rows:
        r["obligation_id"] = obligation_id_of(r)
    return rows


def _load_anchor_dockets(anchors_csv: Path) -> list[str]:
    with anchors_csv.open("r", encoding="utf-8") as f:
        return [row["docket_id"] for row in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _normalize_cfr(section: object) -> str:
    """Strip + lowercase so '80.1' and ' 80.1 ' compare equal. Used only
    for the SURVIVED-* same-cfr_section gate."""
    if section is None:
        return ""
    return str(section).strip().lower()


def classify_proposed(
    best_cosine: float, same_cfr: bool,
) -> str:
    """Apply the Leahey-style classification to one proposed obligation.
    See module docstring for the exact thresholds and the rationale for
    treating same-text-but-different-cfr matches as MODIFIED."""
    if best_cosine >= SURVIVED_UNCHANGED_THRESHOLD and same_cfr:
        return "SURVIVED-unchanged"
    if best_cosine >= SURVIVED_EDITED_THRESHOLD and same_cfr:
        return "SURVIVED-edited"
    if best_cosine >= MODIFIED_FLOOR:
        return "MODIFIED"
    return "DROPPED"


# ---------------------------------------------------------------------------
# Per-docket matcher
# ---------------------------------------------------------------------------
def compute_outcomes_for_docket(
    docket_id: str,
    *,
    obligations_dir: Path = DEFAULT_OBLIGATIONS_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    embed_fn: Optional[Callable] = None,
    use_cache: bool = True,
) -> tuple[list[dict], DocketOutcome]:
    """Compute outcome rows for one docket. Returns (outcome_rows,
    docket_summary). Rows have shape:
        {obligation_id, outcome_state, matched_final_obligation_id,
         best_cosine, docket_id}

    Args:
        docket_id: the anchor docket.
        obligations_dir: where path_a_obligations_verified_<docket>_<rule>.csv
            live.
        cache_dir: where the Stage 4 prefilter wrote per-docket embedding
            caches. Reused if a `<docket>__obligations.parquet` exists.
        embed_fn: injectable for unit tests; defaults to the sentence-
            transformer wrapped by `stage4_embedding_prefilter._default_embed_fn`.
        use_cache: when False, the cache path is suppressed so the embed
            function always recomputes (used by the
            `test_handles_missing_or_empty_csvs_gracefully` test path).
    """
    import numpy as np

    summary = DocketOutcome(docket_id=docket_id, status="ok")

    proposed_rows = _load_obligations(docket_id, "proposed", obligations_dir)
    final_rows = _load_obligations(docket_id, "final", obligations_dir)

    if proposed_rows is None and final_rows is None:
        summary.status = "skipped"
        summary.reason = "neither proposed nor final verified CSV exists"
        return [], summary
    if proposed_rows is None:
        summary.status = "skipped"
        summary.reason = "proposed verified CSV missing or empty"
        return [], summary
    if final_rows is None:
        summary.status = "skipped"
        summary.reason = "final verified CSV missing or empty"
        return [], summary

    summary.n_proposed = len(proposed_rows)
    summary.n_final = len(final_rows)

    cache_path = (cache_dir / f"{docket_id}__obligations.parquet"
                  if use_cache else None)

    # Embed both sides. `embed_obligations` honors content-hash caching
    # per id, so re-running after Stage 1b updates only re-embeds
    # changed rows.
    emb_p, ids_p = embed_obligations(
        proposed_rows, cache_path=cache_path, embed_fn=embed_fn,
    )
    emb_f, ids_f = embed_obligations(
        final_rows, cache_path=cache_path, embed_fn=embed_fn,
    )

    # Build cfr_section lookup tables for the SURVIVED-* same-cfr gate.
    proposed_cfr = {r["obligation_id"]: _normalize_cfr(r.get("cfr_section"))
                    for r in proposed_rows}
    final_cfr = {r["obligation_id"]: _normalize_cfr(r.get("cfr_section"))
                 for r in final_rows}

    # Cosine matrix. embed_obligations L2-normalizes so cos = dot.
    if emb_p.shape[0] == 0 or emb_f.shape[0] == 0:
        cos = np.zeros((emb_p.shape[0], emb_f.shape[0]), dtype=np.float32)
    else:
        cos = emb_p @ emb_f.T  # (n_p, n_f)

    outcome_rows: list[dict] = []
    state_counts: dict[str, int] = {s: 0 for s in OUTCOME_STATES}

    # --- Proposed-side classification -------------------------------
    if cos.shape[1] == 0:
        # No final obligations at all → every proposed is DROPPED.
        for pid in ids_p:
            outcome_rows.append({
                "obligation_id": pid, "outcome_state": "DROPPED",
                "matched_final_obligation_id": None,
                "best_cosine": 0.0, "docket_id": docket_id,
            })
            state_counts["DROPPED"] += 1
    else:
        best_j = cos.argmax(axis=1)
        best_cos_per_p = cos.max(axis=1)
        for i, pid in enumerate(ids_p):
            j = int(best_j[i])
            best_cos = float(best_cos_per_p[i])
            fid = ids_f[j]
            same_cfr = (proposed_cfr.get(pid, "")
                        == final_cfr.get(fid, "")
                        and proposed_cfr.get(pid, "") != "")
            state = classify_proposed(best_cos, same_cfr)
            outcome_rows.append({
                "obligation_id": pid,
                "outcome_state": state,
                # Only attach a matched_final_obligation_id when the
                # match survived as SURVIVED-* or MODIFIED.
                "matched_final_obligation_id": (
                    fid if state != "DROPPED" else None),
                "best_cosine": best_cos,
                "docket_id": docket_id,
            })
            state_counts[state] += 1

    # --- Final-side NEW detection -----------------------------------
    if cos.shape[0] == 0:
        # No proposed obligations → every final is NEW.
        for fid in ids_f:
            outcome_rows.append({
                "obligation_id": fid, "outcome_state": "NEW",
                "matched_final_obligation_id": None,
                "best_cosine": 0.0, "docket_id": docket_id,
            })
            state_counts["NEW"] += 1
    else:
        best_cos_per_f = cos.max(axis=0)
        for j, fid in enumerate(ids_f):
            if float(best_cos_per_f[j]) < NEW_FLOOR:
                outcome_rows.append({
                    "obligation_id": fid,
                    "outcome_state": "NEW",
                    "matched_final_obligation_id": None,
                    "best_cosine": float(best_cos_per_f[j]),
                    "docket_id": docket_id,
                })
                state_counts["NEW"] += 1

    summary.n_outcomes = len(outcome_rows)
    summary.state_counts = state_counts
    return outcome_rows, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_summary(per_docket: list[DocketOutcome],
                   output_path: Path, dry_run: bool) -> None:
    n_ok = sum(1 for d in per_docket if d.status == "ok")
    n_skip = sum(1 for d in per_docket if d.status == "skipped")
    total_outcomes = sum(d.n_outcomes for d in per_docket)
    rolled: dict[str, int] = {s: 0 for s in OUTCOME_STATES}
    for d in per_docket:
        for s, c in d.state_counts.items():
            rolled[s] = rolled.get(s, 0) + c
    print("\n[outcomes] summary", file=sys.stderr)
    print(f"  dockets: {len(per_docket)} (ok={n_ok}, skipped={n_skip})",
          file=sys.stderr)
    print(f"  total outcome rows: {total_outcomes:,}", file=sys.stderr)
    for s in OUTCOME_STATES:
        print(f"    {s:<22} {rolled.get(s, 0):>6,}", file=sys.stderr)
    if not dry_run:
        print(f"  wrote: {output_path}", file=sys.stderr)
    else:
        print(f"  (dry-run — no parquet written)", file=sys.stderr)
    skipped = [d for d in per_docket if d.status == "skipped"]
    if skipped:
        print("\n  skipped dockets:", file=sys.stderr)
        for d in skipped:
            print(f"    {d.docket_id}: {d.reason}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="Run all 29 anchors from anchor_rules_locked.csv.")
    g.add_argument("--anchor", type=str,
                   help="Run a single docket (for smoke tests).")
    ap.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS_CSV)
    ap.add_argument("--obligations-dir", type=Path,
                    default=DEFAULT_OBLIGATIONS_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="Where Stage 4 prefilter wrote per-docket "
                         "embedding caches. Re-used when available.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute outcomes but don't write parquet.")
    args = ap.parse_args(argv)

    if args.all:
        dockets = _load_anchor_dockets(args.anchors_csv)
        print(f"[outcomes] --all: loaded {len(dockets)} anchors from "
              f"{args.anchors_csv}", file=sys.stderr)
    else:
        dockets = [args.anchor]

    all_rows: list[dict] = []
    summaries: list[DocketOutcome] = []
    for docket_id in dockets:
        print(f"[outcomes] {docket_id}", file=sys.stderr)
        try:
            rows, summary = compute_outcomes_for_docket(
                docket_id,
                obligations_dir=args.obligations_dir,
                cache_dir=args.cache_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [outcomes] FAIL {docket_id}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            summary = DocketOutcome(docket_id=docket_id, status="skipped",
                                    reason=f"{type(e).__name__}: {e}")
            summaries.append(summary)
            continue
        summaries.append(summary)
        if summary.status == "ok":
            all_rows.extend(rows)
            print(f"  [outcomes] {summary.n_outcomes:,} rows "
                  f"(proposed={summary.n_proposed}, final={summary.n_final}); "
                  f"states={summary.state_counts}", file=sys.stderr)
        else:
            print(f"  [outcomes] SKIP — {summary.reason}", file=sys.stderr)

    _print_summary(summaries, args.output, args.dry_run)

    if not args.dry_run and all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
