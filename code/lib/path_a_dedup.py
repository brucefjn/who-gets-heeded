"""
path_a_dedup.py — Stage 1b dedup pass (Path A).

The Stage 1b pipeline emits two kinds of verified rows:
  - found_via='primary'              — verified Stage 1a candidates
  - found_via='safety_net_verified'  — section-level safety-net hits that
                                       were re-verified through the primary
                                       prompt.

The safety net deliberately re-scans each amendatory block, so it
sometimes re-surfaces sentences the primary pass already extracted.
This module drops those duplicates while preserving safety-net rows that
add subsection-level coverage the primary pass missed.

Heuristic: a safety-net row is a duplicate of a primary row if their
character-offset regions in the source FR document overlap by more than
OVERLAP_THRESHOLD of the SHORTER region's length, in the SAME normalized
CFR section. Primary offsets come from the Stage 1a candidates CSV;
safety-net offsets are recomputed by string-matching obligation_text
against the FR document (with whitespace normalization, and a 100-char
prefix fallback for rows where multi-line CSV escaping breaks the full
match).

Public entry points:
    dedup_verified_rows(verified_rows, candidates_offsets, fr_doc_text)
        Pure function on dict rows; returns (kept_rows, summary).
    build_candidates_offsets(candidates_rows)
        Convenience: build the offsets dict from already-loaded candidate
        dicts (so callers don't need to re-read the candidates CSV).
    dedup_verified_csv(verified_csv, candidates_csv, fr_doc_path, ...)
        Read-dedup-write convenience for one-off CLI use.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Tuned 2026-05-09 against the EPA-HQ-OAR-2018-0775 v2.3 verified output.
# Using the SHORTER region's length as the denominator means a small
# safety-net span fully contained in a larger primary span is a 100%
# duplicate, which is the behavior we want.
OVERLAP_THRESHOLD = 0.80

# Multi-line obligation_text values lose their literal newlines through
# csv-escaping round-trips. When the full string fails to locate in the
# FR doc, retry with the first PREFIX_FALLBACK_CHARS chars, then with
# SHORT_PREFIX_FALLBACK_CHARS — the shorter fallback handles cases where
# a Federal Register page-break marker (`[[Page NNNNN]]`) sits inside the
# 100-char window of the LLM-paraphrased obligation text.
PREFIX_FALLBACK_CHARS = 100
SHORT_PREFIX_FALLBACK_CHARS = 60


# ---------------------------------------------------------------------------
# CFR-section normalization
# ---------------------------------------------------------------------------
_CFR_BASE_RE = re.compile(r"^(\d+\.\d+[A-Za-z]?)")


def normalize_cfr_section(section: Optional[str]) -> str:
    """Reduce a CFR section string to its base 'part.section' form so
    that 'X.Y' and 'X.Y(z)(w)' compare equal.

        '80.1435(b)(1)' -> '80.1435'
        '80.27'         -> '80.27'
        ''              -> ''
    """
    if not section:
        return ""
    s = str(section).strip()
    m = _CFR_BASE_RE.match(s)
    return m.group(1) if m else s


# ---------------------------------------------------------------------------
# Whitespace-tolerant string match against the FR document
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_PAGE_BREAK_RE = re.compile(r"\[\[Page \d+\]\]")
# Leading paragraph labels the LLM sometimes synthesizes when extracting
# subsection-level obligations (e.g., it may emit "(E)(2) Indicate ..."
# even though the FR doc only has "(2) Indicate ...", because the (E)
# parent label is implicit from the surrounding context). When the
# normalized full-text match fails, we strip up to a couple of these and
# retry — purely a fallback, never the primary match path.
_LEADING_LABEL_RE = re.compile(r"^\s*(?:\([A-Za-z0-9]+\)\s*){1,3}")


def _normalize_for_match(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _strip_leading_labels(text: str) -> str:
    return _LEADING_LABEL_RE.sub("", text, count=1)


def _build_norm_index(doc_text: str) -> tuple[str, list[int]]:
    """Build a whitespace-collapsed view of doc_text and a parallel index
    that maps each character of the collapsed view back to its position
    in the original. Whitespace runs collapse to a single space whose
    mapped index is the FIRST whitespace char of the run.

    Federal Register `[[Page NNNNN]]` markers are folded into the
    surrounding whitespace so they don't break LLM-paraphrased matches.
    """
    out_chars: list[str] = []
    out_orig: list[int] = []
    in_ws = False
    i = 0
    n = len(doc_text)
    while i < n:
        m = _PAGE_BREAK_RE.match(doc_text, i)
        if m:
            if not in_ws:
                out_chars.append(" ")
                out_orig.append(i)
                in_ws = True
            i = m.end()
            continue
        ch = doc_text[i]
        if ch.isspace():
            if not in_ws:
                out_chars.append(" ")
                out_orig.append(i)
                in_ws = True
        else:
            out_chars.append(ch)
            out_orig.append(i)
            in_ws = False
        i += 1
    return "".join(out_chars), out_orig


def _try_norm_match(
    needle: str,
    doc_norm: str,
    doc_norm_to_orig: list[int],
) -> Optional[tuple[int, int]]:
    if not needle:
        return None
    nidx = doc_norm.find(needle)
    if nidx >= 0 and (nidx + len(needle) - 1) < len(doc_norm_to_orig):
        return (
            doc_norm_to_orig[nidx],
            doc_norm_to_orig[nidx + len(needle) - 1] + 1,
        )
    return None


def _find_offsets_in_doc(
    obligation_text: str,
    doc_text: str,
    doc_norm: str,
    doc_norm_to_orig: list[int],
) -> Optional[tuple[int, int]]:
    """Locate obligation_text inside doc_text. Returns (start, end) in
    the ORIGINAL doc_text, or None if no match is found.

    Match strategy (each step uses the original doc_text for direct
    matches and the page-break-stripped, whitespace-collapsed doc_norm
    for normalized matches):
      1. Direct substring match.
      2. Whitespace-collapsed match on the full obligation_text.
      3. Same as (1)/(2) on the first PREFIX_FALLBACK_CHARS chars
         (handles multi-line CSV escaping that mangles the tail).
      4. Same on the first SHORT_PREFIX_FALLBACK_CHARS chars
         (handles a page-break marker landing inside the 100-char
         window of an LLM-paraphrased obligation).
      5. Same on the obligation_text with leading paragraph labels
         stripped (handles cases where the LLM synthesizes a parent
         label like "(E)(2)" that the FR doc itself doesn't carry).
    """
    if not obligation_text:
        return None

    idx = doc_text.find(obligation_text)
    if idx >= 0:
        return idx, idx + len(obligation_text)

    res = _try_norm_match(
        _normalize_for_match(obligation_text), doc_norm, doc_norm_to_orig,
    )
    if res:
        return res

    for prefix_len in (PREFIX_FALLBACK_CHARS, SHORT_PREFIX_FALLBACK_CHARS):
        prefix = obligation_text[:prefix_len]
        if not prefix or prefix == obligation_text:
            continue
        idx = doc_text.find(prefix)
        if idx >= 0:
            return idx, idx + len(prefix)
        res = _try_norm_match(
            _normalize_for_match(prefix), doc_norm, doc_norm_to_orig,
        )
        if res:
            return res

    stripped = _strip_leading_labels(obligation_text)
    if stripped and stripped != obligation_text:
        res = _try_norm_match(
            _normalize_for_match(stripped), doc_norm, doc_norm_to_orig,
        )
        if res:
            return res
        short = stripped[:SHORT_PREFIX_FALLBACK_CHARS]
        if short and short != stripped:
            res = _try_norm_match(
                _normalize_for_match(short), doc_norm, doc_norm_to_orig,
            )
            if res:
                return res
    return None


# ---------------------------------------------------------------------------
# Overlap math
# ---------------------------------------------------------------------------
def _overlap_fraction(
    a_start: int, a_end: int, b_start: int, b_end: int,
) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_end <= inter_start:
        return 0.0
    inter_len = inter_end - inter_start
    a_len = max(1, a_end - a_start)
    b_len = max(1, b_end - b_start)
    return inter_len / min(a_len, b_len)


# ---------------------------------------------------------------------------
# Candidates-offsets index
# ---------------------------------------------------------------------------
def build_candidates_offsets(
    candidate_rows: list[dict],
) -> dict[int, tuple[str, int, int]]:
    """Build {obligation_idx: (normalized_cfr_section, start, end)} from
    Stage 1a candidate dicts (as read from the candidates CSV)."""
    out: dict[int, tuple[str, int, int]] = {}
    for r in candidate_rows:
        try:
            idx = int(r["obligation_idx"])
            start = int(r["char_offset_start"])
            end = int(r["char_offset_end"])
        except (KeyError, ValueError, TypeError):
            continue
        cfr_norm = normalize_cfr_section(r.get("cfr_section"))
        out[idx] = (cfr_norm, start, end)
    return out


def _load_candidates_offsets_from_csv(
    candidates_csv: Path,
) -> dict[int, tuple[str, int, int]]:
    with candidates_csv.open("r", encoding="utf-8") as f:
        return build_candidates_offsets(list(csv.DictReader(f)))


# ---------------------------------------------------------------------------
# Core dedup
# ---------------------------------------------------------------------------
@dataclass
class DedupResult:
    rows_in: int = 0
    rows_out: int = 0
    primary_in: int = 0
    safety_net_in: int = 0
    safety_net_dropped: int = 0
    safety_net_kept: int = 0
    unmatched_safety_net: int = 0   # text could not be located in FR doc
    dropped_candidate_idxs: list[int] = field(default_factory=list)
    kept_safety_net_idxs: list[int] = field(default_factory=list)


def dedup_verified_rows(
    verified_rows: list[dict],
    candidates_offsets: dict[int, tuple[str, int, int]],
    fr_doc_text: str,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> tuple[list[dict], DedupResult]:
    """Drop safety-net rows whose offsets overlap > overlap_threshold
    with any primary row in the same (normalized) CFR section.

    Args:
        verified_rows: dict rows from the Stage 1b verified CSV. Each row
            must have at least: candidate_idx, found_via, cfr_section,
            obligation_text. Dataclass instances can be passed via
            asdict() upstream.
        candidates_offsets: output of build_candidates_offsets().
        fr_doc_text: the source FR document text used by Stage 1a/1b.
        overlap_threshold: drop a safety-net row if its overlap fraction
            with a same-section primary row exceeds this value.

    Returns:
        (kept_rows, summary). kept_rows preserves input order; summary
        records counts plus the candidate_idx values dropped vs kept on
        the safety-net side.

    Behavior on unlocatable safety-net rows: keep them (preferring under-
    dedup over false-positive drops) and increment unmatched_safety_net.
    """
    doc_norm, doc_norm_to_orig = _build_norm_index(fr_doc_text)

    # Group primary regions by normalized cfr_section. Use a set keyed by
    # (start, end) so compound splits don't double-count the same region.
    primary_by_section: dict[str, set[tuple[int, int]]] = {}
    primary_count = 0
    for r in verified_rows:
        if r.get("found_via") != "primary":
            continue
        primary_count += 1
        try:
            cidx = int(r["candidate_idx"])
        except (KeyError, ValueError, TypeError):
            continue
        offsets = candidates_offsets.get(cidx)
        if offsets is None:
            continue
        cfr_norm, s, e = offsets
        primary_by_section.setdefault(cfr_norm, set()).add((s, e))

    kept: list[dict] = []
    summary = DedupResult(rows_in=len(verified_rows), primary_in=primary_count)

    for r in verified_rows:
        if r.get("found_via") != "safety_net_verified":
            kept.append(r)
            continue
        summary.safety_net_in += 1
        cfr_norm = normalize_cfr_section(r.get("cfr_section"))
        sn_offsets = _find_offsets_in_doc(
            r.get("obligation_text", "") or "",
            fr_doc_text, doc_norm, doc_norm_to_orig,
        )
        try:
            cidx_int = int(r["candidate_idx"])
        except (KeyError, ValueError, TypeError):
            cidx_int = -1

        if sn_offsets is None:
            summary.unmatched_safety_net += 1
            kept.append(r)
            if cidx_int >= 0:
                summary.kept_safety_net_idxs.append(cidx_int)
            continue

        sn_s, sn_e = sn_offsets
        is_dup = False
        for ps, pe in primary_by_section.get(cfr_norm, ()):
            if _overlap_fraction(sn_s, sn_e, ps, pe) > overlap_threshold:
                is_dup = True
                break

        if is_dup:
            summary.safety_net_dropped += 1
            if cidx_int >= 0:
                summary.dropped_candidate_idxs.append(cidx_int)
        else:
            kept.append(r)
            if cidx_int >= 0:
                summary.kept_safety_net_idxs.append(cidx_int)

    summary.rows_out = len(kept)
    summary.safety_net_kept = summary.safety_net_in - summary.safety_net_dropped
    summary.dropped_candidate_idxs = sorted(set(summary.dropped_candidate_idxs))
    summary.kept_safety_net_idxs = sorted(set(summary.kept_safety_net_idxs))
    return kept, summary


# ---------------------------------------------------------------------------
# CLI / one-shot CSV utility
# ---------------------------------------------------------------------------
def dedup_verified_csv(
    verified_csv: Path,
    candidates_csv: Path,
    fr_doc_path: Path,
    out_csv: Optional[Path] = None,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> DedupResult:
    """Read verified_csv, dedup, write back (or to out_csv). Returns the
    DedupResult summary."""
    with verified_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    candidates_offsets = _load_candidates_offsets_from_csv(candidates_csv)
    fr_text = fr_doc_path.read_text(encoding="utf-8", errors="ignore")

    kept, summary = dedup_verified_rows(
        rows, candidates_offsets, fr_text, overlap_threshold=overlap_threshold,
    )

    out_path = out_csv if out_csv is not None else verified_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in kept:
            writer.writerow(r)
    return summary


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verified", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--fr-doc", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path. Default: overwrite --verified.")
    ap.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD)
    args = ap.parse_args()

    summary = dedup_verified_csv(
        args.verified, args.candidates, args.fr_doc,
        out_csv=args.out, overlap_threshold=args.threshold,
    )
    print(f"[path_a_dedup] rows: {summary.rows_in} -> {summary.rows_out}")
    print(f"  primary={summary.primary_in}  safety_net_in={summary.safety_net_in}  "
          f"dropped={summary.safety_net_dropped}  kept={summary.safety_net_kept}  "
          f"unmatched={summary.unmatched_safety_net}")
    print(f"  dropped candidate_idxs: {summary.dropped_candidate_idxs}")
    print(f"  kept safety-net candidate_idxs: {summary.kept_safety_net_idxs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
