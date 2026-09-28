"""
test_path_a_dedup.py — unit tests for the Stage 1b dedup pass.

The headline test runs dedup against a frozen snapshot of the v2.3
verified CSV for the Renewable Fuel Standard / E15 docket
(EPA-HQ-OAR-2018-0775) and pins the exact set of safety-net candidate_idx
values that should drop vs. those that should be kept. The expected
sets were eyeballed by Bruce on 2026-05-09 against the v2.3 output
(88 verified rows pre-dedup).

The fixture is captured at code/lib/test_fixtures/ so this test stays
runnable after the production CSV gets overwritten with the dedup result.

Run:
    python -m pytest code/lib/test_path_a_dedup.py -v
or:
    python code/lib/test_path_a_dedup.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Locate repo root from this file: .../code/lib/test_path_a_dedup.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from path_a_dedup import (  # noqa: E402
    OVERLAP_THRESHOLD,
    _build_norm_index,
    _find_offsets_in_doc,
    _overlap_fraction,
    build_candidates_offsets,
    dedup_verified_rows,
    normalize_cfr_section,
)


# Frozen pre-dedup snapshot — reading the live verified CSV would break
# this test the moment the dedup overwrites it for production use.
VERIFIED_CSV = (_REPO_ROOT
                / "code/lib/test_fixtures"
                / "path_a_v23_verified_EPA-HQ-OAR-2018-0775_final.csv")
CANDIDATES_CSV = (_REPO_ROOT
                  / "data/processed"
                  / "path_a_obligations_candidates_EPA-HQ-OAR-2018-0775_final.csv")
FR_DOC = (_REPO_ROOT
          / "data/raw/federal_register"
          / "EPA-HQ-OAR-2018-0775_final_2019-11653.txt")


# Expected behavior eyeballed by Bruce on 2026-05-09 against v2.3 verified output.
EXPECTED_DROP_IDXS = {
    100100, 100101, 100102, 100103,   # § 80.27(d)(2) — duplicates of primaries 1-4
    100400,                            # § 80.1402(a) — duplicate of primary 8
    100500, 100501, 100502,            # § 80.1435 — duplicates of primaries 11, 15, 16
    100600, 100601, 100602, 100603,   # § 80.1451(c)(2) — duplicates of primaries 23-26
    100800, 100801, 100802, 100803,   # § 80.1454 — duplicates of primaries 30, 31, 33, 34
    100804, 100805, 100806,            # § 80.1454 — duplicates of primaries 35, 36, 37
}
EXPECTED_KEEP_IDXS = {
    # § 80.1451(c)(2) subsection-level data fields not in primary
    100604, 100605, 100606, 100607, 100608,
    # § 80.1464 attest-engagement obligations — primary missed § 80.1464 entirely
    100900, 100901, 100902, 100903, 100904,
    # § 80.1503 subsection-level rows
    101000, 101001,
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_normalize_cfr_section():
    assert normalize_cfr_section("80.1435(b)(1)") == "80.1435"
    assert normalize_cfr_section("80.27") == "80.27"
    assert normalize_cfr_section("80.27(d)(2)") == "80.27"
    assert normalize_cfr_section(None) == ""
    assert normalize_cfr_section("") == ""
    assert normalize_cfr_section(" 80.1503  ") == "80.1503"


def test_overlap_fraction():
    # Identical regions overlap fully
    assert _overlap_fraction(100, 200, 100, 200) == 1.0
    # Disjoint
    assert _overlap_fraction(100, 200, 200, 300) == 0.0
    assert _overlap_fraction(100, 200, 300, 400) == 0.0
    # Smaller fully contained in larger -> shorter denominator -> 1.0
    assert _overlap_fraction(105, 195, 100, 200) == 1.0
    # Partial overlap, ~50% of shorter
    assert abs(_overlap_fraction(100, 200, 150, 250) - 0.5) < 1e-9


def test_norm_index_strips_page_breaks():
    doc = "Foo bar\n\n[[Page 27024]]\n\nbaz."
    norm, idx = _build_norm_index(doc)
    # Page-break marker collapses into surrounding whitespace
    assert "[[Page" not in norm
    # All collapsed-positions still map back into the original
    assert max(idx) < len(doc)


def test_find_offsets_locates_paraphrased_safety_net_rows():
    """The two paraphrased rows from the v2.3 output (page-break and
    LLM-added paragraph label) must locate via fallback strategies."""
    fr_text = FR_DOC.read_text(encoding="utf-8", errors="ignore")
    doc_norm, idx = _build_norm_index(fr_text)

    # 100601: full text contains "RINs separated from a renewable fuel volume",
    # but the FR doc has a [[Page 27024]] page break in the middle of that span.
    needle1 = ("Each report must summarize RIN activities for the reporting "
               "period, separately for RINs separated from a renewable fuel "
               "volume and RINs assigned to a renewable fuel volume.")
    res = _find_offsets_in_doc(needle1, fr_text, doc_norm, idx)
    assert res is not None, "page-break fallback failed for 100601-like text"

    # 100605: LLM emitted "(E)(2) Indicate ..." but the FR doc only has
    # "(2) Indicate ..." (the (E) parent is implicit).
    needle2 = ("(E)(2) Indicate if the submitting party or the submitting "
               "party's corporate affiliate group exceeded the secondary "
               "threshold for any day in the quarter under Sec. 80.1435(c)(2).")
    res = _find_offsets_in_doc(needle2, fr_text, doc_norm, idx)
    assert res is not None, "leading-label fallback failed for 100605-like text"


# ---------------------------------------------------------------------------
# End-to-end dedup against the v2.3 verified CSV
# ---------------------------------------------------------------------------
def _read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_dedup_v23_drops_and_keeps_match_eyeball():
    assert VERIFIED_CSV.exists(), f"missing v2.3 verified CSV: {VERIFIED_CSV}"
    assert CANDIDATES_CSV.exists(), f"missing candidates CSV: {CANDIDATES_CSV}"
    assert FR_DOC.exists(), f"missing FR doc: {FR_DOC}"

    rows = _read_csv_dicts(VERIFIED_CSV)
    candidates = _read_csv_dicts(CANDIDATES_CSV)
    candidates_offsets = build_candidates_offsets(candidates)
    fr_text = FR_DOC.read_text(encoding="utf-8", errors="ignore")

    kept, summary = dedup_verified_rows(rows, candidates_offsets, fr_text)

    # v2.3 input shape: 88 rows total = 46 primary + 42 safety_net_verified
    assert summary.rows_in == 88, f"unexpected input row count: {summary.rows_in}"
    assert summary.primary_in == 46, f"unexpected primary count: {summary.primary_in}"
    assert summary.safety_net_in == 42, (
        f"unexpected safety-net count: {summary.safety_net_in}")

    # Every safety-net row must locate in the FR doc; if not, we can't trust
    # the overlap-based dedup result.
    assert summary.unmatched_safety_net == 0, (
        f"{summary.unmatched_safety_net} safety-net row(s) could not be located "
        "in the FR doc — fallback strategies need to be extended.")

    got_drop = set(summary.dropped_candidate_idxs)
    got_keep = set(summary.kept_safety_net_idxs)

    missing_drop = EXPECTED_DROP_IDXS - got_drop
    extra_drop = got_drop - EXPECTED_DROP_IDXS
    missing_keep = EXPECTED_KEEP_IDXS - got_keep
    extra_keep = got_keep - EXPECTED_KEEP_IDXS
    assert not missing_drop, f"safety-net rows that should drop but were kept: {sorted(missing_drop)}"
    assert not extra_drop, f"safety-net rows that should be kept but were dropped: {sorted(extra_drop)}"
    assert not missing_keep, f"safety-net rows that should be kept but were dropped: {sorted(missing_keep)}"
    assert not extra_keep, f"unexpected safety-net rows in keep set: {sorted(extra_keep)}"

    # Output shape: 46 primary + 20 kept safety-net = 66 rows
    assert summary.rows_out == 66, f"unexpected output row count: {summary.rows_out}"
    assert len(kept) == 66

    # All-primary rows must be preserved verbatim.
    primary_in_input = [r for r in rows if r.get("found_via") == "primary"]
    primary_in_output = [r for r in kept if r.get("found_via") == "primary"]
    assert len(primary_in_input) == len(primary_in_output)


def test_dedup_threshold_value_is_pinned():
    # Pin the threshold so any later tweak is intentional + reviewed.
    assert OVERLAP_THRESHOLD == 0.80


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all() -> int:
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failures.append(name)
    if failures:
        print(f"\nFAIL: {len(failures)} test(s) failed: {failures}")
        return 1
    print("\nPASS: all tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
