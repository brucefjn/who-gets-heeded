"""
test_submitter_classification.py — unit tests for the shared Title-field
classifier. Pure-function tests; no pandas DataFrame fixtures needed
because `classify_title` operates on individual strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from submitter_classification import (  # noqa: E402
    SUBMITTER_KIND_VALUES,
    classify_title,
)


# ---------------------------------------------------------------------------
# Required test cases (from spec Task C, File 4)
# ---------------------------------------------------------------------------
def test_individual_name_only():
    # _RE_INDIV_NAME match, no org keyword in title
    assert classify_title("Comment submitted by Jane Doe") == "individual"


def test_individual_pattern_overridden_by_org_role_keyword():
    # The calibrated case in the original code's inline comment: a title
    # that starts with the individual pattern but also names a role/org
    # tips back to organizational.
    title = "Comment submitted by Jane Roe, Executive Director, WMA"
    assert classify_title(title) == "organizational"


def test_anonymous_public_comment_is_individual():
    assert classify_title("Anonymous Public Comment") == "individual"


def test_comments_of_org_is_organizational():
    assert classify_title("Comments of the Sierra Club") == "organizational"


def test_comments_from_org_is_organizational():
    # The 'from' branch of _RE_ORG_OF (alternation with 'of').
    assert classify_title(
        "Comments from the National Mining Association") == "organizational"


def test_joint_comments_multi_org():
    # Doesn't start with "Comment(s) of/from" so _RE_ORG_OF misses; org
    # keyword 'institute' in _RE_TITLE_HAS_ORG carries the classification.
    title = ("Joint Comments of the American Petroleum Institute and the "
             "National Petrochemical Refiners Association")
    assert classify_title(title) == "organizational"


def test_bare_org_name_via_title_has_org():
    # No leading "Comments of/from" — falls through to _RE_TITLE_HAS_ORG.
    assert classify_title("Sierra Club") == "organizational"


def test_empty_string_is_other():
    assert classify_title("") == "other"


def test_none_is_other():
    # pandas NaN may arrive as None depending on the upstream path —
    # function must be None-safe.
    assert classify_title(None) == "other"


def test_pandas_nan_is_other():
    # And as float('nan') on the other path. `pd.isna(float('nan'))` is True.
    assert classify_title(float("nan")) == "other"


def test_whitespace_only_is_other():
    # Strip + no pattern match → "other".
    assert classify_title("   ") == "other"


def test_no_signal_falls_through_to_other():
    assert classify_title("Some random title with no signal") == "other"


# ---------------------------------------------------------------------------
# Defensive test on the public constants tuple
# ---------------------------------------------------------------------------
def test_submitter_kind_values_is_canonical_tuple():
    assert isinstance(SUBMITTER_KIND_VALUES, tuple), (
        "SUBMITTER_KIND_VALUES must be a tuple to preserve canonical "
        "ordering and prevent accidental mutation.")
    assert SUBMITTER_KIND_VALUES == ("organizational", "individual", "other")


# ---------------------------------------------------------------------------
# Standalone runner (matches code/lib/test_path_a_dedup.py style)
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
        print(f"\nFAIL: {len(failures)} test(s): {failures}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
