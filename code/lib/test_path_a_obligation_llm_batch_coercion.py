"""
test_path_a_obligation_llm_batch_coercion.py — coverage for the
batch-path validator coercion patch (commit 0472f21, 2026-05-12).

The patch applies `_coerce_modal` and `_coerce_enum` at the top of
`_validated_verification_dict` and `_validated_safety_net_dict` BEFORE
the strict closed-enum membership check. Documented near-miss aliases
(e.g. "shall not" -> "may not"; "amend" -> "none") now pass through
the batch path the same way they do real-time, and the validator
writes the canonical form back to the dict.

These tests are isolated from the existing
`test_path_a_obligation_llm_batch.py` so the patch can be reverted
without disturbing the pre-patch test corpus.

No real OpenAI calls. Pure validator-level tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from path_a_obligation_llm import (  # noqa: E402
    ALLOWED_AMENDATORY_ACTIONS,
    ALLOWED_MODALS,
    _validated_safety_net_dict,
    _validated_verification_dict,
)


# ---------------------------------------------------------------------------
# Fixtures — minimum-viable parsed dicts that satisfy everything EXCEPT
# the field under test. Tests mutate one field at a time so coverage
# isolates to the patched coercion paths.
# ---------------------------------------------------------------------------
def _base_primary(**overrides) -> dict:
    """Valid primary-verification dict — caller overrides the field
    under test. All non-overridden fields are inside their enums."""
    base = {
        "verified": True,
        "verified_reason": "amendatory block; clear actor + modal.",
        "complexity": "simple",
        "recital_disposition": "none",
        "obligations": [{
            "subject": "operator", "modal": "must",
            "modal_strength": "strong",
            "action": "submit", "object": "report",
            "passive_voice": False, "is_conditional": False,
            "condition_text": None, "cfr_section": "80.1",
            "amendatory_action": "revise", "cross_reference": None,
        }],
    }
    if overrides:
        base["obligations"][0].update(overrides)
    return base


def _base_safety_net(**overrides) -> dict:
    base = {"missed_obligations": [{
        "subject": "operator", "modal": "must",
        "action": "submit", "object": "records",
        "obligation_text": "operator must submit records.",
        "found_via": "passive_voice",
    }]}
    if overrides:
        base["missed_obligations"][0].update(overrides)
    return base


# ===========================================================================
# Tests for _validated_verification_dict — modal coercion
# ===========================================================================
def test_primary_validator_coerces_shall_not_to_may_not():
    """modal='shall not' is the canonical example from the bug commit
    message (commit 0472f21). Per _coerce_modal's alias map at lines
    ~1033, shall-not coerces to may-not (both prohibit semantics)."""
    out = _validated_verification_dict(_base_primary(modal="shall not"))
    assert out["obligations"][0]["modal"] == "may not", (
        f"expected modal coerced to 'may not'; got "
        f"{out['obligations'][0]['modal']!r}")
    # Canonical form is in the enum.
    assert "may not" in ALLOWED_MODALS


def test_primary_validator_coerces_uppercase_modal_alias():
    """_coerce_modal strips + lowercases before lookup. 'SHALL NOT '
    must also coerce to 'may not'."""
    out = _validated_verification_dict(_base_primary(modal=" SHALL NOT "))
    assert out["obligations"][0]["modal"] == "may not"


def test_primary_validator_coerces_required_to_alias():
    """Another documented alias: 'Required to' → 'is required to'."""
    out = _validated_verification_dict(_base_primary(modal="Required to"))
    assert out["obligations"][0]["modal"] == "is required to"
    assert "is required to" in ALLOWED_MODALS


def test_primary_validator_coerces_empty_modal_to_no_modal_implicit():
    """Edge case from audit checklist: empty-string modal coerces to
    'no_modal_implicit' (line 1042 of the alias map) which IS in
    ALLOWED_MODALS, so it passes the strict check after coercion."""
    out = _validated_verification_dict(_base_primary(modal=""))
    assert out["obligations"][0]["modal"] == "no_modal_implicit"
    assert "no_modal_implicit" in ALLOWED_MODALS


def test_primary_validator_already_canonical_modal_passes_unchanged():
    """A modal already in ALLOWED_MODALS should pass through with no
    change (coercion is idempotent on canonical values)."""
    out = _validated_verification_dict(_base_primary(modal="shall"))
    assert out["obligations"][0]["modal"] == "shall"


def test_primary_validator_uncoercible_modal_alias_falls_back_to_default():
    """_coerce_modal returns 'no_modal_implicit' for anything not in
    its alias map. 'wibble' has no entry and is NOT in ALLOWED_MODALS
    on its own, so it should coerce to 'no_modal_implicit' (default)
    rather than raise. This documents that the patch is permissive —
    truly novel surface forms get the fallback, not a raise."""
    out = _validated_verification_dict(_base_primary(modal="wibble"))
    assert out["obligations"][0]["modal"] == "no_modal_implicit"


def test_primary_validator_raises_on_non_string_modal():
    """The patch still raises on type errors. None, integers, lists —
    none have a documented alias path so they can't be coerced."""
    for bad in (None, 123, [], {"a": 1}, True):
        try:
            _validated_verification_dict(_base_primary(modal=bad))
        except ValueError as e:
            assert "modal" in str(e), f"missing 'modal' in error: {e}"
            assert "string" in str(e), (
                f"expected 'must be string' message for {bad!r}; got: {e}")
            continue
        assert False, f"expected ValueError on modal={bad!r}"


# ===========================================================================
# Tests for _validated_verification_dict — amendatory_action coercion
# ===========================================================================
def test_primary_validator_uncoercible_amendatory_action_writes_None_for_fallthrough():
    """amendatory_action='amend' is one of the canonical near-miss examples.
    Followup fix 2026-05-13: when the LLM returns an uncoercible value, the
    validator writes Python None (not the string 'none') so the row
    builder's `ob.get('amendatory_action') or candidate_row.get(...)`
    pattern correctly falls through to the heuristic-derived structural
    classification rather than collapsing 'LLM uncertain' into 'explicitly
    not in amendatory block'."""
    out = _validated_verification_dict(
        _base_primary(amendatory_action="amend"))
    coerced = out["obligations"][0]["amendatory_action"]
    assert coerced is None, (
        f"expected 'amend' to coerce to Python None (for row-builder "
        f"fallthrough); got {coerced!r}")


def test_primary_validator_uncoercible_amendatory_action_does_not_raise():
    """The validator should accept (not raise on) uncoercible LLM values
    for amendatory_action — uncoercibility is signaled by writing None,
    which the row builder handles via fallthrough."""
    # If this raises, the test fails by exception.
    out = _validated_verification_dict(
        _base_primary(amendatory_action="amend"))
    assert out["obligations"][0]["amendatory_action"] is None
    out2 = _validated_verification_dict(
        _base_primary(amendatory_action="modify"))
    assert out2["obligations"][0]["amendatory_action"] is None


def test_primary_validator_coerces_uppercase_amendatory_action():
    """_coerce_enum strips + lowercases before lookup. 'REVISE ' must
    pass through as 'revise'."""
    out = _validated_verification_dict(
        _base_primary(amendatory_action=" REVISE "))
    assert out["obligations"][0]["amendatory_action"] == "revise"


def test_primary_validator_null_amendatory_action_passes():
    """amendatory_action may be null/missing; row builder falls back
    to the candidate's heuristic-derived field. Validator must not
    raise on null."""
    obs = _base_primary()["obligations"][0]
    obs["amendatory_action"] = None
    parsed = {"verified": True, "verified_reason": "x",
              "complexity": "simple", "recital_disposition": "none",
              "obligations": [obs]}
    out = _validated_verification_dict(parsed)
    # Validator preserved the None — row builder will resolve at
    # write-time using the candidate row.
    assert out["obligations"][0]["amendatory_action"] is None


def test_primary_validator_raises_on_non_string_amendatory_action():
    for bad in (123, [], True, {"a": 1}):
        try:
            _validated_verification_dict(
                _base_primary(amendatory_action=bad))
        except ValueError as e:
            assert "amendatory_action" in str(e)
            assert "string" in str(e)
            continue
        assert False, f"expected ValueError on amendatory_action={bad!r}"


# ===========================================================================
# Tests for _validated_verification_dict — UNTOUCHED fields still strict
# ===========================================================================
def test_primary_validator_still_raises_on_invalid_modal_strength():
    """modal_strength has no documented alias map, so the patch leaves
    it strict. Any out-of-enum value must still raise → routed to recovery."""
    try:
        _validated_verification_dict(
            _base_primary(modal_strength="INVALID"))
    except ValueError as e:
        assert "modal_strength" in str(e)
        return
    assert False, "expected ValueError on invalid modal_strength"


def test_primary_validator_still_raises_on_missing_modal_strength():
    """The patch only added coercion for modal and amendatory_action.
    Missing modal_strength is still a hard schema violation."""
    obs = _base_primary()["obligations"][0]
    del obs["modal_strength"]
    parsed = {"verified": True, "verified_reason": "x",
              "complexity": "simple", "recital_disposition": "none",
              "obligations": [obs]}
    try:
        _validated_verification_dict(parsed)
    except ValueError as e:
        assert "modal_strength" in str(e)
        return
    assert False, "expected ValueError on missing modal_strength"


def test_primary_validator_still_raises_on_invalid_complexity():
    """complexity has no alias map either; out-of-enum still raises."""
    parsed = _base_primary()
    parsed["complexity"] = "INVALID"
    try:
        _validated_verification_dict(parsed)
    except ValueError as e:
        assert "complexity" in str(e)
        return
    assert False, "expected ValueError on invalid complexity"


def test_primary_validator_still_raises_on_non_bool_verified():
    parsed = _base_primary()
    parsed["verified"] = "true"   # string, not bool
    try:
        _validated_verification_dict(parsed)
    except ValueError as e:
        assert "verified" in str(e)
        return
    assert False, "expected ValueError on non-bool verified"


# ===========================================================================
# Tests for _validated_safety_net_dict — mirror modal coercion
# ===========================================================================
def test_safety_net_validator_coerces_shall_not_to_may_not():
    """Same modal-alias logic must apply on the safety-net side."""
    out = _validated_safety_net_dict(_base_safety_net(modal="shall not"))
    assert out["missed_obligations"][0]["modal"] == "may not"


def test_safety_net_validator_coerces_uppercase_modal_alias():
    out = _validated_safety_net_dict(_base_safety_net(modal=" SHALL NOT "))
    assert out["missed_obligations"][0]["modal"] == "may not"


def test_safety_net_validator_already_canonical_passes_unchanged():
    out = _validated_safety_net_dict(_base_safety_net(modal="must"))
    assert out["missed_obligations"][0]["modal"] == "must"


def test_safety_net_validator_raises_on_non_string_modal():
    for bad in (None, 123, [], True):
        try:
            _validated_safety_net_dict(_base_safety_net(modal=bad))
        except ValueError as e:
            assert "modal" in str(e)
            assert "string" in str(e)
            continue
        assert False, f"expected ValueError on safety-net modal={bad!r}"


def test_safety_net_validator_empty_modal_coerces_to_no_modal_implicit():
    out = _validated_safety_net_dict(_base_safety_net(modal=""))
    assert out["missed_obligations"][0]["modal"] == "no_modal_implicit"


# ===========================================================================
# Cross-cutting: canonical form is written back to the dict
# ===========================================================================
def test_canonical_form_written_back_to_dict_on_both_validators():
    """Both validators must mutate (or write back) the canonical form
    so downstream consumers (row builder, CSV writer) see the same
    canonical value the real-time path emits.

    Modal: 'shall not' → 'may not' (per _coerce_modal alias map).
    Amendatory_action: 'amend' → None (per followup fix 2026-05-13;
    uncoercible LLM values are written as Python None so the row
    builder's `or candidate_row.get(...)` falls through to the
    heuristic-derived structural classification)."""
    p = _base_primary(modal="shall not", amendatory_action="amend")
    out_p = _validated_verification_dict(p)
    assert out_p["obligations"][0]["modal"] == "may not"
    assert out_p["obligations"][0]["amendatory_action"] is None

    s = _base_safety_net(modal="Required to")
    out_s = _validated_safety_net_dict(s)
    assert out_s["missed_obligations"][0]["modal"] == "is required to"


# ---------------------------------------------------------------------------
# Standalone runner (same style as code/lib/test_path_a_dedup.py)
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
