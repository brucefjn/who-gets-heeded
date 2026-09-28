"""
test_layer_c_judge.py — unit tests for the Layer C LLM-as-judge.

No real OpenAI calls. Batch and real-time paths are exercised by
injecting custom `call_batch` / `call_json` / `parse_json` callables via
`run_layer_c(...)`'s keyword arguments, mirroring the Stage 4 + Stage 1b
test pattern. CLI-integration tests are intentionally out of scope per
Task F's "Skip orchestrator integration tests".
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from layer_c_judge import (  # noqa: E402
    LAYER_C_BINARIES,
    LAYER_C_ESTIMATED_PER_CALL_USD,
    TRUNCATION_HEAD_CHARS,
    TRUNCATION_MARKER,
    TRUNCATION_TAIL_CHARS,
    TRUNCATION_TRIGGER_CHARS,
    _validated_layer_c_dict,
    estimate_cost,
    run_layer_c,
    truncate_for_layer_c,
)
from path_a_obligation_llm import CostCapExceeded  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _all_zero_payload() -> dict:
    """Valid 23-binary dict, all 0."""
    return {b: 0 for b in LAYER_C_BINARIES}


def _support_payload() -> dict:
    """Valid 23-binary dict — explicit support + targeted-provision +
    technical frame fires; everything else 0."""
    payload = _all_zero_payload()
    payload["expresses_explicit_support"] = 1
    payload["names_targeted_provision"] = 1
    payload["technical_scientific_frame_present"] = 1
    return payload


def _make_comments_df(n: int = 3, *, long_comment_at: int = -1):
    """Build a synthetic comments DataFrame. `long_comment_at` is the
    0-indexed slot to populate with a >TRUNCATION_TRIGGER_CHARS text;
    -1 means no long comments."""
    import pandas as pd
    rows = []
    for i in range(n):
        if i == long_comment_at:
            text = "A" * (TRUNCATION_HEAD_CHARS + 100) \
                   + "M" * 500 + "Z" * (TRUNCATION_TAIL_CHARS + 100)
        else:
            text = f"Synthetic short comment number {i}."
        rows.append({
            "document_id": f"EPA-COMMENT-{i:04d}",
            "docket_id": "EPA-TEST-DOCKET",
            "comment": text,
        })
    return pd.DataFrame(rows)


def _mk_call_batch(per_pair_payloads: list[dict | None]):
    """Build a fake `call_batch` that returns the supplied per-pair
    payloads as the `successful` list. Each non-None payload is wrapped
    with `via='batch'` + token counts as `_call_openai_batch` would."""
    captured = {"calls": 0, "validate_fn": None,
                "max_cost_usd": None, "n_pairs": None}

    def fake(messages_list, model, max_cost_usd, *, batch_id_prefix,
             max_tokens, validate_fn=None, estimated_per_call_usd=None):
        captured["calls"] += 1
        captured["validate_fn"] = validate_fn
        captured["max_cost_usd"] = max_cost_usd
        captured["n_pairs"] = len(messages_list)
        results: list[dict | None] = []
        for i, payload in enumerate(per_pair_payloads[:len(messages_list)]):
            if payload is None:
                results.append(None)
                continue
            r = dict(payload)
            r.setdefault("in_tokens", 3000)
            r.setdefault("out_tokens", 500)
            r.setdefault("via", "batch")
            results.append(r)
        while len(results) < len(messages_list):
            r = dict(_all_zero_payload())
            r["in_tokens"] = 3000
            r["out_tokens"] = 500
            r["via"] = "batch"
            results.append(r)
        return results, []

    fake.captured = captured
    return fake


# ===========================================================================
# Test 1 — validator accepts a valid 23-binary dict
# ===========================================================================
def test_validator_accepts_valid_dict():
    out = _validated_layer_c_dict(_all_zero_payload())
    assert set(out.keys()) == set(LAYER_C_BINARIES)
    assert all(out[b] == 0 for b in LAYER_C_BINARIES)
    out2 = _validated_layer_c_dict(_support_payload())
    assert out2["expresses_explicit_support"] == 1
    assert out2["names_targeted_provision"] == 1


# ===========================================================================
# Test 2 — validator rejects missing binary field
# ===========================================================================
def test_validator_rejects_missing_binary():
    bad = _all_zero_payload()
    del bad["expresses_explicit_opposition"]
    try:
        _validated_layer_c_dict(bad)
    except ValueError as e:
        assert "expresses_explicit_opposition" in str(e)
        assert "missing" in str(e).lower()
        return
    assert False, "expected ValueError on missing binary"


# ===========================================================================
# Test 3 — validator rejects non-0/1 values (including bools and "1" strings)
# ===========================================================================
def test_validator_rejects_invalid_value():
    cases = [
        ("burdensome", 2),
        ("burdensome", "1"),
        ("burdensome", True),    # bool must be rejected even though it ==1
        ("burdensome", False),   # bool must be rejected even though it ==0
        ("burdensome", 1.0),
        ("burdensome", None),
    ]
    for field, bad_value in cases:
        bad = _all_zero_payload()
        bad[field] = bad_value
        try:
            _validated_layer_c_dict(bad)
        except ValueError as e:
            assert field in str(e), (
                f"expected ValueError to mention {field!r}; got: {e}")
            continue
        assert False, f"expected ValueError on {field}={bad_value!r}"


# ===========================================================================
# Test 4 — truncation policy applied for long comments
# ===========================================================================
def test_truncation_policy_applied_for_long_comments():
    long_text = (
        "A" * (TRUNCATION_HEAD_CHARS + 200)
        + "MID" * 50
        + "Z" * (TRUNCATION_TAIL_CHARS + 200)
    )
    assert len(long_text) > TRUNCATION_TRIGGER_CHARS
    out, was_trunc = truncate_for_layer_c(long_text)
    assert was_trunc is True
    assert out.startswith("A" * TRUNCATION_HEAD_CHARS)
    assert out.endswith("Z" * TRUNCATION_TAIL_CHARS)
    assert TRUNCATION_MARKER.strip() in out
    # The truncated text should be much smaller than the original.
    assert len(out) < len(long_text)


# ===========================================================================
# Test 5 — short comments pass through unchanged
# ===========================================================================
def test_short_comments_pass_through():
    short = "Brief comment — fewer than 12K chars."
    out, was_trunc = truncate_for_layer_c(short)
    assert out == short
    assert was_trunc is False

    # Right at the trigger boundary.
    boundary = "x" * TRUNCATION_TRIGGER_CHARS
    out2, was_trunc2 = truncate_for_layer_c(boundary)
    assert out2 == boundary
    assert was_trunc2 is False


# ===========================================================================
# Test 6 — cost cap pre-check raises before any API call
# ===========================================================================
def test_cost_cap_pre_check_raises():
    """100K comments × $0.0044/call = $440. With max_cost=$10, the batch
    pre-check inside `_call_openai_batch` raises CostCapExceeded BEFORE
    any client API call is made."""
    import pandas as pd
    big_df = pd.DataFrame([
        {"document_id": f"c{i}", "docket_id": "EPA-X",
         "comment": "filler"}
        for i in range(100_000)
    ])

    call_count = {"n": 0}
    def fake(messages_list, *args, **kwargs):
        call_count["n"] += 1
        return [None] * len(messages_list), []

    try:
        run_layer_c(big_df, batch=True, max_cost_usd=10.0,
                    call_batch=fake)
    except CostCapExceeded as e:
        # Pre-check fires before our fake batch is invoked.
        assert call_count["n"] == 0, (
            f"expected pre-check to fire before call_batch; "
            f"got {call_count['n']} invocations")
        assert "10" in str(e) or "exceeded" in str(e).lower()
        return
    assert False, "expected CostCapExceeded"


# ===========================================================================
# Test 7 — batch-mode happy path: shape + dtypes
# ===========================================================================
def test_batch_mode_happy_path():
    import pandas as pd
    df = _make_comments_df(n=3)
    call_batch = _mk_call_batch([
        _support_payload(), _all_zero_payload(), _support_payload(),
    ])

    out = run_layer_c(df, batch=True, max_cost_usd=10.0,
                      call_batch=call_batch)
    # Shape
    assert len(out) == 3
    expected_cols = {"comment_id", "docket_id", *LAYER_C_BINARIES,
                     "truncation_flag", "in_tokens", "out_tokens",
                     "cost_usd", "via"}
    assert expected_cols.issubset(set(out.columns))

    # Dtypes — binaries must be plain int, truncation_flag bool.
    for b in LAYER_C_BINARIES:
        assert out[b].dtype.kind in ("i", "u"), (
            f"{b} dtype must be integer, got {out[b].dtype}")
    assert out["truncation_flag"].dtype == bool

    # Submission-order preservation.
    assert list(out["comment_id"]) == [
        "EPA-COMMENT-0000", "EPA-COMMENT-0001", "EPA-COMMENT-0002"]
    assert list(out["docket_id"]) == ["EPA-TEST-DOCKET"] * 3

    # Cost accounting at batched gpt-5 rates: 3000 in × 0.625/1M + 500 out × 5/1M
    expected_cost = 3000 * 0.625 / 1_000_000 + 500 * 5.0 / 1_000_000
    assert all(abs(c - expected_cost) < 1e-9 for c in out["cost_usd"])

    # Validator hook plumbed correctly.
    from layer_c_judge import _validated_layer_c_dict
    assert call_batch.captured["validate_fn"] is _validated_layer_c_dict


# ===========================================================================
# Test 8 — real-time mode happy path (regression check)
# ===========================================================================
def test_realtime_mode_happy_path():
    import json
    df = _make_comments_df(n=2)

    def fake_call(system_prompt, user_prompt, model, max_tokens):
        return json.dumps(_support_payload()), 3000, 500

    def fake_parse(text, sp, up, *, provider, model, max_tokens):
        return json.loads(text), 0, 0

    out = run_layer_c(df, batch=False, max_cost_usd=10.0,
                      call_json=fake_call, parse_json=fake_parse,
                      progress_every=0)
    assert len(out) == 2
    expected_cols = {"comment_id", "docket_id", *LAYER_C_BINARIES,
                     "truncation_flag", "in_tokens", "out_tokens",
                     "cost_usd", "via"}
    assert expected_cols.issubset(set(out.columns))
    for b in LAYER_C_BINARIES:
        assert out[b].dtype.kind in ("i", "u")
    assert (out["via"] == "realtime").all()
    # Real-time path bills at standard rates: 3000 × 1.25/1M + 500 × 10/1M
    expected_cost = 3000 * 1.25 / 1_000_000 + 500 * 10.0 / 1_000_000
    assert all(abs(c - expected_cost) < 1e-9 for c in out["cost_usd"])


# ===========================================================================
# Bonus — truncation flag round-trips through run_layer_c
# ===========================================================================
def test_truncation_flag_propagates_through_run_layer_c():
    df = _make_comments_df(n=3, long_comment_at=1)
    call_batch = _mk_call_batch([_all_zero_payload()] * 3)
    out = run_layer_c(df, batch=True, max_cost_usd=10.0,
                      call_batch=call_batch)
    flags = list(out["truncation_flag"])
    assert flags == [False, True, False], (
        f"expected truncation_flag=[False, True, False]; got {flags}")


def test_estimate_cost_anchor():
    # PROJECT_FACTS §7 anchor: 70K Tier B × $0.0044 ≈ $308.
    est = estimate_cost(70_075)
    assert 300 < est < 312, f"expected ~$308 estimate, got ${est:.2f}"


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
