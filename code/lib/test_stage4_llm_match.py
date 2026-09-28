"""
test_stage4_llm_match.py — unit tests for the Stage 4 LLM matcher.

No real OpenAI calls — we inject the call wrappers via the `call_json` /
`parse_json` parameters on `match_one_pair` and `match_pairs`. Tests
cover JSON schema enforcement, closed-enum validation, truncation, and
the cost-cap raise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from stage4_llm_match import (  # noqa: E402
    ALLOWED_STANCES,
    CostCapExceeded,
    TRUNCATION_HEAD_CHARS,
    TRUNCATION_TAIL_CHARS,
    TRUNCATION_TRIGGER_CHARS,
    _validate_match_payload,
    call_cost_usd,
    estimate_cost,
    match_one_pair,
    match_pairs,
    truncate_comment,
)


# ---------------------------------------------------------------------------
# Mocked LLM helpers
# ---------------------------------------------------------------------------
def _make_call_returning(payload: dict, in_tok: int = 1500, out_tok: int = 200):
    def call(system_prompt, user_prompt, *, model, max_tokens):
        return json.dumps(payload), in_tok, out_tok
    return call


def _passthrough_parse(text, system_prompt, user_prompt,
                       *, provider, model, max_tokens):
    return json.loads(text), 0, 0


def _ob():
    return {
        "cfr_section": "80.27(d)(2)", "obligation_text": "Each operator must X.",
        "subject": "operator", "modal": "must", "action": "submit",
        "object": "report",
    }


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------
def test_validate_accepts_well_formed_payload():
    result = _validate_match_payload({
        "addressed": True, "stance": "SUPPORTING",
        "justification": "Comment endorses the rule.",
    })
    assert result.addressed is True
    assert result.stance == "SUPPORTING"
    assert "endorses" in result.justification


def test_validate_rejects_unknown_stance():
    try:
        _validate_match_payload({
            "addressed": True, "stance": "NEUTRAL",   # not in enum
            "justification": "x",
        })
    except ValueError as e:
        assert "stance" in str(e)
        return
    assert False, "expected ValueError on unknown stance"


def test_validate_rejects_missing_addressed():
    try:
        _validate_match_payload({"stance": "NONE", "justification": "x"})
    except ValueError as e:
        assert "addressed" in str(e)
        return
    assert False, "expected ValueError on missing addressed"


def test_validate_coerces_string_bool():
    # Some models emit "true"/"false" instead of literal bools.
    r1 = _validate_match_payload({
        "addressed": "true", "stance": "NONE", "justification": ""})
    assert r1.addressed is True
    r2 = _validate_match_payload({
        "addressed": "false", "stance": "NONE", "justification": ""})
    assert r2.addressed is False


def test_validate_rejects_non_bool_addressed():
    try:
        _validate_match_payload({"addressed": "maybe", "stance": "NONE",
                                 "justification": ""})
    except ValueError:
        return
    assert False, "expected ValueError on non-bool addressed"


def test_match_one_pair_returns_token_counts():
    call = _make_call_returning(
        {"addressed": False, "stance": "NONE", "justification": "off-topic"},
        in_tok=1234, out_tok=56,
    )
    res = match_one_pair(
        "Some comment text.", _ob(),
        call_json=call, parse_json=_passthrough_parse,
    )
    assert res.in_tokens == 1234
    assert res.out_tokens == 56
    assert res.stance == "NONE"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------
def test_short_comment_not_truncated():
    short = "x" * (TRUNCATION_TRIGGER_CHARS - 1)
    out, was = truncate_comment(short)
    assert out == short
    assert was is False


def test_long_comment_truncated_with_marker():
    long = ("A" * (TRUNCATION_HEAD_CHARS + 50)
            + "MIDDLE"
            + "Z" * (TRUNCATION_TAIL_CHARS + 100))
    assert len(long) > TRUNCATION_TRIGGER_CHARS
    out, was = truncate_comment(long)
    assert was is True
    assert out.startswith("A" * TRUNCATION_HEAD_CHARS)
    assert out.endswith("Z" * TRUNCATION_TAIL_CHARS)
    assert "[...middle elided" in out


def test_truncation_flag_propagates_through_match_one_pair():
    long_text = "X" * (TRUNCATION_TRIGGER_CHARS + 100)
    call = _make_call_returning(
        {"addressed": True, "stance": "SUPPORTING", "justification": "ok"})
    res = match_one_pair(
        long_text, _ob(),
        call_json=call, parse_json=_passthrough_parse,
    )
    assert getattr(res, "_was_truncated") is True


# ---------------------------------------------------------------------------
# Cost cap
# ---------------------------------------------------------------------------
def test_estimate_cost_anchor():
    # PROJECT_FACTS §7 anchor: 700,750 calls × $0.0019 ≈ $1,331.
    est = estimate_cost(700_750)
    assert 1300 < est < 1360


def test_call_cost_matches_batch_pricing():
    # 1500 in + 200 out → 1500 * 0.625/1M + 200 * 5/1M
    expected = 1500 * 0.625 / 1_000_000 + 200 * 5.0 / 1_000_000
    assert abs(call_cost_usd(1500, 200) - expected) < 1e-9


def test_cost_cap_raises_after_threshold():
    """Drive enough mocked calls (each costing the anchor $0.0019) to
    cross a tight cap and confirm we raise rather than silently overrun."""
    import pandas as pd

    pairs = [(f"c{i}", f"EPA-TEST__proposed__{i}_0", 0.7) for i in range(20)]
    cdf = pd.DataFrame({
        "comment_id": [p[0] for p in pairs],
        "docket_id": ["EPA-TEST"] * len(pairs),
        "comment": ["short comment text."] * len(pairs),
    })
    odf = pd.DataFrame({
        "obligation_id": [p[1] for p in pairs],
        "cfr_section": ["80.27"] * len(pairs),
        "obligation_text": ["x"] * len(pairs),
        "subject": ["op"] * len(pairs), "modal": ["must"] * len(pairs),
        "action": ["do"] * len(pairs), "object": ["thing"] * len(pairs),
    })
    call = _make_call_returning(
        {"addressed": False, "stance": "NONE", "justification": ""},
        in_tok=1500, out_tok=200,
    )

    # Each call costs ~$0.001938 (batched). 5 calls → $0.00969. Cap at $0.005.
    try:
        match_pairs(
            pairs, cdf, odf,
            call_json=call, parse_json=_passthrough_parse,
            batch=False, max_cost_usd=0.005, progress_every=0,
        )
    except CostCapExceeded as e:
        assert "0.005" in str(e) or "exceeded" in str(e).lower()
        return
    assert False, "expected CostCapExceeded"


def test_match_pairs_returns_expected_dataframe_shape():
    import pandas as pd

    pairs = [("c0", "EPA-TEST__proposed__1_0", 0.83),
             ("c1", "EPA-TEST__proposed__2_0", 0.67)]
    cdf = pd.DataFrame({
        "comment_id": ["c0", "c1"],
        "docket_id": ["EPA-TEST", "EPA-TEST"],
        "comment": ["First comment.", "Second comment."],
    })
    odf = pd.DataFrame({
        "obligation_id": ["EPA-TEST__proposed__1_0", "EPA-TEST__proposed__2_0"],
        "cfr_section": ["80.27", "80.28"],
        "obligation_text": ["text1", "text2"],
        "subject": ["op", "op"], "modal": ["must", "shall"],
        "action": ["submit", "retain"], "object": ["r", "r"],
    })
    call = _make_call_returning(
        {"addressed": True, "stance": "OPPOSING", "justification": "x"})
    df = match_pairs(
        pairs, cdf, odf,
        call_json=call, parse_json=_passthrough_parse,
        batch=False, max_cost_usd=10.0, progress_every=0,
    )
    expected_cols = {
        "comment_id", "obligation_id", "cosine_similarity",
        "addressed", "stance", "justification",
        "uncertainty_truncated", "in_tokens", "out_tokens", "cost_usd",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"missing columns: {expected_cols - set(df.columns)}")
    assert len(df) == 2
    assert (df["stance"] == "OPPOSING").all()


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
        print(f"\nFAIL: {len(failures)} test(s): {failures}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
