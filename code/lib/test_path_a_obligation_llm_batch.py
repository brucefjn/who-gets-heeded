"""
test_path_a_obligation_llm_batch.py — unit tests for the OpenAI Batch
API wrapper in path_a_obligation_llm._call_openai_batch.

All OpenAI SDK calls are mocked. No network. No real cost. The clock
is injectable so we can fast-forward the 24h polling window in tests
without actually waiting.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

import path_a_obligation_llm as palm  # noqa: E402
from path_a_obligation_llm import (  # noqa: E402
    CostCapExceeded,
    _call_openai_batch,
)


# ---------------------------------------------------------------------------
# Mocked OpenAI SDK client + helpers
# ---------------------------------------------------------------------------
def _mk_message(idx: int) -> list[dict]:
    return [
        {"role": "system", "content": "Match obligation."},
        {"role": "user", "content": f"OBLIGATION {idx}\nCOMMENT {idx}"},
    ]


def _mk_response_line(custom_id: str, content: str, prompt_tok: int = 1500,
                      completion_tok: int = 200) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"req_{custom_id}",
            "body": {
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": completion_tok,
                    "total_tokens": prompt_tok + completion_tok,
                },
            },
        },
        "error": None,
    }


def _mk_error_line(custom_id: str, code: str = "invalid_request") -> dict:
    return {
        "custom_id": custom_id,
        "response": None,
        "error": {"code": code, "message": "synthetic test error"},
    }


def _valid_payload(addressed: bool = True, stance: str = "SUPPORTING") -> str:
    return json.dumps({
        "addressed": addressed, "stance": stance,
        "justification": "synthetic test justification",
    })


def _build_client(*, output_text: str, statuses=("completed",)):
    """Mock OpenAI client. `statuses` is the sequence of `status` values
    returned on successive batches.retrieve() calls (the last value is
    sticky once consumed)."""
    status_iter = list(statuses)
    state = {"i": 0}

    file_obj = mock.MagicMock()
    file_obj.id = "file_test"
    batch_obj = mock.MagicMock()
    batch_obj.id = "batch_test_001"
    batch_obj.status = status_iter[0]
    batch_obj.output_file_id = "file_out"
    batch_obj.errors = None

    def retrieve(batch_id):
        idx = min(state["i"], len(status_iter) - 1)
        state["i"] += 1
        b = mock.MagicMock()
        b.id = batch_id
        b.status = status_iter[idx]
        b.output_file_id = "file_out"
        b.errors = None
        return b

    content_resp = mock.MagicMock()
    content_resp.text = output_text

    client = mock.MagicMock()
    client.files.create.return_value = file_obj
    client.batches.create.return_value = batch_obj
    client.batches.retrieve.side_effect = retrieve
    client.files.content.return_value = content_resp
    return client


class _FakeClock:
    def __init__(self, step: float = 60.0):
        self.t = 0.0
        self.step = step

    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += float(dt) if dt else 1.0


def _stage4_validator(parsed: dict) -> dict:
    """Stage 4 schema enforcement, inlined to keep this test file
    independent of the stage4_llm_match import path."""
    if not isinstance(parsed, dict):
        raise ValueError("non-dict")
    addressed = parsed.get("addressed")
    if not isinstance(addressed, bool):
        raise ValueError(f"addressed not bool: {addressed!r}")
    stance = parsed.get("stance")
    if stance not in {"SUPPORTING", "OPPOSING", "SUGGESTING_MODIFICATION", "NONE"}:
        raise ValueError(f"stance invalid: {stance!r}")
    return {
        "addressed": addressed,
        "stance": stance,
        "justification": str(parsed.get("justification", ""))[:240],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_successful_batch_flow(tmp_path: Path = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(i) for i in range(3)]
    output_lines = "\n".join(json.dumps(_mk_response_line(
        f"pair_{i}", _valid_payload())) for i in range(3))

    clock = _FakeClock()
    client = _build_client(output_text=output_lines, statuses=("completed",))

    successful, failed = _call_openai_batch(
        messages_list, model="gpt-5", max_cost_usd=10.0,
        validate_fn=_stage4_validator,
        sleep_fn=clock.sleep, time_fn=clock.time,
        client=client, output_dir=tmp_path,
    )
    assert len(successful) == 3
    assert failed == []
    assert all(r is not None for r in successful)
    assert all(r["via"] == "batch" for r in successful)
    assert all(r["in_tokens"] == 1500 for r in successful)
    assert all(r["out_tokens"] == 200 for r in successful)
    # JSONL audit trail persisted
    input_files = list(tmp_path.glob("*__input.jsonl"))
    output_files = list(tmp_path.glob("*__output.jsonl"))
    assert len(input_files) == 1 and len(output_files) == 1


def test_mixed_batch_with_realtime_recovery(tmp_path: Path = None):
    """2 batch successes + 1 malformed JSON → recovery yields 3/3
    successful, 0 failed. Verify real-time was called exactly once."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(i) for i in range(3)]
    output_lines = "\n".join([
        json.dumps(_mk_response_line("pair_0", _valid_payload())),
        json.dumps(_mk_response_line("pair_1", "{not valid json}")),
        json.dumps(_mk_response_line("pair_2", _valid_payload(stance="OPPOSING"))),
    ])
    clock = _FakeClock()
    client = _build_client(output_text=output_lines)

    recovery_calls = {"n": 0}

    def fake_call_openai_json(system_prompt, user_prompt, model, max_tokens):
        recovery_calls["n"] += 1
        return _valid_payload(stance="SUGGESTING_MODIFICATION"), 1234, 56

    def fake_parse(text, sp, up, *, provider, model, max_tokens):
        return json.loads(text), 0, 0

    with mock.patch.object(palm, "_call_openai_json", fake_call_openai_json), \
         mock.patch.object(palm, "_parse_json_with_repair", fake_parse):
        successful, failed = _call_openai_batch(
            messages_list, model="gpt-5", max_cost_usd=10.0,
            validate_fn=_stage4_validator,
            sleep_fn=clock.sleep, time_fn=clock.time,
            client=client, output_dir=tmp_path,
        )

    assert recovery_calls["n"] == 1, (
        f"real-time recovery should have been called exactly once for the "
        f"malformed pair; got {recovery_calls['n']} calls.")
    assert len(successful) == 3
    assert all(r is not None for r in successful)
    assert failed == []
    # pair_1 went via recovery; pair_0 + pair_2 via batch
    assert successful[1]["via"] == "recovery"
    assert successful[0]["via"] == "batch"
    assert successful[2]["via"] == "batch"
    # Recovery row preserves real-time tokens
    assert successful[1]["in_tokens"] == 1234
    assert successful[1]["out_tokens"] == 56


def test_cost_cap_pre_check_raises_without_api_call(tmp_path: Path = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(i) for i in range(10_000)]
    client = mock.MagicMock()
    try:
        _call_openai_batch(
            messages_list, model="gpt-5", max_cost_usd=1.0,
            client=client, output_dir=tmp_path,
        )
    except CostCapExceeded as e:
        assert "1.00" in str(e) or "cap" in str(e).lower()
        # No API call should have been made.
        assert not client.files.create.called
        assert not client.batches.create.called
        # No JSONL files persisted.
        assert list(tmp_path.glob("*__input.jsonl")) == []
        return
    assert False, "expected CostCapExceeded"


def test_batch_failed_status_raises_with_batch_id(tmp_path: Path = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(0)]
    clock = _FakeClock()
    client = _build_client(output_text="", statuses=("failed",))
    try:
        _call_openai_batch(
            messages_list, model="gpt-5", max_cost_usd=10.0,
            sleep_fn=clock.sleep, time_fn=clock.time,
            client=client, output_dir=tmp_path,
        )
    except RuntimeError as e:
        assert "batch_test_001" in str(e), f"batch id missing in error: {e}"
        assert "failed" in str(e).lower()
        return
    assert False, "expected RuntimeError on status=failed"


def test_timeout_after_24h_raises_with_batch_id(tmp_path: Path = None):
    """Mock status as always in_progress + a fast-forward clock. The
    polling loop should raise TimeoutError once elapsed > timeout."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(0)]

    # Each sleep advances the fake clock by 1 hour → ~25 polls until
    # the 24h timeout triggers (cheap, no real sleeping).
    class HourClock:
        def __init__(self):
            self.t = 0.0
        def time(self):
            return self.t
        def sleep(self, dt):
            self.t += 3600.0

    clock = HourClock()
    # always in_progress — the iterator is consumed; status sticks at last value
    client = _build_client(output_text="",
                           statuses=("in_progress",) * 100)
    try:
        _call_openai_batch(
            messages_list, model="gpt-5", max_cost_usd=10.0,
            sleep_fn=clock.sleep, time_fn=clock.time,
            client=client, output_dir=tmp_path,
            initial_poll_seconds=1.0, max_poll_seconds=1.0,   # tight
        )
    except TimeoutError as e:
        assert "batch_test_001" in str(e), f"batch id missing in error: {e}"
        assert "24h" in str(e) or "manual recovery" in str(e).lower()
        return
    assert False, "expected TimeoutError"


def test_schema_validation_failure_routes_to_recovery(tmp_path: Path = None):
    """Output JSONL contains stance='INVALID_VALUE' — should not be
    silently included; should be routed through real-time recovery."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(0)]
    bad_payload = json.dumps({
        "addressed": True, "stance": "INVALID_VALUE",
        "justification": "x",
    })
    output_lines = json.dumps(_mk_response_line("pair_0", bad_payload))
    clock = _FakeClock()
    client = _build_client(output_text=output_lines)

    recovery_calls = {"n": 0}
    def fake_call(sp, up, model, max_tokens):
        recovery_calls["n"] += 1
        return _valid_payload(stance="OPPOSING"), 100, 20
    def fake_parse(text, sp, up, *, provider, model, max_tokens):
        return json.loads(text), 0, 0

    with mock.patch.object(palm, "_call_openai_json", fake_call), \
         mock.patch.object(palm, "_parse_json_with_repair", fake_parse):
        successful, failed = _call_openai_batch(
            messages_list, model="gpt-5", max_cost_usd=10.0,
            validate_fn=_stage4_validator,
            sleep_fn=clock.sleep, time_fn=clock.time,
            client=client, output_dir=tmp_path,
        )
    assert recovery_calls["n"] == 1
    assert failed == []
    assert successful[0] is not None
    assert successful[0]["stance"] == "OPPOSING"
    assert successful[0]["via"] == "recovery"


def test_custom_id_roundtrip_with_shuffled_output(tmp_path: Path = None):
    """Output lines arrive in a non-submission order. Result indices
    must follow `custom_id`, not output order."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(i) for i in range(5)]
    # Tag each pair's payload with its index in justification so we can
    # verify which output slot it landed in.
    payloads = {
        i: json.dumps({"addressed": True, "stance": "SUPPORTING",
                       "justification": f"payload-for-pair-{i}"})
        for i in range(5)
    }
    # Shuffled emission order: 3, 0, 4, 2, 1
    order = [3, 0, 4, 2, 1]
    output_lines = "\n".join(
        json.dumps(_mk_response_line(f"pair_{i}", payloads[i])) for i in order
    )
    clock = _FakeClock()
    client = _build_client(output_text=output_lines)
    successful, failed = _call_openai_batch(
        messages_list, model="gpt-5", max_cost_usd=10.0,
        validate_fn=_stage4_validator,
        sleep_fn=clock.sleep, time_fn=clock.time,
        client=client, output_dir=tmp_path,
    )
    assert failed == []
    for i in range(5):
        assert successful[i] is not None
        assert successful[i]["justification"] == f"payload-for-pair-{i}", (
            f"slot {i} got payload {successful[i]['justification']!r} "
            "— custom_id round-trip broken.")


def test_all_batch_failure_then_recovery_failure(tmp_path: Path = None):
    """Every pair errors in batch and recovery also raises — all pairs
    land in failed_pairs with custom_id preserved."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(i) for i in range(2)]
    output_lines = "\n".join([
        json.dumps(_mk_error_line("pair_0")),
        json.dumps(_mk_error_line("pair_1")),
    ])
    clock = _FakeClock()
    client = _build_client(output_text=output_lines)

    def fake_call(sp, up, model, max_tokens):
        raise RuntimeError("recovery also down")

    with mock.patch.object(palm, "_call_openai_json", fake_call):
        successful, failed = _call_openai_batch(
            messages_list, model="gpt-5", max_cost_usd=10.0,
            validate_fn=_stage4_validator,
            sleep_fn=clock.sleep, time_fn=clock.time,
            client=client, output_dir=tmp_path,
        )
    assert successful == [None, None]
    assert len(failed) == 2
    custom_ids = sorted(fp["custom_id"] for fp in failed)
    assert custom_ids == ["pair_0", "pair_1"]
    for fp in failed:
        assert "recovery_error" in fp
        assert "batch_error" in fp


def test_jsonl_audit_trail_is_persisted(tmp_path: Path = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="batch_test_"))
    messages_list = [_mk_message(0)]
    output_lines = json.dumps(_mk_response_line("pair_0", _valid_payload()))
    clock = _FakeClock()
    client = _build_client(output_text=output_lines)
    _call_openai_batch(
        messages_list, model="gpt-5", max_cost_usd=10.0,
        validate_fn=_stage4_validator,
        sleep_fn=clock.sleep, time_fn=clock.time,
        client=client, output_dir=tmp_path,
    )
    input_files = list(tmp_path.glob("*__input.jsonl"))
    output_files = list(tmp_path.glob("*__output.jsonl"))
    assert len(input_files) == 1
    assert len(output_files) == 1
    in_text = input_files[0].read_text(encoding="utf-8")
    body = json.loads(in_text.splitlines()[0])
    assert body["custom_id"] == "pair_0"
    assert body["method"] == "POST"
    assert body["url"] == "/v1/chat/completions"
    assert body["body"]["model"] == "gpt-5"


# ===========================================================================
# Task E — run_primary_verification batch wiring
#
# These tests inject a fake `call_batch` directly into
# `run_primary_verification(batch=True, call_batch=...)` so we exercise
# the wiring without spinning up the OpenAI Batch API or mocking the
# whole client. The contract is: call_batch returns
# (successful: list[dict|None], failed: list[dict]) with successful[i]
# being the validated dict + {in_tokens, out_tokens, via}.
# ===========================================================================
def _write_candidates_csv(path: Path, n_candidates: int = 3) -> None:
    """Write a minimal Stage 1a-shaped candidates CSV."""
    import csv as _csv
    fieldnames = ["obligation_idx", "docket_id", "rule_type", "cfr_part",
                  "cfr_section", "subject", "modal", "modal_strength",
                  "action", "object", "amendatory_action", "is_conditional",
                  "condition_text", "passive_voice", "obligation_text",
                  "char_offset_start", "char_offset_end"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n_candidates):
            w.writerow({
                "obligation_idx": str(i + 1), "docket_id": "EPA-TEST",
                "rule_type": "final", "cfr_part": "80",
                "cfr_section": f"80.{i+1}",
                "subject": f"Operator {i}", "modal": "must",
                "modal_strength": "strong",
                "action": "submit", "object": f"report {i}",
                "amendatory_action": "revise",
                "is_conditional": "False", "condition_text": "",
                "passive_voice": "False",
                "obligation_text": f"Operator {i} must submit report {i}.",
                "char_offset_start": "0", "char_offset_end": "50",
            })


def _valid_verification_dict(*, n_obligations: int = 1) -> dict:
    """Return a dict matching the strict validator's schema."""
    return {
        "verified": True,
        "verified_reason": "amendatory block; clear actor + modal",
        "complexity": "simple",
        "recital_disposition": "none",
        "obligations": [
            {
                "subject": "Operator", "modal": "must",
                "modal_strength": "strong", "action": "submit",
                "object": "report",
                "passive_voice": False, "is_conditional": False,
                "condition_text": None, "cfr_section": "80.1",
                "amendatory_action": "revise", "cross_reference": None,
            }
            for _ in range(n_obligations)
        ],
    }


def _make_call_batch(per_call_results: list, expected_validate_fn=None):
    """Build a fake call_batch that returns the supplied results list as
    `successful`. Each entry is either a dict (will be augmented with
    in_tokens/out_tokens/via='batch') or None (failure)."""
    captured = {"calls": 0, "last_messages": None, "last_validate_fn": None}

    def fake(messages_list, model, max_cost_usd, *, batch_id_prefix,
             max_tokens, validate_fn=None):
        captured["calls"] += 1
        captured["last_messages"] = messages_list
        captured["last_validate_fn"] = validate_fn
        results = []
        for i, payload in enumerate(per_call_results[:len(messages_list)]):
            if payload is None:
                results.append(None)
                continue
            r = dict(payload)
            r.setdefault("in_tokens", 1500)
            r.setdefault("out_tokens", 200)
            r.setdefault("via", "batch")
            results.append(r)
        # Pad with valid results if fewer payloads than messages
        while len(results) < len(messages_list):
            r = dict(_valid_verification_dict())
            r["in_tokens"] = 1500
            r["out_tokens"] = 200
            r["via"] = "batch"
            results.append(r)
        return results, []

    fake.captured = captured
    return fake


def test_run_primary_verification_batch_returns_same_shape_as_realtime(
        tmp_path: Path = None):
    """Task E test 1: batch=True with mocked call_batch produces a
    verified CSV with the same row schema as the real-time path."""
    import tempfile, csv as _csv
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="task_e_test_"))
    candidates_csv = tmp_path / "candidates.csv"
    binding_text = tmp_path / "fr.txt"
    out_csv = tmp_path / "verified.csv"
    _write_candidates_csv(candidates_csv, n_candidates=3)
    binding_text.write_text(
        "Each operator must submit report 0. " * 200, encoding="utf-8")

    call_batch = _make_call_batch(
        [_valid_verification_dict() for _ in range(3)])

    from path_a_obligation_llm import run_primary_verification
    summary = run_primary_verification(
        candidates_csv=candidates_csv,
        binding_text_path=binding_text,
        out_csv=out_csv,
        provider="openai", model="gpt-5",
        run_safety_net=False, batch=True,
        call_batch=call_batch, max_cost_usd=10.0,
    )
    assert summary["verified_candidates"] == 3
    assert summary["verified_rows_written"] >= 3
    assert summary["total_in_tokens"] == 4500   # 3 × 1500
    assert summary["total_out_tokens"] == 600    # 3 × 200

    # Verify the CSV columns match the canonical VerifiedObligation schema.
    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 3
    required_cols = {"candidate_idx", "docket_id", "rule_type", "split_idx",
                     "verified", "verified_reason", "complexity",
                     "recital_disposition", "modal", "modal_strength",
                     "amendatory_action", "obligation_text", "found_via",
                     "in_tokens", "out_tokens"}
    assert required_cols.issubset(set(rows[0].keys()))
    # Submission-order mapping: candidate_idx should be 1, 2, 3 in order.
    assert [int(r["candidate_idx"]) for r in rows] == [1, 2, 3]

    # The validator passed to call_batch must be the strict one.
    from path_a_obligation_llm import _validated_verification_dict
    assert call_batch.captured["last_validate_fn"] is _validated_verification_dict


def test_run_primary_verification_no_batch_regression(tmp_path: Path = None):
    """Task E test 2: batch=False keeps the existing real-time path
    functional. We mock _call_llm_json + _parse_json_with_repair to
    avoid actual API calls while still exercising verify_candidate."""
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="task_e_test_"))
    candidates_csv = tmp_path / "candidates.csv"
    binding_text = tmp_path / "fr.txt"
    out_csv = tmp_path / "verified.csv"
    _write_candidates_csv(candidates_csv, n_candidates=2)
    binding_text.write_text("Each operator must submit. " * 200, encoding="utf-8")

    def fake_call_llm_json(system_prompt, user_prompt, *, provider, model,
                            max_tokens):
        return json.dumps(_valid_verification_dict()), 1500, 200

    def fake_parse(text, sp, up, *, provider, model, max_tokens):
        return json.loads(text), 0, 0

    with mock.patch.object(palm, "_call_llm_json", fake_call_llm_json), \
         mock.patch.object(palm, "_parse_json_with_repair", fake_parse):
        from path_a_obligation_llm import run_primary_verification
        summary = run_primary_verification(
            candidates_csv=candidates_csv,
            binding_text_path=binding_text,
            out_csv=out_csv,
            provider="openai", model="gpt-5",
            run_safety_net=False, batch=False,
        )
    assert summary["verified_candidates"] == 2
    assert summary["total_in_tokens"] == 3000
    assert summary["total_out_tokens"] == 400


def test_run_primary_verification_schema_failure_routes_to_recovery(
        tmp_path: Path = None):
    """Task E test 3: a batch row with an invalid modal_strength must
    route to recovery (real-time _call_openai_json), and the real-time
    call must be invoked exactly once for that single failed candidate."""
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="task_e_test_"))
    candidates_csv = tmp_path / "candidates.csv"
    binding_text = tmp_path / "fr.txt"
    out_csv = tmp_path / "verified.csv"
    _write_candidates_csv(candidates_csv, n_candidates=2)
    binding_text.write_text("Each operator must submit. " * 200, encoding="utf-8")

    # Build a real-OpenAI-client mock that returns one good row + one
    # bad row in batch output, then the bad row routes to recovery and
    # the mocked _call_openai_json returns a valid payload.
    bad_payload = json.dumps({
        "verified": True, "verified_reason": "x",
        "complexity": "simple", "recital_disposition": "none",
        "obligations": [{
            "subject": "Operator", "modal": "must",
            "modal_strength": "INVALID_VALUE",   # ← will fail strict validation
            "action": "submit", "object": "x",
            "amendatory_action": "revise",
        }],
    })
    good_payload = json.dumps(_valid_verification_dict())
    output_lines = "\n".join([
        json.dumps(_mk_response_line("pair_0", good_payload)),
        json.dumps(_mk_response_line("pair_1", bad_payload)),
    ])
    clock = _FakeClock()
    client = _build_client(output_text=output_lines)

    recovery_calls = {"n": 0}
    def fake_call(sp, up, model, max_tokens):
        recovery_calls["n"] += 1
        return good_payload, 1234, 56

    def fake_parse(text, sp, up, *, provider, model, max_tokens):
        return json.loads(text), 0, 0

    # The batch function uses the real _call_openai_batch; intercept its
    # internal recovery hooks (_call_openai_json + _parse_json_with_repair).
    # Bind the mocked client + clock through a partial.
    import functools
    real_batch = palm._call_openai_batch
    bound_batch = functools.partial(
        real_batch,
        sleep_fn=clock.sleep, time_fn=clock.time,
        client=client, output_dir=tmp_path,
    )

    with mock.patch.object(palm, "_call_openai_json", fake_call), \
         mock.patch.object(palm, "_parse_json_with_repair", fake_parse):
        from path_a_obligation_llm import run_primary_verification
        summary = run_primary_verification(
            candidates_csv=candidates_csv,
            binding_text_path=binding_text,
            out_csv=out_csv,
            provider="openai", model="gpt-5",
            run_safety_net=False, batch=True,
            call_batch=bound_batch, max_cost_usd=10.0,
        )
    assert recovery_calls["n"] == 1, (
        f"expected exactly one real-time recovery call; got {recovery_calls['n']}")
    # Both candidates ultimately succeed (one via batch, one via recovery).
    assert summary["verified_candidates"] == 2
    assert summary["verified_rows_written"] >= 2


def test_run_primary_verification_cost_cap_raises_before_batch_created(
        tmp_path: Path = None):
    """Task E test 4: max_cost_usd=1.0 with 10K candidates raises
    CostCapExceeded BEFORE the batch is created (no API call made)."""
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="task_e_test_"))
    candidates_csv = tmp_path / "candidates.csv"
    binding_text = tmp_path / "fr.txt"
    out_csv = tmp_path / "verified.csv"
    _write_candidates_csv(candidates_csv, n_candidates=10_000)
    binding_text.write_text("operator must submit. " * 100, encoding="utf-8")

    fake_client = mock.MagicMock()
    import functools
    bound_batch = functools.partial(
        palm._call_openai_batch,
        sleep_fn=lambda d: None, time_fn=lambda: 0.0,
        client=fake_client, output_dir=tmp_path,
    )
    from path_a_obligation_llm import (
        run_primary_verification, CostCapExceeded,
    )
    raised = False
    try:
        run_primary_verification(
            candidates_csv=candidates_csv,
            binding_text_path=binding_text,
            out_csv=out_csv,
            provider="openai", model="gpt-5",
            run_safety_net=False, batch=True,
            call_batch=bound_batch, max_cost_usd=1.0,
        )
    except CostCapExceeded as e:
        raised = True
        assert "1.00" in str(e) or "exceeded" in str(e).lower()
    assert raised, "expected CostCapExceeded with 10K candidates @ $1 cap"
    # No file upload happened.
    assert not fake_client.files.create.called
    assert not fake_client.batches.create.called


def test_run_safety_net_batch_returns_same_shape(tmp_path: Path = None):
    """Task E test 5: safety-net batch path. Mock the batch function so
    Phase 1 returns one missed obligation; Phase 2 verifies it. Result
    rows should appear in the verified CSV with found_via='safety_net_verified'."""
    import tempfile, csv as _csv
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="task_e_test_"))
    candidates_csv = tmp_path / "candidates.csv"
    binding_text = tmp_path / "fr.txt"
    out_csv = tmp_path / "verified.csv"
    _write_candidates_csv(candidates_csv, n_candidates=1)
    # Need a binding text with at least one amendatory block so safety
    # net has something to scan.
    binding_text.write_text(
        "Section 80.1 is revised to read as follows:\n\n"
        "§ 80.1  Scope.\n\nEach operator shall maintain records. " * 50,
        encoding="utf-8",
    )

    call_state = {"n": 0}

    def call_batch(messages_list, model, max_cost_usd, *, batch_id_prefix,
                   max_tokens, validate_fn=None):
        call_state["n"] += 1
        # Three batch invocations expected: primary, safety-net Phase 1,
        # safety-net Phase 2.
        if batch_id_prefix == "stage1b_primary":
            return ([{**_valid_verification_dict(), "in_tokens": 1500,
                      "out_tokens": 200, "via": "batch"}], [])
        if batch_id_prefix == "stage1b_safety_net":
            return ([{"missed_obligations": [{
                "subject": "auditor", "modal": "shall",
                "action": "review", "object": "records",
                "obligation_text": "auditor shall review records.",
                "found_via": "no_modal",
            }], "in_tokens": 800, "out_tokens": 100, "via": "batch"}], [])
        if batch_id_prefix == "stage1b_safety_net_verify":
            return ([{**_valid_verification_dict(), "in_tokens": 1500,
                      "out_tokens": 200, "via": "batch"}], [])
        raise AssertionError(f"unexpected batch_id_prefix={batch_id_prefix}")

    from path_a_obligation_llm import run_primary_verification
    summary = run_primary_verification(
        candidates_csv=candidates_csv,
        binding_text_path=binding_text,
        out_csv=out_csv,
        provider="openai", model="gpt-5",
        run_safety_net=True, batch=True,
        call_batch=call_batch, max_cost_usd=10.0,
    )
    # 3 batch calls — primary + safety_net + safety_net_verify
    assert call_state["n"] == 3, (
        f"expected 3 batch invocations; got {call_state['n']}")
    assert summary["safety_net_hits"] >= 1
    assert summary["safety_net_verified"] >= 1
    # Output CSV should contain at least one row with
    # found_via='safety_net_verified'.
    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    sn_rows = [r for r in rows if r["found_via"] == "safety_net_verified"]
    assert len(sn_rows) >= 1, "expected at least one safety_net_verified row"


def test_run_safety_net_batch_disabled_when_no_safety_net(tmp_path: Path = None):
    """Companion to test 5: when run_safety_net=False, the safety-net
    batch invocations don't happen."""
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="task_e_test_"))
    candidates_csv = tmp_path / "candidates.csv"
    binding_text = tmp_path / "fr.txt"
    out_csv = tmp_path / "verified.csv"
    _write_candidates_csv(candidates_csv, n_candidates=1)
    binding_text.write_text("operator must submit. " * 100, encoding="utf-8")

    call_state = {"prefixes": []}
    def call_batch(messages_list, model, max_cost_usd, *, batch_id_prefix,
                   max_tokens, validate_fn=None):
        call_state["prefixes"].append(batch_id_prefix)
        return ([{**_valid_verification_dict(), "in_tokens": 1500,
                  "out_tokens": 200, "via": "batch"}], [])

    from path_a_obligation_llm import run_primary_verification
    run_primary_verification(
        candidates_csv=candidates_csv,
        binding_text_path=binding_text,
        out_csv=out_csv,
        provider="openai", model="gpt-5",
        run_safety_net=False, batch=True,
        call_batch=call_batch, max_cost_usd=10.0,
    )
    assert call_state["prefixes"] == ["stage1b_primary"], (
        f"expected only primary batch call; got {call_state['prefixes']}")


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
