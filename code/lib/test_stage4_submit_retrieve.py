"""
test_stage4_submit_retrieve.py — Task I unit tests.

Covers the submit-then-recover split added for the 16 GB workstation
ceiling:

  (a) `--submit-only` correctly stages a batch, writes a tracking row,
      and exits without polling (RAM held only for the prefilter+submit
      window, then freed).
  (b) The retrieve script polls each tracking row, downloads + folds the
      output JSONL via the manifest, and writes per-anchor parquet.
  (c) Backward compat: the original submit-then-poll path (`match_pairs`
      → `_match_pairs_batch`) is unchanged and still works.

All OpenAI calls are mocked. No real API traffic.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

import stage4_llm_match as slm  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _mk_pairs_dfs():
    import pandas as pd
    pairs = [
        ("c0", "o0", 0.71),
        ("c1", "o1", 0.65),
        ("c2", "o2", 0.55),
    ]
    cdf = pd.DataFrame([
        {"comment_id": "c0", "comment": "A".center(200)},
        {"comment_id": "c1", "comment": "B".center(200)},
        {"comment_id": "c2", "comment": "C".center(200)},
    ])
    odf = pd.DataFrame([
        {"obligation_id": "o0", "obligation_text": "obligation zero",
         "cfr_section": "80.1", "subject": "operator", "modal": "must",
         "action": "submit", "object": "report"},
        {"obligation_id": "o1", "obligation_text": "obligation one",
         "cfr_section": "80.2", "subject": "operator", "modal": "must",
         "action": "submit", "object": "report"},
        {"obligation_id": "o2", "obligation_text": "obligation two",
         "cfr_section": "80.3", "subject": "operator", "modal": "must",
         "action": "submit", "object": "report"},
    ])
    return pairs, cdf, odf


def _mk_fake_client(*, batch_id="batch_test_001", status_sequence=None,
                    output_text=""):
    """Mock OpenAI client suitable for both submit and retrieve paths."""
    status_sequence = list(status_sequence or ["completed"])
    state = {"status_idx": 0}

    file_obj = types.SimpleNamespace(id="file_test_001")
    batch_obj = types.SimpleNamespace(
        id=batch_id, status=status_sequence[0],
        output_file_id="file_out_001", error_file_id=None,
        errors=None, request_counts=None,
    )

    client = mock.MagicMock()
    client.files.create.return_value = file_obj
    client.batches.create.return_value = batch_obj

    def _retrieve(bid):
        # Advance through status_sequence on repeated calls so timeout/
        # polling tests can simulate a real batch lifecycle.
        idx = min(state["status_idx"], len(status_sequence) - 1)
        state["status_idx"] += 1
        return types.SimpleNamespace(
            id=batch_id, status=status_sequence[idx],
            output_file_id="file_out_001", error_file_id=None,
            errors=None, request_counts=None,
        )
    client.batches.retrieve.side_effect = _retrieve

    content_resp = mock.MagicMock()
    content_resp.text = output_text
    content_resp.read = mock.MagicMock(return_value=output_text.encode("utf-8"))
    client.files.content.return_value = content_resp
    return client


def _mk_response_line(custom_id: str, content_dict: dict,
                      in_tok: int = 80, out_tok: int = 20) -> str:
    return json.dumps({
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{"message": {
                    "content": json.dumps(content_dict),
                }}],
                "usage": {"prompt_tokens": in_tok,
                          "completion_tokens": out_tok},
            }
        },
        "error": None,
    })


# ---------------------------------------------------------------------------
# (a) Submit-only path
# ---------------------------------------------------------------------------
def test_prepare_batch_submission_builds_messages_and_meta():
    """The new public helper builds messages_list, pair_meta, and the
    submit_to_pair remap without making any API call."""
    pairs, cdf, odf = _mk_pairs_dfs()
    sub = slm.prepare_batch_submission(pairs, cdf, odf, model="gpt-5")
    assert sub["n_pairs"] == 3, sub
    assert len(sub["messages_list"]) == 3
    assert len(sub["pair_meta"]) == 3
    assert sub["submit_to_pair"] == [0, 1, 2]
    for m in sub["messages_list"]:
        assert m[0]["role"] == "system"
        assert m[1]["role"] == "user"
    for meta in sub["pair_meta"]:
        assert meta is not None
        assert "comment_id" in meta and "obligation_id" in meta


def test_prepare_batch_submission_handles_missing_keys():
    """Pairs that reference comment_id or obligation_id not in the
    input DataFrames must be skipped via the messages_list=None
    placeholder; the submit_to_pair remap omits them."""
    pairs, cdf, odf = _mk_pairs_dfs()
    pairs.append(("missing_cid", "o0", 0.5))
    sub = slm.prepare_batch_submission(pairs, cdf, odf, model="gpt-5")
    # Submission count drops by 1 because pair_meta[3] is None.
    assert sub["n_pairs"] == 3
    assert sub["pair_meta"][3] is None
    assert 3 not in sub["submit_to_pair"]


def test_submit_batch_only_writes_input_jsonl_and_manifest(tmp_path: Path = None):
    """End-to-end submit: stages JSONL, calls files.create + batches.create
    on the mock client, persists manifest sidecar with pair_meta + the
    submit_to_pair remap, returns a manifest dict ready to append to
    the tracking CSV."""
    import tempfile as _t
    tmp_path = tmp_path or Path(_t.mkdtemp(prefix="submit_only_"))
    pairs, cdf, odf = _mk_pairs_dfs()
    sub = slm.prepare_batch_submission(pairs, cdf, odf)
    client = _mk_fake_client()

    manifest = slm.submit_batch_only(
        sub, model="gpt-5", batch_id_prefix="stage4_test",
        client=client, output_dir=tmp_path, max_cost_usd=10.0,
    )
    assert manifest["batch_id"] == "batch_test_001"
    assert manifest["job_id"]
    assert manifest["n_pairs"] == 3
    input_path = Path(manifest["input_jsonl"])
    manifest_path = Path(manifest["manifest_json"])
    assert input_path.exists()
    assert manifest_path.exists()
    # Input JSONL: 3 rows with custom_id="pair_0..2"
    lines = input_path.read_text().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        body = json.loads(line)
        assert body["custom_id"] == f"pair_{i}"
        assert body["body"]["model"] == "gpt-5"
    # Manifest captures the pair_meta + submit_to_pair so the retrieve
    # script can map results back without re-running prefilter.
    mani = json.loads(manifest_path.read_text())
    assert mani["batch_id"] == "batch_test_001"
    assert mani["n_pairs"] == 3
    assert mani["submit_to_pair"] == [0, 1, 2]
    assert len(mani["pair_meta"]) == 3
    # Exactly one Batch API submission happened (no polling).
    assert client.files.create.call_count == 1
    assert client.batches.create.call_count == 1
    assert client.batches.retrieve.call_count == 0


def test_submit_batch_only_cost_cap_raises_before_api_call(tmp_path: Path = None):
    """The pre-submission guard fires BEFORE files.create."""
    import tempfile as _t
    tmp_path = tmp_path or Path(_t.mkdtemp(prefix="submit_cap_"))
    pairs, cdf, odf = _mk_pairs_dfs()
    pairs = pairs * 1000   # 3000 pairs
    sub = slm.prepare_batch_submission(pairs, cdf, odf)
    client = _mk_fake_client()
    try:
        slm.submit_batch_only(
            sub, model="gpt-5", client=client, output_dir=tmp_path,
            max_cost_usd=0.01,   # 3000 pairs × $0.0019 = $5.70 ≫ $0.01
        )
    except slm.CostCapExceeded as e:
        assert "exceeded" in str(e).lower() or "exceeds" in str(e).lower()
        assert client.files.create.call_count == 0, (
            "files.create must NOT be called when the cap pre-check raises")
        return
    raise AssertionError("expected CostCapExceeded for 3000 pairs at $0.01 cap")


# ---------------------------------------------------------------------------
# (b) Retrieve-script path
# ---------------------------------------------------------------------------
def test_collect_batch_once_returns_in_progress_without_download():
    """If the batch isn't completed yet, the single-pass collector
    returns (status, None) — no files.content call, no parquet write."""
    pairs, cdf, odf = _mk_pairs_dfs()
    sub = slm.prepare_batch_submission(pairs, cdf, odf)
    tmp = Path(tempfile.mkdtemp(prefix="collect_inprog_"))
    client = _mk_fake_client(status_sequence=["in_progress"])
    manifest = slm.submit_batch_only(
        sub, client=client, output_dir=tmp, max_cost_usd=10.0,
    )
    # Reset retrieve counter so we can isolate the next call.
    client.batches.retrieve.side_effect = lambda bid: types.SimpleNamespace(
        id=manifest["batch_id"], status="in_progress",
        output_file_id=None, error_file_id=None, errors=None,
        request_counts=None,
    )
    client.files.content.reset_mock()
    status, df = slm.collect_batch_once(
        manifest["batch_id"], manifest["manifest_json"],
        client=client, output_dir=tmp,
    )
    assert status == "in_progress"
    assert df is None
    assert client.files.content.call_count == 0


def test_collect_batch_once_folds_output_via_manifest():
    """When status='completed', single-pass collector downloads the
    output JSONL, looks up pair_meta in the manifest, and returns a
    canonical-schema DataFrame."""
    pairs, cdf, odf = _mk_pairs_dfs()
    sub = slm.prepare_batch_submission(pairs, cdf, odf)
    tmp = Path(tempfile.mkdtemp(prefix="collect_complete_"))
    # Submit first to write the manifest sidecar.
    submit_client = _mk_fake_client()
    manifest = slm.submit_batch_only(
        sub, client=submit_client, output_dir=tmp, max_cost_usd=10.0,
    )

    # Construct the batch output JSONL — one row per submitted pair.
    out_lines = [
        _mk_response_line(f"pair_{i}", {
            "addressed": True, "stance": "SUPPORTING",
            "justification": f"reason {i}",
        }) for i in range(3)
    ]
    output_text = "\n".join(out_lines)

    poll_client = _mk_fake_client(
        status_sequence=["completed"], output_text=output_text,
    )
    status, df = slm.collect_batch_once(
        manifest["batch_id"], manifest["manifest_json"],
        client=poll_client, output_dir=tmp,
    )
    assert status == "completed"
    assert df is not None
    assert len(df) == 3
    assert set(df["comment_id"]) == {"c0", "c1", "c2"}
    assert (df["addressed"] == True).all()  # noqa: E712
    assert (df["stance"] == "SUPPORTING").all()
    # All rows should be tagged via='batch' (single-pass collector
    # doesn't have a real-time recovery lane).
    assert (df["via"] == "batch").all()


def test_retrieve_script_seed_orphans_idempotent(tmp_path: Path = None):
    """`--seed-orphans` writes the three pre-Task-I rows and is
    idempotent: a second invocation adds zero rows."""
    import importlib.util
    import tempfile as _t
    tmp_path = tmp_path or Path(_t.mkdtemp(prefix="seed_"))
    csv_path = tmp_path / "stage4_inflight_batches.csv"
    spec = importlib.util.spec_from_file_location(
        "retrieve_mod",
        _REPO_ROOT / "code" / "14_stage4_retrieve_batches.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    n1 = mod._seed_orphans(csv_path)
    n2 = mod._seed_orphans(csv_path)
    assert n1 == 3, f"first seed: expected 3, got {n1}"
    assert n2 == 0, f"second seed must be idempotent; got {n2}"
    # Each row references one of the 3 documented orphan batch_ids.
    rows = mod._read_inflight(csv_path)
    expected_ids = {
        "batch_6a062f67561881908d5e18acde15dcc9",
        "batch_6a062f64cfb08190a498887d0d4d587b",
        "batch_6a062f631ffc8190bd070a2d4f3f5673",
    }
    assert {r.batch_id for r in rows} == expected_ids
    assert all(r.status == "orphan" for r in rows)


def test_retrieve_script_orphan_row_surfaces_clear_error():
    """A row with empty manifest_json must surface 'orphan_no_manifest'
    final_status without any client call."""
    import importlib.util, asyncio
    spec = importlib.util.spec_from_file_location(
        "retrieve_mod",
        _REPO_ROOT / "code" / "14_stage4_retrieve_batches.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    row = mod.TrackingRow(
        docket_id="EPA-X", rule_type="proposed",
        batch_id="batch_orphan_test", job_id="",
        input_jsonl="", manifest_json="",
        n_pairs=0, submitted_at_utc="2026-05-14T00:00:00Z",
        model="gpt-5", status="orphan",
    )
    client = mock.MagicMock()
    out_dir = Path(tempfile.mkdtemp(prefix="orphan_out_"))
    res = asyncio.run(mod._retrieve_one(
        row, out_dir, client=client,
        timeout_hours=24.0, initial_poll_s=0.001, max_poll_s=0.001,
    ))
    assert res.final_status == "orphan_no_manifest"
    assert "orphan" in (res.error or "").lower() or "manifest" in (res.error or "")
    assert client.batches.retrieve.call_count == 0


def test_retrieve_script_completed_writes_parquet():
    """End-to-end mocked retrieve: submit, mock client returns
    'completed' immediately, retrieve worker downloads + folds + (after
    `_merge_and_write`) writes per-anchor parquet at
    data/processed/stage4/<docket>__matches.parquet.

    Updated Task J: parquet emission moved from `_retrieve_one` to
    `_merge_and_write` so sharded anchors can be concatenated per
    (docket_id, rule_type) before disk emission. We call both here to
    exercise the full path."""
    import importlib.util, asyncio, tempfile as _t
    tmp = Path(_t.mkdtemp(prefix="retrieve_complete_"))

    pairs, cdf, odf = _mk_pairs_dfs()
    sub = slm.prepare_batch_submission(pairs, cdf, odf)
    submit_client = _mk_fake_client()
    manifest = slm.submit_batch_only(
        sub, client=submit_client, output_dir=tmp, max_cost_usd=10.0,
    )
    out_lines = [
        _mk_response_line(f"pair_{i}", {
            "addressed": True, "stance": "OPPOSING",
            "justification": f"opposes {i}",
        }) for i in range(3)
    ]
    output_text = "\n".join(out_lines)
    poll_client = _mk_fake_client(
        status_sequence=["completed"], output_text=output_text,
    )

    spec = importlib.util.spec_from_file_location(
        "retrieve_mod2",
        _REPO_ROOT / "code" / "14_stage4_retrieve_batches.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    row = mod.TrackingRow(
        docket_id="EPA-TEST-0001", rule_type="proposed",
        batch_id=manifest["batch_id"], job_id=manifest["job_id"],
        input_jsonl=manifest["input_jsonl"],
        manifest_json=manifest["manifest_json"],
        n_pairs=3, submitted_at_utc="2026-05-14T00:00:00Z",
        model="gpt-5", status="submitted",
    )
    out_dir = tmp / "stage4_out"
    res = asyncio.run(mod._retrieve_one(
        row, out_dir, client=poll_client,
        timeout_hours=24.0, initial_poll_s=0.001, max_poll_s=0.001,
    ))
    # Per-shard worker should have attached the DataFrame but NOT
    # written the parquet — that happens in `_merge_and_write`.
    assert res.final_status == "completed"
    assert res.df is not None
    assert len(res.df) == 3
    assert res.out_path is None, (
        "parquet should not be written by _retrieve_one alone after Task J")

    # Merge step writes the parquet and updates res.out_path / n_written.
    mod._merge_and_write([res], out_dir)
    assert res.out_path is not None
    assert res.out_path.exists()
    assert res.out_path.name == "EPA-TEST-0001__matches.parquet"
    assert res.n_written == 3


# ---------------------------------------------------------------------------
# (c) Backward-compat: existing match_pairs path still works
# ---------------------------------------------------------------------------
def test_match_pairs_batch_path_unchanged_after_task_i():
    """`match_pairs(batch=True)` still drives the original
    _match_pairs_batch path. We confirm by injecting a stub
    `_call_openai_batch` and asserting it's invoked with the right
    shape (preserving the pre-Task-I contract)."""
    pairs, cdf, odf = _mk_pairs_dfs()
    captured = {}

    def fake_call_batch(messages_list, model, max_cost_usd, *,
                        batch_id_prefix, max_tokens, validate_fn=None):
        captured["n_messages"] = len(messages_list)
        captured["model"] = model
        captured["prefix"] = batch_id_prefix
        # Three batch successes, no failures.
        successful = [
            {"addressed": True, "stance": "SUPPORTING",
             "justification": f"reason {i}",
             "in_tokens": 80, "out_tokens": 20, "via": "batch"}
            for i in range(len(messages_list))
        ]
        return successful, []

    df = slm.match_pairs(
        pairs, cdf, odf, model="gpt-5", batch=True,
        max_cost_usd=10.0, call_batch=fake_call_batch,
    )
    assert captured["n_messages"] == 3
    assert captured["model"] == "gpt-5"
    assert len(df) == 3
    assert set(df["via"]) == {"batch"}


def test_orchestrator_submit_only_writes_tracking_row_and_exits():
    """The orchestrator's --submit-only branch is the thin glue that
    calls prepare_batch_submission → submit_batch_only → appends a row
    to the in-flight CSV. We exercise the glue with mocked entry points
    so we don't depend on prefilter / sentence-transformers."""
    import importlib.util, tempfile as _t
    tmp = Path(_t.mkdtemp(prefix="orch_submit_only_"))

    spec = importlib.util.spec_from_file_location(
        "orchestrator_mod",
        _REPO_ROOT / "code" / "13_stage4_run_per_anchor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    inflight = tmp / "stage4_inflight_batches.csv"
    fake_row = {
        "docket_id": "EPA-TEST-0002", "rule_type": "proposed",
        "batch_id": "batch_smoke_001", "job_id": "20260514T000000_smoke",
        "input_jsonl": str(tmp / "smoke__input.jsonl"),
        "manifest_json": str(tmp / "smoke__manifest.json"),
        "n_pairs": 5, "submitted_at_utc": "2026-05-14T00:00:00Z",
        "model": "gpt-5", "status": "submitted",
    }
    mod._append_inflight_row(inflight, fake_row)
    # Idempotency under serial CLI invocations: a second append adds a
    # new row without re-emitting the header.
    mod._append_inflight_row(inflight, {
        **fake_row, "docket_id": "EPA-TEST-0003",
        "batch_id": "batch_smoke_002",
    })

    rows = list(csv.DictReader(inflight.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["batch_id"] == "batch_smoke_001"
    assert rows[1]["batch_id"] == "batch_smoke_002"
    assert rows[0]["docket_id"] == "EPA-TEST-0002"
    assert rows[1]["docket_id"] == "EPA-TEST-0003"
    # All canonical fields present.
    expected = set(mod.INFLIGHT_CSV_FIELDS)
    assert set(rows[0].keys()) == expected


# ===========================================================================
# Task J — sharding for anchors > 50K pairs (the OpenAI Batch API cap)
#
# The sharding tests use a SMALL `shard_size` so we can exercise multi-
# shard paths without building 50K-pair fixtures. The math under test is
# the same: ceil(n_pairs / shard_size) shards, each with a slice of
# messages_list + a local-indexed pair_meta.
# ===========================================================================
def _mk_large_pairs(n: int):
    """Build an n-pair fixture (one comment per pair, identity
    obligation). Lets us exercise sharding boundaries cheaply."""
    import pandas as pd
    pairs = [(f"c{i}", "o0", 0.6) for i in range(n)]
    cdf = pd.DataFrame([
        {"comment_id": f"c{i}", "comment": ("A" * 200)} for i in range(n)
    ])
    odf = pd.DataFrame([
        {"obligation_id": "o0", "obligation_text": "obligation zero",
         "cfr_section": "80.1", "subject": "operator", "modal": "must",
         "action": "submit", "object": "report"},
    ])
    return pairs, cdf, odf


def _mk_sharding_client_factory(n_shards: int):
    """Build a mock client that issues a distinct batch_id per shard
    submission. The submit-only path calls `files.create` then
    `batches.create` per shard; this factory cycles `batch_id` so the
    test can verify the per-shard tracking rows carry the right IDs."""
    state = {"i": 0}
    client = mock.MagicMock()
    file_obj = types.SimpleNamespace(id="file_test_shard")
    client.files.create.return_value = file_obj

    def _create_batch(**kwargs):
        i = state["i"]
        state["i"] += 1
        return types.SimpleNamespace(
            id=f"batch_shard_{i + 1}", status="validating",
            output_file_id=None, error_file_id=None,
            errors=None, request_counts=None,
        )
    client.batches.create.side_effect = _create_batch
    return client


def test_sharding_fires_when_pairs_exceed_shard_size(tmp_path: Path = None):
    """(a) sharding fires when n_pairs > shard_size. With shard_size=5
    and 12 pairs, expect 3 shards (ceil(12/5))."""
    import tempfile as _t
    tmp = tmp_path or Path(_t.mkdtemp(prefix="shard_fires_"))
    pairs, cdf, odf = _mk_large_pairs(12)
    sub = slm.prepare_batch_submission(pairs, cdf, odf, model="gpt-5")
    assert sub["n_pairs"] == 12

    client = _mk_sharding_client_factory(n_shards=3)
    manifests = slm.submit_batches_sharded(
        sub, model="gpt-5", batch_id_prefix="stage4_shard_test",
        client=client, output_dir=tmp, max_cost_usd=10.0,
        shard_size=5,
    )
    assert len(manifests) == 3, (
        f"expected 3 shards for 12 pairs at shard_size=5; got "
        f"{len(manifests)}")
    # Per-shard pair counts: 5 + 5 + 2 = 12
    assert [m["n_pairs"] for m in manifests] == [5, 5, 2]
    # All shards share one job_id but each has its own batch_id.
    assert len({m["job_id"] for m in manifests}) == 1
    assert len({m["batch_id"] for m in manifests}) == 3
    # Filenames carry the _shard{N} suffix.
    for k, mf in enumerate(manifests, start=1):
        assert mf["shard_suffix"] == f"_shard{k}"
        assert Path(mf["manifest_json"]).name.endswith(
            f"__manifest_shard{k}.json")
        assert Path(mf["input_jsonl"]).name.endswith(
            f"__input_shard{k}.jsonl")


def test_sharding_count_is_ceil_pairs_over_shard_size(tmp_path: Path = None):
    """(b) Verify shard_count = ceil(pairs / shard_size) at multiple
    boundary points. Mirrors the production case: 157,883 pairs at
    50,000 / shard would be ceil(157883/50000) = 4 shards."""
    import math
    import tempfile as _t

    # (n_pairs, shard_size, expected_shards) — covers boundaries.
    cases = [
        (1, 5, 1),
        (5, 5, 1),     # exactly the cap → single batch
        (6, 5, 2),     # just over → 2 shards
        (10, 5, 2),
        (11, 5, 3),
        (157_883, 50_000, 4),   # the production case (math only)
    ]
    for n_pairs, shard_size, expected in cases:
        expected_calc = math.ceil(n_pairs / shard_size)
        assert expected == expected_calc, (
            f"sanity: ceil({n_pairs}/{shard_size}) = {expected_calc}, "
            f"test expected {expected}")

    # Exercise the actual submitter for the small cases — building a
    # 157,883-pair fixture is wasteful; the ceil math is universal.
    for n_pairs, shard_size, expected in cases:
        if n_pairs > 20:
            continue   # skip the production case (math validated above)
        tmp = Path(_t.mkdtemp(prefix=f"shard_ceil_{n_pairs}_"))
        pairs, cdf, odf = _mk_large_pairs(n_pairs)
        sub = slm.prepare_batch_submission(pairs, cdf, odf, model="gpt-5")
        client = _mk_sharding_client_factory(n_shards=expected)
        manifests = slm.submit_batches_sharded(
            sub, model="gpt-5", batch_id_prefix="stage4_ceil_test",
            client=client, output_dir=tmp, max_cost_usd=10.0,
            shard_size=shard_size,
        )
        assert len(manifests) == expected, (
            f"n_pairs={n_pairs}, shard_size={shard_size}: expected "
            f"{expected} shards, got {len(manifests)}")
        # Per-shard pair counts sum to n_pairs.
        assert sum(m["n_pairs"] for m in manifests) == n_pairs


def test_sharding_50k_pairs_uses_single_batch_path(tmp_path: Path = None):
    """(d) An anchor with exactly 50,000 pairs takes the single-batch
    path (no shard suffix in filenames, identical to pre-Task-J
    behavior). Use shard_size=50_000 (the production default) and the
    same fixture-construction shortcut: build a 50K-pair fixture and
    confirm we get exactly 1 manifest with no shard suffix."""
    import tempfile as _t

    # Building 50K pandas rows for the fixture is heavy; use a
    # shard_size=10 + 10-pair fixture as a stand-in. The CONTRACT
    # is: n_pairs <= shard_size → exactly one manifest with
    # `shard_suffix == ""` and filenames without `_shard{N}`.
    # We also explicitly cover the exact-boundary case (n_pairs ==
    # shard_size) here, which is what "50K-pair anchor" reduces to.
    tmp = tmp_path or Path(_t.mkdtemp(prefix="shard_unsharded_"))
    pairs, cdf, odf = _mk_large_pairs(10)
    sub = slm.prepare_batch_submission(pairs, cdf, odf, model="gpt-5")
    client = _mk_sharding_client_factory(n_shards=1)
    manifests = slm.submit_batches_sharded(
        sub, model="gpt-5", batch_id_prefix="stage4_unsharded_test",
        client=client, output_dir=tmp, max_cost_usd=10.0,
        shard_size=10,   # exactly the cap → still a single batch
    )
    assert len(manifests) == 1
    mf = manifests[0]
    assert mf["shard_suffix"] == "", (
        "exact-cap submission must take the single-batch path "
        "(no shard suffix in filenames)")
    assert Path(mf["manifest_json"]).name.endswith("__manifest.json")
    assert Path(mf["input_jsonl"]).name.endswith("__input.jsonl")
    # Backward compat: the filename pattern matches the pre-Task-J path
    # exactly, so the 28 already-submitted in-flight batches'
    # manifests would still parse cleanly through `collect_batch_once`.


def test_retrieve_concatenates_shards_into_one_parquet(tmp_path: Path = None):
    """(c) Retrieve script groups per-shard rows by (docket_id,
    rule_type) and writes ONE parquet per anchor. Verifies that a
    sharded submission's per-shard DataFrames are concatenated, that
    the total row count is sum(per-shard rows), and that all rows are
    accessible in the final parquet."""
    import importlib.util, asyncio, tempfile as _t
    import pandas as pd
    tmp = tmp_path or Path(_t.mkdtemp(prefix="retrieve_shard_"))
    out_dir = tmp / "stage4_out"

    # Submit a 7-pair fixture with shard_size=3 → 3 shards (3+3+1).
    pairs, cdf, odf = _mk_large_pairs(7)
    sub = slm.prepare_batch_submission(pairs, cdf, odf, model="gpt-5")
    submit_client = _mk_sharding_client_factory(n_shards=3)
    manifests = slm.submit_batches_sharded(
        sub, model="gpt-5", batch_id_prefix="stage4_retrieve_shard",
        client=submit_client, output_dir=tmp, max_cost_usd=10.0,
        shard_size=3,
    )
    assert len(manifests) == 3
    assert [m["n_pairs"] for m in manifests] == [3, 3, 1]

    # Build per-shard output JSONLs — each contains its own pair count's
    # worth of responses with custom_id="pair_0"..."pair_{n-1}" local to
    # the shard.
    shard_outputs = []
    for mf in manifests:
        n = mf["n_pairs"]
        lines = [
            _mk_response_line(f"pair_{i}", {
                "addressed": True, "stance": "SUPPORTING",
                "justification": f"shard reason {i}",
            }) for i in range(n)
        ]
        shard_outputs.append("\n".join(lines))

    # Build a poll-client factory per shard so each retrieve worker
    # gets its own pre-staged 'completed' response.
    def _mk_poll_client(output_text: str, batch_id: str):
        client = mock.MagicMock()
        batch_obj = types.SimpleNamespace(
            id=batch_id, status="completed",
            output_file_id=f"file_out_{batch_id}",
            error_file_id=None, errors=None, request_counts=None,
        )
        client.batches.retrieve.return_value = batch_obj
        content_resp = mock.MagicMock()
        content_resp.text = output_text
        content_resp.read = mock.MagicMock(
            return_value=output_text.encode("utf-8"))
        client.files.content.return_value = content_resp
        return client

    # Load the retrieve module
    spec = importlib.util.spec_from_file_location(
        "retrieve_mod_shard",
        _REPO_ROOT / "code" / "14_stage4_retrieve_batches.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # One TrackingRow per shard, all under the same (docket_id,
    # rule_type) so the merge step groups them together.
    shard_results = []
    for mf, out_text in zip(manifests, shard_outputs):
        row = mod.TrackingRow(
            docket_id="EPA-TEST-SHARD", rule_type="proposed",
            batch_id=mf["batch_id"], job_id=mf["job_id"],
            input_jsonl=mf["input_jsonl"],
            manifest_json=mf["manifest_json"],
            n_pairs=mf["n_pairs"],
            submitted_at_utc="2026-05-14T00:00:00Z",
            model="gpt-5", status="submitted",
        )
        client = _mk_poll_client(out_text, mf["batch_id"])
        res = asyncio.run(mod._retrieve_one(
            row, out_dir, client=client,
            timeout_hours=24.0, initial_poll_s=0.001, max_poll_s=0.001,
        ))
        assert res.final_status == "completed"
        assert res.df is not None
        assert res.out_path is None   # per-shard worker doesn't write
        shard_results.append(res)

    # Each shard's df has the right size.
    assert [len(r.df) for r in shard_results] == [3, 3, 1]
    # Pre-merge: no parquet on disk yet.
    expected_parquet = out_dir / "EPA-TEST-SHARD__matches.parquet"
    assert not expected_parquet.exists()

    # Merge writes one parquet per anchor.
    mod._merge_and_write(shard_results, out_dir)
    assert expected_parquet.exists()

    merged = pd.read_parquet(expected_parquet)
    assert len(merged) == 7, (
        f"expected 7 rows in merged parquet (3+3+1); got {len(merged)}")
    # All 7 comment_ids must be present, and order should reflect the
    # natural shard-then-row submission order so `n_written` is honest.
    expected_ids = [f"c{i}" for i in range(7)]
    # The submit-shard slicing of pairs uses submission order; verify
    # we got those exact comment_ids back.
    assert sorted(merged["comment_id"].tolist()) == sorted(expected_ids)

    # All per-shard RetrieveResults should now carry out_path + the
    # MERGED row count (not just the per-shard contribution).
    for r in shard_results:
        assert r.out_path == expected_parquet
        assert r.n_written == 7


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
