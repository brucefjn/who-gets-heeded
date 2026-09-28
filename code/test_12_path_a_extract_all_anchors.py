"""
test_12_path_a_extract_all_anchors.py — unit tests for the Path A
Stage 2 production orchestrator.

The Stage 1a + Stage 1b entry points are injected via the orchestrator's
`run_orchestrator(stage1a_fn=..., stage1b_fn=..., write_csv_fn=...)`
kwargs so we can exercise the cost-cap, per-anchor failure resilience,
and skip-existing behavior without making any real LLM calls.

We load `12_path_a_extract_all_anchors.py` via `importlib.util` because
the filename starts with a digit (Python module names can't, but file
loaders don't care).
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))


def _load_orchestrator():
    """Load 12_path_a_extract_all_anchors.py by file path. Cached so
    successive tests don't reimport the spaCy stack via Stage 1a.

    The `sys.modules[spec.name] = mod` step is required: Python's
    dataclass machinery resolves type annotations by looking up the
    declaring module via `sys.modules.get(cls.__module__)`, and a
    spec_from_file_location load doesn't register the module by name
    unless we do it explicitly."""
    if "_stage2_module" in globals():
        return globals()["_stage2_module"]
    path = _REPO_ROOT / "code" / "12_path_a_extract_all_anchors.py"
    spec = importlib.util.spec_from_file_location("stage2_orch", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    globals()["_stage2_module"] = mod
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _mk_fixture(tmp_path: Path, n_anchors: int = 3) -> dict:
    """Build a self-contained fixture: an anchors CSV, an FR index CSV
    pointing at synthetic FR text files, and an output dir. Returns a
    dict of all the paths."""
    anchors_csv = tmp_path / "anchors.csv"
    fr_index_csv = tmp_path / "fr_index.csv"
    fr_dir = tmp_path / "data" / "raw" / "federal_register"
    output_dir = tmp_path / "data" / "processed"
    fr_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = [{"docket_id": f"EPA-TEST-{i:04d}"} for i in range(n_anchors)]
    _write_csv(anchors_csv, anchors, ["docket_id"])

    fr_rows = []
    for i in range(n_anchors):
        docket = f"EPA-TEST-{i:04d}"
        for rt in ("proposed", "final"):
            fr_text_path = fr_dir / f"{docket}_{rt}_doc{i}.txt"
            fr_text_path.write_text(
                f"synthetic FR text for {docket} {rt}", encoding="utf-8")
            fr_rows.append({
                "docket_id": docket, "rule_type": rt,
                "fr_doc_number": f"doc{i}",
                "publication_date": "2020-01-01",
                "title": f"Test {docket} {rt}",
                "subtype": "",
                "n_chars": "100",
                # Write the absolute path so the orchestrator's
                # repo-root-relative fallback isn't required in tests.
                "text_path": str(fr_text_path),
                "raw_text_url": "",
                "abstract": "",
            })
    _write_csv(fr_index_csv, fr_rows, list(fr_rows[0].keys()))
    return {
        "anchors_csv": anchors_csv,
        "fr_index_csv": fr_index_csv,
        "output_dir": output_dir,
        "tmp_path": tmp_path,
    }


def _ns(**kwargs) -> types.SimpleNamespace:
    """argparse.Namespace stand-in with the orchestrator's expected fields.
    Tests can override any subset."""
    base = dict(
        all_anchors=True, docket=None, rule_type="both",
        batch=True, skip_existing=False,
        max_cost=80.0, model="gpt-5",
        anchors_csv=None, fr_index=None, output_dir=None,
        dry_run=False,
        concurrency=1,   # Task H default for tests — backward compat
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _fake_extract(text_path, docket, rule_type):
    # Return a few synthetic candidate dataclass-like objects. Stage 1a's
    # `write_candidates_csv` accepts any iterable, but we replace it
    # too — so we don't need the real CandidateObligation shape here.
    return [
        types.SimpleNamespace(amendatory_action="revise", modal_strength="strong",
                              passive_voice=False, is_conditional=False)
        for _ in range(3)
    ]


def _fake_write_csv(cands, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("obligation_idx\n1\n2\n3\n", encoding="utf-8")
    return len(list(cands))


def _fake_verify(*, candidates_csv, binding_text_path, out_csv, provider, model,
                 **kwargs):
    """Test stub for run_primary_verification. Accepts and ignores
    `batch=` / `max_cost_usd=` (added in Task E) so existing tests don't
    break when the orchestrator passes them through."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("candidate_idx\n1\n2\n", encoding="utf-8")
    return {
        "candidates_processed": 3,
        "verified_rows_written": 2,
        "verified_candidates": 2,
        "rejected_candidates": 1,
        "compound_candidates": 0,
        "safety_net_hits": 0,
        "safety_net_verified": 0,
        "safety_net_dedup_dropped": 0,
        "safety_net_dedup_unmatched": 0,
        "total_in_tokens": 1500,
        "total_out_tokens": 200,
        "out_csv": str(out_csv),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_dry_run_plan_lists_all_targets(tmp_path: Path = None):
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage2_test_"))
    fx = _mk_fixture(tmp_path, n_anchors=3)
    mod = _load_orchestrator()

    args = _ns(
        all_anchors=True, dry_run=True, rule_type="both",
        anchors_csv=fx["anchors_csv"], fr_index=fx["fr_index_csv"],
        output_dir=fx["output_dir"],
    )
    results = mod.run_orchestrator(args)
    planned = [r for r in results if r.status == "dry-run"]
    # 3 anchors × 2 rule_types = 6 planned invocations
    assert len(planned) == 6, f"expected 6 planned, got {len(planned)}"
    assert {r.docket_id for r in planned} == {"EPA-TEST-0000",
                                              "EPA-TEST-0001", "EPA-TEST-0002"}
    assert {r.rule_type for r in planned} == {"proposed", "final"}


def test_dry_run_with_rule_type_final_only(tmp_path: Path = None):
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage2_test_"))
    fx = _mk_fixture(tmp_path, n_anchors=4)
    mod = _load_orchestrator()
    args = _ns(
        all_anchors=True, dry_run=True, rule_type="final",
        anchors_csv=fx["anchors_csv"], fr_index=fx["fr_index_csv"],
        output_dir=fx["output_dir"],
    )
    results = mod.run_orchestrator(args)
    planned = [r for r in results if r.status == "dry-run"]
    assert len(planned) == 4
    assert all(r.rule_type == "final" for r in planned)


def test_skip_existing_marks_existing_verified_csvs(tmp_path: Path = None):
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage2_test_"))
    fx = _mk_fixture(tmp_path, n_anchors=2)

    # Pre-create a verified CSV for one (docket, rule_type) combo.
    existing = (fx["output_dir"]
                / "path_a_obligations_verified_EPA-TEST-0001_final.csv")
    existing.write_text("candidate_idx\n", encoding="utf-8")

    mod = _load_orchestrator()
    args = _ns(
        all_anchors=True, dry_run=False, rule_type="both",
        skip_existing=True, max_cost=10.0,
        anchors_csv=fx["anchors_csv"], fr_index=fx["fr_index_csv"],
        output_dir=fx["output_dir"],
    )
    results = mod.run_orchestrator(
        args,
        stage1a_fn=_fake_extract,
        stage1b_fn=_fake_verify,
        write_csv_fn=_fake_write_csv,
    )

    skipped = [r for r in results if r.status == "skipped"]
    ran = [r for r in results if r.status == "ok"]
    # The (EPA-TEST-0001, final) combo should be skipped via --skip-existing.
    assert any(r.docket_id == "EPA-TEST-0001" and r.rule_type == "final"
               and "skip-existing" in r.reason for r in skipped), (
        f"expected skip-existing for EPA-TEST-0001/final; got skipped={skipped}")
    # The remaining 3 combos should have run.
    assert len(ran) == 3, f"expected 3 ran, got {len(ran)}: {[(r.docket_id, r.rule_type) for r in ran]}"


def test_per_anchor_failure_does_not_kill_run(tmp_path: Path = None):
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage2_test_"))
    fx = _mk_fixture(tmp_path, n_anchors=3)
    mod = _load_orchestrator()

    call_count = {"n": 0}
    def flaky_verify(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic Stage 1b failure")
        return _fake_verify(**kwargs)

    args = _ns(
        all_anchors=True, rule_type="both", max_cost=10.0,
        anchors_csv=fx["anchors_csv"], fr_index=fx["fr_index_csv"],
        output_dir=fx["output_dir"],
    )
    results = mod.run_orchestrator(
        args,
        stage1a_fn=_fake_extract,
        stage1b_fn=flaky_verify,
        write_csv_fn=_fake_write_csv,
    )
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "failed")
    # 6 targets total; one fails, five complete.
    assert n_fail == 1, f"expected 1 failure, got {n_fail}; results={results}"
    assert n_ok == 5, f"expected 5 ok, got {n_ok}"
    # Verify the failure record carries the error category.
    fail_rec = next(r for r in results if r.status == "failed")
    assert "synthetic Stage 1b failure" in fail_rec.reason
    # Verify subsequent anchors DID execute (call_count > the failure index).
    assert call_count["n"] >= 6, (
        f"expected verify to be called for all targets; got {call_count['n']}")


def test_max_cost_cap_raises_and_stops_remaining(tmp_path: Path = None):
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage2_test_"))
    fx = _mk_fixture(tmp_path, n_anchors=4)
    mod = _load_orchestrator()

    # Each Stage 1b call produces enough tokens to put cost over $0.01
    # at batched rates → cap of $0.005 should fire after the FIRST call's
    # post-check, before the 2nd target's pre-check.
    def expensive_verify(**kwargs):
        out_csv = kwargs["out_csv"]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("candidate_idx\n", encoding="utf-8")
        return {
            "verified_rows_written": 1,
            "total_in_tokens": 1_000_000,   # 1M in × $0.625 = $0.625 batched
            "total_out_tokens": 1_000_000,
        }

    args = _ns(
        all_anchors=True, rule_type="final", max_cost=0.005,
        anchors_csv=fx["anchors_csv"], fr_index=fx["fr_index_csv"],
        output_dir=fx["output_dir"],
    )
    from path_a_obligation_llm import CostCapExceeded
    raised = False
    try:
        mod.run_orchestrator(
            args,
            stage1a_fn=_fake_extract,
            stage1b_fn=expensive_verify,
            write_csv_fn=_fake_write_csv,
        )
    except CostCapExceeded as e:
        raised = True
        assert "0.005" in str(e) or "exceeded" in str(e).lower()
    assert raised, "expected CostCapExceeded"

    # After the cap fires on target #1, target #2 should NOT have an
    # output CSV (it never ran).
    second_target_csv = (fx["output_dir"]
                         / "path_a_obligations_verified_EPA-TEST-0001_final.csv")
    assert not second_target_csv.exists(), (
        "second anchor's verified CSV should not exist after cap fires")


def test_pick_canonical_fr_chooses_largest_n_chars():
    mod = _load_orchestrator()
    index = [
        {"docket_id": "EPA-X", "rule_type": "proposed",
         "fr_doc_number": "doc-small", "n_chars": "5000"},
        {"docket_id": "EPA-X", "rule_type": "proposed",
         "fr_doc_number": "doc-big", "n_chars": "300000"},
        {"docket_id": "EPA-X", "rule_type": "final",
         "fr_doc_number": "final-doc", "n_chars": "200000"},
    ]
    picked = mod._pick_canonical_fr(index, "EPA-X", "proposed")
    assert picked is not None
    assert picked["fr_doc_number"] == "doc-big", (
        "canonical picker must prefer the larger n_chars when multiple "
        "(docket, rule_type) entries exist")
    assert mod._pick_canonical_fr(index, "EPA-X", "final")["fr_doc_number"] == "final-doc"
    assert mod._pick_canonical_fr(index, "EPA-X", "no-such-type") is None
    assert mod._pick_canonical_fr(index, "EPA-MISSING", "final") is None


def test_missing_fr_doc_is_skipped_not_failed(tmp_path: Path = None):
    import tempfile
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage2_test_"))
    fx = _mk_fixture(tmp_path, n_anchors=2)

    # Inject an anchor with no FR-index entry — should appear as "skipped".
    extra_anchor_csv = tmp_path / "anchors_with_orphan.csv"
    _write_csv(extra_anchor_csv,
               [{"docket_id": "EPA-TEST-0000"},
                {"docket_id": "EPA-TEST-0001"},
                {"docket_id": "EPA-ORPHAN-9999"}],
               ["docket_id"])

    mod = _load_orchestrator()
    args = _ns(
        all_anchors=True, dry_run=True, rule_type="both",
        anchors_csv=extra_anchor_csv, fr_index=fx["fr_index_csv"],
        output_dir=fx["output_dir"],
    )
    results = mod.run_orchestrator(args)
    skipped = [r for r in results if r.status == "skipped"
               and r.docket_id == "EPA-ORPHAN-9999"]
    assert len(skipped) == 2, (
        "orphan docket should appear as skipped for both rule_types; "
        f"got {len(skipped)}")
    assert all("no entry in federal_register_index" in r.reason
               for r in skipped)


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
