"""
test_compute_obligation_outcomes.py — Task L unit tests for
`code/17_compute_obligation_outcomes.py`.

Covers:
  (a) high-similarity proposed-final pair with same cfr_section →
      SURVIVED-edited (and ≥0.95 + same cfr → SURVIVED-unchanged)
  (b) medium-similarity → MODIFIED (with both same-cfr and different-cfr
      variants — the latter is a documented spec interpretation)
  (c) no match → DROPPED (cosine < 0.55 to every final)
  (d) final-only obligation → NEW (max cosine to any proposed < 0.55)
  (e) reuses cached embeddings when available (content_hash hits skip
      embed_fn invocation)
  (f) handles missing or empty docket CSVs gracefully (returns
      status='skipped' with a clear reason and no exceptions)

The script is loaded by file path because its filename starts with a
digit (`17_compute_...`). Tests inject a deterministic synthetic
`embed_fn` so they don't load the sentence-transformer model.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

# Reuse the prefilter's canonical EMBED_DIM so the synthetic embeddings
# match the shape `embed_obligations` enforces.
from stage4_embedding_prefilter import EMBED_DIM  # noqa: E402


def _load_outcomes_script():
    spec = importlib.util.spec_from_file_location(
        "outcomes_mod",
        _REPO_ROOT / "code" / "17_compute_obligation_outcomes.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic-embedding factory
#
# Maps each unique text to one of a small set of "concept" vectors in
# the EMBED_DIM space. Same concept → cosine == 1.0; different concepts
# → small overlap so we can control the cosine via the partial-overlap
# trick (linear combinations of concept basis vectors).
# ---------------------------------------------------------------------------
def _make_concept_embeddings(rng_seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    concepts: dict[str, np.ndarray] = {}
    # 8 distinct "concept" basis vectors — orthogonalized via QR so any
    # pair has near-zero cosine, allowing us to mix them for controlled
    # similarities.
    basis_raw = rng.standard_normal(size=(8, EMBED_DIM)).astype(np.float32)
    q, _ = np.linalg.qr(basis_raw.T)
    basis = q.T[:8]
    for i in range(8):
        v = basis[i].astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        concepts[f"concept_{i}"] = v
    return concepts


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def _make_text_to_vec(concepts: dict[str, np.ndarray]):
    """Return a callable that maps a text → unit vector.

    Test inputs use the convention:
        "mix:<concept_a>:<alpha>:<concept_b>:<beta>"
    → unit-normalized (alpha * vec_a + beta * vec_b).

    Plain texts ("foo", "bar") hash to a stable random vector so unknown
    texts can also be embedded without polluting the controlled cases.
    """
    text_cache: dict[str, np.ndarray] = {}

    def _embed_one(text: str) -> np.ndarray:
        if text in text_cache:
            return text_cache[text]
        if text.startswith("mix:"):
            parts = text.split(":")
            # mix:<a>:<alpha>:<b>:<beta>
            a, alpha, b, beta = parts[1], float(parts[2]), parts[3], float(parts[4])
            v = alpha * concepts[a] + beta * concepts[b]
            text_cache[text] = _l2_norm(v)
            return text_cache[text]
        if text in concepts:
            text_cache[text] = concepts[text].astype(np.float32)
            return text_cache[text]
        # Stable random vector for any uncontrolled text.
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.standard_normal(size=(EMBED_DIM,)).astype(np.float32)
        text_cache[text] = _l2_norm(v)
        return text_cache[text]

    def embed_fn(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.stack([_embed_one(t) for t in texts], axis=0)

    return embed_fn


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
_OBLIGATIONS_FIELDS = [
    "candidate_idx", "docket_id", "rule_type", "split_idx",
    "cfr_part", "cfr_section", "subject", "modal", "modal_strength",
    "action", "object", "amendatory_action", "is_conditional",
    "condition_text", "passive_voice", "obligation_text",
    "cross_reference", "found_via", "verified", "verified_reason",
    "complexity", "recital_disposition", "llm_provider", "llm_model",
    "in_tokens", "out_tokens",
]


def _write_verified_csv(path: Path, docket_id: str, rule_type: str,
                       rows: list[dict]) -> Path:
    """Write a Stage 1b-shaped verified CSV.

    NOTE: the prefilter's `format_obligation_for_embedding` prepends
    "{subject} {modal} {action} {object}\\n" to `obligation_text`
    before embedding. To make the synthetic-embedding tests control the
    cosine score directly via `obligation_text`, we leave the structured
    fields empty — the formatter then returns only `obligation_text`,
    and the test's `embed_fn` can key off that string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    full_rows = []
    for r in rows:
        full = {f: "" for f in _OBLIGATIONS_FIELDS}
        full.update({
            "candidate_idx": str(r.get("candidate_idx", 1)),
            "docket_id": docket_id, "rule_type": rule_type,
            "split_idx": str(r.get("split_idx", 0)),
            "cfr_section": r.get("cfr_section", ""),
            "obligation_text": r.get("obligation_text", ""),
            # Structured fields left empty so the formatter returns
            # `obligation_text` verbatim (see NOTE in docstring).
            "subject": r.get("subject", ""),
            "modal": r.get("modal", ""),
            "modal_strength": r.get("modal_strength", ""),
            "action": r.get("action", ""),
            "object": r.get("object", ""),
            "amendatory_action": r.get("amendatory_action", "revise"),
            "verified": "True",
        })
        full_rows.append(full)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_OBLIGATIONS_FIELDS)
        w.writeheader()
        w.writerows(full_rows)
    return path


def _mk_fixture(docket_id: str = "EPA-TEST-0001"):
    """Return (tmp_dir_path, concepts) — a tempdir with `data/processed/`
    populated, plus the concept-embedding dict the caller uses to drive
    cosine scores."""
    tmp = Path(tempfile.mkdtemp(prefix="task_l_"))
    obligations_dir = tmp / "data" / "processed"
    obligations_dir.mkdir(parents=True, exist_ok=True)
    concepts = _make_concept_embeddings()
    return tmp, obligations_dir, concepts


# ---------------------------------------------------------------------------
# (a) High-similarity → SURVIVED-edited / SURVIVED-unchanged
# ---------------------------------------------------------------------------
def test_high_similarity_same_cfr_yields_survived_edited():
    """Proposed and final share cfr_section + carry a 0.90-cosine text
    pair → SURVIVED-edited per spec."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)

    # alpha=0.9 + beta=sqrt(1 - 0.9^2) gives unit vector with cos=0.9 to
    # concept_0.
    import math
    beta = math.sqrt(1 - 0.9 ** 2)
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": f"mix:concept_0:0.9:concept_7:{beta}"},
        ])

    rows, summary = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn,
        use_cache=False,
    )
    assert summary.status == "ok"
    proposed = [r for r in rows if r["outcome_state"] != "NEW"]
    assert len(proposed) == 1
    assert proposed[0]["outcome_state"] == "SURVIVED-edited", (
        f"expected SURVIVED-edited at cos~0.9 + same cfr; got "
        f"{proposed[0]['outcome_state']} (cos={proposed[0]['best_cosine']:.3f})")
    assert proposed[0]["matched_final_obligation_id"] is not None


def test_perfect_similarity_same_cfr_yields_survived_unchanged():
    """Cos≥0.95 + same cfr → SURVIVED-unchanged."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},   # identical → cos=1.0
        ])
    rows, _ = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    proposed = [r for r in rows if r["outcome_state"] != "NEW"]
    assert proposed[0]["outcome_state"] == "SURVIVED-unchanged"


# ---------------------------------------------------------------------------
# (b) Medium-similarity → MODIFIED
# ---------------------------------------------------------------------------
def test_medium_similarity_yields_modified():
    """cosine in [0.55, 0.85) → MODIFIED, regardless of cfr_section."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)
    import math
    alpha = 0.7   # well within [0.55, 0.85)
    beta = math.sqrt(1 - alpha ** 2)
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": f"mix:concept_0:{alpha}:concept_5:{beta}"},
        ])
    rows, _ = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    proposed = [r for r in rows if r["outcome_state"] != "NEW"]
    assert proposed[0]["outcome_state"] == "MODIFIED"
    assert abs(proposed[0]["best_cosine"] - alpha) < 1e-3


def test_high_similarity_different_cfr_section_is_modified_not_survived():
    """Strong text similarity (≥0.85) but a DIFFERENT cfr_section is
    not SURVIVED-* (which gates on same cfr) — interpreted as MODIFIED.
    This documents the script's resolution of the spec's mid-band gap
    around 0.85-0.95 cosine + cross-section moves."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)
    import math
    alpha = 0.9
    beta = math.sqrt(1 - alpha ** 2)
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            # text strongly matches but the section moved
            {"candidate_idx": 1, "cfr_section": "80.99",
             "obligation_text": f"mix:concept_0:{alpha}:concept_7:{beta}"},
        ])
    rows, _ = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    proposed = [r for r in rows if r["outcome_state"] != "NEW"]
    assert proposed[0]["outcome_state"] == "MODIFIED", (
        f"expected MODIFIED for 0.9 cosine + different cfr; got "
        f"{proposed[0]['outcome_state']}")


# ---------------------------------------------------------------------------
# (c) No-match → DROPPED
# ---------------------------------------------------------------------------
def test_no_match_yields_dropped():
    """Proposed obligation whose closest final is below the 0.55 floor
    → DROPPED with matched_final_obligation_id=None."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            # Orthogonal concept → cos ≈ 0
            {"candidate_idx": 1, "cfr_section": "80.99",
             "obligation_text": "concept_7"},
        ])
    rows, _ = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    proposed = [r for r in rows if r["outcome_state"] != "NEW"]
    assert proposed[0]["outcome_state"] == "DROPPED"
    assert proposed[0]["matched_final_obligation_id"] is None
    assert proposed[0]["best_cosine"] < 0.55


# ---------------------------------------------------------------------------
# (d) Final-only obligation → NEW
# ---------------------------------------------------------------------------
def test_final_only_obligation_emits_new_row():
    """A final obligation whose max cosine to any proposed is < 0.55
    emits a row with obligation_id=final_id, outcome_state=NEW,
    matched_final_obligation_id=None."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},  # SURVIVED-unchanged
            # A SECOND final obligation has no analog in proposed.
            {"candidate_idx": 2, "cfr_section": "80.42",
             "obligation_text": "concept_4"},
        ])
    rows, summary = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    new_rows = [r for r in rows if r["outcome_state"] == "NEW"]
    assert len(new_rows) == 1, (
        f"expected exactly one NEW row; got {len(new_rows)}")
    assert new_rows[0]["matched_final_obligation_id"] is None
    # The NEW row's obligation_id should be the FINAL side.
    assert "__final__" in new_rows[0]["obligation_id"]
    assert summary.state_counts["NEW"] == 1
    assert summary.state_counts["SURVIVED-unchanged"] == 1


# ---------------------------------------------------------------------------
# (e) Cached embeddings reuse
# ---------------------------------------------------------------------------
def test_reuses_cached_embeddings_when_available():
    """A second invocation hits the cache and does NOT call embed_fn.
    The hash check inside `_embed_with_cache` uses the obligation_text
    content hash, so identical inputs short-circuit to cached vectors."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    cache_dir = tmp / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    call_log = {"texts_passed": []}
    base_fn = _make_text_to_vec(concepts)

    def counting_embed_fn(texts):
        call_log["texts_passed"].extend(texts)
        return base_fn(texts)

    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_proposed.csv",
        "EPA-TEST-0001", "proposed", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-TEST-0001_final.csv",
        "EPA-TEST-0001", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])

    # First pass: cold cache → embed_fn called for every unique text.
    _, _ = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        cache_dir=cache_dir,
        embed_fn=counting_embed_fn,
    )
    first_pass_calls = list(call_log["texts_passed"])
    # We expect at least one cold call. The same text appears in both
    # proposed and final but cache only stores per id+hash, and the
    # ids differ across rule_type so two distinct entries.
    assert len(first_pass_calls) >= 1, (
        f"first pass should have called embed_fn at least once; got "
        f"{len(first_pass_calls)}")

    # Second pass: warm cache → no new embed_fn calls.
    call_log["texts_passed"] = []
    _, _ = mod.compute_outcomes_for_docket(
        "EPA-TEST-0001",
        obligations_dir=obligations_dir,
        cache_dir=cache_dir,
        embed_fn=counting_embed_fn,
    )
    assert call_log["texts_passed"] == [], (
        f"second pass should hit cache for every id; embed_fn got "
        f"{call_log['texts_passed']!r}")


# ---------------------------------------------------------------------------
# (f) Missing or empty docket CSVs → graceful skip
# ---------------------------------------------------------------------------
def test_handles_missing_or_empty_csvs_gracefully():
    """When either the proposed or final CSV is missing OR exists but
    has zero rows, compute_outcomes_for_docket returns status='skipped'
    with a clear reason and no rows. No exception."""
    mod = _load_outcomes_script()
    tmp, obligations_dir, concepts = _mk_fixture()
    embed_fn = _make_text_to_vec(concepts)

    # Case 1: BOTH missing.
    rows, summary = mod.compute_outcomes_for_docket(
        "EPA-DOES-NOT-EXIST",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    assert rows == []
    assert summary.status == "skipped"
    assert "neither" in summary.reason or "missing" in summary.reason

    # Case 2: proposed missing, final present.
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-FINAL-ONLY_final.csv",
        "EPA-FINAL-ONLY", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    rows, summary = mod.compute_outcomes_for_docket(
        "EPA-FINAL-ONLY",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    assert rows == []
    assert summary.status == "skipped"
    assert "proposed" in summary.reason.lower()

    # Case 3: empty proposed CSV (header only, 0 data rows).
    empty_proposed = (obligations_dir
                      / "path_a_obligations_verified_EPA-EMPTY-PROP_proposed.csv")
    _write_verified_csv(empty_proposed, "EPA-EMPTY-PROP", "proposed", [])
    _write_verified_csv(
        obligations_dir / "path_a_obligations_verified_EPA-EMPTY-PROP_final.csv",
        "EPA-EMPTY-PROP", "final", [
            {"candidate_idx": 1, "cfr_section": "80.1",
             "obligation_text": "concept_0"},
        ])
    rows, summary = mod.compute_outcomes_for_docket(
        "EPA-EMPTY-PROP",
        obligations_dir=obligations_dir,
        embed_fn=embed_fn, use_cache=False,
    )
    assert rows == []
    assert summary.status == "skipped"
    assert "empty" in summary.reason.lower() or "missing" in summary.reason.lower()


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
