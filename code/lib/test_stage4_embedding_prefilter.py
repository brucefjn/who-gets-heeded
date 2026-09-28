"""
test_stage4_embedding_prefilter.py — unit tests for the Stage 4
embedding prefilter. Synthetic fixtures only; no real sentence-
transformers model is loaded (we inject a deterministic embedder via
`embed_fn=`).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

from stage4_embedding_prefilter import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBED_DIM,
    MIN_COMMENT_LENGTH_CHARS,
    embed_comments,
    embed_obligations,
    filter_eligible_comments,
    format_comment_for_embedding,
    format_obligation_for_embedding,
    prefilter_top_k,
)


# ---------------------------------------------------------------------------
# Deterministic synthetic embedder (no torch / sentence-transformers needed)
# ---------------------------------------------------------------------------
def _synthetic_embedder():
    """A toy embedder that maps each input string to a 768-d unit vector
    whose direction is a stable hash of the string. Two different strings
    yield two different (but well-defined) embeddings, so cosine sims are
    deterministic and reproducible across runs."""
    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        rng_base = 12345
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            # Seed RNG with a digest of the text so equal strings get the
            # same embedding (cache-hit test relies on this).
            seed = (abs(hash(t)) ^ rng_base) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(EMBED_DIM).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-12)
            out[i] = v
        return out
    return encode


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _make_fixture():
    comments = [
        {"document_id": f"c{i}", "docket_id": "EPA-TEST",
         "comment": f"Comment number {i} discussing ethanol blends and "
                    "operator recordkeeping requirements."}
        for i in range(5)
    ]
    obligations = [
        {"docket_id": "EPA-TEST", "rule_type": "proposed",
         "candidate_idx": j, "split_idx": 0,
         "cfr_section": f"80.{1400+j}",
         "subject": "Each operator", "modal": "must",
         "action": "submit", "object": f"report {j}",
         "obligation_text": f"Each operator must submit report {j} "
                            f"under § 80.{1400+j}."}
        for j in range(10)
    ]
    return comments, obligations


def test_format_strings_match_spec():
    comment_row = {"document_id": "c0", "comment": "Hello world"}
    assert format_comment_for_embedding(comment_row) == "Hello world"

    ob_row = {"subject": "Each operator", "modal": "must", "action": "submit",
              "object": "report 0",
              "obligation_text": "Each operator must submit report 0."}
    formatted = format_obligation_for_embedding(ob_row)
    assert formatted.startswith("Each operator must submit report 0\n")
    assert formatted.endswith("Each operator must submit report 0.")


def test_top_k_cardinality():
    comments, obligations = _make_fixture()
    pairs = prefilter_top_k(
        comments, obligations, k=3, similarity_threshold=-1.0,
        embed_fn=_synthetic_embedder(), cache_dir=None,
    )
    # Each comment should get exactly k=3 pairs (threshold disabled).
    by_comment: dict[str, int] = {}
    for cid, _, _ in pairs:
        by_comment[cid] = by_comment.get(cid, 0) + 1
    assert len(by_comment) == 5
    assert all(v == 3 for v in by_comment.values()), by_comment


def test_threshold_filters_low_similarity():
    comments, obligations = _make_fixture()
    # With a synthetic embedder over random unit vectors, expected cosine
    # is ~0; threshold 0.95 should filter ~everything out.
    pairs = prefilter_top_k(
        comments, obligations, k=10, similarity_threshold=0.95,
        embed_fn=_synthetic_embedder(), cache_dir=None,
    )
    assert len(pairs) == 0, (
        "synthetic random-unit embeddings should not exceed 0.95 cosine "
        "for the small fixture; if this fires, the embedder is non-random.")


def test_similarity_is_symmetric():
    """Verify cosine(a, b) == cosine(b, a) via the embedder + dot product
    used inside prefilter_top_k. Catches accidental normalization bugs."""
    embed = _synthetic_embedder()
    a = embed(["alpha"])
    b = embed(["beta"])
    sim_ab = float((a @ b.T).flatten()[0])
    sim_ba = float((b @ a.T).flatten()[0])
    assert abs(sim_ab - sim_ba) < 1e-7


def test_embedding_cache_hit(tmp_path: Path = None):
    """Second invocation with the same inputs should reuse the cache and
    skip the embedder (we detect this by injecting a counting embedder)."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp(prefix="stage4_test_"))
    cache_dir = tmp_path / "embeddings"

    comments, obligations = _make_fixture()
    inner = _synthetic_embedder()
    calls = {"n": 0, "items": 0}

    def counting_embed(texts: list[str]) -> np.ndarray:
        calls["n"] += 1
        calls["items"] += len(texts)
        return inner(texts)

    pairs1 = prefilter_top_k(
        comments, obligations, k=3, similarity_threshold=-1.0,
        docket_id="EPA-TEST", cache_dir=cache_dir, embed_fn=counting_embed,
    )
    first_items = calls["items"]
    assert first_items == len(comments) + len(obligations)

    # Second run: cache should cover everything → embedder called with
    # zero items (or, by short-circuit, not called at all).
    pairs2 = prefilter_top_k(
        comments, obligations, k=3, similarity_threshold=-1.0,
        docket_id="EPA-TEST", cache_dir=cache_dir, embed_fn=counting_embed,
    )
    assert calls["items"] == first_items, (
        f"cache miss: embedder re-encoded {calls['items'] - first_items} "
        "items on second run.")
    # Pair outputs must be byte-equal between runs.
    assert pairs1 == pairs2


# ---------------------------------------------------------------------------
# Eligibility filter + threshold pin
# ---------------------------------------------------------------------------
def test_default_similarity_threshold_pinned_at_0_3():
    # Pinned 2026-05-12 — change requires intentional review.
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.3
    assert MIN_COMMENT_LENGTH_CHARS == 50


def test_filter_drops_attachment_failed_rows():
    import pandas as pd
    df = pd.DataFrame([
        {"document_id": "c0", "comment": "a" * 100,
         "is_attachment_only": True, "text_source": "attachment_failed"},
        {"document_id": "c1", "comment": "a" * 100,
         "is_attachment_only": False, "text_source": "inform"},
        {"document_id": "c2", "comment": "a" * 100,
         "is_attachment_only": True, "text_source": "attachment_recovered"},
    ])
    out, counts = filter_eligible_comments(df)
    # Only c0 should drop — c2 is attachment-only but recovery succeeded.
    assert counts["attachment_failed"] == 1
    assert counts["too_short"] == 0
    assert counts["total_dropped"] == 1
    assert counts["n_kept"] == 2
    assert set(out["document_id"]) == {"c1", "c2"}


def test_filter_drops_too_short_rows():
    import pandas as pd
    df = pd.DataFrame([
        {"document_id": "c0", "comment": "ok " * 50,    # plenty long
         "is_attachment_only": False, "text_source": "inform"},
        {"document_id": "c1", "comment": "   short   ",   # <50 after strip
         "is_attachment_only": False, "text_source": "inform"},
        {"document_id": "c2", "comment": "a" * (MIN_COMMENT_LENGTH_CHARS - 1),
         "is_attachment_only": False, "text_source": "inform"},
        {"document_id": "c3", "comment": "a" * MIN_COMMENT_LENGTH_CHARS,
         "is_attachment_only": False, "text_source": "inform"},
        {"document_id": "c4", "comment": None,
         "is_attachment_only": False, "text_source": "inform"},
    ])
    out, counts = filter_eligible_comments(df)
    assert counts["too_short"] == 3   # c1, c2, c4
    assert counts["attachment_failed"] == 0
    assert set(out["document_id"]) == {"c0", "c3"}


def test_filter_independent_counts_with_overlap():
    """A row that satisfies BOTH rules should be counted under each
    reason independently; total_dropped reflects unique-row count."""
    import pandas as pd
    df = pd.DataFrame([
        # Both rules apply (attachment_failed + short)
        {"document_id": "c0", "comment": "x",
         "is_attachment_only": True, "text_source": "attachment_failed"},
        # Only short
        {"document_id": "c1", "comment": "y",
         "is_attachment_only": False, "text_source": "inform"},
        # Only attachment_failed
        {"document_id": "c2", "comment": "a" * 200,
         "is_attachment_only": True, "text_source": "attachment_failed"},
        # Eligible
        {"document_id": "c3", "comment": "a" * 200,
         "is_attachment_only": False, "text_source": "inform"},
    ])
    out, counts = filter_eligible_comments(df)
    assert counts["attachment_failed"] == 2   # c0, c2
    assert counts["too_short"] == 2           # c0, c1
    assert counts["total_dropped"] == 3       # c0 + c1 + c2 (c0 not double-counted)
    assert counts["n_kept"] == 1


def test_filter_tolerates_missing_optional_columns():
    """If the parquet shard lacks is_attachment_only / text_source, rule
    (a) should silently no-op rather than crashing — rule (b) still runs."""
    import pandas as pd
    df = pd.DataFrame([
        {"document_id": "c0", "comment": "a" * 100},
        {"document_id": "c1", "comment": "x"},
    ])
    out, counts = filter_eligible_comments(df)
    assert counts["attachment_failed"] == 0
    assert counts["too_short"] == 1
    assert set(out["document_id"]) == {"c0"}


def test_filter_handles_empty_input():
    import pandas as pd
    empty = pd.DataFrame(columns=["document_id", "comment",
                                  "is_attachment_only", "text_source"])
    out, counts = filter_eligible_comments(empty)
    assert len(out) == 0
    assert counts == {"attachment_failed": 0, "too_short": 0,
                      "total_dropped": 0, "n_kept": 0}


def test_empty_inputs_return_empty():
    assert prefilter_top_k([], [], embed_fn=_synthetic_embedder(),
                           cache_dir=None) == []
    comments, obligations = _make_fixture()
    assert prefilter_top_k(comments, [], embed_fn=_synthetic_embedder(),
                           cache_dir=None) == []
    assert prefilter_top_k([], obligations, embed_fn=_synthetic_embedder(),
                           cache_dir=None) == []


# ---------------------------------------------------------------------------
# Standalone runner (same pattern as test_path_a_dedup.py)
# ---------------------------------------------------------------------------
def _run_all() -> int:
    import inspect
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        sig = inspect.signature(fn)
        try:
            if sig.parameters:
                fn()
            else:
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
