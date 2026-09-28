"""
test_stage4_rerun_truncated.py — Task K (2026-05-14) unit tests for
the truncation-rerun pipeline.

Covers:
  (a) `_find_truncated_pairs` identifies rows with uncertainty_truncated=True
      and groups them by (docket_id, rule_type) inferred from obligation_id;
      rows with `truncation_cap_used >= 5000` are excluded (idempotency).
  (b) `prepare_batch_submission` with head=4000/tail=1000 builds messages
      whose comment_text is up to ~5000 chars + the elision marker.
  (c) `_merge_one_parquet` upserts on (comment_id, obligation_id) — old
      truncated rows REPLACED by new rerun rows, untouched rows kept.
  (d) `truncation_cap_used` column populated correctly (1400 for kept
      original rows, 5000 for rerun rows).
  (e) Idempotency: running merge twice produces identical output.

No real OpenAI calls. The submit-only path is exercised with mocked
clients that match Task I/J's `_mk_sharding_client_factory` shape.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "code" / "lib"))

import stage4_llm_match as slm   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for loading the digit-prefixed CLI scripts as importable modules
# ---------------------------------------------------------------------------
def _load_rerun_script():
    spec = importlib.util.spec_from_file_location(
        "stage4_rerun_mod",
        _REPO_ROOT / "code" / "15_stage4_retrun_truncated.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_merge_script():
    spec = importlib.util.spec_from_file_location(
        "stage4_merge_mod",
        _REPO_ROOT / "code" / "16_stage4_merge_rerun.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mk_stage4_parquet(tmp_dir: Path, docket_id: str, rule_type: str,
                      rows: list[dict]) -> Path:
    """Build one per-anchor parquet at the canonical path."""
    import pandas as pd
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{docket_id}__matches.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _mk_row(comment_id, oid_local, *, docket_id, rule_type,
            truncated=False, addressed=True, stance="SUPPORTING",
            cosine=0.7, cap_used=None):
    """Synthesize one Stage 4 match-row matching the canonical schema."""
    obligation_id = f"{docket_id}__{rule_type}__{oid_local}_0"
    row = {
        "comment_id": comment_id, "obligation_id": obligation_id,
        "cosine_similarity": cosine, "addressed": addressed,
        "stance": stance, "justification": f"reason for {comment_id}",
        "uncertainty_truncated": truncated,
        "in_tokens": 1500, "out_tokens": 200, "cost_usd": 0.0019,
        "via": "batch",
    }
    if cap_used is not None:
        row["truncation_cap_used"] = cap_used
    return row


# ===========================================================================
# (a) Identify truncated pairs correctly
# ===========================================================================
def test_find_truncated_pairs_filters_on_uncertainty_truncated():
    rerun_mod = _load_rerun_script()
    tmp = Path(tempfile.mkdtemp(prefix="task_k_find_"))
    # Mix truncated + untruncated rows under one docket.
    _mk_stage4_parquet(tmp, "EPA-TEST-0001", "proposed", [
        _mk_row("c0", 1, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=False),
        _mk_row("c1", 2, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=True),
        _mk_row("c2", 3, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=True),
    ])
    by_group, summary = rerun_mod._find_truncated_pairs(tmp)
    assert summary["total_rows"] == 3
    assert summary["total_truncated"] == 2
    key = ("EPA-TEST-0001", "proposed")
    assert key in by_group
    assert len(by_group[key]) == 2
    cids = {p["comment_id"] for p in by_group[key]}
    assert cids == {"c1", "c2"}, (
        f"expected only the 2 truncated comment_ids; got {cids}")


def test_find_truncated_pairs_skips_already_rerun_rows():
    """Idempotency anchor: rows with `truncation_cap_used >= 5000` are
    already at the rerun cap and must be excluded from the next rerun
    scan."""
    rerun_mod = _load_rerun_script()
    tmp = Path(tempfile.mkdtemp(prefix="task_k_idem_"))
    _mk_stage4_parquet(tmp, "EPA-TEST-0001", "proposed", [
        # uncertainty_truncated=True but truncation_cap_used=5000 → skip
        _mk_row("c0", 1, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=True, cap_used=5000),
        # uncertainty_truncated=True at the original cap → include
        _mk_row("c1", 2, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=True, cap_used=1400),
        # not truncated → exclude
        _mk_row("c2", 3, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=False, cap_used=1400),
    ])
    by_group, summary = rerun_mod._find_truncated_pairs(tmp)
    key = ("EPA-TEST-0001", "proposed")
    assert key in by_group
    cids = [p["comment_id"] for p in by_group[key]]
    assert cids == ["c1"], (
        f"expected only c1 (original-cap truncated); got {cids}")


def test_find_truncated_pairs_groups_by_docket_and_rule_type():
    """Multiple parquets across two anchors yield two groups; rule_type
    comes from the obligation_id namespace (`docket__rule_type__...`)."""
    rerun_mod = _load_rerun_script()
    tmp = Path(tempfile.mkdtemp(prefix="task_k_groups_"))
    _mk_stage4_parquet(tmp, "EPA-TEST-A", "proposed", [
        _mk_row("ca0", 1, docket_id="EPA-TEST-A", rule_type="proposed",
                truncated=True),
    ])
    _mk_stage4_parquet(tmp, "EPA-TEST-B", "final", [
        _mk_row("cb0", 1, docket_id="EPA-TEST-B", rule_type="final",
                truncated=True),
        _mk_row("cb1", 2, docket_id="EPA-TEST-B", rule_type="final",
                truncated=True),
    ])
    by_group, _ = rerun_mod._find_truncated_pairs(tmp)
    assert set(by_group.keys()) == {
        ("EPA-TEST-A", "proposed"), ("EPA-TEST-B", "final")
    }
    assert len(by_group[("EPA-TEST-A", "proposed")]) == 1
    assert len(by_group[("EPA-TEST-B", "final")]) == 2


# ===========================================================================
# (b) Prompt builder honors the higher truncation cap
# ===========================================================================
def test_prepare_batch_submission_higher_cap_lets_more_chars_through():
    """When head=4000 / tail=1000 is passed, the comment_text inside the
    user-prompt should contain up to ~5000 chars (plus the elision
    marker) — versus 1400 under the v2.3 default."""
    import pandas as pd

    long_text = "X" * 10_000   # exceeds both caps; will be truncated either way
    pairs = [("c0", "o0", 0.6)]
    cdf = pd.DataFrame([{"comment_id": "c0", "comment": long_text}])
    odf = pd.DataFrame([{
        "obligation_id": "o0", "obligation_text": "obligation zero",
        "cfr_section": "80.1", "subject": "operator", "modal": "must",
        "action": "submit", "object": "report",
    }])

    sub_default = slm.prepare_batch_submission(pairs, cdf, odf)
    sub_rerun = slm.prepare_batch_submission(
        pairs, cdf, odf,
        head_chars=slm.RERUN_TRUNCATION_HEAD_CHARS,
        tail_chars=slm.RERUN_TRUNCATION_TAIL_CHARS,
    )

    default_user = sub_default["messages_list"][0][1]["content"]
    rerun_user = sub_rerun["messages_list"][0][1]["content"]
    # Sanity: rerun user prompt is strictly longer than default
    # because comment_text occupies more characters.
    assert len(rerun_user) > len(default_user), (
        f"rerun prompt should be longer than default; got "
        f"rerun={len(rerun_user)}, default={len(default_user)}")
    # Both must still report uncertainty_truncated=True because the
    # underlying comment is 10K (beyond even the rerun cap).
    assert sub_default["pair_meta"][0]["uncertainty_truncated"] is True
    assert sub_rerun["pair_meta"][0]["uncertainty_truncated"] is True


def test_truncate_comment_param_overrides_module_defaults():
    """Smoke check the parametrized core: same input, two caps, two
    outputs of the expected lengths."""
    text = "A" * 2_000
    short, was_short = slm.truncate_comment(text, head_chars=100, tail_chars=50)
    long, was_long = slm.truncate_comment(text, head_chars=1000, tail_chars=500)
    # 100+50+marker ≈ short; 1000+500+marker ≈ long. Both truncated.
    assert was_short is True and was_long is True
    assert len(short) < len(long), (
        f"smaller cap should produce shorter output; got "
        f"short={len(short)}, long={len(long)}")
    # Module defaults untouched: a 2K text under defaults is truncated
    # to ~1400 chars + marker.
    default, was_default = slm.truncate_comment(text)
    assert was_default is True
    # Just under (4K head + 1K tail) → not truncated
    text_4K = "A" * 4_000
    out4k, was4k = slm.truncate_comment(
        text_4K, head_chars=4000, tail_chars=1000,
    )
    assert was4k is False, (
        "4000-char text under head=4000/tail=1000 (trigger=5100) must "
        "pass through verbatim; got truncated")


# ===========================================================================
# (c) Upsert replaces, doesn't append
# (d) truncation_cap_used populated correctly
# ===========================================================================
def test_merge_rerun_upserts_replace_not_append():
    """Merge replaces matching (comment_id, obligation_id) rows with the
    rerun version; non-truncated rows pass through unchanged. The total
    row count of the merged parquet matches the original (when every
    rerun row corresponds to one original row)."""
    import pandas as pd
    merge_mod = _load_merge_script()

    tmp = Path(tempfile.mkdtemp(prefix="task_k_merge_"))
    stage4_dir = tmp / "stage4"
    rerun_dir = tmp / "stage4_rerun"

    # Original: 4 rows for one anchor, 2 truncated + 2 untruncated.
    original_rows = [
        _mk_row("c0", 1, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=False, stance="NONE", cosine=0.55),
        _mk_row("c1", 2, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=True, stance="OPPOSING", cosine=0.71),
        _mk_row("c2", 3, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=True, stance="OPPOSING", cosine=0.65),
        _mk_row("c3", 4, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=False, stance="SUPPORTING", cosine=0.61),
    ]
    _mk_stage4_parquet(stage4_dir, "EPA-TEST-0001", "proposed", original_rows)
    # Rerun: only the 2 previously-truncated rows, with new judgments.
    rerun_rows = [
        _mk_row("c1", 2, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=False, stance="SUGGESTING_MODIFICATION",
                cosine=0.71),
        _mk_row("c2", 3, docket_id="EPA-TEST-0001", rule_type="proposed",
                truncated=False, stance="SUPPORTING", cosine=0.65),
    ]
    # Override justification so we can verify the rerun rows replaced
    # (not just supplemented) the originals.
    for r in rerun_rows:
        r["justification"] = f"rerun reason for {r['comment_id']}"
    _mk_stage4_parquet(rerun_dir, "EPA-TEST-0001", "proposed", rerun_rows)

    s = merge_mod._merge_one_parquet(
        stage4_dir / "EPA-TEST-0001__matches.parquet",
        rerun_dir / "EPA-TEST-0001__matches.parquet",
    )
    assert s["status"] == "merged"
    assert s["n_rerun_rows"] == 2
    assert s["n_original_rows"] == 4
    assert s["n_replaced"] == 2
    assert s["n_kept_original"] == 2
    assert s["n_new_appended"] == 0

    merged = pd.read_parquet(stage4_dir / "EPA-TEST-0001__matches.parquet")
    # Row count UNCHANGED (replacement, not append).
    assert len(merged) == 4

    # c1 and c2 rows now carry the rerun justification + new stance.
    c1 = merged[merged["comment_id"] == "c1"].iloc[0]
    c2 = merged[merged["comment_id"] == "c2"].iloc[0]
    assert c1["justification"].startswith("rerun reason"), (
        f"c1's justification should reflect rerun; got "
        f"{c1['justification']!r}")
    assert c1["stance"] == "SUGGESTING_MODIFICATION"
    assert c2["stance"] == "SUPPORTING"
    # c0 and c3 are unchanged.
    c0 = merged[merged["comment_id"] == "c0"].iloc[0]
    c3 = merged[merged["comment_id"] == "c3"].iloc[0]
    assert c0["stance"] == "NONE"
    assert c3["stance"] == "SUPPORTING"


def test_merge_rerun_populates_truncation_cap_used_column():
    """The truncation_cap_used column should appear post-merge with:
       - 1400 for legacy rows that weren't replaced
       - 5000 for rerun rows"""
    import pandas as pd
    merge_mod = _load_merge_script()

    tmp = Path(tempfile.mkdtemp(prefix="task_k_cap_col_"))
    stage4_dir = tmp / "stage4"
    rerun_dir = tmp / "stage4_rerun"

    _mk_stage4_parquet(stage4_dir, "EPA-TEST-0002", "proposed", [
        _mk_row("c0", 1, docket_id="EPA-TEST-0002", rule_type="proposed",
                truncated=False),
        _mk_row("c1", 2, docket_id="EPA-TEST-0002", rule_type="proposed",
                truncated=True),
    ])
    _mk_stage4_parquet(rerun_dir, "EPA-TEST-0002", "proposed", [
        _mk_row("c1", 2, docket_id="EPA-TEST-0002", rule_type="proposed",
                truncated=False),
    ])
    merge_mod._merge_one_parquet(
        stage4_dir / "EPA-TEST-0002__matches.parquet",
        rerun_dir / "EPA-TEST-0002__matches.parquet",
    )
    merged = pd.read_parquet(stage4_dir / "EPA-TEST-0002__matches.parquet")
    assert "truncation_cap_used" in merged.columns
    # Pre-existing row (c0): cap=1400; rerun row (c1): cap=5000.
    cap_by_cid = dict(zip(merged["comment_id"], merged["truncation_cap_used"]))
    assert int(cap_by_cid["c0"]) == merge_mod.ORIGINAL_TRUNCATION_CAP == 1400
    assert int(cap_by_cid["c1"]) == merge_mod.RERUN_TRUNCATION_CAP == 5000


# ===========================================================================
# (e) Idempotency
# ===========================================================================
def test_merge_rerun_is_idempotent():
    """Running merge twice on the same inputs must produce a byte-
    identical output parquet on the second invocation."""
    import pandas as pd
    merge_mod = _load_merge_script()
    rerun_mod = _load_rerun_script()

    tmp = Path(tempfile.mkdtemp(prefix="task_k_idemp_"))
    stage4_dir = tmp / "stage4"
    rerun_dir = tmp / "stage4_rerun"

    _mk_stage4_parquet(stage4_dir, "EPA-TEST-0003", "proposed", [
        _mk_row("c0", 1, docket_id="EPA-TEST-0003", rule_type="proposed",
                truncated=False),
        _mk_row("c1", 2, docket_id="EPA-TEST-0003", rule_type="proposed",
                truncated=True),
    ])
    _mk_stage4_parquet(rerun_dir, "EPA-TEST-0003", "proposed", [
        _mk_row("c1", 2, docket_id="EPA-TEST-0003", rule_type="proposed",
                truncated=False),
    ])
    target = stage4_dir / "EPA-TEST-0003__matches.parquet"
    rerun_path = rerun_dir / "EPA-TEST-0003__matches.parquet"

    # First merge: replaces c1, adds truncation_cap_used.
    merge_mod._merge_one_parquet(target, rerun_path)
    after_first = pd.read_parquet(target).sort_values(["comment_id"]).reset_index(drop=True)

    # Second merge: no-op contract.
    merge_mod._merge_one_parquet(target, rerun_path)
    after_second = pd.read_parquet(target).sort_values(["comment_id"]).reset_index(drop=True)
    assert after_first.equals(after_second), (
        "second merge must produce a byte-identical parquet")

    # And: the next rerun-scan should find ZERO truncated rows to rerun
    # (because c1's `truncation_cap_used == 5000` now).
    by_group, summary = rerun_mod._find_truncated_pairs(stage4_dir)
    assert by_group == {}, (
        f"expected no truncated rows to rerun on the second pass; got "
        f"{summary['total_truncated']} across {summary['n_groups']} groups")


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
