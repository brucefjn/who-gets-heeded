"""
stage4_llm_match.py — Stage 4 LLM matching pass for Path A.

For each (comment, obligation) candidate pair from the embedding
prefilter, ask gpt-5 three questions per the Stage 4 sub-spec (notes
/2026-05-07_stage1_rubric_design.md §549–580):

  1. addressed (bool)              — does the comment substantively
                                     engage with this specific obligation?
  2. stance (closed 4-value enum)  — SUPPORTING / OPPOSING
                                     / SUGGESTING_MODIFICATION / NONE
  3. justification (one sentence)  — comment span justifying the call.

Per PROJECT_FACTS §7, the model is locked at gpt-5 — Stage 4 is regression
input, so a weaker model adds noise to the downstream causal estimates.

Call infrastructure (JSON output, repair-once retry policy, rate-limit
backoff) is delegated to `path_a_obligation_llm._call_openai_json` /
`_parse_json_with_repair` so we don't reimplement them. Cost is computed
per-call against batched gpt-5 pricing (PROJECT_FACTS §7):
  $0.625 / 1M input tokens, $5 / 1M output tokens.
A hard `max_cost_usd` guardrail raises before the cap is exceeded.

Comment-text truncation: a long comment is truncated to first 1,200 +
last 200 characters when the raw length exceeds 1,500 (per Stage 4 spec).
The output DataFrame carries an `uncertainty_truncated` flag so
downstream analysis can stratify on truncation.

Public entry points:
  match_pairs(pairs, comments_df, obligations_df, model, batch, max_cost_usd)
      → pd.DataFrame
  estimate_cost(n_pairs) → float
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

# Reuse Stage 1b call infrastructure — same retry policy, same JSON repair,
# same rate-limit handling. Do not reimplement.
from path_a_obligation_llm import (  # type: ignore  (top-level path injection by orchestrator)
    CostCapExceeded,
    DEFAULT_OPENAI_MODEL,
    GPT5_STANDARD_INPUT_PRICE_PER_M,
    GPT5_STANDARD_OUTPUT_PRICE_PER_M,
    PATH_A_DEBUG,
    _call_openai_batch,
    _call_openai_json,
    _parse_json_with_repair,
)


# Closed enums per Stage 4 spec. Out-of-enum values raise.
ALLOWED_STANCES = {
    "SUPPORTING", "OPPOSING", "SUGGESTING_MODIFICATION", "NONE",
}

# Pricing per PROJECT_FACTS §7 (gpt-5 batched). Real-time is ~2x; cost
# accounting uses the production-target batched numbers so a dry-run
# estimate aligns with the budget in the spec.
GPT5_BATCH_PRICE_PER_M_INPUT = 0.625
GPT5_BATCH_PRICE_PER_M_OUTPUT = 5.0

# Per-call cost anchor used by `estimate_cost`. PROJECT_FACTS §7 quotes
# $0.0019 per call assuming 1.5K in + 200 out — anchor the dry-run
# estimate to that.
ESTIMATED_PER_CALL_USD = 0.0019

# Match-call output budget. The JSON shape is tiny so 400 tokens is more
# than enough; we keep some headroom for the `justification` string.
MATCH_MAX_TOKENS = 400

# Truncation policy locked v2.3 (Stage 4 sub-spec, distinct from the
# Layer C 10K/2K policy for full-corpus coding).
TRUNCATION_TRIGGER_CHARS = 1500
TRUNCATION_HEAD_CHARS = 1200
TRUNCATION_TAIL_CHARS = 200

# Task K rerun (2026-05-14): for the truncation-rerun pipeline that
# surgically re-processes the 18.77% of pairs whose comments were
# truncated under the v2.3 cap, the head/tail are raised to give the
# matcher more of the comment body. Used as the recommended preset by
# `code/15_stage4_retrun_truncated.py`; downstream merge writes 5000 to
# the per-row `truncation_cap_used` audit column.
RERUN_TRUNCATION_HEAD_CHARS = 4000
RERUN_TRUNCATION_TAIL_CHARS = 1000


# ---------------------------------------------------------------------------
# Prompt (compact ≤200 words — locked v2.3 per PROJECT_FACTS §9b
# methodology-format finding)
# ---------------------------------------------------------------------------
MATCH_SYSTEM_PROMPT = """\
You are evaluating whether a public comment substantively addresses a specific
regulatory obligation in a U.S. EPA proposed rule. Return JSON only — no prose,
no markdown.
"""


MATCH_USER_TEMPLATE = """\
OBLIGATION:
  cfr_section: {cfr_section}
  text: {obligation_text}
  parsed: actor="{subject}", modal="{modal}", action="{action}", object="{object}"

COMMENT (excerpt, may be truncated to first 1200 + last 200 chars):
{comment_text}

Return a JSON object with these fields:
{{
  "addressed": true | false,
    // true if the comment substantively engages with this specific
    // obligation: cites it, requests change to it, or supports/opposes it
    // as a discrete provision. False if the comment is about a different
    // obligation, the rule generally, or off-topic.
  "stance": "SUPPORTING" | "OPPOSING" | "SUGGESTING_MODIFICATION" | "NONE",
    // SUPPORTING: comment endorses this obligation as written.
    // OPPOSING: comment opposes this obligation; wants it removed/blocked.
    // SUGGESTING_MODIFICATION: comment proposes specific changes to it.
    // NONE: addressed=false, OR addressed=true but stance is unclear.
  "justification": "<one sentence, <=200 chars, plain English>"
    // Quote or paraphrase the comment span that justifies the addressed/stance call.
}}

Output JSON only.
"""


# ---------------------------------------------------------------------------
# Comment-text truncation
# ---------------------------------------------------------------------------
def truncate_comment(
    text: str,
    *,
    head_chars: Optional[int] = None,
    tail_chars: Optional[int] = None,
    trigger_chars: Optional[int] = None,
) -> tuple[str, bool]:
    """Return (text_for_prompt, was_truncated). Long comments keep the
    first head_chars and the last tail_chars joined by an explicit
    elision marker so the model can tell the middle was omitted.

    `head_chars` / `tail_chars` / `trigger_chars` default to the
    module-level v2.3 constants (1200/200/1500). Callers may override
    them to enlarge the budget on a per-call basis — Task K's rerun
    pipeline uses head=4000, tail=1000 to recover the 18.77% of pairs
    truncated under the v2.3 cap.

    `trigger_chars` (default = head_chars + tail_chars + a small safety
    margin equal to the original 100-char gap) is the length at which
    truncation kicks in. Texts shorter than this threshold pass through
    verbatim with was_truncated=False.
    """
    if head_chars is None:
        head_chars = TRUNCATION_HEAD_CHARS
    if tail_chars is None:
        tail_chars = TRUNCATION_TAIL_CHARS
    if trigger_chars is None:
        # Preserve the original v2.3 invariant `trigger == head + tail +
        # 100` for the default-arg path so existing behavior is
        # byte-identical (1200 + 200 + 100 = 1500).
        trigger_chars = head_chars + tail_chars + 100
    if not text:
        return "", False
    if len(text) <= trigger_chars:
        return text, False
    head = text[:head_chars]
    tail = text[-tail_chars:]
    return f"{head}\n[...middle elided for length...]\n{tail}", True


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------
def call_cost_usd(in_tokens: int, out_tokens: int) -> float:
    return (
        in_tokens * GPT5_BATCH_PRICE_PER_M_INPUT / 1_000_000.0
        + out_tokens * GPT5_BATCH_PRICE_PER_M_OUTPUT / 1_000_000.0
    )


def estimate_cost(n_pairs: int) -> float:
    """Spec-locked anchor for dry-run cost reporting. Per PROJECT_FACTS
    §7, gpt-5 batched Stage 4 is $0.0019/call assuming 1.5K in + 200 out."""
    return n_pairs * ESTIMATED_PER_CALL_USD


# ---------------------------------------------------------------------------
# Per-pair LLM call (sequential — see __doc__ for the batch-API caveat)
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    addressed: bool
    stance: str
    justification: str
    in_tokens: int
    out_tokens: int


def _validate_match_payload(parsed: dict) -> MatchResult:
    """Enforce the closed JSON shape. Raises ValueError on out-of-schema
    output. The Stage 4 spec says no field beyond addressed / stance /
    justification, and an out-of-enum stance is an error (not a coerced
    default) so we surface model drift quickly instead of silently
    polluting the regression input."""
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-object JSON: {type(parsed).__name__}")

    if "addressed" not in parsed:
        raise ValueError("LLM JSON missing 'addressed' field.")
    addressed_raw = parsed["addressed"]
    if isinstance(addressed_raw, bool):
        addressed = addressed_raw
    elif isinstance(addressed_raw, str) and addressed_raw.lower() in ("true", "false"):
        addressed = addressed_raw.lower() == "true"
    else:
        raise ValueError(f"'addressed' must be bool, got {addressed_raw!r}")

    stance = parsed.get("stance")
    if not isinstance(stance, str):
        raise ValueError(f"'stance' must be string, got {type(stance).__name__}")
    stance = stance.strip().upper()
    if stance not in ALLOWED_STANCES:
        raise ValueError(
            f"'stance' must be one of {sorted(ALLOWED_STANCES)}, got {stance!r}"
        )

    justification = parsed.get("justification") or ""
    if not isinstance(justification, str):
        raise ValueError(
            f"'justification' must be string, got {type(justification).__name__}"
        )
    justification = justification.strip()[:240]   # 200-char target + 40-char slack

    return MatchResult(addressed, stance, justification, 0, 0)


def match_one_pair(
    comment_text: str,
    obligation_row: dict,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    call_json: Callable[..., tuple[str, int, int]] = _call_openai_json,
    parse_json: Callable[..., tuple[dict, int, int]] = _parse_json_with_repair,
) -> MatchResult:
    """Run one match call. `call_json` and `parse_json` are injectable
    for unit tests; production uses the Stage 1b call wrappers."""
    truncated_text, was_trunc = truncate_comment(comment_text)
    user_prompt = MATCH_USER_TEMPLATE.format(
        cfr_section=obligation_row.get("cfr_section") or "<unknown>",
        obligation_text=str(obligation_row.get("obligation_text") or "")[:2000],
        subject=obligation_row.get("subject") or "",
        modal=obligation_row.get("modal") or "",
        action=obligation_row.get("action") or "",
        object=obligation_row.get("object") or "",
        comment_text=truncated_text,
    )
    raw_text, in_tok, out_tok = call_json(
        MATCH_SYSTEM_PROMPT, user_prompt,
        model=model, max_tokens=MATCH_MAX_TOKENS,
    )
    parsed, repair_in, repair_out = parse_json(
        raw_text, MATCH_SYSTEM_PROMPT, user_prompt,
        provider="openai", model=model, max_tokens=MATCH_MAX_TOKENS,
    )
    in_tok += repair_in
    out_tok += repair_out
    result = _validate_match_payload(parsed)
    result.in_tokens = in_tok
    result.out_tokens = out_tok
    # `was_trunc` is attached by the caller via the per-pair record so it
    # appears in the DataFrame; we keep MatchResult schema fixed here.
    setattr(result, "_was_truncated", was_trunc)
    return result


def _validated_match_dict(parsed: dict) -> dict:
    """Adapter from `_validate_match_payload` (which returns the
    `MatchResult` dataclass) to the dict shape the batch wrapper
    expects. Raises on schema failure — the batch wrapper routes
    raising pairs to recovery."""
    res = _validate_match_payload(parsed)
    return {
        "addressed": bool(res.addressed),
        "stance": res.stance,
        "justification": res.justification,
    }


def _build_messages_for_pair(
    comment_text: str,
    obligation_row: dict,
    *,
    head_chars: Optional[int] = None,
    tail_chars: Optional[int] = None,
) -> tuple[list[dict], bool]:
    """Construct the `[system, user]` messages list for a single pair,
    sharing the truncation policy + prompt templates with `match_one_pair`.

    `head_chars` / `tail_chars` are passed through to `truncate_comment`;
    None means module-level v2.3 defaults (1200/200). Task K rerun
    passes 4000/1000 to recover truncated pairs."""
    truncated_text, was_trunc = truncate_comment(
        comment_text, head_chars=head_chars, tail_chars=tail_chars,
    )
    user_prompt = MATCH_USER_TEMPLATE.format(
        cfr_section=obligation_row.get("cfr_section") or "<unknown>",
        obligation_text=str(obligation_row.get("obligation_text") or "")[:2000],
        subject=obligation_row.get("subject") or "",
        modal=obligation_row.get("modal") or "",
        action=obligation_row.get("action") or "",
        object=obligation_row.get("object") or "",
        comment_text=truncated_text,
    )
    return [
        {"role": "system", "content": MATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], was_trunc


def match_pairs(
    pairs: Iterable[tuple[str, str, float]],
    comments_df,
    obligations_df,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    batch: bool = True,
    max_cost_usd: float = 1500.0,
    progress_every: int = 100,
    call_json: Callable[..., tuple[str, int, int]] = _call_openai_json,
    parse_json: Callable[..., tuple[dict, int, int]] = _parse_json_with_repair,
    call_batch: Callable[..., tuple[list, list]] = _call_openai_batch,
    batch_id_prefix: str = "stage4",
):
    """Run Stage 4 matching across a list of prefilter pairs.

    Args:
        pairs: iterable of (comment_id, obligation_id, cosine_similarity).
        comments_df: pandas DataFrame indexed by comment_id (or with a
            comment_id / document_id column) containing the `comment`
            text column.
        obligations_df: pandas DataFrame indexed by obligation_id (or
            with a synthesized obligation_id column matching what
            `stage4_embedding_prefilter.obligation_id_of` produces).
        model: defaults to gpt-5 per PROJECT_FACTS §7.
        batch: True → use OpenAI Batch API (24h async, half-price gpt-5).
            False → per-pair real-time calls. Both paths share the same
            JSON-schema validation, cost-cap behavior, and DataFrame
            shape; only the LLM-call lane differs.
        max_cost_usd: hard ceiling; raises CostCapExceeded before exceeding.
        progress_every: real-time path only — one-line stderr progress
            update at this cadence (0 to disable).
        call_json, parse_json: injectable for unit tests (real-time path).
        call_batch: injectable for unit tests (batch path).
        batch_id_prefix: passed through to the batch wrapper for audit.

    Returns:
        pandas.DataFrame with columns
            [comment_id, obligation_id, cosine_similarity,
             addressed, stance, justification,
             uncertainty_truncated, in_tokens, out_tokens, cost_usd, via]
        — one row per successfully-matched pair. Pairs whose LLM call
        failed after all retries (and recovery, in batch mode) are
        excluded from the DataFrame; each is logged with its custom_id
        to stderr so downstream tools can investigate without parsing
        log noise.
    """
    import pandas as pd

    # Build comment + obligation lookups once. Accept either an
    # already-indexed DataFrame or a column we can index by.
    comments_idx = _ensure_indexed(comments_df, ("comment_id", "document_id"))
    obligations_idx = _ensure_indexed(obligations_df, ("obligation_id",))

    if batch:
        return _match_pairs_batch(
            pairs, comments_idx, obligations_idx,
            model=model, max_cost_usd=max_cost_usd,
            call_batch=call_batch, batch_id_prefix=batch_id_prefix,
        )

    out_rows: list[dict] = []
    cumulative_cost = 0.0
    n_done = 0
    n_skipped = 0
    n_total_in = 0
    n_total_out = 0

    for cid, oid, cosine in pairs:
        try:
            c_row = comments_idx.loc[cid]
            o_row = obligations_idx.loc[oid]
        except KeyError:
            n_skipped += 1
            print(f"  [stage4-match] WARN: pair ({cid}, {oid}) missing in "
                  "comments_df or obligations_df; skipping.", file=sys.stderr)
            continue

        # Pre-flight cost-cap check using the spec anchor. We add the
        # *actual* per-call cost after the call too — this catches the
        # case where a single call's measured cost would push past the
        # cap and lets us raise before doing it.
        if cumulative_cost + ESTIMATED_PER_CALL_USD > max_cost_usd:
            raise CostCapExceeded(
                f"Stage 4 cost cap ${max_cost_usd:,.2f} would be exceeded "
                f"by the next call (cumulative=${cumulative_cost:.4f}). "
                f"{n_done} pairs completed; aborting cleanly."
            )

        comment_text = str(_row_get(c_row, "comment") or _row_get(c_row, "text") or "")

        try:
            result = match_one_pair(
                comment_text, _row_to_dict(o_row),
                model=model, call_json=call_json, parse_json=parse_json,
            )
        except Exception as e:
            n_skipped += 1
            print(f"  [stage4-match] pair ({cid}, {oid}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue

        call_cost = call_cost_usd(result.in_tokens, result.out_tokens)
        cumulative_cost += call_cost
        n_total_in += result.in_tokens
        n_total_out += result.out_tokens
        n_done += 1

        out_rows.append({
            "comment_id": cid,
            "obligation_id": oid,
            "cosine_similarity": float(cosine),
            "addressed": result.addressed,
            "stance": result.stance,
            "justification": result.justification,
            "uncertainty_truncated": bool(getattr(result, "_was_truncated", False)),
            "in_tokens": result.in_tokens,
            "out_tokens": result.out_tokens,
            "cost_usd": call_cost,
            "via": "realtime",
        })

        if cumulative_cost > max_cost_usd:
            raise CostCapExceeded(
                f"Stage 4 cost cap ${max_cost_usd:,.2f} exceeded after "
                f"{n_done} pairs (cumulative=${cumulative_cost:.4f}). "
                "Last call completed; aborting before the next."
            )

        if progress_every and (n_done % progress_every == 0):
            print(f"  [stage4-match] {n_done} pairs done; "
                  f"cost=${cumulative_cost:.4f}; "
                  f"tokens in={n_total_in:,} out={n_total_out:,}; "
                  f"skipped={n_skipped}", file=sys.stderr)

    print(f"  [stage4-match] DONE — {n_done} pairs matched, "
          f"{n_skipped} skipped; cost=${cumulative_cost:.4f}; "
          f"tokens in={n_total_in:,} out={n_total_out:,}", file=sys.stderr)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Small DataFrame helpers
# ---------------------------------------------------------------------------
def _ensure_indexed(df, candidate_id_cols: tuple[str, ...]):
    """Return df indexed by the first of candidate_id_cols that exists.
    If df is already indexed by one of them (or any non-default index),
    leave it alone."""
    import pandas as pd
    if df is None:
        raise ValueError("DataFrame is None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    # Already explicitly indexed?
    if df.index.name in candidate_id_cols:
        return df
    for col in candidate_id_cols:
        if col in df.columns:
            return df.set_index(col, drop=False)
    raise KeyError(
        f"DataFrame has no recognizable id column. Expected one of "
        f"{candidate_id_cols}; got {list(df.columns)}."
    )


def _row_get(row, key):
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _row_to_dict(row) -> dict:
    try:
        return dict(row)
    except (TypeError, ValueError):
        # pandas.Series fallback
        return {k: row[k] for k in row.index}


# ---------------------------------------------------------------------------
# Batch dispatch — Stage 4 wire-up for the OpenAI Batch API wrapper
# ---------------------------------------------------------------------------
def _match_pairs_batch(
    pairs,
    comments_idx,
    obligations_idx,
    *,
    model: str,
    max_cost_usd: float,
    call_batch,
    batch_id_prefix: str,
):
    """Stage 4 wire-up for `_call_openai_batch`. Builds the per-pair
    messages list, hands it off to the batch wrapper with the Stage 4
    schema validator, and folds the wrapper's positional results back
    into a DataFrame matching the real-time path's column set."""
    import pandas as pd

    pair_list = list(pairs)
    messages_list: list[list[dict]] = []
    pair_meta: list[Optional[dict]] = []

    for cid, oid, cosine in pair_list:
        try:
            c_row = comments_idx.loc[cid]
            o_row = obligations_idx.loc[oid]
        except KeyError:
            print(f"  [stage4-match] WARN: pair ({cid}, {oid}) missing in "
                  "comments_df or obligations_df; skipping.", file=sys.stderr)
            messages_list.append(None)  # placeholder to preserve index
            pair_meta.append(None)
            continue
        comment_text = str(_row_get(c_row, "comment") or _row_get(c_row, "text") or "")
        messages, was_trunc = _build_messages_for_pair(
            comment_text, _row_to_dict(o_row),
        )
        messages_list.append(messages)
        pair_meta.append({
            "comment_id": cid,
            "obligation_id": oid,
            "cosine_similarity": float(cosine),
            "uncertainty_truncated": was_trunc,
        })

    # Drop placeholders before calling the batch wrapper — the wrapper
    # treats its input positionally and we don't want None entries in
    # the JSONL submission. Track a remap so we can fold results back.
    submit_messages: list[list[dict]] = []
    submit_to_pair: list[int] = []
    for i, m in enumerate(messages_list):
        if m is not None:
            submit_to_pair.append(i)
            submit_messages.append(m)

    if not submit_messages:
        print("  [stage4-match] no submittable pairs; returning empty DataFrame.",
              file=sys.stderr)
        return pd.DataFrame(columns=[
            "comment_id", "obligation_id", "cosine_similarity",
            "addressed", "stance", "justification",
            "uncertainty_truncated", "in_tokens", "out_tokens",
            "cost_usd", "via",
        ])

    successful, failed_pairs = call_batch(
        submit_messages, model, max_cost_usd,
        batch_id_prefix=batch_id_prefix,
        max_tokens=MATCH_MAX_TOKENS,
        validate_fn=_validated_match_dict,
    )

    out_rows: list[dict] = []
    for submit_idx, result in enumerate(successful):
        if result is None:
            continue
        pair_idx = submit_to_pair[submit_idx]
        meta = pair_meta[pair_idx]
        if meta is None:
            continue
        in_tok = int(result.get("in_tokens", 0) or 0)
        out_tok = int(result.get("out_tokens", 0) or 0)
        via = str(result.get("via", "batch"))
        # Cost depends on which lane actually serviced this pair.
        if via == "recovery":
            cost = (in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
                    + out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0)
        else:
            cost = call_cost_usd(in_tok, out_tok)
        out_rows.append({
            "comment_id": meta["comment_id"],
            "obligation_id": meta["obligation_id"],
            "cosine_similarity": meta["cosine_similarity"],
            "addressed": bool(result.get("addressed", False)),
            "stance": str(result.get("stance", "NONE")),
            "justification": str(result.get("justification", "")),
            "uncertainty_truncated": bool(meta["uncertainty_truncated"]),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "cost_usd": cost,
            "via": via,
        })

    # Surface failed_pairs explicitly — exclude from DataFrame but log
    # each so downstream can investigate without parsing log noise.
    for fp in failed_pairs:
        submit_idx = fp.get("index", -1)
        if 0 <= submit_idx < len(submit_to_pair):
            pair_idx = submit_to_pair[submit_idx]
            meta = pair_meta[pair_idx] or {}
            cid = meta.get("comment_id", "<unknown>")
            oid = meta.get("obligation_id", "<unknown>")
        else:
            cid = oid = "<unknown>"
        print(f"  [stage4-match] FAILED final pair {fp.get('custom_id', '?')} "
              f"(comment_id={cid}, obligation_id={oid}): "
              f"batch_error={fp.get('batch_error')!r}; "
              f"recovery_error={fp.get('recovery_error')!r}", file=sys.stderr)

    print(f"  [stage4-match] batch DONE — {len(out_rows)} matched, "
          f"{len(failed_pairs)} failed-final out of {len(submit_messages)} submitted.",
          file=sys.stderr)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Submit-then-recover split (Task I, 2026-05-14)
#
# The default Stage 4 path (`match_pairs` → `_match_pairs_batch` →
# `_call_openai_batch`) holds ~1.5 GB resident memory while polling for
# 9-15 hr per anchor. On a 16 GB workstation that prevents parallel
# anchor firing and has caused OOM crashes. The functions below split
# the batch into two phases:
#
#   1. `submit_batch_only`     — build messages_list (already done by
#      `prepare_batch_submission`), stage JSONL, upload, create batch,
#      write a manifest sidecar. Exit. ~60 s wall-clock and the
#      embedding model can be freed immediately afterward.
#
#   2. `collect_batch_once`    — single-pass polling step. Call
#      `client.batches.retrieve(batch_id)`; if status='completed',
#      download + parse using the manifest and return the DataFrame.
#      Otherwise return None. The polling loop lives in the retrieval
#      script (`code/14_stage4_retrieve_batches.py`) so many in-flight
#      batches can be polled in parallel without holding the embedding
#      model.
#
# These helpers do NOT replace `_match_pairs_batch`; that path remains
# the backward-compat default behavior of `match_pairs`. See Yue's
# `collect_orphan_layer_c_batches.py` / `parse_orphan_layer_c_outputs.py`
# for the Layer C analog these functions mirror.
# ---------------------------------------------------------------------------
def prepare_batch_submission(
    pairs,
    comments_df,
    obligations_df,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    head_chars: Optional[int] = None,
    tail_chars: Optional[int] = None,
) -> dict:
    """Build `messages_list` + `pair_meta` + `submit_to_pair` without
    submitting anything. Mirrors the first half of `_match_pairs_batch`
    so the submit-only path produces byte-identical JSONL to what
    `match_pairs(batch=True)` would.

    Returns a dict with:
        messages_list: list[list[dict]]  — one [system, user] per pair
        pair_meta:     list[dict | None] — comment_id, obligation_id,
                                            cosine_similarity, uncertainty_truncated
                                            (None for pairs the comments or
                                             obligations index couldn't resolve)
        submit_to_pair: list[int]        — submission-order index → pair_meta index
        n_pairs:        int              — len(messages_list submitted)
        model:          str              — passed through for the manifest
    """
    pair_list = list(pairs)
    comments_idx = _ensure_indexed(comments_df, ("comment_id", "document_id"))
    obligations_idx = _ensure_indexed(obligations_df, ("obligation_id",))

    messages_list: list[Optional[list[dict]]] = []
    pair_meta: list[Optional[dict]] = []

    for cid, oid, cosine in pair_list:
        try:
            c_row = comments_idx.loc[cid]
            o_row = obligations_idx.loc[oid]
        except KeyError:
            messages_list.append(None)
            pair_meta.append(None)
            continue
        comment_text = str(_row_get(c_row, "comment")
                           or _row_get(c_row, "text") or "")
        messages, was_trunc = _build_messages_for_pair(
            comment_text, _row_to_dict(o_row),
            head_chars=head_chars, tail_chars=tail_chars,
        )
        messages_list.append(messages)
        pair_meta.append({
            "comment_id": cid, "obligation_id": oid,
            "cosine_similarity": float(cosine),
            "uncertainty_truncated": bool(was_trunc),
        })

    submit_messages: list[list[dict]] = []
    submit_to_pair: list[int] = []
    for i, m in enumerate(messages_list):
        if m is not None:
            submit_to_pair.append(i)
            submit_messages.append(m)

    return {
        "messages_list": submit_messages,
        "pair_meta": pair_meta,
        "submit_to_pair": submit_to_pair,
        "n_pairs": len(submit_messages),
        "model": model,
    }


# OpenAI Batch API hard cap (2026-05-14): 50,000 requests per batch.
# Anchors whose prefilter pair count exceeds this must be split into
# N = ceil(n_pairs / BATCH_API_MAX_REQUESTS_PER_BATCH) shards, each
# submitted as an independent batch. See `submit_batches_sharded`.
BATCH_API_MAX_REQUESTS_PER_BATCH = 50_000


def _submit_batch_inner(
    messages_list: list,
    pair_meta: list,
    submit_to_pair: list,
    *,
    model: str,
    batch_id_prefix: str,
    client,
    output_dir,
    job_id: str,
    shard_suffix: str,
    time_fn,
) -> dict:
    """Stage JSONL + manifest sidecar, upload, create batch. Returns the
    manifest dict. The caller is responsible for cost-cap pre-checks
    (so when sharded submission checks the WHOLE pair count once at the
    top, individual shard submits don't re-check fractions).

    `shard_suffix` is "" for unsharded submissions (default
    `submit_batch_only` path) or "_shard{N}" for sharded submissions.
    Both filenames (input JSONL + manifest JSON) carry the same suffix
    so a single `ls` groups shards visually.
    """
    import json as _json

    from path_a_obligation_llm import (
        _is_reasoning_model, DEFAULT_REASONING_EFFORT,
        _ALLOWED_REASONING_EFFORTS, _openai_call_with_retry,
    )
    import time as _time

    n_pairs = len(messages_list)
    if n_pairs == 0:
        raise ValueError("_submit_batch_inner: nothing to submit (n_pairs=0)")

    input_path = output_dir / f"{job_id}__input{shard_suffix}.jsonl"
    manifest_path = output_dir / f"{job_id}__manifest{shard_suffix}.json"

    is_reasoning = _is_reasoning_model(model)
    effort = (DEFAULT_REASONING_EFFORT
              if DEFAULT_REASONING_EFFORT in _ALLOWED_REASONING_EFFORTS
              else "minimal")
    with input_path.open("w", encoding="utf-8") as f:
        for idx, messages in enumerate(messages_list):
            body = {
                "model": model, "messages": messages,
                "max_completion_tokens": MATCH_MAX_TOKENS,
                "response_format": {"type": "json_object"},
            }
            if is_reasoning:
                body["reasoning_effort"] = effort
            f.write(_json.dumps({
                "custom_id": f"pair_{idx}", "method": "POST",
                "url": "/v1/chat/completions", "body": body,
            }) + "\n")

    def _create_file():
        with open(input_path, "rb") as f:
            return client.files.create(file=f, purpose="batch")
    file_obj = _openai_call_with_retry(_create_file, attempt_label="batch-upload")

    def _create_batch():
        return client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"prefix": batch_id_prefix, "job_id": job_id,
                      "shard": shard_suffix or "single"},
        )
    batch = _openai_call_with_retry(_create_batch, attempt_label="batch-create")
    batch_id = batch.id
    submitted_at_utc = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(time_fn()))

    manifest = {
        "batch_id": batch_id, "job_id": job_id, "model": model,
        "batch_id_prefix": batch_id_prefix,
        "shard_suffix": shard_suffix,
        "submitted_at_utc": submitted_at_utc,
        "n_pairs": n_pairs,
        "pair_meta": pair_meta,
        "submit_to_pair": submit_to_pair,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        _json.dump(manifest, f)

    label = f"submit-only{shard_suffix}" if shard_suffix else "submit-only"
    print(f"  [{label}] batch_id={batch_id} job_id={job_id} "
          f"pairs={n_pairs:,} input={input_path.name} "
          f"manifest={manifest_path.name}", file=sys.stderr)

    return {
        "batch_id": batch_id, "job_id": job_id,
        "input_jsonl": str(input_path),
        "manifest_json": str(manifest_path),
        "n_pairs": n_pairs,
        "submitted_at_utc": submitted_at_utc,
        "model": model,
        "shard_suffix": shard_suffix,
    }


def _ensure_submit_environment(client, output_dir, time_fn):
    """Resolve the OpenAI client + output_dir + time function defaults.
    Extracted from `submit_batch_only` so the sharded entry point can
    reuse the same fall-throughs."""
    import os as _os
    import time as _time
    from pathlib import Path as _Path
    from path_a_obligation_llm import BATCH_DEFAULT_OUTPUT_DIR
    if time_fn is None:
        time_fn = _time.time
    if output_dir is None:
        output_dir = BATCH_DEFAULT_OUTPUT_DIR
    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        import openai
        api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is not set.")
        client = openai.OpenAI(api_key=api_key)
    return client, output_dir, time_fn


def _new_job_id(time_fn) -> str:
    import time as _time
    import uuid as _uuid
    ts = _time.strftime("%Y%m%dT%H%M%S", _time.gmtime(time_fn()))
    short_uuid = _uuid.uuid4().hex[:8]
    return f"{ts}_{short_uuid}"


def submit_batch_only(
    submission: dict,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    batch_id_prefix: str = "stage4",
    client=None,
    output_dir=None,
    max_cost_usd: float = 1500.0,
    estimated_per_call_usd: float = ESTIMATED_PER_CALL_USD,
    time_fn=None,
) -> dict:
    """Submit a prepared batch and return the manifest. No polling.

    For anchors with > 50,000 pairs (the OpenAI Batch API hard cap),
    use `submit_batches_sharded` instead — this function raises a
    descriptive error if asked to submit more than the cap.

    Writes two files under `output_dir`:
        {job_id}__input.jsonl     — Batch API input (preserved as the
                                    primary audit trail)
        {job_id}__manifest.json   — pair_meta + submit_to_pair + model,
                                    so `collect_batch_once` can fold
                                    results back without re-running the
                                    prefilter
    Raises CostCapExceeded BEFORE creating the batch if the estimated
    cost exceeds `max_cost_usd`.

    Returns a dict suitable for appending to the in-flight tracking CSV:
        {batch_id, job_id, input_jsonl, manifest_json, n_pairs,
         submitted_at_utc, model, shard_suffix}
    `shard_suffix` is "" for the unsharded path.
    """
    messages_list = submission["messages_list"]
    n_pairs = len(messages_list)

    if n_pairs > BATCH_API_MAX_REQUESTS_PER_BATCH:
        raise ValueError(
            f"submit_batch_only: n_pairs={n_pairs:,} exceeds the OpenAI "
            f"Batch API cap of {BATCH_API_MAX_REQUESTS_PER_BATCH:,}; "
            "use submit_batches_sharded(...) instead.")

    # Pre-submission cost guard. Mirrors `_call_openai_batch`'s pre-check.
    estimated_total = n_pairs * estimated_per_call_usd
    if estimated_total > max_cost_usd:
        raise CostCapExceeded(
            f"submit_batch_only pre-check: estimated cost "
            f"${estimated_total:.2f} for {n_pairs:,} pairs "
            f"(anchor ${estimated_per_call_usd:.4f}/call) exceeds cap "
            f"${max_cost_usd:,.2f}. No API call made."
        )
    if n_pairs == 0:
        raise ValueError("submit_batch_only: nothing to submit (n_pairs=0)")

    client, output_dir, time_fn = _ensure_submit_environment(
        client, output_dir, time_fn,
    )
    job_id = _new_job_id(time_fn)
    return _submit_batch_inner(
        messages_list,
        submission["pair_meta"],
        submission["submit_to_pair"],
        model=model, batch_id_prefix=batch_id_prefix,
        client=client, output_dir=output_dir,
        job_id=job_id, shard_suffix="", time_fn=time_fn,
    )


def submit_batches_sharded(
    submission: dict,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    batch_id_prefix: str = "stage4",
    client=None,
    output_dir=None,
    max_cost_usd: float = 1500.0,
    estimated_per_call_usd: float = ESTIMATED_PER_CALL_USD,
    shard_size: int = BATCH_API_MAX_REQUESTS_PER_BATCH,
    time_fn=None,
) -> list:
    """Submit one or more batches, sharding the submission when
    `n_pairs > shard_size`. Returns a list of manifest dicts, one per
    shard. The unsharded case returns a single-element list whose
    manifest is byte-identical to what `submit_batch_only` would write
    (same filename pattern: `{job_id}__input.jsonl` /
    `{job_id}__manifest.json`, no shard suffix).

    Sharded case:
      - All shards share one `job_id` (generated once at the top); each
        shard gets its own `batch_id` from OpenAI. Filenames use a
        `_shard{N}` suffix (N is 1-indexed) so `ls
        data/intermediate/openai_batches/` groups the shards together.
      - The cost-cap pre-check runs ONCE on the total pair count. If
        any single shard would exceed `max_cost_usd / n_shards`, the
        check still passes because the cap is enforced on the total.
      - Each shard's manifest contains the SLICE of pair_meta covering
        only that shard's pairs, with a local-indexed `submit_to_pair`.
        This means `collect_batch_once` can fold each shard
        independently without cross-shard knowledge — the retrieve
        script concatenates DataFrames at the (docket, rule_type)
        level after all shards complete.

    Args:
        submission: output of `prepare_batch_submission`.
        shard_size: max pairs per shard (default 50,000 — the OpenAI
            Batch API hard cap; can be lowered for testing).
    """
    messages_list = submission["messages_list"]
    pair_meta = submission["pair_meta"]
    submit_to_pair = submission["submit_to_pair"]
    n_pairs = len(messages_list)

    if n_pairs == 0:
        raise ValueError("submit_batches_sharded: nothing to submit (n_pairs=0)")

    # ONE cost-cap pre-check on the total. Each per-shard submit
    # bypasses re-checking because we've already validated against the
    # whole submission's pair count.
    estimated_total = n_pairs * estimated_per_call_usd
    if estimated_total > max_cost_usd:
        raise CostCapExceeded(
            f"submit_batches_sharded pre-check: estimated total cost "
            f"${estimated_total:.2f} for {n_pairs:,} pairs "
            f"(anchor ${estimated_per_call_usd:.4f}/call) exceeds cap "
            f"${max_cost_usd:,.2f}. No API call made."
        )

    client, output_dir, time_fn = _ensure_submit_environment(
        client, output_dir, time_fn,
    )
    job_id = _new_job_id(time_fn)

    if n_pairs <= shard_size:
        # Unsharded fast path — byte-identical to `submit_batch_only`.
        # We do not call `submit_batch_only` (which would re-run the
        # cost pre-check); just delegate to the inner helper.
        return [_submit_batch_inner(
            messages_list, pair_meta, submit_to_pair,
            model=model, batch_id_prefix=batch_id_prefix,
            client=client, output_dir=output_dir,
            job_id=job_id, shard_suffix="", time_fn=time_fn,
        )]

    n_shards = (n_pairs + shard_size - 1) // shard_size   # ceil
    print(f"  [submit-sharded] {n_pairs:,} pairs > {shard_size:,} cap → "
          f"splitting into {n_shards} shards (job_id={job_id})",
          file=sys.stderr)

    manifests: list[dict] = []
    for k in range(n_shards):
        start = k * shard_size
        end = min(start + shard_size, n_pairs)
        shard_messages = messages_list[start:end]
        # `submit_to_pair[start:end]` are the pair_meta indices that
        # this shard's submission indices refer to. We carve out the
        # corresponding slice of pair_meta and rebuild a LOCAL
        # submit_to_pair (identity, since the slice is dense).
        shard_pair_meta_indices = submit_to_pair[start:end]
        shard_pair_meta = [pair_meta[i] for i in shard_pair_meta_indices]
        shard_submit_to_pair = list(range(len(shard_messages)))

        shard_suffix = f"_shard{k + 1}"
        manifest = _submit_batch_inner(
            shard_messages, shard_pair_meta, shard_submit_to_pair,
            model=model,
            batch_id_prefix=f"{batch_id_prefix}{shard_suffix}",
            client=client, output_dir=output_dir,
            job_id=job_id, shard_suffix=shard_suffix, time_fn=time_fn,
        )
        manifests.append(manifest)

    return manifests


def _result_row_from_validated(result: dict, meta: dict) -> dict:
    """Shape a parsed-and-validated batch row into the canonical
    DataFrame schema. Shared by `collect_batch_once` and (potential)
    orphan-reconstruction tools."""
    in_tok = int(result.get("in_tokens", 0) or 0)
    out_tok = int(result.get("out_tokens", 0) or 0)
    via = str(result.get("via", "batch"))
    if via == "recovery":
        cost = (in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
                + out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0)
    else:
        cost = call_cost_usd(in_tok, out_tok)
    return {
        "comment_id": meta["comment_id"],
        "obligation_id": meta["obligation_id"],
        "cosine_similarity": float(meta["cosine_similarity"]),
        "addressed": bool(result.get("addressed", False)),
        "stance": str(result.get("stance", "NONE")),
        "justification": str(result.get("justification", "")),
        "uncertainty_truncated": bool(meta["uncertainty_truncated"]),
        "in_tokens": in_tok, "out_tokens": out_tok,
        "cost_usd": cost, "via": via,
    }


def collect_batch_once(
    batch_id: str,
    manifest_path,
    *,
    client=None,
    output_dir=None,
    download_only_if_complete: bool = True,
):
    """Single-pass collector for one in-flight batch.

    Returns:
        (status: str, df_or_none: pd.DataFrame | None)
        - status: 'completed' | 'failed' | 'expired' | 'cancelled' |
                  'in_progress' | 'finalizing' | 'validating' | ...
        - df_or_none: a canonical-schema DataFrame when status='completed'
                      and the output was downloaded; None otherwise.

    No polling loop — the caller (retrieval script) decides whether to
    sleep and retry.
    """
    import json as _json
    import pandas as pd
    from pathlib import Path as _Path

    from path_a_obligation_llm import (
        BATCH_DEFAULT_OUTPUT_DIR, _openai_call_with_retry,
        _read_batch_output_text, _strip_codefence,
    )
    if output_dir is None:
        output_dir = BATCH_DEFAULT_OUTPUT_DIR
    output_dir = _Path(output_dir)

    if client is None:
        import openai
        import os as _os
        api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is not set.")
        client = openai.OpenAI(api_key=api_key)

    batch = _openai_call_with_retry(
        lambda: client.batches.retrieve(batch_id),
        attempt_label="batch-poll",
    )
    status = str(getattr(batch, "status", "unknown"))
    if status != "completed":
        return status, None

    if download_only_if_complete:
        # Load manifest to fold results.
        manifest = _json.loads(_Path(manifest_path).read_text(encoding="utf-8"))
        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError(
                f"Batch {batch_id} reported completed but has no "
                "output_file_id; manifest preserved for manual recovery.")
        content_resp = _openai_call_with_retry(
            lambda: client.files.content(output_file_id),
            attempt_label="batch-download",
        )
        out_text = _read_batch_output_text(content_resp)
        output_path = output_dir / f"{manifest['job_id']}__output.jsonl"
        output_path.write_text(out_text, encoding="utf-8")

        pair_meta = manifest["pair_meta"]
        submit_to_pair = manifest["submit_to_pair"]

        out_rows: list[dict] = []
        for line in out_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            custom_id = row.get("custom_id", "")
            if not isinstance(custom_id, str) or not custom_id.startswith("pair_"):
                continue
            try:
                submit_idx = int(custom_id.split("_", 1)[1])
            except ValueError:
                continue
            if submit_idx < 0 or submit_idx >= len(submit_to_pair):
                continue
            pair_idx = submit_to_pair[submit_idx]
            meta = pair_meta[pair_idx] if 0 <= pair_idx < len(pair_meta) else None
            if meta is None:
                continue

            err = row.get("error")
            response = row.get("response")
            if err is not None or response is None:
                continue
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or 0)
            try:
                choices = body.get("choices") or []
                content = choices[0]["message"]["content"]
            except (IndexError, KeyError, TypeError):
                continue
            try:
                parsed = _json.loads(_strip_codefence(content))
            except _json.JSONDecodeError:
                continue
            try:
                validated = _validated_match_dict(parsed)
            except Exception:  # noqa: BLE001
                # Skip schema-failed rows — the submit-then-recover path
                # doesn't have a real-time fallback; downstream analysis
                # can re-batch these later if desired.
                continue
            result_dict = dict(validated)
            result_dict["in_tokens"] = in_tok
            result_dict["out_tokens"] = out_tok
            result_dict["via"] = "batch"
            out_rows.append(_result_row_from_validated(result_dict, meta))

        return status, pd.DataFrame(out_rows)

    return status, None


# ---------------------------------------------------------------------------
# CLI entry — single-anchor convenience
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    import csv as _csv
    import json as _json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs-jsonl", type=Path, required=True,
                    help="Prefilter output (one JSON per line).")
    ap.add_argument("--comments-parquet", type=Path, required=True)
    ap.add_argument("--obligations-csv", type=Path, required=True)
    ap.add_argument("--docket-id", type=str, required=True)
    ap.add_argument("--rule-type", default="proposed",
                    help="proposed | final (rule_type column in obligations CSV).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--max-cost", type=float, default=1500.0)
    ap.add_argument("--no-batch", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    cdf = pd.read_parquet(args.comments_parquet)
    cdf = cdf[cdf["docket_id"] == args.docket_id]

    with args.obligations_csv.open("r", encoding="utf-8") as f:
        odf_rows = list(_csv.DictReader(f))
    # Synthesize obligation_id matching the prefilter contract.
    from stage4_embedding_prefilter import obligation_id_of
    for r in odf_rows:
        r["obligation_id"] = obligation_id_of(r)
    odf = pd.DataFrame(odf_rows)

    pairs: list[tuple[str, str, float]] = []
    with args.pairs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = _json.loads(line)
            pairs.append((j["comment_id"], j["obligation_id"], float(j["cosine"])))

    df = match_pairs(
        pairs, cdf, odf,
        model=args.model,
        batch=not args.no_batch,
        max_cost_usd=args.max_cost,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[stage4-match] wrote {len(df)} matched rows to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
