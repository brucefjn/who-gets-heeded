"""
layer_c_judge.py — Layer C LLM-as-judge for the 23 binary rhetoric
features locked in notes/2026-05-08_prompt_template_v1.md (v1 LOCK).

Public entry point:
    run_layer_c(comments_df, model="gpt-5", batch=True, max_cost_usd=200.0)
        → pd.DataFrame [comment_id, docket_id, <23 binaries>,
                        truncation_flag, in_tokens, out_tokens, cost_usd, via]

Both the batch (OpenAI Batch API via `_call_openai_batch`) and real-time
(`_call_openai_json`) paths share the prompt, validator, and DataFrame
shape so cost-accounting and downstream consumers see one canonical
output regardless of dispatch lane.

Truncation: comments longer than `TRUNCATION_TRIGGER_CHARS` (12,000)
are clipped to the first `TRUNCATION_HEAD_CHARS` (10,000) + the last
`TRUNCATION_TAIL_CHARS` (2,000) separated by an explicit marker. This
mirrors PROJECT_FACTS §12 item 6's preamble + signature retention
policy for attachment-recovered comments.

Schema enforcement: `_validated_layer_c_dict` is strict — every one of
the 23 binaries must be present as a plain `0` or `1` integer (bools
like `True` and strings like `"1"` are rejected). Strictness is the
contract `_call_openai_batch.validate_fn` expects: any pathological row
routes through the wrapper's recovery path (one real-time retry).

Out of scope (per Task F): the 6 hybrid regex/heuristic features, the
Sonnet 4.5 cross-LLM check on the 5% sample, prompt-stability variants.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Reuse Path A / Stage 4 LLM call infrastructure — same retry policy,
# same JSON repair, same rate-limit handling. Do not reimplement.
from path_a_obligation_llm import (  # type: ignore  (lib injected on sys.path)
    CostCapExceeded,
    DEFAULT_OPENAI_MODEL,
    GPT5_BATCH_INPUT_PRICE_PER_M,
    GPT5_BATCH_OUTPUT_PRICE_PER_M,
    GPT5_STANDARD_INPUT_PRICE_PER_M,
    GPT5_STANDARD_OUTPUT_PRICE_PER_M,
    _call_openai_batch,
    _call_openai_json,
    _parse_json_with_repair,
)


# ---------------------------------------------------------------------------
# The 23 binary fields, verbatim from notes/2026-05-08_prompt_template_v1.md.
# Order preserved to match the prompt's documented anchor table; downstream
# consumers can iterate in this order for stable per-binary reporting.
# ---------------------------------------------------------------------------
SOPHISTICATION_BINARIES = (
    "names_targeted_provision",
    "states_requested_change",
    "provides_example",
    "provides_legal_or_empirical_background",
)
STANCE_BINARIES = (
    "expresses_explicit_support",
    "expresses_explicit_opposition",
)
ARGUMENT_CLAIM_BINARIES = (
    "burdensome",
    "lacks_flexibility",
    "not_sufficient_time",
    "conflicting_interests",
    "disputed_information",
    "legal_challenge",
    "overreach",
    "requests_clarification",
    "lacks_clarity",
    "seeks_exclusion",
    "too_broad",
    "too_narrow",
)
FRAME_BINARIES = (
    "technical_scientific_frame_present",
    "legal_statutory_frame_present",
    "justice_equity_frame_present",
    "economic_cost_benefit_frame_present",
    "lived_experience_frame_present",
)
LAYER_C_BINARIES = (
    SOPHISTICATION_BINARIES
    + STANCE_BINARIES
    + ARGUMENT_CLAIM_BINARIES
    + FRAME_BINARIES
)
assert len(LAYER_C_BINARIES) == 23, "Layer C binary count must be 23"


# ---------------------------------------------------------------------------
# Pricing + cost anchors (per PROJECT_FACTS §7)
# ---------------------------------------------------------------------------
# Anchor batched cost-per-call from PROJECT_FACTS §7: 3K input + 500
# output @ gpt-5 batched = ~$0.00438. Round to $0.0044 for the dry-run
# estimate, matching the spec's $155/70K guideline.
LAYER_C_ESTIMATED_PER_CALL_USD = 0.0044

# Real-time (non-batched) anchor: 3K input × $1.25/M + 500 output × $10/M
# = $0.00375 + $0.005 = $0.00875. Used by the concurrent real-time path's
# pre-check and the orchestrator's dry-run when --no-batch is set.
LAYER_C_REALTIME_PER_CALL_USD = 0.0089

# Output-token budget per call. Locked at 600 to leave headroom over
# the spec's 500-token output anchor — gpt-5 with reasoning sometimes
# spills a little.
LAYER_C_MAX_OUTPUT_TOKENS = 600


# ---------------------------------------------------------------------------
# Truncation policy (PROJECT_FACTS §12 item 6 + Task F spec)
# ---------------------------------------------------------------------------
TRUNCATION_TRIGGER_CHARS = 12_000
TRUNCATION_HEAD_CHARS = 10_000
TRUNCATION_TAIL_CHARS = 2_000
TRUNCATION_MARKER = "\n... [truncation marker] ...\n"


def truncate_for_layer_c(text: str) -> tuple[str, bool]:
    """Apply the locked truncation policy. Returns (text_for_prompt,
    was_truncated). Short comments pass through unchanged."""
    if not text:
        return "", False
    if len(text) <= TRUNCATION_TRIGGER_CHARS:
        return text, False
    head = text[:TRUNCATION_HEAD_CHARS]
    tail = text[-TRUNCATION_TAIL_CHARS:]
    return f"{head}{TRUNCATION_MARKER}{tail}", True


# ---------------------------------------------------------------------------
# Prompt template (verbatim from notes/2026-05-08_prompt_template_v1.md)
# ---------------------------------------------------------------------------
LAYER_C_SYSTEM_PROMPT = """\
You are coding U.S. EPA public comments on federal regulatory rulemakings (covering air, water, waste, and toxics programs). For the comment below, return a JSON object with these 23 binary fields. Use 1 if the feature is present in the comment text, 0 if absent. Code surface form — what the commenter explicitly says — not inferred intent.

SOPHISTICATION (Cuéllar 2005 five-question checklist):
- names_targeted_provision: 1 if the comment cites a specific CFR section, statute, FR provision, or named rule subsection
- states_requested_change: 1 if the comment articulates a specific change to the rule (not just a general direction)
- provides_example: 1 if the comment includes at least one specific example, scenario, or case
- provides_legal_or_empirical_background: 1 if the comment cites legal authority, scientific data, agency reports, or named empirical evidence

STANCE (Eidelman & Grom 2019 hierarchy — code separately from claim types):
- expresses_explicit_support: 1 if the commenter explicitly endorses the proposed rule
- expresses_explicit_opposition: 1 if the commenter explicitly opposes the proposed rule
(Both 0 = neutral, off-topic, or pure clarification request. Both 1 = coding error.)

ARGUMENT-CLAIM TYPES (Eidelman & Grom 2019 — 12 specific types, multi-label):
- burdensome: rule imposes disproportionate financial or administrative burden
- lacks_flexibility: rule is overly prescriptive, lacks accommodation
- not_sufficient_time: comment period or implementation timeline too short
- conflicting_interests: rule favors one party at expense of another
- disputed_information: factual claims, studies, or premises in rule are flawed
- legal_challenge: comment threatens litigation or invokes specific case law
- overreach: agency exceeds statutory authority
- requests_clarification: commenter explicitly asks the agency to clarify a provision
- lacks_clarity: commenter asserts the rule itself is ambiguous
- seeks_exclusion: commenter seeks exemption or carve-out for a group
- too_broad: rule is over-inclusive
- too_narrow: rule is under-inclusive

FRAME (multi-label, presence of frame in argument):
- technical_scientific_frame_present
- legal_statutory_frame_present
- justice_equity_frame_present
- economic_cost_benefit_frame_present
- lived_experience_frame_present

Output: JSON object with all 23 binary keys. No explanation, no narrative. If the comment is empty, off-topic, or unparseable, set all fields to 0.
"""


LAYER_C_USER_TEMPLATE = """\
COMMENT:
{comment_text}
"""


def _build_layer_c_messages(comment_text: str) -> list[dict]:
    """One [system, user] messages list per comment. Shared by batch
    and real-time paths so the prompt is byte-identical regardless of
    dispatch lane."""
    return [
        {"role": "system", "content": LAYER_C_SYSTEM_PROMPT},
        {"role": "user", "content": LAYER_C_USER_TEMPLATE.format(
            comment_text=comment_text)},
    ]


# ---------------------------------------------------------------------------
# Schema validator — strict; raises on any drift from the 23-binary
# contract. The batch wrapper catches the ValueError and routes the row
# to its real-time recovery path (one retry through `_call_openai_json`),
# so transient drift recovers without polluting the regression input.
# ---------------------------------------------------------------------------
def _validated_layer_c_dict(parsed: dict) -> dict:
    """Strict 23-binary schema validator. Returns the canonical dict
    (binaries-only, integer 0/1) or raises ValueError on:
      - non-dict payload
      - missing binary field
      - value that is not exactly int 0 or int 1 (bools rejected even
        though Python treats `bool` as `int`; "1"/"0" strings rejected)
    """
    if not isinstance(parsed, dict):
        raise ValueError(f"non-dict payload: {type(parsed).__name__}")

    out: dict = {}
    for name in LAYER_C_BINARIES:
        if name not in parsed:
            raise ValueError(f"missing binary field: {name!r}")
        v = parsed[name]
        # Reject bools — `isinstance(True, int)` is True in Python, but
        # the schema contract is plain int. Same for strings and floats.
        if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1):
            raise ValueError(
                f"binary field {name!r} must be int 0 or 1, got {v!r} "
                f"({type(v).__name__})"
            )
        out[name] = v
    return out


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------
def _layer_c_cost(in_tok: int, out_tok: int, *, via: str) -> float:
    if via == "recovery":
        return (in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
                + out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0)
    return (in_tok * GPT5_BATCH_INPUT_PRICE_PER_M / 1_000_000.0
            + out_tok * GPT5_BATCH_OUTPUT_PRICE_PER_M / 1_000_000.0)


def estimate_cost(n_comments: int) -> float:
    """Pre-API cost estimate anchored to PROJECT_FACTS §7's $0.0044/call
    batched gpt-5 rate (3K in + 500 out). Used by `--dry-run`."""
    return n_comments * LAYER_C_ESTIMATED_PER_CALL_USD


def estimate_cost_realtime(n_comments: int) -> float:
    """Real-time variant of `estimate_cost` — uses standard (non-batched)
    gpt-5 pricing for the per-comment dispatch path. Anchor: 3K input ×
    $1.25/M + 500 output × $10/M ≈ $0.0089/call. ~2× the batched anchor.
    Used by the concurrent real-time path's pre-check and by the
    orchestrator's --dry-run when --no-batch is set."""
    return n_comments * LAYER_C_REALTIME_PER_CALL_USD


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_layer_c(
    comments_df,
    model: str = DEFAULT_OPENAI_MODEL,
    batch: bool = True,
    max_cost_usd: float = 200.0,
    *,
    concurrency: int = 1,
    call_batch: Callable = _call_openai_batch,
    call_json: Callable = _call_openai_json,
    parse_json: Callable = _parse_json_with_repair,
    progress_every: int = 100,
):
    """Code each comment in `comments_df` with the 23 Layer C binaries.

    Args:
        comments_df: pandas DataFrame with at least `document_id` (or
            `comment_id`), `docket_id`, and `comment` (or `text`).
        model: production target, gpt-5 per PROJECT_FACTS §7.
        batch: True → OpenAI Batch API; False → per-comment real-time.
            Cost-accounted at batched or standard rates accordingly.
        max_cost_usd: hard cap; raised as CostCapExceeded before the
            batch is created (when batch=True) or before the next
            real-time call exceeds the cap.
        concurrency: real-time-only. When `batch=False` and
            `concurrency > 1`, dispatches via the concurrent real-time
            path (asyncio.to_thread + asyncio.Semaphore), cribbed from
            Bruce's Stage 1b Task H pattern in
            `path_a_obligation_llm._run_primary_realtime_concurrent`.
            Default 1 = sequential. Ignored when `batch=True`.
        call_batch, call_json, parse_json: injectable for tests.
        progress_every: real-time-only stderr cadence; 0 disables.

    Returns:
        pandas.DataFrame with columns:
          [comment_id, docket_id, <23 binaries>, truncation_flag,
           in_tokens, out_tokens, cost_usd, via]
        Rows preserve `comments_df`'s input order. Comments whose LLM
        call failed (after batch + recovery, or after real-time + retry)
        are excluded from the DataFrame; each is logged to stderr.
    """
    import pandas as pd

    rows = _prepare_comment_rows(comments_df)
    if not rows:
        return _empty_layer_c_df()

    if batch:
        return _run_layer_c_batch(
            rows, model=model, max_cost_usd=max_cost_usd,
            call_batch=call_batch,
        )
    if concurrency and concurrency > 1:
        return _run_layer_c_realtime_concurrent(
            rows, model=model, max_cost_usd=max_cost_usd,
            concurrency=concurrency,
            call_json=call_json, parse_json=parse_json,
            progress_every=progress_every,
        )
    return _run_layer_c_realtime(
        rows, model=model, max_cost_usd=max_cost_usd,
        call_json=call_json, parse_json=parse_json,
        progress_every=progress_every,
    )


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def _prepare_comment_rows(comments_df) -> list[dict]:
    """Normalize input DataFrame into a list of {comment_id, docket_id,
    comment} dicts, dropping rows with empty text."""
    if comments_df is None or len(comments_df) == 0:
        return []
    cid_col = "comment_id" if "comment_id" in comments_df.columns else "document_id"
    text_col = "comment" if "comment" in comments_df.columns else "text"
    docket_col = "docket_id" if "docket_id" in comments_df.columns else None

    out: list[dict] = []
    for _, row in comments_df.iterrows():
        cid = row.get(cid_col)
        text = row.get(text_col) or ""
        if not isinstance(text, str) or not text.strip():
            continue
        out.append({
            "comment_id": str(cid) if cid is not None else "",
            "docket_id": str(row.get(docket_col, "")) if docket_col else "",
            "comment": str(text),
        })
    return out


def _empty_layer_c_df():
    import pandas as pd
    cols = ["comment_id", "docket_id", *LAYER_C_BINARIES,
            "truncation_flag", "in_tokens", "out_tokens", "cost_usd", "via"]
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------
def _run_layer_c_batch(
    rows: list[dict],
    *,
    model: str,
    max_cost_usd: float,
    call_batch: Callable,
):
    """Submit all comments as one batch, fold validated results back into
    a DataFrame in submission order."""
    import pandas as pd

    # Pre-API cost guard. _call_openai_batch enforces this internally
    # in production, but we duplicate the check here so the raise also
    # fires when tests inject a custom `call_batch` (and so the orchestrator
    # never gets to JSONL staging when the cap is impossible).
    estimated = estimate_cost(len(rows))
    if estimated > max_cost_usd:
        raise CostCapExceeded(
            f"Layer C batch pre-check: estimated cost ${estimated:,.2f} "
            f"for {len(rows):,} comments (anchor "
            f"${LAYER_C_ESTIMATED_PER_CALL_USD:.4f}/call) exceeds cap "
            f"${max_cost_usd:,.2f}. No API call made."
        )

    truncated_flags: list[bool] = []
    messages_list: list[list[dict]] = []
    for r in rows:
        truncated_text, was_trunc = truncate_for_layer_c(r["comment"])
        messages_list.append(_build_layer_c_messages(truncated_text))
        truncated_flags.append(was_trunc)

    successful, failed_pairs = call_batch(
        messages_list, model, max_cost_usd,
        batch_id_prefix="layer_c",
        max_tokens=LAYER_C_MAX_OUTPUT_TOKENS,
        validate_fn=_validated_layer_c_dict,
        estimated_per_call_usd=LAYER_C_ESTIMATED_PER_CALL_USD,
    )

    out_rows: list[dict] = []
    for i, result in enumerate(successful):
        if result is None:
            continue
        record = _result_to_record(rows[i], result, truncated_flags[i])
        out_rows.append(record)

    for fp in failed_pairs:
        idx = fp.get("index", -1)
        cid = rows[idx]["comment_id"] if 0 <= idx < len(rows) else "<unknown>"
        print(f"  [layer-c] FAILED final {fp.get('custom_id', '?')} "
              f"(comment_id={cid}): batch_error={fp.get('batch_error')!r}; "
              f"recovery_error={fp.get('recovery_error')!r}", file=sys.stderr)

    print(f"  [layer-c] batch DONE — {len(out_rows)} coded, "
          f"{len(failed_pairs)} failed out of {len(rows)} submitted.",
          file=sys.stderr)
    return pd.DataFrame(out_rows, columns=_canonical_columns())


def _result_to_record(comment_row: dict, result: dict,
                      was_truncated: bool) -> dict:
    """Combine a validated 23-binary dict + token usage into a single
    DataFrame row matching the canonical column set."""
    in_tok = int(result.get("in_tokens", 0) or 0)
    out_tok = int(result.get("out_tokens", 0) or 0)
    via = str(result.get("via", "batch"))
    cost = _layer_c_cost(in_tok, out_tok, via=via)
    record = {
        "comment_id": comment_row["comment_id"],
        "docket_id": comment_row["docket_id"],
    }
    for name in LAYER_C_BINARIES:
        record[name] = int(result.get(name, 0))
    record["truncation_flag"] = bool(was_truncated)
    record["in_tokens"] = in_tok
    record["out_tokens"] = out_tok
    record["cost_usd"] = cost
    record["via"] = via
    return record


def _canonical_columns() -> list[str]:
    return ["comment_id", "docket_id", *LAYER_C_BINARIES,
            "truncation_flag", "in_tokens", "out_tokens", "cost_usd", "via"]


# ---------------------------------------------------------------------------
# Real-time path (smoke / regression only)
# ---------------------------------------------------------------------------
def _run_layer_c_realtime(
    rows: list[dict],
    *,
    model: str,
    max_cost_usd: float,
    call_json: Callable,
    parse_json: Callable,
    progress_every: int,
):
    """Per-comment real-time loop. Same DataFrame shape as the batch path.
    The cost cap is checked before each call using the spec anchor; if
    the cumulative would exceed it, raise before the next API call."""
    import pandas as pd

    out_rows: list[dict] = []
    cumulative_cost = 0.0
    n_done = 0
    n_failed = 0

    for r in rows:
        if cumulative_cost + LAYER_C_ESTIMATED_PER_CALL_USD > max_cost_usd:
            raise CostCapExceeded(
                f"Layer C real-time cost cap ${max_cost_usd:,.2f} would "
                f"be exceeded by the next call "
                f"(cumulative=${cumulative_cost:.4f}). {n_done} coded; "
                "aborting cleanly."
            )

        truncated_text, was_trunc = truncate_for_layer_c(r["comment"])
        messages = _build_layer_c_messages(truncated_text)
        sys_prompt, usr_prompt = messages[0]["content"], messages[1]["content"]

        try:
            raw, in_tok, out_tok = call_json(
                sys_prompt, usr_prompt, model=model,
                max_tokens=LAYER_C_MAX_OUTPUT_TOKENS,
            )
            parsed, repair_in, repair_out = parse_json(
                raw, sys_prompt, usr_prompt,
                provider="openai", model=model,
                max_tokens=LAYER_C_MAX_OUTPUT_TOKENS,
            )
            in_tok += repair_in
            out_tok += repair_out
            validated = _validated_layer_c_dict(parsed)
        except CostCapExceeded:
            raise
        except Exception as e:
            n_failed += 1
            print(f"  [layer-c] comment {r['comment_id']}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue

        # Real-time path: bill at standard (non-batched) rates.
        validated["in_tokens"] = in_tok
        validated["out_tokens"] = out_tok
        validated["via"] = "realtime"
        record = _result_to_record(r, validated, was_trunc)
        # Override cost using standard rates since via='realtime' (the
        # batched bill is reserved for batch path; real-time = standard).
        record["cost_usd"] = (
            in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
            + out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0
        )
        cumulative_cost += record["cost_usd"]
        out_rows.append(record)
        n_done += 1

        if progress_every and (n_done % progress_every == 0):
            print(f"  [layer-c] {n_done} comments done; "
                  f"cost=${cumulative_cost:.4f}", file=sys.stderr)

    print(f"  [layer-c] real-time DONE — {n_done} coded, "
          f"{n_failed} failed; cost=${cumulative_cost:.4f}", file=sys.stderr)
    return pd.DataFrame(out_rows, columns=_canonical_columns())


# ---------------------------------------------------------------------------
# Concurrent real-time path
# ---------------------------------------------------------------------------
# Cribbed from Bruce's Stage 1b Task H pattern
# (path_a_obligation_llm._run_primary_realtime_concurrent, commits 0286237).
#
# Architecture:
#   - asyncio.Semaphore(concurrency) for backpressure
#   - asyncio.to_thread wrapping the existing sync per-comment work so we
#     don't duplicate the JSON-repair / retry edge cases that already live
#     in _call_openai_json + _parse_json_with_repair
#   - Pre-check the cap once on n_rows × LAYER_C_REALTIME_PER_CALL_USD;
#     no per-call gating inside the loop (concurrent gating with shared
#     cumulative cost would require a lock and adds complexity without
#     proportional safety — the pre-check is the right boundary)
#   - Tag each in-flight task with its input index; asyncio.gather
#     preserves submission order, so the output DataFrame is order-stable
#   - Per-call failures captured in-band as ("fail", idx, exc) — never
#     crash the whole gather
#
# Throughput anchor: gpt-5 Tier-4 TPM ~800K. Layer C call is ~3.5K tokens
# in + 500 out = ~4K tokens per call. 800K / 4K = 200 calls/min sustained
# ceiling. Default concurrency=20 with ~30s wall per call lands at ~40 calls/min
# — well under the ceiling, leaves room to dial up.
# ---------------------------------------------------------------------------
def _run_layer_c_realtime_concurrent(
    rows: list[dict],
    *,
    model: str,
    max_cost_usd: float,
    concurrency: int,
    call_json: Callable,
    parse_json: Callable,
    progress_every: int = 100,
):
    """Concurrent variant of `_run_layer_c_realtime`. Runs up to
    `concurrency` in-flight real-time calls. DataFrame output preserves
    input order; failures are excluded from the DataFrame and logged.
    """
    import pandas as pd

    if not rows:
        return _empty_layer_c_df()

    # Pre-check using the real-time anchor (not the batched anchor).
    estimated_total = len(rows) * LAYER_C_REALTIME_PER_CALL_USD
    if estimated_total > max_cost_usd:
        raise CostCapExceeded(
            f"Layer C concurrent real-time pre-check: estimated cost "
            f"${estimated_total:,.2f} for {len(rows):,} comments "
            f"(anchor ${LAYER_C_REALTIME_PER_CALL_USD:.4f}/call) exceeds "
            f"cap ${max_cost_usd:,.2f}. No API call made."
        )

    import asyncio

    def _code_one_sync(r: dict) -> tuple:
        """Sync per-comment pipeline: truncate → build messages → call
        → parse → validate → record. Returns the canonical-shape record
        dict. Raises on any failure; the async wrapper catches and tags."""
        truncated_text, was_trunc = truncate_for_layer_c(r["comment"])
        messages = _build_layer_c_messages(truncated_text)
        sys_p, usr_p = messages[0]["content"], messages[1]["content"]

        raw, in_tok, out_tok = call_json(
            sys_p, usr_p, model=model,
            max_tokens=LAYER_C_MAX_OUTPUT_TOKENS,
        )
        parsed, rep_in, rep_out = parse_json(
            raw, sys_p, usr_p,
            provider="openai", model=model,
            max_tokens=LAYER_C_MAX_OUTPUT_TOKENS,
        )
        in_tok += rep_in
        out_tok += rep_out
        validated = _validated_layer_c_dict(parsed)

        validated["in_tokens"] = in_tok
        validated["out_tokens"] = out_tok
        validated["via"] = "realtime"
        record = _result_to_record(r, validated, was_trunc)
        # Real-time path is billed at standard (non-batched) rates;
        # override the _result_to_record default (which uses batched
        # pricing when via != "recovery").
        record["cost_usd"] = (
            in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
            + out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0
        )
        return record

    async def _run() -> list[tuple]:
        sem = asyncio.Semaphore(concurrency)
        n_done = [0]

        async def _one(idx: int, r: dict):
            async with sem:
                try:
                    record = await asyncio.to_thread(_code_one_sync, r)
                    n_done[0] += 1
                    if progress_every and n_done[0] % progress_every == 0:
                        print(
                            f"  [layer-c-concurrent] {n_done[0]}/"
                            f"{len(rows)} done", file=sys.stderr,
                        )
                    return ("ok", idx, record)
                except Exception as e:  # noqa: BLE001
                    n_done[0] += 1
                    return ("fail", idx, e)

        tasks = [asyncio.create_task(_one(i, r))
                 for i, r in enumerate(rows)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())

    out_records: list[dict] = []
    n_failed = 0
    # asyncio.gather preserves submission order — iterate in that order
    # so the resulting DataFrame's row order matches the input.
    for status, idx, payload in results:
        if status == "fail":
            n_failed += 1
            cid = rows[idx].get("comment_id", "<unknown>")
            print(
                f"  [layer-c-concurrent] comment {cid}: "
                f"{type(payload).__name__}: {payload}", file=sys.stderr,
            )
            continue
        out_records.append(payload)

    total_cost = sum(r["cost_usd"] for r in out_records)
    print(
        f"  [layer-c-concurrent] real-time DONE — {len(out_records)} "
        f"coded, {n_failed} failed; cost=${total_cost:.4f}",
        file=sys.stderr,
    )
    return pd.DataFrame(out_records, columns=_canonical_columns())
