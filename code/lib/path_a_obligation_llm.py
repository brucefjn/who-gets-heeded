"""
path_a_obligation_llm.py — Stage 1b of Path A obligation extraction.

Per notes/2026-05-09_path_a_stage1_obligation_extraction_v2.md.

Stage 1b takes Stage 1a candidate obligations (from
path_a_obligation_heuristic.py) and runs:

  1. PRIMARY VERIFICATION (gpt-5 by default; Sonnet 4.5 for cross-LLM check).
     Returns verified=true|false, structured-field corrections, complexity,
     recital_disposition (closed enum), and split obligations if compound.

  2. SECTION-LEVEL SAFETY NET (per-amendatory-block call).
     Returns obligations the primary extractor may have missed.

  3. SAFETY-NET SECOND-PASS VERIFICATION (Major 5 fix).
     Each safety-net hit is routed back through the primary verification
     prompt before merging. Eliminates the v1 asymmetry that inflated FPs.

Closed-enum constraints + null behavior + retry-on-malformed-JSON are all
enforced per Minor 10 of the audit.

Outputs (per docket):
  data/processed/path_a_obligations_verified_<docket_id>.csv   ← primary
  data/processed/path_a_obligations_missed_<docket_id>.csv     ← safety-net
                                                                 (post-second-pass)
  data/processed/path_a_obligations_<docket_id>.csv            ← merged final

Cross-LLM consistency (validation rules + 5% Stage 2 production):
  data/processed/path_a_xllm_consistency_<docket_id>.csv

Usage (single-provider primary pass):
    python -m code.lib.path_a_obligation_llm verify \\
        --candidates data/processed/path_a_obligations_candidates_<docket>_<rule>.csv \\
        --binding-text data/raw/federal_register/<file>.txt \\
        --provider openai --model gpt-5 \\
        --out data/processed/path_a_obligations_verified_<docket>_<rule>.csv

Usage (cross-LLM consistency, validation rules):
    python -m code.lib.path_a_obligation_llm xllm \\
        --primary data/processed/path_a_obligations_verified_<docket>_<rule>.csv \\
        --candidates data/processed/path_a_obligations_candidates_<docket>_<rule>.csv \\
        --binding-text data/raw/federal_register/<file>.txt \\
        --out data/processed/path_a_xllm_consistency_<docket>_<rule>.csv

Smoke test (no API call; validates JSON-schema enforcement on canned LLM
outputs):
    python -m code.lib.path_a_obligation_llm --smoke-test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# Closed enums (Minor 10 — JSON robustness)
# ---------------------------------------------------------------------------
ALLOWED_MODALS = {
    "must", "shall", "may not", "must not",
    "may", "is required to", "are required to",
    "is prohibited from", "are prohibited from",
    "is permitted to", "are permitted to",
    "is authorized to", "are authorized to",
    "no_modal_implicit",
}
ALLOWED_MODAL_STRENGTHS = {"strong", "permissive"}
ALLOWED_AMENDATORY_ACTIONS = {
    "add", "revise", "remove", "replace", "redesignate", "none",
}
ALLOWED_RECITAL_DISPOSITIONS = {
    "none", "amendatory_quote", "statutory_quote",
    "preamble_narrative", "definition_embedded",
}
ALLOWED_COMPLEXITIES = {"simple", "compound"}

# ---------------------------------------------------------------------------
# Default models (per v2.3 model lock + PROJECT_FACTS § 7)
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"

# Pricing (per PROJECT_FACTS §7). Shared with downstream cost guards
# in Stage 4 and any other batch-eligible call site.
GPT5_BATCH_INPUT_PRICE_PER_M = 0.625
GPT5_BATCH_OUTPUT_PRICE_PER_M = 5.0
GPT5_STANDARD_INPUT_PRICE_PER_M = 1.25
GPT5_STANDARD_OUTPUT_PRICE_PER_M = 10.0


class CostCapExceeded(RuntimeError):
    """Raised when an LLM call (real-time or batched) would exceed the
    caller-supplied `max_cost_usd` budget. Canonical definition lives
    here so the batch + real-time paths surface the same exception
    type — downstream code does a single `except CostCapExceeded`."""

# Context window for primary verification (±3 sentences ≈ ~600 chars each side)
CONTEXT_RADIUS_CHARS = 600

# Section-text radius for safety-net (full amendatory block, capped)
SAFETY_NET_SECTION_CHAR_CAP = 8000

# Per-call retry budget (rate-limit + transient errors)
LLM_RETRY_LIMIT = 3
LLM_RETRY_BASE_SLEEP_S = 6  # exponential backoff base

# Per-call max output tokens. GPT-5 is a REASONING MODEL: max_completion_tokens
# includes internal reasoning tokens, so we need a generous cap or the JSON
# output gets truncated mid-stream and parsing fails. Empirically, a primary
# verification call needs ~6-8K total budget (most going to reasoning) for
# reliable structured-JSON output. SAFETYNET is bigger because the prompt
# carries section text + existing-obligations list as input.
PRIMARY_MAX_TOKENS = 8000
SAFETYNET_MAX_TOKENS = 12000

# OpenAI reasoning_effort for reasoning models (gpt-5 family). Structured
# extraction with closed-enum constraints does NOT need deep multi-step
# reasoning — `minimal` skips the long internal chain-of-thought and
# returns the JSON faster + cheaper. Caller can override via env var.
DEFAULT_REASONING_EFFORT = os.environ.get(
    "PATH_A_REASONING_EFFORT", "minimal"
).strip().lower() or "minimal"
# Allowed values per OpenAI API: "minimal", "low", "medium", "high"
_ALLOWED_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}

# Verbose-debug toggle: when true, print raw LLM responses on parse failure.
PATH_A_DEBUG = os.environ.get("PATH_A_DEBUG", "").strip().lower() in (
    "1", "true", "yes",
)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class VerifiedObligation:
    """One row in path_a_obligations_verified_<docket>.csv.

    A single Stage 1a candidate may produce zero (verified=false), one
    (simple), or multiple (compound, split) VerifiedObligation rows.
    """
    candidate_idx: int           # links back to Stage 1a obligation_idx
    docket_id: str
    rule_type: str               # 'proposed' | 'final'
    split_idx: int               # 0 if simple, 0..N-1 if compound
    verified: bool
    verified_reason: str
    complexity: str              # 'simple' | 'compound'
    recital_disposition: str     # closed enum
    cfr_part: Optional[str]
    cfr_section: Optional[str]
    subject: str
    modal: str                   # closed enum
    modal_strength: str          # closed enum
    action: str
    object: str
    passive_voice: bool
    is_conditional: bool
    condition_text: Optional[str]
    amendatory_action: str       # closed enum
    obligation_text: str         # verbatim from candidate (for traceability)
    cross_reference: Optional[str]   # e.g. "§ 80.10" if cited in obligation
    found_via: str               # 'primary' | 'safety_net' | 'safety_net_verified'
    llm_provider: str            # 'openai' | 'anthropic'
    llm_model: str
    in_tokens: int
    out_tokens: int


@dataclass
class CrossLLMRow:
    """One row in path_a_xllm_consistency_<docket>.csv."""
    candidate_idx: int
    docket_id: str
    rule_type: str
    primary_provider: str
    primary_model: str
    secondary_provider: str
    secondary_model: str
    primary_verified: bool
    secondary_verified: bool
    verified_match: bool
    primary_complexity: str
    secondary_complexity: str
    complexity_match: bool
    primary_subjects: str   # ; -joined
    secondary_subjects: str
    primary_modals: str
    secondary_modals: str
    primary_actions: str
    secondary_actions: str
    primary_objects: str
    secondary_objects: str


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
PRIMARY_SYSTEM_PROMPT = """\
You are an expert in U.S. federal regulatory text analysis. You verify
whether a candidate text span is a binding deontic obligation in CFR
amendatory text, and return its structured representation.

VERIFICATION RULE:

An obligation is BINDING (verified=true) if EITHER:
(a) The obligation appears within an amendatory block (i.e., follows or is
    contained within text matching: "is revised to read as follows",
    "is amended to read as follows", "is added to read as follows",
    "is amended by adding paragraph (X) to read as follows", "is added",
    "is amended"). Mark verified=true regardless of whether the obligation
    cites or restates an existing CFR section. The amendatory block makes
    the language new binding text.
(b) The obligation is in a non-amendatory binding-text section (e.g.,
    new subpart added without an explicit amendatory wrapper) AND is not
    a recital, definition, or preamble narrative.

DEFINITION-EMBEDDED OBLIGATIONS: definitions can embed implicit binding
requirements. If a definition specifies an actor, threshold, deadline, or
quantitative criterion, treat the embedded requirement AS the obligation.
Examples:
- "Eligible facility means one that has filed Form A by March 1." →
  obligation: "facility must file Form A by March 1"; verified=true.
- "Routine maintenance means activities performed at intervals not
  exceeding 12 months." → obligation: "maintenance must occur at
  intervals ≤ 12 months"; verified=true.

NOT BINDING — DEFINITION SCOPE-CLARIFICATION (v2.1, 2026-05-09; narrowed v2.2):
Within a definition (text matching `\\bX means\\b` or `\\bX is defined as\\b` or
appearing inside a definitions section like § 80.1401), short sentences
that ONLY describe what the defined term covers — using descriptive verbs
like "may or may not", "need not", "includes", "shall include" referring
to membership/scope of the term itself — are NOT binding obligations.
Concrete examples (verified=false):
- "This other party may or may not be registered under the RFS program."
  (clarifies scope of "Contractual affiliate" — no actor + obligation)
- "Such party need not be a producer or importer of fuel."
  (clarifies scope of a definition — no actor + action duty)

This rule does NOT apply when:
- The candidate has a clear actor (e.g., "operators", "the Administrator",
  "each party") performing a clear action verb under a clear modal
  (must/shall/may not/etc.). Such candidates are binding even when
  inside a definition section.
- The candidate prescribes any temporal, quantitative, or eligibility
  criterion (e.g., "must file Form A by March 1") — those are
  definition-embedded obligations, verified=true.

When in doubt, prefer verified=true; the rule above only covers narrow
descriptive scope-clarification, not borderline cases.

NOT BINDING (verified=false) ONLY if:
- Comment is in preamble (SUPPLEMENTARY INFORMATION) AND no amendatory
  block contains it.
- Verbatim quote of statutory text WITHOUT restatement as new amendment.
- Pure definition without embedded obligation (e.g., "Major source means
  any stationary source ..." with no temporal/quantitative criterion).
- Narrow definition scope-clarification sentences (per v2.1/v2.2 rule
  above).
- Preamble narrative ("Commenters argued ...", "EPA agrees that ...").

CONSTRAINTS (closed enums — do NOT return novel values):
- modal MUST be one of: ["must", "shall", "may not", "must not", "may",
  "is required to", "are required to", "is prohibited from",
  "are prohibited from", "is permitted to", "are permitted to",
  "is authorized to", "are authorized to", "no_modal_implicit"]
- modal_strength MUST be one of: ["strong", "permissive"]
- amendatory_action MUST be one of: ["add", "revise", "remove",
  "replace", "redesignate", "none"]
- recital_disposition MUST be one of: ["none", "amendatory_quote",
  "statutory_quote", "preamble_narrative", "definition_embedded"]
- complexity MUST be one of: ["simple", "compound"]
- cfr_section MUST be a string of form "X.Y" or "X.Y(z)" or null
  (literal JSON null, NOT empty string) if not extractable.
- cross_reference MUST be a string like "§ 80.10" or null if absent.

OUTPUT: return ONLY a JSON object (no markdown, no preamble) with shape:
{
  "verified": <bool>,
  "verified_reason": "<one sentence>",
  "complexity": "simple" | "compound",
  "recital_disposition": "<closed_enum>",
  "obligations": [
    {
      "subject": "<str>",
      "modal": "<closed_enum>",
      "modal_strength": "<closed_enum>",
      "action": "<str>",
      "object": "<str>",
      "passive_voice": <bool>,
      "is_conditional": <bool>,
      "condition_text": "<str>" | null,
      "cfr_section": "<str>" | null,
      "amendatory_action": "<closed_enum>",
      "cross_reference": "<str>" | null
    }
  ]
}

If verified=false, return obligations: [].
If complexity=compound, return one obligation object per split obligation.
"""


PRIMARY_USER_TEMPLATE = """\
Candidate obligation:
\"\"\"{candidate_sentence}\"\"\"

Surrounding context (±3 sentences, with amendatory-block header if any):
\"\"\"{context_text}\"\"\"

CFR location: Part {cfr_part}, § {cfr_section}
Amendatory action (heuristic-detected): {amendatory_action}

Pre-extracted structured fields (from spaCy, may need correction):
- subject: {subject}
- modal: {modal}
- modal_strength: {modal_strength}
- action: {action}
- object: {object}
- passive_voice: {passive_voice}
- is_conditional: {is_conditional}
- condition_text: {condition_text}

Tasks:
1. Decide verified: true | false. Apply the BINDING / NOT BINDING rules.
2. If verified=true, correct structured fields where the spaCy parse is wrong.
3. Set complexity: "simple" (one actor + one modal + one action) or
   "compound" (split rule: modals differ OR actions target different
   objects). If compound, return a list of split obligations.
4. Set recital_disposition for downstream sub-metric analysis.

Return JSON only.
"""


SAFETY_NET_SYSTEM_PROMPT = """\
You are reviewing a CFR-amendment text section for any binding deontic
obligations that may have been missed by the primary extractor.

Apply the same BINDING / NOT BINDING rules as the primary verification:
- Amendatory blocks (is revised to read as follows / is added / etc.)
  make their text new binding text — even if it cites an existing section.
- Definitions with temporal/quantitative criteria embed obligations.
- Preamble narrative ("Commenters argued ...") is NOT binding.

PRECISION ANCHOR: only return obligations you are HIGHLY CONFIDENT are
missed. When in doubt, OMIT — false positives here cascade into the
downstream regression.

OUTPUT: return ONLY a JSON object (no markdown) with shape:
{
  "missed_obligations": [
    {
      "subject": "<str>",
      "modal": "<closed_enum>",
      "action": "<str>",
      "object": "<str>",
      "obligation_text": "<verbatim sentence from the section>",
      "found_via": "passive_voice" | "no_modal" | "cross_reference"
                 | "definition_embedded" | "other"
    }
  ]
}

Closed enum on `modal`: ["must", "shall", "may not", "must not", "may",
"is required to", "are required to", "is prohibited from",
"are prohibited from", "is permitted to", "are permitted to",
"is authorized to", "are authorized to", "no_modal_implicit"].
"""


SAFETY_NET_USER_TEMPLATE = """\
Section text (binding-text only):
\"\"\"{section_text}\"\"\"

Already-extracted obligations from this section (may be incomplete):
{json_existing}

Return JSON only with the missed_obligations list. If you find no missed
obligations, return: {{"missed_obligations": []}}.
"""


RETRY_REPAIR_SYSTEM = (
    "Your previous response was not valid JSON. "
    "Return ONLY the JSON object with no preamble, no markdown formatting, "
    "and no commentary."
)


# ---------------------------------------------------------------------------
# LLM call wrappers — pluggable provider
# ---------------------------------------------------------------------------
def _is_reasoning_model(model: str) -> bool:
    """gpt-5 family + o-series are reasoning models; they accept
    reasoning_effort and treat max_completion_tokens as inclusive of
    reasoning tokens."""
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _call_openai_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Call OpenAI with JSON-output expectation. Returns (text, in_tokens,
    out_tokens). One repair-retry on malformed JSON happens at the
    caller level (verify_candidate / find_missed_obligations).

    For reasoning models (gpt-5 / o-series), max_completion_tokens INCLUDES
    internal reasoning tokens — keep PRIMARY_MAX_TOKENS generous (8K+) and
    pass reasoning_effort='minimal' to short-circuit deep chain-of-thought
    on a structured-extraction task that doesn't need it.
    """
    import openai
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")
    client = openai.OpenAI(api_key=api_key)

    is_reasoning = _is_reasoning_model(model)
    effort = (DEFAULT_REASONING_EFFORT
              if DEFAULT_REASONING_EFFORT in _ALLOWED_REASONING_EFFORTS
              else "minimal")

    base_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    last_err = None
    for attempt in range(LLM_RETRY_LIMIT):
        try:
            kwargs = dict(base_kwargs)
            if is_reasoning:
                kwargs["reasoning_effort"] = effort
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            # Detect API-level truncation (finish_reason='length') — surface it
            # explicitly so the caller can distinguish truncation from a model
            # that returned legitimately-short JSON.
            finish_reason = (resp.choices[0].finish_reason
                             if resp.choices else None)
            if finish_reason == "length":
                print(f"  [openai] WARN: finish_reason='length' "
                      f"(out_tokens={out_tok}/{max_tokens}); response likely truncated. "
                      f"Bump max_completion_tokens or use reasoning_effort='minimal'.",
                      file=sys.stderr)
            return text, in_tok, out_tok
        except Exception as e:
            last_err = e
            etype = type(e).__name__
            msg = str(e)
            # Some models reject reasoning_effort or response_format=json_object
            # with a TypeError / BadRequestError. Retry once without those.
            if (etype == "BadRequestError"
                    and ("reasoning_effort" in msg or "response_format" in msg)):
                print(f"  [openai] BadRequestError on optional kwarg; "
                      f"retrying without it. ({msg[:120]})",
                      file=sys.stderr)
                # Strip whichever kwarg was rejected
                if "reasoning_effort" in msg:
                    is_reasoning = False
                if "response_format" in msg:
                    base_kwargs.pop("response_format", None)
                continue
            if etype in ("RateLimitError", "APIConnectionError", "APITimeoutError"):
                sleep_s = LLM_RETRY_BASE_SLEEP_S * (2 ** attempt)
                print(f"  [openai] {etype}; sleep {sleep_s}s "
                      f"(attempt {attempt + 1}/{LLM_RETRY_LIMIT})",
                      file=sys.stderr)
                time.sleep(sleep_s)
                continue
            raise
    raise RuntimeError(f"OpenAI call failed after {LLM_RETRY_LIMIT} retries: "
                       f"{type(last_err).__name__}: {last_err}")


def _call_anthropic_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Call Anthropic Messages API with JSON-output expectation. Returns
    (text, in_tokens, out_tokens). The Messages API doesn't have a strict
    JSON-mode flag, so we rely on the system-prompt instruction + caller-
    side parse-with-repair."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY env var is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    last_err = None
    for attempt in range(LLM_RETRY_LIMIT):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = msg.content[0].text if msg.content else ""
            in_tok = msg.usage.input_tokens
            out_tok = msg.usage.output_tokens
            return text, in_tok, out_tok
        except Exception as e:
            last_err = e
            etype = type(e).__name__
            if etype in ("RateLimitError", "APIConnectionError", "APITimeoutError"):
                sleep_s = LLM_RETRY_BASE_SLEEP_S * (2 ** attempt)
                print(f"  [anthropic] {etype}; sleep {sleep_s}s "
                      f"(attempt {attempt + 1}/{LLM_RETRY_LIMIT})",
                      file=sys.stderr)
                time.sleep(sleep_s)
                continue
            raise
    raise RuntimeError(f"Anthropic call failed after {LLM_RETRY_LIMIT} retries: "
                       f"{type(last_err).__name__}: {last_err}")


def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    if provider == "openai":
        return _call_openai_json(system_prompt, user_prompt, model, max_tokens)
    elif provider == "anthropic":
        return _call_anthropic_json(system_prompt, user_prompt, model, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# JSON parsing with one repair-retry (Minor 10 fix)
# ---------------------------------------------------------------------------
def _strip_codefence(text: str) -> str:
    """Strip ``` and ```json fences; some models add them despite instruction."""
    s = text.strip()
    if s.startswith("```"):
        # Drop first fence line
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        # Drop trailing fence
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def _parse_json_with_repair(
    text: str,
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> tuple[dict, int, int]:
    """Parse JSON output. If malformed, do exactly one repair retry with
    the RETRY_REPAIR_SYSTEM prefix appended to the system prompt (per spec).

    On second failure, raise RuntimeError with truncated preview by default
    or full response if PATH_A_DEBUG=1.
    """
    cleaned = _strip_codefence(text)
    try:
        return json.loads(cleaned), 0, 0
    except json.JSONDecodeError:
        if PATH_A_DEBUG:
            print(f"  [debug] first-parse failed; raw response "
                  f"(len={len(cleaned)}):\n{cleaned}\n---",
                  file=sys.stderr)
        # One repair retry
        repair_system = system_prompt + "\n\n" + RETRY_REPAIR_SYSTEM
        repair_user = (
            "Your previous response was not valid JSON. The original task is "
            "below; return ONLY a valid JSON object answering it, with no "
            "preamble, no commentary, and no markdown fences.\n\n"
            f"ORIGINAL TASK:\n{user_prompt}"
        )
        text2, in_tok, out_tok = _call_llm_json(
            repair_system, repair_user, provider, model, max_tokens,
        )
        cleaned2 = _strip_codefence(text2)
        try:
            return json.loads(cleaned2), in_tok, out_tok
        except json.JSONDecodeError as e:
            preview = cleaned2 if PATH_A_DEBUG else (cleaned2[:300] + "...")
            preview_note = "" if PATH_A_DEBUG else (
                "  (set PATH_A_DEBUG=1 to print the full response)")
            raise RuntimeError(
                f"LLM produced malformed JSON twice. Last response "
                f"(len={len(cleaned2)}):\n{preview}{preview_note}"
            ) from e


# ---------------------------------------------------------------------------
# OpenAI Batch API wrapper (24h asynchronous lane, half-price gpt-5)
# ---------------------------------------------------------------------------
# Default per-call cost anchor for the batch pre-submission guard. The
# anchor matches PROJECT_FACTS §7 (Stage 4: 1.5K in + 200 out @ batched
# gpt-5 rates ≈ $0.0019). Callers can override via the parameter.
BATCH_DEFAULT_ESTIMATED_PER_CALL_USD = 0.0019

# Polling defaults — start at 30s, double each retry, cap at 5 min.
# Hard timeout matches OpenAI's batch completion_window=24h.
BATCH_DEFAULT_INITIAL_POLL_S = 30.0
BATCH_DEFAULT_MAX_POLL_S = 300.0
BATCH_DEFAULT_TIMEOUT_S = 24 * 3600

# Default output dir for the audit-trail JSONL files. Caller can override.
BATCH_DEFAULT_OUTPUT_DIR = Path("data/intermediate/openai_batches")


def _split_messages_for_recovery(messages: list[dict]) -> tuple[str, str]:
    """Extract (system_prompt, user_prompt) from a `[system, user]`
    messages list so a failed batch pair can be retried through the
    existing real-time `_call_openai_json` path."""
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    usr_msg = next((m for m in messages if m.get("role") == "user"), None)
    if sys_msg is None or usr_msg is None:
        raise ValueError(
            "Recovery requires [system, user] messages shape; got roles="
            f"{[m.get('role') for m in messages]}."
        )
    return str(sys_msg.get("content") or ""), str(usr_msg.get("content") or "")


def _openai_call_with_retry(fn, *, attempt_label: str = "openai-batch"):
    """Run an OpenAI SDK call with the same rate-limit / transient-error
    backoff the real-time path uses. `fn` must be idempotent — we may
    call it multiple times."""
    last_err = None
    for attempt in range(LLM_RETRY_LIMIT):
        try:
            return fn()
        except Exception as e:
            last_err = e
            etype = type(e).__name__
            if etype in ("RateLimitError", "APIConnectionError", "APITimeoutError"):
                s = LLM_RETRY_BASE_SLEEP_S * (2 ** attempt)
                print(f"  [{attempt_label}] {etype}; sleep {s}s "
                      f"(attempt {attempt + 1}/{LLM_RETRY_LIMIT})",
                      file=sys.stderr)
                time.sleep(s)
                continue
            raise
    raise RuntimeError(
        f"{attempt_label} failed after {LLM_RETRY_LIMIT} retries: "
        f"{type(last_err).__name__}: {last_err}"
    )


def _read_batch_output_text(content_resp) -> str:
    """The OpenAI SDK has shipped a couple of different return types
    from `client.files.content(...)`; cover both .text and .content."""
    if hasattr(content_resp, "text") and isinstance(content_resp.text, str):
        return content_resp.text
    if hasattr(content_resp, "content"):
        raw = content_resp.content
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    if hasattr(content_resp, "read"):
        raw = content_resp.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    return str(content_resp)


def _call_openai_batch(
    messages_list: list[list[dict]],
    model: str,
    max_cost_usd: float,
    *,
    batch_id_prefix: str = "stage4",
    max_tokens: int = 400,
    validate_fn: Optional["callable"] = None,
    estimated_per_call_usd: float = BATCH_DEFAULT_ESTIMATED_PER_CALL_USD,
    timeout_seconds: int = BATCH_DEFAULT_TIMEOUT_S,
    initial_poll_seconds: float = BATCH_DEFAULT_INITIAL_POLL_S,
    max_poll_seconds: float = BATCH_DEFAULT_MAX_POLL_S,
    sleep_fn=None,
    time_fn=None,
    client=None,
    output_dir: Path = BATCH_DEFAULT_OUTPUT_DIR,
) -> tuple[list[Optional[dict]], list[dict]]:
    """Submit a batch of chat-completions requests through OpenAI's
    Batch API (24h async, ~half-price) and return per-pair results.

    Args:
        messages_list: one `[system, user]` (or compatible) messages list
            per pair. The function preserves submission order via the
            `custom_id="pair_{idx}"` round-trip.
        model: the production target (`gpt-5` for Stage 4).
        max_cost_usd: hard cap; raises `CostCapExceeded` if
            `len(messages_list) * estimated_per_call_usd` would exceed
            it. The pre-check happens BEFORE the batch is created, so
            no API call is made on cap-violations.
        batch_id_prefix: label captured in the batch metadata + audit
            file names. Default `stage4`.
        max_tokens: per-call output-budget for the body's
            `max_completion_tokens` (reasoning + visible text combined
            for gpt-5).
        validate_fn: optional schema validator applied to each parsed
            JSON output. Signature: `(parsed: dict) -> dict`; raises on
            schema failure. Failing pairs are routed to recovery. If
            None, only `json.loads` is enforced.
        estimated_per_call_usd: anchor used by the pre-submission guard.
        timeout_seconds: hard wall-clock cap on the polling loop.
            Raises `TimeoutError` (with the batch id in the message) if
            the batch hasn't completed by then — do NOT silently fall
            back to real-time; the caller decides whether to wait.
        initial_poll_seconds, max_poll_seconds: exponential-backoff
            bounds. Start at 30 s, double, cap at 5 min.
        sleep_fn, time_fn: injectable for tests (default to
            `time.sleep` / `time.time`).
        client: injectable OpenAI client (defaults to a fresh
            `openai.OpenAI()` reading `OPENAI_API_KEY`).
        output_dir: where input/output JSONL audit files are persisted.
            NEVER auto-cleaned — these files are the recovery trail.

    Returns:
        (successful, failed_pairs):
          - successful: list with the same length as messages_list. Each
            entry is either None (pair never produced a valid result) or
            a dict containing the parsed/validated fields plus
            `in_tokens`, `out_tokens`, and `via` ('batch' | 'recovery').
          - failed_pairs: list of dicts {custom_id, index, batch_error,
            recovery_error} for pairs the batch couldn't produce and
            the real-time recovery couldn't either. successful[i] is
            None for these indices.

    Recovery: each pair that fails inside the batch lane (either a row-
    level `error`, missing/malformed JSON, or `validate_fn` raising) is
    retried EXACTLY ONCE via the real-time `_call_openai_json` path. If
    that real-time retry also fails, the pair is appended to
    `failed_pairs`. Recovery cost is computed at standard (non-batched)
    gpt-5 rates so the per-call cost stays honest.

    Side effects: two JSONL files are persisted, NEVER auto-cleaned —
        {output_dir}/{timestamp}_{short_uuid}__input.jsonl
        {output_dir}/{timestamp}_{short_uuid}__output.jsonl
    """
    import uuid

    if sleep_fn is None:
        sleep_fn = time.sleep
    if time_fn is None:
        time_fn = time.time

    n_pairs = len(messages_list)

    # Pre-submission cost guard — BEFORE any API call.
    estimated_total = n_pairs * estimated_per_call_usd
    if estimated_total > max_cost_usd:
        raise CostCapExceeded(
            f"Batch pre-check: estimated cost ${estimated_total:.2f} for "
            f"{n_pairs:,} pairs (anchor ${estimated_per_call_usd:.4f}/call) "
            f"exceeds cap ${max_cost_usd:,.2f}. No API call made."
        )

    if n_pairs == 0:
        return [], []

    # Build the audit-trail file paths.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(time_fn()))
    short_uuid = uuid.uuid4().hex[:8]
    job_id = f"{ts}_{short_uuid}"
    input_path = output_dir / f"{job_id}__input.jsonl"
    output_path = output_dir / f"{job_id}__output.jsonl"

    # Stage the input JSONL.
    is_reasoning = _is_reasoning_model(model)
    effort = (DEFAULT_REASONING_EFFORT
              if DEFAULT_REASONING_EFFORT in _ALLOWED_REASONING_EFFORTS
              else "minimal")
    with input_path.open("w", encoding="utf-8") as f:
        for idx, messages in enumerate(messages_list):
            body = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if is_reasoning:
                body["reasoning_effort"] = effort
            f.write(json.dumps({
                "custom_id": f"pair_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }) + "\n")

    # Acquire client lazily so tests can inject mocks.
    if client is None:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is not set.")
        client = openai.OpenAI(api_key=api_key)

    # Submit input file + batch with the real-time path's retry semantics.
    def _create_file():
        with open(input_path, "rb") as f:
            return client.files.create(file=f, purpose="batch")
    file_obj = _openai_call_with_retry(_create_file, attempt_label="batch-upload")

    def _create_batch():
        return client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"prefix": batch_id_prefix, "job_id": job_id},
        )
    batch = _openai_call_with_retry(_create_batch, attempt_label="batch-create")
    batch_id = batch.id
    print(f"  [batch] submitted batch_id={batch_id} job_id={job_id} "
          f"pairs={n_pairs:,} input_file={input_path}", file=sys.stderr)

    # Polling with exponential backoff, hard 24h wall-clock cap.
    t_start = time_fn()
    delay = float(initial_poll_seconds)
    while True:
        batch = _openai_call_with_retry(
            lambda: client.batches.retrieve(batch_id),
            attempt_label="batch-poll",
        )
        status = getattr(batch, "status", "unknown")
        elapsed = time_fn() - t_start
        print(f"  [batch] {batch_id} status={status} "
              f"elapsed={elapsed:.0f}s", file=sys.stderr)
        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled", "cancelling"):
            errors_info = getattr(batch, "errors", None)
            raise RuntimeError(
                f"Batch {batch_id} ended with status={status}. "
                f"Errors: {errors_info!r}. Input JSONL preserved at "
                f"{input_path}."
            )
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within 24h; "
                f"manual recovery required. Input JSONL preserved at "
                f"{input_path}."
            )
        sleep_fn(delay)
        delay = min(delay * 2.0, float(max_poll_seconds))

    # Download output JSONL and persist.
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(
            f"Batch {batch_id} reported completed but has no output_file_id."
        )
    content_resp = _openai_call_with_retry(
        lambda: client.files.content(output_file_id),
        attempt_label="batch-download",
    )
    out_text = _read_batch_output_text(content_resp)
    output_path.write_text(out_text, encoding="utf-8")
    print(f"  [batch] downloaded output_file_id={output_file_id} -> "
          f"{output_path}", file=sys.stderr)

    # Parse output, route failures to recovery.
    successful: list[Optional[dict]] = [None] * n_pairs
    pre_recovery_failures: list[tuple[int, str]] = []
    batched_in_tok = 0
    batched_out_tok = 0
    seen_idxs: set[int] = set()

    for line in out_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [batch] bad output JSONL line: {e}", file=sys.stderr)
            continue
        custom_id = row.get("custom_id", "")
        if not isinstance(custom_id, str) or not custom_id.startswith("pair_"):
            continue
        try:
            idx = int(custom_id.split("_", 1)[1])
        except ValueError:
            continue
        if idx < 0 or idx >= n_pairs:
            continue
        seen_idxs.add(idx)

        err = row.get("error")
        response = row.get("response")
        if err is not None or response is None:
            pre_recovery_failures.append((idx, str(err) or "no response body"))
            continue

        body = response.get("body") or {}
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)

        try:
            choices = body.get("choices") or []
            content = choices[0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as e:
            pre_recovery_failures.append((idx, f"missing content: {e}"))
            continue

        try:
            parsed = json.loads(_strip_codefence(content))
        except json.JSONDecodeError as e:
            pre_recovery_failures.append((idx, f"JSON decode: {e}"))
            continue

        if validate_fn is not None:
            try:
                validated = validate_fn(parsed)
            except Exception as e:
                pre_recovery_failures.append(
                    (idx, f"validate_fn: {type(e).__name__}: {e}"))
                continue
        else:
            validated = parsed

        result_dict = dict(validated) if isinstance(validated, dict) else {"raw": validated}
        result_dict["in_tokens"] = in_tok
        result_dict["out_tokens"] = out_tok
        result_dict["via"] = "batch"
        successful[idx] = result_dict
        batched_in_tok += in_tok
        batched_out_tok += out_tok

    # Pairs missing from the output entirely → also recovery candidates.
    missing_idxs = [i for i in range(n_pairs) if i not in seen_idxs]
    for idx in missing_idxs:
        pre_recovery_failures.append((idx, "missing from batch output"))

    # Recovery — one real-time retry per failed pair.
    failed_pairs: list[dict] = []
    recovery_in_tok = 0
    recovery_out_tok = 0
    for idx, batch_err in pre_recovery_failures:
        messages = messages_list[idx]
        try:
            sys_prompt, usr_prompt = _split_messages_for_recovery(messages)
            raw_text, in_tok, out_tok = _call_openai_json(
                sys_prompt, usr_prompt, model=model, max_tokens=max_tokens,
            )
            parsed, repair_in, repair_out = _parse_json_with_repair(
                raw_text, sys_prompt, usr_prompt,
                provider="openai", model=model, max_tokens=max_tokens,
            )
            in_tok += repair_in
            out_tok += repair_out
            if validate_fn is not None:
                validated = validate_fn(parsed)
            else:
                validated = parsed
            result_dict = dict(validated) if isinstance(validated, dict) else {"raw": validated}
            result_dict["in_tokens"] = in_tok
            result_dict["out_tokens"] = out_tok
            result_dict["via"] = "recovery"
            successful[idx] = result_dict
            recovery_in_tok += in_tok
            recovery_out_tok += out_tok
            print(f"  [batch] pair_{idx} recovered via real-time "
                  f"(batch_err={batch_err!r})", file=sys.stderr)
        except Exception as e:
            failed_pairs.append({
                "custom_id": f"pair_{idx}",
                "index": idx,
                "batch_error": batch_err,
                "recovery_error": f"{type(e).__name__}: {e}",
            })
            print(f"  [batch] pair_{idx} FAILED FINAL — batch_err={batch_err!r}; "
                  f"recovery_err={type(e).__name__}: {e}", file=sys.stderr)

    # Cost summary
    batched_cost = (
        batched_in_tok * GPT5_BATCH_INPUT_PRICE_PER_M / 1_000_000.0
        + batched_out_tok * GPT5_BATCH_OUTPUT_PRICE_PER_M / 1_000_000.0
    )
    recovery_cost = (
        recovery_in_tok * GPT5_STANDARD_INPUT_PRICE_PER_M / 1_000_000.0
        + recovery_out_tok * GPT5_STANDARD_OUTPUT_PRICE_PER_M / 1_000_000.0
    )
    total_cost = batched_cost + recovery_cost
    n_succeeded = sum(1 for v in successful if v is not None and v.get("via") == "batch")
    n_recovered = sum(1 for v in successful if v is not None and v.get("via") == "recovery")
    n_failed_final = len(failed_pairs)
    print(f"[batch] {n_succeeded} succeeded, {n_recovered} "
          f"failed-then-recovered, {n_failed_final} failed-final | "
          f"batched_cost=${batched_cost:.4f} "
          f"recovery_cost=${recovery_cost:.4f} "
          f"total=${total_cost:.4f}", file=sys.stderr)

    return successful, failed_pairs


# ---------------------------------------------------------------------------
# Closed-enum coercion (graceful fallback if model returns near-miss)
# ---------------------------------------------------------------------------
def _coerce_modal(raw: object) -> str:
    if not isinstance(raw, str):
        return "no_modal_implicit"
    s = raw.strip().lower()
    if s in ALLOWED_MODALS:
        return s
    # Common near-misses
    aliases = {
        "shall not": "may not",   # treat shall-not as may-not (both prohibit)
        "must not": "must not",
        "required to": "is required to",
        "prohibited from": "is prohibited from",
        "permitted to": "is permitted to",
        "authorized to": "is authorized to",
        "no modal": "no_modal_implicit",
        "implicit": "no_modal_implicit",
        "": "no_modal_implicit",
    }
    return aliases.get(s, "no_modal_implicit")


def _coerce_enum(raw: object, allowed: set, default: str) -> str:
    if not isinstance(raw, str):
        return default
    s = raw.strip().lower()
    return s if s in allowed else default


def _coerce_optional_str(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        return None if s == "" or s.lower() == "null" else s
    return None


def _coerce_bool(raw: object, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1")
    if isinstance(raw, (int, float)):
        return bool(raw)
    return default


# ---------------------------------------------------------------------------
# Context extraction — pull ±3 sentences around a candidate's char offsets
# ---------------------------------------------------------------------------
def extract_context(
    binding_text: str,
    char_offset_start: int,
    char_offset_end: int,
    radius: int = CONTEXT_RADIUS_CHARS,
) -> str:
    """Pull ±radius characters around a candidate's offsets in the
    binding text. Caller is responsible for offsets being relative to the
    binding-text segment used at call time; the heuristic stores offsets
    relative to the WHOLE FR document, so callers should pass that here too.
    """
    start = max(0, char_offset_start - radius)
    end = min(len(binding_text), char_offset_end + radius)
    return binding_text[start:end].strip()


# ---------------------------------------------------------------------------
# Shared helpers for the primary + safety-net verification paths
#
# Two validator variants share the same module-level ALLOWED_* enum sets:
#
#   _coerce_verification_dict   — permissive, snaps near-misses to defaults
#                                 (used by the real-time `verify_candidate`)
#   _validated_verification_dict — strict, raises ValueError on any closed-
#                                 enum violation (used by the batch path
#                                 via `_call_openai_batch.validate_fn` so
#                                 pathological outputs route to recovery)
#
# Both return the same normalized dict shape; the downstream row builder
# `_verified_obligation_rows_from_validated` consumes that shape so the
# real-time and batch paths produce byte-identical VerifiedObligation rows.
# ---------------------------------------------------------------------------
def _coerce_verification_dict(parsed: dict) -> dict:
    """Permissive normalization for primary-verification JSON. Near-miss
    enum values get coerced to safe defaults rather than raising — this
    matches the inline behavior the real-time `verify_candidate` shipped
    before Task E."""
    if not isinstance(parsed, dict):
        parsed = {}
    obligations_raw = parsed.get("obligations") or []
    if not isinstance(obligations_raw, list):
        obligations_raw = []
    return {
        "verified": _coerce_bool(parsed.get("verified"), default=False),
        "verified_reason": str(parsed.get("verified_reason", "")).strip()[:500],
        "complexity": _coerce_enum(parsed.get("complexity"),
                                   ALLOWED_COMPLEXITIES, "simple"),
        "recital_disposition": _coerce_enum(parsed.get("recital_disposition"),
                                            ALLOWED_RECITAL_DISPOSITIONS, "none"),
        "obligations": [ob for ob in obligations_raw if isinstance(ob, dict)],
    }


def _validated_verification_dict(parsed: dict) -> dict:
    """Strict primary-verification schema validator. Raises ValueError on
    any closed-enum violation. Used by `_call_openai_batch.validate_fn`
    so pathological outputs trigger the wrapper's recovery path (which
    falls back to the real-time call, which then coerces gracefully).

    Net behavior end-to-end is the same as pure real-time, with the
    recovery layer catching pathological cases for observability and
    surfacing them in the failure summary."""
    if not isinstance(parsed, dict):
        raise ValueError(f"non-dict payload: {type(parsed).__name__}")

    verified = parsed.get("verified")
    if not isinstance(verified, bool):
        raise ValueError(f"verified must be bool, got {verified!r}")

    complexity = parsed.get("complexity")
    if not isinstance(complexity, str) or complexity not in ALLOWED_COMPLEXITIES:
        raise ValueError(f"complexity invalid: {complexity!r}")

    recital_disposition = parsed.get("recital_disposition")
    if (not isinstance(recital_disposition, str)
            or recital_disposition not in ALLOWED_RECITAL_DISPOSITIONS):
        raise ValueError(f"recital_disposition invalid: {recital_disposition!r}")

    obligations_raw = parsed.get("obligations")
    if obligations_raw is None:
        obligations_raw = []
    if not isinstance(obligations_raw, list):
        raise ValueError("obligations must be a list")

    validated_obs: list[dict] = []
    for ob in obligations_raw:
        if not isinstance(ob, dict):
            raise ValueError(f"obligation must be dict, got {type(ob).__name__}")
        modal_raw = ob.get("modal")
        if not isinstance(modal_raw, str):
            raise ValueError(f"modal must be string, got {modal_raw!r}")
        # Apply documented alias coercion BEFORE strict membership check
        # (e.g. "shall not" -> "may not"; "amend" -> "revise" downstream).
        # The alias map at _coerce_modal is the canonical design intent
        # per PROJECT_FACTS §9b; bypassing it makes the batch path stricter
        # than the documented schema. (Bug fix 2026-05-13: batch path was
        # rejecting "shall not" and "amendatory_action='amend'" which the
        # real-time path coerces gracefully — empirically ~50% lost rate
        # for "shall not" pairs in tonight's first batched production run.)
        modal = _coerce_modal(modal_raw)
        if modal not in ALLOWED_MODALS:
            raise ValueError(f"modal invalid (uncoercible): {modal_raw!r} -> {modal!r}")
        ob["modal"] = modal  # write canonical form back
        modal_strength = ob.get("modal_strength")
        if (not isinstance(modal_strength, str)
                or modal_strength not in ALLOWED_MODAL_STRENGTHS):
            raise ValueError(f"modal_strength invalid: {modal_strength!r}")
        amendatory_action = ob.get("amendatory_action")
        # amendatory_action design intent: row builder falls back to the
        # candidate's heuristic-derived amendatory_action via the pattern
        # `ob.get("amendatory_action") or candidate_row.get(...)` when
        # the LLM's value is null/missing/uncoercible.
        #
        # On uncoercible LLM input (e.g. modal value "amend" that's not
        # in ALLOWED_AMENDATORY_ACTIONS), write Python None back to
        # `ob` rather than the literal string "none". Writing the
        # truthy string "none" would defeat the row builder's `or`
        # fall-through and incorrectly collapse "LLM was uncertain"
        # with "explicitly not in an amendatory block."  (Fix follow-
        # ing 2026-05-13 audit finding; pre-existing latent bug also
        # present in the real-time `_coerce_verification_dict` path —
        # consider fixing there too in a future iteration.)
        if amendatory_action is not None:
            if not isinstance(amendatory_action, str):
                raise ValueError(f"amendatory_action must be string, got {amendatory_action!r}")
            s = amendatory_action.strip().lower()
            if s in ALLOWED_AMENDATORY_ACTIONS:
                ob["amendatory_action"] = s  # canonical lowercased form
            else:
                ob["amendatory_action"] = None  # let row builder fall through
        validated_obs.append(ob)

    return {
        "verified": verified,
        "verified_reason": str(parsed.get("verified_reason", "")).strip()[:500],
        "complexity": complexity,
        "recital_disposition": recital_disposition,
        "obligations": validated_obs,
    }


def _validated_safety_net_dict(parsed: dict) -> dict:
    """Strict safety-net schema validator (section-level find_missed
    output). Raises ValueError on closed-enum violations on `modal`."""
    if not isinstance(parsed, dict):
        raise ValueError(f"non-dict payload: {type(parsed).__name__}")
    missed_raw = parsed.get("missed_obligations")
    if missed_raw is None:
        missed_raw = []
    if not isinstance(missed_raw, list):
        raise ValueError("missed_obligations must be a list")
    validated_missed: list[dict] = []
    for m in missed_raw:
        if not isinstance(m, dict):
            raise ValueError(f"missed obligation must be dict, got {type(m).__name__}")
        modal_raw = m.get("modal")
        if not isinstance(modal_raw, str):
            raise ValueError(f"modal must be string, got {modal_raw!r}")
        # Apply alias coercion before strict check (same fix as
        # _validated_verification_dict; see comment there).
        modal = _coerce_modal(modal_raw)
        if modal not in ALLOWED_MODALS:
            raise ValueError(f"modal invalid (uncoercible): {modal_raw!r} -> {modal!r}")
        m["modal"] = modal  # write canonical form back
        validated_missed.append(m)
    return {"missed_obligations": validated_missed}


def _build_primary_messages(candidate_row: dict,
                            binding_text: str) -> list[dict]:
    """Construct the `[system, user]` messages list for one primary
    verification call. Used by both the real-time and batch paths so the
    prompt is byte-identical regardless of dispatch lane."""
    context_text = extract_context(
        binding_text,
        int(candidate_row["char_offset_start"]),
        int(candidate_row["char_offset_end"]),
    )
    user_prompt = PRIMARY_USER_TEMPLATE.format(
        candidate_sentence=candidate_row.get("obligation_text", ""),
        context_text=context_text,
        cfr_part=candidate_row.get("cfr_part") or "<unknown>",
        cfr_section=candidate_row.get("cfr_section") or "<unknown>",
        amendatory_action=candidate_row.get("amendatory_action", "none"),
        subject=candidate_row.get("subject", "<unknown>"),
        modal=candidate_row.get("modal", "<unknown>"),
        modal_strength=candidate_row.get("modal_strength", "strong"),
        action=candidate_row.get("action", "<unknown>"),
        object=candidate_row.get("object", ""),
        passive_voice=str(candidate_row.get("passive_voice", False)),
        is_conditional=str(candidate_row.get("is_conditional", False)),
        condition_text=candidate_row.get("condition_text") or "",
    )
    return [
        {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_safety_net_messages(section_text: str,
                               existing_obligations: list[dict]) -> list[dict]:
    """Construct the `[system, user]` messages list for one safety-net
    section-level call."""
    capped = section_text[:SAFETY_NET_SECTION_CHAR_CAP]
    existing_summary = [
        {"subject": e.get("subject", ""), "modal": e.get("modal", ""),
         "obligation_text": (e.get("obligation_text", "") or "")[:200]}
        for e in existing_obligations
    ]
    user_prompt = SAFETY_NET_USER_TEMPLATE.format(
        section_text=capped,
        json_existing=json.dumps(existing_summary, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": SAFETY_NET_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _verified_obligation_rows_from_validated(
    candidate_row: dict,
    validated: dict,
    *,
    found_via: str,
    provider: str,
    model: str,
    in_tok: int,
    out_tok: int,
) -> list["VerifiedObligation"]:
    """Build VerifiedObligation rows from either a coerced (real-time)
    or validated (batch) dict. Same logic on both paths so CSV output is
    byte-identical regardless of dispatch lane."""
    if not validated.get("verified") or not validated.get("obligations"):
        return []
    verified_reason = validated["verified_reason"]
    complexity = validated["complexity"]
    recital_disposition = validated["recital_disposition"]
    out_rows: list[VerifiedObligation] = []
    for split_idx, ob in enumerate(validated["obligations"]):
        if not isinstance(ob, dict):
            continue
        out_rows.append(VerifiedObligation(
            candidate_idx=int(candidate_row["obligation_idx"]),
            docket_id=str(candidate_row["docket_id"]),
            rule_type=str(candidate_row["rule_type"]),
            split_idx=split_idx,
            verified=True,
            verified_reason=verified_reason,
            complexity=complexity,
            recital_disposition=recital_disposition,
            cfr_part=_coerce_optional_str(candidate_row.get("cfr_part")),
            cfr_section=_coerce_optional_str(
                ob.get("cfr_section") or candidate_row.get("cfr_section")),
            subject=str(ob.get("subject", "<unknown>")).strip()[:300],
            modal=_coerce_modal(ob.get("modal")),
            modal_strength=_coerce_enum(ob.get("modal_strength"),
                                        ALLOWED_MODAL_STRENGTHS, "strong"),
            action=str(ob.get("action", "")).strip()[:300],
            object=str(ob.get("object", "")).strip()[:500],
            passive_voice=_coerce_bool(ob.get("passive_voice"), default=False),
            is_conditional=_coerce_bool(ob.get("is_conditional"), default=False),
            condition_text=_coerce_optional_str(ob.get("condition_text")),
            amendatory_action=_coerce_enum(
                ob.get("amendatory_action") or candidate_row.get("amendatory_action"),
                ALLOWED_AMENDATORY_ACTIONS, "none"),
            obligation_text=str(candidate_row.get("obligation_text", ""))[:1500],
            cross_reference=_coerce_optional_str(ob.get("cross_reference")),
            found_via=found_via,
            llm_provider=provider,
            llm_model=model,
            in_tokens=in_tok if split_idx == 0 else 0,
            out_tokens=out_tok if split_idx == 0 else 0,
        ))
    return out_rows


def _normalize_missed_obligation(m: dict) -> dict:
    """Mirror find_missed_obligations' per-missed normalization for the
    batch consumer."""
    return {
        "subject": str(m.get("subject", "")).strip()[:300],
        "modal": _coerce_modal(m.get("modal")),
        "action": str(m.get("action", "")).strip()[:300],
        "object": str(m.get("object", "")).strip()[:500],
        "obligation_text": str(m.get("obligation_text", "")).strip()[:1500],
        "found_via": str(m.get("found_via", "other")).strip()[:50],
    }


# ---------------------------------------------------------------------------
# Primary verification — one candidate at a time
# ---------------------------------------------------------------------------
def verify_candidate(
    candidate_row: dict,
    binding_text: str,
    provider: str = "openai",
    model: Optional[str] = None,
    found_via: str = "primary",
) -> tuple[list[VerifiedObligation], int, int]:
    """Run primary verification on one Stage 1a candidate via the
    REAL-TIME path. Returns (list_of_verified_rows, in_tokens, out_tokens).
    list_of_verified_rows has length 0 (verified=false), 1 (simple), or
    N (compound).

    Used by the per-candidate real-time loop in `run_primary_verification`
    when `batch=False`. The batch path (`_run_primary_batch`) shares the
    prompt builder, validator, and row builder helpers so both paths
    produce byte-identical VerifiedObligation rows.
    """
    if model is None:
        model = (DEFAULT_OPENAI_MODEL if provider == "openai"
                 else DEFAULT_ANTHROPIC_MODEL)

    messages = _build_primary_messages(candidate_row, binding_text)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    raw_text, in_tok, out_tok = _call_llm_json(
        system_prompt, user_prompt,
        provider=provider, model=model,
        max_tokens=PRIMARY_MAX_TOKENS,
    )
    parsed, repair_in, repair_out = _parse_json_with_repair(
        raw_text, system_prompt, user_prompt,
        provider=provider, model=model,
        max_tokens=PRIMARY_MAX_TOKENS,
    )
    in_tok += repair_in
    out_tok += repair_out

    coerced = _coerce_verification_dict(parsed)
    out_rows = _verified_obligation_rows_from_validated(
        candidate_row, coerced,
        found_via=found_via, provider=provider, model=model,
        in_tok=in_tok, out_tok=out_tok,
    )
    return (out_rows, in_tok, out_tok)


# ---------------------------------------------------------------------------
# Safety net — section-level call, then second-pass verification (Major 5)
# ---------------------------------------------------------------------------
def find_missed_obligations(
    section_text: str,
    existing_obligations: list[dict],
    provider: str = "openai",
    model: Optional[str] = None,
) -> tuple[list[dict], int, int]:
    """REAL-TIME safety-net call on one section. Returns
    (list_of_missed, in_tok, out_tok). Each missed-obligation dict has:
    subject, modal, action, object, obligation_text, found_via.

    The batch path (`_run_safety_net_batch`) shares the message builder
    and the per-missed normalizer so both paths produce byte-identical
    missed-obligation dicts."""
    if model is None:
        model = (DEFAULT_OPENAI_MODEL if provider == "openai"
                 else DEFAULT_ANTHROPIC_MODEL)

    messages = _build_safety_net_messages(section_text, existing_obligations)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    raw_text, in_tok, out_tok = _call_llm_json(
        system_prompt, user_prompt,
        provider=provider, model=model,
        max_tokens=SAFETYNET_MAX_TOKENS,
    )
    parsed, repair_in, repair_out = _parse_json_with_repair(
        raw_text, system_prompt, user_prompt,
        provider=provider, model=model,
        max_tokens=SAFETYNET_MAX_TOKENS,
    )
    in_tok += repair_in
    out_tok += repair_out

    missed_raw = parsed.get("missed_obligations") or []
    missed = [_normalize_missed_obligation(m) for m in missed_raw
              if isinstance(m, dict)]
    return missed, in_tok, out_tok


def verify_safety_net_hits(
    missed_obligations: list[dict],
    section_text: str,
    section_amendatory_action: str,
    section_cfr_part: Optional[str],
    section_cfr_section: Optional[str],
    docket_id: str,
    rule_type: str,
    provider: str = "openai",
    model: Optional[str] = None,
    starting_candidate_idx: int = 100000,
) -> tuple[list[VerifiedObligation], int, int]:
    """Major 5 fix: run each safety-net hit back through the primary
    verification prompt. Eliminates the v1 asymmetry where safety-net
    outputs got fewer LLM passes than primary candidates.

    `starting_candidate_idx` ensures safety-net rows have non-overlapping
    candidate_idx values vs. primary candidates (default 100000+ floor)."""
    out_rows: list[VerifiedObligation] = []
    total_in = 0
    total_out = 0
    for i, m in enumerate(missed_obligations):
        # Build a synthetic candidate row matching the heuristic schema
        synthetic = {
            "obligation_idx": starting_candidate_idx + i,
            "docket_id": docket_id,
            "rule_type": rule_type,
            "cfr_part": section_cfr_part,
            "cfr_section": section_cfr_section,
            "subject": m["subject"],
            "modal": m["modal"],
            "modal_strength": "strong",  # safety-net default; LLM will correct
            "action": m["action"],
            "object": m["object"],
            "amendatory_action": section_amendatory_action,
            "is_conditional": False,
            "condition_text": "",
            "passive_voice": False,
            "obligation_text": m["obligation_text"],
            # Use start-of-section as conservative offset; context will use
            # the section text directly
            "char_offset_start": 0,
            "char_offset_end": min(len(m["obligation_text"]), len(section_text)),
        }
        verified_rows, ti, to = verify_candidate(
            synthetic, section_text,
            provider=provider, model=model,
            found_via="safety_net_verified",
        )
        total_in += ti
        total_out += to
        out_rows.extend(verified_rows)
    return out_rows, total_in, total_out


# ---------------------------------------------------------------------------
# Batch helpers for the primary + safety-net passes (Task E)
# ---------------------------------------------------------------------------
def _run_primary_batch(
    candidates: list[dict],
    binding_text: str,
    *,
    model: str,
    provider: str,
    max_cost_usd: float,
    call_batch=None,
) -> tuple[list[VerifiedObligation], int, int, dict]:
    """Run primary verification on all candidates as a single batch.
    Returns (verified_rows, total_in_tok, total_out_tok, summary).

    Args:
        candidates: Stage 1a candidate dicts (in submission order).
        binding_text: full FR doc — used for ±context extraction.
        model, provider: routed through to the OpenAI batch call.
        max_cost_usd: hard cap; pre-check raises CostCapExceeded.
        call_batch: injectable for tests; defaults to `_call_openai_batch`.

    The batch validator is `_validated_verification_dict` (strict); any
    pathological row is routed to the wrapper's recovery path (real-time).
    """
    if call_batch is None:
        call_batch = _call_openai_batch
    if not candidates:
        return [], 0, 0, {
            "n_batch_succeeded": 0, "n_recovered": 0, "n_failed": 0,
            "verified_candidates": 0, "rejected_candidates": 0,
            "compound_candidates": 0,
        }

    messages_list = [_build_primary_messages(c, binding_text) for c in candidates]
    successful, failed_pairs = call_batch(
        messages_list, model, max_cost_usd,
        batch_id_prefix="stage1b_primary",
        max_tokens=PRIMARY_MAX_TOKENS,
        validate_fn=_validated_verification_dict,
    )

    verified_rows: list[VerifiedObligation] = []
    total_in = total_out = 0
    n_succeeded = n_recovered = 0
    n_verified = n_rejected = n_compound = 0

    for i, result in enumerate(successful):
        if result is None:
            n_rejected += 1   # treated like a rejected candidate
            continue
        in_tok = int(result.get("in_tokens", 0) or 0)
        out_tok = int(result.get("out_tokens", 0) or 0)
        total_in += in_tok
        total_out += out_tok
        if result.get("via") == "recovery":
            n_recovered += 1
        else:
            n_succeeded += 1
        rows = _verified_obligation_rows_from_validated(
            candidates[i], result,
            found_via="primary", provider=provider, model=model,
            in_tok=in_tok, out_tok=out_tok,
        )
        if rows:
            n_verified += 1
            if rows[0].complexity == "compound":
                n_compound += 1
            verified_rows.extend(rows)
        else:
            n_rejected += 1

    summary = {
        "n_batch_succeeded": n_succeeded,
        "n_recovered": n_recovered,
        "n_failed": len(failed_pairs),
        "verified_candidates": n_verified,
        "rejected_candidates": n_rejected,
        "compound_candidates": n_compound,
    }
    return verified_rows, total_in, total_out, summary


def _run_safety_net_batch(
    binding_text: str,
    primary_verified_rows: list[VerifiedObligation],
    docket_id: str,
    rule_type: str,
    *,
    model: str,
    provider: str,
    max_cost_usd: float,
    call_batch=None,
) -> tuple[list[VerifiedObligation], int, int, dict]:
    """Run the section-level safety-net pass + the second-pass missed-
    obligation verification, each as its own batch. Returns
    (verified_rows, total_in_tok, total_out_tok, summary).

    Phase 1: batch all amendatory-block `find_missed` calls.
    Phase 2: batch all synthetic-candidate verifications (using the same
             primary-verification prompt as `verify_candidate`).
    """
    from path_a_obligation_heuristic import (
        split_preamble_binding, find_amendatory_blocks, _last_cfr_anchor_before,
    )
    if call_batch is None:
        call_batch = _call_openai_batch

    _, binding_only = split_preamble_binding(binding_text)
    blocks = find_amendatory_blocks(binding_only)

    # Group primary verified rows by section so the safety-net prompt
    # can be told what's already been found.
    from collections import defaultdict
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for r in primary_verified_rows:
        sec = r.cfr_section or "<unknown>"
        by_sec[sec].append({
            "subject": r.subject, "modal": r.modal,
            "obligation_text": r.obligation_text,
        })

    # Phase 1 — section-level find_missed batch.
    block_meta: list[tuple[int, int, int, str, str, str]] = []
    sn_messages: list[list[dict]] = []
    for block_i, (s, e, action) in enumerate(blocks):
        if action == "none":
            continue
        section_text = binding_only[s:e]
        if len(section_text.strip()) < 50:
            continue
        cfr_part, cfr_section = _last_cfr_anchor_before(binding_only, s + 50)
        existing = by_sec.get(cfr_section or "<unknown>", [])
        block_meta.append((block_i, s, e, action, cfr_part or "", cfr_section or ""))
        sn_messages.append(_build_safety_net_messages(section_text, existing))

    summary = {
        "n_section_calls": len(sn_messages),
        "n_section_recovered": 0,
        "n_section_failed": 0,
        "safety_net_hits": 0,
        "safety_net_verified": 0,
        "n_verify_recovered": 0,
        "n_verify_failed": 0,
    }
    if not sn_messages:
        return [], 0, 0, summary

    successful_sn, failed_sn = call_batch(
        sn_messages, model, max_cost_usd,
        batch_id_prefix="stage1b_safety_net",
        max_tokens=SAFETYNET_MAX_TOKENS,
        validate_fn=_validated_safety_net_dict,
    )
    summary["n_section_failed"] = len(failed_sn)

    total_in = total_out = 0
    synthetic_candidates: list[dict] = []
    section_texts: list[str] = []
    starting_sn_idx = 100000
    for sn_i, result in enumerate(successful_sn):
        if result is None:
            continue
        if result.get("via") == "recovery":
            summary["n_section_recovered"] += 1
        in_tok = int(result.get("in_tokens", 0) or 0)
        out_tok = int(result.get("out_tokens", 0) or 0)
        total_in += in_tok
        total_out += out_tok
        block_i, s, e, action, cfr_part, cfr_section = block_meta[sn_i]
        section_text = binding_only[s:e]
        missed = [_normalize_missed_obligation(m)
                  for m in result.get("missed_obligations", [])]
        summary["safety_net_hits"] += len(missed)
        for j, m in enumerate(missed):
            synthetic = {
                "obligation_idx": starting_sn_idx + block_i * 100 + j,
                "docket_id": docket_id, "rule_type": rule_type,
                "cfr_part": cfr_part or None,
                "cfr_section": cfr_section or None,
                "subject": m["subject"], "modal": m["modal"],
                "modal_strength": "strong",
                "action": m["action"], "object": m["object"],
                "amendatory_action": action,
                "is_conditional": False, "condition_text": "",
                "passive_voice": False,
                "obligation_text": m["obligation_text"],
                "char_offset_start": 0,
                "char_offset_end": min(len(m["obligation_text"]),
                                       len(section_text)),
            }
            synthetic_candidates.append(synthetic)
            section_texts.append(section_text)
        if missed:
            print(f"  [safety-net-batch] §{cfr_section}: "
                  f"{len(missed)} missed", file=sys.stderr)

    if not synthetic_candidates:
        return [], total_in, total_out, summary

    # Phase 2 — verify the missed obligations as a batch (uses the
    # SAME primary-verification prompt + validator as Phase 1 of the
    # primary pass).
    verify_messages = [
        _build_primary_messages(c, section_texts[i])
        for i, c in enumerate(synthetic_candidates)
    ]
    successful_v, failed_v = call_batch(
        verify_messages, model, max_cost_usd,
        batch_id_prefix="stage1b_safety_net_verify",
        max_tokens=PRIMARY_MAX_TOKENS,
        validate_fn=_validated_verification_dict,
    )
    summary["n_verify_failed"] = len(failed_v)

    verified_rows: list[VerifiedObligation] = []
    for i, result in enumerate(successful_v):
        if result is None:
            continue
        if result.get("via") == "recovery":
            summary["n_verify_recovered"] += 1
        in_tok = int(result.get("in_tokens", 0) or 0)
        out_tok = int(result.get("out_tokens", 0) or 0)
        total_in += in_tok
        total_out += out_tok
        rows = _verified_obligation_rows_from_validated(
            synthetic_candidates[i], result,
            found_via="safety_net_verified",
            provider=provider, model=model,
            in_tok=in_tok, out_tok=out_tok,
        )
        verified_rows.extend(rows)

    summary["safety_net_verified"] = len(verified_rows)
    return verified_rows, total_in, total_out, summary


# ---------------------------------------------------------------------------
# Concurrent real-time helpers (Task H, 2026-05-13)
#
# Dispatches multiple `verify_candidate` (or `find_missed_obligations`)
# calls in parallel via asyncio + `asyncio.to_thread`. The sync per-call
# function is unmodified — `to_thread` runs each on a separate worker
# thread so the OpenAI HTTP client's blocking I/O multiplexes naturally.
# This preserves all the existing retry/backoff semantics inside
# `_call_openai_json` (rate-limit handling, BadRequestError fallback,
# etc.) without re-implementing them against `AsyncOpenAI`.
#
# Throughput anchor (PROJECT_FACTS §7): gpt-5 Tier-4 TPM ~800K with
# ~1,847 tokens/call → ~430 calls/min sustained ceiling. Default
# `--concurrency 20` keeps us at ~90K TPM (well under the ceiling) and
# lets the user dial up via the CLI flag.
# ---------------------------------------------------------------------------
# Per-call cost anchor for the concurrent-realtime pre-check. Matches
# Task H spec ($0.0038/call real-time).
REALTIME_ESTIMATED_PER_CALL_USD = 0.0038


def _run_primary_realtime_concurrent(
    candidates: list[dict],
    binding_text: str,
    *,
    model: str,
    provider: str,
    concurrency: int,
    max_cost_usd: float,
    verify_fn=None,
    progress_every: int = 25,
) -> tuple[list[VerifiedObligation], list[dict], int, int, dict]:
    """Run primary verification on all candidates with up to `concurrency`
    in-flight real-time calls. Returns (verified_rows, failed_pairs,
    total_in_tok, total_out_tok, summary). Submission order is preserved
    in `verified_rows` because we iterate the gather results in order.

    Args:
        verify_fn: injectable for tests; defaults to `verify_candidate`.
            Must accept (candidate_row, binding_text, *, provider, model,
            found_via) and return (rows_list, in_tok, out_tok).
    """
    if verify_fn is None:
        verify_fn = verify_candidate
    if not candidates:
        return [], [], 0, 0, {
            "verified_candidates": 0, "rejected_candidates": 0,
            "compound_candidates": 0, "n_failed": 0,
        }

    # Cost-cap pre-check — fire BEFORE any API call.
    estimated_total = len(candidates) * REALTIME_ESTIMATED_PER_CALL_USD
    if estimated_total > max_cost_usd:
        raise CostCapExceeded(
            f"Concurrent real-time pre-check: estimated cost "
            f"${estimated_total:,.2f} for {len(candidates):,} candidates "
            f"(anchor ${REALTIME_ESTIMATED_PER_CALL_USD:.4f}/call) exceeds "
            f"cap ${max_cost_usd:,.2f}. No API call made."
        )

    import asyncio

    async def _run() -> list[tuple]:
        sem = asyncio.Semaphore(concurrency)
        n_done = [0]

        async def _one(idx: int, cand: dict):
            async with sem:
                try:
                    result = await asyncio.to_thread(
                        verify_fn, cand, binding_text,
                        provider=provider, model=model, found_via="primary",
                    )
                    n_done[0] += 1
                    if progress_every and n_done[0] % progress_every == 0:
                        print(f"  [primary-concurrent] {n_done[0]}/"
                              f"{len(candidates)} done", file=sys.stderr)
                    return ("ok", idx, result)
                except Exception as e:  # noqa: BLE001
                    n_done[0] += 1
                    return ("fail", idx, e)

        tasks = [asyncio.create_task(_one(i, c))
                 for i, c in enumerate(candidates)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())

    verified_rows: list[VerifiedObligation] = []
    failed_pairs: list[dict] = []
    total_in = 0
    total_out = 0
    n_verified = 0
    n_rejected = 0
    n_compound = 0

    # asyncio.gather returns results in submission order even when
    # completion is out-of-order; iterate in that order so verified_rows
    # is also order-stable.
    for r in results:
        status, idx, payload = r
        if status == "fail":
            cand = candidates[idx]
            failed_pairs.append({
                "obligation_idx": cand.get("obligation_idx"),
                "index": idx,
                "error": f"{type(payload).__name__}: {payload}",
            })
            print(f"  [primary-concurrent] candidate "
                  f"#{cand.get('obligation_idx')}: "
                  f"{type(payload).__name__}: {payload}", file=sys.stderr)
            continue
        rows, ti, to = payload
        total_in += ti
        total_out += to
        if rows:
            n_verified += 1
            if rows[0].complexity == "compound":
                n_compound += 1
            verified_rows.extend(rows)
        else:
            n_rejected += 1

    summary = {
        "verified_candidates": n_verified,
        "rejected_candidates": n_rejected,
        "compound_candidates": n_compound,
        "n_failed": len(failed_pairs),
    }
    print(f"  [primary-concurrent] N={n_verified + n_rejected} succeeded, "
          f"M={len(failed_pairs)} failed-after-retries | "
          f"in_tok={total_in:,} out_tok={total_out:,}", file=sys.stderr)
    return verified_rows, failed_pairs, total_in, total_out, summary


def _run_safety_net_realtime_concurrent(
    binding_text: str,
    primary_verified_rows: list[VerifiedObligation],
    docket_id: str,
    rule_type: str,
    *,
    model: str,
    provider: str,
    concurrency: int,
    max_cost_usd: float,
    find_missed_fn=None,
    verify_fn=None,
) -> tuple[list[VerifiedObligation], list[dict], int, int, dict]:
    """Concurrent counterpart of `_run_safety_net_batch` for the real-
    time path. Phase 1 (find_missed per amendatory block) and Phase 2
    (per-missed second-pass verify) are each dispatched as their own
    semaphore-bounded gather. Returns (verified_rows, failed_pairs,
    in_tok, out_tok, summary).

    Args:
        find_missed_fn: injectable for tests; defaults to
            `find_missed_obligations`.
        verify_fn: injectable for tests; defaults to `verify_candidate`.
    """
    from path_a_obligation_heuristic import (
        split_preamble_binding, find_amendatory_blocks, _last_cfr_anchor_before,
    )
    import asyncio

    if find_missed_fn is None:
        find_missed_fn = find_missed_obligations
    if verify_fn is None:
        verify_fn = verify_candidate

    _, binding_only = split_preamble_binding(binding_text)
    blocks = find_amendatory_blocks(binding_only)

    from collections import defaultdict
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for r in primary_verified_rows:
        sec = r.cfr_section or "<unknown>"
        by_sec[sec].append({
            "subject": r.subject, "modal": r.modal,
            "obligation_text": r.obligation_text,
        })

    # Phase 1 plan: one task per amendatory block.
    block_meta: list[tuple] = []
    for block_i, (s, e, action) in enumerate(blocks):
        if action == "none":
            continue
        section_text = binding_only[s:e]
        if len(section_text.strip()) < 50:
            continue
        cfr_part, cfr_section = _last_cfr_anchor_before(binding_only, s + 50)
        existing = by_sec.get(cfr_section or "<unknown>", [])
        block_meta.append((block_i, s, e, action, cfr_part or "",
                           cfr_section or "", section_text, existing))

    summary = {
        "n_section_calls": len(block_meta),
        "n_section_failed": 0,
        "safety_net_hits": 0,
        "safety_net_verified": 0,
        "n_verify_failed": 0,
    }

    if not block_meta:
        return [], [], 0, 0, summary

    # Cost-cap pre-check covers Phase 1 + Phase 2 in the worst case.
    # Phase 2 is unbounded a priori — we accept the under-estimate and
    # rely on max_cost_usd being the user's hard ceiling at the
    # orchestrator level (run_one_target enforces it across anchors).
    estimated_phase_1 = len(block_meta) * REALTIME_ESTIMATED_PER_CALL_USD
    if estimated_phase_1 > max_cost_usd:
        raise CostCapExceeded(
            f"Safety-net Phase 1 pre-check: estimated cost "
            f"${estimated_phase_1:,.2f} for {len(block_meta)} blocks "
            f"exceeds cap ${max_cost_usd:,.2f}."
        )

    failed_pairs: list[dict] = []
    total_in = 0
    total_out = 0

    async def _phase1() -> list[tuple]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(idx: int, meta: tuple):
            block_i, s, e, action, cfr_part, cfr_section, section_text, existing = meta
            async with sem:
                try:
                    missed, ti, to = await asyncio.to_thread(
                        find_missed_fn,
                        section_text, existing,
                        provider=provider, model=model,
                    )
                    return ("ok", idx, missed, ti, to)
                except Exception as ex:  # noqa: BLE001
                    return ("fail", idx, ex)

        tasks = [asyncio.create_task(_one(i, m))
                 for i, m in enumerate(block_meta)]
        return await asyncio.gather(*tasks)

    p1_results = asyncio.run(_phase1())

    synthetic_candidates: list[dict] = []
    section_texts: list[str] = []
    starting_sn_idx = 100000
    for r in p1_results:
        status, idx, *rest = r
        if status == "fail":
            summary["n_section_failed"] += 1
            block_i = block_meta[idx][0]
            cfr_section = block_meta[idx][5]
            err = rest[0]
            print(f"  [safety-net-concurrent] block {block_i} "
                  f"(§{cfr_section}): {type(err).__name__}: {err}",
                  file=sys.stderr)
            failed_pairs.append({
                "phase": "find_missed", "block_index": block_i,
                "error": f"{type(err).__name__}: {err}",
            })
            continue
        missed, ti, to = rest
        total_in += ti
        total_out += to
        block_i, s, e, action, cfr_part, cfr_section, section_text, _existing = block_meta[idx]
        summary["safety_net_hits"] += len(missed)
        for j, m in enumerate(missed):
            synthetic_candidates.append({
                "obligation_idx": starting_sn_idx + block_i * 100 + j,
                "docket_id": docket_id, "rule_type": rule_type,
                "cfr_part": cfr_part or None,
                "cfr_section": cfr_section or None,
                "subject": m["subject"], "modal": m["modal"],
                "modal_strength": "strong",
                "action": m["action"], "object": m["object"],
                "amendatory_action": action,
                "is_conditional": False, "condition_text": "",
                "passive_voice": False,
                "obligation_text": m["obligation_text"],
                "char_offset_start": 0,
                "char_offset_end": min(len(m["obligation_text"]),
                                       len(section_text)),
            })
            section_texts.append(section_text)

    if not synthetic_candidates:
        return [], failed_pairs, total_in, total_out, summary

    # Phase 2 — verify each missed obligation in parallel.
    async def _phase2() -> list[tuple]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(idx: int, cand: dict, section: str):
            async with sem:
                try:
                    result = await asyncio.to_thread(
                        verify_fn, cand, section,
                        provider=provider, model=model,
                        found_via="safety_net_verified",
                    )
                    return ("ok", idx, result)
                except Exception as ex:  # noqa: BLE001
                    return ("fail", idx, ex)

        tasks = [asyncio.create_task(_one(i, c, section_texts[i]))
                 for i, c in enumerate(synthetic_candidates)]
        return await asyncio.gather(*tasks)

    p2_results = asyncio.run(_phase2())
    verified_rows: list[VerifiedObligation] = []
    for r in p2_results:
        status, idx, payload = r
        if status == "fail":
            summary["n_verify_failed"] += 1
            failed_pairs.append({
                "phase": "verify_missed",
                "synthetic_idx": synthetic_candidates[idx]["obligation_idx"],
                "error": f"{type(payload).__name__}: {payload}",
            })
            continue
        rows, ti, to = payload
        total_in += ti
        total_out += to
        verified_rows.extend(rows)

    summary["safety_net_verified"] = len(verified_rows)
    print(f"  [safety-net-concurrent] section_calls={summary['n_section_calls']} "
          f"section_failed={summary['n_section_failed']}  "
          f"hits={summary['safety_net_hits']}  "
          f"verified={summary['safety_net_verified']}  "
          f"verify_failed={summary['n_verify_failed']}",
          file=sys.stderr)
    return verified_rows, failed_pairs, total_in, total_out, summary


# ---------------------------------------------------------------------------
# Cross-LLM consistency comparison
# ---------------------------------------------------------------------------
def _join_field(rows: list[VerifiedObligation], field_name: str) -> str:
    return "; ".join(getattr(r, field_name) or "" for r in rows)


def compare_cross_llm(
    candidate_row: dict,
    primary_rows: list[VerifiedObligation],
    primary_provider: str,
    primary_model: str,
    secondary_provider: str = "anthropic",
    secondary_model: Optional[str] = None,
) -> tuple[CrossLLMRow, list[VerifiedObligation], int, int]:
    """Run the secondary LLM on the same candidate; return a CrossLLMRow
    capturing per-obligation field agreement, plus the secondary-rows
    list and its token usage (for cost accounting)."""
    if secondary_model is None:
        secondary_model = DEFAULT_ANTHROPIC_MODEL

    binding_text = candidate_row.get("__binding_text__", "")
    secondary_rows, ti, to = verify_candidate(
        candidate_row, binding_text,
        provider=secondary_provider, model=secondary_model,
        found_via="primary",
    )

    primary_verified = bool(primary_rows)
    secondary_verified = bool(secondary_rows)
    primary_complexity = primary_rows[0].complexity if primary_rows else "n/a"
    secondary_complexity = secondary_rows[0].complexity if secondary_rows else "n/a"

    return (CrossLLMRow(
        candidate_idx=int(candidate_row["obligation_idx"]),
        docket_id=str(candidate_row["docket_id"]),
        rule_type=str(candidate_row["rule_type"]),
        primary_provider=primary_provider,
        primary_model=primary_model,
        secondary_provider=secondary_provider,
        secondary_model=secondary_model,
        primary_verified=primary_verified,
        secondary_verified=secondary_verified,
        verified_match=(primary_verified == secondary_verified),
        primary_complexity=primary_complexity,
        secondary_complexity=secondary_complexity,
        complexity_match=(primary_complexity == secondary_complexity),
        primary_subjects=_join_field(primary_rows, "subject"),
        secondary_subjects=_join_field(secondary_rows, "subject"),
        primary_modals=_join_field(primary_rows, "modal"),
        secondary_modals=_join_field(secondary_rows, "modal"),
        primary_actions=_join_field(primary_rows, "action"),
        secondary_actions=_join_field(secondary_rows, "action"),
        primary_objects=_join_field(primary_rows, "object"),
        secondary_objects=_join_field(secondary_rows, "object"),
    ), secondary_rows, ti, to)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def _read_candidates_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # cast booleans
            row["passive_voice"] = row.get("passive_voice", "").lower() == "true"
            row["is_conditional"] = row.get("is_conditional", "").lower() == "true"
            rows.append(row)
    return rows


def _write_verified_csv(rows: list[VerifiedObligation], path: Path) -> int:
    if not rows:
        fieldnames = list(VerifiedObligation.__dataclass_fields__.keys())
    else:
        fieldnames = list(asdict(rows[0]).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return len(rows)


def _write_verified_dicts_csv(rows: list[dict], path: Path) -> int:
    """Write already-dict-shaped verified rows. Used after the dedup pass
    so we don't have to round-trip back through VerifiedObligation."""
    fieldnames = list(VerifiedObligation.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return len(rows)


def _write_xllm_csv(rows: list[CrossLLMRow], path: Path) -> int:
    if not rows:
        fieldnames = list(CrossLLMRow.__dataclass_fields__.keys())
    else:
        fieldnames = list(asdict(rows[0]).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return len(rows)


# ---------------------------------------------------------------------------
# Orchestrator entry points
# ---------------------------------------------------------------------------
def run_primary_verification(
    candidates_csv: Path,
    binding_text_path: Path,
    out_csv: Path,
    provider: str,
    model: Optional[str],
    limit: int = 0,
    run_safety_net: bool = True,
    *,
    batch: bool = True,
    max_cost_usd: float = 1500.0,
    call_batch=None,
    concurrency: int = 1,
) -> dict:
    """Run primary verification on every Stage 1a candidate, then (if
    run_safety_net=True) run the section-level safety net + second-pass
    verification on safety-net hits. Writes merged results to CSV.

    Args:
        ...
        batch: When True (default), submit candidates through the OpenAI
            Batch API via `_call_openai_batch`. When False, fall back to
            the per-candidate real-time loop. Cost accounting is honest
            in both modes — the per-candidate token totals come from the
            wrapper's per-row usage report.
        max_cost_usd: hard cap for the batch pre-check. Ignored on the
            real-time path (which has its own per-anchor budgeting via
            the orchestrator's cumulative-cost tracker).
        call_batch: injectable for tests; defaults to `_call_openai_batch`.

    Per v2 spec § Stage 1b: safety net catches obligations the heuristic's
    modal-anchor pass misses (e.g., implicit-modal CFR auditor instructions
    that use imperatives like 'Obtain', 'Select', 'Compute' without
    'must'/'shall')."""
    from path_a_obligation_heuristic import (
        split_preamble_binding, find_amendatory_blocks, _last_cfr_anchor_before,
    )

    candidates = _read_candidates_csv(candidates_csv)
    if limit > 0:
        candidates = candidates[:limit]
    binding_text = binding_text_path.read_text(encoding="utf-8", errors="ignore")

    verified_rows: list[VerifiedObligation] = []
    total_in = 0
    total_out = 0
    n_verified = 0
    n_rejected = 0
    n_compound = 0

    if batch:
        # ---- PRIMARY pass (batched) ----
        primary_rows, ti, to, primary_summary = _run_primary_batch(
            candidates, binding_text,
            model=model or DEFAULT_OPENAI_MODEL, provider=provider,
            max_cost_usd=max_cost_usd, call_batch=call_batch,
        )
        verified_rows.extend(primary_rows)
        total_in += ti
        total_out += to
        n_verified = primary_summary["verified_candidates"]
        n_rejected = primary_summary["rejected_candidates"]
        n_compound = primary_summary["compound_candidates"]
        print(f"  [primary-batch] verified={n_verified}  rejected={n_rejected}  "
              f"compound={n_compound}  recovered={primary_summary['n_recovered']}  "
              f"failed={primary_summary['n_failed']}  "
              f"in_tok={total_in:,} out_tok={total_out:,}", file=sys.stderr)
    elif concurrency > 1:
        # ---- PRIMARY pass (concurrent real-time, Task H) ----
        primary_rows, _failed, ti, to, primary_summary = (
            _run_primary_realtime_concurrent(
                candidates, binding_text,
                model=model or DEFAULT_OPENAI_MODEL, provider=provider,
                concurrency=concurrency, max_cost_usd=max_cost_usd,
            )
        )
        verified_rows.extend(primary_rows)
        total_in += ti
        total_out += to
        n_verified = primary_summary["verified_candidates"]
        n_rejected = primary_summary["rejected_candidates"]
        n_compound = primary_summary["compound_candidates"]
    else:
        # ---- PRIMARY pass (sequential real-time, concurrency=1) ----
        for i, cand in enumerate(candidates, 1):
            try:
                rows, ti, to = verify_candidate(
                    cand, binding_text,
                    provider=provider, model=model,
                    found_via="primary",
                )
            except Exception as e:
                print(f"  [primary] candidate #{cand.get('obligation_idx')}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                continue
            total_in += ti
            total_out += to
            if rows:
                n_verified += 1
                if rows[0].complexity == "compound":
                    n_compound += 1
                verified_rows.extend(rows)
            else:
                n_rejected += 1
            if i % 25 == 0 or i == len(candidates):
                print(f"  [primary] {i}/{len(candidates)} candidates  "
                      f"verified={n_verified}  rejected={n_rejected}  "
                      f"compound={n_compound}  in_tok={total_in:,}  out_tok={total_out:,}")

    # ---- SAFETY NET pass (Major 5 fix from v2 spec) ----
    n_safety_net_hits = 0
    n_safety_net_verified = 0
    if run_safety_net and batch:
        sn_rows, ti_sn, to_sn, sn_summary = _run_safety_net_batch(
            binding_text, verified_rows,
            docket_id=str(candidates[0].get("docket_id", "")) if candidates else "",
            rule_type=str(candidates[0].get("rule_type", "final")) if candidates else "final",
            model=model or DEFAULT_OPENAI_MODEL, provider=provider,
            max_cost_usd=max_cost_usd, call_batch=call_batch,
        )
        verified_rows.extend(sn_rows)
        total_in += ti_sn
        total_out += to_sn
        n_safety_net_hits = sn_summary["safety_net_hits"]
        n_safety_net_verified = sn_summary["safety_net_verified"]
        print(f"  [safety-net-batch] section_calls={sn_summary['n_section_calls']}  "
              f"hits={n_safety_net_hits}  verified={n_safety_net_verified}  "
              f"section_recovered={sn_summary['n_section_recovered']}  "
              f"verify_recovered={sn_summary['n_verify_recovered']}",
              file=sys.stderr)
    elif run_safety_net and concurrency > 1:
        # Concurrent real-time safety-net pass (Task H).
        sn_rows, _sn_failed, ti_sn, to_sn, sn_summary = (
            _run_safety_net_realtime_concurrent(
                binding_text, verified_rows,
                docket_id=str(candidates[0].get("docket_id", "")) if candidates else "",
                rule_type=str(candidates[0].get("rule_type", "final")) if candidates else "final",
                model=model or DEFAULT_OPENAI_MODEL, provider=provider,
                concurrency=concurrency, max_cost_usd=max_cost_usd,
            )
        )
        verified_rows.extend(sn_rows)
        total_in += ti_sn
        total_out += to_sn
        n_safety_net_hits = sn_summary["safety_net_hits"]
        n_safety_net_verified = sn_summary["safety_net_verified"]
    elif run_safety_net:
        # Sequential real-time safety-net pass (concurrency=1, unchanged).
        preamble, binding_only = split_preamble_binding(binding_text)
        blocks = find_amendatory_blocks(binding_only)

        from collections import defaultdict
        by_sec: dict[str, list[dict]] = defaultdict(list)
        for r in verified_rows:
            sec = r.cfr_section or "<unknown>"
            by_sec[sec].append({
                "subject": r.subject, "modal": r.modal,
                "obligation_text": r.obligation_text,
            })

        starting_sn_idx = 100000
        for block_i, (s, e, action) in enumerate(blocks):
            if action == "none":
                continue
            section_text = binding_only[s:e]
            if len(section_text.strip()) < 50:
                continue
            cfr_part, cfr_section = _last_cfr_anchor_before(binding_only, s + 50)
            existing = by_sec.get(cfr_section or "<unknown>", [])

            try:
                missed, ti_sn, to_sn = find_missed_obligations(
                    section_text, existing,
                    provider=provider, model=model,
                )
            except Exception as ex:
                print(f"  [safety-net] block {block_i} (§{cfr_section}): "
                      f"{type(ex).__name__}: {ex}", file=sys.stderr)
                continue
            total_in += ti_sn
            total_out += to_sn
            if not missed:
                continue
            n_safety_net_hits += len(missed)

            try:
                sn_rows, ti_v, to_v = verify_safety_net_hits(
                    missed, section_text, action, cfr_part, cfr_section,
                    docket_id=str(candidates[0].get("docket_id", "")) if candidates else "",
                    rule_type=str(candidates[0].get("rule_type", "final")) if candidates else "final",
                    provider=provider, model=model,
                    starting_candidate_idx=starting_sn_idx + block_i * 100,
                )
            except Exception as ex:
                print(f"  [safety-net-verify] block {block_i}: "
                      f"{type(ex).__name__}: {ex}", file=sys.stderr)
                continue
            total_in += ti_v
            total_out += to_v
            n_safety_net_verified += len(sn_rows)
            verified_rows.extend(sn_rows)
            print(f"  [safety-net] §{cfr_section}: {len(missed)} missed "
                  f"→ {len(sn_rows)} verified after second-pass")

    # ---- DEDUP pass ----
    # The safety net deliberately re-scans each amendatory block, so it
    # sometimes re-surfaces sentences the primary pass already extracted.
    # Drop those duplicates before writing — same-section overlap > 80%
    # against any primary row's offsets means the safety-net row is a
    # duplicate. See path_a_dedup for the heuristic rationale.
    from path_a_dedup import build_candidates_offsets, dedup_verified_rows
    candidates_offsets = build_candidates_offsets(candidates)
    verified_dicts = [asdict(r) for r in verified_rows]
    kept_dicts, dedup_summary = dedup_verified_rows(
        verified_dicts, candidates_offsets, binding_text,
    )
    if dedup_summary.safety_net_dropped:
        print(f"  [dedup] dropped {dedup_summary.safety_net_dropped} "
              f"safety-net duplicates of primary rows "
              f"(rows {dedup_summary.rows_in} -> {dedup_summary.rows_out}; "
              f"unmatched={dedup_summary.unmatched_safety_net})")
    n = _write_verified_dicts_csv(kept_dicts, out_csv)
    return {
        "candidates_processed": len(candidates),
        "verified_rows_written": n,
        "verified_candidates": n_verified,
        "rejected_candidates": n_rejected,
        "compound_candidates": n_compound,
        "safety_net_hits": n_safety_net_hits,
        "safety_net_verified": n_safety_net_verified,
        "safety_net_dedup_dropped": dedup_summary.safety_net_dropped,
        "safety_net_dedup_unmatched": dedup_summary.unmatched_safety_net,
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "out_csv": str(out_csv),
    }


def run_cross_llm_consistency(
    candidates_csv: Path,
    binding_text_path: Path,
    primary_verified_csv: Path,
    out_csv: Path,
    primary_provider: str = "openai",
    primary_model: Optional[str] = None,
    secondary_provider: str = "anthropic",
    secondary_model: Optional[str] = None,
    sample_frac: float = 1.0,
    sample_seed: int = 20260510,
    limit: int = 0,
) -> dict:
    """Run secondary LLM on the same candidates used for primary; emit
    cross-LLM CSV with verified-flag agreement + structured-field agreement.

    sample_frac<1.0 enables stratified-by-amendatory-action subsampling
    for the 5% Stage 2 production cross-validation use case."""
    if primary_model is None:
        primary_model = (DEFAULT_OPENAI_MODEL if primary_provider == "openai"
                         else DEFAULT_ANTHROPIC_MODEL)
    if secondary_model is None:
        secondary_model = (DEFAULT_ANTHROPIC_MODEL if secondary_provider == "anthropic"
                           else DEFAULT_OPENAI_MODEL)

    candidates = _read_candidates_csv(candidates_csv)
    binding_text = binding_text_path.read_text(encoding="utf-8", errors="ignore")

    # Load primary verified for comparison
    primary_by_cand: dict[int, list[VerifiedObligation]] = {}
    with primary_verified_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cidx = int(row["candidate_idx"])
            primary_by_cand.setdefault(cidx, []).append(_dict_to_verified(row))

    # Optional stratified subsample on amendatory_action
    if sample_frac < 1.0:
        import random
        rng = random.Random(sample_seed)
        by_action: dict[str, list[dict]] = {}
        for c in candidates:
            by_action.setdefault(c.get("amendatory_action", "none"), []).append(c)
        sampled: list[dict] = []
        for act, group in by_action.items():
            k = max(1, int(round(len(group) * sample_frac)))
            sampled.extend(rng.sample(group, k=min(k, len(group))))
        candidates = sampled

    if limit > 0:
        candidates = candidates[:limit]

    xllm_rows: list[CrossLLMRow] = []
    total_in = 0
    total_out = 0
    for i, cand in enumerate(candidates, 1):
        cand["__binding_text__"] = binding_text
        primary_rows = primary_by_cand.get(int(cand["obligation_idx"]), [])
        try:
            xrow, _, ti, to = compare_cross_llm(
                cand, primary_rows,
                primary_provider=primary_provider,
                primary_model=primary_model,
                secondary_provider=secondary_provider,
                secondary_model=secondary_model,
            )
        except Exception as e:
            print(f"  [xllm] candidate #{cand.get('obligation_idx')}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        total_in += ti
        total_out += to
        xllm_rows.append(xrow)
        if i % 25 == 0 or i == len(candidates):
            print(f"  [xllm] {i}/{len(candidates)}  "
                  f"in_tok={total_in:,}  out_tok={total_out:,}")

    n = _write_xllm_csv(xllm_rows, out_csv)

    # Quick agreement summary
    if xllm_rows:
        verified_match = sum(1 for r in xllm_rows if r.verified_match)
        complexity_match = sum(
            1 for r in xllm_rows
            if r.complexity_match and r.primary_verified and r.secondary_verified
        )
        verified_match_pct = verified_match / len(xllm_rows)
    else:
        verified_match_pct = float("nan")

    return {
        "compared": len(xllm_rows),
        "verified_match_pct": verified_match_pct,
        "secondary_in_tokens": total_in,
        "secondary_out_tokens": total_out,
        "out_csv": str(out_csv),
    }


def _dict_to_verified(row: dict) -> VerifiedObligation:
    return VerifiedObligation(
        candidate_idx=int(row["candidate_idx"]),
        docket_id=row["docket_id"],
        rule_type=row["rule_type"],
        split_idx=int(row.get("split_idx", 0)),
        verified=row.get("verified", "True").lower() == "true",
        verified_reason=row.get("verified_reason", ""),
        complexity=row.get("complexity", "simple"),
        recital_disposition=row.get("recital_disposition", "none"),
        cfr_part=_coerce_optional_str(row.get("cfr_part")),
        cfr_section=_coerce_optional_str(row.get("cfr_section")),
        subject=row.get("subject", ""),
        modal=row.get("modal", "no_modal_implicit"),
        modal_strength=row.get("modal_strength", "strong"),
        action=row.get("action", ""),
        object=row.get("object", ""),
        passive_voice=row.get("passive_voice", "False").lower() == "true",
        is_conditional=row.get("is_conditional", "False").lower() == "true",
        condition_text=_coerce_optional_str(row.get("condition_text")),
        amendatory_action=row.get("amendatory_action", "none"),
        obligation_text=row.get("obligation_text", ""),
        cross_reference=_coerce_optional_str(row.get("cross_reference")),
        found_via=row.get("found_via", "primary"),
        llm_provider=row.get("llm_provider", "openai"),
        llm_model=row.get("llm_model", DEFAULT_OPENAI_MODEL),
        in_tokens=int(row.get("in_tokens", 0) or 0),
        out_tokens=int(row.get("out_tokens", 0) or 0),
    )


# ---------------------------------------------------------------------------
# Smoke test (no API calls — validates schema enforcement on canned outputs)
# ---------------------------------------------------------------------------
def _smoke_test() -> int:
    print("[smoke-test] Stage 1b — schema enforcement (no API calls)\n")

    # Test 1: parse a clean primary-verification JSON
    canned_primary = """
    {
      "verified": true,
      "verified_reason": "Within an amendatory block; specifies actor and deadline.",
      "complexity": "simple",
      "recital_disposition": "none",
      "obligations": [
        {
          "subject": "Each operator",
          "modal": "shall",
          "modal_strength": "strong",
          "action": "maintain",
          "object": "records for five years",
          "passive_voice": false,
          "is_conditional": false,
          "condition_text": null,
          "cfr_section": "80.1",
          "amendatory_action": "revise",
          "cross_reference": null
        }
      ]
    }
    """
    parsed = json.loads(canned_primary)
    print("  ✓ canned primary JSON parses")
    ob0 = parsed["obligations"][0]
    assert _coerce_modal(ob0["modal"]) == "shall"
    assert _coerce_enum(ob0["modal_strength"], ALLOWED_MODAL_STRENGTHS, "strong") == "strong"
    assert _coerce_enum(ob0["amendatory_action"], ALLOWED_AMENDATORY_ACTIONS, "none") == "revise"
    assert _coerce_optional_str(ob0["cfr_section"]) == "80.1"
    assert _coerce_optional_str(ob0["condition_text"]) is None
    print("  ✓ enum coercion + null coercion work on clean JSON")

    # Test 2: model returns near-miss values — coercion should snap them
    out_of_enum = {
        "modal": "Required to",          # case + missing "is "
        "modal_strength": "STRONG",      # case
        "amendatory_action": "amend",    # not in enum
        "recital_disposition": "FOO",    # not in enum
        "complexity": "Compound",        # case
    }
    assert _coerce_modal(out_of_enum["modal"]) == "is required to"
    assert _coerce_enum(out_of_enum["modal_strength"], ALLOWED_MODAL_STRENGTHS, "strong") == "strong"
    assert _coerce_enum(out_of_enum["amendatory_action"], ALLOWED_AMENDATORY_ACTIONS, "none") == "none"
    assert _coerce_enum(out_of_enum["recital_disposition"], ALLOWED_RECITAL_DISPOSITIONS, "none") == "none"
    assert _coerce_enum(out_of_enum["complexity"], ALLOWED_COMPLEXITIES, "simple") == "compound"
    print("  ✓ near-miss enum values coerce to safe defaults / canonical forms")

    # Test 3: code-fenced JSON output strips correctly
    fenced = "```json\n{\"verified\": true, \"obligations\": []}\n```"
    cleaned = _strip_codefence(fenced)
    assert json.loads(cleaned) == {"verified": True, "obligations": []}
    print("  ✓ code-fenced JSON strip works")

    # Test 4: bare-empty / 'null' string coerce to Python None
    assert _coerce_optional_str("") is None
    assert _coerce_optional_str("null") is None
    assert _coerce_optional_str(None) is None
    assert _coerce_optional_str("  some text  ") == "some text"
    print("  ✓ optional-str null behavior")

    # Test 5: empty obligations list when verified=false
    canned_reject = json.loads("""
    {
      "verified": false,
      "verified_reason": "This sentence is preamble narrative, not amendatory text.",
      "complexity": "simple",
      "recital_disposition": "preamble_narrative",
      "obligations": []
    }
    """)
    assert canned_reject["obligations"] == []
    assert _coerce_enum(canned_reject["recital_disposition"],
                       ALLOWED_RECITAL_DISPOSITIONS, "none") == "preamble_narrative"
    print("  ✓ verified=false with empty obligations list")

    # Test 6: safety-net JSON shape
    canned_safety = json.loads("""
    {
      "missed_obligations": [
        {
          "subject": "the operator",
          "modal": "shall",
          "action": "submit",
          "object": "the report",
          "obligation_text": "The report shall be submitted by the operator.",
          "found_via": "passive_voice"
        }
      ]
    }
    """)
    assert len(canned_safety["missed_obligations"]) == 1
    m = canned_safety["missed_obligations"][0]
    assert _coerce_modal(m["modal"]) == "shall"
    print("  ✓ safety-net JSON shape")

    # Test 7: closed enum sets are non-empty + sane
    assert "must" in ALLOWED_MODALS
    assert "no_modal_implicit" in ALLOWED_MODALS
    assert "revise" in ALLOWED_AMENDATORY_ACTIONS
    assert "preamble_narrative" in ALLOWED_RECITAL_DISPOSITIONS
    print("  ✓ closed-enum sets sanity-check")

    # Test 8: safety-net hit roundtrip — synthetic missed-obligation dict
    # transforms into a valid candidate-shaped dict for re-verification
    missed = {
        "subject": "operators", "modal": "shall", "action": "submit",
        "object": "records", "obligation_text": "Operators shall submit records.",
        "found_via": "passive_voice",
    }
    synthetic = {
        "obligation_idx": 100001,
        "docket_id": "EPA-TEST",
        "rule_type": "final",
        "cfr_part": None,
        "cfr_section": None,
        "subject": missed["subject"],
        "modal": missed["modal"],
        "modal_strength": "strong",
        "action": missed["action"],
        "object": missed["object"],
        "amendatory_action": "revise",
        "is_conditional": False,
        "condition_text": "",
        "passive_voice": False,
        "obligation_text": missed["obligation_text"],
        "char_offset_start": 0,
        "char_offset_end": len(missed["obligation_text"]),
    }
    # Just confirm all required keys are present for verify_candidate
    required_keys = {"obligation_idx", "docket_id", "rule_type",
                     "obligation_text", "char_offset_start", "char_offset_end"}
    assert required_keys.issubset(synthetic.keys())
    print("  ✓ safety-net second-pass synthetic-candidate shape valid")

    print("\n[smoke-test] PASS (8/8 sub-tests). API-calling tests require keys.")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="cmd")

    ap.add_argument("--smoke-test", action="store_true",
                    help="Run inline schema-enforcement smoke test (no API).")

    pv = sub.add_parser("verify",
                        help="Run primary verification on Stage 1a candidates "
                             "and (by default) the section-level safety net.")
    pv.add_argument("--candidates", type=Path, required=True)
    pv.add_argument("--binding-text", type=Path, required=True)
    pv.add_argument("--out", type=Path, required=True)
    pv.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    pv.add_argument("--model", default=None)
    pv.add_argument("--limit", type=int, default=0,
                    help="Cap candidates for a smoke run; 0 = no cap.")
    pv.add_argument("--no-safety-net", action="store_true",
                    help="Skip the section-level safety-net pass. The default "
                         "is to run safety net + second-pass verification on "
                         "missed obligations (per v2 spec § Stage 1b).")
    pv_batch = pv.add_mutually_exclusive_group()
    pv_batch.add_argument("--batch", dest="batch", action="store_true",
                          default=True,
                          help="Submit candidates through the OpenAI Batch API "
                               "(24h async, ~50%% gpt-5 pricing). Default.")
    pv_batch.add_argument("--no-batch", dest="batch", action="store_false",
                          help="Use the per-candidate real-time path (legacy).")
    pv.add_argument("--max-cost", type=float, default=1500.0,
                    help="Hard cost cap for the batch pre-check (USD).")

    px = sub.add_parser("xllm",
                        help="Cross-LLM consistency check on already-verified candidates.")
    px.add_argument("--candidates", type=Path, required=True)
    px.add_argument("--binding-text", type=Path, required=True)
    px.add_argument("--primary", type=Path, required=True,
                    help="Path to primary verified CSV (output of `verify`).")
    px.add_argument("--out", type=Path, required=True)
    px.add_argument("--primary-provider", choices=["openai", "anthropic"],
                    default="openai")
    px.add_argument("--primary-model", default=None)
    px.add_argument("--secondary-provider", choices=["openai", "anthropic"],
                    default="anthropic")
    px.add_argument("--secondary-model", default=None)
    px.add_argument("--sample-frac", type=float, default=1.0,
                    help="Stratified subsample fraction (1.0 = full validation; "
                         "0.05 = Stage 2 production 5%% cross-validation).")
    px.add_argument("--sample-seed", type=int, default=20260510)
    px.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    if args.smoke_test or args.cmd is None:
        return _smoke_test()

    if args.cmd == "verify":
        summary = run_primary_verification(
            candidates_csv=args.candidates,
            binding_text_path=args.binding_text,
            out_csv=args.out,
            provider=args.provider,
            model=args.model,
            limit=args.limit,
            run_safety_net=not args.no_safety_net,
            batch=args.batch,
            max_cost_usd=args.max_cost,
        )
        print("\n[stage-1b verify] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    if args.cmd == "xllm":
        summary = run_cross_llm_consistency(
            candidates_csv=args.candidates,
            binding_text_path=args.binding_text,
            primary_verified_csv=args.primary,
            out_csv=args.out,
            primary_provider=args.primary_provider,
            primary_model=args.primary_model,
            secondary_provider=args.secondary_provider,
            secondary_model=args.secondary_model,
            sample_frac=args.sample_frac,
            sample_seed=args.sample_seed,
            limit=args.limit,
        )
        print("\n[stage-1b xllm] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
