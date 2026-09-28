# Path A Stage 1 — Obligation Extraction Pipeline Design (v2)

**Date:** 2026-05-09
**Author:** Yue (with Claude)
**Status:** v2 — incorporates Bruce's 2026-05-08 audit (`notes/2026-05-08_path_a_stage1_audit.md`) + my blocker-resolution plan (`notes/2026-05-09_path_a_stage1_blockers_addressed.md`) + Bruce's 2026-05-09 sign-offs + PROJECT_FACTS Leahey-entry primary-source-verified framing.
**Replaces:** `notes/2026-05-09_path_a_stage1_obligation_extraction.md` (v1).
**Approval state:** Bruce greenlit code-writing pending v2 absorption.

## Goal

For each of our 36 anchor dockets, extract every binding deontic obligation from the NPRM and the Final Rule as a structured record. Each obligation becomes an analytic unit per the **obligation-level unit of analysis (Leahey 2026)**, implemented with a hybrid NLP+LLM verification pipeline that extends Leahey's deterministic structural parsing to capture semantic continuity.

**Citation framing (locked per PROJECT_FACTS § 9 verified by Bruce 2026-05-08):** cite as "obligation-level unit of analysis (Leahey 2026), implemented with a hybrid NLP+LLM verification pipeline that extends Leahey's deterministic structural parsing to capture semantic continuity that pure structural matching may miss; Leahey 2026 acknowledges this limitation." **DO NOT** cite as "Leahey-style deontic parsing" — that overclaims direct methodological inheritance.

## Critical scope rule (per Love 2026)

**Extract from binding regulatory text ONLY, not the preamble.**

Federal Register documents have two parts:
- **Preamble** (`SUMMARY`, `DATES`, `ADDRESSES`, `SUPPLEMENTARY INFORMATION`): agency reasoning, response to comments, justification — NOT legally binding.
- **Binding regulatory text**: precise text codified at e.g. 40 CFR § 63.6590(b)(2)(iii) — legally binding.

Pre-processing must split preamble from binding text BEFORE obligation extraction.

**Section-detection markers** (FR style guide):
- Binding text starts at: `For the reasons set forth in the preamble, … is amended as follows:` OR `PART X — [TITLE]` OR `Authority:` block.
- Practical heuristic: binding text lives between `For the reasons stated in the preamble, [agency] amends [N] CFR …` and the document end.

Hand-validate the split on each of the 3 validation rules before running on full set.

## Two-stage extraction pipeline

### Stage 1a — Heuristic + spaCy pre-pass

For each binding-text section:

1. **Sentence segmentation** via spaCy (`en_core_web_lg`).
2. **Modal-verb anchor detection.** Mark sentence as candidate if it contains:
   - **Strong modals:** `must`, `must not`, `shall`, `shall not`, `may not`, `is/are required to`, `is/are prohibited from`, `no person shall`, `no operator shall`, `no [entity] may`
   - **Permissive modals** (per Bruce sign-off — tagged `modal_strength=permissive` for downstream filtering): `may`, `is/are permitted to`, `is/are authorized to`
   - **Conditional triggers** (sentence-leading): `If …, [actor] must …`, `When …`, `Upon …`
   - **Mid-sentence/end-sentence conditionals** (Major 2 fix): `provided that`, `unless`, `except as`, `subject to`, `if applicable`. Disambiguation: `subject to § X.Y` is cross-reference, not conditional.
3. **Amendatory-block detection** (Blocker 3 + Major 3 fix): regex pre-pass for amendatory-block markers, attach `amendatory_action ∈ {add, revise, remove, replace, redesignate, none}` to every candidate based on the enclosing block:
   - `is revised to read as follows` → `revise`
   - `is amended to read as follows` → `revise`
   - `is added to read as follows` → `add`
   - `is amended by adding paragraph (X) to read as follows` → `add`
   - `paragraph (X) is removed` → `remove`
   - `is redesignated as` → `redesignate`
   - `is replaced by` → `replace`
   - No enclosing amendatory marker → `none` (e.g., new subpart added without explicit amendatory wrapper)
4. **Dependency parse** each candidate (Major 1 fix):
   - Try `nsubj` first (active voice).
   - **If `nsubjpass` (passive voice)** with no nsubj: look for `agent` child; if agent → use as actor; else mark `actor=<implicit>`, `passive_voice=true`.
   - Extract `modal` (modal-verb token), `action` (head verb + immediate VP), `object` (direct-object NP if any).
5. **CFR-section detection:** for each candidate, walk back to nearest preceding `§\s*\d+\.\d+(?:\([a-z0-9ivx]+\))*` regex match. Also extract `cfr_part` from nearest preceding `PART X — [TITLE]` header (Major 7 / multi-part-rules handling).
6. **Output candidate-obligations CSV:** `data/processed/path_a_obligations_candidates_<docket_id>.csv` with fields: `obligation_idx, rule_type (proposed|final), cfr_part, cfr_section, subject, modal, modal_strength, action, object, amendatory_action, is_conditional, condition_text, passive_voice, obligation_text, char_offset_start, char_offset_end`.

### Stage 1b — gpt-5 verification + structured extraction

Per v2.3 model lock: gpt-5 real-time, ~$0.011 average per call.

**Two LLM passes per candidate:**
1. **Verification + structured-field correction** (primary)
2. **Section-level safety net** (catches missed obligations)

**NEW Major 5 fix:** safety-net outputs run back through the verification prompt before merging. Asymmetric pipeline (primary 2 passes, safety-net 1 pass) caused inflated false positives in v1.

### Stage 1b prompt — Primary verification (revised v2)

```
System:
You are an expert in U.S. federal regulatory text analysis. You verify
whether a candidate text span is a binding deontic obligation in CFR
amendatory text, and return its structured representation.

VERIFICATION RULE (revised v2 — replaces v1's recital rule):

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

DEFINITION-EMBEDDED OBLIGATIONS (Major 7 fix): definitions can embed
implicit binding requirements. If a definition specifies an actor,
threshold, deadline, or quantitative criterion, treat the embedded
requirement AS the obligation. Examples:
- "Eligible facility means one that has filed Form A by March 1." →
  obligation: "facility must file Form A by March 1"; verified=true.
- "Routine maintenance means activities performed at intervals not
  exceeding 12 months." → obligation: "maintenance must occur at
  intervals ≤ 12 months"; verified=true.

NOT BINDING (verified=false) ONLY if:
- Comment is in preamble (SUPPLEMENTARY INFORMATION) AND no amendatory
  block contains it.
- Verbatim quote of statutory text WITHOUT restatement as new amendment.
- Pure definition without embedded obligation (e.g., "Major source
  means any stationary source ..." with no temporal/quantitative
  criterion).
- Preamble narrative ("Commenters argued ...", "EPA agrees that ...").

User:
Candidate obligation:
"""{candidate_sentence}"""

Surrounding context (±3 sentences, with amendatory-block header if any):
"""{context_text}"""

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
1. Decide verified: true | false. Apply the rules above.
2. If verified=true, correct structured fields where spaCy parse is wrong.
3. Flag complexity: "simple" (one actor + one modal + one action) or
   "compound" (split rule: modals differ OR actions target different
   objects). If compound, return a list of split obligations.
4. Mark `recital_disposition` ∈ {none, amendatory_quote, statutory_quote,
   preamble_narrative, definition_embedded} for downstream sub-metric
   analysis.

CONSTRAINTS (Minor 10 — JSON robustness):
- modal MUST be one of: ["must", "shall", "may not", "must not",
  "may", "is required to", "are required to", "is prohibited from",
  "are prohibited from", "is permitted to", "are permitted to",
  "is authorized to", "are authorized to", "no_modal_implicit"]
  Do NOT return novel modal phrasings.
- cfr_section: MUST be a string in form "X.Y" or "X.Y(z)" or null
  (literal null, NOT empty string) if not extractable.
- modal_strength: MUST be one of: ["strong", "permissive"].
- amendatory_action: MUST be one of: ["add", "revise", "remove",
  "replace", "redesignate", "none"].
- recital_disposition: MUST be one of: ["none", "amendatory_quote",
  "statutory_quote", "preamble_narrative", "definition_embedded"].

If the LLM returns malformed JSON, retry once with the system message
"Your previous response was not valid JSON. Return only the JSON object
with no preamble or markdown formatting."

Output JSON only:
{
  "verified": <bool>,
  "verified_reason": "<one sentence>",
  "complexity": "simple" | "compound",
  "recital_disposition": "<enum>",
  "obligations": [
    {
      "subject": "<str>",
      "modal": "<closed_enum_value>",
      "modal_strength": "<closed_enum_value>",
      "action": "<str>",
      "object": "<str>",
      "passive_voice": <bool>,
      "is_conditional": <bool>,
      "condition_text": "<str>" | null,
      "cfr_section": "<str>" | null,
      "amendatory_action": "<closed_enum_value>"
    }
  ]
}

If verified=false, return obligations: [].
```

### Stage 1b prompt — Safety-net (revised v2 with second-pass verification)

```
System:
You are reviewing a CFR-amendment text section for any binding deontic
obligations that may have been missed by the primary extractor.

You are given:
- Section text (binding-text only)
- The list of obligations the primary extractor found

Return ONLY missed obligations. Apply the same BINDING/NOT BINDING
rules as the primary verification prompt. If you find no missed
obligations, return missed_obligations: [].

PRECISION ANCHOR (Major 5 fix): only return obligations you are
HIGHLY CONFIDENT are missed. When in doubt, OMIT — false positives
here cascade into the regression.

User:
Section text: "{section_text}"
Already-extracted obligations: {json_existing}

Output (JSON only):
{
  "missed_obligations": [
    {
      "subject": "<str>",
      "modal": "<closed_enum>",
      "action": "<str>",
      "object": "<str>",
      "obligation_text": "<verbatim>",
      "found_via": "passive_voice" | "no_modal" | "cross_reference"
                 | "definition_embedded" | "other"
    }
  ]
}
```

**NEW (Major 5 fix):** each item from `missed_obligations` is then routed back through the **primary verification prompt** before merging. This eliminates the asymmetry where safety-net outputs got fewer LLM passes than primary candidates.

## Validation plan (Blocker 1 + Bruce's preferred 1+2 split)

**3-rule validation set across CFR programs:**

| Role | Anchor | CFR Part | Year cluster | Expected obligations (proxy from regex) |
|---|---|---|---|---|
| **In-sample** (prompt tuning) | EPA-HQ-OAR-2018-0775 | 40 CFR 80 (RFS / E15) | 2019-2022 | ~75 |
| **Held-out A** | EPA-HQ-OAR-2017-0757 | 40 CFR 60 (Oil & Gas NSPS) | 2019-2022 | ~107 |
| **Held-out B** | EPA-HQ-OAR-2002-0058 | 40 CFR 63 (Boiler MACT) | 2010-2012 | ~453 (largest, but oldest era — strongest generalization test) |

**Per Bruce's audit (1+2 split, stronger than my 2+1 proposal):**
- Iterate prompts ONLY on the in-sample (2018-0775).
- Run frozen pipeline on both held-outs separately.
- Report per-program recall + precision.

**Acceptance criteria (Blocker 4 + Major 6 fix):**

| Metric | Target | Action if missed |
|---|---|---|
| In-sample recall | ≥ 0.95 | Iterate heuristic + LLM prompts |
| **In-sample precision (NEW v2 floor 0.85)** | ≥ 0.85 | Tighten LLM verification rules |
| **Cross-LLM consistency on validation** (Blocker 2) | gpt-5 vs Sonnet 4.5 verified-flag κ ≥ 0.80 | Iterate prompts to reduce model dependency |
| **Held-out generalization** | Recall + precision within 5pp of in-sample on each held-out | Second iteration on full validation set |
| **Stretch target** | Precision ≥ 0.90 | Lock 0.90 if achieved without recall loss |

**If held-out A or B falls > 5pp below in-sample**, this is a generalization-failure flag — triggers second iteration round across all 3 rules.

## Cross-LLM parity (Blocker 2 fix)

Sonnet 4.5 verification pass on the 3 validation rules. Per-obligation:
- **Verified flag agreement:** Cohen's κ on `verified=true/false` (target ≥ 0.80)
- **Structured-fields agreement:** lemma-fuzzy match rate on `subject`, `modal`, `action`, `object` (target ≥ 0.85)
- **Compound-flag agreement:** Cohen's κ on `complexity=simple/compound` (target ≥ 0.70)

**Plus 5% cross-LLM on Stage 2 production** (per Bruce sign-off, mirrors Layer C design): Sonnet 4.5 on stratified 5% subsample of all 36 anchors' obligations. ~$5-10 marginal.

## Stage 2 state taxonomy (NEW 7-state, Bruce sign-off)

Replaces v1's 4-state {SURVIVED, MODIFIED, DROPPED, NEW}:

| State | Definition |
|---|---|
| `SURVIVED-unchanged` | Final amendatory action = revise; similarity to NPRM ≥ 0.95 |
| `SURVIVED-edited` | Final amendatory action = revise; similarity 0.80-0.95 |
| `MODIFIED` | Final amendatory action = revise; similarity 0.50-0.80 |
| `REPLACED` | Final amendatory action = replace, OR similarity < 0.50 |
| `DROPPED-explicit` | NPRM obligation appears in `paragraph (X) is removed` block in Final |
| `DROPPED-silent` | NPRM obligation has no Final counterpart, no explicit removal |
| `NEW` | Final obligation has `amendatory_action=add` and no NPRM antecedent |

Per Bruce: "SURVIVED-unchanged/edited captures Love's procedural-vs-substantive at outcome level; DROPPED-explicit/silent is the precision diagnostic."

## Output artifacts per anchor

```
data/processed/path_a_obligations_candidates_<docket_id>.csv  ← Stage 1a output
data/processed/path_a_obligations_verified_<docket_id>.csv    ← Stage 1b primary
data/processed/path_a_obligations_missed_<docket_id>.csv      ← Stage 1b safety-net (post-verification)
data/processed/path_a_obligations_<docket_id>.csv             ← merged final, dedup
data/processed/path_a_xllm_consistency_<docket_id>.csv        ← gpt-5 vs Sonnet 4.5 (validation only)
```

## Implementation files

```
code/lib/path_a_obligation_heuristic.py        ← spaCy + regex Stage 1a (NEXT)
code/lib/path_a_obligation_llm.py              ← gpt-5 + Sonnet 4.5 Stage 1b
code/path_a_stage1_extract_obligations.py      ← orchestrator
code/path_a_validation_report.py               ← per-rule recall/precision
data/processed/path_a_obligations_GROUND_TRUTH_<docket_id>.csv  ← hand-coded × 3
data/processed/path_a_obligation_validation_report.md            ← combined metrics
```

## Cost summary (revised v2)

| Item | Volume | Cost |
|---|---|---|
| Stage 1b primary verification (per candidate) | ~6,000 across 3 validation rules + iteration | ~$70 |
| Stage 1b safety net + second-pass verification | ~600 sections | ~$30 |
| Sonnet 4.5 cross-validation on 3 validation rules | ~600 obligations | ~$10 |
| Stage 2 production on full 36 anchors | ~70K (comment, obligation) pairs as input to Path A | ~$60 |
| Sonnet 4.5 5% cross-validation on Stage 2 | ~3.5K pairs | ~$5 |
| **Total Stage 1 (post-iteration) + initial Stage 2** | | **~$155-175** |

Within v2.3's $50 validation buffer + Stage 4 budget. Update PROJECT_FACTS § 8 budget table when convenient.

## Open methodological questions (carrying forward to limitations)

Per Bruce sign-off + audit:

1. **Permissive modals** (`may`, `is permitted to`) — included in Stage 1, tagged `modal_strength=permissive` for downstream filtering. Path A regression can subset to strong modals only if needed.
2. **Compound splitting rule** — split when modals differ OR actions target different objects; keep merged when same-actor sequential workflow on same target.
3. **Cross-reference attribution** — `Records under § X.Y are due by …` attaches to citing section, not cited; record `cross_reference` field.
4. **Recital handling** — replaced by amendatory-block detection (Blocker 3); recital_disposition sub-metric tracks LLM accuracy per category.
5. **Multi-part rules** — `cfr_part` field extracted from `PART X` headers; Stage 2 NPRM↔Final matching respects Part boundaries.
6. **Incorporation by reference** (`as required by 40 CFR Part 60 Appendix A, Method 5`) — **OUT OF SCOPE for May 15**, document as limitation. The incorporated standard's content is NOT extracted as Path A obligations.
7. **Definition-embedded obligations** — handled in verification prompt's BINDING examples (Bruce's audit fix). Definitions with temporal/quantitative criteria are obligations.
8. **Single-rule-validation generalization** — addressed via 1+2 split; held-out per-rule metrics reported.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Heuristic over-extracts (false positives slow Stage 1b) | Acceptable — Stage 1b verifies. Cost monitored at validation step. |
| Heuristic under-extracts (low recall on implicit obligations) | Stage 1b safety-net per-section call covers this. Validation step measures recall directly against hand-coded truth. |
| LLM hallucinates obligations not in text | Closed-enum constraints on JSON output + validation step catches LLM-extracted not in ground truth = false positive (now gated by precision floor 0.85). |
| Preamble text leaks into extraction | Pre-processing splits preamble from binding text; hand-validate split on each of 3 validation rules. |
| Compound-obligation splitting introduces analytic-unit ambiguity | Document splitting rule explicitly; Stage 2 robustness check at both compound-not-split and compound-split levels. |
| spaCy parse errors on legal-language constructions | Stage 1b LLM verification corrects spaCy fields. Acceptable. |
| **NEW: safety-net asymmetry inflates false positives** (Major 5) | Run safety-net through verification prompt before merging. Closed. |
| **NEW: malformed JSON / parse failures** (Minor 10) | Closed enum on modal/amendatory_action; null literals for missing fields; retry-on-malformed clause in prompt orchestration. |
| **Held-out A or B fails generalization** | Triggers second iteration round; if still fails, document per-CFR-program recall as a limitation. |

## Implementation timeline

1. **Today:** Yue writes `code/lib/path_a_obligation_heuristic.py` (~2 hrs). Smoke-test on 2018-0775 Final.
2. **Tomorrow joint:** Bruce + Yue hand-code ground truth on 2018-0775 (~2 hrs). Yue runs Stage 1a, iterates heuristic until plain-modal recall ≥ 0.85 on in-sample.
3. **Tomorrow PM:** Stage 1b implementation. Run end-to-end on 2018-0775 in-sample. Iterate prompts until pipeline recall ≥ 0.95 AND precision ≥ 0.85.
4. **Day after:** Hand-code held-out A (2017-0757) and held-out B (2002-0058) ground truth. Run frozen pipeline. Report per-rule + cross-LLM metrics. Decide whether to lock for Stage 2 production.
5. **Day after that:** Sonnet 4.5 cross-validation on 3 validation rules. Stage 2 production rollout to all 36 anchors if validation passes.

## Coordination notes

- Independent of Layer C Path 1 lock (Bruce's `aedcdce` commit). Both pipelines use the same Stage 1 obligation extraction.
- Stage 1's output feeds Stage 2 (NPRM↔Final matching), Stage 3 (state classification), Stage 4 (comment-obligation linkage), Stage 5 (Cross-PPI debiased aggregation).
- Stage 4 K=10 prefilter validation (Bruce's prior sign-off) is SEPARATE from Stage 1 validation — different sampling protocol.

— End of v2.
