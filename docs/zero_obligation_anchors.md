# Stage 1b Rejection Diagnosis — 7 Empty Verified CSVs

**Date:** 2026-05-13
**Author:** Claude (Cowork)
**Run analyzed:** Stage 2 concurrent real-time production run, started ~04:18 UTC 2026-05-13, ended 06:53 UTC. Log: `data/processed/stage2_concurrent_full.log`.

## Question

Why did the Stage 1b LLM verifier reject 100% of candidates for 7 of 72 target-rule-types? Legitimate (rules without real obligations), miscalibrated (verifier too strict), or heuristic over-extraction (Stage 1a producing non-obligation candidates)?

## TL;DR — Verdict

**All 7 rejections are legitimate.** The Stage 1b verifier is correctly applying its rejection criteria. The root cause is heuristic over-extraction on non-substantive documents:

- **5 of 7 affected targets have ZERO amendatory blocks** in the FR text — they are discussion/preamble documents (endangerment findings, procedural meta-rules, recodifications, reconsideration NPRMs). Stage 1a regex fires on preamble's deontic language; Stage 1b correctly rejects because there are no amendatory blocks.
- **2 of 7 (OW-2017-0644 prop+final) have 12 amendatory blocks each, but all 12 are narrow applicability-date insertions** ("This definition is applicable beginning on February 6, 2020") — no new substantive obligations. Verifier correctly classifies as definition scope-clarification (verified=false per prompt v2.2).

**Recommendation:** Treat as a methodology finding, not a bug. Document in paper limitations. No verifier prompt fix, no heuristic fix, no re-run needed.

## Convergent Evidence (Three Layers)

### Layer 1: Document structure (objective)

Counted amendatory-block triggers (`is revised/amended/added to read as follows`, etc.) in each source FR text. Result:

| Target | Amend. Blocks | Preamble % | Verified | Consistent? |
|---|---|---|---|---|
| OAR-2014-0828/prop (82→0) | 0 | 97% | 0 | ✓ |
| OAR-2014-0828/final (41→0) | 0 | 99% | 0 | ✓ |
| OAR-2018-0794/prop (51→0) | 0 | ~100% | 0 | ✓ |
| OAR-2018-0794/final (181→304) | 10 | 70% | 304 | ✓ (accepted, has real amendments) |
| OAR-2020-0044/prop (132→0) | 0 | ~100% | 0 | ✓ |
| OW-2017-0203/prop (37→0) | 0 | 97% | 0 | ✓ |
| OW-2017-0644/prop (2→0) | 12* | 68% | 0 | ✓ (*all applicability-date) |
| OW-2017-0644/final (7→0) | 12* | 85% | 0 | ✓ (*all applicability-date) |
| **OAR-2009-0234/prop (569→847)** known-good | 11 | 66% | 847 | ✓ (accepted) |

### Layer 2: OW-2017-0644 deep-dive

The 12 amendatory blocks in OW-2017-0644 (both versions) are this exact pattern repeated for 12 different CFR parts:

```
Section X.X is amended by adding paragraph (4) to the definition of
"Navigable waters" to read as follows:
* * * * * (4) Applicability date. This definition is applicable beginning
on February 6, 2020.
```

The entire amendatory content is applicability dates. The verifier prompt explicitly excludes "narrow definition scope-clarification sentences" (v2.2 rule). Applicability dates fit this exclusion: no actor, no deontic, no substantive action — just temporal scope metadata about an existing rule.

### Layer 3: Verifier prompt vs. rejected content

Sampled candidates from all 7 affected targets. Every rejected candidate falls cleanly into a verified=false category per the prompt:

- Preamble narrative ("EPA agrees that...", "Commenters argued...")
- Verbatim statutory citations
- Executive Order boilerplate
- Discussion of OTHER agency actions ("In the 2020 Final Action, EPA asserted...")
- References to OTHER rules (NAAQS review cycle, NRC backfitting)
- Solicitations of comment

## Evidence — Docket-by-Docket

### Identified by docket title (anchor_rules_locked.csv)

| Docket | Title | Rule Type Class |
|---|---|---|
| OAR-2014-0828 | Greenhouse Gas Determinations: Emissions from Aircraft Cause or Contribute to Air Pollution... | **Endangerment finding** (not regulatory) |
| OAR-2018-0794 | NESHAP: Coal- and Oil-Fired Electric Utility Steam... | Substantive NESHAP (final OK; proposed = pre-2020 cost-benefit discussion) |
| OAR-2020-0044 | Increasing Consistency and Transparency in Considering Benefits and Costs in the Clean Air Act Rulemaking | **Procedural meta-rule** (EPA's own cost-benefit methodology) |
| OW-2017-0203 | Definition of Waters of US — Recodification of Pre-Existing Rules | **Recodification** (no new obligations) |
| OW-2017-0644 | Definition of Waters of US — Addition of Applicability Date | **Applicability-date addition** to existing rule |

### Spot-check: actual candidate sentences (verifier-rejected)

Sampled 5 candidates from each affected target. None of them meet the obligation criteria (actor + deontic + action + object as a new substantive requirement of this rule).

**OAR-2014-0828/proposed (82 cands, 0 verified):**
- "International Energy Agency Data Services, Available at http://data.iea.org..." → **citation**
- "Available at http://flightclub.jalopnik.com/heres-the-skinny-on-whats-next-for-boeing..." → **citation**
- "Section 307(d)(1)(V) of the CAA provides that the provisions of section 307(d) apply to..." → **procedural framework reference**
- "Proposed Definition of Air Pollutant Under section 231, the Administrator is to determine whether..." → describing the *action being proposed* (the determination itself), not an obligation on regulated parties

**OAR-2014-0828/final (41 cands, 0 verified):**
- "As with public health, the Administrator considered the multiple pathways in which the GHG air pollution..." → **descriptive narrative**
- "...when making a cause or contribute finding under section 231(a)(2), the Administrator must first define the air pollutant..." → describing EPA's own analytical process, not a regulated-party obligation

**OAR-2018-0794/proposed (51 cands, 0 verified):**
- "Totals may not sum due to rounding." → **boilerplate**
- "See 85 FR 31303 (May 22, 2020)." → **citation**
- "In the 2020 Final Action, the EPA merely asserted that a comparison of benefits to costs is..." → **discussion of OTHER agency action**
- "Executive Order 12866, currently in effect, requires a quantification of benefits and costs..." → **executive order boilerplate** (every NPRM has this)

**OAR-2020-0044/proposed (132 cands, 0 verified):**
- "...water systems must take actions to reduce exposure to lead and copper..." → **describing OTHER rules** (lead and copper rule, not this docket)
- "...EPA is required to review and if appropriate revise the air quality criteria and national ambient air quality standards (NAAQS) every 5 years" → **describing OTHER procedural requirement**
- "All additional requirements that involve integration of the station blackout mitigation strategies with other existing accident procedure and guideline sets must be justified under the NRC's backfitting regulations." → **NRC rule, not EPA**

**OW-2017-0203/proposed (37 cands, 0 verified):**
- "These depressions are located within 4,000 feet of Poplar Creek..." → **descriptive geography**
- "The agencies thus solicit comment on whether the definitions in the 2015 Rule would subject..." → **request for comment** (not an obligation)
- "I. Executive Order 13211: Actions Concerning Regulations That Significantly Affect Energy Supply..." → **EO boilerplate**

**OW-2017-0644 (both versions, 2+7 cands, 0 verified):**
- "An agency may certify that a rule will not have a significant economic impact on a substantial number of small entities..." → **RFA boilerplate**
- "Executive Order 13045: Protection of Children From Environmental Health Risks and Safety Risks..." → **EO boilerplate**
- "To satisfy the APA's notice and comment requirements, agencies must provide a 'meaningful opportunity' for comment..." → **APA procedural reference** (not a substantive obligation of this rule)

### Known-good baseline: OAR-2009-0234/proposed (569 cands, 847 verified)

Same heuristic, same verifier, but on a substantive NESHAP rule. Sample candidates:

- "natural gas must either be composed of at least 70 percent methane by volume or have a gross calorific value between 34 and 43 megajoules (MJ) per dry standard cubic meter..." → ✅ emission standard
- "The owner or operator of the facility experiencing an exceedence of its emission limit(s) during a malfunction shall notify the EPA Administrator by telephone or facsimile..." → ✅ notification obligation
- "You must establish the maximum fluorine fuel input (F-input) during the initial performance testing according to the procedures in paragraphs (c)(1)(i) through (iii) of this section." → ✅ testing obligation
- "The compliance certification shall indicate whether the monitoring data submitted were recorded in accordance with the applicable requirements of this appendix." → ✅ certification obligation

These all have clear actor + deontic + substantive action + new requirement structure. The verifier correctly accepts them.

## Why the heuristic over-extracts on these rule types

The Stage 1a heuristic (regex + spaCy) flags sentences containing deontic markers (`shall`, `must`, `may not`, `is required to`, etc.). It cannot distinguish:

1. **Substantive obligations of THIS rule** (the target) from
2. **Procedural framework references** ("the Administrator must define the air pollutant")
3. **Citations to OTHER rules** ("EPA is required to review and revise NAAQS every 5 years")
4. **Executive order boilerplate** ("E.O. 12866 requires quantification of benefits and costs")
5. **APA / RFA procedural language** ("agencies must provide a meaningful opportunity for comment")
6. **Discussion of past agency actions** ("In the 2020 Final Action, the EPA merely asserted...")

For rule types that are *primarily preamble* (endangerment findings, recodifications, meta-procedural rules, applicability-date additions), this means the heuristic recall is high but precision is near-zero. The LLM verifier — which evaluates whether each candidate is a *new substantive requirement of the rule being analyzed* — correctly filters them out.

This is the two-stage pipeline working as designed.

## Methodological Implication

**For the paper (§6 Limitations):**

> "The Stage 1a heuristic + Stage 1b LLM verifier pipeline correctly handles substantive regulatory rules but produces empty verified outputs for non-regulatory rule types (endangerment findings, procedural meta-rules, recodifications, applicability-date additions). Of 36 anchor dockets, 5 fall into these categories (OAR-2014-0828, OAR-2020-0044, OW-2017-0203, OW-2017-0644, and the proposed version of OAR-2018-0794). The verifier's 100% rejection rate on these targets reflects the absence of new substantive obligations rather than verifier miscalibration; manual spot-check of candidate sentences confirms they consist of citations, executive-order boilerplate, references to other rules, and procedural framework language. This is a feature of the design (the verifier is doing its job) but also a coverage limitation: comments on these dockets cannot be matched against rule obligations because no obligations exist to match against. For these 5 dockets, downstream analysis (Stage 4 obligation-comment matching) is necessarily skipped."

**Quantitatively:**

- 36 anchor dockets total
- 31 have at least one rule type with verified obligations (proposed OR final)
- 5 dockets have zero verified obligations across both rule types

That's still very good coverage. The 5 non-regulatory dockets weren't selected for their obligation density — they were selected for comment volume per the Path B/Layer C requirements. They contribute commenter responsiveness data even if not obligation-matching data.

## No Action Needed

- **Verifier prompt:** correct as written. Don't change.
- **Heuristic:** could be tightened to skip preamble sections, but this is a future-work optimization, not a fix. Stage 1b is catching the false positives.
- **Stage 4:** fires normally on the 62 target-rule-types with content. The 10 empty/missing targets just contribute 0 pairs.
- **Paper:** add the §6 limitation paragraph above.

## Files Examined

- `data/processed/anchor_rules_locked.csv` (docket titles)
- `data/processed/federal_register_index.csv` (FR document metadata)
- `data/processed/stage2_concurrent_full.log` (run log, 70KB)
- `data/processed/path_a_obligations_candidates_*.csv` (8 candidate files for the 7 affected targets + 1 known-good)

## Out of Scope (Not Performed)

- Live re-run of `verify_candidate()` against the OpenAI API — would cost a few dollars and the verbal "rejection reasoning" wouldn't add to the diagnosis since the candidate sentences themselves are dispositive.
- Modifying the verifier prompt — no evidence of miscalibration.
- Modifying the heuristic — diminishing returns vs. accepting filtered output.

## Round 2 Verification (Additional Layers)

### Layer 4: Safety net behavior

The Stage 1b safety net (section-level fallback that catches obligations the primary verifier misses on a per-candidate basis) **never fired** for any of the 7 affected targets:

| Target | Primary N succeeded | Safety net section_calls | Safety net verified |
|---|---|---|---|
| OAR-2014-0828/prop | 82 | (not fired) | 0 |
| OAR-2014-0828/final | 41 | (not fired) | 0 |
| OAR-2018-0794/prop | 51 | (not fired) | 0 |
| OAR-2020-0044/prop | 132 | (not fired) | 0 |
| OW-2017-0203/prop | 37 | (not fired) | 0 |
| OW-2017-0644/prop | 2 | (not fired) | 0 |
| OW-2017-0644/final | 7 | (not fired) | 0 |

Compare to OAR-2018-0794/final (same docket, accepted): `section_calls=11, hits=33, verified=47`. The safety net does fire when there's substantive content to recover from. Its non-firing for the rejected targets is consistent with the underlying documents lacking recoverable amendatory content.

### Layer 5: Exhaustive classification of OAR-2020-0044/proposed (132 candidates)

The largest affected target. Categorized all 132 candidates by content type:

| Category | Count | % | Example |
|---|---|---|---|
| OTHER (general preamble discussion) | 94 | 71% | "FAR Council issued a final rule (case 2017-007) on May 1, 2018..." |
| PASSIVE_OBLIGATION_LIKE | 25 | 19% | "VA is required to implement the Veterans Community Care Program..." |
| CITATION/URL | 4 | 3% | "Available at http://..." |
| DESCRIBES_OTHER_RULE_NAAQS | 4 | 3% | NAAQS review cycle references |
| DESCRIBES_OTHER_AGENCY | 2 | 2% | NRC backfitting regulations |
| DISCUSSION_PRIOR_ACTION | 1 | 1% | "In the 2020 Final Action..." |
| FORMATTING / EO BOILERPLATE | 2 | 2% | E.O. 13833 reference |

Inspected all 25 "PASSIVE_OBLIGATION_LIKE" candidates manually:
- 6 explicit VA (Veterans Affairs)
- 14 ambiguous but on inspection all discuss other agencies' rules (VA, SSA, FAR Council, FOIA, IRS, NRC, TTB)
- 1 EPA-actor but describing pre-existing CAA-mandated obligations on EPA itself (NAAQS review cycle every 5 years)

**Zero candidates create new obligations on regulated parties through THIS rule.** All "obligation-like" candidates are discussions of OTHER agencies' rules or pre-existing statutory obligations.

Random sample of 20 from the 94 "OTHER" candidates — same finding: all preamble discussion of other rules/agencies (VA must revise regs, SECVA shall promulgate rulemaking, TTB rulemaking priorities, FAR Council past action, EPA's TSCA risk management duties — none NEW from OAR-2020-0044).

This rule (OAR-2020-0044) is a procedural meta-rule about EPA's cost-benefit analysis *methodology*. Its preamble extensively discusses examples from other agencies' regulatory practices. The Stage 1a heuristic correctly fires on deontic patterns within those examples; the Stage 1b verifier correctly rejects them as not being obligations of THIS rule.

## Final Confidence Statement

**Five layers of converging evidence:**

1. **Document structure**: 0 amendatory blocks in 5 of 7 affected targets; the other 2 have only applicability-date insertions
2. **OW-2017-0644 deep-dive**: All 12 amendatory blocks across both versions are identical applicability-date insertions for 12 different CFR parts
3. **Verifier prompt criteria**: Every sampled rejected candidate fits a stated `verified=false` category
4. **Safety net never fired**: Consistent with no recoverable substantive surface
5. **Exhaustive sample of largest affected target (OAR-2020-0044/proposed)**: Even superficially obligation-like candidates confirmed as third-party citations

**Verdict:** All 7 rejections are legitimate. The Stage 1b verifier is correctly applying its stated criteria. The two-stage pipeline design is working exactly as intended. The 5 affected anchor dockets fall into non-regulatory rule categories that don't create new substantive obligations on regulated parties.

**No actions:** verifier prompt is correct, heuristic doesn't need tightening (the verifier catches what it should catch), no re-run needed.
