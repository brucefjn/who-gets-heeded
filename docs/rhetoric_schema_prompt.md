# Comment-rhetoric schema: production prompt

The 23-indicator schema in Appendix A.1 of the paper was coded with GPT-5 using the prompt below, copied verbatim from `code/lib/layer_c_judge.py` (`LAYER_C_SYSTEM_PROMPT` and `LAYER_C_USER_TEMPLATE`). Each line of the system prompt gives the operational definition of one indicator. The prompt is zero-shot: it contains no worked examples. Long comments are truncated as described in `code/lib/layer_c_judge.py` before `{comment_text}` is filled in.

## System prompt

```text
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
```

## User message

```text
COMMENT:
{comment_text}
```
