# Outcome-state audit rubric — LOCKED 2026-05-16 (revised)

**Status:** Locked before coding begins. Do NOT modify after coding starts; rubric drift invalidates inter-rater statistics.

**Audit object:** the five-state outcome classifier (§5, §6.3.1 of paper).
**Sample size:** ~150 obligation pairs total. **Tier 1 = 116 F3-determining obligations** (the load-bearing subset for §6.4 Finding 3); **Tier 2 ≈ 34 supplementary pairs** for general 5-class metrics.
**Produced by:** `code/23_outcome_audit_sampler.py` with `--seed 20260516`.
**Canonical blind template:** `data/processed/outcome_audit_BLIND.csv` — do NOT edit.
**Per-rater files:**
  - `data/processed/outcome_audit_yue.csv` (Yue codes `rater_1_label` here)
  - `data/processed/outcome_audit_bruce.csv` (Bruce codes `rater_2_label` here)

Each rater copies the BLIND template to their own per-rater file via `cp`,
codes their column only (leaves the other rater's column empty), and commits.
This matches the Stage 4 audit pattern and avoids merge conflicts.

---

## What the audit IS and what it ISN'T

This audit primarily validates the **SURVIVED-edited / MODIFIED boundary** — the decision rule that drives the §6.4 Finding 3 equity claim. Tier 1 audits all 116 obligations in the F3 contingency directly. Tier 2 supplements with rows from outside the F3 subset to give general 5-class classifier metrics. Read the paper's §6.x audit subsection results in this hierarchy: the SE/MO boundary κ is the load-bearing number; the 5-class κ is supporting.

**Important caveat — DROPPED and NEW are audited at a "plausibility-of-stated-absence" level, not full ground-truth absence verification.** A blank `final_obligation_text` (DROPPED) or blank `proposed_obligation_text` (NEW) cannot, by itself, be used to confirm that the obligation is truly absent from the full rule — that would require reading the entire Federal Register document. The `cfr_reference` field provides the CFR section so the coder can spot-check if desired, but the rubric's claim for DROPPED/NEW is bounded to "the row is presented consistently with stated absence and no countervailing context is visible." Sample sizes for these two classes are intentionally small (≤4 each in Tier 2) because the audit cannot strongly validate them.

---

## Blindness

The blind coding sheet does NOT include cosine similarity. The outcome classifier is itself a cosine-threshold function (§5.3), so exposing cosine to coders would leak the classifier's decision boundary and would invalidate the audit. Code based on substantive reading of the two texts (and, for DROPPED/NEW, the CFR reference).

You also do not see:
- The classifier-assigned label (kept in a separate truth file).
- The audit tier (T1_F3 vs T2_general).
- The other rater's labels.

---

## Coding task

For each row, read:
- `proposed_obligation_text` — the obligation as it appears in the Notice of Proposed Rulemaking (blank for NEW rows).
- `final_obligation_text` — the corresponding obligation in the Final Rule (blank for DROPPED rows).
- `cfr_reference` — the CFR Part and section where the obligation appears, for spot-check use on DROPPED/NEW.

Assign exactly one of five labels:

### (1) SURVIVED-unchanged

The proposed text appears **verbatim** in the final rule, OR with only:
- Whitespace, line-break, or paragraph-numbering changes
- Punctuation refinement (Oxford comma, hyphenation)
- Renumbering of cross-references (§80.1234 → §80.1240) with no semantic content change

Use sparingly — most "verbatim" cases have minor surface edits and should be SURVIVED-edited.

### (2) SURVIVED-edited

The **meaning is preserved**; the **surface form is refined**. Examples:
- Terminology updates ("the Administrator shall" → "EPA shall")
- Definitional clarification that doesn't change scope
- Sentence restructuring that preserves what is required
- Addition of explanatory language that does not change the obligation

**Critical:** Numerical thresholds, deadlines, exempted parties, and the *scope* of the obligation are **unchanged**. The compliance behavior a regulated party would adopt is **identical** between proposed and final.

### (3) MODIFIED

The **meaning is changed**. The obligation imposes:
- A different threshold (e.g., 9% → 15% ethanol concentration limit)
- A different scope (different facilities covered; different pollutants)
- A different deadline (compliance date moved)
- A different set of exempted parties (new exemption added/removed)
- A different required action (monitoring frequency, reporting form, etc.)

The obligation is still present in the final, but its substantive requirement differs.

### (4) DROPPED — *plausibility-of-absence judgment only*

The row presents a proposed obligation with no semantically corresponding final-rule text. Verify that:
- The proposed text is a binding obligation (not a recital, definition, or procedural note that wouldn't be carried to the final).
- The `cfr_reference` lies within the rule's scope (i.e., the agency could have either kept or dropped this obligation).
- Nothing in the proposed text suggests a likely replacement that should have been matched in the final.

If all three are satisfied, code DROPPED. If you're not confident, code MODIFIED (the safer descriptive default). The reported κ on DROPPED is bounded by what these isolated rows allow — a reviewer should not read DROPPED κ as full ground-truth validation.

### (5) NEW — *plausibility-of-absence judgment only*

Symmetric to DROPPED. The row presents a final-rule obligation with no proposed counterpart. Verify that:
- The final text is a binding obligation
- The `cfr_reference` is within the rule's scope
- Nothing in the final text suggests this is a near-duplicate of a proposed obligation that should have been matched

If satisfied, code NEW. If unsure, code MODIFIED.

---

## Critical decision rule (the load-bearing one)

When unsure between **SURVIVED-edited** and **MODIFIED**, ask:

> "Would a regulated party adopt a different compliance behavior between proposed and final?"

- **YES → MODIFIED.** The obligation requires something different now.
- **NO → SURVIVED-edited.** Same behavior, refined language.

This is the load-bearing decision rule. The §6.4 equity finding turns on this boundary. Cohen's κ on the {SURVIVED-edited, MODIFIED} restricted subset is the headline audit number.

---

## Other coding rules

1. **Code based on substantive meaning, not textual length.** A 200-character edit can be SURVIVED-edited (refined language) or MODIFIED (different threshold) — read the content.
2. **Cross-references:** if a proposed obligation references §X.Y and the final references §X.Z (renumbered), code SURVIVED-unchanged or SURVIVED-edited depending on whether the substantive obligation differs. Pure renumbering ≠ MODIFIED.
3. **Compound obligations:** if the proposed text contains two distinct obligations and only one is preserved, code MODIFIED.
4. **DROPPED/NEW spot-check:** use the `cfr_reference` to quickly scan the Federal Register text if you have any doubt about the plausibility of absence. Don't spend more than 60 seconds per row on this — the audit is bounded by what the isolated row supports.
5. **Notes column:** for any row where the call was difficult, write a short note in `rater_X_notes` (1–2 sentences). This is the input to the post-blind adjudication step.

---

## Procedure

1. **Bruce produces the sample.** `python3 code/23_outcome_audit_sampler.py` writes `outcome_audit_BLIND.csv`, `outcome_audit_classifier_labels.csv`, and `outcome_audit_sampling_log.json`. Bruce commits and pushes these.
2. **Each author creates their per-rater file by copying the BLIND template:**
   ```
   cp data/processed/outcome_audit_BLIND.csv data/processed/outcome_audit_yue.csv     # Yue
   cp data/processed/outcome_audit_BLIND.csv data/processed/outcome_audit_bruce.csv   # Bruce
   ```
3. Open your own per-rater file in a spreadsheet (Google Sheets, Excel, Numbers).
4. **DO NOT** look at the other rater's file.
5. **DO NOT** look at `outcome_audit_classifier_labels.csv` (the truth file).
6. Code each row by reading the texts and assigning one of the five labels.
   - Yue fills `rater_1_label` in `outcome_audit_yue.csv` (leaves `rater_2_label` blank)
   - Bruce fills `rater_2_label` in `outcome_audit_bruce.csv` (leaves `rater_1_label` blank)
7. Use `rater_X_notes` (your own column) for difficult cases.
8. Take breaks every 30 minutes — close-reading fatigue degrades reliability fast.
9. Each author commits their per-rater file independently:
   ```
   git add data/processed/outcome_audit_yue.csv     # or _bruce.csv
   git commit -m "Audit 1 blind coding: <your name>"
   git push
   ```
10. When **both** per-rater files are committed and pushed, run `code/24_outcome_audit_compute.py`. This produces:
   - Inter-rater κ on 5-class
   - Inter-rater κ on the load-bearing SE/MO binary
   - Confusion matrix
   - **Classifier κ against each rater individually** (NOT just consensus — see below)
8. After step 7, **adjudicate disagreements**. For each row where `rater_1_label != rater_2_label`, both authors review the row jointly and agree on a single `adjudicated_label`. This produces a human gold standard.
9. Re-run `code/24_outcome_audit_compute.py --with-adjudicated` to produce classifier-vs-adjudicated-gold κ.

**Why classifier κ is reported against each rater AND against adjudicated gold, not just consensus:** restricting classifier evaluation to rows where the two raters agree (consensus-only) systematically excludes the ambiguous cases — exactly the cases where classifier errors are most likely — and inflates the reported classifier agreement. The audit reports all three to avoid this bias.

---

## Expected time per author

- 40–60 sec/pair × 150 pairs = 100–150 minutes of pure coding.
- With thinking, breaks, harder cases: realistic block of 3–4 hours per author.
- Plus ~30 minutes of joint adjudication of disagreements after blind coding.

---

## What κ to expect (calibration)

Under reasonable rubric clarity:
- 5-class Cohen's κ: 0.65–0.85 (substantial-to-almost-perfect, Landis-Koch 1977).
- 2-class SURVIVED-edited vs MODIFIED: 0.55–0.80 (the boundary is intrinsically hard).
- Classifier-vs-each-rater κ: 0.45–0.75 (the classifier is a threshold heuristic).
- Classifier-vs-adjudicated-gold κ: typically 0.05–0.10 lower than classifier-vs-each-rater (adjudication reintroduces the hard cases).

**Reviewer-defensibility threshold:** κ ≥ 0.70 on the load-bearing SE/MO contrast under both rater-1 and rater-2 individually (not just consensus). Below 0.70, the paper should narrow the §6.4 claim accordingly.
