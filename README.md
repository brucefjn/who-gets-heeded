# Who Gets Heeded? Replication materials

Code, documentation, and derived data for

> Jianing Fan and Yue Yao. 2026. **Who Gets Heeded? An Obligation-Level Audit of Responsiveness in EPA Rulemaking.** In *Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO '26)*, November 5–7, 2026, Munich, Germany. ACM. [doi:10.1145/3846167.3848608](https://doi.org/10.1145/3846167.3848608) · Preprint: [arXiv:2608.10329](https://arxiv.org/abs/2608.10329)

The paper audits U.S. EPA notice-and-comment rulemaking at the level of individual regulatory obligations. It extracts obligations from the proposed and final rules of 36 EPA rulemakings, matches 70,075 public comments to the obligations they address, classifies how each obligation changed between the proposed and final rule, and tests each AI-assisted step against blind human coding.

Maintained by Jianing Fan ([@brucefjn](https://github.com/brucefjn)).

## What is here

| Folder | Contents |
|---|---|
| `code/` | The analysis pipeline, numbered in run order, and shared modules in `code/lib/` |
| `docs/` | Specifications, audit rubrics, prompts, and diagnostic notes |
| `data/processed/` | Derived tables: anchors, extracted obligations, matcher labels, outcomes, audit coding sheets, robustness results |
| `reproduce/` | `reproduce_key_numbers.py`, which recomputes the headline statistics from `data/processed/` |
| `figures/` | Figures as they appear in the paper |

## Where to find what the paper refers to

The paper points to this repository as "the supplementary repository." Each reference is listed below.

| Paper | Material | Location |
|---|---|---|
| §3, anchor selection | Anchor selection script; locked anchor table | `code/03_anchor_rule_selection.py`; `data/processed/anchor_rule_candidates.csv` (sampling frame); `data/processed/anchor_rules_locked.csv`; `docs/anchor_selection_log.md` |
| §3, attachment recovery | Per-stage recovery rates; OCR cross-validation protocol | `docs/attachment_recovery_diagnostic.md`; `docs/attachment_rate_diagnostic.md`; `docs/ocr_cross_validation.md`; `data/processed/attachment_sample.csv`, `attachment_extraction_log.csv`, `ocr_cross_validation.csv`; `code/04`–`code/10` |
| §4, commenter type | Both commenter-type classifiers | Permissive title rule used for the headline results: `code/20_finding3_analysis.py` (also `code/19`); stricter ternary classifier: `code/lib/submitter_classification.py`; labels for every addressing comment: `data/processed/addressing_comment_types.csv` |
| §5, extraction | Full extraction specification, including out-of-scope obligation classes | `docs/obligation_extraction_spec.md`; `code/12_path_a_extract_all_anchors.py`; `code/lib/path_a_obligation_heuristic.py`; `code/lib/path_a_obligation_llm.py`; extracted obligations in `data/processed/obligations/` |
| §5, extraction audit | Coding sheet for the 66-obligation precision audit | `data/processed/2026-05-11_path_a_audit_2018-0775.csv` |
| §6.3, outcome audit | Rubric; sampler script; coding sheets | `docs/outcome_audit_rubric.md` (and `.pdf`); `code/23_outcome_audit_sampler.py`; `data/processed/outcome_audit_bruce.csv`, `outcome_audit_yue.csv`, `outcome_audit_reconciled.csv`, `outcome_audit_classifier_labels.csv`; results from `code/24` and `code/27` in `outcome_audit_results.json` and `outcome_audit_finding3_corrected.json` |
| App. A.1 | Prompt template and per-indicator operational definitions | `docs/rhetoric_schema_prompt.md`; `code/lib/layer_c_judge.py`; `code/16_layer_c_run_full.py` |
| App. A.2 | Notes on the 9-comment loss; compute and cost | `docs/rhetoric_schema_run_notes.md` |
| App. A.2 | Per-indicator distributions for all 23 indicators | `data/processed/layer_c_indicator_prevalence.csv` |
| App. A.2 | 20% stratified non-anchor baseline | `code/tier_a_sample.py`; `docs/baseline_sample_design.md` |
| App. B.1 | Matcher rubric; full inter-rater statistics; disagreement list | `docs/matcher_audit_rubric.md`; `docs/matcher_audit_results.md`; `data/processed/stage4_validation_audit_bruce.csv`, `_yue.csv`, `_KEY.csv`; `code/15`, `code/18` |
| App. B.2 | Section-gated outcome classification (sensitivity artifact) | `data/processed/obligation_outcomes_section_gated.csv`; `code/17_compute_obligation_outcomes.py` |
| App. B.2 | Per-docket diagnosis of the seven anchors without verified proposed-side obligations | `docs/zero_obligation_anchors.md` |
| App. C.1 | Finding 1 per-docket leave-one-out | `data/processed/finding1_loo_proposed_side.csv`; `code/29_f1_excluding_new.py` |
| App. C.2 | Finding 3 per-docket leave-one-out | `data/processed/finding3_loo_sensitivity.csv`; `code/19_finding3_loo_sensitivity.py` |
| App. C.3 | Form letters, document pairing, comment periods | `code/31_dedup_mass_comments.py`, `code/33_pairing_window_check.py`; `data/processed/robustness_dedup_summary.json`, `robustness_pairing_window_summary.json` |
| App. C.4 | Time periods | `code/32_time_split.py`, `code/35_fig_time_split.py`; `data/processed/robustness_time_split_summary.json` |
| Ethics statement | Audit samples, reconciliation scripts, methodology documentation | The audit files above; `code/18`, `code/24`, `code/27`; `docs/` |

In the coding sheets, columns prefixed `BRUCE_` and `YUE_` (or `rater_1` and `rater_2`) hold the independent codes of Jianing (Bruce) Fan and Yue Yao.

## Data and privacy

The comments are public records on [regulations.gov](https://www.regulations.gov), and the rule text comes from the Federal Register. This repository does not redistribute comment text, comment titles, or submitter names. Tables identify comments by their regulations.gov document ID (for example `EPA-HQ-OAR-2018-0775-0731`), which opens the public record at `https://www.regulations.gov/comment/<ID>`. The audit coding sheets keep the obligation text and the raters' notes but not the comment text. Commenter type is released only as the derived label.

Main tables in `data/processed/`:

| File | One row per |
|---|---|
| `obligations/path_a_obligations_verified_<docket>_<proposed or final>.csv` | Verified obligation extracted from a Federal Register rule |
| `responsiveness_analysis.csv` | Obligation: outcome state, best cosine similarity, number of addressing comments, stance counts |
| `obligation_outcomes.csv`, `_audit_corrected.csv`, `_section_gated.csv` | Obligation: outcome under the text-similarity classifier, after the blind audit, and under the section-gated sensitivity rule |
| `stage4_matches.csv.gz` | Candidate comment–obligation pair (398,509): the matcher's addressing and stance labels |
| `addressing_comment_types.csv` | Comment that addresses at least one obligation: commenter type under both classifiers |
| `finding3_obligations.csv` | High-engagement obligation: addressing counts by commenter type and outcome before and after the audit |
| `federal_register_index.csv`, `fr_metadata_verification.csv` | Federal Register document used to pair proposed and final rules |

## Reproducing the key numbers

```bash
pip install pandas
python3 reproduce/reproduce_key_numbers.py
```

The script prints each statistic next to the value reported in the paper: Findings 1–3, the extraction precision audit, the matcher audit, and the outcome-classifier audit. It also writes `data/processed/finding1_loo_proposed_side.csv` and `docs/matcher_audit_results.md`.

## Running the full pipeline

The scripts in `code/` are the ones used for the paper, numbered in the order they were run. They expect the original project layout (`data/raw/` for the regulations.gov bulk download of EPA comments, 2010–2022, and `data/processed/` for intermediate files). Install dependencies with `pip install -r requirements.txt`. The LLM stages (obligation verification, comment matching, and the rhetoric schema, all GPT-5) need `OPENAI_API_KEY`; the OCR fallback in `code/09` and `code/10` needs `ANTHROPIC_API_KEY`. Re-running the LLM stages will not reproduce the labels bit for bit, which is why the labels used in the paper are released in `data/processed/`.

Finding 1 is reported for the 12,243 proposed-side obligations (`code/29_f1_excluding_new.py`, which reuses the regression code in `code/22_clustered_logistic_f1_f3.py`). `code/20` and `code/22` on their own also print an earlier version that counted final-only obligations as not revised.

## Citation

```bibtex
@inproceedings{fan2026heeded,
  author    = {Fan, Jianing and Yao, Yue},
  title     = {Who Gets Heeded? An Obligation-Level Audit of Responsiveness in EPA Rulemaking},
  booktitle = {Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3846167.3848608}
}
```

GitHub also reads the citation from `CITATION.cff`.

## License

Code is released under the MIT License (`LICENSE`). Data in `data/` and documents in `docs/` are released under CC BY 4.0 (`LICENSE-DATA`), the same license as the paper.

## Contact

Jianing Fan (jf3774@columbia.edu), Columbia University.
