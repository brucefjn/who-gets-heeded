# Comment-rhetoric schema: production run notes

Operational notes on the full coding run of the 23-indicator schema (Appendix A of the paper): coverage, the 9-comment loss, truncation, and compute cost. Excerpted verbatim from the project's internal fact sheet (section 11c, 13 May 2026). File paths refer to the original project layout.

**Coverage:** 70,066 / 70,075 Tier B comments = 99.987%, across all 36 anchor dockets.

| Metric | Value |
|---|---|
| Wall time | 4.48 hr (concurrency=80, gpt-5 standard real-time) |
| Cost actual | $234.53 |
| Cost pre-check (conservative anchor) | $623 ($0.0089/call) |
| Cost per call (observed) | $0.00343 (prompts smaller than the 3K input + 500 output anchor) |
| Truncated comments | 268 (head + marker + tail policy, `TRUNCATION_TRIGGER_CHARS=12,000`) |
| Sync failures | 8 (LLM returned JSON missing one of the 23 required binaries; strict validator rejected) |
| Skip-existing dedup | 1 (orphan-recovered set overlap) |
| Total loss | 9 / 70,075 = 0.013% |

**Failure mode (8 sync failures):** All 8 are SOPHISTICATION-cluster schema misses — 4× `provides_legal_or_empirical_background`, 3× `provides_example`, 1× `burdensome`. Spread across 7 dockets, no docket-side concentration. The strict `_validated_layer_c_dict` validator rejects without recovery on the real-time path (asymmetric with the batch path, which has a safety-net retry).

**Recovery toolchain (4 orphan batches from the failed parallel `--docket` fire):**
- `code/collect_orphan_layer_c_batches.py` — polls OpenAI for batch_ids whose polling pythons died; downloads raw output JSONL
- `code/parse_orphan_layer_c_outputs.py` — re-loads Tier B per docket to recover `comment_id` from `custom_id='pair_{idx}'`; ingests raw JSONL into canonical CSV using existing helpers (`_validated_layer_c_dict`, `_result_to_record`, `_canonical_columns`, `truncate_for_layer_c`)
- Result: 1,626 rows recovered, 0 validation failures, $2.83 batched cost
- Final `data/processed/layer_c_full.csv` = 1,626 recovered + 68,440 sync = 70,066 rows
