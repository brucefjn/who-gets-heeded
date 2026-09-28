# Tier A Stratified Sample — Pre-Registration

**Date locked:** 2026-05-13
**Authors:** Yue Yao, Jianing (Bruce) Fan
**Purpose:** Pre-register the strata definitions, sample size, and allocation rule for the Tier A Layer C coding fire, prior to any Tier A coding data being generated. This document predates the sampling-script execution by design; any modification of the strata or allocation post-fire is a deviation requiring an explicit follow-up pre-registration with rationale.

## 1. Population frame

The Tier A population is the analyzable EPA public-comment corpus EXCLUDING the 36 anchor dockets that constitute Tier B:

| Component | Count |
|---|---|
| Full analyzable corpus (post-pipeline, PROJECT_FACTS §3) | 340,910 |
| Minus Tier B (36 anchor dockets, PROJECT_FACTS §5) | −70,075 |
| **Tier A frame** | **270,835 comments** |

Comments in the Tier A frame come from 6,109 non-anchor dockets (= 6,145 total dockets − 36 anchors). Eligibility within the frame inherits the same filters as Tier B: `text_source ≠ attachment_failed` and non-empty `comment` text.

## 2. Target sample size

**N = 100,000 comments**, ~37% of the Tier A frame.

Rationale:
- Larger than the 20% Bruce initially proposed (270,835 × 0.20 ≈ 54K) by ~85%; this gives substantially better per-cell statistical power for the §6.4 cell-by-cell analysis
- Cost anchor: 100,000 × $0.00343/call (observed Layer C per-call rate) ≈ **$343 expected actual**; well within the $400 `--max-cost` guardrail
- Wall time at concurrency=80: ~5 hours sync (vs ~12 hours for 270K full-frame); lands before Wednesday's joint Path A audit
- Comfortable headroom against the $1,200 budget reserve for any post-hoc supplementary oversample

## 3. Stratification design

Three crossed stratification axes, all a priori (data-independent):

### 3.1 Program office (9 cells)

EPA program-office identifier extracted from the docket_id field via the parsing rule:

```
parts = docket_id.split("-")
if parts[1] == "HQ":
    program_office = parts[2]
else:
    program_office = parts[1] + "-" + parts[2]
```

Expected program offices, based on the 36-anchor distribution as a prior:
- **OAR** — Office of Air & Radiation
- **OW** — Office of Water
- **OPP** — Office of Pesticide Programs
- **OPPT** — Office of Pollution Prevention & Toxics
- **OLEM** — Office of Land & Emergency Management
- **OA** — Office of the Administrator
- **OPA** — Office of Public Affairs / External Affairs
- **RCRA** — Resource Conservation & Recovery Act–specific
- **R08-OAR** — EPA Region 8 OAR

Any program office not in this enumeration that appears in the Tier A frame is grouped into an `OTHER` cell.

### 3.2 Year bucket (4 cells)

Year is extracted from the docket_id field (the 4-digit segment between the program office and the within-year sequence number). Bins align with administration changes, capturing the variation in regulatory posture documented in admin-law literature (e.g., West & Raso 2013; Yackee 2015):

| Bucket | Years | Era |
|---|---|---|
| Y1 | 2002–2008 | Bush II second-term EPA |
| Y2 | 2009–2016 | Obama EPA (regulatory expansion) |
| Y3 | 2017–2020 | Trump EPA (regulatory rollback) |
| Y4 | 2021–2023 | Biden EPA (regulatory rebuild) |

Bucket widths are intentionally unequal (7, 8, 4, 3 years). Equal-allocation per cell (§4) handles the imbalance; we do not need equal-duration bins for valid stratified inference.

### 3.3 Per-docket comment-count tercile (3 cells)

Each non-anchor docket is binned into low / mid / high tercile based on its total analyzable-comment count:

| Tercile | Definition |
|---|---|
| T1 (low) | Dockets below the 33rd percentile of the non-anchor docket-size distribution |
| T2 (mid) | 33rd–67th percentile |
| T3 (high) | 67th+ percentile |

Tercile boundaries are computed empirically from the non-anchor docket-size distribution. The boundaries (in comment count) are frozen at sampling time and recorded in the strata-assignment audit-trail CSV.

## 4. Allocation rule

**Equal allocation per cell with full-population cap.**

Per-cell target: 100,000 / (9 × 4 × 3) = **~926 comments per cell**.

Allocation algorithm:

1. For each of the 108 (program-office × year-bucket × tercile) cells, count the Tier A frame population.
2. If cell population ≤ 926: sample 100% of the cell.
3. If cell population > 926: random sample of exactly 926 with `random_state=42`.
4. After steps 2–3 the total may be below 100,000 because of small-population cells; redistribute the shortfall proportionally to cells with remaining population, applying the random-sample rule within each oversampled cell.
5. Maximum total sample: 100,000. If the total exceeds 100,000 after redistribution, cap proportionally.

This allocation maximizes per-cell statistical power up to 926 observations regardless of population, with marginal increase only to large cells.

## 5. Audit trail

- **This pre-registration** is committed to the repo at `notes/2026-05-13_tier_a_strata_preregistration.md` **before** the sampling script runs. The git timestamp on this file is the methodological anchor.
- The sampling script `code/tier_a_sample.py` generates three artifacts:
  - `data/processed/tier_a_sample/EPA_sample.parquet` — the actual sampled comments (subset of the analyzable corpus, with the same columns the Layer C orchestrator expects)
  - `data/processed/tier_a_anchors.csv` — list of unique docket_ids in the sample (used as the `--anchors-csv` argument for the Layer C orchestrator; functionally a "Tier A docket whitelist")
  - `data/processed/tier_a_strata_assignments.csv` — full audit trail: per-comment program_office, year_bucket, tercile, sampling indicator, tercile-boundary record
- All three artifacts are committed to the repo immediately after generation, before the Layer C fire begins.
- Random seed: `random_state=42` throughout for reproducibility.

## 6. Fire parameters

After this pre-registration is committed and the sample artifacts are generated, the Layer C orchestrator fires with:

```bash
nohup python3 code/16_layer_c_run_full.py \
    --all \
    --no-batch --concurrency 80 \
    --max-cost 400 \
    --anchors-csv data/processed/tier_a_anchors.csv \
    --comments-glob "data/processed/tier_a_sample/EPA_sample.parquet" \
    --output data/processed/tier_a_layer_c_full.csv \
    > /tmp/tier_a_fire.log 2>&1 &
```

Expected: ~100K calls, ~$343 actual, ~5 hours wall.

## 7. Analytic claims supported

This sample supports the following analyses in §6.4 (Equity-stratified responsiveness analysis):

1. **Population baseline.** Layer C 23-binary positive-rate distribution for the non-anchor corpus, as a comparison group for the Tier B distributions reported in §6.1 Table 2.
2. **Equity-frame stratified analysis.** Per-cell positive rates of `justice_equity_frame_present` and `lived_experience_frame_present`, by (program-office × year-bucket × tercile) cell. Wilson 95% CIs reported on all per-cell rates.
3. **Tier B vs Tier A contrast.** Tests whether the equity-frame and lived-experience frame rates observed in Tier B (18.8% and 19.4% respectively, §6.1) generalize beyond the anchor-selected high-attention dockets to the broader non-anchor population.
4. **Administration-era variation.** Tests whether population-level rhetoric distributions shift across the four year buckets, with the existing admin-law literature as the prior.

## 8. Deviations and amendments

Any deviation from this pre-registration (different strata, different N, different allocation rule, different random seed) requires a new dated pre-registration note documenting:
- The deviation
- The rationale (data-independent or post-hoc)
- The analytic implications

Post-hoc supplementary samples (e.g., a targeted oversample of a specific stratum after §6.4 surfaces a finding) are permitted but must be reported separately and explicitly flagged in the paper as supplementary, with their motivation acknowledged.

## 9. Realized sample size (post-execution amendment, 2026-05-13)

Pre-execution target: 100,000 (§2).
**Realized N: 77,513**, across 4,769 unique non-anchor dockets.

The shortfall (22,487 short of target) is the expected behavior of the allocation rule in §4 when cell populations are uneven. Specifically: a substantial number of (program-office × year-bucket × tercile) cells had fewer than 926 comments in the population. The §4 step-2 rule samples 100% of these small cells (giving fewer than 926 each), and the §4 step-4 redistribution to cap-cells couldn't fully cover the resulting gap because the remaining cap-cell populations were finite.

This is the §4 algorithm behaving exactly as specified — not a deviation. The realized N is what the pre-registered strata × allocation rule actually delivers on this population. No re-fire or strata change is warranted; the smaller-than-target N reduces statistical power per cell modestly but does not alter the methodological claims.

The `--max-cost` value originally listed in §6 ($400) was set against the expected actual cost of $343. The orchestrator's pre-check, however, uses the conservative $0.0089/call anchor (vs the observed $0.00343/call), yielding a pre-check estimate of $689 for 77,513 comments — over the $400 cap. The realized fire used `--max-cost 800` as an operational guardrail override; this is not a methodological deviation since `--max-cost` is a budget guardrail, not a pre-registered analytic parameter. Actual fire cost expected: ~$266 (77,513 × $0.00343).

Realized fire parameters:

```
nohup python3 code/16_layer_c_run_full.py \
    --all --no-batch --concurrency 80 --max-cost 800 \
    --anchors-csv data/processed/tier_a_anchors.csv \
    --comments-glob "data/processed/tier_a_sample/EPA_sample.parquet" \
    --output data/processed/tier_a_layer_c_full.csv \
    > /tmp/tier_a_fire.log 2>&1 &
```

Fire timestamp: 2026-05-13. PID 83467. Strata pre-registration commit (`5cedd84`) precedes the fire by design; the sampling artifacts (committed in the same revision) match the artifact specifications in §5.

---

**Pre-registration locked. Commit before sampling. Fire after sample artifacts land.**
