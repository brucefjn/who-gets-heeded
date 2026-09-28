# Anchor Selection Log — Option 2 stratified attachment retrieval

_Generated 2026-05-06T02:16:13+00:00_

## Eligibility filter

Single rule. **No text-content threshold.** A `>=30% text-content`
filter was considered in an earlier draft and withdrawn — it would
have collapsed the 2010-2012 cluster to a single Pesticide Petitions
docket (`EPA-HQ-OPP-2010-0889`), and filtering on text-content while
stratifying on era (which correlates with attachment-only rate) is
methodologically tense. Text-content rate is treated as a downstream
analysis variable rather than a sample-frame filter.

Active criteria:

- `n_comments >= 500`
  - Rigor: published-and-cited (Libgober & Rashin 2023, high-stakes-volume cutoff).
- Docket has at least one Proposed Rule AND one Final Rule (or Withdrawal) posted 2010-2022.
  - Rigor: published-and-cited (standard scope filter for rulemaking outcome studies).

**Total dockets observed:** 6,145
**Eligible after filter:** 72

Per-cluster eligible counts:

| Cluster | Eligible | Total comments | Min n | Median | Max n |
|---|---:|---:|---:|---:|---:|
| 2010-2012 | 21 | 113,026 | 563 | 2,993 | 16,868 |
| 2013-2015 | 19 | 105,585 | 506 | 2,399 | 34,479 |
| 2016-2018 | 18 | 108,801 | 549 | 2,259 | 26,206 |
| 2019-2022 | 14 | 19,113 | 586 | 897 | 5,294 |

## Step 1 — Stratified random sample (PRIMARY)

- Seed: `20260505`
- Strata: 4 year clusters x 3 within-cluster volume tertiles = 12
- N per stratum: 2
- Tertile boundaries computed within each cluster (33rd, 67th percentiles, method='lower') so volume distribution shifts across years don't collapse strata.
- Rigor: published-and-cited (Lohr 2009; Sage Encyclopedia of Educational Research).

Per-stratum draws:

| Cluster | Tertile | Pool size | Tertile boundaries | Drawn | Shortfall | Picks |
|---|---|---:|---|---:|---:|---|
| 2010-2012 | low | 6 | <1,452 | 1,452-5,811 | >=5,812 | 2 | 0 | `EPA-HQ-OW-2008-0667`, `EPA-HQ-OAR-2009-0926` |
| 2010-2012 | mid | 7 | <1,452 | 1,452-5,811 | >=5,812 | 2 | 0 | `EPA-HQ-OW-2009-0596`, `EPA-HQ-OAR-2003-0119` |
| 2010-2012 | high | 8 | <1,452 | 1,452-5,811 | >=5,812 | 2 | 0 | `EPA-HQ-OAR-2009-0517`, `EPA-HQ-OAR-2010-0505` |
| 2013-2015 | low | 6 | <1,028 | 1,028-4,040 | >=4,041 | 2 | 0 | `EPA-HQ-OAR-2014-0828`, `EPA-HQ-OPA-2006-0090` |
| 2013-2015 | mid | 6 | <1,028 | 1,028-4,040 | >=4,041 | 2 | 0 | `EPA-HQ-OPP-2011-0184`, `EPA-HQ-OA-2014-0012` |
| 2013-2015 | high | 7 | <1,028 | 1,028-4,040 | >=4,041 | 2 | 0 | `EPA-HQ-OW-2011-0880`, `EPA-HQ-OAR-2013-0602` |
| 2016-2018 | low | 5 | <1,033 | 1,033-4,721 | >=4,722 | 2 | 0 | `EPA-HQ-OW-2017-0644`, `EPA-HQ-OAR-2015-0531` |
| 2016-2018 | mid | 6 | <1,033 | 1,033-4,721 | >=4,722 | 2 | 0 | `EPA-R08-OAR-2015-0463`, `EPA-HQ-OLEM-2017-0286` |
| 2016-2018 | high | 7 | <1,033 | 1,033-4,721 | >=4,722 | 2 | 0 | `EPA-HQ-OPPT-2018-0159`, `EPA-HQ-OW-2017-0203` |
| 2019-2022 | low | 4 | <744 | 744-1,041 | >=1,042 | 2 | 0 | `EPA-HQ-OAR-2021-0668`, `EPA-HQ-OAR-2020-0044` |
| 2019-2022 | mid | 4 | <744 | 744-1,041 | >=1,042 | 2 | 0 | `EPA-HQ-OAR-2015-0072`, `EPA-HQ-OAR-1994-0099` |
| 2019-2022 | high | 6 | <744 | 744-1,041 | >=1,042 | 2 | 0 | `EPA-HQ-OAR-2004-0265`, `EPA-HQ-OAR-2018-0794` |

**Total drawn:** 24 (target 24). Shortfall: 0.

## Step 2 — Extreme-case additions (Obama-Trump regulatory-reversal pairs)

Detected pairs where:

- D1 has Final Rule posted 2014-2016 (Obama-era)
- D2 has Proposed Rule posted 2017-2020 (Trump-era)
- AND (cfr_overlap >= 3 OR textual_reference matches)
- Both halves of every detected pair are added; duplicates against Step-1 sample are dropped.
- Rigor: novel-but-defensible (Klotz 2008, Gerring 2007; specific operationalization is ours).

**Criteria refinements applied 2026-05-06 after pilot run:**

- *CFR threshold raised from 2 to 3.* Parts 60 (NSPS) and 63 (NESHAP) are parent CFR locations for
  hundreds of unrelated EPA rules. Two-part overlap (e.g., `(40,60)+(40,63)`) is weak evidence of
  regulatory relatedness; >=3 targets rule-subsystem-level overlap. Pilot run had 6 pairs that
  matched only on generic Part 60+63 co-occurrence and had no substantive policy connection.
  Rigor: novel-but-defensible.
- *Reversal-keyword vocabulary expanded* from {repeal, rescind, withdraw} to the canonical
  administrative-law reversal-action verb set, including recodify, supersede, reconsider, vacate
  (and inflectional variants). Under-specification fix to canonical category, not post-hoc
  adjustment to capture any specific pair. Vocabulary corresponds to standard APA-section
  regulatory-action verbs. WOTUS-recodification was the surface defect that exposed the
  under-specification (its Trump 2017 proposed rule used `Recodification`, which the original
  three-word set missed).
  Rigor: published-but-adapted.

Per the methodology rigor stopping rule (Bruce 2026-05-06): one more criteria revision is
permissible if a specific surface defect appears in this run; beyond that, the criteria are
locked as-is.

**27 pair(s) detected**, 18 unique docket(s):

| D1 (Obama Final) | D2 (Trump Proposed) | Evidence |
|---|---|---|
| `EPA-HQ-OAR-2002-0058` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sources, standards) |
| `EPA-HQ-OAR-2002-0058` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 5 shared title words (emission, hazardous, national, pollutants, standards) |
| `EPA-HQ-OAR-2003-0119` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 6 shared title words (emission, existing, guidelines, sources, stationary, units) |
| `EPA-HQ-OAR-2003-0119` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sources, standards) |
| `EPA-HQ-OAR-2003-0119` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, standards, units) |
| `EPA-HQ-OAR-2006-0790` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sources, standards) |
| `EPA-HQ-OAR-2006-0790` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 5 shared title words (emission, hazardous, national, pollutants, standards) |
| `EPA-HQ-OAR-2009-0234` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 5 shared title words (electric, emission, generating, units, utility) |
| `EPA-HQ-OAR-2009-0234` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, reconsideration, standards) |
| `EPA-HQ-OAR-2009-0234` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 11 shared title words (coal, electric, emission, fired, generating, hazardous) |
| `EPA-HQ-OAR-2009-0491` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 4 shared title words (existing, generating, pollution, units) |
| `EPA-HQ-OAR-2010-0505` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 4 shared title words (natural, reconsideration, sector, standards) |
| `EPA-HQ-OAR-2010-0682` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sector, standards) |
| `EPA-HQ-OAR-2010-0682` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 4 shared title words (review, risk, standards, technology) |
| `EPA-HQ-OAR-2011-0044` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 5 shared title words (electric, emission, generating, units, utility) |
| `EPA-HQ-OAR-2011-0044` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, reconsideration, standards) |
| `EPA-HQ-OAR-2011-0044` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 13 shared title words (coal, electric, emission, fired, generating, hazardous) |
| `EPA-HQ-OAR-2011-0817` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 5 shared title words (emission, hazardous, national, pollutants, standards) |
| `EPA-HQ-OAR-2013-0495` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 6 shared title words (electric, generating, sources, stationary, units, utility) |
| `EPA-HQ-OAR-2013-0495` | `EPA-HQ-OAR-2017-0483` | textual_ref: reversal keyword `reconsider` + 4 shared title words (modified, reconstructed, sources, standards) |
| `EPA-HQ-OAR-2013-0495` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 5 shared title words (electric, generating, standards, units, utility) |
| `EPA-HQ-OAR-2013-0602` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 11 shared title words (carbon, electric, emission, existing, generating, guidelines) |
| `EPA-HQ-OAR-2013-0602` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 5 shared title words (electric, emission, generating, units, utility) |
| `EPA-HQ-OW-2009-0819` | `EPA-HQ-OAR-2017-0355` | textual_ref: reversal keyword `repeal` + 3 shared title words (electric, generating, guidelines) |
| `EPA-HQ-OW-2009-0819` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 4 shared title words (electric, generating, standards, steam) |
| `EPA-HQ-OW-2011-0880` | `EPA-HQ-OW-2017-0203` | textual_ref: reversal keyword `recodification` + 4 shared title words (definition, states, united, waters) |
| `EPA-HQ-RCRA-2009-0640` | `EPA-HQ-OAR-2018-0794` | textual_ref: reversal keyword `reconsider` + 3 shared title words (coal, electric, hazardous) |

Pair detail:

- **`EPA-HQ-OAR-2002-0058` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2002-0058-3941): _National Emission Standards for Hazardous Air Pollutants for Major Sources: Industrial, Commercial, _  FR: 2015-29186
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sources, standards)
- **`EPA-HQ-OAR-2002-0058` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2002-0058-3941): _National Emission Standards for Hazardous Air Pollutants for Major Sources: Industrial, Commercial, _  FR: 2015-29186
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 5 shared title words (emission, hazardous, national, pollutants, standards)
- **`EPA-HQ-OAR-2003-0119` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OAR-2003-0119-2722): _Standards of Performance for New Stationary Sources and Emission Guidelines for Existing Sources: Co_  FR: 2016-13687
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 6 shared title words (emission, existing, guidelines, sources, stationary, units)
- **`EPA-HQ-OAR-2003-0119` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2003-0119-2722): _Standards of Performance for New Stationary Sources and Emission Guidelines for Existing Sources: Co_  FR: 2016-13687
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sources, standards)
- **`EPA-HQ-OAR-2003-0119` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2003-0119-2722): _Standards of Performance for New Stationary Sources and Emission Guidelines for Existing Sources: Co_  FR: 2016-13687
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, standards, units)
- **`EPA-HQ-OAR-2006-0790` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2006-0790-2576): _National Emission Standards: Hazardous Air Pollutants for Area Sources; Industrial, Commercial, and _  FR: 2016-21334
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sources, standards)
- **`EPA-HQ-OAR-2006-0790` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2006-0790-2576): _National Emission Standards: Hazardous Air Pollutants for Area Sources; Industrial, Commercial, and _  FR: 2016-21334
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 5 shared title words (emission, hazardous, national, pollutants, standards)
- **`EPA-HQ-OAR-2009-0234` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OAR-2009-0234-20450): _National Emission Standards for Hazardous Air Pollutants and Standards of Performance: Reconsiderati_  FR: 2014-27125
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 5 shared title words (electric, emission, generating, units, utility)
- **`EPA-HQ-OAR-2009-0234` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2009-0234-20450): _National Emission Standards for Hazardous Air Pollutants and Standards of Performance: Reconsiderati_  FR: 2014-27125
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, reconsideration, standards)
- **`EPA-HQ-OAR-2009-0234` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2009-0234-20448): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Steam Generat_  FR: 2014-27126
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 11 shared title words (coal, electric, emission, fired, generating, hazardous)
- **`EPA-HQ-OAR-2009-0491` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OAR-2009-0491-5033): _Data on Allocations of Cross-State Air Pollution Rule Allowances to Existing Electricity Generating _  FR: 2014-28281
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 4 shared title words (existing, generating, pollution, units)
- **`EPA-HQ-OAR-2010-0505` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2010-0505-4745): _Oil and Natural Gas Sector: Reconsideration of Additional Provisions of New Source Performance Stand_  FR: 2014-30630
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 4 shared title words (natural, reconsideration, sector, standards)
- **`EPA-HQ-OAR-2010-0682` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2010-0682-0871): _National Emission Standards for Hazardous Air Pollutant Emissions: Petroleum Refinery Sector_  FR: 2016-16451
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, sector, standards)
- **`EPA-HQ-OAR-2010-0682` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2010-0682-0700): _Petroleum Refinery Sector Risk and Technology Review and New Source Performance Standards_  FR: 2015-26486
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 4 shared title words (review, risk, standards, technology)
- **`EPA-HQ-OAR-2011-0044` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OAR-2011-0044-5852): _National Emission Standards for Hazardous Air Pollutants and Standards of Performance: Reconsiderati_  FR: 2014-27125
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 5 shared title words (electric, emission, generating, units, utility)
- **`EPA-HQ-OAR-2011-0044` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2011-0044-5852): _National Emission Standards for Hazardous Air Pollutants and Standards of Performance: Reconsiderati_  FR: 2014-27125
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (emission, reconsideration, standards)
- **`EPA-HQ-OAR-2011-0044` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2011-0044-5852): _National Emission Standards for Hazardous Air Pollutants and Standards of Performance: Reconsiderati_  FR: 2014-27125
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 13 shared title words (coal, electric, emission, fired, generating, hazardous)
- **`EPA-HQ-OAR-2011-0817` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2011-0817-0865): _National Emission Standards for Hazardous Air Pollutants: Portland Cement Manufacturing Industry and_  FR: 2015-16811
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 5 shared title words (emission, hazardous, national, pollutants, standards)
- **`EPA-HQ-OAR-2013-0495` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OAR-2013-0495-11310): _Standards of Performance: Greenhouse Gas Emissions from New, Modified, and Reconstructed Stationary _  FR: 2015-22837
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 6 shared title words (electric, generating, sources, stationary, units, utility)
- **`EPA-HQ-OAR-2013-0495` -> `EPA-HQ-OAR-2017-0483`**
  - D1 (EPA-HQ-OAR-2013-0495-11310): _Standards of Performance: Greenhouse Gas Emissions from New, Modified, and Reconstructed Stationary _  FR: 2015-22837
  - D2 (EPA-HQ-OAR-2017-0483-0005): _Oil and Natural Gas Sector: Emission Standards for New, Reconstructed, and Modified Sources Reconsid_
  - Evidence: textual_ref: reversal keyword `reconsider` + 4 shared title words (modified, reconstructed, sources, standards)
- **`EPA-HQ-OAR-2013-0495` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2013-0495-11310): _Standards of Performance: Greenhouse Gas Emissions from New, Modified, and Reconstructed Stationary _  FR: 2015-22837
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 5 shared title words (electric, generating, standards, units, utility)
- **`EPA-HQ-OAR-2013-0602` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OAR-2013-0602-36051): _Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Generating Un_  FR: 2015-22842
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 11 shared title words (carbon, electric, emission, existing, generating, guidelines)
- **`EPA-HQ-OAR-2013-0602` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OAR-2013-0602-36051): _Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Generating Un_  FR: 2015-22842
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 5 shared title words (electric, emission, generating, units, utility)
- **`EPA-HQ-OW-2009-0819` -> `EPA-HQ-OAR-2017-0355`**
  - D1 (EPA-HQ-OW-2009-0819-5558): _Effluent Limitations Guidelines and Standards for the Steam Electric Power Generating Point Source C_  FR: 2015-25663
  - D2 (EPA-HQ-OAR-2017-0355-0002): _Repeal of Carbon Pollution Emission Guidelines for Existing Stationary Sources: Electric Utility Gen_
  - Evidence: textual_ref: reversal keyword `repeal` + 3 shared title words (electric, generating, guidelines)
- **`EPA-HQ-OW-2009-0819` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-OW-2009-0819-5558): _Effluent Limitations Guidelines and Standards for the Steam Electric Power Generating Point Source C_  FR: 2015-25663
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 4 shared title words (electric, generating, standards, steam)
- **`EPA-HQ-OW-2011-0880` -> `EPA-HQ-OW-2017-0203`**
  - D1 (EPA-HQ-OW-2011-0880-20862): _Clean Water Rule: Definition of Waters of the United States_  FR: 2015-13435
  - D2 (EPA-HQ-OW-2017-0203-0001): _Definition of Waters of United States - Recodification of Pre-Existing
Rules_
  - Evidence: textual_ref: reversal keyword `recodification` + 4 shared title words (definition, states, united, waters)
- **`EPA-HQ-RCRA-2009-0640` -> `EPA-HQ-OAR-2018-0794`**
  - D1 (EPA-HQ-RCRA-2009-0640-11970): _Hazardous and Solid Waste Management Systems: Disposal of Coal Combustion Residuals from Electric Ut_  FR: 2015-00257
  - D2 (EPA-HQ-OAR-2018-0794-0001): _National Emission Standards for Hazardous Air Pollutants: Coal- and Oil-Fired Electric Utility Steam_
  - Evidence: textual_ref: reversal keyword `reconsider` + 3 shared title words (coal, electric, hazardous)

## Composite locked anchor list

- Stratified-only: 18
- Extreme-only: 12
- Both (stratified + also extreme): 6
- **Total unique anchor dockets: 36**

| docket_id | year_cluster | proposed | final | n_comments | %attach | method |
|---|---|---:|---:|---:|---:|---|
| `EPA-HQ-OAR-2009-0234` | 2010-2012 | 2011 | 2012 | 16,868 | 95% | extreme_case |
| `EPA-HQ-OAR-2009-0517` | 2010-2012 | 2012 | 2010 | 13,405 | 100% | stratified_random |
| `EPA-HQ-OAR-2010-0505` | 2010-2012 | 2011 | 2012 | 11,869 | 84% | stratified_random+extreme_case |
| `EPA-HQ-RCRA-2009-0640` | 2010-2012 | 2010 | 2015 | 11,493 | 91% | extreme_case |
| `EPA-HQ-OAR-2011-0044` | 2010-2012 | 2011 | 2012 | 5,812 | 97% | extreme_case |
| `EPA-HQ-OAR-2009-0491` | 2010-2012 | 2010 | 2011 | 3,992 | 91% | extreme_case |
| `EPA-HQ-OAR-2002-0058` | 2010-2012 | 2010 | 2011 | 2,920 | 88% | extreme_case |
| `EPA-HQ-OAR-2003-0119` | 2010-2012 | 2010 | 2011 | 2,527 | 96% | stratified_random+extreme_case |
| `EPA-HQ-OAR-2006-0790` | 2010-2012 | 2010 | 2011 | 2,325 | 94% | extreme_case |
| `EPA-HQ-OW-2009-0596` | 2010-2012 | 2010 | 2010 | 1,452 | 83% | stratified_random |
| `EPA-HQ-OW-2008-0667` | 2010-2012 | 2011 | 2014 | 1,409 | 89% | stratified_random |
| `EPA-HQ-OAR-2009-0926` | 2010-2012 | 2010 | 2010 | 792 | 97% | stratified_random |
| `EPA-HQ-OAR-2011-0817` | 2010-2012 | 2012 | 2013 | 563 | 78% | extreme_case |
| `EPA-HQ-OAR-2013-0602` | 2013-2015 | 2014 | 2015 | 34,479 | 87% | stratified_random+extreme_case |
| `EPA-HQ-OW-2011-0880` | 2013-2015 | 2014 | 2015 | 20,594 | 68% | stratified_random+extreme_case |
| `EPA-HQ-OAR-2013-0495` | 2013-2015 | 2014 | 2015 | 11,935 | 79% | extreme_case |
| `EPA-HQ-OW-2009-0819` | 2013-2015 | 2013 | 2015 | 4,161 | 81% | extreme_case |
| `EPA-HQ-OPP-2011-0184` | 2013-2015 | 2013 | 2015 | 2,399 | 10% | stratified_random |
| `EPA-HQ-OA-2014-0012` | 2013-2015 | 2014 | 2014 | 1,409 | 34% | stratified_random |
| `EPA-HQ-OAR-2014-0828` | 2013-2015 | 2015 | 2016 | 806 | 20% | stratified_random |
| `EPA-HQ-OPA-2006-0090` | 2013-2015 | 2015 | 2021 | 596 | 5% | stratified_random |
| `EPA-HQ-OAR-2010-0682` | 2013-2015 | 2014 | 2015 | 506 | 64% | extreme_case |
| `EPA-HQ-OAR-2017-0355` | 2016-2018 | 2017 | 2019 | 26,206 | 31% | extreme_case |
| `EPA-HQ-OW-2017-0203` | 2016-2018 | 2017 | 2019 | 15,618 | 39% | stratified_random+extreme_case |
| `EPA-HQ-OPPT-2018-0159` | 2016-2018 | 2018 | 2019 | 5,894 | 1% | stratified_random |
| `EPA-HQ-OLEM-2017-0286` | 2016-2018 | 2018 | 2018 | 2,371 | 36% | stratified_random |
| `EPA-HQ-OAR-2017-0483` | 2016-2018 | 2018 | 2020 | 2,148 | 50% | extreme_case |
| `EPA-R08-OAR-2015-0463` | 2016-2018 | 2016 | 2016 | 1,033 | 4% | stratified_random |
| `EPA-HQ-OW-2017-0644` | 2016-2018 | 2017 | 2018 | 707 | 33% | stratified_random |
| `EPA-HQ-OAR-2015-0531` | 2016-2018 | 2016 | 2017 | 549 | 83% | stratified_random |
| `EPA-HQ-OAR-2018-0794` | 2019-2022 | 2019 | 2020 | 5,294 | 54% | stratified_random+extreme_case |
| `EPA-HQ-OAR-2004-0265` | 2019-2022 | 2020 | 2020 | 1,196 | 100% | stratified_random |
| `EPA-HQ-OAR-2015-0072` | 2019-2022 | 2020 | 2020 | 958 | 56% | stratified_random |
| `EPA-HQ-OAR-1994-0099` | 2019-2022 | 2022 | 2022 | 770 | 100% | stratified_random |
| `EPA-HQ-OAR-2021-0668` | 2019-2022 | 2022 | 2022 | 709 | 23% | stratified_random |
| `EPA-HQ-OAR-2020-0044` | 2019-2022 | 2020 | 2020 | 690 | 10% | stratified_random |

## Step 3 — Robustness samples

- **alt-1**: stratified random with seed `20260506`, same strata, same n_per_stratum. Drawn anchors written to `anchor_rules_alt1_random_seed2.csv`.
  - Total drawn: 24.
- **alt-2**: top-25 dockets by `n_comments` (Libgober & Rashin 2023 style). Written to `anchor_rules_alt2_top_volume.csv`.
  - Volume range: 3,913 – 34,479

## Citations

- Lohr, S. L. (2009). _Sampling: Design and Analysis_ (2nd ed.). Brooks/Cole. — stratified sampling
- _Sage Encyclopedia of Educational Research, Measurement, and Evaluation_ — strata should minimize within-stratum variance
- Klotz, A. (2008). Case selection. In _Qualitative Methods in International Relations_ (pp. 43-58). — extreme-case methodology
- Gerring, J. (2007). _Case Study Research: Principles and Practices_. Cambridge University Press. — extreme-case methodology
- Libgober, B., & Rashin, S. (2023). [pending Columbia retrieval]. — high-stakes comment-volume framing; top-volume robustness specification