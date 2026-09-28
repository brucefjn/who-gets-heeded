# Attachment-Rate Diagnostic — 39 regulations.gov CSVs

_Generated 2026-05-05T21:10:46+00:00_

## Critical findings (read before interpreting tables below)

### 1. FCC and SEC bulk-download CSVs contain ZERO public comments

Across all 26 FCC + SEC files, every row is an agency-issued document — `Document Type` is exclusively `Notice` / `Rule` / `Proposed Rule` / `Other`. Zero rows have a non-empty `Comment on Document ID`. Zero rows are `Document Type == 'Public Submission'`.

- FCC: 6,470 documents across 13 years; 0 comments.
- SEC: 27,652 documents across 13 years; 0 comments.

This is consistent with how those agencies route public comments: FCC operates its own ECFS comment system, and SEC accepts comment letters via its own file-number-indexed system. Neither agency routes comments through regulations.gov.

**Implication for the project:** the comment-influence analysis cannot proceed for FCC and SEC with the current bulk downloads. Options to flag:
1. Acquire FCC ECFS comment data and SEC comment-letter data via separate scrapers.
2. Restrict project scope to EPA only.
3. Revisit whether the FCC/SEC bulk downloads need different parameters (long shot — these CSVs are bulk dumps and don't appear to be filtered).

### 2. Submitter-identity fields are PII-redacted

For all 71,251 EPA 2018 public submissions, `Organization Name`, `Submitter Representative`, `First Name`, and `Last Name` are all empty (0% fill rate). The only identity signal is `Title`, which is 100% populated and follows recognizable patterns:

| Title pattern | n | meaning |
|---|---:|---|
| `Comment submitted by X.` | 35,391 | almost always individual |
| `Anonymous public comment` (in "other") | many | anonymous individual |
| `Comment from / Comments of <Org>` | 21 | mostly organizational |
| `Public Comment on …` | 4 | likely transcript-style from hearings |

**Implication for the project:** the Org-attached / Indiv-attached column in the per-file table below shows 0/all because the Organization Name field is uniformly empty — *not* because no orgs submitted attachment-only comments. To distinguish org from individual submitters we'd need to parse the `Title` field (heuristic: "submitted by [Initial. Lastname]" → individual, free-form name → likely org).

---

**Definitions used:**
- A *comment* row = `Comment on Document ID` non-empty OR `Document Type == 'Public Submission'` (union).
- An *attachment-only* comment = `Comment` field is empty/null OR matches a placeholder pattern after normalization (lowercase, whitespace-collapsed, leading 'please' stripped, trailing punctuation stripped). Patterns include:
  `see attached`, `see attached file(s)`, `see attached document`, `see attachment(s)`, `see the attached letter`, `see my/our attached comments`, `attached`, `comment(s) attached`, `letter attached`, `n/a`, `none`, etc. (full regex in script).
- *Org-attached* = attachment-only comment with `Organization Name` populated; *Indiv-attached* = attachment-only with empty `Organization Name`.

## Per-file

| File | Total rows | Comments | Attachment-only | % attach | Org-attached | Indiv-attached |
|---|---:|---:|---:|---:|---:|---:|
| EPA_2010_full.csv | 103,280 | 84,709 | 67,917 | 80.2% | 0 | 67,917 |
| EPA_2011_full.csv | 85,441 | 63,688 | 55,963 | 87.9% | 0 | 55,963 |
| EPA_2012_full.csv | 78,394 | 60,700 | 48,673 | 80.2% | 0 | 48,673 |
| EPA_2013_full.csv | 61,084 | 42,430 | 26,901 | 63.4% | 0 | 26,901 |
| EPA_2014_full.csv | 111,228 | 92,091 | 60,949 | 66.2% | 0 | 60,949 |
| EPA_2015_full.csv | 51,818 | 30,607 | 18,760 | 61.3% | 0 | 18,760 |
| EPA_2016_full.csv | 53,802 | 32,505 | 17,270 | 53.1% | 0 | 17,270 |
| EPA_2017_full.csv | 148,166 | 132,060 | 44,845 | 34.0% | 0 | 44,845 |
| EPA_2018_full.csv | 85,680 | 71,251 | 17,570 | 24.7% | 0 | 17,570 |
| EPA_2019_full.csv | 73,010 | 48,628 | 10,598 | 21.8% | 0 | 10,598 |
| EPA_2020_full.csv | 66,097 | 39,247 | 15,540 | 39.6% | 0 | 15,540 |
| EPA_2021_full.csv | 72,552 | 39,176 | 26,293 | 67.1% | 0 | 26,293 |
| EPA_2022_full.csv | 78,155 | 49,105 | 35,374 | 72.0% | 0 | 35,374 |
| FCC_2010_full.csv | 494 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2011_full.csv | 567 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2012_full.csv | 423 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2013_full.csv | 515 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2014_full.csv | 558 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2015_full.csv | 465 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2016_full.csv | 453 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2017_full.csv | 516 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2018_full.csv | 482 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2019_full.csv | 432 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2020_full.csv | 561 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2021_full.csv | 571 | 0 | 0 | 0.0% | 0 | 0 |
| FCC_2022_full.csv | 433 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2010_full.csv | 2,031 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2011_full.csv | 2,058 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2012_full.csv | 2,186 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2013_full.csv | 2,270 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2014_full.csv | 2,215 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2015_full.csv | 2,184 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2016_full.csv | 2,360 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2017_full.csv | 2,189 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2018_full.csv | 2,067 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2019_full.csv | 2,000 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2020_full.csv | 2,219 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2021_full.csv | 1,968 | 0 | 0 | 0.0% | 0 | 0 |
| SEC_2022_full.csv | 1,905 | 0 | 0 | 0.0% | 0 | 0 |

## Rollup by agency

| Agency | Files | Total rows | Comments | Attachment-only | % attach |
|---|---:|---:|---:|---:|---:|
| EPA | 13 | 1,068,707 | 786,197 | 446,653 | 56.8% |
| FCC | 13 | 6,470 | 0 | 0 | nan% |
| SEC | 13 | 27,652 | 0 | 0 | nan% |

## Rollup by year

| Year | Total rows | Comments | Attachment-only | % attach |
|---|---:|---:|---:|---:|
| 2010 | 105,805 | 84,709 | 67,917 | 80.2% |
| 2011 | 88,066 | 63,688 | 55,963 | 87.9% |
| 2012 | 81,003 | 60,700 | 48,673 | 80.2% |
| 2013 | 63,869 | 42,430 | 26,901 | 63.4% |
| 2014 | 114,001 | 92,091 | 60,949 | 66.2% |
| 2015 | 54,467 | 30,607 | 18,760 | 61.3% |
| 2016 | 56,615 | 32,505 | 17,270 | 53.1% |
| 2017 | 150,871 | 132,060 | 44,845 | 34.0% |
| 2018 | 88,229 | 71,251 | 17,570 | 24.7% |
| 2019 | 75,442 | 48,628 | 10,598 | 21.8% |
| 2020 | 68,877 | 39,247 | 15,540 | 39.6% |
| 2021 | 75,091 | 39,176 | 26,293 | 67.1% |
| 2022 | 80,493 | 49,105 | 35,374 | 72.0% |

## Overall

- Total rows across all 39 files: **1,102,829**
- Total comments: **786,197**
- Total attachment-only: **446,653** (56.8% of comments)
- Of attachment-only: **0** with Organization Name, **446,653** without

## Placeholder discovery

The working version of this diagnostic also listed the most frequent short comment texts, which were used to refine the attachment-only definition. That list is omitted from the public release because it reproduces comment text, including some submitter names.
