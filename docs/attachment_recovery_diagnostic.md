# Attachment-Failure Diagnostic — Falsification Check on the OCR-Candidate Set

_Generated 2026-05-06 (post-attachment-pull)_

## Question

Are the 148 OCR-candidate attachments (the failures pdfplumber + python-docx couldn't extract usable text from) **systematically biased** on year × submitter × docket × file-size dimensions, or **randomly distributed**? If systematic, recovery via OCR is methodologically necessary; if random, they can be documented as a known limitation and skipped.

## Definitions

- **OCR-candidate set (n=148):** comments where the extraction either (a) failed via `extract_fail:` with no specific reason recorded (pdfplumber gave up — usually image-only PDFs), or (b) succeeded but returned `n_chars < 100` (very-low-yield extracts, almost certainly scans where pdfplumber found page metadata only). These are the attachments OCR could plausibly recover.
- **Successful set (n=1,489):** ok extractions with `n_chars ≥ 100` (pdfplumber returned analytically usable text). After the +5 docx recovery completed earlier today.
- **Excluded from this comparison (n=132):** 127 `no_url_nan` (no attachment URL in the source CSV — OCR can't fix), 4 legacy `.doc` files (pre-2007 MS Word — OCR not relevant), 1 `MalformedPDFException`. These are out of scope for the OCR-vs-skip decision.

## Side-by-side distributions

### 1. Docket-year distribution (proxy from `docket_id`)

The docket year is the year the docket was opened (not when the comment was filed). Earlier-cohort dockets accumulated paper-era backfills that were scanned in.

| Docket year | Successful (%) | OCR-candidate (%) | Direction |
|---|---:|---:|---|
| 2002 | 2.5 | 8.8 | **Over (3.5×)** |
| 2003 | 2.9 | 4.7 | Over (1.6×) |
| 2008 | 2.6 | 7.4 | **Over (2.8×)** |
| 2009 | 19.6 | **36.5** | **Over (1.9×)** |
| 2010 | 6.3 | 4.1 | Under |
| 2011 | 11.8 | 12.8 | ~Equal |
| 2013 | 6.4 | 2.7 | Under |
| 2014 | 6.6 | 1.4 | Under |
| 2015 | 8.6 | 7.4 | ~Equal |
| 2017 | 15.9 | 6.8 | **Under (0.4×)** |
| 2018 | 5.4 | 1.4 | Under |
| 2020 | 3.2 | 1.4 | Under |
| 2021 | 3.4 | **0.0** | **Under (gone)** |

**Verdict on year: SYSTEMATIC EARLY-PERIOD CONCENTRATION.** Pre-2010 dockets (2002, 2003, 2008, 2009) account for **57.4% of OCR candidates** vs only 27.6% of successful extracts. Post-2017 dockets (2017–2021) are under-represented. Consistent with paper-era submissions backfilled into older dockets and never re-OCR'd.

### 2. Submitter-kind distribution

| Submitter kind | Successful (%) | OCR-candidate (%) | Direction |
|---|---:|---:|---|
| Individual | 45.9 | 14.2 | **Under (0.3×)** |
| Organizational | 36.6 | **70.3** | **Over (1.9×, +33.7 pp)** |
| Other / anonymous | 17.5 | 15.5 | ~Equal |

**Verdict on submitter: STRONGLY BIASED TOWARD ORGANIZATIONAL.** 70.3% of OCR-candidates are organizational submitters vs only 36.6% in the successful set — a **+33.7-percentage-point** over-representation. Lay individuals dominate the successful set. Organizational comments are systematically over-represented in the failures, almost certainly because industry / advocacy groups submit on scanned-letterhead PDFs while lay individuals type comments inline or attach simple text PDFs. **Excluding OCR-candidates would mute organized-interest voices specifically — the exact bias EAAMO/FAccT reviewers will scrutinize.**

### 3. Docket concentration

Top 10 dockets by OCR-candidate count, with within-docket OCR rate:

| Docket | OCR count | % of OCR set | OCR / sample-size | % of docket sample |
|---|---:|---:|---|---:|
| EPA-HQ-OAR-2002-0058 | 13 | 8.8% | 13/50 | **26.0%** |
| EPA-HQ-RCRA-2009-0640 | 11 | 7.4% | 11/50 | 22.0% |
| EPA-HQ-OW-2008-0667 | 11 | 7.4% | 11/50 | 22.0% |
| EPA-HQ-OAR-2009-0491 | 10 | 6.8% | 10/50 | 20.0% |
| EPA-HQ-OW-2009-0596 (FL Water QS) | 10 | 6.8% | 10/50 | 20.0% |
| EPA-HQ-OAR-2015-0531 | 9 | 6.1% | 9/50 | 18.0% |
| EPA-HQ-OAR-2009-0234 (MATS) | 9 | 6.1% | 9/50 | 18.0% |
| EPA-HQ-OAR-2011-0817 | 8 | 5.4% | 8/50 | 16.0% |
| EPA-HQ-OAR-2003-0119 | 7 | 4.7% | 7/50 | 14.0% |
| EPA-HQ-OAR-2006-0790 | 7 | 4.7% | 7/50 | 14.0% |

**Verdict on docket: MODERATE CONCENTRATION.** Top-10 dockets account for ~62% of all OCR candidates, with several pre-2010 dockets having 18–26% of their per-docket sample in the OCR-candidate set. If we skip OCR, those specific dockets lose 1/5 to 1/4 of their analytical sample.

### 4. File-size distribution

PDFs with embedded image data (scans) are much larger than text-bearing PDFs.

| Statistic | Successful (KB) | OCR-candidate (KB) | Ratio |
|---|---:|---:|---:|
| n | 1,489 | 148 | — |
| Median | 37 | **370** | **10.0×** |
| Mean | 385 | 1,048 | 2.7× |
| p25 | 14 | 92 | 6.6× |
| p75 | 175 | 971 | 5.5× |
| p95 | 1,091 | 4,145 | 3.8× |
| Max | 138,397 | 15,109 | — |

**Verdict on file size: STRONGLY BIASED TOWARD LARGE FILES.** Median OCR candidate is 10× the size of the median successful extract (370 KB vs 37 KB). This is the unmistakable signature of image-only PDFs vs text-bearing PDFs.

## Final verdict

**The failure pattern is biased on three of four dimensions** — year, submitter, file size — with the fourth (docket concentration) showing moderate clustering as well. Specifically:

- **Early-period docket cohorts** are over-represented (pre-2010 dockets = 57% of OCR set, 28% of baseline)
- **Organizational submitters** are dramatically over-represented (70% vs 37%)
- **File-size signature** is unambiguously consistent with image-only scanned PDFs

Skipping OCR and treating the 148 attachments as "documented limitation" would produce a sample that **systematically mutes organized-interest voices on pre-2010 rulemakings**. That is an analytically material exclusion, not noise. A reviewer will read our methodology section, see "we excluded 148 attachments where pdfplumber returned <100 chars," and immediately ask whether the exclusion is correlated with submitter type. We would have to answer "yes — strongly," which damages the analysis.

**Recommendation: OCR recovery is methodologically necessary, not optional.**

Implementation paths, in order of cost and quality:

1. **pytesseract on the 148 OCR-candidate PDFs.** Free, ~30 sec/file = ~75 minutes total. Quality: excellent on printed-then-scanned letterhead (95%+), poor on handwritten content (~30–50%). Rough estimate: recovers 100–125 of 148 with usable text.
2. **Multimodal LLM (Claude/GPT-4 vision) on the 148.** ~$0.01–0.03 per page. For an avg ~5-page scan and 148 files = $7–22 total. Quality: handles printed and most handwritten content well. Recovers nearly all 148.
3. **Hybrid:** pytesseract first; for the residual where pytesseract returns <100 chars, fall back to vision-LLM. Best quality/cost ratio.

For the May 9 course paper, **pytesseract is sufficient** if executed by Friday morning — recovers ~85% of the bias-introducing exclusions. For EAAMO (May 15), the hybrid is recommended.

Either way, the 148 are not random data loss. They're a non-random sample of organized-interest commentary on older rulemakings, and they need to come back into the corpus.
