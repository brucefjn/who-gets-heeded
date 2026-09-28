# OCR Cross-Validation Summary — pytesseract vs vision-LLM

_Generated 2026-05-07T17:22:37+00:00_

## Method

For each of the pytesseract-extracted attachment PDFs (Stage 1 OCR), ran Claude Sonnet vision-LLM on the same source PDF and computed:

- `char_similarity`: rapidfuzz Levenshtein ratio (0-1) after lowercasing and whitespace normalization
- `word_recall`, `word_precision`, `word_f1`: set-based word-level metrics, vision-LLM treated as the asymmetric reference
- Word tokenization: alphanumeric only, lowercased, set-based

**Important:** vision-LLM is the reference for the metric only — *not* ground truth. Both methods can be wrong, in the same or different directions. High agreement here is evidence of consistency between two stochastic OCR methods, not of correctness. Direct ground-truth validation is captured separately in `notes/2026-05-07_vision_llm_validation.md`.

## Sample

- N total pytesseract files: 141
- N with successful vision-LLM run: 141

## F1 distribution

| Statistic | Value |
|---|---:|
| min          | 0.144 |
| 10th pct     | 0.912 |
| 25th pct     | 0.971 |
| median (50)  | 0.986 |
| mean         | 0.953 |
| 75th pct     | 0.991 |
| 90th pct     | 0.995 |
| 95th pct     | 0.997 |
| max          | 1.000 |

## Histogram (F1)

| F1 range | Count | % |
|---|---:|---:|
| [0.00, 0.50) | 3 | 2.1% |
| [0.50, 0.70) | 4 | 2.8% |
| [0.70, 0.85) | 2 | 1.4% |
| [0.85, 0.95) | 10 | 7.1% |
| [0.95, 1.00] | 122 | 86.5% |

```
  [0.00, 0.50)   |  3
  [0.50, 0.70)   | # 4
  [0.70, 0.85)   |  2
  [0.85, 0.95)   | ### 10
  [0.95, 1.00]   | ######################################## 122
```

## Distribution shape (auto-described)

Heavily concentrated in the high-agreement range. The two OCR methods agree on most word content for the typical file. Outliers in the low-F1 tail are candidates for individual inspection rather than evidence of systematic disagreement.

Char-similarity distribution (sanity reference): min=0.533, median=0.988, max=1.000.

## Top 5 worst-F1 files (candidates for manual inspection)

| doc_id | F1 | char_sim | pyt chars | vision chars |
|---|---:|---:|---:|---:|
| `EPA-HQ-RCRA-2009-0640-11185` | 0.144 | 0.533 | 2,147 | 2,915 |
| `EPA-HQ-OW-2009-0596-3024` | 0.430 | 0.694 | 531 | 651 |
| `EPA-HQ-OW-2009-0596-3027` | 0.447 | 0.795 | 1,781 | 1,844 |
| `EPA-HQ-OAR-2009-0234-16649` | 0.563 | 0.837 | 947 | 975 |
| `EPA-HQ-OW-2009-0596-1261` | 0.652 | 0.592 | 52,724 | 62,103 |

## Top 5 best-F1 files (sanity check)

| doc_id | F1 | char_sim | pyt chars | vision chars |
|---|---:|---:|---:|---:|
| `EPA-HQ-OAR-2010-0505-2525` | 1.000 | 1.000 | 1,741 | 1,742 |
| `EPA-HQ-OAR-2010-0505-1525` | 1.000 | 0.999 | 1,571 | 1,722 |
| `EPA-HQ-OAR-2015-0531-0317` | 1.000 | 0.991 | 2,803 | 2,802 |
| `EPA-HQ-OAR-2010-0682-0617` | 0.998 | 0.998 | 8,416 | 8,393 |
| `EPA-HQ-OA-2014-0012-0186` | 0.998 | 0.993 | 8,999 | 9,009 |

## Cost

- Tokens: 1,206,019 input + 431,898 output
- Estimated cost (Sonnet 4.5 @ $3/M input + $15/M output): $10.10

## Next step (joint decision — Bruce locks)

Inspect the distribution above. The threshold-setting decision space:

- **Binary**: accept (use pytesseract output as-is) / replace (overwrite with vision-LLM) at one F1 cutoff.
- **Tiered**: accept / sensitivity-flag / replace at two cutoffs.
- **No thresholds**: keep all pytesseract output, document distribution as descriptive robustness check.

The cutoff choice should be defensible against natural breaks in the empirical F1 distribution observed above. Document the calibration rationale in `data/processed/anchor_selection_log.md` and reference in the methods section. The methodology paragraph reads: "thresholds were calibrated post-hoc to the empirical F1 distribution observed across our N pytesseract extractions; we report distributional statistics in [appendix]."