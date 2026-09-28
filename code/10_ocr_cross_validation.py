"""
10_ocr_cross_validation.py — Cross-validate pytesseract via Claude vision-LLM.

For all 141 pytesseract extracts, run Claude Sonnet vision-LLM on the same
source PDFs and compute pairwise agreement metrics. **Measurement only** —
no acceptance thresholds set in this script. Thresholds get calibrated post-hoc
by Bruce after inspecting the empirical F1 distribution.

WHY ASYMMETRIC METRICS WITH VISION-LLM AS REFERENCE
  Vision-LLM has fewer documented systematic failure modes on handwriting,
  cursive, and low-resolution scans — the dominant content in this 141-file
  set (selected because pytesseract returned <100 chars or failed entirely
  before re-OCR). It is *not* ground truth; both methods can be wrong, in
  the same or different directions. High agreement is evidence of consistency,
  not accuracy. Direct ground-truth check is in Bruce's hand-validation pass
  on `notes/2026-05-07_vision_llm_validation.md`.

OUTPUTS
  data/processed/ocr_cross_validation.csv          per-file metrics
  data/processed/ocr_cross_validation_summary.md   distribution summary +
                                                   top-5 worst / best
  data/processed/ocr_cross_validation_vision_llm/<docket>/<doc>__1_vision.txt
                                                   raw vision-LLM transcripts,
                                                   for spot-check on outliers.

DEPENDENCIES
  pip3 install --user anthropic rapidfuzz pypdfium2
  export ANTHROPIC_API_KEY='...'

CITATIONS / RIGOR
  - Word-level F1 as cross-method agreement metric: novel-but-defensible
    operationalization (set-based, formatting-agnostic).
  - Levenshtein ratio: standard string-similarity metric (Levenshtein 1965;
    rapidfuzz implementation is the C++ reimplementation of fuzzywuzzy).
  - Empirical post-hoc threshold calibration when no canonical OCR-vs-OCR
    threshold exists in the literature: standard measurement-methodology
    practice. Reporting the full distribution lets reviewers evaluate cutoff
    choices against the actual data.

USAGE
  python3 code/10_ocr_cross_validation.py
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACH = PROJECT_ROOT / "data" / "raw" / "attachments"
LOG_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_extraction_log.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_sample.csv"
CROSS_VAL_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_cross_validation_vision_llm"
CROSS_VAL_CSV = PROJECT_ROOT / "data" / "processed" / "ocr_cross_validation.csv"
CROSS_VAL_SUMMARY = PROJECT_ROOT / "data" / "processed" / "ocr_cross_validation_summary.md"

DEFAULT_MODEL = "claude-sonnet-4-5"
RENDER_DPI = 150
MAX_OUTPUT_TOKENS = 4000

TRANSCRIBE_PROMPT = """Transcribe all text content visible in this PDF page image.

Output requirements:
1. Preserve paragraph structure with blank lines between paragraphs.
2. Preserve line breaks within addresses, signature blocks, lists, and tables.
3. Do not add commentary, summaries, or "[image of...]" descriptions.
4. If the page has handwritten content, transcribe it verbatim.
5. If a portion is illegible, write [illegible] in place of that portion.
6. Return only the transcribed text, no other content.
"""


def normalize_for_chars(text: str) -> str:
    """Normalize for Levenshtein: lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokenize_words(text: str) -> set:
    """Word-level tokenization for set comparison.
    Lowercased, whitespace-collapsed, alphanumeric only."""
    s = normalize_for_chars(text)
    return set(re.findall(r"[a-z0-9]+", s))


def char_similarity(a: str, b: str) -> float:
    """Levenshtein ratio (0-1) on normalized strings."""
    from rapidfuzz import fuzz
    na, nb = normalize_for_chars(a), normalize_for_chars(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return fuzz.ratio(na, nb) / 100.0


def word_metrics(pytesseract_text: str, vision_text: str) -> dict:
    """Asymmetric word-level set metrics. Vision-LLM is treated as reference."""
    p = tokenize_words(pytesseract_text)
    v = tokenize_words(vision_text)
    if not v and not p:
        return {"word_recall": 1.0, "word_precision": 1.0, "word_f1": 1.0}
    if not v or not p:
        return {"word_recall": 0.0, "word_precision": 0.0, "word_f1": 0.0}
    inter = p & v
    recall = len(inter) / len(v)        # of vision-LLM words, what fraction did pytesseract recover?
    precision = len(inter) / len(p)     # of pytesseract words, what fraction also appear in vision-LLM?
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {"word_recall": recall, "word_precision": precision, "word_f1": f1}


def pdf_pages_as_b64(pdf_path: Path, dpi: int = RENDER_DPI) -> list:
    import pypdfium2
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    out = []
    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            try:
                bitmap = page.render(scale=dpi / 72)
                pil = bitmap.to_pil()
            finally:
                page.close()
            buf = io.BytesIO()
            pil.save(buf, format="PNG", optimize=True)
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    finally:
        pdf.close()
    return out


def vision_llm_transcribe(pdf_path: Path, client, model: str) -> tuple:
    """Returns (text, in_tokens, out_tokens). Retries up to 3x on rate limit."""
    import anthropic
    pages = pdf_pages_as_b64(pdf_path)
    parts = []
    in_tok = 0
    out_tok = 0
    for b64 in pages:
        delay = 15.0
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": b64
                            }},
                            {"type": "text", "text": TRANSCRIBE_PROMPT},
                        ],
                    }],
                )
                parts.append(msg.content[0].text if msg.content else "")
                in_tok += msg.usage.input_tokens
                out_tok += msg.usage.output_tokens
                break
            except anthropic.RateLimitError:
                time.sleep(delay)
                delay *= 2
            except anthropic.APIError:
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
    return "\n\n".join(parts), in_tok, out_tok


def write_summary(df: pd.DataFrame, total_in: int, total_out: int) -> None:
    import numpy as np
    ok = df[df["vision_status"] == "ok"]
    f1 = ok["word_f1"]
    cs = ok["char_similarity"]

    L: list[str] = []
    L.append("# OCR Cross-Validation Summary — pytesseract vs vision-LLM")
    L.append("")
    L.append(f"_Generated {pd.Timestamp.utcnow().isoformat(timespec='seconds')}_")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append("For each of the pytesseract-extracted attachment PDFs (Stage 1 OCR), "
             "ran Claude Sonnet vision-LLM on the same source PDF and computed:")
    L.append("")
    L.append("- `char_similarity`: rapidfuzz Levenshtein ratio (0-1) after lowercasing and whitespace normalization")
    L.append("- `word_recall`, `word_precision`, `word_f1`: set-based word-level metrics, vision-LLM treated as the asymmetric reference")
    L.append("- Word tokenization: alphanumeric only, lowercased, set-based")
    L.append("")
    L.append("**Important:** vision-LLM is the reference for the metric only — *not* ground truth. "
             "Both methods can be wrong, in the same or different directions. High agreement here is "
             "evidence of consistency between two stochastic OCR methods, not of correctness. Direct "
             "ground-truth validation is captured separately in `notes/2026-05-07_vision_llm_validation.md`.")
    L.append("")

    L.append("## Sample")
    L.append("")
    L.append(f"- N total pytesseract files: {len(df)}")
    L.append(f"- N with successful vision-LLM run: {len(ok)}")
    n_fail = (df["vision_status"] != "ok").sum()
    if n_fail:
        L.append(f"- N with failed vision-LLM run (excluded from distribution stats): {n_fail}")
    L.append("")

    if len(ok) == 0:
        L.append("## No successful comparisons — diagnose `vision_status` values in the CSV.")
        CROSS_VAL_SUMMARY.write_text("\n".join(L), encoding="utf-8")
        return

    L.append("## F1 distribution")
    L.append("")
    L.append("| Statistic | Value |")
    L.append("|---|---:|")
    L.append(f"| min          | {f1.min():.3f} |")
    L.append(f"| 10th pct     | {np.percentile(f1, 10):.3f} |")
    L.append(f"| 25th pct     | {np.percentile(f1, 25):.3f} |")
    L.append(f"| median (50)  | {f1.median():.3f} |")
    L.append(f"| mean         | {f1.mean():.3f} |")
    L.append(f"| 75th pct     | {np.percentile(f1, 75):.3f} |")
    L.append(f"| 90th pct     | {np.percentile(f1, 90):.3f} |")
    L.append(f"| 95th pct     | {np.percentile(f1, 95):.3f} |")
    L.append(f"| max          | {f1.max():.3f} |")
    L.append("")

    L.append("## Histogram (F1)")
    L.append("")
    L.append("| F1 range | Count | % |")
    L.append("|---|---:|---:|")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0001)]
    for lo, hi in bins:
        if hi > 1.0:
            n = ((f1 >= lo) & (f1 <= 1.0)).sum()
            label = f"[{lo:.2f}, 1.00]"
        else:
            n = ((f1 >= lo) & (f1 < hi)).sum()
            label = f"[{lo:.2f}, {hi:.2f})"
        pct = n / len(f1) * 100
        L.append(f"| {label} | {n} | {pct:.1f}% |")
    L.append("")
    # Visual ASCII histogram
    L.append("```")
    max_count = max(((f1 >= lo) & ((f1 < hi) if hi <= 1.0 else (f1 <= 1.0))).sum() for lo, hi in bins) or 1
    for lo, hi in bins:
        if hi > 1.0:
            n = ((f1 >= lo) & (f1 <= 1.0)).sum()
            label = f"[{lo:.2f}, 1.00]"
        else:
            n = ((f1 >= lo) & (f1 < hi)).sum()
            label = f"[{lo:.2f}, {hi:.2f})"
        bar = "#" * int(40 * n / max_count) if max_count else ""
        L.append(f"  {label:<14} | {bar} {n}")
    L.append("```")
    L.append("")

    L.append("## Distribution shape (auto-described)")
    L.append("")
    median_f1 = f1.median()
    p25, p75 = np.percentile(f1, 25), np.percentile(f1, 75)
    iqr = p75 - p25
    if median_f1 > 0.90 and p25 > 0.85:
        shape = ("Heavily concentrated in the high-agreement range. The two OCR methods agree on most "
                 "word content for the typical file. Outliers in the low-F1 tail are candidates for "
                 "individual inspection rather than evidence of systematic disagreement.")
    elif median_f1 > 0.70 and iqr < 0.30:
        shape = ("Right-skewed distribution: most files achieve moderate-to-high agreement, with a tail "
                 "of low-agreement files. Inspect the low-F1 tail to determine whether disagreement "
                 "reflects pytesseract failure modes or vision-LLM artifacts.")
    elif iqr > 0.30:
        shape = ("Broad distribution: substantial within-sample variation in cross-method agreement. "
                 "Multiple modes or a near-uniform spread would suggest pytesseract performs differently "
                 "on different document classes (e.g., printed vs handwritten). Worth sub-grouping.")
    else:
        shape = ("Distribution is bounded and concentrated. Look for natural breaks in the histogram "
                 "above to inform threshold choice.")
    L.append(shape)
    L.append("")
    L.append(f"Char-similarity distribution (sanity reference): "
             f"min={cs.min():.3f}, median={cs.median():.3f}, max={cs.max():.3f}.")
    L.append("")

    L.append("## Top 5 worst-F1 files (candidates for manual inspection)")
    L.append("")
    L.append("| doc_id | F1 | char_sim | pyt chars | vision chars |")
    L.append("|---|---:|---:|---:|---:|")
    for _, r in ok.nsmallest(5, "word_f1").iterrows():
        L.append(f"| `{r['doc_id']}` | {r['word_f1']:.3f} | {r['char_similarity']:.3f} | "
                 f"{int(r['pytesseract_chars']):,} | {int(r['vision_llm_chars']):,} |")
    L.append("")

    L.append("## Top 5 best-F1 files (sanity check)")
    L.append("")
    L.append("| doc_id | F1 | char_sim | pyt chars | vision chars |")
    L.append("|---|---:|---:|---:|---:|")
    for _, r in ok.nlargest(5, "word_f1").iterrows():
        L.append(f"| `{r['doc_id']}` | {r['word_f1']:.3f} | {r['char_similarity']:.3f} | "
                 f"{int(r['pytesseract_chars']):,} | {int(r['vision_llm_chars']):,} |")
    L.append("")

    cost = total_in * 3 / 1_000_000 + total_out * 15 / 1_000_000
    L.append("## Cost")
    L.append("")
    L.append(f"- Tokens: {total_in:,} input + {total_out:,} output")
    L.append(f"- Estimated cost (Sonnet 4.5 @ $3/M input + $15/M output): ${cost:.2f}")
    L.append("")

    L.append("## Next step (joint decision — Bruce locks)")
    L.append("")
    L.append("Inspect the distribution above. The threshold-setting decision space:")
    L.append("")
    L.append("- **Binary**: accept (use pytesseract output as-is) / replace (overwrite with vision-LLM) at one F1 cutoff.")
    L.append("- **Tiered**: accept / sensitivity-flag / replace at two cutoffs.")
    L.append("- **No thresholds**: keep all pytesseract output, document distribution as descriptive robustness check.")
    L.append("")
    L.append("The cutoff choice should be defensible against natural breaks in the empirical F1 distribution observed above. "
             "Document the calibration rationale in `data/processed/anchor_selection_log.md` and reference in the methods section. "
             "The methodology paragraph reads: \"thresholds were calibrated post-hoc to the empirical F1 distribution observed "
             "across our N pytesseract extractions; we report distributional statistics in [appendix].\"")

    CROSS_VAL_SUMMARY.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY env var.", file=sys.stderr)
        return 2

    try:
        import anthropic  # noqa: F401
        from rapidfuzz import fuzz  # noqa: F401
        import pypdfium2  # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dep {e.name}.", file=sys.stderr)
        print("Install: pip3 install --user anthropic rapidfuzz pypdfium2", file=sys.stderr)
        return 2

    log = pd.read_csv(LOG_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    docket_for = dict(zip(sample.document_id, sample.docket_id))

    pytess = log[log["kind"] == "pytesseract"].copy()
    print(f"[targets] {len(pytess)} pytesseract extracts to cross-validate against vision-LLM")
    if not len(pytess):
        print("Nothing to do.")
        return 0

    import anthropic
    # 120s timeout per API call so a single hung request fails fast instead of
    # stalling the whole run invisibly.
    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)

    rows: list[dict] = []
    total_in = 0
    total_out = 0
    t_start = time.time()

    for i, (_, r) in enumerate(pytess.iterrows(), 1):
        docid = r["document_id"]
        docket = docket_for.get(docid)
        if not docket:
            continue
        pdf_path = ATTACH / docket / f"{docid}__1.pdf"
        pyt_txt_path = ATTACH / docket / f"{docid}__1.txt"
        if not pdf_path.exists() or not pyt_txt_path.exists():
            rows.append({
                "doc_id": docid, "docket_id": docket,
                "pytesseract_chars": 0, "vision_llm_chars": 0,
                "char_similarity": 0.0,
                "word_recall": 0.0, "word_precision": 0.0, "word_f1": 0.0,
                "vision_status": "file_missing",
                "in_tokens": 0, "out_tokens": 0,
            })
            continue

        pyt_text = pyt_txt_path.read_text(encoding="utf-8", errors="replace")

        # Idempotency: if vision-LLM transcript already on disk from a prior
        # interrupted run, skip the API call and just compute metrics. Saves
        # cost on restart.
        v_out = CROSS_VAL_DIR / docket / f"{docid}__1_vision.txt"
        if v_out.exists() and v_out.stat().st_size > 0:
            v_text = v_out.read_text(encoding="utf-8", errors="replace")
            in_tok = 0
            out_tok = 0
            cached = True
        else:
            try:
                v_text, in_tok, out_tok = vision_llm_transcribe(pdf_path, client, DEFAULT_MODEL)
            except Exception as e:
                rows.append({
                    "doc_id": docid, "docket_id": docket,
                    "pytesseract_chars": len(pyt_text), "vision_llm_chars": 0,
                    "char_similarity": 0.0,
                    "word_recall": 0.0, "word_precision": 0.0, "word_f1": 0.0,
                    "vision_status": f"fail:{type(e).__name__}",
                    "in_tokens": 0, "out_tokens": 0,
                })
                continue
            v_out.parent.mkdir(parents=True, exist_ok=True)
            v_out.write_text(v_text, encoding="utf-8")
            cached = False

        cs = char_similarity(pyt_text, v_text)
        wm = word_metrics(pyt_text, v_text)
        rows.append({
            "doc_id": docid, "docket_id": docket,
            "pytesseract_chars": len(pyt_text), "vision_llm_chars": len(v_text),
            "char_similarity": cs,
            "word_recall": wm["word_recall"],
            "word_precision": wm["word_precision"],
            "word_f1": wm["word_f1"],
            "vision_status": "ok",
            "in_tokens": in_tok, "out_tokens": out_tok,
        })
        total_in += in_tok
        total_out += out_tok
        cost_so_far = total_in * 3 / 1_000_000 + total_out * 15 / 1_000_000
        tag = "[CACHED]" if cached else ""
        print(f"  [{i:3d}/{len(pytess)}] {time.time()-t_start:.0f}s elapsed; "
              f"F1={wm['word_f1']:.3f}  cost ${cost_so_far:.2f}  {tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CROSS_VAL_CSV, index=False)
    print(f"\n[write] {CROSS_VAL_CSV}")

    write_summary(df, total_in, total_out)
    print(f"[write] {CROSS_VAL_SUMMARY}")

    cost = total_in * 3 / 1_000_000 + total_out * 15 / 1_000_000
    n_ok = (df["vision_status"] == "ok").sum()
    print(f"\n=== Cross-validation complete ===")
    print(f"  files compared: {n_ok}/{len(df)}")
    print(f"  total runtime: {(time.time()-t_start)/60:.1f} min")
    print(f"  estimated cost: ${cost:.2f}")
    print()
    print("Next: open data/processed/ocr_cross_validation_summary.md and review the F1 distribution.")
    print("      Then jointly lock thresholds with Claude before any merge changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
