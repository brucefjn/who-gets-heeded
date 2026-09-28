"""
08_ocr_pytesseract.py — Stage 1 OCR recovery via pytesseract.

PURPOSE
  After 06_download_extract_attachments.py and the diagnostic in
  data/processed/attachment_failure_diagnostic.md, recover the 148
  OCR-candidate attachments — pdfplumber failures + ok-but-low-yield
  extractions, biased toward pre-2010 dockets and organizational submitters.

CANDIDATE DEFINITION
  attachment_extraction_log.csv rows where:
    - status == 'extract_fail:'  (pdfplumber gave up, usually image-only PDF), OR
    - status == 'ok' AND n_chars < 100  (very-low-yield extraction)

PIPELINE
  For each candidate:
    1. Render PDF pages to PIL images via pypdfium2 (no poppler dep needed).
    2. Run pytesseract on each page image with English language model.
    3. Concatenate page texts → write to <docket>/<doc_id>__1.txt
       (overwrites prior low-yield extraction).
    4. Apply quality gate:
        n_chars < 500              → low_quality:short
        non_ascii_ratio > 0.20     → low_quality:non_ascii
        else                       → ok
    5. Record per-row: method=pytesseract, status, n_chars, n_pages,
       quality_flag, runtime_sec.

OUTPUTS
  data/processed/ocr_stage1_log.csv
      per-attachment Stage-1 detail; consumed by 09_ocr_vision_llm.py.
  data/processed/attachment_extraction_log.csv
      master log updated for ok rows: kind='pytesseract', n_chars updated.
  data/raw/attachments/<docket>/<doc_id>__1.txt
      replaces low-yield extraction with OCR text.

DEPENDENCIES (one-time)
    brew install tesseract
    pip3 install --user pytesseract pypdfium2 pillow

CITATIONS (rigor: published-and-cited)
  Smith, R. (2007). An overview of the Tesseract OCR engine. ICDAR 2007.
  Patel, C., Patel, A., & Patel, D. (2012). Optical Character Recognition by
    Open Source OCR Tool Tesseract: A Case Study. IJCA, 55(10).

USAGE
  python3 code/08_ocr_pytesseract.py
"""
from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACH = PROJECT_ROOT / "data" / "raw" / "attachments"
LOG_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_extraction_log.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_sample.csv"
OCR_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "ocr_stage1_log.csv"

# Quality-gate thresholds — flag for Stage-2 vision-LLM if either fires
MIN_CHARS = 500
MAX_NON_ASCII_RATIO = 0.20

# Tesseract config — automatic page segmentation with orientation/script detection
TESSERACT_CONFIG = "--psm 1"
OCR_DPI = 200  # PDF→image render resolution; 200 is the sweet spot for text-heavy scans

SPOT_CHECK_SEED = 20260507
SPOT_CHECK_N = 10


def is_ocr_candidate(row) -> bool:
    """Bruce's definition of OCR candidate (per attachment_failure_diagnostic.md)."""
    s = str(row.get("status", "")).strip()
    if s in ("extract_fail:", "extract_fail::"):
        return True
    if s == "ok":
        try:
            return float(row.get("n_chars", 0)) < 100
        except (TypeError, ValueError):
            return False
    return False


def render_pdf_pages(pdf_path: Path, dpi: int = OCR_DPI) -> list:
    """Render PDF → PIL.Image list via pypdfium2 (no poppler / external binary)."""
    import pypdfium2
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    images = []
    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            try:
                bitmap = page.render(scale=dpi / 72)
                images.append(bitmap.to_pil())
            finally:
                page.close()
    finally:
        pdf.close()
    return images


def ocr_pdf(pdf_path: Path) -> tuple[str, dict]:
    """Run pytesseract on each page of a PDF; concatenate."""
    import pytesseract
    pages = render_pdf_pages(pdf_path)
    parts: list[str] = []
    for img in pages:
        t = pytesseract.image_to_string(img, lang="eng", config=TESSERACT_CONFIG) or ""
        parts.append(t)
    text = "\n\n".join(parts)
    return text, {
        "n_pages": len(pages),
        "n_pages_with_text": sum(1 for t in parts if t.strip()),
    }


def quality_flag(text: str) -> str:
    if len(text) < MIN_CHARS:
        return f"low_quality:short ({len(text)} chars)"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(1, len(text))
    if ratio > MAX_NON_ASCII_RATIO:
        return f"low_quality:non_ascii ({ratio*100:.1f}%)"
    return "ok"


def main() -> int:
    # Pre-flight
    try:
        import pytesseract  # noqa: F401
        import pypdfium2  # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dep {e.name}. Run: pip3 install --user pytesseract pypdfium2", file=sys.stderr)
        return 2
    try:
        import pytesseract
        v = pytesseract.get_tesseract_version()
        print(f"[preflight] tesseract {v}")
    except Exception as e:
        print(f"ERROR: tesseract binary not found ({e}). Run: brew install tesseract", file=sys.stderr)
        return 2

    log = pd.read_csv(LOG_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    docket_for = dict(zip(sample.document_id, sample.docket_id))
    url_for = dict(zip(sample.document_id, sample.attachment_files))

    candidates = log[log.apply(is_ocr_candidate, axis=1)].copy()
    print(f"[stage 1] {len(candidates)} OCR candidates")

    rows: list[dict] = []
    t_start = time.time()
    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        docid = row["document_id"]
        docket = docket_for.get(docid)
        if not docket:
            rows.append({"document_id": docid, "docket_id": "",
                         "method": "pytesseract", "status": "no_docket",
                         "n_chars": 0, "n_pages": 0,
                         "quality_flag": "n/a", "runtime_sec": 0})
            continue
        url = str(url_for.get(docid, "")).split(",")[0].lower()
        ext_match = re.search(r"\.([a-z0-9]+)(?:$|\?)", url)
        ext = ext_match.group(1) if ext_match else "pdf"
        if ext != "pdf":
            rows.append({"document_id": docid, "docket_id": docket,
                         "method": "pytesseract", "status": f"non_pdf:{ext}",
                         "n_chars": 0, "n_pages": 0,
                         "quality_flag": "n/a", "runtime_sec": 0})
            continue
        fp = ATTACH / docket / f"{docid}__1.{ext}"
        if not fp.exists():
            rows.append({"document_id": docid, "docket_id": docket,
                         "method": "pytesseract", "status": "file_missing",
                         "n_chars": 0, "n_pages": 0,
                         "quality_flag": "n/a", "runtime_sec": 0})
            continue

        t0 = time.time()
        try:
            text, info = ocr_pdf(fp)
        except Exception as e:
            elapsed = time.time() - t0
            rows.append({"document_id": docid, "docket_id": docket,
                         "method": "pytesseract", "status": f"ocr_fail:{type(e).__name__}",
                         "n_chars": 0, "n_pages": 0,
                         "quality_flag": "n/a", "runtime_sec": round(elapsed, 1)})
            print(f"  [{i:3d}/{len(candidates)}] {docid}: FAIL {type(e).__name__}")
            continue
        elapsed = time.time() - t0
        flag = quality_flag(text)

        txt_path = fp.with_suffix(".txt")
        txt_path.write_text(text, encoding="utf-8")
        rows.append({
            "document_id": docid, "docket_id": docket,
            "method": "pytesseract", "status": "ok",
            "n_chars": len(text), "n_pages": info["n_pages"],
            "n_pages_with_text": info["n_pages_with_text"],
            "quality_flag": flag, "runtime_sec": round(elapsed, 1),
        })
        if i % 10 == 0:
            print(f"  [{i:3d}/{len(candidates)}] {time.time()-t_start:.0f}s elapsed; "
                  f"last={elapsed:.1f}s  {flag}")

    ocr_df = pd.DataFrame(rows)
    OCR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ocr_df.to_csv(OCR_LOG_PATH, index=False)

    # Update master log for successful OCR rows.
    ok = ocr_df[ocr_df["status"] == "ok"]
    for _, r in ok.iterrows():
        mask = log["document_id"] == r["document_id"]
        log.loc[mask, "status"] = "ok"
        log.loc[mask, "kind"] = "pytesseract"
        log.loc[mask, "n_chars"] = int(r["n_chars"])
        log.loc[mask, "info"] = (
            f"method=pytesseract n_pages={int(r['n_pages'])} "
            f"quality={r['quality_flag']}"
        )
    log.to_csv(LOG_PATH, index=False)

    # Summary
    n_total = len(ocr_df)
    n_ok = (ocr_df["status"] == "ok").sum()
    n_high = ((ocr_df["status"] == "ok") & (ocr_df["quality_flag"] == "ok")).sum()
    n_low = ((ocr_df["status"] == "ok") & (ocr_df["quality_flag"].str.startswith("low_quality"))).sum()
    n_fail = n_total - n_ok
    elapsed_total = (time.time() - t_start) / 60
    print(f"\n=== Stage 1 (pytesseract) summary ===")
    print(f"  total candidates      : {n_total}")
    print(f"  succeeded             : {n_ok}")
    print(f"    high quality        : {n_high}")
    print(f"    low quality (→ S2)  : {n_low}")
    print(f"  failed (→ S2 retry)   : {n_fail}")
    print(f"  total runtime         : {elapsed_total:.1f} min")

    # Spot-check: 10 random high-quality results for manual eyeballing
    rng = random.Random(SPOT_CHECK_SEED)
    high_ok_ids = ok.loc[ok["quality_flag"] == "ok", "document_id"].tolist()
    spot_sample = rng.sample(high_ok_ids, k=min(SPOT_CHECK_N, len(high_ok_ids)))
    print(f"\n=== Spot-check sample ({len(spot_sample)} random high-quality files) ===")
    print("Eyeball each .txt against the original PDF before approving Stage 2.")
    print("Open commands:")
    for docid in spot_sample:
        d = ok.loc[ok["document_id"] == docid, "docket_id"].iloc[0]
        pdf = ATTACH / d / f"{docid}__1.pdf"
        txt = ATTACH / d / f"{docid}__1.txt"
        print(f"  open '{pdf}' && open '{txt}'")
    print()
    print("After approving Stage 1 quality, run:")
    print("  export ANTHROPIC_API_KEY='...'")
    print("  python3 code/09_ocr_vision_llm.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
