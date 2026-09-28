"""
copy_ocr_validation_pdfs.py — One-off helper.

Copies the 148 OCR-candidate PDFs from data/raw/attachments/ (which is
gitignored for binary files) into data/processed/ocr_validation_pdfs/
(which IS tracked in git), so reviewers / Yue / future-self can do
byte-level audit of the pytesseract + vision-LLM extractions against the
original PDFs.

Why these specifically (not all 1,769):
  - The 148 OCR-candidate set was processed via non-deterministic methods
    (pytesseract OCR + Claude vision API). They need ground-truth check
    that pdfplumber-extracted PDFs do not.
  - 148 PDFs at typical ~500 KB each ≈ 75 MB — manageable repo addition.

Source of truth for the candidate list:
  data/processed/ocr_stage1_log.csv  (all 148 rows have method=pytesseract).

Usage:
  python3 code/copy_ocr_validation_pdfs.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "data" / "raw" / "attachments"
DST = PROJECT_ROOT / "data" / "processed" / "ocr_validation_pdfs"
STAGE1_LOG = PROJECT_ROOT / "data" / "processed" / "ocr_stage1_log.csv"


def main() -> int:
    if not STAGE1_LOG.exists():
        print(f"ERROR: {STAGE1_LOG} not found — run 08_ocr_pytesseract.py first.", file=sys.stderr)
        return 2

    log = pd.read_csv(STAGE1_LOG)
    print(f"[stage1 log] {len(log)} OCR-candidate rows")

    copied = 0
    missing = 0
    for _, r in log.iterrows():
        docid = r["document_id"]
        docket = r["docket_id"]
        if pd.isna(docket) or not docket:
            continue
        src = SRC / docket / f"{docid}__1.pdf"
        if not src.exists():
            # Some non_pdf candidates (.docx, .jpg) won't have a .pdf — skip silently
            missing += 1
            continue
        dst = DST / docket / f"{docid}__1.pdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    total_size = sum(p.stat().st_size for p in DST.rglob("*.pdf"))
    print(f"[done] copied {copied} PDFs into {DST.relative_to(PROJECT_ROOT)}")
    print(f"       missing/skipped (no .pdf on disk): {missing}")
    print(f"       total size: {total_size / 1024 / 1024:.1f} MB")
    print()
    print("Add a README to the new folder explaining its purpose, then commit:")
    print(f"  git add {DST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
