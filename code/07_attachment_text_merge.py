"""
07_attachment_text_merge.py
Step 5 of Option 2 — merge extracted attachment text back into the comments
parquet so Stage 1 (LLM-as-judge) sees real content for the sampled comments.

For each row in attachment_extraction_log.csv with status=='ok' and a
non-empty extracted .txt:
  - If extracted text length >= MIN_CHARS, replace `comment` field with the
    extracted text in a new "augmented" comments parquet.
  - Set is_attachment_only = False (we now have content).
  - Set text_source per the canonical taxonomy (Bruce 2026-05-07):

        'inform'                   -> original CSV comment text was real content
        'attachment_pdfplumber'    -> pdfplumber (PDF) or python-docx (DOCX) native
                                      extraction succeeded
        'attachment_pytesseract'   -> pytesseract OCR succeeded (08_ocr_pytesseract.py)
        'attachment_vision_llm'    -> vision-LLM transcription succeeded (09_ocr_vision_llm.py)
        'attachment_failed'        -> attachment-only comment, all extraction
                                      methods failed (or attachment URL was missing)

This taxonomy lets Stage-1 LLM-as-judge filter or weight comments by source,
lets the methodology section report per-source counts, and enables a robustness
falsification check: do Stage-1 features differ systematically by extraction
method? (If yes, the LLM judge is conflating extraction artifacts with content.)

Output:
  data/processed/comments_augmented/EPA_{YEAR}.parquet
  (each per-year file mirrors data/processed/comments/EPA_{YEAR}.parquet but
   with the merged text + text_source column)

Usage:
  python code/07_attachment_text_merge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "comments"
AUG_DIR = PROJECT_ROOT / "data" / "processed" / "comments_augmented"
LOG_CSV = PROJECT_ROOT / "data" / "processed" / "attachment_extraction_log.csv"
ATTACH_DIR = PROJECT_ROOT / "data" / "raw" / "attachments"

MIN_CHARS = 100  # below this we treat as failed extraction


def _load_one(path: Path) -> pd.DataFrame:
    pq = path.with_suffix(".parquet")
    pk = path.with_suffix(".pkl")
    if pq.exists():
        return pd.read_parquet(pq)
    if pk.exists():
        return pd.read_pickle(pk)
    raise FileNotFoundError(path)


def _write_one(df: pd.DataFrame, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(target, index=False)
        return target
    except ImportError:
        pkl = target.with_suffix(".pkl")
        df.to_pickle(pkl)
        return pkl


def _kind_to_text_source(kind) -> str:
    """Map extraction-log `kind` field to canonical text_source taxonomy."""
    k = (str(kind) if kind is not None else "").lower().strip()
    if k in ("pdf", "docx", "doc", "html", "txt"):
        return "attachment_pdfplumber"   # native (non-OCR) extraction
    if k == "pytesseract":
        return "attachment_pytesseract"
    if k in ("vision_llm", "vision-llm", "claude_vision"):
        return "attachment_vision_llm"
    return "attachment_pdfplumber"


def main() -> int:
    if not LOG_CSV.exists():
        print(f"ERROR: extraction log not found at {LOG_CSV}", file=sys.stderr)
        return 2
    log = pd.read_csv(LOG_CSV)
    ok_log = log[log["status"] == "ok"].copy()
    print(f"[log] {len(ok_log):,}/{len(log):,} successful extractions")
    if "kind" in ok_log.columns:
        kind_counts = ok_log["kind"].fillna("(missing)").value_counts().to_dict()
        print(f"  kind breakdown: {kind_counts}")

    sample = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "attachment_sample.csv",
                         usecols=["docket_id", "document_id"])
    ok_log = ok_log.merge(sample, on="document_id", how="left")
    print(f"  joined to sample CSV: {len(ok_log):,}")

    year_files = sorted(COMMENTS_DIR.glob("EPA_*.parquet")) + \
                 sorted(COMMENTS_DIR.glob("EPA_*.pkl"))
    seen_years: set[str] = set()
    n_replaced_total = 0
    counts_by_source: dict[str, int] = {}
    for path in year_files:
        if path.stem in seen_years:
            continue
        seen_years.add(path.stem)
        year = int(path.stem.split("_")[1])
        c = _load_one(path).copy()
        # Default: attachment-only with no successful extraction → 'attachment_failed';
        # in-form (text content from CSV) → 'inform'.
        c["text_source"] = c["is_attachment_only"].map({
            True: "attachment_failed",
            False: "inform",
        })

        sub = ok_log[ok_log["document_id"].isin(c["document_id"])]
        n_replaced_year = 0
        for _, row in sub.iterrows():
            url = str(row.get("url", "") or "")
            ext = Path(url).suffix.lower().lstrip(".") or "pdf"
            txt_path = ATTACH_DIR / str(row["docket_id"]) / f"{row['document_id']}__1.{ext}"
            txt_path = txt_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            content = txt_path.read_text(encoding="utf-8", errors="replace")
            if len(content) < MIN_CHARS:
                continue
            mask = c["document_id"] == row["document_id"]
            text_source_val = _kind_to_text_source(row.get("kind"))
            c.loc[mask, "comment"] = content
            c.loc[mask, "is_attachment_only"] = False
            c.loc[mask, "text_source"] = text_source_val
            n_replaced_year += int(mask.sum())

        out = AUG_DIR / f"EPA_{year}.parquet"
        written = _write_one(c, out)
        for src, n in c["text_source"].value_counts().items():
            counts_by_source[src] = counts_by_source.get(src, 0) + int(n)
        print(f"  EPA_{year}: replaced {n_replaced_year:,} rows -> {written.name}")
        n_replaced_total += n_replaced_year

    print(f"\n[done] total comments augmented with attachment text: {n_replaced_total:,}")
    print("text_source distribution across all years:")
    for src, n in sorted(counts_by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<28} {n:>12,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
