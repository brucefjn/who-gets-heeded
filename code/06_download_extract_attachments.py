"""
06_download_extract_attachments.py
Step 4 of Option 2 — download attachment files and extract text.

Reads attachment_sample.csv (from 05), downloads each attachment, detects
file type, and extracts text:
  - PDF      -> pdfplumber
  - DOCX     -> python-docx
  - DOC      -> textract (fallback) or skip with warning
  - TXT      -> read directly
  - HTML     -> beautifulsoup
  - other    -> skip with warning

Per-file output:
  data/raw/attachments/{docket_id}/{document_id}__{idx}.{ext}     (raw download)
  data/raw/attachments/{docket_id}/{document_id}__{idx}.txt        (extracted)
  data/processed/attachment_extraction_log.csv                     (log row)

Network behavior:
  - REQUIRES api key — set via REGULATIONS_GOV_API_KEY env var. Sent as the
    X-Api-Key header on every download request. Hitting the downloads URL
    without a key returns AWS S3 AccessDenied (verified 2026-05-06).
  - exponential backoff on 429 (Retry-After honored)
  - browser-like User-Agent
  - skips files already downloaded successfully (idempotent)

Usage:
  REGULATIONS_GOV_API_KEY=$KEY python code/06_download_extract_attachments.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "attachment_sample.csv"
ATTACH_DIR = PROJECT_ROOT / "data" / "raw" / "attachments"
LOG_CSV = PROJECT_ROOT / "data" / "processed" / "attachment_extraction_log.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
# 1,000 calls/hour budget = 3.6s/call. Plus retry headroom.
THROTTLE_SEC = 4.0


def download(url: str, dest: Path, api_key: str, max_retries: int = 4) -> tuple[bool, str]:
    """Returns (ok, message). Idempotent: skip if dest exists and >0 bytes.

    Sends X-Api-Key header — required by downloads.regulations.gov; raw
    fetches without a key get S3 AccessDenied.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"skip (already {dest.stat().st_size} bytes)"
    headers = {"User-Agent": USER_AGENT, "X-Api-Key": api_key}
    delay = 2.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return True, f"ok {len(data)} bytes"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = int(e.headers.get("Retry-After", str(int(delay))))
                time.sleep(ra)
                delay *= 2
                continue
            if e.code == 403:
                # Likely AccessDenied -> fall back to query-string variant once.
                if attempt == 0 and "api_key=" not in url:
                    sep = "&" if "?" in url else "?"
                    fallback_url = f"{url}{sep}api_key={api_key}"
                    req = urllib.request.Request(fallback_url, headers={"User-Agent": USER_AGENT})
                    try:
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            data = resp.read()
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(data)
                            return True, f"ok-querystring {len(data)} bytes"
                    except Exception:
                        pass
            return False, f"HTTP {e.code}"
        except urllib.error.URLError as e:
            time.sleep(delay)
            delay *= 2
    return False, "max retries exceeded"


def detect_kind(path: Path) -> str:
    """Cheap content-sniff. PDFs start with '%PDF-'; docx are zip; etc."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "missing"
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        # zip-based; could be docx/xlsx/pptx
        suf = path.suffix.lower().lstrip(".")
        return suf or "zip"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "doc"  # legacy MS Office
    if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or head.startswith(b"<HTML"):
        return "html"
    if head[:3] == b"\xef\xbb\xbf" or head.isascii():
        return "txt"
    return "unknown"


def extract_pdf(path: Path) -> tuple[str, dict]:
    import pdfplumber
    text_parts: list[str] = []
    n_pages = 0
    n_pages_with_text = 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                n_pages_with_text += 1
            text_parts.append(t)
    return "\n\n".join(text_parts), {"n_pages": n_pages, "n_pages_with_text": n_pages_with_text}


def extract_docx(path: Path) -> tuple[str, dict]:
    import docx  # python-docx
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs]
    return "\n".join(paras), {"n_paragraphs": len(paras)}


def extract_text_file(path: Path) -> tuple[str, dict]:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(enc), {"encoding": enc}
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), {"encoding": "latin-1-replace"}


def extract_html(path: Path) -> tuple[str, dict]:
    from bs4 import BeautifulSoup
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text("\n", strip=True), {}


def extract(path: Path) -> tuple[str, dict, str]:
    """Returns (text, info_dict, kind_used)."""
    kind = detect_kind(path)
    try:
        if kind == "pdf":
            t, info = extract_pdf(path); return t, info, "pdf"
        if kind == "docx":
            t, info = extract_docx(path); return t, info, "docx"
        if kind == "txt":
            t, info = extract_text_file(path); return t, info, "txt"
        if kind == "html":
            t, info = extract_html(path); return t, info, "html"
        # legacy doc / xlsx / images / unknown -> skip extraction
        return "", {"reason": f"unsupported kind={kind}"}, kind
    except Exception as e:
        return "", {"reason": f"{type(e).__name__}: {e}"}, kind


def filename_from_url(url: str) -> str:
    return Path(re.sub(r"\?.*$", "", url)).name


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="cap rows for a smoke run")
    args = p.parse_args()

    api_key = os.environ.get("REGULATIONS_GOV_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set REGULATIONS_GOV_API_KEY env var (rotated key required).",
              file=sys.stderr)
        return 2

    if not SAMPLE_CSV.exists():
        print(f"ERROR: sample CSV not found at {SAMPLE_CSV}", file=sys.stderr)
        return 2

    sample = pd.read_csv(SAMPLE_CSV)
    if args.limit:
        sample = sample.head(args.limit)

    log_rows: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(sample.itertuples(index=False), 1):
        # Guard against NaN / empty attachment_files (legitimate edge case:
        # ~0.5% of attachment-only comments lack an attachment URL even though
        # the comment text says 'See attached'. They leak through the sampler.)
        raw = row.attachment_files
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            log_rows.append({"document_id": row.document_id, "url": "",
                             "status": "no_url_nan", "kind": "", "n_chars": 0,
                             "info": "attachment_files was NaN"})
            continue
        s = str(raw).strip()
        if not s or s.lower() == "nan":
            log_rows.append({"document_id": row.document_id, "url": "",
                             "status": "no_url_blank", "kind": "", "n_chars": 0,
                             "info": ""})
            continue
        urls = [u.strip() for u in s.split(",") if u.strip() and u.strip().lower() != "nan"]
        urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]
        if not urls:
            log_rows.append({"document_id": row.document_id, "url": s[:200],
                             "status": "no_valid_url", "kind": "", "n_chars": 0,
                             "info": ""})
            continue
        # Prefer PDF over .docx if both present (Bruce: most are PDF)
        pdf_urls = [u for u in urls if u.lower().endswith(".pdf")]
        chosen = pdf_urls[0] if pdf_urls else urls[0]
        ext = Path(chosen).suffix.lower().lstrip(".") or "bin"
        dest = ATTACH_DIR / row.docket_id / f"{row.document_id}__1.{ext}"
        ok, msg = download(chosen, dest, api_key)
        if not ok:
            log_rows.append({"document_id": row.document_id, "url": chosen,
                             "status": f"download_fail:{msg}", "kind": "",
                             "n_chars": 0, "info": ""})
            time.sleep(THROTTLE_SEC)
            continue
        text, info, kind = extract(dest)
        if text:
            txt_path = dest.with_suffix(".txt")
            txt_path.write_text(text, encoding="utf-8")
        log_rows.append({
            "document_id": row.document_id,
            "url": chosen,
            "status": "ok" if text else f"extract_fail:{info.get('reason','')}",
            "kind": kind,
            "n_chars": len(text),
            "info": str(info),
        })
        if i % 25 == 0:
            print(f"  [{i}/{len(sample)}] {time.time()-t0:.0f}s elapsed")
        time.sleep(THROTTLE_SEC)

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False)
    print(f"\n[write] {LOG_CSV}")
    print(f"  ok rows: {sum(1 for r in log_rows if r['status']=='ok'):,}")
    print(f"  total:   {len(log_rows):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
