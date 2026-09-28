"""
smoke_test_attachments.py — URL validation + 5-PDF extraction sanity check.

Stand-alone. Doesn't touch the full sample pipeline (05/06). Goal: in 1-2
minutes, verify end-to-end that:
  - downloads.regulations.gov returns real PDFs when given X-Api-Key header
  - pdfplumber extracts clean text from those real attachments
  - filenames, paths, encodings round-trip cleanly

Picks 5 attachment-only comments at random from anchor_rules_locked.csv's
dockets, downloads each, runs pdfplumber, prints per-file diagnostics.

PREREQUISITES
  - anchor_rules_locked.csv present
  - REGULATIONS_GOV_API_KEY env var set (rotated key)
  - pyarrow + pdfplumber installed in the python that runs this

Usage:
  REGULATIONS_GOV_API_KEY=$KEY python3 code/smoke_test_attachments.py
"""
from __future__ import annotations

import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKED_CSV = PROJECT_ROOT / "data" / "processed" / "anchor_rules_locked.csv"
COMMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "comments"
SMOKE_DIR = PROJECT_ROOT / "data" / "raw" / "attachments_smoke"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
SMOKE_SEED = 20260506
N_SAMPLES = 5


def first_pdf_url(attachment_files_field: str) -> str | None:
    if not attachment_files_field or pd.isna(attachment_files_field):
        return None
    urls = [u.strip() for u in str(attachment_files_field).split(",") if u.strip()]
    pdf_urls = [u for u in urls if u.lower().endswith(".pdf")]
    if pdf_urls:
        return pdf_urls[0]
    return urls[0] if urls else None


def download(url: str, dest: Path, api_key: str, retries: int = 3) -> tuple[bool, str]:
    headers = {"User-Agent": USER_AGENT, "X-Api-Key": api_key}
    delay = 2.0
    last_err = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return True, f"ok {len(data)} bytes"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 429:
                ra = int(e.headers.get("Retry-After", str(int(delay))))
                time.sleep(ra)
                delay *= 2
                continue
            if e.code == 403 and attempt == 0:
                # Try query-string fallback once.
                sep = "&" if "?" in url else "?"
                fallback = f"{url}{sep}api_key={api_key}"
                try:
                    req2 = urllib.request.Request(fallback, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req2, timeout=60) as resp:
                        data = resp.read()
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                        return True, f"ok-querystring {len(data)} bytes"
                except Exception as e2:
                    last_err = f"HTTP 403; query-string fallback also failed: {e2}"
            return False, last_err
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
            time.sleep(delay)
            delay *= 2
    return False, last_err or "max retries exceeded"


def detect_kind(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "docx-or-zip"
    if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or head.startswith(b"<HTML"):
        return "html"
    if head[:5].decode("ascii", errors="replace").isprintable():
        # Looks like text — peek further
        try:
            sample = path.read_text(encoding="utf-8", errors="replace")[:200]
            if "<Error>" in sample or "AccessDenied" in sample:
                return "s3-error-xml"
        except Exception:
            pass
    return "unknown"


def extract_pdf(path: Path) -> tuple[str, dict]:
    import pdfplumber
    parts: list[str] = []
    n_pages, n_text = 0, 0
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                n_text += 1
            parts.append(t)
    return "\n\n".join(parts), {"n_pages": n_pages, "n_pages_with_text": n_text}


def main() -> int:
    api_key = os.environ.get("REGULATIONS_GOV_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set REGULATIONS_GOV_API_KEY env var (rotated key required).", file=sys.stderr)
        return 2
    if not LOCKED_CSV.exists():
        print(f"ERROR: locked anchor list not found at {LOCKED_CSV}", file=sys.stderr)
        return 2

    locked = pd.read_csv(LOCKED_CSV)
    docket_ids = set(locked["docket_id"])
    print(f"[load] {len(docket_ids)} locked anchor dockets")

    # Load comments parquet, filter to attachment-only with non-empty attachment_files
    print("[load] comments parquet for locked dockets ...")
    paths = sorted(COMMENTS_DIR.glob("EPA_*.parquet"))
    parts: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p, columns=["document_id", "docket_id", "title",
                                         "is_attachment_only", "attachment_files",
                                         "comment"])
        df = df[df["docket_id"].isin(docket_ids)]
        df = df[df["is_attachment_only"] & df["attachment_files"].fillna("").str.strip().ne("")]
        if not df.empty:
            parts.append(df)
    if not parts:
        print("ERROR: no candidate attachments found in locked anchors", file=sys.stderr)
        return 2
    pool = pd.concat(parts, ignore_index=True)
    print(f"  pool: {len(pool):,} attachment-only comments across {pool['docket_id'].nunique()} dockets")

    rng = np.random.default_rng(SMOKE_SEED)
    idx = rng.permutation(len(pool))[:N_SAMPLES]
    sample = pool.iloc[idx].copy()
    sample["url"] = sample["attachment_files"].apply(first_pdf_url)
    sample = sample.reset_index(drop=True)

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Smoke test on {N_SAMPLES} attachments (seed={SMOKE_SEED}) ===\n")
    rows = []
    for i, r in sample.iterrows():
        url = r["url"]
        if not url:
            print(f"[{i+1}/{N_SAMPLES}] {r['document_id']}: no URL")
            continue
        # Build a clean filename: docket/document_id.pdf
        ext = Path(re.sub(r"\?.*$", "", url)).suffix or ".pdf"
        dest = SMOKE_DIR / r["docket_id"] / f"{r['document_id']}{ext}"
        print(f"[{i+1}/{N_SAMPLES}] docket={r['docket_id']}  doc_id={r['document_id']}")
        print(f"    title: {str(r['title'])[:90]}")
        print(f"    url:   {url}")
        ok, msg = download(url, dest, api_key)
        print(f"    download: {msg}")
        if not ok:
            rows.append({"document_id": r["document_id"], "url": url, "status": f"download_fail: {msg}"})
            continue
        kind = detect_kind(dest)
        print(f"    kind: {kind}  size: {dest.stat().st_size:,} bytes")
        if kind != "pdf":
            rows.append({"document_id": r["document_id"], "url": url, "status": f"non_pdf: {kind}"})
            continue
        try:
            text, info = extract_pdf(dest)
        except Exception as e:
            rows.append({"document_id": r["document_id"], "url": url,
                         "status": f"extract_fail: {type(e).__name__}: {e}"})
            print(f"    extract: FAIL ({e})")
            continue
        print(f"    extract: ok  pages={info['n_pages']}  with_text={info['n_pages_with_text']}  "
              f"chars={len(text):,}")
        first_line = re.sub(r"\s+", " ", text[:300]).strip()
        print(f"    head: {first_line!r}")
        rows.append({
            "document_id": r["document_id"], "url": url, "status": "ok",
            "n_pages": info["n_pages"], "n_chars": len(text),
            "first_300": first_line,
        })
        print()
        time.sleep(1.0)

    # Summary
    ok_count = sum(1 for r in rows if r["status"] == "ok")
    print(f"=== Summary: {ok_count}/{len(rows)} successful end-to-end ===")
    if ok_count < N_SAMPLES:
        print("Failures:")
        for r in rows:
            if r["status"] != "ok":
                print(f"  {r['document_id']}: {r['status']}")
    return 0 if ok_count >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
