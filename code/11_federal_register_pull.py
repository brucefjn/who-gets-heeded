"""
code/11_federal_register_pull.py

Workstream 1 — Federal Register integration.

Pulls proposed-rule and final-rule documents for the 36 anchor EPA dockets
in data/processed/anchor_rules_locked.csv, using the Federal Register API
(federalregister.gov/api/v1). Saves raw text + JSON metadata for each
matched document; writes a per-document index to data/processed/.

Decisions encoded:
  - Pull both raw_text and full JSON metadata. raw_text is plain text suitable
    for diffing; JSON metadata exposes section structure, abstract, RIN, CFR
    references, agencies, etc.
  - Section-decomposition strategy (proposed by Yue 2026-05-07, pending Bruce
    sign-off): use the JSON `topics` and `subheaders` fields plus markdown-
    style heading detection on the raw_text to derive section boundaries.
    Implemented as a separate module after WS1 confirms the API works.
  - Filter to document type ∈ {"Rule", "Proposed Rule"}. Withdrawals, Notices,
    and Corrections are not part of the responsiveness regression but are
    logged in the index for completeness.
  - Fail-soft: a docket missing proposed or final rules is logged with
    rule_type="missing" rather than aborting the run.

Inputs:
  data/processed/anchor_rules_locked.csv  -- 36 anchor docket IDs

Outputs:
  data/raw/federal_register/{docket_id}_{rule_type}_{fr_doc_number}.txt
  data/raw/federal_register/{docket_id}_{rule_type}_{fr_doc_number}.json
  data/processed/federal_register_index.csv

Usage:
  # Smoke test on one docket with verbose output (recommended first run):
  python3 code/11_federal_register_pull.py --smoke-test

  # Pull a specific docket:
  python3 code/11_federal_register_pull.py --docket EPA-HQ-OAR-2009-0234

  # Full pull across all 36 anchors:
  python3 code/11_federal_register_pull.py

API reference:
  https://www.federalregister.gov/developers/api/v1
  Documents endpoint: GET /api/v1/documents.json
  Filter by docket:   conditions[docket_id]=EPA-HQ-OAR-2009-0234
  No auth required; gentle throttle (~0.5s between calls) to be a good citizen.

Bruce 2026-05-07: critical-path workstream. Start with smoke test before
building out. Coverage acceptance: ≥30/36 dockets recovered with both
proposed and final text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CSV = REPO_ROOT / "data" / "processed" / "anchor_rules_locked.csv"
RAW_DIR = REPO_ROOT / "data" / "raw" / "federal_register"
INDEX_CSV = REPO_ROOT / "data" / "processed" / "federal_register_index.csv"

FR_API = "https://www.federalregister.gov/api/v1/documents.json"

# Per the Federal Register API guidelines, identify ourselves with a UA + contact
USER_AGENT = "regulatory-comments-project/1.0 (academic research; yy3462@columbia.edu)"

# Field set to request from the API. Fewer fields = smaller payloads.
# Names verified against https://www.federalregister.gov/developers/api/v1
DOC_FIELDS = [
    "document_number",
    "title",
    "type",
    "subtype",
    "publication_date",
    "abstract",
    "body_html_url",
    "json_url",
    "raw_text_url",
    "html_url",
    "pdf_url",
    "regulation_id_numbers",
    "agencies",
    "docket_ids",
    "topics",
    "cfr_references",
]

THROTTLE_SECONDS = 0.5


def http_get_json(url: str, *, retries: int = 3) -> dict:
    """GET a URL, return parsed JSON. Retries with exponential backoff."""
    from urllib.error import HTTPError
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            })
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            # Capture the response body so we can see what the API is complaining about
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = "(could not read response body)"
            last_err = e
            print(f"  HTTP {e.code} on {url[:120]}...", file=sys.stderr)
            print(f"  response body: {body[:500]}", file=sys.stderr)
            if e.code in (400, 404):  # don't retry client errors
                break
            wait = 1.5 * (2 ** attempt)
            print(f"  retry {attempt+1}/{retries} after {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = 1.5 * (2 ** attempt)
            print(f"  retry {attempt+1}/{retries} after {wait:.1f}s: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url} ({last_err})")


def http_get_text(url: str, *, retries: int = 3) -> str | None:
    """GET a URL, return body as text. Returns None on failure."""
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (2 ** attempt))
    print(f"  text fetch failed: {url} ({last_err})", file=sys.stderr)
    return None


def list_documents_for_docket(docket_id: str, *, verbose: bool = False) -> list[dict]:
    """List all FR documents for a docket, paging through results."""
    base = FR_API + "?" + urlencode({
        "conditions[docket_id]": docket_id,
        "per_page": 100,
    }) + "".join(f"&fields[]={f}" for f in DOC_FIELDS)

    docs: list[dict] = []
    url = base
    page = 0
    while url:
        page += 1
        if verbose:
            print(f"  page {page}: {url[:120]}...")
        data = http_get_json(url)
        results = data.get("results", [])
        docs.extend(results)
        if verbose:
            print(f"    got {len(results)} docs (total so far {len(docs)})")
        url = data.get("next_page_url")
        if url:
            time.sleep(THROTTLE_SECONDS)
    return docs


def filter_rules(docs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split documents into (proposed, final, other)."""
    proposed = [d for d in docs if d.get("type") == "Proposed Rule"]
    final = [d for d in docs if d.get("type") == "Rule"]
    other = [d for d in docs if d.get("type") not in ("Proposed Rule", "Rule")]
    return proposed, final, other


def safe_filename(s: str) -> str:
    """Return a filesystem-safe slug for a string."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def save_doc_artifacts(docket_id: str, rule_type: str, doc: dict) -> tuple[Path | None, Path]:
    """
    Save the document's raw text and JSON metadata. Returns (text_path, json_path).
    text_path may be None if the body could not be fetched.
    """
    fr_doc = doc.get("document_number", "unknown")
    stem = f"{docket_id}_{rule_type}_{safe_filename(fr_doc)}"
    text_path = RAW_DIR / f"{stem}.txt"
    json_path = RAW_DIR / f"{stem}.json"

    # Save the JSON metadata always
    json_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Try raw_text first, fall back to stripping HTML
    text = None
    if doc.get("raw_text_url"):
        text = http_get_text(doc["raw_text_url"])
    if text is None and doc.get("body_html_url"):
        html = http_get_text(doc["body_html_url"])
        if html is not None:
            # Crude strip; section-aware extraction happens in WS2
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()

    if text is not None:
        text_path.write_text(text, encoding="utf-8")
        return text_path, json_path
    else:
        return None, json_path


def index_row(docket_id: str, rule_type: str, doc: dict, text_path: Path | None,
              n_chars: int | None) -> dict:
    """Build one row for federal_register_index.csv."""
    return {
        "docket_id": docket_id,
        "rule_type": rule_type,
        "fr_doc_number": doc.get("document_number", ""),
        "publication_date": doc.get("publication_date", ""),
        "title": (doc.get("title", "") or "").strip(),
        "subtype": doc.get("subtype", "") or "",
        "n_chars": n_chars if n_chars is not None else 0,
        "text_path": str(text_path.relative_to(REPO_ROOT)) if text_path else "",
        "raw_text_url": doc.get("raw_text_url", "") or "",
        "abstract": (doc.get("abstract", "") or "").strip()[:500],
    }


def missing_row(docket_id: str, rule_type: str) -> dict:
    return {
        "docket_id": docket_id,
        "rule_type": "missing",
        "fr_doc_number": "",
        "publication_date": "",
        "title": f"NO {rule_type.upper()} FOUND for {docket_id}",
        "subtype": "",
        "n_chars": 0,
        "text_path": "",
        "raw_text_url": "",
        "abstract": "",
    }


def process_docket(docket_id: str, *, verbose: bool = False) -> list[dict]:
    """Fetch + save all FR documents for a docket. Returns index rows."""
    print(f"\n=== {docket_id} ===")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    docs = list_documents_for_docket(docket_id, verbose=verbose)
    print(f"  found {len(docs)} total documents in this docket")

    proposed, final, other = filter_rules(docs)
    print(f"  → {len(proposed)} Proposed Rule(s), {len(final)} Rule(s), {len(other)} other")

    if verbose:
        for d in proposed + final:
            print(f"    [{d.get('type', '?')}] {d.get('document_number', '?')} "
                  f"({d.get('publication_date', '?')}): "
                  f"{(d.get('title') or '')[:80]}")

    rows: list[dict] = []

    if not proposed:
        rows.append(missing_row(docket_id, "Proposed Rule"))
    for d in proposed:
        text_path, _ = save_doc_artifacts(docket_id, "proposed", d)
        n_chars = text_path.stat().st_size if text_path else 0
        rows.append(index_row(docket_id, "proposed", d, text_path, n_chars))
        if verbose:
            print(f"    saved proposed: {d.get('document_number')} → {n_chars:,} chars")
        time.sleep(THROTTLE_SECONDS)

    if not final:
        rows.append(missing_row(docket_id, "Rule"))
    for d in final:
        text_path, _ = save_doc_artifacts(docket_id, "final", d)
        n_chars = text_path.stat().st_size if text_path else 0
        rows.append(index_row(docket_id, "final", d, text_path, n_chars))
        if verbose:
            print(f"    saved final:    {d.get('document_number')} → {n_chars:,} chars")
        time.sleep(THROTTLE_SECONDS)

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--smoke-test", action="store_true",
                    help="run on EPA-HQ-OAR-2009-0234 only with verbose output")
    ap.add_argument("--docket", type=str, default=None, action="append",
                    help="specific docket ID to pull (skip the anchor list); "
                         "repeatable: --docket A --docket B")
    ap.add_argument("--anchors-csv", type=Path, default=ANCHOR_CSV,
                    help="path to anchor_rules_locked.csv")
    args = ap.parse_args()

    if args.smoke_test:
        target_dockets = ["EPA-HQ-OAR-2009-0234"]
        verbose = True
    elif args.docket:
        # action="append" gives a list; preserve order, dedupe.
        target_dockets = list(dict.fromkeys(args.docket))
        verbose = True
    else:
        df = pd.read_csv(args.anchors_csv)
        target_dockets = df["docket_id"].tolist()
        verbose = False

    print(f"Pulling Federal Register data for {len(target_dockets)} docket(s)")
    print(f"Output dir: {RAW_DIR}")

    all_rows: list[dict] = []
    n_success = 0
    n_failed = 0
    for i, docket_id in enumerate(target_dockets, 1):
        try:
            rows = process_docket(docket_id, verbose=verbose)
            all_rows.extend(rows)
            n_success += 1
        except Exception as e:
            print(f"  ERROR processing {docket_id}: {e}", file=sys.stderr)
            all_rows.append(missing_row(docket_id, "Proposed Rule"))
            all_rows.append(missing_row(docket_id, "Rule"))
            n_failed += 1
        if i < len(target_dockets):
            time.sleep(THROTTLE_SECONDS)

    # Write the index. CRITICAL FIX (Bruce 2026-05-07):
    # Partial-run modes (--smoke-test, --docket) must MERGE into the existing
    # index, not OVERWRITE it. The earlier overwrite-on-partial behavior caused
    # an index-loss event during WS1 recovery — we got it back from disk, but
    # don't bet on disk recovery a second time.
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(all_rows)
    is_partial = bool(args.smoke_test or args.docket)
    if is_partial and INDEX_CSV.exists():
        existing = pd.read_csv(INDEX_CSV)
        target_set = set(target_dockets)
        # Drop the existing index's rows for dockets we just re-pulled
        existing = existing[~existing["docket_id"].isin(target_set)]
        out_df = pd.concat([existing, new_df], ignore_index=True)
        print(f"\nMerged {len(new_df)} new rows for {len(target_set)} docket(s) "
              f"into existing index (kept {len(existing)} rows for other dockets)")
    else:
        out_df = new_df
    out_df.to_csv(INDEX_CSV, index=False)
    print(f"Wrote {INDEX_CSV} ({len(out_df)} rows total)")

    # Coverage report
    print("\n=== coverage ===")
    print(f"Dockets attempted: {len(target_dockets)}")
    print(f"Dockets succeeded: {n_success}")
    print(f"Dockets failed:    {n_failed}")
    if not args.smoke_test and not args.docket:
        # Per-docket coverage check for the full pull
        cov = (out_df.assign(has_text=out_df["n_chars"] > 0)
                     .groupby("docket_id")["rule_type"]
                     .apply(lambda s: {"proposed" in s.values, "final" in s.values}))
        n_complete = sum(1 for v in cov if v == {True})
        print(f"Dockets with both proposed and final text: {n_complete}/{len(target_dockets)}")


if __name__ == "__main__":
    main()
