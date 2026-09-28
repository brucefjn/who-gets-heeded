"""
04_fetch_attachment_metadata.py
Step 2 of Option 2 — OPTIONAL fallback.

In ~99.5% of attachment-only EPA 2018 comments, the `attachment_files` column
already holds direct download URLs. So for our locked anchor rules we can mostly
skip the API metadata endpoint and just download from the URLs directly.

This script is for the residual edge cases: comments where attachment_files is
empty but the comment text says 'See attached'. It hits
    GET https://api.regulations.gov/v4/documents/{COMMENT-DOC-ID}/attachments
to enumerate attachments and resolve their download URLs.

Inputs:
  data/processed/anchor_rules_locked.csv   (Bruce-approved anchor list)
  REGULATIONS_GOV_API_KEY  (env var; rotated key per Bruce)

Outputs:
  data/processed/attachments_metadata.parquet
    one row per attachment: comment_doc_id, attachment_id, file_url,
    file_format, page_count, anchor_docket_id

Rate limit: 1,000 requests/hour. Default: respects retry-after on 429,
exponential backoff up to 5 attempts.

Usage:
  REGULATIONS_GOV_API_KEY=$KEY python code/04_fetch_attachment_metadata.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "comments"
ANCHOR_LOCKED = PROJECT_ROOT / "data" / "processed" / "anchor_rules_locked.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "attachments_metadata.parquet"

API_BASE = "https://api.regulations.gov/v4"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _read_parquet_or_pickle(path: Path) -> pd.DataFrame:
    pq = path.with_suffix(".parquet")
    pk = path.with_suffix(".pkl")
    if pq.exists():
        return pd.read_parquet(pq)
    if pk.exists():
        return pd.read_pickle(pk)
    raise FileNotFoundError(f"Neither parquet nor pickle: {path}")


def fetch_attachments(comment_doc_id: str, api_key: str) -> list[dict]:
    """Hit /comments/{id}?include=attachments and return the 'included' list.

    Verified pattern (2026-05-06): the regulations.gov frontend itself uses
    this endpoint with X-Api-Key header. Each comment's attachments are
    returned in the top-level `included` array, with fileFormats[].fileUrl
    pointing to https://downloads.regulations.gov/{id}/attachment_N.{ext}
    (which then also requires X-Api-Key to download).
    """
    url = f"{API_BASE}/comments/{comment_doc_id}?include=attachments"
    headers = {"User-Agent": USER_AGENT, "X-Api-Key": api_key,
               "Accept": "application/vnd.api+json"}
    delay = 2.0
    for attempt in range(5):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
                return payload.get("included", []) or []
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", str(int(delay))))
                print(f"  [rate-limit] sleeping {retry_after}s (attempt {attempt+1}/5)", file=sys.stderr)
                time.sleep(retry_after)
                delay *= 2
                continue
            if e.code == 404:
                return []
            raise
        except urllib.error.URLError as e:
            print(f"  [network] {e} (attempt {attempt+1}/5)", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    return []


def parse_attachment(rec: dict, comment_doc_id: str, anchor_docket_id: str) -> list[dict]:
    """Each attachment 'data' record may have multiple fileFormats entries
    (e.g. one PDF + one PNG preview). Yield one row per fileFormat."""
    out = []
    aid = rec.get("id")
    attrs = rec.get("attributes", {}) or {}
    formats = attrs.get("fileFormats", []) or []
    for f in formats:
        out.append({
            "comment_doc_id": comment_doc_id,
            "attachment_id": aid,
            "anchor_docket_id": anchor_docket_id,
            "file_url": f.get("fileUrl"),
            "file_format": f.get("format"),
            "page_count": attrs.get("pageCount"),
            "title": attrs.get("title") or "",
        })
    if not formats:
        out.append({
            "comment_doc_id": comment_doc_id,
            "attachment_id": aid,
            "anchor_docket_id": anchor_docket_id,
            "file_url": None,
            "file_format": None,
            "page_count": attrs.get("pageCount"),
            "title": attrs.get("title") or "",
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="cap the number of API calls (0 = no cap)")
    args = p.parse_args()

    api_key = os.environ.get("REGULATIONS_GOV_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set REGULATIONS_GOV_API_KEY env var (rotate your key first!)",
              file=sys.stderr)
        return 2
    if not ANCHOR_LOCKED.exists():
        print(f"ERROR: anchor list not found at {ANCHOR_LOCKED}", file=sys.stderr)
        print("       Run 03_anchor_rule_selection.py and have Bruce approve.",
              file=sys.stderr)
        return 2

    anchors = pd.read_csv(ANCHOR_LOCKED)
    print(f"[anchors] {len(anchors)} dockets locked")

    # Comments without a usable attachment_files URL but flagged attachment-only.
    print("[load] comments parquet ...")
    paths = sorted(COMMENTS_DIR.glob("EPA_*.parquet")) + sorted(COMMENTS_DIR.glob("EPA_*.pkl"))
    seen = set(); parts = []
    for path in paths:
        if path.stem in seen:
            continue
        seen.add(path.stem)
        parts.append(pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path))
    comments = pd.concat(parts, ignore_index=True)
    print(f"  total comments: {len(comments):,}")

    in_anchors = comments[comments["docket_id"].isin(anchors["docket_id"])]
    needs_api = in_anchors[
        in_anchors["is_attachment_only"]
        & (in_anchors["attachment_files"].fillna("").str.strip() == "")
    ]
    print(f"  attachment-only in anchor dockets needing API metadata: {len(needs_api):,}")

    if args.limit > 0:
        needs_api = needs_api.head(args.limit)
        print(f"  capped to {args.limit} per --limit")

    rows: list[dict] = []
    n_calls = 0
    t0 = time.time()
    for r in needs_api.itertuples():
        if n_calls and n_calls % 50 == 0:
            print(f"  [{n_calls}] {time.time()-t0:.0f}s elapsed; {len(rows)} attachments resolved")
        atts = fetch_attachments(r.document_id, api_key)
        for a in atts:
            rows.extend(parse_attachment(a, r.document_id, r.docket_id))
        n_calls += 1
        # Conservative throttle: ~1.4 req/sec under the 1000/hour cap
        time.sleep(0.7)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(OUT_PATH, index=False)
        print(f"[write] {OUT_PATH} ({len(out):,} attachment-format rows)")
    except ImportError:
        pkl = OUT_PATH.with_suffix(".pkl")
        out.to_pickle(pkl)
        print(f"[write] {pkl} (pyarrow unavailable; fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
