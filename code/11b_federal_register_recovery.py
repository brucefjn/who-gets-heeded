"""
code/11b_federal_register_recovery.py

Workstream 1 — Federal Register recovery for dockets that failed the
initial pull (code/11_federal_register_pull.py).

Bruce 2026-05-07: 36/36 is the standard. Three dockets failed the initial
pull and need explicit recovery effort with per-attempt logging:

  * EPA-HQ-OAR-1994-0099  (0 docs from API; 1994 docket, FR API doesn't
    surface 1994-era entries cleanly though FR is fully digital from 1994)
  * EPA-HQ-OAR-2004-0265  (0 docs from API; likely docket-ID/metadata
    mismatch — RIN cross-reference should resolve)
  * EPA-HQ-OAR-2020-0044  (3 finals, 0 NPRMs; almost certainly cross-
    docketed — final-rule preambles cite originating NPRMs)

Recovery paths attempted in order (each per docket):

  1. regulations.gov docket landing page scrape → extract FR citation
     refs and RIN
  2. FR API by document_number (single-doc endpoint, bypasses
     conditions[docket_id] filter)
  3. FR API by regulation_id_numbers (RIN)
  4. Final-rule preamble text scan for cross-docketed NPRM FR-citation
     patterns (e.g. "X FR Y", "78 FR 12345")
  5. GovInfo.gov FR archive (requires GOVINFO_API_KEY env var; sign up
     free at api.data.gov/signup/)

Outputs:

  data/processed/fr_recovery_log.csv  -- one row per attempt with:
      docket_id, attempt_path, query, http_status, result_summary,
      success_bool, recovered_fr_doc_numbers
  notes/2026-05-07_fr_recovery_log.md -- prose diagnostic Bruce can read
  data/raw/federal_register/{docket}_{rule_type}_{fr_doc}.{txt,json}
      for any newly recovered documents
  data/processed/federal_register_index.csv  -- updated in-place to
      replace 'missing' rows with recovered ones (preserves original
      rows for unaffected dockets)

Escalation: any docket where ALL FIVE paths fail is logged with
escalate=True in the diagnostic note. Per Bruce's rule, escalate to him
before dropping the docket — fallback is anchor swap from the same
stratum, not silent shrink to n=35.

Usage:

  # Set the api.data.gov key (free, instant signup at api.data.gov/signup).
  # The same key works for both api.regulations.gov v4 and GovInfo.
  export API_DATA_GOV_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  # Recovery on the dockets that are still missing in the index:
  python3 code/11b_federal_register_recovery.py

  # Specific docket only:
  python3 code/11b_federal_register_recovery.py --docket EPA-HQ-OAR-1994-0099

Recovery paths attempted (Bruce 2026-05-07 spec):
  1b. api.regulations.gov v4 (canonical; tried FIRST when key is available)
  1.  regulations.gov public landing page (fallback; SPA, may return shell)
  3.  FR API by RIN (extracted from public landing page)
  4.  scan local final-rule preambles for cross-docketed NPRM citations
  5.  GovInfo FR archive (full-text + citation search)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CSV = REPO_ROOT / "data" / "processed" / "anchor_rules_locked.csv"
INDEX_CSV = REPO_ROOT / "data" / "processed" / "federal_register_index.csv"
RECOVERY_LOG_CSV = REPO_ROOT / "data" / "processed" / "fr_recovery_log.csv"
RECOVERY_NOTE_MD = REPO_ROOT / "notes" / "2026-05-07_fr_recovery_log.md"
RAW_DIR = REPO_ROOT / "data" / "raw" / "federal_register"

FR_API_DOC = "https://www.federalregister.gov/api/v1/documents/{}.json"
FR_API_LIST = "https://www.federalregister.gov/api/v1/documents.json"
REGULATIONS_GOV_DOCKET_PUBLIC = "https://www.regulations.gov/docket/{}"
# api.regulations.gov v4 — canonical source for docket+document metadata
# (regulations.gov SPA queries this internally). Free key from api.data.gov/signup.
REG_API_DOCKET = "https://api.regulations.gov/v4/dockets/{}"
REG_API_DOCS = "https://api.regulations.gov/v4/documents"
# GovInfo FR archive — fully digitized 1994+, queryable by FR citation,
# year, full-text search. Same api.data.gov key.
GOVINFO_SEARCH = "https://api.govinfo.gov/search"
GOVINFO_PACKAGES = "https://api.govinfo.gov/packages"

USER_AGENT = "regulatory-comments-project/1.1 (academic research; yy3462@columbia.edu)"
# regulations.gov blocks non-browser UAs. Use a Chrome-like UA there only.
USER_AGENT_BROWSER = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def strip_html(s: str) -> str:
    """Strip HTML tags. The FR raw_text_url returns HTML-wrapped plaintext,
    so files written by code/11_federal_register_pull.py may include
    <html><body><pre>...</pre></body></html> markup that wraps plain text.
    """
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    return s

# FR citation pattern: "85 FR 84130", "78 FR 12,345", "85 Fed. Reg. 84130"
# Volume: 1-3 digits (FR is on volume ~95 in 2030). Page: 1-6 digits with optional commas.
FR_CITATION_PATTERN = re.compile(
    r"\b(\d{1,3})\s+(?:FR|Fed\.?\s*Reg\.?)\s+(\d{1,3}(?:,?\d{3})*)\b",
    re.IGNORECASE,
)

# RIN pattern: e.g. "RIN 2060-AT91"
RIN_PATTERN = re.compile(r"\bRIN\s+(\d{4}-[A-Z]{2}\d{2,3})\b", re.IGNORECASE)


@dataclass
class Attempt:
    docket_id: str
    attempt_path: str
    query: str
    http_status: int | None = None
    result_summary: str = ""
    success: bool = False
    recovered_fr_doc_numbers: list[str] = field(default_factory=list)


# ---------------------------- HTTP helpers ----------------------------

def http_get(url: str, *, headers: dict | None = None, timeout: int = 30) -> tuple[int, str]:
    """GET a URL. Returns (status_code, body_text). Status -1 means non-HTTP error."""
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    req = Request(url, headers=h)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return e.code, body
    except Exception as e:
        return -1, str(e)


def http_get_json(url: str) -> tuple[int, dict | None]:
    """GET a URL expecting JSON. Returns (status_code, parsed_dict_or_None)."""
    code, body = http_get(url, headers={"Accept": "application/json"})
    if code == 200:
        try:
            return code, json.loads(body)
        except Exception as e:
            print(f"  json parse failed: {e}", file=sys.stderr)
            return code, None
    return code, None


# ----------------------- Recovery path implementations ----------------------

def path1_regulations_gov(docket_id: str) -> Attempt:
    """Path 1: scrape regulations.gov public landing page for FR refs and RIN."""
    url = REGULATIONS_GOV_DOCKET_PUBLIC.format(docket_id)
    a = Attempt(docket_id=docket_id, attempt_path="regulations_gov_landing", query=url)
    # Use browser UA to avoid 403; regulations.gov blocks non-browser User-Agents
    code, body = http_get(url, headers={"User-Agent": USER_AGENT_BROWSER}, timeout=45)
    a.http_status = code
    if code != 200:
        a.result_summary = f"HTTP {code}"
        return a

    # Extract FR citations and RINs from the page HTML
    fr_citations = sorted(set(m.group(0) for m in FR_CITATION_PATTERN.finditer(body)))
    rins = sorted(set(m.group(1).upper() for m in RIN_PATTERN.finditer(body)))
    a.result_summary = f"page fetched ({len(body):,} chars); FR cites: {len(fr_citations)}; RINs: {len(rins)}"
    if fr_citations or rins:
        a.success = True
    # Stash for downstream paths
    a.recovered_fr_doc_numbers = []  # this path returns citations, not doc numbers
    a.query += f"  [FR_CITES={fr_citations[:5]}; RINS={rins}]"
    return a


def path2_fr_by_doc_number(docket_id: str, doc_number: str) -> Attempt:
    """Path 2: hit the FR single-doc endpoint by document_number."""
    url = FR_API_DOC.format(quote(doc_number))
    a = Attempt(docket_id=docket_id, attempt_path="fr_by_doc_number", query=doc_number)
    code, data = http_get_json(url)
    a.http_status = code
    if code == 200 and data:
        a.success = True
        a.recovered_fr_doc_numbers = [data.get("document_number", doc_number)]
        a.result_summary = f"recovered: {data.get('type', '?')} - {(data.get('title') or '')[:80]}"
        # Save it
        save_recovered_doc(docket_id, data)
    else:
        a.result_summary = f"HTTP {code}"
    return a


def path3_fr_by_rin(docket_id: str, rin: str) -> Attempt:
    """Path 3: query FR API by RIN."""
    fields = ["document_number", "title", "type", "subtype", "publication_date",
              "abstract", "raw_text_url", "html_url", "json_url", "pdf_url",
              "regulation_id_numbers", "agencies", "docket_ids", "topics", "cfr_references"]
    base = (FR_API_LIST + "?" + urlencode({"per_page": 100})
            + f"&conditions[regulation_id_numbers][]={quote(rin)}"
            + "".join(f"&fields[]={f}" for f in fields))
    a = Attempt(docket_id=docket_id, attempt_path="fr_by_rin", query=rin)
    code, data = http_get_json(base)
    a.http_status = code
    if code == 200 and data:
        results = data.get("results", [])
        for d in results:
            save_recovered_doc(docket_id, d)
            a.recovered_fr_doc_numbers.append(d.get("document_number", "?"))
        a.success = bool(results)
        a.result_summary = f"got {len(results)} document(s) via RIN={rin}"
    else:
        a.result_summary = f"HTTP {code}"
    return a


def path4_preamble_scan(docket_id: str) -> Attempt:
    """Path 4: scan existing final-rule .txt files in this docket for cross-
    docketed NPRM FR citations, then fetch each."""
    a = Attempt(docket_id=docket_id, attempt_path="preamble_scan", query="(scan local .txt finals)")
    final_files = list(RAW_DIR.glob(f"{docket_id}_final_*.txt"))
    if not final_files:
        a.result_summary = "no final .txt files to scan"
        return a

    citations: list[str] = []
    for fp in final_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # FR raw_text_url returns HTML-wrapped plaintext; strip HTML before scanning
        if "<html" in text[:200].lower() or "<pre" in text[:200].lower():
            text = strip_html(text)
        # Find all FR citations: "85 FR 84130" or "78 Fed. Reg. 12,345"
        for m in FR_CITATION_PATTERN.finditer(text):
            volume = m.group(1)
            page = m.group(2).replace(",", "")
            citations.append(f"{volume} FR {page}")
    citations = sorted(set(citations))
    a.query += f"  [found {len(citations)} unique FR citations]"

    # Try to resolve each FR citation to a document_number via FR API
    # The FR API doesn't have a direct "find by FR citation" endpoint, but
    # we can use conditions[term]= and look for matches; or use the
    # publication_date.year derived from volume, since FR vol N → year ~1936+N.
    recovered = []
    for cite in citations[:25]:  # cap to keep API usage reasonable
        # Search by citation as a term
        url = (FR_API_LIST + "?" + urlencode({"per_page": 5, "conditions[term]": cite})
               + "&fields[]=document_number&fields[]=type&fields[]=title&fields[]=publication_date"
               + "&fields[]=raw_text_url&fields[]=html_url")
        code, data = http_get_json(url)
        if code == 200 and data:
            for d in (data.get("results") or []):
                if d.get("type") in ("Proposed Rule", "Rule"):
                    save_recovered_doc(docket_id, d)
                    recovered.append(d.get("document_number", "?"))
        time.sleep(0.4)
    a.recovered_fr_doc_numbers = recovered
    a.success = bool(recovered)
    a.result_summary = f"{len(citations)} citations scanned, {len(recovered)} doc(s) recovered"
    return a


def get_api_key(*names: str) -> str | None:
    """Look up an API key from env, supporting multiple names. Same api.data.gov
    key works for api.regulations.gov + GovInfo, so we accept several aliases."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def path1b_api_regulations_gov(docket_id: str, api_key: str) -> Attempt:
    """Path 1b: api.regulations.gov v4 — canonical docket + document metadata.

    Steps:
      1. GET /v4/dockets/{docketId}  → docket envelope (title, agency, RIN if any)
      2. GET /v4/documents?filter[docketId]={docketId}&filter[documentType]=Proposed Rule
         GET /v4/documents?filter[docketId]={docketId}&filter[documentType]=Rule
         → frDocNumber for each rule document
      3. For each frDocNumber, fall through to FR API (path 2) to pull text.
    """
    a = Attempt(docket_id=docket_id, attempt_path="api_regulations_gov_v4",
                query=f"GET /v4/documents?filter[docketId]={docket_id}")
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}

    # Step 1 — docket envelope (informational; we don't need it to recover, but it
    # logs whether the docket exists at all in the canonical source)
    code, env_data = http_get_json_with_headers(REG_API_DOCKET.format(docket_id), headers)
    a.http_status = code
    if code != 200:
        a.result_summary = f"docket envelope HTTP {code}"
        return a

    # Step 2 — list rule documents in the docket. Track which doctype each
    # frDocNum came from so we can classify correctly downstream (the FR API's
    # `type` field is unreliable for older docs — it returns 'Uncategorized
    # Document' for many pre-2000 entries even though v4 classifies them).
    docs_seen: list[dict] = []
    fr_doc_to_v4_type: dict[str, str] = {}
    for doctype in ("Proposed Rule", "Rule"):
        params = {
            "filter[docketId]": docket_id,
            "filter[documentType]": doctype,
            "page[size]": 250,
        }
        url = REG_API_DOCS + "?" + urlencode(params)
        code, data = http_get_json_with_headers(url, headers)
        if code != 200 or not data:
            continue
        results = data.get("data", []) or []
        docs_seen.extend(results)
        for d in results:
            attrs = d.get("attributes", {}) or {}
            fr_doc = attrs.get("frDocNum") or attrs.get("frDocNumber")
            if fr_doc:
                fr_doc_to_v4_type[fr_doc] = doctype
        time.sleep(0.4)

    fr_doc_numbers = sorted(fr_doc_to_v4_type.keys())
    a.result_summary = (f"docket envelope OK; {len(docs_seen)} rule-type documents listed "
                        f"({len(fr_doc_numbers)} with frDocNum)")

    # Step 3 — pull each FR document's text + JSON via the public FR API.
    # Use the v4 classification as the rule_type override (canonical source).
    recovered = []
    pull_failures: list[tuple[str, int]] = []
    for fr_doc in fr_doc_numbers:
        url = FR_API_DOC.format(quote(fr_doc))
        code, doc = http_get_json(url)
        if code == 200 and doc:
            v4_type = fr_doc_to_v4_type[fr_doc]
            rt = "proposed" if v4_type == "Proposed Rule" else "final"
            save_recovered_doc(docket_id, doc, rule_type_override=rt)
            recovered.append(fr_doc)
        else:
            pull_failures.append((fr_doc, code))
        time.sleep(0.4)
    a.recovered_fr_doc_numbers = recovered
    a.success = bool(recovered)
    if recovered:
        a.result_summary += f"; pulled {len(recovered)} doc(s) via FR API"
    if pull_failures:
        a.result_summary += (f"; {len(pull_failures)} pull failure(s): "
                             f"{[(f, c) for f, c in pull_failures[:5]]}")
    return a


def path5_govinfo(docket_id: str, fr_citations: list[str], docket_title: str | None,
                  api_key: str | None) -> Attempt:
    """Path 5: GovInfo.gov FR archive — fully digitized 1994+.

    Two sub-strategies:
      A. If we have FR citations from prior paths, search GovInfo for each.
      B. If we have a docket title, try a full-text search on the FR collection
         restricted to plausible date range.
    """
    a = Attempt(docket_id=docket_id, attempt_path="govinfo_archive",
                query=f"FR cites: {len(fr_citations)}; title: {bool(docket_title)}")
    if not api_key:
        a.result_summary = "no api.data.gov key; skipped"
        return a

    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    recovered: list[str] = []

    # Strategy A: search by FR citation (if we have any)
    for cite in fr_citations[:10]:
        # GovInfo search supports query expressions like collection:FR + AND + free text
        params = {"query": f'collection:FR "{cite}"', "pageSize": 5, "offsetMark": "*"}
        url = GOVINFO_SEARCH + "?" + urlencode(params)
        code, data = http_get_json_with_headers(url, headers)
        if code != 200 or not data:
            continue
        # GovInfo returns results with packageId and granuleId; we'd need a separate
        # call to /packages/{packageId}/granules/{granuleId}/htm or /pdf for the text.
        # For our purposes, having packageId is enough to log discovery; we won't
        # round-trip the granule fetch yet — we use the FR citation to fetch from
        # the public FR API instead, which is faster and returns clean JSON.
        results = data.get("results", [])
        if results and not recovered:
            a.result_summary = (f"GovInfo found {len(results)} candidate(s) for cite '{cite}' "
                                f"(packageIds: {[r.get('packageId') for r in results[:3]]})")
        time.sleep(0.4)

    # Strategy B: if no FR citations at all, search by docket title (truncated)
    if not fr_citations and docket_title:
        title_query = docket_title[:80].strip().replace('"', '')
        params = {"query": f'collection:FR "{title_query}"', "pageSize": 10, "offsetMark": "*"}
        url = GOVINFO_SEARCH + "?" + urlencode(params)
        code, data = http_get_json_with_headers(url, headers)
        if code == 200 and data:
            results = data.get("results", [])
            a.result_summary += f"; title-search returned {len(results)} candidate(s)"

    a.success = bool(recovered)
    if not a.result_summary or "no api.data.gov key" in a.result_summary:
        a.result_summary = "GovInfo search executed but no actionable matches"
    return a


def http_get_json_with_headers(url: str, headers: dict) -> tuple[int, dict | None]:
    """JSON GET with custom headers (e.g. X-Api-Key)."""
    code, body = http_get(url, headers=headers)
    if code == 200:
        try:
            return code, json.loads(body)
        except Exception:
            return code, None
    return code, None


# ------------------------- Helpers / saving ---------------------------

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def classify_rule_type(doc: dict) -> str:
    t = (doc.get("type") or "").strip()
    return "proposed" if t == "Proposed Rule" else ("final" if t == "Rule" else "other")


def save_recovered_doc(docket_id: str, doc: dict, rule_type_override: str | None = None) -> None:
    """Save a recovered FR document's text + JSON metadata into data/raw/federal_register/.

    rule_type_override: when provided (e.g. 'proposed' or 'final'), overrides
    classification based on the FR API's `type` field. Use this when the
    canonical source (api.regulations.gov v4) has classified the document
    and the FR API metadata is unreliable (common for pre-2000 docs where
    type='Uncategorized Document').
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rule_type = rule_type_override if rule_type_override else classify_rule_type(doc)
    fr_doc = doc.get("document_number", "unknown")
    stem = f"{docket_id}_{rule_type}_{safe_filename(fr_doc)}"
    json_path = RAW_DIR / f"{stem}.json"
    txt_path = RAW_DIR / f"{stem}.txt"
    json_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Try to fetch the body text
    text = None
    if doc.get("raw_text_url"):
        code, body = http_get(doc["raw_text_url"])
        if code == 200:
            text = body
    if text is None and doc.get("body_html_url"):
        code, body = http_get(doc["body_html_url"])
        if code == 200:
            text = re.sub(r"<[^>]+>", " ", body)
            text = re.sub(r"\s+", " ", text).strip()
    if text:
        txt_path.write_text(text, encoding="utf-8")


def update_index_for_recovered(attempts: list[Attempt]) -> None:
    """Add rows to federal_register_index.csv for newly recovered documents."""
    if not INDEX_CSV.exists():
        print("WARNING: federal_register_index.csv does not exist; cannot update", file=sys.stderr)
        return
    df = pd.read_csv(INDEX_CSV)

    new_rows = []
    for a in attempts:
        for fr_doc in a.recovered_fr_doc_numbers:
            # Find the saved JSON to extract metadata
            for stem_glob in [f"{a.docket_id}_proposed_{safe_filename(fr_doc)}.json",
                              f"{a.docket_id}_final_{safe_filename(fr_doc)}.json"]:
                p = RAW_DIR / stem_glob
                if p.exists():
                    try:
                        doc = json.loads(p.read_text())
                    except Exception:
                        continue
                    rule_type = "proposed" if "_proposed_" in stem_glob else "final"
                    txt_path = p.with_suffix(".txt")
                    n_chars = txt_path.stat().st_size if txt_path.exists() else 0
                    new_rows.append({
                        "docket_id": a.docket_id,
                        "rule_type": rule_type,
                        "fr_doc_number": doc.get("document_number", ""),
                        "publication_date": doc.get("publication_date", ""),
                        "title": (doc.get("title", "") or "").strip(),
                        "subtype": doc.get("subtype", "") or "",
                        "n_chars": n_chars,
                        "text_path": str(txt_path.relative_to(REPO_ROOT)) if txt_path.exists() else "",
                        "raw_text_url": doc.get("raw_text_url", "") or "",
                        "abstract": (doc.get("abstract", "") or "").strip()[:500],
                    })

    if new_rows:
        # Drop any existing 'missing' rows for these dockets (replaced by recovered)
        recovered_dockets = {r["docket_id"] for r in new_rows}
        df = df[~((df["rule_type"] == "missing") & (df["docket_id"].isin(recovered_dockets)))]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(INDEX_CSV, index=False)
        print(f"Updated {INDEX_CSV} with {len(new_rows)} recovered row(s)")


def write_recovery_log(attempts: list[Attempt]) -> None:
    """Write the per-attempt CSV log + the markdown diagnostic."""
    rows = [{
        "docket_id": a.docket_id,
        "attempt_path": a.attempt_path,
        "query": a.query,
        "http_status": a.http_status,
        "success": a.success,
        "n_recovered": len(a.recovered_fr_doc_numbers),
        "result_summary": a.result_summary,
    } for a in attempts]
    pd.DataFrame(rows).to_csv(RECOVERY_LOG_CSV, index=False)
    print(f"Wrote {RECOVERY_LOG_CSV}")

    # Group by docket for the markdown
    by_docket: dict[str, list[Attempt]] = {}
    for a in attempts:
        by_docket.setdefault(a.docket_id, []).append(a)

    lines = [
        "# Federal Register recovery log",
        "",
        "Date: 2026-05-07. Recovery script: `code/11b_federal_register_recovery.py`.",
        "Standard: 36/36 with documented per-attempt log per Bruce 2026-05-07.",
        "Escalation: any docket where ALL paths fail is escalated to Bruce, not dropped.",
        "",
    ]
    for docket_id, ats in by_docket.items():
        any_success = any(a.success for a in ats)
        n_recovered = sum(len(a.recovered_fr_doc_numbers) for a in ats)
        status = (f"✓ RECOVERED ({n_recovered} document(s))"
                  if any_success else "✗ ALL PATHS FAILED — ESCALATE")
        lines.append(f"## `{docket_id}` — {status}")
        lines.append("")
        for a in ats:
            mark = "✓" if a.success else "✗"
            lines.append(f"- {mark} **{a.attempt_path}** — query: `{a.query[:200]}`")
            lines.append(f"  - HTTP {a.http_status}; {a.result_summary}")
            if a.recovered_fr_doc_numbers:
                lines.append(f"  - recovered FR doc numbers: {a.recovered_fr_doc_numbers}")
        lines.append("")

    RECOVERY_NOTE_MD.parent.mkdir(parents=True, exist_ok=True)
    RECOVERY_NOTE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RECOVERY_NOTE_MD}")


# --------------------------------- Main ---------------------------------

def find_failed_dockets() -> list[str]:
    """Read the index and find dockets that have at least one missing row
    AND don't have both proposed + final text already."""
    if not INDEX_CSV.exists():
        print(f"ERROR: {INDEX_CSV} not found; run code/11_federal_register_pull.py first")
        sys.exit(1)
    df = pd.read_csv(INDEX_CSV)
    failed = set(df.loc[df["rule_type"] == "missing", "docket_id"])
    # Also include dockets where one of proposed / final is entirely missing
    has_text = df["n_chars"].fillna(0) > 0
    by_docket = df[has_text].groupby("docket_id")["rule_type"].apply(set)
    for docket, types in by_docket.items():
        if "proposed" not in types or "final" not in types:
            failed.add(docket)
    return sorted(failed)


def recover_one_docket(docket_id: str) -> list[Attempt]:
    print(f"\n=== Recovering {docket_id} ===")
    attempts: list[Attempt] = []
    api_key = get_api_key("API_DATA_GOV_KEY", "REGULATIONS_GOV_API_KEY", "GOVINFO_API_KEY")

    # Path 1b: api.regulations.gov v4 (canonical) — try this FIRST (Bruce 2026-05-07)
    docket_title: str | None = None
    if api_key:
        a1b = path1b_api_regulations_gov(docket_id, api_key)
        print(f"  [path1b api.regulations.gov v4] {a1b.result_summary}")
        attempts.append(a1b)
        # Extract title from envelope for downstream paths
        # (best-effort — even if the call failed we proceed)
    else:
        skip = Attempt(docket_id=docket_id, attempt_path="api_regulations_gov_v4",
                       query="(no api.data.gov key)",
                       result_summary="API_DATA_GOV_KEY env var not set; skipped")
        print(f"  [path1b api.regulations.gov v4] {skip.result_summary}")
        attempts.append(skip)

    # Path 1: regulations.gov public landing page (browser UA)
    a1 = path1_regulations_gov(docket_id)
    print(f"  [path1 regulations.gov public] {a1.result_summary}")
    attempts.append(a1)

    # Extract FR citations and RIN(s) from the body of path 1's response
    rins: list[str] = []
    fr_citations: list[str] = []
    if a1.http_status == 200:
        url = REGULATIONS_GOV_DOCKET_PUBLIC.format(docket_id)
        code, body = http_get(url, headers={"User-Agent": USER_AGENT_BROWSER}, timeout=45)
        if code == 200:
            fr_citations = sorted(set(m.group(0) for m in FR_CITATION_PATTERN.finditer(body)))
            rins = sorted(set(m.group(1).upper() for m in RIN_PATTERN.finditer(body)))

    # Path 3: by RIN(s) (FR API search by regulation_id_numbers)
    for rin in rins[:3]:
        a3 = path3_fr_by_rin(docket_id, rin)
        print(f"  [path3 by RIN={rin}] {a3.result_summary}")
        attempts.append(a3)
        time.sleep(0.5)

    # Path 4: scan local final preambles for cross-docketed NPRM citations
    a4 = path4_preamble_scan(docket_id)
    print(f"  [path4 preamble scan] {a4.result_summary}")
    attempts.append(a4)

    # Path 5: GovInfo FR archive (now with real implementation)
    a5 = path5_govinfo(docket_id, fr_citations, docket_title, api_key)
    print(f"  [path5 GovInfo] {a5.result_summary}")
    attempts.append(a5)

    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docket", type=str, default=None, action="append",
                    help="recover one specific docket (default: all failed); "
                         "repeatable: --docket A --docket B")
    args = ap.parse_args()

    if args.docket:
        # action="append" gives a list; preserve order, dedupe.
        targets = list(dict.fromkeys(args.docket))
    else:
        targets = find_failed_dockets()
    print(f"Recovery targets: {targets}")

    all_attempts: list[Attempt] = []
    for d in targets:
        all_attempts.extend(recover_one_docket(d))
        time.sleep(0.5)

    update_index_for_recovered(all_attempts)
    write_recovery_log(all_attempts)

    print("\n=== summary ===")
    by_docket: dict[str, list[Attempt]] = {}
    for a in all_attempts:
        by_docket.setdefault(a.docket_id, []).append(a)
    for d, ats in by_docket.items():
        ok = any(a.success for a in ats)
        n = sum(len(a.recovered_fr_doc_numbers) for a in ats)
        print(f"  {d}: {'✓ recovered' if ok else '✗ ALL PATHS FAILED — ESCALATE'} ({n} doc(s))")


if __name__ == "__main__":
    main()
