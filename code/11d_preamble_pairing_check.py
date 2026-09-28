"""
11d_preamble_pairing_check.py — Preamble-citation verification for swap
candidates (Bruce 2026-05-07 #3).

CONTEXT
  Each swap candidate from 11c_anchor_swap_finder.py reports v4-listed
  Proposed Rule(s) and Final Rule(s) with FR pub dates. v4 lists them in the
  same docket envelope, but listing != pairing. The 1994-0099 audit
  established that "two NPRMs + one Final in the same docket" can mean:
    (a) Final finalizes one of the NPRMs (clean pair),
    (b) Final finalizes a different NPRM (distinct rulemaking, no pair),
    (c) Final cites multiple NPRMs (multi-NPRM pairing).
  We need preamble evidence to distinguish (a)/(c) from (b).

PROCEDURE
  For each (docket, final_fr_doc_number) we hit the FR API and read:
    - `action` — short summary line that often says "EPA is finalizing the
                  proposal published [date] at [citation]".
    - `abstract` — slightly longer summary.
    - First ~3000 chars of the `raw_text` body — the SUPPLEMENTARY
                  INFORMATION preamble where "this rule finalizes ..."
                  cross-references typically live.
  We extract:
    * Any FR citations (vol FR page) in the preamble;
    * Any explicit dates in the preamble's first paragraph;
    * Any FR doc numbers (XXXX-NNNNN format).
  Then we cross-check against the v4-listed Proposed Rule frDocNums and
  pub dates to determine pairing status.

USAGE
  python code/11d_preamble_pairing_check.py
  python code/11d_preamble_pairing_check.py --docket EPA-HQ-OAR-2018-0775
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

UA = "regulatory-comments-project/1.0 (academic; yy3462@columbia.edu)"
FR_API_DOC = "https://www.federalregister.gov/api/v1/documents/{}.json"
FR_API_LIST = "https://www.federalregister.gov/api/v1/documents.json"

# Default candidates from 11c v4 envelope query (pub dates exact, frDocNum
# resolved via FR API search to avoid hard-coding).
SWAP_CANDIDATES = {
    "EPA-HQ-OAR-2018-0775": {
        "proposed_pub_dates": ["2019-03-18", "2019-03-21"],
        "final_pub_dates":    ["2019-06-10"],
    },
    "EPA-HQ-OAR-2017-0757": {
        "proposed_pub_dates": ["2019-09-24", "2019-09-20"],
        "final_pub_dates":    ["2020-09-14"],
    },
}

# FR citation regex (handles "85 FR 12345" and 5-6 digit page numbers,
# matches the regex from code/11b after Bruce's fix)
FR_CITE_RE = re.compile(r"\b(\d{1,3})\s+(?:FR|Fed\.?\s*Reg\.?)\s+(\d{1,3}(?:,?\d{3})*)\b",
                        re.IGNORECASE)
FR_DOC_NUM_RE = re.compile(r"\b(20\d{2}-\d{4,6}|\d{2}-\d{4,6})\b")
DATE_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|"
                     r"September|October|November|December)\s+\d{1,2},?\s+(\d{4})\b")


def http_get_json(url: str) -> dict | None:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=45) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, json.JSONDecodeError):
        return None


def http_get_text(url: str) -> str | None:
    req = Request(url, headers={"User-Agent": UA})
    try:
        with urlopen(req, timeout=45) as r:
            return r.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError):
        return None


def find_doc_by_docket_and_date(docket_id: str, pub_date: str, doc_type: str) -> dict | None:
    """Find an FR document by docket + pub date + type."""
    type_filter = "RULE" if doc_type.lower() == "rule" else "PRORULE"
    params = {
        "conditions[term]": docket_id,
        "conditions[type][]": type_filter,
        "conditions[publication_date][is]": pub_date,
        "fields[]": ["document_number", "title", "type", "publication_date",
                     "raw_text_url", "html_url", "regulation_id_numbers",
                     "action", "abstract", "dates", "agencies"],
        "per_page": 5,
    }
    # urlencode with doseq for repeated keys
    qs = []
    for k, v in params.items():
        if isinstance(v, list):
            for x in v:
                qs.append((k, x))
        else:
            qs.append((k, v))
    url = FR_API_LIST + "?" + urlencode(qs, doseq=True)
    data = http_get_json(url)
    if not data or not (data.get("results") or []):
        return None
    return data["results"][0]


def vol_to_year(vol: int) -> int:
    """FR continuous volume: Vol N = 1935 + N."""
    return 1935 + vol


def scan_preamble(text: str) -> dict:
    """Extract pairing-relevant evidence from preamble text."""
    text = text or ""
    # First ~5000 chars usually covers SUMMARY + ACTION + DATES + first paragraph
    # of SUPPLEMENTARY INFORMATION. Strip HTML if present.
    if "<" in text:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
    snippet = text[:5000]

    fr_cites = list({(int(m.group(1)), m.group(2).replace(",", ""))
                     for m in FR_CITE_RE.finditer(snippet)})
    fr_cites = sorted(fr_cites)
    fr_cite_years = sorted({vol_to_year(v) for v, _ in fr_cites if 50 <= v <= 200})

    fr_doc_nums = sorted(set(m.group(1) for m in FR_DOC_NUM_RE.finditer(snippet)))
    dates = sorted({m.group(0) for m in DATE_RE.finditer(snippet)})
    return {
        "fr_citations": fr_cites,
        "fr_cite_years": fr_cite_years,
        "fr_doc_nums": fr_doc_nums,
        "dates_mentioned": dates,
        "snippet_head": snippet[:600],
    }


def check_candidate(docket_id: str, info: dict) -> None:
    print(f"\n{'='*72}\n{docket_id}\n{'='*72}")

    # Resolve Proposed Rule doc numbers
    proposed_meta = []
    for pd in info["proposed_pub_dates"]:
        d = find_doc_by_docket_and_date(docket_id, pd, "Proposed Rule")
        if d:
            proposed_meta.append({
                "fr_doc_number": d.get("document_number"),
                "publication_date": d.get("publication_date"),
                "title": d.get("title"),
                "rin": d.get("regulation_id_numbers"),
                "raw_text_url": d.get("raw_text_url"),
            })
        time.sleep(0.4)
    print(f"\nProposed Rules in docket (resolved via FR search):")
    for p in proposed_meta:
        print(f"  - {p['fr_doc_number']}  pub {p['publication_date']}  RIN={p['rin']}")
        print(f"    title: {(p['title'] or '')[:120]}")

    # Resolve Final Rule + read its preamble
    for fd in info["final_pub_dates"]:
        f = find_doc_by_docket_and_date(docket_id, fd, "Rule")
        if not f:
            print(f"\n!! Could not find Final at pub {fd} via FR search")
            continue
        time.sleep(0.4)
        print(f"\nFinal Rule:  {f.get('document_number')}  pub {f.get('publication_date')}")
        print(f"  title:    {(f.get('title') or '')[:120]}")
        print(f"  RIN:      {f.get('regulation_id_numbers')}")
        if f.get('action'):
            print(f"  action:   {f['action'][:300]}")
        if f.get('dates'):
            print(f"  dates:    {f['dates'][:300]}")
        if f.get('abstract'):
            print(f"  abstract: {f['abstract'][:300]}")

        # Pull first chunk of preamble text
        text = http_get_text(f.get('raw_text_url')) if f.get('raw_text_url') else None
        if text:
            ev = scan_preamble(text)
            print(f"\n  Preamble FR citations:  {ev['fr_citations'][:8]}")
            print(f"  Preamble FR cite years: {ev['fr_cite_years']}")
            print(f"  Preamble FR doc numbers: {ev['fr_doc_nums'][:10]}")
            print(f"  Preamble dates mentioned: {ev['dates_mentioned'][:6]}")

            # Cross-check: does preamble mention any Proposed pub date or
            # FR doc number from our v4 listing?
            proposed_dates_short = []
            for p in proposed_meta:
                pd_str = p["publication_date"]
                if pd_str:
                    y, m, d = pd_str.split("-")
                    month_name = ["", "January", "February", "March", "April", "May",
                                  "June", "July", "August", "September", "October",
                                  "November", "December"][int(m)]
                    proposed_dates_short.append(f"{month_name} {int(d)}, {y}")
                    proposed_dates_short.append(f"{month_name} {int(d)} {y}")

            preamble_lower = ev['snippet_head'].lower()
            print("\n  Pairing cross-check:")
            for p in proposed_meta:
                doc_match = (p["fr_doc_number"] in ev["fr_doc_nums"]) if p["fr_doc_number"] else False
                date_match = False
                if p["publication_date"]:
                    y, mo, d = p["publication_date"].split("-")
                    month_name = ["", "January", "February", "March", "April", "May",
                                  "June", "July", "August", "September", "October",
                                  "November", "December"][int(mo)]
                    date_match = (f"{month_name} {int(d)}, {y}" in ev['snippet_head']
                                  or f"{month_name} {int(d)} {y}" in ev['snippet_head'])
                rin_match = (p["rin"] and f["regulation_id_numbers"]
                             and bool(set(p["rin"] or []) & set(f["regulation_id_numbers"] or [])))
                tag = "PAIRED" if (doc_match or date_match or rin_match) else "  no link found"
                print(f"    NPRM {p['fr_doc_number']} ({p['publication_date']}): "
                      f"doc#={doc_match}, date={date_match}, RIN={rin_match}  →  {tag}")
        else:
            print("  (could not retrieve preamble text)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docket", help="restrict to one docket")
    args = ap.parse_args()

    items = SWAP_CANDIDATES.items()
    if args.docket:
        items = [(args.docket, SWAP_CANDIDATES[args.docket])]
    for did, info in items:
        check_candidate(did, info)


if __name__ == "__main__":
    main()
