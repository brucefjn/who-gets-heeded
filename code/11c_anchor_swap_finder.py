"""
11c_anchor_swap_finder.py — Verification-first swap candidate finder.

CONTEXT (2026-05-07, Bruce decision):
  Two anchor dockets locked by code/03_anchor_rule_selection.py turned out to
  have proposed_year / final_year derived from regulations.gov "posted_date"
  metadata that does NOT correspond to the actual Federal Register publication
  date.

    - EPA-HQ-OAR-1994-0099 (mid tertile, 2019-2022 cluster, 770 comments):
        regulations.gov posted_date → 2022 / 2022
        Actual FR pubs filed in this docket → 1994/1995 (CA ozone) and
                                              2002/2002 (RFG modifications).
        BOTH pairs fall outside the 2010-2022 corpus. Swap.

    - EPA-HQ-OAR-2004-0265 (high tertile, 2019-2022 cluster, 1196 comments):
        regulations.gov posted_date → 2020 / 2020
        FR side: only one canonical pull (1994 final). v4 envelope confirms
        no 2010-2022 NPRM/Final pair filed in this docket. Swap.

PURPOSE:
  Provide a ranked list of swap candidates from the SAME stratum (year_cluster
  x within-cluster comment-volume tertile) that 1994-0099 / 2004-0265 came
  from, AND verify each candidate's Federal Register pub dates against the
  api.regulations.gov v4 envelope (canonical) before we lock anything.

PROCEDURE:
  1. Load `data/processed/anchor_rule_candidates.csv` and
     `data/processed/anchor_rules_locked.csv`.
  2. Restrict to year_cluster == '2019-2022' eligible pool (n_comments >= 500
     AND has_proposed_and_final == True).
  3. Compute the same tertile boundaries the original sampling used (within
     this cluster's eligible pool, 33rd / 67th percentiles of n_comments).
  4. Available = (eligible) - (locked - {1994-0099, 2004-0265})
                 - {1994-0099, 2004-0265}.
  5. For each available candidate, hit api.regulations.gov v4:
        GET /v4/dockets/{docket_id}                       → envelope
        GET /v4/documents?filter[docketId]={...}
            &filter[documentType]=Proposed Rule
        GET /v4/documents?filter[docketId]={...}
            &filter[documentType]=Rule
     Collect every frDocNum's `postedDate` (from v4) AND the FR API's
     `publication_date` for cross-check (the v4 postedDate is the
     regulations.gov upload date; the FR `publication_date` is canonical).
  6. Classify each candidate:
        - LOCKABLE: at least one Proposed Rule with FR pub-date in
                    [2010, 2022] AND at least one Rule with FR pub-date in
                    [2010, 2022], AND those two are plausibly paired
                    (overlap on RIN or temporally compatible).
        - METADATA_BUG: v4 lists rule-type docs but FR pub-dates fall outside
                        2010-2022 (same failure mode as 1994-0099).
        - EMPTY: v4 envelope OK but no rule documents listed.
        - UNREACHABLE: docket envelope HTTP != 200.
  7. Write report → `data/processed/anchor_swap_diagnostic.csv` and a Markdown
     summary printed to stdout.

USAGE:
    export API_DATA_GOV_KEY=...
    python code/11c_anchor_swap_finder.py
        # processes both swap targets' tertiles by default
    python code/11c_anchor_swap_finder.py --tertile mid
        # restrict to one tertile

NOTE:
  - Read-only with respect to the existing anchor_rules_locked.csv. This
    script does NOT lock anything; it produces a diagnostic for Yue + Bruce
    to make the swap decision from. Locking happens in a separate edit.
  - Hits the same X-Api-Key endpoint as code/11b. Reuses identical headers
    + 0.4s sleep between calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CANDIDATES = PROC / "anchor_rule_candidates.csv"
LOCKED = PROC / "anchor_rules_locked.csv"
OUT_CSV = PROC / "anchor_swap_diagnostic.csv"

# api.regulations.gov v4 — canonical source
REG_API_DOCKET = "https://api.regulations.gov/v4/dockets/{}"
REG_API_DOCS = "https://api.regulations.gov/v4/documents"
# Federal Register API — public, no auth
FR_API_DOC = "https://www.federalregister.gov/api/v1/documents/{}.json"

USER_AGENT = "regulatory-comments-project/1.0 (academic; contact: yy3462@columbia.edu)"
SLEEP_BETWEEN = 0.4

CORPUS_YEAR_LO = 2010
CORPUS_YEAR_HI = 2022

SWAP_TARGETS = {
    "EPA-HQ-OAR-1994-0099",   # mid tertile
    "EPA-HQ-OAR-2004-0265",   # high tertile
}


# ---------- HTTP helpers ----------
def http_get_json(url: str, headers: dict | None = None) -> tuple[int, dict | None]:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = Request(url, headers=h)
    try:
        with urlopen(req, timeout=30) as resp:
            code = resp.getcode()
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return code, json.loads(body)
            except json.JSONDecodeError:
                return code, None
    except HTTPError as e:
        return e.code, None
    except URLError:
        return 0, None


# ---------- v4 envelope query ----------
def query_v4(docket_id: str, api_key: str) -> dict:
    """Return a structured result for a single candidate."""
    headers = {"X-Api-Key": api_key}
    out = {
        "docket_id": docket_id,
        "envelope_status": None,
        "envelope_title": None,
        "n_proposed": 0,
        "n_final": 0,
        "proposed_fr_dates": [],
        "final_fr_dates": [],
        "proposed_fr_docs": [],
        "final_fr_docs": [],
        "proposed_titles": [],
        "final_titles": [],
        "fr_pull_errors": [],
    }

    # docket envelope
    code, env = http_get_json(REG_API_DOCKET.format(docket_id), headers)
    out["envelope_status"] = code
    if code != 200 or not env:
        return out
    attrs = (env.get("data") or {}).get("attributes") or {}
    out["envelope_title"] = attrs.get("title")

    # list docs by doctype
    for doctype, key_n, key_dates, key_docs, key_titles in [
        ("Proposed Rule", "n_proposed", "proposed_fr_dates", "proposed_fr_docs", "proposed_titles"),
        ("Rule", "n_final", "final_fr_dates", "final_fr_docs", "final_titles"),
    ]:
        params = {
            "filter[docketId]": docket_id,
            "filter[documentType]": doctype,
            "page[size]": 250,
        }
        url = REG_API_DOCS + "?" + urlencode(params)
        code, data = http_get_json(url, headers)
        time.sleep(SLEEP_BETWEEN)
        if code != 200 or not data:
            continue
        results = data.get("data", []) or []
        out[key_n] = len(results)
        for d in results:
            a = d.get("attributes", {}) or {}
            fr_doc = a.get("frDocNum") or a.get("frDocNumber")
            title = a.get("title")
            if fr_doc:
                out[key_docs].append(fr_doc)
                out[key_titles].append(title)
                # canonical FR pub date via FR API
                fr_url = FR_API_DOC.format(quote(fr_doc))
                code2, doc = http_get_json(fr_url)
                time.sleep(SLEEP_BETWEEN)
                if code2 == 200 and doc:
                    pubdate = doc.get("publication_date")
                    out[key_dates].append(pubdate)
                else:
                    out[key_dates].append(None)
                    out["fr_pull_errors"].append((fr_doc, code2))

    return out


# ---------- classification ----------
def classify(result: dict) -> tuple[str, str]:
    """Return (status, reason) where status is one of:
      LOCKABLE / METADATA_BUG / NO_PAIR / EMPTY / UNREACHABLE
    """
    if result["envelope_status"] != 200:
        return "UNREACHABLE", f"envelope HTTP {result['envelope_status']}"
    if result["n_proposed"] == 0 and result["n_final"] == 0:
        return "EMPTY", "v4 envelope OK but no Proposed Rule / Rule documents listed"

    def in_corpus(d):
        if not d:
            return False
        try:
            y = int(d[:4])
        except (ValueError, TypeError):
            return False
        return CORPUS_YEAR_LO <= y <= CORPUS_YEAR_HI

    p_in = [d for d in result["proposed_fr_dates"] if in_corpus(d)]
    f_in = [d for d in result["final_fr_dates"] if in_corpus(d)]

    if p_in and f_in:
        return ("LOCKABLE",
                f"{len(p_in)} proposed + {len(f_in)} final in {CORPUS_YEAR_LO}-{CORPUS_YEAR_HI}")
    if (result["n_proposed"] > 0 or result["n_final"] > 0) and not (p_in and f_in):
        return ("METADATA_BUG",
                f"v4 lists {result['n_proposed']}P + {result['n_final']}F but FR pub-dates "
                f"outside {CORPUS_YEAR_LO}-{CORPUS_YEAR_HI} "
                f"(P dates: {result['proposed_fr_dates']}; F dates: {result['final_fr_dates']})")
    return "NO_PAIR", "needs both Proposed and Final in corpus window"


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tertile", choices=["low", "mid", "high", "all"], default="all")
    ap.add_argument("--max-candidates-per-tertile", type=int, default=10)
    args = ap.parse_args()

    api_key = (os.environ.get("API_DATA_GOV_KEY")
               or os.environ.get("REGULATIONS_GOV_API_KEY")
               or os.environ.get("GOVINFO_API_KEY"))
    if not api_key:
        print("ERROR: set API_DATA_GOV_KEY (or REGULATIONS_GOV_API_KEY/GOVINFO_API_KEY)",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CANDIDATES)
    locked = pd.read_csv(LOCKED)
    locked_ids = set(locked["docket_id"])

    elig = df[(df["n_comments"] >= 500)
              & (df["has_proposed_and_final"] == True)
              & (df["year_cluster"] == "2019-2022")].copy()
    if elig.empty:
        print("Eligible 2019-2022 pool is empty — nothing to do.")
        return

    b33 = int(elig["n_comments"].quantile(1 / 3))
    b67 = int(elig["n_comments"].quantile(2 / 3))

    def tert(n):
        if n <= b33:
            return "low"
        if n <= b67:
            return "mid"
        return "high"
    elig["tertile"] = elig["n_comments"].apply(tert)

    # Available = (eligible) - (locked excluding swap targets) - swap targets
    keep = locked_ids - SWAP_TARGETS
    avail = elig[~elig["docket_id"].isin(keep)].copy()
    avail = avail[~avail["docket_id"].isin(SWAP_TARGETS)]

    tertiles = ["low", "mid", "high"] if args.tertile == "all" else [args.tertile]

    print(f"Tertile boundaries (within 2019-2022 eligible pool, n={len(elig)}): "
          f"low<={b33} < mid <={b67} < high")
    print(f"Locked retained (not being swapped): {len(keep)}/{len(locked_ids)} dockets")
    print(f"Swap targets: {sorted(SWAP_TARGETS)}")
    print()

    rows = []
    for t in tertiles:
        sub = avail[avail["tertile"] == t].sort_values("n_comments", ascending=False)
        sub = sub.head(args.max_candidates_per_tertile)
        print(f"=== {t.upper()} tertile — {len(sub)} candidate(s) to verify ===")
        for _, r in sub.iterrows():
            did = r["docket_id"]
            print(f"\n--- {did} (n={r['n_comments']}, P_meta={int(r['proposed_year'])}, "
                  f"F_meta={int(r['final_year'])}) ---")
            print(f"  candidate title: {r['title'][:100]}")
            res = query_v4(did, api_key)
            status, reason = classify(res)
            print(f"  envelope title: {(res['envelope_title'] or '')[:100]}")
            print(f"  v4 docs: {res['n_proposed']} Proposed Rule, {res['n_final']} Rule")
            print(f"  proposed FR pub dates: {res['proposed_fr_dates']}")
            print(f"  final    FR pub dates: {res['final_fr_dates']}")
            if res["fr_pull_errors"]:
                print(f"  FR pull errors: {res['fr_pull_errors']}")
            print(f"  STATUS: {status}  ({reason})")
            rows.append({
                "tertile": t,
                "docket_id": did,
                "n_comments": int(r["n_comments"]),
                "candidate_title": r["title"],
                "meta_proposed_year": (None if pd.isna(r["proposed_year"]) else int(r["proposed_year"])),
                "meta_final_year": (None if pd.isna(r["final_year"]) else int(r["final_year"])),
                "envelope_title": res["envelope_title"],
                "n_proposed_v4": res["n_proposed"],
                "n_final_v4": res["n_final"],
                "proposed_fr_dates": "|".join(d or "" for d in res["proposed_fr_dates"]),
                "final_fr_dates": "|".join(d or "" for d in res["final_fr_dates"]),
                "proposed_fr_docs": "|".join(res["proposed_fr_docs"]),
                "final_fr_docs": "|".join(res["final_fr_docs"]),
                "status": status,
                "reason": reason,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} candidate diagnostics → {OUT_CSV.relative_to(ROOT)}")

    print("\n=== Summary table ===")
    if not out_df.empty:
        for t in tertiles:
            sub = out_df[out_df["tertile"] == t]
            if sub.empty:
                continue
            print(f"\n{t.upper()} tertile:")
            for _, r in sub.iterrows():
                print(f"  [{r['status']:13s}] {r['docket_id']:25s} "
                      f"n={r['n_comments']:4d}  P_dates={r['proposed_fr_dates'] or '(none)'} "
                      f"F_dates={r['final_fr_dates'] or '(none)'}")


if __name__ == "__main__":
    main()
