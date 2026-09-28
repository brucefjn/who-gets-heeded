"""
03_anchor_rule_selection.py — Methodology rigor standard 2026-05-06.

PURPOSE
  Produce a locked anchor-rule list and two robustness samples for Option 2
  stratified attachment retrieval.

ELIGIBILITY FILTER (single rule, intentionally no text-content threshold)
  - n_comments >= 500
      Rigor: published-and-cited. High-stakes-comment-volume cutoff is
      consistent with Libgober & Rashin (2023) — see paper_notes/.
  - docket has at least one Proposed Rule AND one Final Rule (or Withdrawal)
    posted in 2010-2022.
      Rigor: published-and-cited. Standard scope filter for rulemaking outcome
      studies.
  Note: a >=30% text-content threshold was considered and withdrawn — it
  collapsed the 2010-2012 cluster to 1 narrow Pesticide Petitions docket,
  and filtering on text-content while stratifying on era (which correlates
  with attachment-only rate) is methodologically tense. Text-content rate is
  treated as a downstream analysis variable instead.

STEP 1 — STRATIFIED RANDOM SAMPLE (24 anchors)
  Strata: year_cluster x within-cluster comment-volume tertile.
    Year clusters: 2010-2012, 2013-2015, 2016-2018, 2019-2022.
    Tertile boundaries: 33rd and 67th percentiles of n_comments computed
    *within each cluster's eligible pool* (not on the global pool).
  4 clusters x 3 tertiles = 12 strata; 2 dockets per stratum, fixed seed
  PRIMARY_SEED = 20260505. Strata short of 2 are documented in the log.
    Rigor: published-and-cited (Lohr 2009 Sampling: Design & Analysis;
    Sage Encyclopedia of Educational Research, Measurement, and Evaluation —
    'strata should minimize within-stratum variance').

STEP 2 — EXTREME-CASE ADDITIONS (Obama-Trump regulatory-reversal pairs)
  Detect pairs (D1_docket, D2_docket) where:
    * D1 has a Final Rule (document_type == "Rule") posted 2014-2016
    * D2 has a Proposed Rule (document_type == "Proposed Rule") posted 2017-2020
    * Both D1 and D2 are in the eligible pool
    * AND: cfr_overlap(D1, D2) >= 2 -OR- textual_reference(D1, D2)
  Where:
    cfr_overlap     = | D1.cfr_set intersect D2.cfr_set |
                      where cfr_set is the set of (CFR title, CFR part)
                      tuples parsed from the docs' CFR field.
    textual_reference = D2's Proposed-Rule title or abstract contains:
        - D1's docket_id (substring match)  OR
        - D1's federal_register_number (substring match)  OR
        - any of {"repeal","rescind","withdraw","withdrawal","rescission",
          "repeals","rescinds","withdraws"} AND >=3 distinctive title-words
          from D1's Final-Rule title (>=4 chars, non-stopword).
  Both halves of every detected pair (D1_docket and D2_docket) are added to
  the anchor list, deduplicated against the stratified random sample.
    Rigor: novel-but-defensible. Extreme-case selection is published
    methodology (Klotz 2008; Gerring 2007). The Obama-Trump operationalization
    and the specific criteria (CFR overlap, keyword+title-word reference) are
    ours. Each criterion is objectively verifiable from primary documents.
  Per Bruce 2026-05-06: NO HAND-ADDING. If criteria miss CPP<->repeal or
  WOTUS<->recodification, the criteria — not the result — get fixed.

STEP 3 — ROBUSTNESS SAMPLES
  alt-1: stratified random sample with ALT1_SEED = 20260506. Same strata,
         same n_per_stratum.
  alt-2: top-25 dockets by n_comments (Libgober & Rashin 2023 style).
    Rigor: published-and-cited. Sensitivity analysis with alternative sample
    frames is standard.

OUTPUTS
  data/processed/anchor_rules_locked.csv
  data/processed/anchor_rules_alt1_random_seed2.csv
  data/processed/anchor_rules_alt2_top_volume.csv
  data/processed/anchor_selection_log.md
  data/processed/anchor_rule_candidates.csv  (full ranked candidate list,
                                              same shape as before)

USAGE
  python code/03_anchor_rule_selection.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"
COMMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "comments"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

PRIMARY_SEED = 20260505
ALT1_SEED = 20260506
N_PER_STRATUM = 2
N_TERTILES = 3

YEAR_CLUSTERS = [
    ("2010-2012", 2010, 2012),
    ("2013-2015", 2013, 2015),
    ("2016-2018", 2016, 2018),
    ("2019-2022", 2019, 2022),
]

OBAMA_FINAL_YEARS = (2014, 2016)
TRUMP_PROPOSED_YEARS = (2017, 2020)

# Bumped from 2 -> 3 on 2026-05-06 after pilot run showed FP cluster from
# generic Part 60 (NSPS) + Part 63 (NESHAP) co-occurrence. Parts 60 and 63 are
# parent CFR locations for hundreds of unrelated EPA rules; two-part overlap
# is weak evidence of regulatory relatedness. >=3 targets rule-subsystem-level
# overlap. Rigor: novel-but-defensible.
CFR_OVERLAP_MIN = 3

TITLE_WORDS_MIN = 3

# Canonical administrative-law reversal vocabulary. Expanded 2026-05-06 to
# include recodify/supersede/reconsider/vacate after WOTUS-recodification
# diagnostic showed the original 3-word set under-specified the category
# (not a post-hoc adjustment to capture a specific pair). Vocabulary
# corresponds to standard APA-section regulatory-action verbs.
# Rigor: published-but-adapted.
REVERSAL_KEYWORDS = {
    # original three (repeal-class)
    "repeal", "repeals", "repealed", "repealing",
    "rescind", "rescinds", "rescinded", "rescinding", "rescission",
    "withdraw", "withdraws", "withdrawn", "withdrawing", "withdrawal",
    # added 2026-05-06
    "recodify", "recodifies", "recodified", "recodifying", "recodification",
    "supersede", "supersedes", "superseded", "superseding",
    "reconsider", "reconsiders", "reconsidered", "reconsidering", "reconsideration",
    "vacate", "vacates", "vacated", "vacating", "vacatur",
}

# Conservative stopword list — primarily English structure words. Kept short
# on purpose; we want the title-word-overlap filter to be reasonably permissive.
STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "in", "on", "at", "to", "from",
    "is", "are", "was", "were", "be", "been", "being", "by", "with", "as",
    "or", "that", "this", "it", "its", "their", "our", "rule", "rules",
    "proposed", "final", "act",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_parquet(path_dir: Path, glob: str = "EPA_*.parquet") -> pd.DataFrame:
    paths = sorted(path_dir.glob(glob))
    if not paths:
        raise FileNotFoundError(f"no parquet under {path_dir} matching {glob}")
    parts = [pd.read_parquet(p) for p in paths]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# CFR parsing (set of (title:int, part:int) tuples per document)
# ---------------------------------------------------------------------------

# Examples encountered in EPA bulk:
#   "40 CFR Part 19"
#   "40 CFR Parts 60, 63"
#   "40 CFR 60.5777"
#   "33 CFR Part 328; 40 CFR Part 230"
# We extract title + part numbers; subpart letters are ignored. Part numbers
# can be followed by ".N" (decimal section) — we ignore the decimal and keep
# the integer part.

_CFR_TITLE_RE = re.compile(r"\b(\d{1,3})\s*CFR\b", re.I)
_CFR_PART_RE = re.compile(r"\b(\d{1,4})\b")


def parse_cfr(text) -> set:
    """Parse a CFR field into a set of (title:int, part:int) tuples.
    Splits the text on each CFR-title occurrence so part numbers from one
    title don't bleed into the next."""
    if not text or pd.isna(text):
        return set()
    s = str(text)
    out: set[tuple[int, int]] = set()
    # Find each "<title> CFR" occurrence and the substring up to the next one.
    matches = list(_CFR_TITLE_RE.finditer(s))
    if not matches:
        return out
    for i, m in enumerate(matches):
        title = int(m.group(1))
        # Substring of part-numbers belonging to this title
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        seg = s[start:end]
        # Find part-number tokens, ignoring decimals (.5777 etc.)
        for pm in _CFR_PART_RE.finditer(seg):
            try:
                p = int(pm.group(1))
                if 1 <= p <= 9999:
                    out.add((title, p))
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Textual-reference matching
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+")


def title_keywords(title) -> set:
    if not title or pd.isna(title):
        return set()
    return {
        w.lower()
        for w in _WORD_RE.findall(str(title))
        if len(w) >= 4 and w.lower() not in STOPWORDS
    }


def textual_reference(d1_docket: str, d1_fr: str, d1_title: str,
                      d2_text: str) -> tuple[bool, str]:
    """Return (matched, reason)."""
    if not d2_text:
        return False, ""
    s = str(d2_text)
    s_low = s.lower()

    if d1_docket and d1_docket in s:
        return True, f"D1 docket_id `{d1_docket}` appears in D2 text"
    if d1_fr and str(d1_fr).strip() and str(d1_fr) in s:
        return True, f"D1 FR number `{d1_fr}` appears in D2 text"

    has_kw = next((kw for kw in REVERSAL_KEYWORDS if kw in s_low), None)
    if has_kw:
        d1_kw = title_keywords(d1_title)
        d2_kw = title_keywords(d2_text)
        shared = d1_kw & d2_kw
        if len(shared) >= TITLE_WORDS_MIN:
            sample = sorted(shared)[:6]
            return True, f"reversal keyword `{has_kw}` + {len(shared)} shared title words ({', '.join(sample)})"
    return False, ""


# ---------------------------------------------------------------------------
# Per-docket aggregation
# ---------------------------------------------------------------------------

def cluster_for_year(y) -> str:
    if pd.isna(y):
        return "no-proposed-rule"
    yi = int(y)
    for name, lo, hi in YEAR_CLUSTERS:
        if lo <= yi <= hi:
            return name
    return "out-of-window"


# ---------------------------------------------------------------------------
# Canonical FR-year derivation (Bruce 2026-05-07 metadata-trust fix)
# ---------------------------------------------------------------------------
# `posted_date` in the regulations.gov bulk reflects when the docket /
# document was uploaded to regulations.gov, NOT when the rule was published in
# the Federal Register. For "legacy dockets" — typically pre-2002 dockets that
# regulations.gov ingested in bulk on a single later date (e.g. EPA Air Docket
# A-94-09 → EPA-HQ-OAR-1994-0099, all rule docs posted_date=2022-04-29) — this
# means `posted_date.year` is essentially uniform and misclassifies the
# rulemaking era. The same failure mode hit EPA-HQ-OAR-2004-0265 in our 36-
# anchor pull.
#
# The fix below derives the rule's year from the Federal Register itself, in
# this priority order:
#   (1) Volume number parsed from `fr_citation`            (most reliable)
#         FR continuous volume numbering: Vol N = 1935 + N
#         (Vol 75 = 2010, Vol 87 = 2022, etc.)
#   (2) Year prefix parsed from `federal_register_number`  (fallback)
#         Pre-2000: 2-digit prefix `XX-NNNN` → 1900+XX (e.g. `94-11399` → 1994)
#         Post-2000 (regulations.gov bulk has no docs from 2000-2009 with
#         2-digit prefix in this dataset): 4-digit prefix `YYYY-NNNNN`
#         (e.g. `2022-18320` → 2022)
#   (3) `posted_date.year`                                 (legacy fallback)
#
# This DOES NOT regenerate the existing locked anchors — the anchor list is
# CSV-cached at data/processed/anchor_rules_locked.csv, and the 33 verified
# pulls remain locked. Re-running this script will produce a corrected
# candidate list (anchor_rule_candidates.csv) where year_cluster is no longer
# distorted by legacy-docket bulk-upload dates.

_FR_VOL_RE = re.compile(r"\b(\d{2,3})\s+FR\b")
_FR_NUM_4Y_RE = re.compile(r"^(20\d{2})-")
_FR_NUM_2Y_RE = re.compile(r"^(\d{2})-")


def _fr_year_from_citation(citation: str | float | None) -> int | None:
    """Parse FR Volume → publication year. Vol N = 1935 + N."""
    if not isinstance(citation, str):
        return None
    m = _FR_VOL_RE.search(citation)
    if not m:
        return None
    try:
        vol = int(m.group(1))
    except ValueError:
        return None
    if 59 <= vol <= 200:  # Vol 59 = 1994 (oldest plausible) ... room for 2050
        return 1935 + vol
    return None


def _fr_year_from_doc_number(fr_num: str | float | None) -> int | None:
    """Parse year from FR document number like `2022-18320` or `94-11399`."""
    if not isinstance(fr_num, str):
        return None
    m = _FR_NUM_4Y_RE.match(fr_num)
    if m:
        return int(m.group(1))
    m = _FR_NUM_2Y_RE.match(fr_num)
    if m:
        yy = int(m.group(1))
        # FR Doc# 2-digit prefix: pre-2000 only. `94`..`99` → 1994..1999.
        # Post-2000 docs use 4-digit prefix and won't match this branch.
        return 1900 + yy if yy >= 80 else 2000 + yy
    return None


def derive_fr_year(row: pd.Series) -> int | None:
    """Best-effort canonical year for a rule document.
    Priority: fr_citation volume > federal_register_number prefix > posted_year."""
    y = _fr_year_from_citation(row.get("fr_citation"))
    if y is not None:
        return y
    y = _fr_year_from_doc_number(row.get("federal_register_number"))
    if y is not None:
        return y
    pd_year = row.get("posted_year")
    if pd.notna(pd_year):
        return int(pd_year)
    return None


def build_candidates(comments: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per docket with all the columns the rest of the script needs."""
    by_docket = comments.groupby("docket_id", dropna=True).agg(
        n_comments=("document_id", "size"),
        n_attach_only=("is_attachment_only", "sum"),
    )
    by_docket["pct_attach_only"] = by_docket["n_attach_only"] / by_docket["n_comments"] * 100
    by_docket = by_docket.reset_index()
    by_docket["n_comments"] = by_docket["n_comments"].astype(int)
    by_docket["n_attach_only"] = by_docket["n_attach_only"].astype(int)

    # Document-type presence
    dt = docs.assign(document_type=docs["document_type"].fillna("Unknown"))
    dt_pivot = dt.groupby(["docket_id", "document_type"]).size().unstack(fill_value=0)
    for c in ["Proposed Rule", "Rule"]:
        if c not in dt_pivot.columns:
            dt_pivot[c] = 0
    dt_pivot["has_proposed_rule"] = dt_pivot["Proposed Rule"] > 0
    dt_pivot["has_final_rule"] = dt_pivot["Rule"] > 0
    dt_pivot = dt_pivot.reset_index()

    # Withdrawal: regulations.gov bulk doesn't use a separate Document Type,
    # but Final-Rule rows whose title contains 'withdraw' qualify as withdrawals.
    final_rules = docs[docs["document_type"] == "Rule"].copy()
    has_withdrawal = (
        final_rules[final_rules["title"].fillna("").str.lower().str.contains("withdraw")]
        .groupby("docket_id").size().gt(0)
    )
    has_withdrawal_set = set(has_withdrawal[has_withdrawal].index.tolist())

    docs = docs.copy()
    docs["posted_year"] = pd.to_datetime(
        docs["posted_date"], errors="coerce", utc=True
    ).dt.year
    # Bruce 2026-05-07: derive year from canonical FR pub date (volume number
    # in fr_citation, or year prefix in federal_register_number) rather than
    # regulations.gov posted_date — see helper docstring above. This is what
    # `proposed_year` / `final_year` should mean: the year the NPRM / Final
    # was published in the Federal Register, NOT the year the docket was
    # uploaded to regulations.gov.
    docs["fr_year"] = docs.apply(derive_fr_year, axis=1).astype("Int64")

    proposed = docs[docs["document_type"] == "Proposed Rule"].dropna(subset=["fr_year"])
    final = docs[docs["document_type"] == "Rule"].dropna(subset=["fr_year"])
    proposed_first = proposed.groupby("docket_id")["fr_year"].min().astype("Int64")
    final_first = final.groupby("docket_id")["fr_year"].min().astype("Int64")
    rep_title = (
        proposed.sort_values(["docket_id", "posted_date"])
        .drop_duplicates("docket_id")
        .set_index("docket_id")["title"]
    )

    cand = (
        by_docket
        .merge(dt_pivot[["docket_id", "Proposed Rule", "Rule",
                         "has_proposed_rule", "has_final_rule"]],
               on="docket_id", how="left")
        .merge(proposed_first.rename("proposed_year"), on="docket_id", how="left")
        .merge(final_first.rename("final_year"), on="docket_id", how="left")
    )
    cand["title"] = cand["docket_id"].map(rep_title).fillna("")
    cand["has_withdrawal"] = cand["docket_id"].isin(has_withdrawal_set)
    cand["has_proposed_and_final"] = (
        cand["has_proposed_rule"].fillna(False)
        & (cand["has_final_rule"].fillna(False) | cand["has_withdrawal"])
    )
    cand["year_cluster"] = cand["proposed_year"].apply(cluster_for_year)
    return cand


# ---------------------------------------------------------------------------
# Stratified random sampling
# ---------------------------------------------------------------------------

def compute_tertile_boundaries(volumes: list[int]) -> tuple[int, int]:
    """Return (b33, b67) — the boundaries that partition the sorted volumes
    into three roughly equal-sized strata. Lower-bound inclusive in the
    upper tertile."""
    if not volumes:
        return (0, 0)
    arr = np.asarray(sorted(volumes))
    p33 = int(np.percentile(arr, 100 / 3, method="lower"))
    p67 = int(np.percentile(arr, 200 / 3, method="lower"))
    return p33, p67


def assign_tertile(n_comments: int, b33: int, b67: int) -> str:
    if n_comments < b33:
        return "low"
    if n_comments < b67:
        return "mid"
    return "high"


def stratified_random_sample(eligible: pd.DataFrame, seed: int,
                              ) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (sample_df, log_rows)."""
    rng = np.random.default_rng(seed)
    log: list[dict] = []
    picks: list[pd.DataFrame] = []

    for cluster_name, _, _ in YEAR_CLUSTERS:
        cluster_pool = eligible[eligible["year_cluster"] == cluster_name].copy()
        if cluster_pool.empty:
            log.append({"cluster": cluster_name, "tertile": "(all)",
                        "pool_size": 0, "boundaries": (None, None),
                        "drawn": 0, "shortfall": N_PER_STRATUM * N_TERTILES,
                        "picks": []})
            continue
        b33, b67 = compute_tertile_boundaries(cluster_pool["n_comments"].tolist())
        cluster_pool["tertile"] = cluster_pool["n_comments"].apply(
            lambda n: assign_tertile(int(n), b33, b67)
        )
        for tertile in ["low", "mid", "high"]:
            stratum = cluster_pool[cluster_pool["tertile"] == tertile]
            if stratum.empty:
                log.append({"cluster": cluster_name, "tertile": tertile,
                            "pool_size": 0, "boundaries": (b33, b67),
                            "drawn": 0, "shortfall": N_PER_STRATUM,
                            "picks": []})
                continue
            n = min(N_PER_STRATUM, len(stratum))
            # Deterministic permutation
            idx = rng.permutation(len(stratum))[:n]
            chosen = stratum.iloc[idx].copy()
            chosen["stratum_cluster"] = cluster_name
            chosen["stratum_tertile"] = tertile
            picks.append(chosen)
            log.append({
                "cluster": cluster_name, "tertile": tertile,
                "pool_size": len(stratum), "boundaries": (b33, b67),
                "drawn": n, "shortfall": max(0, N_PER_STRATUM - n),
                "picks": chosen["docket_id"].tolist(),
            })

    if not picks:
        return pd.DataFrame(columns=eligible.columns), log
    sample = pd.concat(picks, ignore_index=True)
    return sample, log


# ---------------------------------------------------------------------------
# Extreme-case detection
# ---------------------------------------------------------------------------

def detect_extreme_cases(eligible: pd.DataFrame, docs: pd.DataFrame
                          ) -> tuple[list[dict], set[str]]:
    """Return (pairs_log, extreme_dockets_set).
    pairs_log has one row per (D1_docket, D2_docket) pair detected, with the
    matching criterion (cfr_overlap, textual_reference, or both)."""
    docs = docs.copy()
    docs["posted_year"] = pd.to_datetime(
        docs["posted_date"], errors="coerce", utc=True
    ).dt.year
    # Bruce 2026-05-07: extreme-case era windows must also use canonical FR
    # year (Obama-era = FR pub 2014-2016, Trump-era = FR pub 2017-2020), not
    # regulations.gov upload year. See derive_fr_year docstring above.
    docs["fr_year"] = docs.apply(derive_fr_year, axis=1).astype("Int64")

    elig_dockets = set(eligible["docket_id"])

    obama_finals = docs[
        (docs["document_type"] == "Rule")
        & docs["fr_year"].between(*OBAMA_FINAL_YEARS, inclusive="both")
        & docs["docket_id"].isin(elig_dockets)
    ].copy()
    trump_proposed = docs[
        (docs["document_type"] == "Proposed Rule")
        & docs["fr_year"].between(*TRUMP_PROPOSED_YEARS, inclusive="both")
        & docs["docket_id"].isin(elig_dockets)
    ].copy()

    obama_finals["cfr_set"] = obama_finals["cfr"].apply(parse_cfr)
    trump_proposed["cfr_set"] = trump_proposed["cfr"].apply(parse_cfr)

    pairs: dict[tuple[str, str], dict] = {}
    for _, d1 in obama_finals.iterrows():
        d1_docket = d1["docket_id"]
        d1_fr = str(d1.get("federal_register_number", "") or "")
        d1_title = d1.get("title", "") or ""
        d1_cfr = d1["cfr_set"]
        d1_doc_id = d1.get("document_id", "")

        for _, d2 in trump_proposed.iterrows():
            d2_docket = d2["docket_id"]
            if d2_docket == d1_docket:
                continue
            d2_text_parts = []
            for col in ("title", "abstract"):
                v = d2.get(col, "")
                if v and not pd.isna(v):
                    d2_text_parts.append(str(v))
            d2_text = "\n".join(d2_text_parts)
            d2_cfr = d2["cfr_set"]
            d2_doc_id = d2.get("document_id", "")

            cfr_n = len(d1_cfr & d2_cfr)
            cfr_match = cfr_n >= CFR_OVERLAP_MIN
            tr_match, tr_reason = textual_reference(d1_docket, d1_fr, d1_title, d2_text)
            if not (cfr_match or tr_match):
                continue

            key = (d1_docket, d2_docket)
            existing = pairs.get(key)
            evidence = []
            if cfr_match:
                evidence.append(f"cfr_overlap={cfr_n} (shared: "
                                f"{', '.join(f'{t}-{p}' for t, p in sorted(d1_cfr & d2_cfr)[:5])})")
            if tr_match:
                evidence.append(f"textual_ref: {tr_reason}")
            if existing is None or cfr_n > existing["cfr_overlap"]:
                pairs[key] = {
                    "d1_docket": d1_docket,
                    "d1_doc_id": d1_doc_id,
                    "d1_title": str(d1_title)[:100],
                    "d1_fr": d1_fr,
                    "d2_docket": d2_docket,
                    "d2_doc_id": d2_doc_id,
                    "d2_title": str(d2.get("title", "") or "")[:100],
                    "cfr_overlap": cfr_n,
                    "cfr_match": cfr_match,
                    "textual_ref_match": tr_match,
                    "evidence": "; ".join(evidence),
                }

    pairs_list = list(pairs.values())
    pairs_list.sort(key=lambda p: (p["d1_docket"], p["d2_docket"]))
    extreme_dockets = set()
    for p in pairs_list:
        extreme_dockets.add(p["d1_docket"])
        extreme_dockets.add(p["d2_docket"])
    return pairs_list, extreme_dockets


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

OUT_COLS = [
    "docket_id", "year_cluster", "proposed_year", "final_year",
    "n_comments", "n_attach_only", "pct_attach_only", "title",
    "selection_method",
]


def write_anchor_csv(df: pd.DataFrame, path: Path,
                     selection_method: str | None = None) -> None:
    out = df.copy()
    if selection_method is not None and "selection_method" not in out.columns:
        out["selection_method"] = selection_method
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = None
    out[OUT_COLS].to_csv(path, index=False)


def write_selection_log(
    eligible: pd.DataFrame,
    cand: pd.DataFrame,
    primary_log: list[dict],
    extreme_pairs: list[dict],
    extreme_dockets: set[str],
    locked_with_extreme: pd.DataFrame,
    alt1_log: list[dict],
    alt2_top: pd.DataFrame,
    path: Path,
) -> None:
    L: list[str] = []
    L.append("# Anchor Selection Log — Option 2 stratified attachment retrieval\n")
    L.append(f"_Generated {pd.Timestamp.utcnow().isoformat(timespec='seconds')}_\n")

    L.append("## Eligibility filter\n")
    L.append("Single rule. **No text-content threshold.** A `>=30% text-content`")
    L.append("filter was considered in an earlier draft and withdrawn — it would")
    L.append("have collapsed the 2010-2012 cluster to a single Pesticide Petitions")
    L.append("docket (`EPA-HQ-OPP-2010-0889`), and filtering on text-content while")
    L.append("stratifying on era (which correlates with attachment-only rate) is")
    L.append("methodologically tense. Text-content rate is treated as a downstream")
    L.append("analysis variable rather than a sample-frame filter.\n")
    L.append("Active criteria:\n")
    L.append("- `n_comments >= 500`")
    L.append("  - Rigor: published-and-cited (Libgober & Rashin 2023, high-stakes-volume cutoff).")
    L.append("- Docket has at least one Proposed Rule AND one Final Rule (or Withdrawal) posted 2010-2022.")
    L.append("  - Rigor: published-and-cited (standard scope filter for rulemaking outcome studies).\n")
    L.append(f"**Total dockets observed:** {len(cand):,}")
    L.append(f"**Eligible after filter:** {len(eligible):,}\n")

    L.append("Per-cluster eligible counts:\n")
    L.append("| Cluster | Eligible | Total comments | Min n | Median | Max n |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for cluster_name, _, _ in YEAR_CLUSTERS:
        block = eligible[eligible["year_cluster"] == cluster_name]
        if block.empty:
            L.append(f"| {cluster_name} | 0 | — | — | — | — |")
            continue
        L.append(
            f"| {cluster_name} | {len(block):,} | "
            f"{int(block['n_comments'].sum()):,} | "
            f"{int(block['n_comments'].min()):,} | "
            f"{int(block['n_comments'].median()):,} | "
            f"{int(block['n_comments'].max()):,} |"
        )
    L.append("")

    L.append("## Step 1 — Stratified random sample (PRIMARY)\n")
    L.append(f"- Seed: `{PRIMARY_SEED}`")
    L.append(f"- Strata: 4 year clusters x 3 within-cluster volume tertiles = 12")
    L.append(f"- N per stratum: {N_PER_STRATUM}")
    L.append("- Tertile boundaries computed within each cluster (33rd, 67th percentiles, "
             "method='lower') so volume distribution shifts across years don't collapse strata.")
    L.append("- Rigor: published-and-cited (Lohr 2009; Sage Encyclopedia of Educational Research).\n")
    L.append("Per-stratum draws:\n")
    L.append("| Cluster | Tertile | Pool size | Tertile boundaries | Drawn | Shortfall | Picks |")
    L.append("|---|---|---:|---|---:|---:|---|")
    for r in primary_log:
        b = r["boundaries"]
        bnd = "—" if b == (None, None) else f"<{b[0]:,} | {b[0]:,}-{b[1]-1:,} | >={b[1]:,}"
        picks = ", ".join(f"`{d}`" for d in r["picks"]) or "—"
        L.append(f"| {r['cluster']} | {r['tertile']} | {r['pool_size']} | {bnd} | "
                 f"{r['drawn']} | {r['shortfall']} | {picks} |")
    total_drawn = sum(r["drawn"] for r in primary_log)
    total_shortfall = sum(r["shortfall"] for r in primary_log)
    L.append(f"\n**Total drawn:** {total_drawn} (target {N_PER_STRATUM*N_TERTILES*len(YEAR_CLUSTERS)}). "
             f"Shortfall: {total_shortfall}.\n")

    L.append("## Step 2 — Extreme-case additions (Obama-Trump regulatory-reversal pairs)\n")
    L.append("Detected pairs where:\n")
    L.append(f"- D1 has Final Rule posted {OBAMA_FINAL_YEARS[0]}-{OBAMA_FINAL_YEARS[1]} (Obama-era)")
    L.append(f"- D2 has Proposed Rule posted {TRUMP_PROPOSED_YEARS[0]}-{TRUMP_PROPOSED_YEARS[1]} (Trump-era)")
    L.append(f"- AND (cfr_overlap >= {CFR_OVERLAP_MIN} OR textual_reference matches)")
    L.append("- Both halves of every detected pair are added; duplicates against Step-1 sample are dropped.")
    L.append("- Rigor: novel-but-defensible (Klotz 2008, Gerring 2007; specific operationalization is ours).\n")

    L.append("**Criteria refinements applied 2026-05-06 after pilot run:**\n")
    L.append("- *CFR threshold raised from 2 to 3.* Parts 60 (NSPS) and 63 (NESHAP) are parent CFR locations for")
    L.append("  hundreds of unrelated EPA rules. Two-part overlap (e.g., `(40,60)+(40,63)`) is weak evidence of")
    L.append("  regulatory relatedness; >=3 targets rule-subsystem-level overlap. Pilot run had 6 pairs that")
    L.append("  matched only on generic Part 60+63 co-occurrence and had no substantive policy connection.")
    L.append("  Rigor: novel-but-defensible.")
    L.append("- *Reversal-keyword vocabulary expanded* from {repeal, rescind, withdraw} to the canonical")
    L.append("  administrative-law reversal-action verb set, including recodify, supersede, reconsider, vacate")
    L.append("  (and inflectional variants). Under-specification fix to canonical category, not post-hoc")
    L.append("  adjustment to capture any specific pair. Vocabulary corresponds to standard APA-section")
    L.append("  regulatory-action verbs. WOTUS-recodification was the surface defect that exposed the")
    L.append("  under-specification (its Trump 2017 proposed rule used `Recodification`, which the original")
    L.append("  three-word set missed).")
    L.append("  Rigor: published-but-adapted.\n")
    L.append("Per the methodology rigor stopping rule (Bruce 2026-05-06): one more criteria revision is")
    L.append("permissible if a specific surface defect appears in this run; beyond that, the criteria are")
    L.append("locked as-is.\n")
    if not extreme_pairs:
        L.append("**No pairs detected.** This is unexpected — verify the criteria.\n")
    else:
        L.append(f"**{len(extreme_pairs)} pair(s) detected**, "
                 f"{len(extreme_dockets)} unique docket(s):\n")
        L.append("| D1 (Obama Final) | D2 (Trump Proposed) | Evidence |")
        L.append("|---|---|---|")
        for p in extreme_pairs:
            L.append(f"| `{p['d1_docket']}` | `{p['d2_docket']}` | {p['evidence']} |")
        L.append("")
        L.append("Pair detail:\n")
        for p in extreme_pairs:
            L.append(f"- **`{p['d1_docket']}` -> `{p['d2_docket']}`**")
            L.append(f"  - D1 ({p['d1_doc_id']}): _{p['d1_title']}_  FR: {p['d1_fr'] or '—'}")
            L.append(f"  - D2 ({p['d2_doc_id']}): _{p['d2_title']}_")
            L.append(f"  - Evidence: {p['evidence']}")
        L.append("")

    L.append("## Composite locked anchor list\n")
    n_random = (locked_with_extreme["selection_method"]
                .str.contains("stratified_random", na=False).sum())
    n_extreme_only = (locked_with_extreme["selection_method"]
                      == "extreme_case").sum()
    n_both = (locked_with_extreme["selection_method"]
              == "stratified_random+extreme_case").sum()
    L.append(f"- Stratified-only: {n_random - n_both}")
    L.append(f"- Extreme-only: {n_extreme_only}")
    L.append(f"- Both (stratified + also extreme): {n_both}")
    L.append(f"- **Total unique anchor dockets: {len(locked_with_extreme)}**\n")
    L.append("| docket_id | year_cluster | proposed | final | n_comments | %attach | method |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    for _, r in locked_with_extreme.sort_values(
        ["year_cluster", "n_comments"], ascending=[True, False]
    ).iterrows():
        py = "" if pd.isna(r["proposed_year"]) else int(r["proposed_year"])
        fy = "" if pd.isna(r["final_year"]) else int(r["final_year"])
        L.append(
            f"| `{r['docket_id']}` | {r['year_cluster']} | {py} | {fy} | "
            f"{int(r['n_comments']):,} | {r['pct_attach_only']:.0f}% | "
            f"{r['selection_method']} |"
        )
    L.append("")

    L.append("## Step 3 — Robustness samples\n")
    L.append(f"- **alt-1**: stratified random with seed `{ALT1_SEED}`, same strata, same n_per_stratum. "
             "Drawn anchors written to `anchor_rules_alt1_random_seed2.csv`.")
    n_alt1_drawn = sum(r["drawn"] for r in alt1_log)
    L.append(f"  - Total drawn: {n_alt1_drawn}.")
    L.append(f"- **alt-2**: top-25 dockets by `n_comments` (Libgober & Rashin 2023 style). "
             "Written to `anchor_rules_alt2_top_volume.csv`.")
    L.append(f"  - Volume range: {int(alt2_top['n_comments'].min()):,} – {int(alt2_top['n_comments'].max()):,}\n")

    L.append("## Citations\n")
    L.append("- Lohr, S. L. (2009). _Sampling: Design and Analysis_ (2nd ed.). Brooks/Cole. — stratified sampling")
    L.append("- _Sage Encyclopedia of Educational Research, Measurement, and Evaluation_ — strata should minimize within-stratum variance")
    L.append("- Klotz, A. (2008). Case selection. In _Qualitative Methods in International Relations_ (pp. 43-58). — extreme-case methodology")
    L.append("- Gerring, J. (2007). _Case Study Research: Principles and Practices_. Cambridge University Press. — extreme-case methodology")
    L.append("- Libgober, B., & Rashin, S. (2023). [pending Columbia retrieval]. — high-stakes comment-volume framing; top-volume robustness specification")

    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("[load] documents and comments parquet ...")
    docs = load_all_parquet(DOCS_DIR)
    comments = load_all_parquet(COMMENTS_DIR)
    print(f"  documents: {len(docs):,}  comments: {len(comments):,}")

    print("[aggregate] per-docket candidate frame ...")
    cand = build_candidates(comments, docs)
    cand_sorted = cand.sort_values("n_comments", ascending=False)
    cand_sorted.to_csv(OUT_DIR / "anchor_rule_candidates.csv", index=False)

    eligible = cand[
        (cand["n_comments"] >= 500)
        & (cand["has_proposed_and_final"])
        & (cand["year_cluster"].isin([n for n, _, _ in YEAR_CLUSTERS]))
    ].copy().sort_values("n_comments", ascending=False).reset_index(drop=True)
    print(f"[eligible] {len(eligible):,} dockets after filter "
          "(>=500 comments + Proposed+Final, no text-content threshold)")

    print(f"[step 1] stratified random sample (seed={PRIMARY_SEED}) ...")
    primary, primary_log = stratified_random_sample(eligible, PRIMARY_SEED)
    primary["selection_method"] = "stratified_random"
    print(f"  drew {len(primary)} dockets")

    print("[step 2] extreme-case detection ...")
    extreme_pairs, extreme_dockets = detect_extreme_cases(eligible, docs)
    print(f"  {len(extreme_pairs)} pair(s); {len(extreme_dockets)} unique extreme dockets")
    for p in extreme_pairs:
        print(f"    {p['d1_docket']} -> {p['d2_docket']}  ({p['evidence']})")

    extreme_only = eligible[
        eligible["docket_id"].isin(extreme_dockets)
        & ~eligible["docket_id"].isin(primary["docket_id"])
    ].copy()
    extreme_only["selection_method"] = "extreme_case"

    primary_marked = primary.copy()
    primary_marked.loc[
        primary_marked["docket_id"].isin(extreme_dockets), "selection_method"
    ] = "stratified_random+extreme_case"

    locked = pd.concat([primary_marked, extreme_only], ignore_index=True)
    locked = locked.drop_duplicates("docket_id").sort_values(
        ["year_cluster", "n_comments"], ascending=[True, False]
    ).reset_index(drop=True)

    write_anchor_csv(locked, OUT_DIR / "anchor_rules_locked.csv")
    print(f"[write] anchor_rules_locked.csv  ({len(locked)} dockets)")

    print(f"[step 3a] alt-1 stratified random (seed={ALT1_SEED}) ...")
    alt1, alt1_log = stratified_random_sample(eligible, ALT1_SEED)
    alt1["selection_method"] = "alt1_stratified_seed2"
    write_anchor_csv(alt1, OUT_DIR / "anchor_rules_alt1_random_seed2.csv")
    print(f"  alt-1: {len(alt1)} dockets")

    print("[step 3b] alt-2 top-25 by volume ...")
    alt2 = eligible.head(25).copy()
    alt2["selection_method"] = "alt2_top_volume"
    write_anchor_csv(alt2, OUT_DIR / "anchor_rules_alt2_top_volume.csv")
    print(f"  alt-2: {len(alt2)} dockets")

    write_selection_log(
        eligible=eligible, cand=cand,
        primary_log=primary_log,
        extreme_pairs=extreme_pairs,
        extreme_dockets=extreme_dockets,
        locked_with_extreme=locked,
        alt1_log=alt1_log,
        alt2_top=alt2,
        path=OUT_DIR / "anchor_selection_log.md",
    )
    print(f"[write] anchor_selection_log.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
