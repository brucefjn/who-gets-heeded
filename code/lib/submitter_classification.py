"""
submitter_classification.py — canonical Title-field → submitter_kind
classifier for the regulations.gov pipeline.

Why this lives in a shared module:
  The regulations.gov bulk export universally returns an empty
  `Organization Name` field (PII-redacted at source — see PROJECT_FACTS
  §2), so the analytic pipeline reconstructs the submitter type from
  the comment Title field via a small regex cascade. The same cascade
  is needed in two places (the stratified sample step and the Stage 4
  obligation-level aggregator), so a single source of truth eliminates
  drift between them.

Pure relocation — patterns, priorities, and branch logic are byte-
identical to the original in `code/05_stratified_sample.py`. Any future
refinement is a separate methodological decision (PROJECT_FACTS §2
note: requires both authors' input).
"""
from __future__ import annotations

import re

import pandas as pd


# Heuristics on the Title field. Calibrate as we see edge cases.
_RE_INDIV_NAME = re.compile(r"^comment\s+submitted\s+by\s+", re.I)
_RE_ANON      = re.compile(r"^anonymous\s+public\s+comment\b", re.I)
_RE_ORG_OF    = re.compile(r"^comments?\s+(?:of|from)\s+", re.I)
_RE_TITLE_HAS_ORG = re.compile(
    r"\b(association|coalition|institute|council|federation|society|"
    r"corporation|company|inc\.?|llc|llp|university|center|"
    r"environmental defense|sierra club|nrdc|chamber|federation|"
    r"alliance|department|committee|agency|administration|union|"
    r"bureau|board|consortium|"
    r"executive director|president|vice president|director|"
    r"general counsel|associate general counsel)\b",
    re.I,
)


# Canonical ordering matches the stratification quota dict in
# `code/05_stratified_sample.py` (organizational 40% / individual 40% /
# other 20%). Downstream code can import this tuple to validate that a
# value is one of the canonical kinds without re-listing them.
SUBMITTER_KIND_VALUES = ("organizational", "individual", "other")


def classify_title(title: str) -> str:
    if not title or pd.isna(title):
        return "other"
    s = str(title).strip()
    if _RE_ORG_OF.search(s):
        return "organizational"
    if _RE_INDIV_NAME.search(s):
        # If the title also mentions a clear org/role, count as organizational
        # (e.g. 'Comment submitted by Jane Roe, Executive Director, WMA')
        if _RE_TITLE_HAS_ORG.search(s):
            return "organizational"
        return "individual"
    if _RE_ANON.search(s):
        return "individual"  # anonymous individual
    if _RE_TITLE_HAS_ORG.search(s):
        return "organizational"
    return "other"
