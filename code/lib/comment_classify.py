"""
Shared comment-classification helpers for the regulations.gov pipeline.

The single source of truth for the attachment-only placeholder matcher.
Both 01_ingest_regulations_csv.py and 02_attachment_rate_diagnostic.py
import from here.

A comment is *attachment-only* iff:
  - the normalized comment text is empty, OR
  - the normalized text matches one of the placeholder regex patterns below
    ("see attached", "see attached file(s)", "see attachment for comment",
     "comments are attached", "see pdf for comment", etc.)

Normalization (deterministic, applied before matching):
  1. lowercase
  2. "(s)" -> "s",   "(es)" -> "es"      (handles "file(s)" -> "files")
  3. collapse runs of whitespace, strip
  4. strip leading "please " / "kindly "
  5. strip stray surrounding/trailing punctuation/quotes

Tested in code/02_attachment_rate_diagnostic.py.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd  # only used for pd.isna check on float NaN

# Building blocks for the placeholder regex.
_NOUN = (
    r"(?:attached|attachment|attachments|file|files|document|documents|"
    r"letter|letters|comments?|submission|submissions|pdf|response|response\s+letter|"
    r"email|emails|memo|memos|message|messages|materials?|paper|papers)"
)
_DET = r"(?:the|my|our|this|enclosed|attached|uploaded|provided|following|below)"
_VERB = r"(?:see|refer\s+to|view|read|review|find|please\s+see|please\s+refer\s+to|please\s+find)"

# Bare-noun matches must be unambiguously placeholder-y. "comment"/"letter"
# alone are too ambiguous to count as attachment-only — treated as content.
_PLACEHOLDER_BARE = (
    r"(?:attached|attachment|attachments|file|files|document|documents|"
    r"pdf|submission|submissions|materials)"
)

PLACEHOLDER_PATTERNS = [
    # "see [the/my] attached", "see attached file", "see attached comments", etc.
    rf"^{_VERB}(?:\s+{_DET})*\s+{_NOUN}(?:\s+{_NOUN}|\s+for\s+{_NOUN}|\s+of\s+\w+(?:\s+\w+){{0,5}})*$",
    # "attached comments", "attached letter", "the attached document"
    rf"^{_DET}\s+{_NOUN}(?:\s+{_NOUN})*$",
    # bare placeholder noun: "attached", "attachment", "documents", "pdf"
    rf"^{_PLACEHOLDER_BARE}$",
    # "comments are attached", "letter attached", "see comments attached"
    rf"^(?:{_VERB}\s+)?{_NOUN}(?:\s+{_NOUN})*\s+(?:is|are)?\s*attached(?:\s+\w+(?:\s+\w+){{0,5}})?$",
    # "comments of <Org> attached"
    rf"^{_NOUN}(?:\s+{_NOUN})*\s+of\s+\w+(?:\s+\w+){{0,8}}\s+(?:is|are)?\s*attached$",
    # "attached please find <stuff>"
    rf"^attached\s+(?:please\s+)?(?:find|are|is)?(?:\s+(?:the|my|our))?(?:\s+{_NOUN})?(?:\s+\w+(?:\s+\w+){{0,8}})?$",
    # "see attached letter for comments", "see pdf for comment"
    rf"^{_VERB}(?:\s+{_DET})*\s+{_NOUN}\s+for\s+{_NOUN}$",
    # "no content here" placeholders
    r"^(?:n\s*/\s*a|none|no\s+comment|no\s+comments)$",
]

PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS))


def normalize_for_match(text: Any) -> str:
    """Deterministic normalization applied before placeholder matching."""
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(text).lower()
    # Normalize "(s)" / "(es)" before stripping parens.
    s = re.sub(r"\(\s*s\s*\)", "s", s)
    s = re.sub(r"\(\s*es\s*\)", "es", s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    # Strip leading politeness.
    s = re.sub(r"^(?:please|kindly)\s+", "", s)
    # Strip stray surrounding/trailing punctuation/quotes.
    s = re.sub(r'^[.,;:!\-_"\'“”‘’()\[\]]+', "", s)
    s = re.sub(r'[.,;:!\-_"\'“”‘’()\[\]]+$', "", s).strip()
    return s


def is_attachment_only(text: Any) -> bool:
    """True if the comment text is empty or a known attachment-only placeholder."""
    n = normalize_for_match(text)
    if n == "":
        return True
    return bool(PLACEHOLDER_RE.match(n))
