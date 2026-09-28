"""
path_a_obligation_heuristic.py — Stage 1a of Path A obligation extraction.

Per notes/2026-05-09_path_a_stage1_obligation_extraction_v2.md.

Hybrid heuristic + spaCy dependency-parse extractor for candidate deontic
obligations from Federal Register binding regulatory text. Output flows
to Stage 1b (gpt-5 verification + structured correction).

Pipeline within this module:
  1. Pre-process: split preamble from binding text.
  2. Walk binding text by amendatory blocks; tag amendatory_action.
  3. Sentence-segment per amendatory block; track CFR Part / Section anchors.
  4. Per sentence: detect modal anchor (strong / permissive / no_modal_implicit
     fallback). If modal present, run dependency parse.
  5. Extract subject (nsubj OR nsubjpass with agent recovery), modal,
     action, object. Tag passive_voice + conditional fields.
  6. Emit candidate record.

Usage:
    from code.lib.path_a_obligation_heuristic import extract_candidates_from_file
    candidates = extract_candidates_from_file(
        text_path="data/raw/federal_register/EPA-HQ-OAR-2018-0775_final_2019-11653.txt",
        docket_id="EPA-HQ-OAR-2018-0775",
        rule_type="final",
    )

CLI:
    python -m code.lib.path_a_obligation_heuristic \\
        --text-path data/raw/federal_register/<file>.txt \\
        --docket EPA-HQ-OAR-2018-0775 \\
        --rule-type final \\
        --out data/processed/path_a_obligations_candidates_<docket>_<rule>.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# spaCy model loader (lazy — module import shouldn't pay the load cost)
# ---------------------------------------------------------------------------
_NLP = None


def _load_nlp():
    """Lazy load spaCy en_core_web_lg. Falls back to en_core_web_sm if lg
    isn't installed, with a stderr warning. Use lg in production."""
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy
    try:
        _NLP = spacy.load("en_core_web_lg")
    except OSError:
        try:
            _NLP = spacy.load("en_core_web_sm")
            print("[path_a_heuristic] WARNING: en_core_web_lg not installed; "
                  "falling back to en_core_web_sm. For production use:\n"
                  "  python -m spacy download en_core_web_lg",
                  file=sys.stderr)
        except OSError:
            raise RuntimeError(
                "Neither en_core_web_lg nor en_core_web_sm is installed. "
                "Install with: python -m spacy download en_core_web_lg"
            )
    return _NLP


# ---------------------------------------------------------------------------
# Constants — modal lists, amendatory markers, conditional triggers
# Per v2 spec § "Stage 1a heuristic + spaCy pre-pass"
# ---------------------------------------------------------------------------

# Strong modals (binding requirement)
STRONG_MODALS = {
    "must", "must not",
    "shall", "shall not",
    "may not",
    "is required to", "are required to",
    "is prohibited from", "are prohibited from",
    "no person shall", "no person may",
    "no operator shall", "no operator may",
    "no [entity] may", "no [entity] shall",
}
# We compile a regex that catches single-token strongs + multi-word phrases
STRONG_MODAL_RE = re.compile(
    r"\b(?:"
    r"must(?:\s+not)?"
    r"|shall(?:\s+not)?"
    r"|may\s+not"
    r"|(?:is|are)\s+required\s+to"
    r"|(?:is|are)\s+prohibited\s+from"
    r"|no\s+(?:person|operator|owner|entity|facility)\s+(?:shall|may)"
    r")\b",
    re.IGNORECASE,
)

# Permissive modals (binding statement that actor IS allowed to do X)
PERMISSIVE_MODAL_RE = re.compile(
    r"\b(?:"
    r"may"  # bare "may" — but need to disambiguate from epistemic uses
    r"|(?:is|are)\s+permitted\s+to"
    r"|(?:is|are)\s+authorized\s+to"
    r")\b",
    re.IGNORECASE,
)

# Sentence-leading conditionals
LEADING_CONDITIONAL_RE = re.compile(
    r"^\s*(?:If|When|Upon|Where|Whenever)\b[^.]+?,",
    re.IGNORECASE | re.MULTILINE,
)

# Mid-sentence conditional markers (Major 2 fix)
MID_CONDITIONAL_RE = re.compile(
    r",\s*(?:provided\s+that|unless|except\s+as|if\s+applicable)\b",
    re.IGNORECASE,
)
# "subject to" — disambiguate: cross-reference if followed by section-citation
SUBJECT_TO_CITATION_RE = re.compile(
    r"\bsubject\s+to\s+(?:§|paragraph|section)",
    re.IGNORECASE,
)
SUBJECT_TO_RE = re.compile(r",\s*subject\s+to\b", re.IGNORECASE)


# Amendatory block markers — order matters (longer/more-specific first).
#
# v2.1 (2026-05-09 post-Bruce-eyeball): dropped bare `is\s+added` and
# `is\s+amended` generic catch-alls because they fire on ordinary English usages
# in preamble text (e.g., "ethanol is added to gasoline", "Section 80 is
# amended by EPA's reading"). Plus, the long forms (e.g., "is added to read
# as follows") cover all real CFR amendatory cases — the bare forms only
# created false positives + duplicates.
AMENDATORY_PATTERNS = [
    (re.compile(r"is\s+revised\s+to\s+read\s+as\s+follows", re.IGNORECASE), "revise"),
    (re.compile(r"is\s+amended\s+to\s+read\s+as\s+follows", re.IGNORECASE), "revise"),
    (re.compile(r"is\s+added\s+to\s+read\s+as\s+follows", re.IGNORECASE), "add"),
    (re.compile(r"is\s+amended\s+by\s+adding\s+(?:paragraph|subparagraph|section)\s*\([a-z0-9ivx]+\)\s+to\s+read\s+as\s+follows", re.IGNORECASE), "add"),
    (re.compile(r"is\s+amended\s+by\s+adding\s+(?:paragraphs?|subparagraphs?|sections?)\s*\([a-z0-9ivx]+\)(?:\s*(?:through|to|and|,)\s*\([a-z0-9ivx]+\))*\s+to\s+read\s+as\s+follows", re.IGNORECASE), "add"),
    (re.compile(r"is\s+amended\s+by\s+adding\s+the\s+following", re.IGNORECASE), "add"),
    (re.compile(r"is\s+amended\s+by\s+revising\s+paragraph\s*\([a-z0-9ivx]+\)\s+to\s+read\s+as\s+follows", re.IGNORECASE), "revise"),
    (re.compile(r"is\s+amended\s+by\s+(?:adding|revising|removing|redesignating)", re.IGNORECASE), "revise"),
    # Colon-instruction form: "Section X.Y is amended by:" with enumerated
    # sub-instructions on subsequent lines (very common in EPA FR docs —
    # § 80.1451, § 80.1452, § 80.1503 in the 2018-0775 RFS rule all use this).
    (re.compile(r"is\s+amended\s+by\s*:", re.IGNORECASE), "revise"),
    (re.compile(r"paragraph\s*\([a-z0-9ivx]+\)\s+is\s+removed", re.IGNORECASE), "remove"),
    (re.compile(r"is\s+removed\s+and\s+reserved", re.IGNORECASE), "remove"),
    (re.compile(r"is\s+removed", re.IGNORECASE), "remove"),
    (re.compile(r"is\s+redesignated\s+as", re.IGNORECASE), "redesignate"),
    (re.compile(r"is\s+replaced\s+by", re.IGNORECASE), "replace"),
    (re.compile(r"the\s+heading\s+is\s+revised\s+to\s+read\s+as\s+follows", re.IGNORECASE), "revise"),
]

# Binding-text section start markers.
#
# v2.1 (2026-05-09 post-Bruce-eyeball): "List of Subjects in 40 CFR Part X"
# was previously included as a binding-start anchor, but in real FR docs it
# precedes the EPA Administrator's signature block + (in some rules)
# verbatim REPRODUCTION of older interpretive rulings (e.g., the 1981
# "Substantially Similar" interpretive ruling reproduced in the 2019 RFS
# rule). That entire region between "List of Subjects" and "For the
# reasons set forth in the preamble" is end-of-preamble, NOT binding text.
# Restrict binding detection to the canonical "For the reasons …"
# amendatory header.
BINDING_START_PATTERNS = [
    re.compile(r"For\s+the\s+reasons\s+(?:set\s+forth|stated)\s+in\s+the\s+preamble", re.IGNORECASE),
]

# Section-being-amended anchor — matches the section-header line that
# precedes each amendatory instruction. This is what cfr_section should
# attribute to (NOT internal cross-references like `§ 86.113` that the
# walk-back-to-nearest-cite heuristic incorrectly grabs).
#
# Forms encountered in real FR docs (all match):
#   `Section 80.1464 is amended by adding paragraphs (a)(4) through (6)…`
#   `Section 80.1402 is added to read as follows:`
#   `Section 80.1503 is amended by:`
AMENDATORY_SECTION_ANCHOR_RE = re.compile(
    r"Section\s+(\d{1,3}\.\d{1,4}(?:\([a-z0-9ivx]+\))*)"
    r"\s+is\s+(?:amended|revised|added|removed|redesignated|replaced)",
    re.IGNORECASE,
)
# Also: section-heading reproduction line (`Sec.  80.1454  What records …`)
# which appears just after the amendatory instruction.
SEC_HEADING_REPRO_RE = re.compile(
    r"Sec\.\s+(\d{1,3}\.\d{1,4}(?:\([a-z0-9ivx]+\))*)\s+",
    re.IGNORECASE,
)

# CFR section + part patterns
CFR_PART_HEADER_RE = re.compile(
    r"^\s*(?:PART|Part)\s+(\d{1,3})\s*[—–-]",
    re.MULTILINE,
)
CFR_SECTION_RE = re.compile(
    r"§\s*(\d{1,3}\.\d{1,4}(?:\([a-z0-9ivx]+\))*)",
)
# "40 C.F.R. § 63.6590" full-cite pattern (Bruce v2.2 dual-pattern fix)
CFR_FULL_CITE_RE = re.compile(
    r"(\d{1,2})\s*C\.?F\.?R\.?\s*(?:§|Part|part|section)?\s*(\d{1,3}\.\d{1,4}(?:\([a-z0-9ivx]+\))*)",
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass
class CandidateObligation:
    """Stage 1a candidate. Fields populated by heuristic + dep-parse;
    Stage 1b LLM verification corrects + finalizes."""
    obligation_idx: int
    docket_id: str
    rule_type: str        # 'proposed' | 'final'
    cfr_part: Optional[str]
    cfr_section: Optional[str]
    subject: str          # NP or "<implicit>" if passive without agent
    modal: str            # token from STRONG/PERMISSIVE_MODAL_RE
    modal_strength: str   # 'strong' | 'permissive'
    action: str           # head verb + immediate VP children
    object: str           # direct-object NP, "" if absent
    amendatory_action: str   # 'add' | 'revise' | 'remove' | 'replace' | 'redesignate' | 'none'
    is_conditional: bool
    condition_text: Optional[str]
    passive_voice: bool
    obligation_text: str  # verbatim sentence
    char_offset_start: int
    char_offset_end: int


# ---------------------------------------------------------------------------
# Pre-processing — split preamble vs binding text
# ---------------------------------------------------------------------------
def split_preamble_binding(text: str) -> tuple[str, str]:
    """Return (preamble_text, binding_text). If no clear binding-text marker,
    fall back to last 40% of doc (heuristic).

    Caller should hand-validate the split on each rule before relying on it.
    """
    earliest_match = None
    for pat in BINDING_START_PATTERNS:
        m = pat.search(text)
        if m and (earliest_match is None or m.start() < earliest_match):
            earliest_match = m.start()
    if earliest_match is None:
        # Fallback: last 40%
        cutoff = int(0.6 * len(text))
        return text[:cutoff], text[cutoff:]
    return text[:earliest_match], text[earliest_match:]


# ---------------------------------------------------------------------------
# Amendatory-block detection
# ---------------------------------------------------------------------------
def find_amendatory_blocks(binding_text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_offset, end_offset, action) tuples.
    end_offset is the start of the next amendatory block, or end of text.
    Per v2 spec: amendatory_action enum {add, revise, remove, replace, redesignate, none}.

    v2.1 (2026-05-09 post-Bruce-eyeball): dedupe overlapping matches —
    multiple AMENDATORY_PATTERNS regexes may match the same physical span
    (e.g., "is added to read as follows" matches both the long-specific
    pattern and would have matched the dropped `is added` generic). When
    matches overlap (within 200 chars), keep only the first one.
    """
    raw_matches = []
    for pat, action in AMENDATORY_PATTERNS:
        for m in pat.finditer(binding_text):
            raw_matches.append((m.start(), m.end(), action))
    # Sort by (start ascending, end DESCENDING) so the LONGEST match wins
    # when multiple patterns match the same position. This is critical:
    # specific patterns ("is amended by adding paragraph (c) to read as
    # follows" → "add") must be preferred over broader ones ("is amended
    # by adding" → "revise") when they overlap.
    raw_matches.sort(key=lambda x: (x[0], -x[1]))

    # Dedupe: when two matches start within 200 chars of each other, keep
    # the first (which is now also the longest, hence the most-specific).
    matches: list[tuple[int, str]] = []
    last_kept_end: Optional[int] = None
    for start, end, action in raw_matches:
        if last_kept_end is not None and start < last_kept_end + 200:
            continue
        matches.append((start, action))
        last_kept_end = end

    if not matches:
        return [(0, len(binding_text), "none")]
    blocks = []
    for i, (start, action) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(binding_text)
        blocks.append((start, end, action))
    # Prepend a "none" block for any text before the first amendatory marker
    if matches[0][0] > 0:
        blocks.insert(0, (0, matches[0][0], "none"))
    return blocks


# ---------------------------------------------------------------------------
# CFR Part / Section anchor walker
# ---------------------------------------------------------------------------
def _last_cfr_anchor_before(text: str, offset: int) -> tuple[Optional[str], Optional[str]]:
    """Walk backward from offset; find:
      - cfr_part from nearest preceding `PART X — TITLE` header
      - cfr_section from nearest preceding `Section X.Y is amended/revised/
        added/removed` anchor (NOT from internal `§` cross-references — that
        walk-back grabbed wrong sections per Bruce 2026-05-09 audit, e.g.,
        attributing § 80.1402 candidates to § 86.113 because § 86.113 was
        cited inline in the new section's text).

    v2.1 (2026-05-09): rewrote section attribution from walk-back-to-nearest-
    `§`-cite to walk-back-to-nearest-amendatory-section-header. Falls back
    to `Sec. X.Y` heading reproduction (which appears immediately after each
    amendatory instruction). Internal `§ X.Y` cross-references are no longer
    used to attribute the candidate to a section.
    """
    head = text[:offset]

    # PART number: nearest preceding `PART X — TITLE` header. Last-match wins.
    part_match = None
    for m in CFR_PART_HEADER_RE.finditer(head):
        part_match = m
    cfr_part = part_match.group(1) if part_match else None

    # Section being amended: nearest preceding "Section X.Y is amended/..." anchor.
    section_anchor = None
    for m in AMENDATORY_SECTION_ANCHOR_RE.finditer(head):
        section_anchor = m
    if section_anchor is not None:
        return cfr_part, section_anchor.group(1)

    # Fallback: nearest preceding "Sec. X.Y" heading reproduction line.
    sec_heading = None
    for m in SEC_HEADING_REPRO_RE.finditer(head):
        sec_heading = m
    if sec_heading is not None:
        return cfr_part, sec_heading.group(1)

    # No amendatory anchor found — return (cfr_part, None). Don't fall back
    # to internal `§` cross-references (the v1 bug).
    return cfr_part, None


# ---------------------------------------------------------------------------
# Modal detection per sentence
# ---------------------------------------------------------------------------
def detect_modal(sentence: str) -> tuple[Optional[str], Optional[str]]:
    """Return (modal_text, strength) or (None, None) if no modal.
    Strong takes precedence over permissive."""
    m = STRONG_MODAL_RE.search(sentence)
    if m:
        return (m.group(0).lower().strip(), "strong")
    m = PERMISSIVE_MODAL_RE.search(sentence)
    if m:
        modal = m.group(0).lower().strip()
        # Disambiguate bare "may": skip if epistemic ("EPA may believe...")
        if modal == "may":
            # Rough heuristic: epistemic "may" usually after subject-verb
            # patterns like "EPA may believe", "we may consider".
            # If the sentence STARTS with subject-only then "may", more
            # likely deontic. Lazy filter — let LLM verification decide.
            pass
        return (modal, "permissive")
    return (None, None)


# ---------------------------------------------------------------------------
# Conditional detection per sentence
# ---------------------------------------------------------------------------
def detect_conditional(sentence: str) -> tuple[bool, Optional[str]]:
    """Return (is_conditional, condition_text)."""
    # Sentence-leading
    m = LEADING_CONDITIONAL_RE.match(sentence)
    if m:
        return (True, m.group(0).rstrip(",").strip())
    # Mid/end-sentence: provided that / unless / except as / if applicable
    m = MID_CONDITIONAL_RE.search(sentence)
    if m:
        return (True, sentence[m.start():].strip(", "))
    # "subject to" — only if NOT followed by section-citation
    m = SUBJECT_TO_RE.search(sentence)
    if m:
        # Look ahead — is next ~30 chars a citation?
        tail = sentence[m.start():m.start() + 50]
        if not SUBJECT_TO_CITATION_RE.search(tail):
            return (True, sentence[m.start():].strip(", "))
    return (False, None)


# ---------------------------------------------------------------------------
# spaCy dep-parse to extract subject / action / object (Major 1 + 4 fixes)
# ---------------------------------------------------------------------------
def parse_obligation_structure(sentence: str, modal: str) -> dict:
    """Run spaCy dep-parse and pull subject (nsubj OR nsubjpass+agent OR
    <implicit>), action (verb head + VP), object (dobj NP).
    Returns dict with keys: subject, action, object, passive_voice."""
    nlp = _load_nlp()
    doc = nlp(sentence)

    # Find the modal token (heuristic-matched modal might be multi-word)
    modal_token = None
    modal_first_word = modal.split()[0]
    for tok in doc:
        if tok.text.lower() == modal_first_word.lower():
            modal_token = tok
            break

    if modal_token is None:
        # Fallback: locate first verb and use it
        for tok in doc:
            if tok.pos_ == "VERB":
                modal_token = tok
                break

    if modal_token is None:
        return {"subject": "<unknown>", "action": "<unknown>", "object": "",
                "passive_voice": False}

    # The action verb is the head of the modal (or the modal's head if modal is AUX)
    head = modal_token.head if modal_token.dep_ in ("aux", "auxpass") else modal_token

    # Subject extraction: try nsubj first, fall back to nsubjpass + agent
    subject_tok = None
    passive = False
    for child in head.children:
        if child.dep_ == "nsubj":
            subject_tok = child
            break
    if subject_tok is None:
        for child in head.children:
            if child.dep_ == "nsubjpass":
                subject_tok = child
                passive = True
                break
        # Recover actor from agent ("by the operator")
        if passive:
            agent_actor = None
            for child in head.children:
                if child.dep_ == "agent":
                    # agent typically has a pobj child = real actor
                    for sub in child.children:
                        if sub.dep_ == "pobj":
                            agent_actor = sub
                            break
                    break
            if agent_actor is not None:
                subject_tok = agent_actor

    if subject_tok is None:
        subject = "<implicit>" if passive else "<unknown>"
    else:
        # NP span: include left-children that are det/amod/compound
        subject_span = _np_span(subject_tok)
        subject = subject_span.strip()

    # Action: verb head + adverbial children that immediately follow
    action_tokens = [head]
    for child in head.rights:
        if child.dep_ in ("prt", "advmod"):
            action_tokens.append(child)
        else:
            break
    action = " ".join(t.text for t in action_tokens)

    # Object: direct object NP
    obj_tok = None
    for child in head.children:
        if child.dep_ == "dobj":
            obj_tok = child
            break
    obj = _np_span(obj_tok).strip() if obj_tok is not None else ""

    return {
        "subject": subject,
        "action": action,
        "object": obj,
        "passive_voice": passive,
    }


def _np_span(token) -> str:
    """Reconstruct an NP span around a head token by including its
    left-children (det, amod, compound, poss, nummod) up to the head + any
    'of'/'in' prepositional modifier."""
    if token is None:
        return ""
    nlp_doc = token.doc
    left = token.i
    for child in token.lefts:
        if child.dep_ in ("det", "amod", "compound", "poss", "nummod", "nmod"):
            left = min(left, child.i)
    right = token.i + 1
    # Include immediate post-head prepositional phrase if it's "of/in/under"
    for child in token.rights:
        if child.dep_ == "prep" and child.text.lower() in ("of", "in", "under", "to"):
            # Walk to end of pp
            tail = max((t.i for t in child.subtree), default=child.i)
            right = max(right, tail + 1)
            break
    return nlp_doc[left:right].text


# ---------------------------------------------------------------------------
# Public API — extract candidates from a single FR text file
# ---------------------------------------------------------------------------
def extract_candidates_from_file(
    text_path: str | Path,
    docket_id: str,
    rule_type: str,
) -> list[CandidateObligation]:
    """Extract candidate obligations from one FR text file.
    Returns a list of CandidateObligation records."""
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
    return extract_candidates_from_text(text, docket_id, rule_type)


def extract_candidates_from_text(
    full_text: str,
    docket_id: str,
    rule_type: str,
) -> list[CandidateObligation]:
    """Extract candidate obligations from the full FR document text.
    Splits preamble vs binding text internally."""
    # 1. Split
    preamble, binding = split_preamble_binding(full_text)
    binding_offset_in_doc = len(preamble)

    # 2. Identify amendatory blocks within binding text
    amend_blocks = find_amendatory_blocks(binding)

    # 3. Sentence-segment per amendatory block, walk each
    nlp = _load_nlp()
    candidates: list[CandidateObligation] = []
    obligation_idx = 0

    for block_start, block_end, action in amend_blocks:
        block_text = binding[block_start:block_end]
        # Skip very short blocks (header-only)
        if len(block_text.strip()) < 20:
            continue

        doc = nlp(block_text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) < 10:
                continue

            # Skip likely-non-obligation sentences
            if _is_likely_recital_or_header(sent_text):
                continue

            # Modal detection
            modal, strength = detect_modal(sent_text)
            if modal is None:
                continue

            # Conditional detection
            is_cond, cond_text = detect_conditional(sent_text)

            # Dep-parse
            try:
                parsed = parse_obligation_structure(sent_text, modal)
            except Exception:
                parsed = {"subject": "<parse_error>", "action": modal,
                          "object": "", "passive_voice": False}

            # CFR anchors — walk back from this sentence's offset in binding
            sent_offset_in_binding = block_start + sent.start_char
            cfr_part, cfr_section = _last_cfr_anchor_before(binding, sent_offset_in_binding)

            obligation_idx += 1
            cand = CandidateObligation(
                obligation_idx=obligation_idx,
                docket_id=docket_id,
                rule_type=rule_type,
                cfr_part=cfr_part,
                cfr_section=cfr_section,
                subject=parsed["subject"],
                modal=modal,
                modal_strength=strength,
                action=parsed["action"],
                object=parsed["object"],
                amendatory_action=action,
                is_conditional=is_cond,
                condition_text=cond_text,
                passive_voice=parsed["passive_voice"],
                obligation_text=sent_text[:1500],  # cap for CSV size
                char_offset_start=binding_offset_in_doc + sent_offset_in_binding,
                char_offset_end=binding_offset_in_doc + block_start + sent.end_char,
            )
            candidates.append(cand)

    return candidates


def _is_likely_recital_or_header(sent_text: str) -> bool:
    """Cheap heuristic to filter obvious non-obligations BEFORE dep-parse.
    The LLM verification will catch anything this misses; this is just to
    save spaCy compute on obvious headers and recitals."""
    s = sent_text.strip()
    # Section headers: "§ 63.6590 Standards for ..."
    if re.match(r"^§\s*\d", s):
        # Could be a section heading — usually short, no modal
        if len(s) < 80 and not STRONG_MODAL_RE.search(s):
            return True
    # All-caps headings
    if s.isupper() and len(s) < 100:
        return True
    # "Authority:" line
    if s.lower().startswith("authority:"):
        return True
    # PART headers
    if re.match(r"^(?:PART|Part)\s+\d", s):
        return True
    return False


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def write_candidates_csv(candidates: Iterable[CandidateObligation],
                         out_path: str | Path) -> int:
    """Write candidates to CSV. Returns number of rows written."""
    candidates = list(candidates)
    if not candidates:
        # Still write header
        fieldnames = list(CandidateObligation.__dataclass_fields__.keys())
    else:
        fieldnames = list(asdict(candidates[0]).keys())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow(asdict(c))
    return len(candidates)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--text-path", type=Path, default=None,
                    help="Path to FR text file (required unless --smoke-test)")
    ap.add_argument("--docket", type=str, default=None,
                    help="Docket ID e.g. EPA-HQ-OAR-2018-0775")
    ap.add_argument("--rule-type", choices=["proposed", "final"], default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output candidates CSV path")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run inline smoke test (no inputs needed)")
    args = ap.parse_args()

    if args.smoke_test:
        return _smoke_test()

    # Validate required args for non-smoke-test mode
    missing = [k for k in ("text_path", "docket", "rule_type", "out")
               if getattr(args, k) is None]
    if missing:
        ap.error(f"Missing required arguments for non-smoke-test mode: {missing}")

    cands = extract_candidates_from_file(args.text_path, args.docket, args.rule_type)
    n = write_candidates_csv(cands, args.out)
    print(f"[path_a_heuristic] wrote {n} candidates to {args.out}")
    if n > 0:
        # Quick distribution report
        from collections import Counter
        actions = Counter(c.amendatory_action for c in cands)
        modals = Counter(c.modal_strength for c in cands)
        passive_n = sum(1 for c in cands if c.passive_voice)
        cond_n = sum(1 for c in cands if c.is_conditional)
        print(f"  amendatory_action: {dict(actions)}")
        print(f"  modal_strength:    {dict(modals)}")
        print(f"  passive_voice:     {passive_n}")
        print(f"  is_conditional:    {cond_n}")


SYNTHETIC_TEST_TEXT = """
    SUMMARY: This rule modifies fuel regulations.

    SUPPLEMENTARY INFORMATION:
    Commenters argued that the proposed rule was burdensome.

    For the reasons set forth in the preamble, EPA amends 40 CFR part 80 as follows:

    PART 80 — REGULATION OF FUELS

    Authority: 42 U.S.C. 7401-7671q.

    Section 80.1 is revised to read as follows:

    § 80.1  Scope.

    Each operator shall maintain records for five years.

    Records under § 80.10 must be submitted by January 15.

    No person may distribute fuel that fails to meet the standards in § 80.20.

    Section 80.5 is amended by adding paragraph (c) to read as follows:

    (c) The Administrator may grant an exemption, provided that the operator
    demonstrates compliance with § 80.6.

    Records shall be retained by the facility, except as provided in § 80.7.

    Paragraph (a) of § 80.99 is removed.
    """


def _smoke_test():
    """Smoke test — uses a synthetic FR-like text snippet. If spaCy isn't
    installed, runs a regex-only layer test instead (no dep-parse; subject/
    action/object will be unknown but other fields validate)."""
    try:
        import spacy
        spacy.load("en_core_web_lg")
        spacy_ok = True
    except (ImportError, OSError):
        try:
            import spacy
            spacy.load("en_core_web_sm")
            spacy_ok = True
        except (ImportError, OSError):
            spacy_ok = False

    if not spacy_ok:
        return _smoke_test_regex_only()
    return _smoke_test_full()


def _smoke_test_regex_only():
    """Regex-only validation — runs in any env without spaCy.
    Validates: preamble/binding split, amendatory-block detection,
    modal detection, conditional detection, CFR-anchor walking."""
    print("[smoke-test] spaCy model not available; running REGEX-ONLY layer test")
    print("[smoke-test] (full smoke-test requires `pip install spacy && python -m spacy download en_core_web_lg`)")
    print()

    # Test 1: preamble/binding split
    pre, bind = split_preamble_binding(SYNTHETIC_TEST_TEXT)
    assert "For the reasons set forth" in bind, "Binding split failed"
    assert "Commenters argued" in pre, "Preamble split failed"
    print(f"  ✓ preamble/binding split: preamble={len(pre)}c, binding={len(bind)}c")

    # Test 2: amendatory blocks
    blocks = find_amendatory_blocks(bind)
    actions = [a for _, _, a in blocks]
    print(f"  ✓ amendatory blocks: {len(blocks)} blocks, actions={actions}")
    assert "revise" in actions, "Should detect 'is revised to read as follows'"
    assert "add" in actions, "Should detect 'is amended by adding paragraph'"
    assert "remove" in actions, "Should detect 'is removed'"

    # Test 3: modal detection on synthetic sentences
    test_sentences = [
        ("Each operator shall maintain records for five years.", "shall", "strong"),
        ("Records under § 80.10 must be submitted by January 15.", "must", "strong"),
        ("No person may distribute fuel that fails to meet the standards.", "no person may", "strong"),
        ("The Administrator may grant an exemption.", "may", "permissive"),
        ("Records shall be retained by the facility.", "shall", "strong"),
        ("This is a recital with no obligation.", None, None),
    ]
    for sent, exp_modal, exp_strength in test_sentences:
        modal, strength = detect_modal(sent)
        ok = (modal is None and exp_modal is None) or (
            modal is not None and exp_modal is not None
            and exp_modal in modal and strength == exp_strength
        )
        marker = "✓" if ok else "✗"
        print(f"  {marker} modal: '{sent[:50]}...' → modal={modal} ({strength})  "
              f"[expected: {exp_modal} ({exp_strength})]")
        if not ok:
            return 1

    # Test 4: conditional detection
    cond_tests = [
        ("If the operator fails, EPA shall act.", True),
        ("Operators must comply, provided that conditions apply.", True),
        ("Records shall be retained, except as provided in § 80.7.", True),
        ("Each operator shall maintain records.", False),
        ("The agency may act, subject to § 80.10.", False),  # subject-to-citation = NOT conditional
    ]
    for sent, expected in cond_tests:
        is_cond, _ = detect_conditional(sent)
        marker = "✓" if is_cond == expected else "✗"
        print(f"  {marker} conditional: '{sent[:50]}...' → {is_cond}  [expected: {expected}]")
        if is_cond != expected:
            return 1

    # Test 5: CFR Part / Section anchor walking
    pos_after_section_one = bind.find("Each operator shall")
    cfr_part, cfr_section = _last_cfr_anchor_before(bind, pos_after_section_one)
    print(f"  ✓ CFR anchor before 'Each operator shall': part={cfr_part}, section={cfr_section}")
    assert cfr_part == "80", f"Expected Part 80, got {cfr_part}"

    print()
    print("[smoke-test] REGEX-LAYER PASS (5/5 sub-tests). Full dep-parse pending spaCy install.")
    return 0


def _smoke_test_full():
    """Full smoke test with spaCy — extracts candidates end-to-end."""
    print("[smoke-test] Running full heuristic on synthetic FR text...")
    cands = extract_candidates_from_text(SYNTHETIC_TEST_TEXT, "EPA-TEST-0001", "final")
    print(f"[smoke-test] extracted {len(cands)} candidates")
    for c in cands:
        print(f"  #{c.obligation_idx:>2}  Part={c.cfr_part}  §={c.cfr_section}  "
              f"action={c.amendatory_action}  modal={c.modal} ({c.modal_strength})  "
              f"passive={c.passive_voice}  cond={c.is_conditional}")
        print(f"        subject='{c.subject}'  verb='{c.action}'  obj='{c.object}'")
        print(f"        text: {c.obligation_text[:120]}")
    expected_min = 5  # at least: shall maintain, must be submitted, no person may, may grant, shall be retained
    if len(cands) < expected_min:
        print(f"[smoke-test] WARN: expected ≥{expected_min} candidates, got {len(cands)}")
        return 1
    print(f"[smoke-test] FULL PASS ({len(cands)} candidates ≥ {expected_min} expected)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
