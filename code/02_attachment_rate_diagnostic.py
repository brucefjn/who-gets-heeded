"""
02_attachment_rate_diagnostic.py
One-off pre-ingestion diagnostic: across all 39 regulations.gov CSVs, count
total rows, comments, and attachment-only-placeholder comments. Discover
non-obvious placeholder variants for the attachment-only definition.

Output: data/processed/attachment_rate_diagnostic.md
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Use the shared placeholder matcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.comment_classify import is_attachment_only, normalize_for_match  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "regulations_gov"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "attachment_rate_diagnostic.md"

# Columns we care about from each CSV (original casing as in the file headers).
USE_COLS = ["Document Type", "Comment on Document ID", "Comment", "Organization Name"]
CHUNK_SIZE = 50_000

# --- Attachment-only placeholder matcher ---------------------------------

# Patterns matching what a human would call "the comment field has no real
# content, the actual comment is in the attachment". Matched after a strict
# normalization:
#   1. lowercase
#   2. replace runs of whitespace with single space
#   3. strip leading/trailing whitespace
#   4. strip leading "please " (politeness prefix)
#   5. strip trailing punctuation . , ; : ! - and quotes
#
# A comment is attachment-only iff:
#   - the normalized text is empty, OR
#   - the normalized text matches PLACEHOLDER_RE
# Placeholder matcher lives in lib.comment_classify; both 01_ingest and this
# diagnostic import the same definitions.

# --- File processing -----------------------------------------------------

def discover_csvs(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*_full.csv"))


def diagnose_one(path: Path) -> tuple[dict, Counter]:
    """Return (per-file stats, Counter of short-comment normalized values).
    The Counter is for placeholder discovery — only collected for short
    comments (<=80 chars normalized) to keep memory bounded."""
    stem = path.stem.replace("_full", "")
    agency = stem.split("_")[0]
    year = int(stem.split("_")[1])

    total = 0
    n_comments = 0
    n_attach_only = 0
    n_attach_with_org = 0
    n_attach_no_org = 0
    short_counter: Counter[str] = Counter()

    reader = pd.read_csv(
        path,
        encoding="utf-8",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
        on_bad_lines="skip",
        usecols=lambda c: c in USE_COLS,
    )
    for chunk in reader:
        total += len(chunk)
        # Comments by union: link-based OR type=='Public Submission'
        link = chunk.get("Comment on Document ID", pd.Series([pd.NA] * len(chunk))).fillna("").astype(str).str.strip()
        dt = chunk.get("Document Type", pd.Series([pd.NA] * len(chunk))).fillna("").astype(str).str.strip()
        is_comment = (link != "") | (dt.str.lower() == "public submission")
        if not is_comment.any():
            continue
        ccm = chunk.loc[is_comment]
        n_comments += len(ccm)

        comment_text = ccm.get("Comment", pd.Series([pd.NA] * len(ccm)))
        org = ccm.get("Organization Name", pd.Series([pd.NA] * len(ccm))).fillna("").astype(str).str.strip()

        # Vectorized normalization for placeholder match
        attach_mask = comment_text.apply(is_attachment_only)
        n_attach_only += int(attach_mask.sum())
        # org population among attachment-only
        org_pop = (org != "")
        n_attach_with_org += int((attach_mask & org_pop).sum())
        n_attach_no_org += int((attach_mask & ~org_pop).sum())

        # Discovery: count normalized short comments (<=80 chars after norm)
        norm_short = comment_text.apply(lambda t: normalize_for_match(t))
        for v in norm_short[norm_short.str.len().between(1, 80)]:
            short_counter[v] += 1

    pct = (n_attach_only / n_comments * 100.0) if n_comments else 0.0
    return ({
        "file": path.name,
        "agency": agency,
        "year": year,
        "total_rows": total,
        "n_comments": n_comments,
        "n_attach_only": n_attach_only,
        "pct_attach_only": pct,
        "n_attach_with_org": n_attach_with_org,
        "n_attach_no_org": n_attach_no_org,
    }, short_counter)


# --- Main ----------------------------------------------------------------

def main() -> int:
    files = discover_csvs(RAW_DIR)
    print(f"diagnostic over {len(files)} files")
    rows: list[dict] = []
    placeholder_counter: Counter[str] = Counter()
    for p in files:
        stats, short_counter = diagnose_one(p)
        rows.append(stats)
        placeholder_counter.update(short_counter)
        print(f"  {p.name}: total={stats['total_rows']:>7,}  "
              f"comments={stats['n_comments']:>7,}  "
              f"attach={stats['n_attach_only']:>7,}  "
              f"({stats['pct_attach_only']:.1f}%)")

    df = pd.DataFrame(rows)

    # Rollups
    by_agency = df.groupby("agency").agg(
        n_files=("file", "size"),
        total_rows=("total_rows", "sum"),
        n_comments=("n_comments", "sum"),
        n_attach_only=("n_attach_only", "sum"),
    )
    by_agency["pct_attach_only"] = by_agency["n_attach_only"] / by_agency["n_comments"] * 100

    by_year = df.groupby("year").agg(
        total_rows=("total_rows", "sum"),
        n_comments=("n_comments", "sum"),
        n_attach_only=("n_attach_only", "sum"),
    )
    by_year["pct_attach_only"] = by_year["n_attach_only"] / by_year["n_comments"] * 100

    overall_total = df["total_rows"].sum()
    overall_comments = df["n_comments"].sum()
    overall_attach = df["n_attach_only"].sum()
    overall_pct = (overall_attach / overall_comments * 100) if overall_comments else 0
    overall_attach_with_org = df["n_attach_with_org"].sum()
    overall_attach_no_org = df["n_attach_no_org"].sum()

    # --- Markdown output ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Attachment-Rate Diagnostic — 39 regulations.gov CSVs")
    lines.append("")
    lines.append(f"_Generated {pd.Timestamp.utcnow().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("**Definitions used:**")
    lines.append("- A *comment* row = `Comment on Document ID` non-empty OR `Document Type == 'Public Submission'` (union).")
    lines.append("- An *attachment-only* comment = `Comment` field is empty/null OR matches a placeholder pattern after normalization (lowercase, whitespace-collapsed, leading 'please' stripped, trailing punctuation stripped). Patterns include:")
    lines.append("  `see attached`, `see attached file(s)`, `see attached document`, `see attachment(s)`, `see the attached letter`, `see my/our attached comments`, `attached`, `comment(s) attached`, `letter attached`, `n/a`, `none`, etc. (full regex in script).")
    lines.append("- *Org-attached* = attachment-only comment with `Organization Name` populated; *Indiv-attached* = attachment-only with empty `Organization Name`.")
    lines.append("")

    # Per-file table
    lines.append("## Per-file")
    lines.append("")
    lines.append("| File | Total rows | Comments | Attachment-only | % attach | Org-attached | Indiv-attached |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df.sort_values(["agency", "year"]).iterrows():
        lines.append(f"| {r['file']} | {r['total_rows']:,} | {r['n_comments']:,} "
                     f"| {r['n_attach_only']:,} | {r['pct_attach_only']:.1f}% "
                     f"| {r['n_attach_with_org']:,} | {r['n_attach_no_org']:,} |")
    lines.append("")

    # By agency
    lines.append("## Rollup by agency")
    lines.append("")
    lines.append("| Agency | Files | Total rows | Comments | Attachment-only | % attach |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ag, r in by_agency.iterrows():
        lines.append(f"| {ag} | {int(r['n_files']):,} | {int(r['total_rows']):,} | "
                     f"{int(r['n_comments']):,} | {int(r['n_attach_only']):,} | "
                     f"{r['pct_attach_only']:.1f}% |")
    lines.append("")

    # By year
    lines.append("## Rollup by year")
    lines.append("")
    lines.append("| Year | Total rows | Comments | Attachment-only | % attach |")
    lines.append("|---|---:|---:|---:|---:|")
    for yr, r in by_year.iterrows():
        lines.append(f"| {int(yr)} | {int(r['total_rows']):,} | {int(r['n_comments']):,} "
                     f"| {int(r['n_attach_only']):,} | {r['pct_attach_only']:.1f}% |")
    lines.append("")

    # Overall
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Total rows across all 39 files: **{overall_total:,}**")
    lines.append(f"- Total comments: **{overall_comments:,}**")
    lines.append(f"- Total attachment-only: **{overall_attach:,}** "
                 f"({overall_pct:.1f}% of comments)")
    lines.append(f"- Of attachment-only: **{overall_attach_with_org:,}** with Organization Name, "
                 f"**{overall_attach_no_org:,}** without")
    lines.append("")

    # --- Placeholder discovery ---
    # Take the most common short normalized comment values that are NOT yet
    # caught by our matcher. These are candidates to add.
    not_caught: list[tuple[str, int]] = []
    for txt, cnt in placeholder_counter.most_common(500):
        if not is_attachment_only(txt):
            not_caught.append((txt, cnt))

    lines.append("## Placeholder discovery — top short comments NOT matched by current regex")
    lines.append("")
    lines.append("These are the most common normalized comment values (length 1–80 chars) "
                 "that the current attachment-only definition does **not** capture. Skim for "
                 "additional placeholder variants worth folding into the definition.")
    lines.append("")
    lines.append("| n | normalized text |")
    lines.append("|---:|---|")
    for txt, cnt in not_caught[:60]:
        safe = txt.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {cnt:,} | {safe} |")
    lines.append("")
    lines.append("## Placeholder discovery — top short comments ALREADY matched")
    lines.append("")
    lines.append("Sanity check: these are the most common short comments that our regex DOES "
                 "consider attachment-only.")
    lines.append("")
    lines.append("| n | normalized text |")
    lines.append("|---:|---|")
    caught_top: list[tuple[str, int]] = []
    for txt, cnt in placeholder_counter.most_common(500):
        if is_attachment_only(txt):
            caught_top.append((txt, cnt))
        if len(caught_top) >= 30:
            break
    for txt, cnt in caught_top:
        safe = txt.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {cnt:,} | {safe} |")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
