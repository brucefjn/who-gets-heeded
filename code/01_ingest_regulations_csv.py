"""
01_ingest_regulations_csv.py
Ingestion pipeline for regulations.gov bulk-download CSVs (EPA-only, 2010-2022).

Scope (Bruce, 2026-05-06): EPA only. FCC and SEC bulk downloads contain zero
public comments — those agencies route comments through their own systems
(FCC ECFS, SEC file-number-indexed letters). Their raw CSVs remain in
data/raw/regulations_gov/{FCC,SEC}/ for a possible later FAccT 2027 extension.

Decisions encoded:
  - Output is per-year parquet (or pickle if pyarrow missing).
        data/processed/documents/EPA_{YEAR}.parquet
        data/processed/comments/EPA_{YEAR}.parquet
  - Canonical schema = union of snake_case column names across the 13 EPA CSVs,
    cached at data/processed/canonical_schema.json. Both documents and comments
    parquet files use this schema, NaN-filled for missing columns. The schema
    cache *can* still include FCC/SEC columns from earlier scans — those just
    become NaN on EPA outputs, harmless.
  - Classification rule: a row is a comment iff `Comment on Document ID` is
    populated. Document Type is *also* recorded; any disagreement between the
    link-based and type-based classifications is logged to data_quality_log.md.
  - Encoding is detected per file (utf-8 → utf-8-sig → latin-1 → cp1252).
  - Each comment row gets:
        comment_text_hash       SHA256 of normalized comment text (form-letter signal)
        is_attachment_only      True iff comment is empty or matches a
                                placeholder pattern from lib.comment_classify
    Document rows have both columns set to NaN/False.
  - Date columns are parsed to UTC datetime.
  - Memory: chunked CSV reads (50k rows/chunk).

Usage:
    python 01_ingest_regulations_csv.py --build-schema-only
    python 01_ingest_regulations_csv.py --file EPA/EPA_2018_full.csv --report
    python 01_ingest_regulations_csv.py --all-epa --report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# Make `code/lib` importable when running `python code/01_ingest_regulations_csv.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.comment_classify import is_attachment_only  # noqa: E402

try:
    import pyarrow  # noqa: F401
    PARQUET_OK = True
except ImportError:
    PARQUET_OK = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "regulations_gov"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROCESSED_DIR / "documents"
COMMENTS_DIR = PROCESSED_DIR / "comments"
SCHEMA_PATH = PROCESSED_DIR / "canonical_schema.json"
DQ_LOG_PATH = PROCESSED_DIR / "data_quality_log.md"

CHUNK_SIZE = 50_000

# Date columns in the regulations.gov schema; parsed to UTC datetime.
DATE_COLS = [
    "posted_date",
    "comment_start_date",
    "comment_due_date",
    "received_date",
    "postmark_date",
    "effective_date",
    "implementation_date",
    "author_date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNAKE_RE = re.compile(r"[^A-Za-z0-9]+")


def snake_case(name: str) -> str:
    """'Document ID' -> 'document_id', 'Display Properties (Name, Label, Tooltip)'
    -> 'display_properties_name_label_tooltip'."""
    s = _SNAKE_RE.sub("_", name.strip()).strip("_").lower()
    # Collapse repeats (defensive; regex above handles most cases)
    return re.sub(r"_+", "_", s)


ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def detect_encoding(path: Path, sample_bytes: int = 1_000_000) -> str:
    """Return first encoding that decodes the sample without error.
    'latin-1' is the universal fallback (it never raises)."""
    with open(path, "rb") as f:
        sample = f.read(sample_bytes)
    for enc in ENCODING_CANDIDATES:
        try:
            sample.decode(enc, errors="strict")
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


_FNAME_RE = re.compile(r"^(EPA|FCC|SEC)_(\d{4})_full\.csv$", re.IGNORECASE)


def parse_agency_year(path: Path) -> tuple[str, int]:
    m = _FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Cannot parse agency/year from filename: {path.name}")
    return m.group(1).upper(), int(m.group(2))


# ---------------------------------------------------------------------------
# Comment hashing
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_comment_text(text: Any) -> str:
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(text).lower().strip()
    return _WS_RE.sub(" ", s)


def comment_hash(text: Any) -> str | None:
    norm = normalize_comment_text(text)
    if not norm:
        return None
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

def discover_csvs(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*_full.csv"))


def build_canonical_schema(raw_dir: Path) -> dict:
    """Union snake_case column names across all CSVs in raw_dir.
    Records per-file encoding + original column list for the data quality log."""
    files = discover_csvs(raw_dir)
    union_cols: list[str] = []
    seen: set[str] = set()
    per_file: list[dict] = []

    for path in files:
        enc = detect_encoding(path)
        header_df = pd.read_csv(path, nrows=0, encoding=enc)
        original = list(header_df.columns)
        snake = [snake_case(c) for c in original]
        for c in snake:
            if c not in seen:
                union_cols.append(c)
                seen.add(c)
        per_file.append({
            "path": str(path.relative_to(raw_dir)),
            "encoding": enc,
            "n_columns": len(original),
            "original_columns": original,
            "snake_columns": snake,
        })

    return {
        "union_columns_snake": union_cols,
        "union_n_columns": len(union_cols),
        "per_file": per_file,
    }


def load_or_build_schema(rebuild: bool = False) -> dict:
    if rebuild or not SCHEMA_PATH.exists():
        schema = build_canonical_schema(RAW_DIR)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
        print(f"[schema] wrote {SCHEMA_PATH} ({schema['union_n_columns']} cols, "
              f"{len(schema['per_file'])} files)")
    else:
        schema = json.loads(SCHEMA_PATH.read_text())
        print(f"[schema] loaded {SCHEMA_PATH} ({schema['union_n_columns']} cols, "
              f"{len(schema['per_file'])} files)")
    return schema


# ---------------------------------------------------------------------------
# Per-file ingestion
# ---------------------------------------------------------------------------

def _is_comment_by_link(s: pd.Series) -> pd.Series:
    """A row is a comment iff Comment on Document ID is non-empty."""
    return s.fillna("").astype(str).str.strip() != ""


def _parse_dates_inplace(df: pd.DataFrame) -> None:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)


def _write_table(df: pd.DataFrame, target: Path) -> tuple[Path, str]:
    """Write to parquet if pyarrow available, else pickle (sandbox fallback)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if PARQUET_OK:
        df.to_parquet(target, index=False)
        return target, "parquet"
    fallback = target.with_suffix(".pkl")
    df.to_pickle(fallback)
    return fallback, "pickle"


def count_records_with_csv(csv_path: Path, encoding: str) -> tuple[int, int]:
    """Use the stdlib csv module (multi-line aware, permissive) to count true
    record count and number of rows whose field count != header field count.
    Returns (n_records, n_field_count_anomalies). Header is excluded from count."""
    import csv as _csv
    n = 0
    bad = 0
    with open(csv_path, encoding=encoding, newline="") as f:
        r = _csv.reader(f)
        header = next(r)
        n_cols = len(header)
        for row in r:
            n += 1
            if len(row) != n_cols:
                bad += 1
    return n, bad


def ingest_one_file(
    csv_path: Path,
    canonical_cols: list[str],
    chunksize: int = CHUNK_SIZE,
    sample_n_comments: int = 0,
) -> dict:
    """Ingest a single regulations.gov CSV.

    Returns a stats dict including paths to written outputs and any
    classification mismatches.
    """
    if not csv_path.is_absolute():
        csv_path = (RAW_DIR / csv_path).resolve()
    agency, year = parse_agency_year(csv_path)
    encoding = detect_encoding(csv_path)

    # Ground-truth record count via stdlib csv (multi-line aware).
    # We compare this against pandas' parsed count to detect silent skips.
    true_n_records, n_field_count_anomalies = count_records_with_csv(csv_path, encoding)

    extra_cols = ["agency", "data_year", "comment_text_hash", "is_attachment_only"]
    final_cols = list(dict.fromkeys(canonical_cols + extra_cols))

    docs_chunks: list[pd.DataFrame] = []
    comments_chunks: list[pd.DataFrame] = []
    mismatches: list[pd.DataFrame] = []

    n_rows = 0
    n_docs = 0
    n_comments = 0
    doc_type_counts: dict[str, int] = {}
    sampled_comments: list[str] = []

    rng = random.Random(42)

    # `on_bad_lines='skip'` skips any row pandas' C parser can't fit to the
    # canonical column count. We compare the parsed total against the csv-module
    # ground truth and log the delta.
    reader = pd.read_csv(
        csv_path,
        encoding=encoding,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
        on_bad_lines="skip",
        engine="c",
    )

    link_col = "comment_on_document_id"
    dt_col = "document_type"

    for chunk in reader:
        chunk.columns = [snake_case(c) for c in chunk.columns]
        chunk["agency"] = agency
        chunk["data_year"] = year

        if link_col not in chunk.columns:
            chunk[link_col] = pd.NA
        if dt_col not in chunk.columns:
            chunk[dt_col] = pd.NA

        is_comment_link = _is_comment_by_link(chunk[link_col])
        dt_clean = chunk[dt_col].fillna("").astype(str).str.strip().replace("", "MISSING")
        for v, c in dt_clean.value_counts().items():
            doc_type_counts[v] = doc_type_counts.get(v, 0) + int(c)
        # Common values for comments: 'Public Submission' (modern), older bulk
        # downloads occasionally use 'Comment'. Both are treated as comment-like.
        is_comment_type = dt_clean.str.lower().isin(["public submission", "comment"])

        mm_mask = (is_comment_link != is_comment_type)
        if mm_mask.any():
            mm = pd.DataFrame({
                "agency": agency,
                "year": year,
                "document_id": chunk.loc[mm_mask, "document_id"] if "document_id" in chunk.columns else "",
                "document_type": dt_clean[mm_mask].values,
                "comment_on_document_id": chunk.loc[mm_mask, link_col].fillna("").astype(str).values,
                "link_says_comment": is_comment_link[mm_mask].values,
                "type_says_comment": is_comment_type[mm_mask].values,
            })
            mismatches.append(mm)

        # Authoritative classification: link-based
        comment_chunk = chunk.loc[is_comment_link].copy()
        doc_chunk = chunk.loc[~is_comment_link].copy()

        if "comment" in comment_chunk.columns:
            comment_chunk["comment_text_hash"] = comment_chunk["comment"].apply(comment_hash)
            comment_chunk["is_attachment_only"] = comment_chunk["comment"].apply(is_attachment_only)
        else:
            comment_chunk["comment_text_hash"] = pd.NA
            comment_chunk["is_attachment_only"] = True  # no comment col → vacuously empty
        # Document rows: never attachment-only (they're not comments).
        doc_chunk["is_attachment_only"] = False
        doc_chunk["comment_text_hash"] = pd.NA

        # Sample some comment texts (reservoir-style, simple)
        if sample_n_comments > 0 and "comment" in comment_chunk.columns:
            for txt in comment_chunk["comment"].dropna().tolist():
                if not txt or not str(txt).strip():
                    continue
                if len(sampled_comments) < sample_n_comments:
                    sampled_comments.append(str(txt))
                else:
                    j = rng.randint(0, n_comments + len(comment_chunk))
                    if j < sample_n_comments:
                        sampled_comments[j] = str(txt)

        comment_chunk = comment_chunk.reindex(columns=final_cols)
        doc_chunk = doc_chunk.reindex(columns=final_cols)

        docs_chunks.append(doc_chunk)
        comments_chunks.append(comment_chunk)

        n_rows += len(chunk)
        n_docs += len(doc_chunk)
        n_comments += len(comment_chunk)

    docs_df = pd.concat(docs_chunks, ignore_index=True) if docs_chunks else pd.DataFrame(columns=final_cols)
    comments_df = pd.concat(comments_chunks, ignore_index=True) if comments_chunks else pd.DataFrame(columns=final_cols)
    mm_df = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame(
        columns=["agency", "year", "document_id", "document_type",
                 "comment_on_document_id", "link_says_comment", "type_says_comment"]
    )

    _parse_dates_inplace(docs_df)
    _parse_dates_inplace(comments_df)

    docs_path, fmt = _write_table(docs_df, DOCS_DIR / f"{agency}_{year}.parquet")
    comments_path, _ = _write_table(comments_df, COMMENTS_DIR / f"{agency}_{year}.parquet")

    if not PARQUET_OK:
        warnings.warn(
            "pyarrow not installed in this environment — wrote pickle (.pkl) "
            "instead of parquet. Install pyarrow and rerun for production.",
            stacklevel=2,
        )

    # Top comment text hashes
    top_hashes: list[dict] = []
    if "comment_text_hash" in comments_df.columns:
        vc = comments_df["comment_text_hash"].dropna().value_counts().head(5)
        for h, count in vc.items():
            sample_text = comments_df.loc[
                comments_df["comment_text_hash"] == h, "comment"
            ].dropna().iloc[0] if "comment" in comments_df.columns else None
            top_hashes.append({
                "hash": h,
                "n": int(count),
                "sample_first_200_chars": (str(sample_text)[:200] if sample_text else None),
            })

    n_attach_only = int(comments_df["is_attachment_only"].sum()) if "is_attachment_only" in comments_df.columns else 0
    pct_attach_only = (100.0 * n_attach_only / n_comments) if n_comments else 0.0

    return {
        "agency": agency,
        "year": year,
        "encoding": encoding,
        "out_format": fmt,
        "true_n_records": true_n_records,
        "n_field_count_anomalies": n_field_count_anomalies,
        "n_rows_parsed": n_rows,
        "n_rows_skipped_by_pandas": true_n_records - n_rows,
        "n_documents": n_docs,
        "n_comments": n_comments,
        "n_attachment_only": n_attach_only,
        "pct_attachment_only": pct_attach_only,
        "doc_type_counts": doc_type_counts,
        "n_mismatches": len(mm_df),
        "mismatches_df": mm_df,
        "top_comment_hashes": top_hashes,
        "sampled_comment_texts": sampled_comments,
        "docs_path": str(docs_path),
        "comments_path": str(comments_path),
        "n_canonical_cols": len(final_cols),
    }


# ---------------------------------------------------------------------------
# Data quality log writer
# ---------------------------------------------------------------------------

def append_quality_log(stats: dict, schema: dict, log_path: Path = DQ_LOG_PATH) -> None:
    """Append a section to data_quality_log.md describing this ingestion run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        header = (
            "# Data Quality Log — regulations.gov ingestion\n\n"
            "Each section below records one `01_ingest_regulations_csv.py` run.\n"
            "Authoritative classification: link-based (`Comment on Document ID` non-empty → comment).\n"
            "Mismatches between link-based and `Document Type`-based classification are logged but not silently fixed.\n\n"
        )
        log_path.write_text(header)

    a, y = stats["agency"], stats["year"]
    lines: list[str] = []
    lines.append(f"\n## {a} {y} — {pd.Timestamp.utcnow().isoformat(timespec='seconds')}\n")
    lines.append(f"- Encoding: `{stats['encoding']}`")
    lines.append(f"- Output format: {stats['out_format']}")
    lines.append(f"- True record count (csv module): {stats['true_n_records']:,}")
    lines.append(f"- Field-count anomalies (rows with != n_header_cols): "
                 f"{stats['n_field_count_anomalies']:,}")
    lines.append(f"- Rows parsed by pandas: {stats['n_rows_parsed']:,} "
                 f"(skipped by pandas: {stats['n_rows_skipped_by_pandas']:,})")
    lines.append(f"- Documents: {stats['n_documents']:,} | "
                 f"Comments: {stats['n_comments']:,}")
    lines.append(f"- Attachment-only comments: {stats['n_attachment_only']:,} "
                 f"({stats['pct_attachment_only']:.1f}% of comments)")
    lines.append(f"- Canonical schema columns applied: {stats['n_canonical_cols']}")
    lines.append(f"- Output: `{stats['docs_path']}`, `{stats['comments_path']}`")
    lines.append("")
    lines.append("**Document Type distribution (raw, before classification):**")
    lines.append("")
    lines.append("| Document Type | Count |")
    lines.append("|---|---:|")
    for v, c in sorted(stats["doc_type_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{v}` | {c:,} |")
    lines.append("")

    n_mm = stats["n_mismatches"]
    lines.append(f"**Classification mismatches** (link-based vs type-based): {n_mm:,}")
    if n_mm > 0:
        mm = stats["mismatches_df"]
        # Pattern summary
        patt = mm.groupby(["document_type", "link_says_comment", "type_says_comment"]).size().reset_index(name="n")
        patt = patt.sort_values("n", ascending=False)
        lines.append("")
        lines.append("Mismatch patterns:")
        lines.append("")
        lines.append("| document_type | link_says_comment | type_says_comment | n |")
        lines.append("|---|---|---|---:|")
        for _, row in patt.iterrows():
            lines.append(f"| `{row['document_type']}` | {row['link_says_comment']} "
                         f"| {row['type_says_comment']} | {int(row['n']):,} |")
        lines.append("")
        lines.append("Sample mismatches (first 10):")
        lines.append("")
        sample = mm.head(10)
        lines.append("| document_id | document_type | link | link_says | type_says |")
        lines.append("|---|---|---|---|---|")
        for _, row in sample.iterrows():
            link_short = (row["comment_on_document_id"] or "")[:40]
            lines.append(f"| `{row['document_id']}` | `{row['document_type']}` "
                         f"| `{link_short}` | {row['link_says_comment']} "
                         f"| {row['type_says_comment']} |")
        lines.append("")

    lines.append("**Top 5 most common `comment_text_hash` (form-letter signal):**")
    lines.append("")
    if not stats["top_comment_hashes"]:
        lines.append("_(none — no hashed comments)_")
    else:
        lines.append("| n | hash (first 16) | sample (first 200 chars) |")
        lines.append("|---:|---|---|")
        for h in stats["top_comment_hashes"]:
            sample_clean = (h["sample_first_200_chars"] or "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {h['n']:,} | `{h['hash'][:16]}` | {sample_clean} |")
    lines.append("")

    if stats["sampled_comment_texts"]:
        lines.append("**Random comment text samples** (text-quality / encoding sanity check):")
        lines.append("")
        for i, t in enumerate(stats["sampled_comment_texts"], 1):
            preview = (t[:500] + "…") if len(t) > 500 else t
            preview = preview.replace("\n", " ")
            lines.append(f"{i}. {preview}")
            lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_stats(stats: dict) -> None:
    summary = {k: v for k, v in stats.items() if k not in ("mismatches_df",)}
    # Truncate doc_type_counts to top 10 for stdout
    dtc = summary.pop("doc_type_counts", {})
    top_dtc = dict(sorted(dtc.items(), key=lambda kv: -kv[1])[:10])
    summary["doc_type_counts_top10"] = top_dtc
    print(json.dumps(summary, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", help="CSV to ingest (absolute, or relative to data/raw/regulations_gov/)")
    p.add_argument("--all", action="store_true", help="Ingest all CSVs (legacy; use --all-epa)")
    p.add_argument("--all-epa", action="store_true", help="Ingest all 13 EPA CSVs (current scope)")
    p.add_argument("--build-schema-only", action="store_true",
                   help="Only build/refresh canonical schema, then exit")
    p.add_argument("--rebuild-schema", action="store_true",
                   help="Rebuild canonical schema even if cached")
    p.add_argument("--report", action="store_true",
                   help="Append a quality-log section for each file ingested")
    p.add_argument("--sample-comments", type=int, default=0,
                   help="If --report, sample N random comment texts per file for the log")
    args = p.parse_args(argv)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    schema = load_or_build_schema(rebuild=args.rebuild_schema)

    if args.build_schema_only:
        return 0

    canonical = schema["union_columns_snake"]
    targets: list[Path] = []
    if args.all_epa:
        targets = sorted((RAW_DIR / "EPA").glob("EPA_*_full.csv"))
    elif args.all:
        targets = discover_csvs(RAW_DIR)
    elif args.file:
        p = Path(args.file)
        if not p.is_absolute():
            p = (RAW_DIR / args.file).resolve()
        targets = [p]
    else:
        print("nothing to do — pass --file PATH or --all", file=sys.stderr)
        return 2

    for t in targets:
        print(f"\n[ingest] {t.name}")
        stats = ingest_one_file(
            t, canonical,
            sample_n_comments=args.sample_comments if args.report else 0,
        )
        _print_stats(stats)
        if args.report:
            append_quality_log(stats, schema)
            print(f"[ingest] appended report to {DQ_LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
