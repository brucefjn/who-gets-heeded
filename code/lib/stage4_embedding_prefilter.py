"""
stage4_embedding_prefilter.py — sentence-transformer prefilter for Path A
Stage 4 (comment-to-obligation matching).

Per notes/2026-05-07_stage1_rubric_design.md §549–580, Stage 4 calls the
LLM on every (comment, obligation) pair that survives an embedding-
similarity prefilter. The prefilter caps each comment at top-K=10
obligations with cosine similarity >= 0.5 (K pre-locked v2.3).

Model: `all-mpnet-base-v2` (768-dim). Chosen for the legal-NLP retrieval
quality / inference-speed balance documented in the v2.3 rubric design.
GPU is auto-detected via torch if available, otherwise CPU.

Embedding inputs (locked):
  - Comment: the raw text from the analytic corpus `comment` column.
  - Obligation: f"{subject} {modal} {action} {object}\n{obligation_text}"
    — pre-extracted structured fields prepended to the verbatim sentence,
    so the cosine score reflects both the parsed predicate-argument
    structure (high signal) and the surface form (catches phrasings the
    parser missed).

Cache layout (per anchor, parquet):
  data/intermediate/stage4_embeddings/<docket_id>__comments.parquet
  data/intermediate/stage4_embeddings/<docket_id>__obligations.parquet
Each cache row carries an SHA-1 content hash so the embedder can detect
when an upstream Stage 1b run changed an obligation's text and recompute
selectively rather than invalidating the whole cache.

Public entry points:
  prefilter_top_k(comments, obligations, k, similarity_threshold)
      → list[(comment_id, obligation_id, cosine)]
  embed_comments(comments, cache_path) → np.ndarray, id_index_map
  embed_obligations(obligations, cache_path) → np.ndarray, id_index_map
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


# Model name locked v2.3.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768

# Prefilter defaults — K locked v2.3; similarity threshold tightened
# 0.5 → 0.3 (2026-05-12) after the initial 0.5 floor was found to drop
# legitimate matches the LLM stage was happy to confirm. The looser
# floor lets gpt-5 make the final call on borderline pairs.
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.3

# Minimum comment length (post-strip) for Stage 4 eligibility. Below
# this, the comment can't carry enough signal for either the embedding
# prefilter or the LLM matcher to produce a meaningful judgement.
MIN_COMMENT_LENGTH_CHARS = 50

# Default cache root.
DEFAULT_CACHE_DIR = Path("data/intermediate/stage4_embeddings")


# ---------------------------------------------------------------------------
# Stage 4 eligibility filter (runs before embedding)
# ---------------------------------------------------------------------------
def filter_eligible_comments(comments_df):
    """Drop rows that are not Stage 4 eligible. Applied BEFORE the
    embedding step so we don't burn embedding compute on rows that would
    be dropped anyway.

    Two exclusion rules:
      (a) Attachment-recovery failures — rows where the attachment was
          the comment's only payload AND the recovery step couldn't
          retrieve text. Concretely: `is_attachment_only == True`
          AND `text_source == 'attachment_failed'`. No semantic content
          to embed.
      (b) Too-short free-text — rows where the `comment` field stripped
          of leading/trailing whitespace is shorter than
          MIN_COMMENT_LENGTH_CHARS (50). Below this, the comment can't
          carry enough signal for the prefilter or the LLM to judge
          whether it addresses a specific obligation.

    Args:
        comments_df: pandas DataFrame with at least `comment` and the
            optional `is_attachment_only` / `text_source` columns. Rows
            with missing optional columns are not dropped by rule (a).

    Returns:
        (filtered_df, drop_counts) where drop_counts has the keys:
          - 'attachment_failed': count satisfying rule (a)
          - 'too_short': count satisfying rule (b)
          - 'total_dropped': len(comments_df) - len(filtered_df)
          - 'n_kept': len(filtered_df)
        Rule (a) and (b) counts are independent — a row dropped by both
        is counted once under each, so they may not sum to total_dropped.
    """
    import pandas as pd
    if comments_df is None or len(comments_df) == 0:
        return comments_df, {"attachment_failed": 0, "too_short": 0,
                             "total_dropped": 0, "n_kept": 0}

    if "is_attachment_only" in comments_df.columns \
            and "text_source" in comments_df.columns:
        att_mask = (
            (comments_df["is_attachment_only"] == True)   # noqa: E712
            & (comments_df["text_source"] == "attachment_failed")
        )
    else:
        att_mask = pd.Series(False, index=comments_df.index)

    if "comment" in comments_df.columns:
        short_mask = comments_df["comment"].fillna("").astype(str) \
            .str.strip().str.len() < MIN_COMMENT_LENGTH_CHARS
    else:
        short_mask = pd.Series(False, index=comments_df.index)

    drop_mask = att_mask | short_mask
    kept = comments_df.loc[~drop_mask].reset_index(drop=True)
    return kept, {
        "attachment_failed": int(att_mask.sum()),
        "too_short": int(short_mask.sum()),
        "total_dropped": int(drop_mask.sum()),
        "n_kept": int(len(kept)),
    }


# ---------------------------------------------------------------------------
# Embedding-input formatters (locked spec — do not vary)
# ---------------------------------------------------------------------------
def format_comment_for_embedding(comment_row: dict) -> str:
    """Comment embedding input is the verbatim `comment` text. The
    Stage 4 LLM call separately handles truncation for context-budget
    reasons; for embedding we want the full available text so similarity
    captures topics raised anywhere in the comment."""
    return str(comment_row.get("comment") or comment_row.get("text") or "").strip()


def format_obligation_for_embedding(obligation_row: dict) -> str:
    """Obligation embedding input — pre-extracted structured fields
    prepended to the verbatim sentence:

        "{subject} {modal} {action} {object}\\n{obligation_text}"

    Locked per Stage 4 spec; do not vary."""
    subject = str(obligation_row.get("subject") or "").strip()
    modal = str(obligation_row.get("modal") or "").strip()
    action = str(obligation_row.get("action") or "").strip()
    obj = str(obligation_row.get("object") or "").strip()
    text = str(obligation_row.get("obligation_text") or "").strip()
    head = " ".join(p for p in (subject, modal, action, obj) if p)
    if head and text:
        return f"{head}\n{text}"
    return head or text


# ---------------------------------------------------------------------------
# ID synthesis
# ---------------------------------------------------------------------------
def comment_id_of(row: dict) -> str:
    cid = row.get("comment_id") or row.get("document_id")
    if not cid:
        raise ValueError("Comment row missing comment_id / document_id.")
    return str(cid)


def obligation_id_of(row: dict) -> str:
    """Stage 1b output has (candidate_idx, split_idx) unique within a
    (docket_id, rule_type). Stage 4 namespaces by docket+rule_type so
    the same obligation_id can survive a join across anchors."""
    cidx = row.get("candidate_idx")
    sidx = row.get("split_idx", 0)
    docket = row.get("docket_id", "")
    rule = row.get("rule_type", "")
    if cidx is None:
        raise ValueError("Obligation row missing candidate_idx.")
    return f"{docket}__{rule}__{cidx}_{sidx}"


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Sentence-transformers wrapper (lazy import)
# ---------------------------------------------------------------------------
def _load_model(model_name: str = DEFAULT_EMBED_MODEL):
    """Lazy-import sentence-transformers and torch. Auto-pick GPU if
    available. Raises ImportError with an actionable install hint if
    the libraries aren't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for the Stage 4 prefilter. "
            "Install via: pip install sentence-transformers"
        ) from e
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass
    return SentenceTransformer(model_name, device=device)


def _default_embed_fn(model_name: str = DEFAULT_EMBED_MODEL) -> Callable[[list[str]], np.ndarray]:
    """Build a callable that encodes a list of strings into an
    (n_strings, EMBED_DIM) float32 numpy array, L2-normalized so dot
    products give cosine similarity directly."""
    model = _load_model(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        out = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return out.astype(np.float32, copy=False)

    return encode


# ---------------------------------------------------------------------------
# Per-anchor cache (parquet)
# ---------------------------------------------------------------------------
def _read_cache(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Returns {id: (content_hash, embedding)}. Empty if cache absent."""
    if not path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [stage4-cache] could not read cache {path}: "
              f"{type(e).__name__}: {e}; treating as empty.",
              file=sys.stderr)
        return {}
    out: dict[str, tuple[str, np.ndarray]] = {}
    for _, row in df.iterrows():
        emb = np.asarray(row["embedding"], dtype=np.float32)
        out[str(row["id"])] = (str(row["content_hash"]), emb)
    return out


def _write_cache(path: Path, cache: dict[str, tuple[str, np.ndarray]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": k, "content_hash": h, "embedding": emb.tolist()}
        for k, (h, emb) in cache.items()
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _embed_with_cache(
    items: list[dict],
    id_of: Callable[[dict], str],
    text_of: Callable[[dict], str],
    cache_path: Optional[Path],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Returns (embeddings, id_order). embeddings[i] is the row vector
    for id_order[i]. Re-uses cached rows whose content_hash matches; only
    items with a missing or mismatched hash get re-encoded."""
    cache: dict[str, tuple[str, np.ndarray]] = (
        _read_cache(cache_path) if cache_path else {}
    )

    todo_texts: list[str] = []
    todo_ids: list[str] = []
    id_to_hash: dict[str, str] = {}

    for r in items:
        rid = id_of(r)
        text = text_of(r)
        h = _content_hash(text)
        id_to_hash[rid] = h
        cached = cache.get(rid)
        if cached is None or cached[0] != h:
            todo_texts.append(text)
            todo_ids.append(rid)

    if todo_texts:
        new_embs = embed_fn(todo_texts)
        if new_embs.shape != (len(todo_texts), EMBED_DIM):
            raise RuntimeError(
                f"embed_fn returned shape {new_embs.shape}, expected "
                f"({len(todo_texts)}, {EMBED_DIM})."
            )
        for rid, emb in zip(todo_ids, new_embs):
            cache[rid] = (id_to_hash[rid], emb.astype(np.float32, copy=False))

    if cache_path and todo_texts:
        _write_cache(cache_path, cache)

    id_order = [id_of(r) for r in items]
    if not id_order:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []
    embeddings = np.stack([cache[rid][1] for rid in id_order], axis=0)
    return embeddings, id_order


def embed_comments(
    comments: list[dict],
    cache_path: Optional[Path] = None,
    *,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Embed comment rows. Output rows are L2-normalized (cosine = dot)."""
    fn = embed_fn or _default_embed_fn()
    return _embed_with_cache(
        comments, comment_id_of, format_comment_for_embedding, cache_path, fn,
    )


def embed_obligations(
    obligations: list[dict],
    cache_path: Optional[Path] = None,
    *,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Embed obligation rows using the locked format string."""
    fn = embed_fn or _default_embed_fn()
    return _embed_with_cache(
        obligations, obligation_id_of, format_obligation_for_embedding,
        cache_path, fn,
    )


# ---------------------------------------------------------------------------
# Top-K cosine prefilter
# ---------------------------------------------------------------------------
def prefilter_top_k(
    comments: list[dict],
    obligations: list[dict],
    k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    *,
    docket_id: Optional[str] = None,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> list[tuple[str, str, float]]:
    """For each comment, return the top-K obligations by cosine
    similarity, filtered to similarity >= similarity_threshold.

    Args:
        comments: dict rows from the analytic-corpus parquet
            (must have document_id/comment_id, comment/text).
        obligations: dict rows from the Stage 1b verified CSV.
        k: max obligations retained per comment (locked v2.3 = 10).
        similarity_threshold: cosine floor (locked v2.3 = 0.5).
        docket_id: used to namespace cache files. If None, embeddings
            are not cached (one-off calls / tests).
        cache_dir: parent dir for cache parquets.
        embed_fn: inject a custom embedder for tests; defaults to
            sentence-transformers/all-mpnet-base-v2.

    Returns:
        list of (comment_id, obligation_id, cosine_similarity) tuples,
        sorted by comment_id ascending then cosine descending.
    """
    if not comments or not obligations:
        return []

    comments_cache: Optional[Path] = None
    obligations_cache: Optional[Path] = None
    if docket_id and cache_dir:
        cache_dir = Path(cache_dir)
        comments_cache = cache_dir / f"{docket_id}__comments.parquet"
        obligations_cache = cache_dir / f"{docket_id}__obligations.parquet"

    c_embs, c_ids = embed_comments(comments, comments_cache, embed_fn=embed_fn)
    o_embs, o_ids = embed_obligations(obligations, obligations_cache, embed_fn=embed_fn)

    if c_embs.shape[0] == 0 or o_embs.shape[0] == 0:
        return []

    # Embeddings are pre-normalized → dot product == cosine.
    sims = c_embs @ o_embs.T  # (n_comments, n_obligations)

    n_obligations = sims.shape[1]
    keep_k = min(k, n_obligations)

    pairs: list[tuple[str, str, float]] = []
    for i, cid in enumerate(c_ids):
        row = sims[i]
        if keep_k < n_obligations:
            top_idx = np.argpartition(-row, kth=keep_k - 1)[:keep_k]
        else:
            top_idx = np.arange(n_obligations)
        # Sort the selected top-K by descending cosine for deterministic output.
        top_idx = top_idx[np.argsort(-row[top_idx])]
        for j in top_idx:
            score = float(row[j])
            if score < similarity_threshold:
                continue
            pairs.append((cid, o_ids[j], score))
    return pairs


# ---------------------------------------------------------------------------
# CLI helper — useful for one-off prefilter runs outside the orchestrator
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    import csv
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comments-parquet", type=Path, required=True)
    ap.add_argument("--obligations-csv", type=Path, required=True)
    ap.add_argument("--docket-id", type=str, required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL: one {comment_id, obligation_id, cosine} per line.")
    args = ap.parse_args()

    import pandas as pd
    cdf = pd.read_parquet(args.comments_parquet)
    cdf = cdf[cdf["docket_id"] == args.docket_id]
    comments = cdf.to_dict("records")

    with args.obligations_csv.open("r", encoding="utf-8") as f:
        obligations = list(csv.DictReader(f))

    pairs = prefilter_top_k(
        comments, obligations,
        k=args.k, similarity_threshold=args.threshold,
        docket_id=args.docket_id,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for cid, oid, sc in pairs:
            f.write(json.dumps({"comment_id": cid, "obligation_id": oid,
                                "cosine": sc}) + "\n")
    print(f"[stage4-prefilter] wrote {len(pairs)} pairs to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
