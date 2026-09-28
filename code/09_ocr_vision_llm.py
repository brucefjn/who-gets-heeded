"""
09_ocr_vision_llm.py — Stage 2 OCR fallback via Claude vision API.

PURPOSE
  Recover the residual after pytesseract Stage 1: low-quality results
  (flagged by 08) plus any rows where pytesseract crashed entirely.
  Optionally include the originally-non-recoverable .doc files.

DECISION POINT (Bruce, before running)
  Per spot-check protocol in attachment_failure_diagnostic.md:
    Path A: Stage-1 pytesseract quality is comparable to pdfplumber on the
            spot-check sample → run THIS script only on flagged + failed
            (Stage-2-targeted set, ~30-50 files).
    Path B: Stage-1 quality is degraded enough to risk Stage-1 LLM →
            re-run THIS script on ALL 148 candidates (--mode all).
  Default: Path A (--mode targeted).

PIPELINE
  For each target file:
    1. Render PDF pages → PNG (pypdfium2, 150 DPI to keep image tokens bounded).
    2. Send each page as base64 PNG to Claude vision API with verbatim
       transcription prompt.
    3. Concatenate page transcriptions → write to <doc_id>__1.txt
       (overwrites Stage-1 result if Stage-1 was low-quality).
    4. Track tokens, runtime, and per-page status.

OUTPUTS
  data/processed/ocr_stage2_log.csv          per-attachment Stage-2 detail
  data/processed/attachment_extraction_log.csv  master log: kind='vision_llm'
  data/raw/attachments/<docket>/<doc_id>__1.txt  vision-LLM transcription

DEPENDENCIES
    pip3 install --user anthropic pypdfium2 pillow
  Env var:
    ANTHROPIC_API_KEY=<your-key>

CITATIONS (rigor: novel-but-defensible)
  Anthropic (2024). Claude 3 Model Card.
    https://www.anthropic.com/news/claude-3-family
  [TBD: add specific 2024-2025 paper on LLM document understanding /
   vision-based OCR comparison once Bruce/Yue selects.]

USAGE
  export ANTHROPIC_API_KEY='...'
  python3 code/09_ocr_vision_llm.py [--mode targeted|all] [--model NAME] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACH = PROJECT_ROOT / "data" / "raw" / "attachments"
LOG_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_extraction_log.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "attachment_sample.csv"
STAGE1_LOG = PROJECT_ROOT / "data" / "processed" / "ocr_stage1_log.csv"
STAGE2_LOG = PROJECT_ROOT / "data" / "processed" / "ocr_stage2_log.csv"

DEFAULT_MODEL = "claude-sonnet-4-5"
RENDER_DPI = 150  # lower DPI than Stage 1 to keep image tokens manageable
MAX_OUTPUT_TOKENS = 4000  # per-page response cap

TRANSCRIBE_PROMPT = """Transcribe all text content visible in this PDF page image.

Output requirements:
1. Preserve paragraph structure with blank lines between paragraphs.
2. Preserve line breaks within addresses, signature blocks, lists, and tables.
3. Do not add commentary, summaries, or "[image of...]" descriptions.
4. If the page has handwritten content, transcribe it verbatim.
5. If a portion is illegible, write [illegible] in place of that portion.
6. If the page is mostly blank or contains only a stamp/letterhead, transcribe whatever is present.
7. Return only the transcribed text, no other content.
"""


def pdf_pages_as_b64(pdf_path: Path, dpi: int = RENDER_DPI) -> list[str]:
    import pypdfium2
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    out: list[str] = []
    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            try:
                bitmap = page.render(scale=dpi / 72)
                pil = bitmap.to_pil()
            finally:
                page.close()
            buf = io.BytesIO()
            pil.save(buf, format="PNG", optimize=True)
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    finally:
        pdf.close()
    return out


def transcribe_page(client, model: str, b64_image: str) -> tuple[str, int, int]:
    """Single-page transcription. Returns (text, in_tokens, out_tokens)."""
    msg = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64_image
                }},
                {"type": "text", "text": TRANSCRIBE_PROMPT},
            ],
        }],
    )
    text = msg.content[0].text if msg.content else ""
    return text, msg.usage.input_tokens, msg.usage.output_tokens


def select_targets(stage1: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`targeted` = Stage-1 low-quality + Stage-1 fails.
       `all`      = every Stage-1 candidate, regardless of Stage-1 outcome.
    """
    if mode == "all":
        return stage1.copy()
    return stage1[
        (
            (stage1["status"] == "ok")
            & (stage1["quality_flag"].astype(str).str.startswith("low_quality"))
        )
        | (stage1["status"].astype(str).str.startswith("ocr_fail"))
    ].copy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["targeted", "all"], default="targeted",
                   help="targeted: run on Stage-1 low-quality + failed only. "
                        "all: run on every Stage-1 candidate (when Stage-1 quality is poor).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0,
                   help="cap files for a smoke run; 0 = no cap.")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY env var.", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip3 install --user anthropic", file=sys.stderr)
        return 2

    if not STAGE1_LOG.exists():
        print(f"ERROR: Stage-1 log missing at {STAGE1_LOG} — run 08 first.", file=sys.stderr)
        return 2

    stage1 = pd.read_csv(STAGE1_LOG)
    targets = select_targets(stage1, args.mode)
    if args.limit > 0:
        targets = targets.head(args.limit)
    print(f"[stage 2] mode={args.mode}  model={args.model}  targets={len(targets)}")
    if len(targets) == 0:
        print("Nothing to do.")
        return 0

    client = anthropic.Anthropic(api_key=api_key)

    rows: list[dict] = []
    total_in = 0
    total_out = 0
    t_start = time.time()

    for i, (_, row) in enumerate(targets.iterrows(), 1):
        docid = row["document_id"]
        docket = row["docket_id"]
        fp = ATTACH / docket / f"{docid}__1.pdf"
        if not fp.exists():
            rows.append({"document_id": docid, "docket_id": docket,
                         "method": "vision_llm", "status": "file_missing",
                         "n_chars": 0, "n_pages": 0,
                         "in_tokens": 0, "out_tokens": 0, "runtime_sec": 0})
            continue

        try:
            b64_pages = pdf_pages_as_b64(fp)
        except Exception as e:
            rows.append({"document_id": docid, "docket_id": docket,
                         "method": "vision_llm",
                         "status": f"render_fail:{type(e).__name__}",
                         "n_chars": 0, "n_pages": 0,
                         "in_tokens": 0, "out_tokens": 0, "runtime_sec": 0})
            continue

        t0 = time.time()
        texts: list[str] = []
        in_tok = 0
        out_tok = 0
        page_status = "ok"
        for j, b64 in enumerate(b64_pages):
            for attempt in range(3):
                try:
                    t, ti, to = transcribe_page(client, args.model, b64)
                    texts.append(t)
                    in_tok += ti
                    out_tok += to
                    break
                except anthropic.RateLimitError:
                    sleep_s = 30 * (2 ** attempt)
                    print(f"    rate limit; sleep {sleep_s}s", file=sys.stderr)
                    time.sleep(sleep_s)
                except anthropic.APIError as e:
                    page_status = f"api_fail_p{j+1}:{type(e).__name__}"
                    break
            else:
                page_status = f"rate_limit_exhausted_p{j+1}"
                break

        full_text = "\n\n".join(texts)
        elapsed = round(time.time() - t0, 1)
        if texts:
            (fp.with_suffix(".txt")).write_text(full_text, encoding="utf-8")
        total_in += in_tok
        total_out += out_tok
        rows.append({
            "document_id": docid, "docket_id": docket,
            "method": "vision_llm",
            "status": "ok" if page_status == "ok" else page_status,
            "n_chars": len(full_text), "n_pages": len(b64_pages),
            "in_tokens": in_tok, "out_tokens": out_tok,
            "runtime_sec": elapsed,
        })
        if i % 5 == 0 or i == len(targets):
            print(f"  [{i:3d}/{len(targets)}] {time.time()-t_start:.0f}s elapsed; "
                  f"last={elapsed:.1f}s  in={in_tok:,}  out={out_tok:,}")

    s2 = pd.DataFrame(rows)
    s2.to_csv(STAGE2_LOG, index=False)
    print(f"\n[write] {STAGE2_LOG}")

    # Update master log
    log = pd.read_csv(LOG_PATH)
    ok = s2[s2["status"] == "ok"]
    for _, r in ok.iterrows():
        mask = log["document_id"] == r["document_id"]
        log.loc[mask, "status"] = "ok"
        log.loc[mask, "kind"] = "vision_llm"
        log.loc[mask, "n_chars"] = int(r["n_chars"])
        log.loc[mask, "info"] = f"method=vision_llm n_pages={int(r['n_pages'])} model={args.model}"
    log.to_csv(LOG_PATH, index=False)

    # Cost estimate (Sonnet 4.5: $3/M input, $15/M output as of 2025-09)
    cost = total_in * 3 / 1_000_000 + total_out * 15 / 1_000_000
    n_ok = (s2["status"] == "ok").sum()
    print(f"\n=== Stage 2 (vision LLM) summary ===")
    print(f"  succeeded         : {n_ok}/{len(s2)}")
    print(f"  total runtime     : {(time.time()-t_start)/60:.1f} min")
    print(f"  tokens            : {total_in:,} in + {total_out:,} out")
    print(f"  est. cost         : ${cost:.2f} (model={args.model})")
    print()
    print("Next: re-run 07_attachment_text_merge.py to fold OCR'd .txt into comments_augmented parquet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
