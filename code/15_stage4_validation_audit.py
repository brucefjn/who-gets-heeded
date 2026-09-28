"""
15_stage4_validation_audit.py — Stage 4 reliability audit.

Two modes:

(1) `--sample` (default): build a 100-pair hand-coding template from one
    anchor's Stage 4 matches. Stratified 50/50 across the LLM's
    addressed=True / addressed=False; within addressed=True, stratified
    across the three stance values. Writes a CSV with empty BRUCE_* and
    YUE_* columns for the two human raters plus RECONCILED_* columns for
    the post-disagreement adjudication.

(2) `--compute-kappa`: given a fully-coded CSV, compute Cohen's κ on the
    binary `addressed` and per-class κ on the 4-value `stance`, both
    Bruce-vs-Yue and LLM-vs-RECONCILED. Reports Wilson 95% CIs via
    n=1000 bootstrap. Writes a markdown report.

PROJECT_FACTS §9b is explicit that we report descriptively and DO NOT
enforce a κ threshold for Stage 4 — Dong et al. 2026 IPM (Fleiss' κ
0.54–0.67) is the calibration anchor for "sub-0.67 κ is publishable".
The script does not raise on low κ.

Usage:
    # Sample mode (default)
    python code/15_stage4_validation_audit.py \\
        --anchor EPA-HQ-OAR-2018-0775 --n-pairs 100

    # Compute kappa from a reconciled CSV
    python code/15_stage4_validation_audit.py \\
        --compute-kappa --input data/processed/stage4_validation_audit_coded.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from pathlib import Path
from typing import Optional

DEFAULT_ANCHOR = "EPA-HQ-OAR-2018-0775"
DEFAULT_N_PAIRS = 100
DEFAULT_SAMPLE_OUT = Path("data/processed/stage4_validation_audit_template.csv")
DEFAULT_KAPPA_REPORT = Path("data/processed/stage4_validation_audit_results.md")
DEFAULT_MATCHES_DIR = Path("data/processed/stage4")
DEFAULT_COMMENTS_GLOB = "data/processed/comments_augmented/EPA_*.parquet"

ALLOWED_STANCES = ("SUPPORTING", "OPPOSING", "SUGGESTING_MODIFICATION", "NONE")


# ---------------------------------------------------------------------------
# Sample mode
# ---------------------------------------------------------------------------
def _load_anchor_matches(anchor: str, matches_dir: Path):
    import pandas as pd
    path = matches_dir / f"{anchor}__matches.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No Stage 4 matches at {path}. Run 13_stage4_run_per_anchor on "
            f"{anchor} first."
        )
    return pd.read_parquet(path)


def _load_comment_text_index(anchor: str, comment_ids: list[str],
                             comments_glob: str) -> dict[str, str]:
    import pandas as pd
    out: dict[str, str] = {}
    needed = set(comment_ids)
    for path in sorted(glob.glob(comments_glob)):
        df = pd.read_parquet(path, columns=["document_id", "docket_id",
                                            "comment"])
        df = df[(df["docket_id"] == anchor) & df["document_id"].isin(needed)]
        for _, row in df.iterrows():
            out[str(row["document_id"])] = str(row["comment"] or "")
        if len(out) == len(needed):
            break
    return out


def _load_obligation_text_index(anchor: str) -> dict[str, str]:
    """Load proposed AND final verified CSVs so audit text resolves
    whichever rule_type the matches came from."""
    out: dict[str, str] = {}
    for rule in ("proposed", "final"):
        path = Path("data/processed") / (
            f"path_a_obligations_verified_{anchor}_{rule}.csv")
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cidx = row.get("candidate_idx")
                sidx = row.get("split_idx", "0")
                if cidx is None:
                    continue
                oid = f"{anchor}__{rule}__{int(cidx)}_{int(sidx)}"
                out[oid] = str(row.get("obligation_text") or "")
    return out


def sample_pairs(
    anchor: str,
    n_pairs: int,
    matches_dir: Path,
    out_path: Path,
    *,
    seed: int = 20260512,
    comments_glob: str = DEFAULT_COMMENTS_GLOB,
) -> int:
    """Sample n_pairs (50 yes / 50 no, stratified by stance within yes)
    and write the hand-coding template CSV. Returns the number of rows
    actually written (may be < n_pairs if a stratum is exhausted)."""
    import pandas as pd
    df = _load_anchor_matches(anchor, matches_dir)
    rng = random.Random(seed)

    half = n_pairs // 2
    yes = df[df["addressed"] == True].copy()    # noqa: E712
    no = df[df["addressed"] == False].copy()    # noqa: E712

    # No stratum: random
    no_n = min(len(no), n_pairs - half)
    no_sample = no.sample(
        n=no_n, random_state=seed) if no_n else no.iloc[0:0]

    # Yes stratum: balance across the 3 informative stances; NONE is
    # rare here (it's mostly addressed=False) but allowed so we capture
    # any addressed-yes / stance-unclear cases the LLM produced.
    yes_strata = {s: yes[yes["stance"] == s] for s in ALLOWED_STANCES}
    target_per = max(1, half // 4)
    yes_chunks = []
    remaining = half
    for s in ALLOWED_STANCES:
        bucket = yes_strata[s]
        take = min(target_per, len(bucket), remaining)
        if take > 0:
            yes_chunks.append(bucket.sample(n=take, random_state=seed))
            remaining -= take
    # Top up from the largest stance bucket if we under-filled.
    if remaining > 0 and yes_chunks:
        leftover = yes.drop(pd.concat(yes_chunks).index, errors="ignore")
        top_up = leftover.sample(n=min(remaining, len(leftover)),
                                 random_state=seed)
        yes_chunks.append(top_up)
    yes_sample = pd.concat(yes_chunks) if yes_chunks else yes.iloc[0:0]

    sampled = pd.concat([yes_sample, no_sample], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(sampled) > n_pairs:
        sampled = sampled.iloc[:n_pairs]

    comment_ids = sampled["comment_id"].astype(str).tolist()
    obligation_ids = sampled["obligation_id"].astype(str).tolist()
    comment_text = _load_comment_text_index(anchor, comment_ids, comments_glob)
    obligation_text = _load_obligation_text_index(anchor)

    rows: list[dict] = []
    for i, (_, r) in enumerate(sampled.iterrows()):
        cid = str(r["comment_id"])
        oid = str(r["obligation_id"])
        rows.append({
            "pair_id": f"{anchor}__pair_{i:04d}",
            "comment_id": cid,
            "obligation_id": oid,
            "comment_text": (comment_text.get(cid, "")[:4000]),
            "obligation_text": obligation_text.get(oid, ""),
            "llm_addressed": bool(r["addressed"]),
            "llm_stance": str(r["stance"]),
            "llm_justification": str(r.get("justification", "")),
            "BRUCE_ADDRESSED": "",
            "BRUCE_STANCE": "",
            "BRUCE_NOTES": "",
            "YUE_ADDRESSED": "",
            "YUE_STANCE": "",
            "YUE_NOTES": "",
            "RECONCILED_ADDRESSED": "",
            "RECONCILED_STANCE": "",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "pair_id", "comment_id", "obligation_id", "comment_text",
        "obligation_text", "llm_addressed", "llm_stance",
        "llm_justification", "BRUCE_ADDRESSED", "BRUCE_STANCE",
        "BRUCE_NOTES", "YUE_ADDRESSED", "YUE_STANCE", "YUE_NOTES",
        "RECONCILED_ADDRESSED", "RECONCILED_STANCE",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[audit] wrote {len(rows)} pairs to {out_path}", file=sys.stderr)
    return len(rows)


# ---------------------------------------------------------------------------
# Kappa mode
# ---------------------------------------------------------------------------
def _normalize_bool(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def _normalize_stance(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper()
    return s if s in ALLOWED_STANCES else None


def cohen_kappa_binary(a: list[bool], b: list[bool]) -> float:
    """Cohen's κ for two binary raters. Returns NaN if there's only one
    observed class (κ is undefined when chance agreement = 1)."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a_true = sum(1 for x in a if x) / n
    p_b_true = sum(1 for x in b if x) / n
    p_chance = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if abs(p_chance - 1.0) < 1e-12:
        return float("nan")
    return (p_obs - p_chance) / (1.0 - p_chance)


def cohen_kappa_per_class(
    a: list[str], b: list[str], classes: tuple[str, ...] = ALLOWED_STANCES,
) -> dict[str, float]:
    """Per-class κ (one-vs-rest, treating each class as a binary)."""
    out: dict[str, float] = {}
    for cls in classes:
        a_bin = [x == cls for x in a]
        b_bin = [x == cls for x in b]
        out[cls] = cohen_kappa_binary(a_bin, b_bin)
    return out


def bootstrap_kappa_ci(
    a: list, b: list, stat_fn, n_bootstrap: int = 1000, seed: int = 20260512,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic over paired data. Returns
    (lo, hi) at 2.5/97.5 percentile."""
    rng = random.Random(seed)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    vals: list[float] = []
    a_arr = list(a)
    b_arr = list(b)
    for _ in range(n_bootstrap):
        idxs = [rng.randint(0, n - 1) for _ in range(n)]
        a_s = [a_arr[i] for i in idxs]
        b_s = [b_arr[i] for i in idxs]
        try:
            v = stat_fn(a_s, b_s)
        except Exception:
            v = float("nan")
        if v == v:   # not NaN
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[max(0, int(0.025 * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)) - 1)]
    return (lo, hi)


def compute_kappa_report(coded_csv: Path, report_path: Path) -> None:
    """Read a fully-coded audit CSV and write the markdown κ report."""
    with coded_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {coded_csv}")

    def col_bool(row, key):
        return _normalize_bool(row.get(key))

    def col_stance(row, key):
        return _normalize_stance(row.get(key))

    # Build paired lists, skipping rows missing any required column.
    bruce_a, yue_a = [], []
    bruce_s, yue_s = [], []
    llm_a, rec_a = [], []
    llm_s, rec_s = [], []
    skipped = 0
    for r in rows:
        b_addr = col_bool(r, "BRUCE_ADDRESSED")
        y_addr = col_bool(r, "YUE_ADDRESSED")
        b_st = col_stance(r, "BRUCE_STANCE")
        y_st = col_stance(r, "YUE_STANCE")
        l_addr = col_bool(r, "llm_addressed")
        l_st = col_stance(r, "llm_stance")
        r_addr = col_bool(r, "RECONCILED_ADDRESSED")
        r_st = col_stance(r, "RECONCILED_STANCE")
        if b_addr is None or y_addr is None or b_st is None or y_st is None:
            skipped += 1
            continue
        bruce_a.append(b_addr); yue_a.append(y_addr)
        bruce_s.append(b_st); yue_s.append(y_st)
        if l_addr is not None and r_addr is not None:
            llm_a.append(l_addr); rec_a.append(r_addr)
        if l_st is not None and r_st is not None:
            llm_s.append(l_st); rec_s.append(r_st)

    if not bruce_a:
        raise ValueError("No fully-coded rows; check BRUCE_*/YUE_* columns.")

    k_addr_by = cohen_kappa_binary(bruce_a, yue_a)
    k_addr_by_ci = bootstrap_kappa_ci(bruce_a, yue_a, cohen_kappa_binary)
    k_stance_by = cohen_kappa_per_class(bruce_s, yue_s)
    k_stance_by_ci = {
        cls: bootstrap_kappa_ci(
            [x == cls for x in bruce_s],
            [x == cls for x in yue_s],
            cohen_kappa_binary,
        )
        for cls in ALLOWED_STANCES
    }

    k_addr_lr = (cohen_kappa_binary(llm_a, rec_a)
                 if llm_a else float("nan"))
    k_addr_lr_ci = (bootstrap_kappa_ci(llm_a, rec_a, cohen_kappa_binary)
                    if llm_a else (float("nan"), float("nan")))
    k_stance_lr = (cohen_kappa_per_class(llm_s, rec_s)
                   if llm_s else {c: float("nan") for c in ALLOWED_STANCES})
    k_stance_lr_ci = {
        cls: (bootstrap_kappa_ci(
            [x == cls for x in llm_s], [x == cls for x in rec_s],
            cohen_kappa_binary) if llm_s else (float("nan"), float("nan")))
        for cls in ALLOWED_STANCES
    }

    def _fmt(v):
        return "nan" if v != v else f"{v:.3f}"

    def _fmt_ci(lo_hi):
        lo, hi = lo_hi
        return f"[{_fmt(lo)}, {_fmt(hi)}]"

    lines: list[str] = []
    lines.append("# Stage 4 validation audit — Cohen's κ report\n")
    lines.append(f"Source CSV: `{coded_csv}`\n")
    lines.append(f"N pairs analyzed: **{len(bruce_a)}** (skipped {skipped} "
                 "rows with missing required columns)\n")
    lines.append("\n## Bruce vs Yue (inter-rater reliability)\n")
    lines.append(f"- `addressed` Cohen's κ: **{_fmt(k_addr_by)}** "
                 f"(95% CI: {_fmt_ci(k_addr_by_ci)}, bootstrap n=1000)\n")
    lines.append("- `stance` per-class κ:\n")
    for cls in ALLOWED_STANCES:
        lines.append(f"    - {cls}: **{_fmt(k_stance_by[cls])}** "
                     f"(95% CI: {_fmt_ci(k_stance_by_ci[cls])})\n")

    lines.append("\n## LLM vs RECONCILED (human consensus)\n")
    if llm_a:
        lines.append(f"- `addressed` Cohen's κ: **{_fmt(k_addr_lr)}** "
                     f"(95% CI: {_fmt_ci(k_addr_lr_ci)}, n={len(llm_a)})\n")
        lines.append("- `stance` per-class κ:\n")
        for cls in ALLOWED_STANCES:
            lines.append(f"    - {cls}: **{_fmt(k_stance_lr[cls])}** "
                         f"(95% CI: {_fmt_ci(k_stance_lr_ci[cls])})\n")
    else:
        lines.append("- (no LLM/RECONCILED rows available — fill the "
                     "RECONCILED_* columns after adjudicating disagreements.)\n")

    lines.append("\n## Interpretation\n")
    lines.append("Stage 4 reports κ descriptively per Dong et al. 2026 IPM "
                 "(Fleiss' κ 0.54–0.67 on summarization-quality dimensions) "
                 "as the calibration anchor for the sub-0.67 publishability "
                 "envelope (PROJECT_FACTS §9b). No pass/fail threshold is "
                 "enforced here.\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[audit] wrote κ report to {report_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR)
    ap.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    ap.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_OUT)
    ap.add_argument("--comments-glob", default=DEFAULT_COMMENTS_GLOB)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--compute-kappa", action="store_true",
                    help="Compute κ from a reconciled CSV instead of sampling.")
    ap.add_argument("--input", type=Path, default=None,
                    help="Reconciled-coded CSV (required with --compute-kappa).")
    ap.add_argument("--report-out", type=Path, default=DEFAULT_KAPPA_REPORT)
    args = ap.parse_args()

    if args.compute_kappa:
        if args.input is None or not args.input.exists():
            print("--compute-kappa requires --input pointing to a reconciled "
                  "coded CSV.", file=sys.stderr)
            return 2
        compute_kappa_report(args.input, args.report_out)
        return 0

    if args.n_pairs > 100:
        print(f"[audit] WARN: n-pairs={args.n_pairs} > 100; we de-scoped "
              "from 500 to 100, but honoring the user override.",
              file=sys.stderr)
    sample_pairs(
        args.anchor, args.n_pairs, args.matches_dir, args.output,
        seed=args.seed, comments_glob=args.comments_glob,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
