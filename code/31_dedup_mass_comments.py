"""
31_dedup_mass_comments.py - Findings 1 and 3 with one vote per text cluster.

Robustness check requested by EAAMO 2026 reviewer DRNX ("run a deduplication
robustness check for mass-comment campaigns") and listed in the paper's
Limitations. Engagement counts in the paper treat every comment as an
independent voice, so a form-letter campaign can push an obligation over the
five-commenter threshold. Here every cluster of identical or near-identical
comments counts once.

Two clustering rules, both applied within each docket to the comments that
address at least one obligation:
  EXACT  identical text after normalization (lowercase, alphanumeric tokens).
  NEAR   Jaccard similarity >= 0.7 on 3-word shingles, which tolerates the
         personalization typical of form letters (a name, an address, a
         changed word). Candidate pairs come from MinHash-LSH (128
         permutations, 32 bands x 4 rows) and every candidate pair is
         verified with the exact Jaccard similarity.
         Clusters are connected components (union-find), so EXACT clusters
         are always nested inside NEAR clusters.

Finding 1 (proposed-side obligations, final-only NEW excluded, as in the
paper): an obligation is high-engagement when it is addressed by >= 5
distinct clusters.
Finding 3: an obligation is organizational-majority when more than half of
its addressing clusters are organizational. A cluster takes the majority
commenter type of its members (ties count as individual). Reported with
classifier-assigned (pre-audit) and audit-corrected outcome labels.

Data loading, the commenter-type rule, and the regression code are reused
from code/22 so inputs and conventions are identical. Before deduplicating,
the script reproduces the published numbers and aborts if they drift.

Usage (repo root, on the machine with the production data):
    python3 code/31_dedup_mass_comments.py

Writes a JSON summary to data/processed/robustness_dedup_summary.json.
Required packages: pandas, pyarrow, numpy, scipy, statsmodels.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import pathlib
import re
import sys
import zlib
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
AUDIT_OUTCOMES = "data/processed/path_a_obligation_outcomes_audit_corrected.parquet"
SUMMARY_JSON = "data/processed/robustness_dedup_summary.json"

REVISED = ["SURVIVED-edited", "MODIFIED", "DROPPED"]
HIGH = 5                      # high-engagement threshold used in the paper

SHINGLE = 3                   # words per shingle
JACCARD = 0.7                 # near-duplicate threshold
NUM_PERM = 128
BANDS, ROWS = 32, 4           # BANDS * ROWS == NUM_PERM
ALL_PAIRS_MAX = 60            # compare all pairs in LSH buckets up to this size
SEED = 20260927

TOKEN_RE = re.compile(r"[a-z0-9]+")
_P = np.uint64((1 << 31) - 1)            # Mersenne prime 2^31 - 1
_rng = np.random.RandomState(SEED)
_A = _rng.randint(1, (1 << 31) - 1, size=NUM_PERM).astype(np.uint64)
_B = _rng.randint(0, (1 << 31) - 1, size=NUM_PERM).astype(np.uint64)


# ---------------------------------------------------------------------------
# Text clustering (pure functions; no dependency on code/22)
# ---------------------------------------------------------------------------
def tokens(text) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def shingles(toks: list[str]) -> set[str]:
    if not toks:
        return set()
    if len(toks) < SHINGLE:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def minhash(sh: set[str], chunk: int = 4096) -> np.ndarray:
    """MinHash signature. crc32 values are < 2^32 and _A < 2^31, so the
    products stay below 2^63. Long documents are processed in chunks to
    keep memory flat."""
    mins = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    buf: list[int] = []

    def _flush(values: list[int]) -> None:
        nonlocal mins
        hv = np.asarray(values, dtype=np.uint64)
        mins = np.minimum(mins, ((np.outer(hv, _A) + _B) % _P).min(axis=0))

    for s in sh:
        buf.append(zlib.crc32(s.encode("utf-8")))
        if len(buf) == chunk:
            _flush(buf)
            buf = []
    if buf:
        _flush(buf)
    return mins


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def cluster_texts(texts: list[str]) -> tuple[list[int], list[int]]:
    """Return (exact_ids, near_ids) for a list of texts from one docket.

    Texts with no alphanumeric content each get their own cluster so that
    empty strings are never merged into one artificial cluster."""
    toks = [tokens(t) for t in texts]
    exact_index: dict[str, int] = {}
    exact_ids: list[int] = []
    for i, tk in enumerate(toks):
        key = " ".join(tk) if tk else f"__empty__{i}"
        exact_ids.append(exact_index.setdefault(key, len(exact_index)))

    # One representative text per exact cluster.
    rep_of_exact: dict[int, int] = {}
    for i, e in enumerate(exact_ids):
        rep_of_exact.setdefault(e, i)
    exact_list = sorted(rep_of_exact)                 # exact ids in order
    reps = [rep_of_exact[e] for e in exact_list]
    pos_of_exact = {e: k for k, e in enumerate(exact_list)}

    shs = [shingles(toks[r]) for r in reps]
    uf = UnionFind(len(reps))
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for k, sh in enumerate(shs):
        if not sh:
            continue
        sig = minhash(sh)
        for b in range(BANDS):
            buckets[(b, sig[b * ROWS:(b + 1) * ROWS].tobytes())].append(k)

    checked: set[tuple[int, int]] = set()

    def _try(i: int, j: int) -> None:
        pair = (i, j) if i < j else (j, i)
        if pair in checked:
            return
        checked.add(pair)
        if jaccard(shs[i], shs[j]) >= JACCARD:
            uf.union(i, j)

    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) <= ALL_PAIRS_MAX:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    _try(members[x], members[y])
        else:
            anchor = members[0]
            for x in range(1, len(members)):
                _try(anchor, members[x])
                _try(members[x - 1], members[x])

    near_root = [uf.find(pos_of_exact[e]) for e in exact_ids]
    relabel: dict[int, int] = {}
    near_ids = [relabel.setdefault(r, len(relabel)) for r in near_root]
    return exact_ids, near_ids


def cluster_comments(texts_df):
    """texts_df: document_id, docket_id, comment. Adds cluster_exact, cluster_near."""
    import pandas as pd
    parts = []
    for docket, g in texts_df.groupby("docket_id", sort=True):
        exact_ids, near_ids = cluster_texts(g["comment"].tolist())
        parts.append(pd.DataFrame({
            "document_id": g["document_id"].values,
            "docket_id": docket,
            "cluster_exact": [f"{docket}::e{i}" for i in exact_ids],
            "cluster_near": [f"{docket}::n{i}" for i in near_ids],
        }))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-obligation aggregation with one vote per cluster
# ---------------------------------------------------------------------------
def dedup_aggregate(addressed, df_outcomes, kinds, clusters, cluster_col, attach_docket):
    """Same columns as code/22's _build_per_obligation_aggregate
    (n_addressing, n_org, n_ind, org_share, docket_id), counted by cluster."""
    a = addressed[["comment_id", "obligation_id"]].drop_duplicates()
    a = a.merge(clusters[["document_id", cluster_col]],
                left_on="comment_id", right_on="document_id", how="left")
    missing = a[cluster_col].isna()
    a.loc[missing, cluster_col] = "solo::" + a.loc[missing, "comment_id"].astype(str)
    a = a.merge(kinds[["document_id", "submitter_kind"]].rename(
        columns={"document_id": "_doc"}), left_on="comment_id", right_on="_doc", how="left")

    # Cluster type = majority type of its members (ties -> individual).
    members = a[["comment_id", cluster_col, "submitter_kind"]].drop_duplicates("comment_id")
    share_org = members.groupby(cluster_col)["submitter_kind"].apply(
        lambda s: (s == "organizational").mean())
    cluster_kind = share_org.gt(0.5).map({True: "organizational", False: "individual"})
    a["cluster_kind"] = a[cluster_col].map(cluster_kind)

    per = a[["obligation_id", cluster_col, "cluster_kind"]].drop_duplicates(
        ["obligation_id", cluster_col])
    agg = per.groupby("obligation_id").agg(
        n_addressing=(cluster_col, "nunique"),
        n_org=("cluster_kind", lambda s: int((s == "organizational").sum())),
        n_ind=("cluster_kind", lambda s: int((s == "individual").sum())),
    ).reset_index()
    agg["org_share"] = agg["n_org"] / (agg["n_org"] + agg["n_ind"]).replace(0, 1)
    out = df_outcomes.merge(agg, on="obligation_id", how="left").fillna(
        {"n_addressing": 0, "n_org": 0, "n_ind": 0, "org_share": 0.0})
    return attach_docket(out)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def f1_stats(out, c22) -> dict:
    stats = c22.stats
    d = out[out["outcome_state"] != "NEW"].copy()
    d["high"] = d["n_addressing"] >= HIGH
    d["rev"] = d["outcome_state"].isin(REVISED)
    ct = c22.pd.crosstab(d["high"], d["rev"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = int(ct.values.sum())
    t1 = c22.test_1_f1_clustered(d)
    return {
        "n": n, "n_high": int(d["high"].sum()),
        "rate_high": float(d.loc[d["high"], "rev"].mean() * 100),
        "rate_low": float(d.loc[~d["high"], "rev"].mean() * 100),
        "chi2": float(chi2), "p": float(p), "V": float((chi2 / n) ** 0.5),
        "fe_OR": t1["OR"], "fe_ci": [t1["ci_lo"], t1["ci_hi"]], "fe_p": t1["p"],
        "k_dockets": t1.get("k_dockets"),
    }


def _cmh(subset) -> dict:
    from statsmodels.stats.contingency_tables import StratifiedTable
    tables = []
    for _, g in subset.groupby("docket_id"):
        se = g["outcome_state"] == "SURVIVED-edited"
        om = g["org_majority"]
        a, b = int((se & om).sum()), int((se & ~om).sum())
        c, d = int((~se & om).sum()), int((~se & ~om).sum())
        if a + b + c + d < 2 or 0 in (a + b, c + d, a + c, b + d):
            continue
        tables.append(np.array([[a, b], [c, d]]))
    if not tables:
        return {"OR": float("nan"), "p": float("nan"), "strata": 0}
    st = StratifiedTable(tables)
    lo, se_ = float(st.logodds_pooled), float(st.logodds_pooled_se)
    return {"OR": float(st.oddsratio_pooled),
            "ci": [float(np.exp(lo - 1.96 * se_)), float(np.exp(lo + 1.96 * se_))],
            "p": float(st.test_null_odds().pvalue), "strata": len(tables),
            "n": int(sum(t.sum() for t in tables))}


def f3_stats(out, c22) -> dict:
    high = out[out["n_addressing"] >= HIGH].copy()
    high["org_majority"] = high["org_share"] > 0.5
    sub = high[high["outcome_state"].isin(["SURVIVED-edited", "MODIFIED"])].copy()
    se = sub["outcome_state"] == "SURVIVED-edited"
    om = sub["org_majority"]
    a, b = int((se & om).sum()), int((se & ~om).sum())
    c, d = int((~se & om).sum()), int((~se & ~om).sum())
    OR, lo, hi, p = c22.fisher_exact_2x2(a, b, c, d)
    t2 = c22.test_2_f3_clustered(out)
    return {
        "cells": [a, b, c, d], "n": a + b + c + d,
        "pct_org_SE": 100 * a / max(a + b, 1), "pct_org_MO": 100 * c / max(c + d, 1),
        "fisher_OR": OR, "wald_ci": [lo, hi], "fisher_p": p,
        "clustered_OR": t2["OR"], "clustered_ci": [t2["ci_lo"], t2["ci_hi"]],
        "clustered_p": t2["p"], "cmh": _cmh(sub),
    }


def _f(x, nd=2) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _p(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.2e}" if x < 0.001 else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "c22", HERE / "22_clustered_logistic_f1_f3.py")
    c22 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c22)
    pd = c22.pd

    addressed, df_outcomes, kinds = c22._load_data(c22.DEFAULT_OUTCOMES)
    out0 = c22._build_per_obligation_aggregate(addressed, df_outcomes, kinds)

    # --- Baseline checks: reproduce the published numbers before deduplicating.
    base_f1 = f1_stats(out0, c22)
    if base_f1["n"] != 12243 or abs(base_f1["chi2"] - 19.71) > 0.05:
        sys.exit(f"ABORT: Finding 1 baseline drifted: n={base_f1['n']}, "
                 f"chi2={base_f1['chi2']:.2f} (expected 12243, 19.71)")
    c22.test_0b_baseline(out0)                      # aborts unless (43,15,28,30)
    base_f3 = f3_stats(out0, c22)
    audit_available = pathlib.Path(AUDIT_OUTCOMES).exists()
    if audit_available:
        df_audit = pd.read_parquet(AUDIT_OUTCOMES)
        out0_a = c22._build_per_obligation_aggregate(addressed, df_audit, kinds)
        base_f3a = f3_stats(out0_a, c22)
        if base_f3a["cells"] != [13, 2, 57, 43]:
            sys.exit(f"ABORT: audit-corrected Finding 3 baseline drifted: "
                     f"{base_f3a['cells']} (expected [13, 2, 57, 43])")
    print("Baseline checks passed (Finding 1 n=12,243, chi2=19.71; Finding 3 "
          "cells 43/15/28/30 and audit-corrected 13/2/57/43).", file=sys.stderr)

    # --- Load text for every comment that addresses at least one obligation.
    needed = set(addressed["comment_id"].astype(str))
    parts = []
    cols = ["document_id", "docket_id", "comment"]
    for pq in sorted(glob.glob(c22.COMMENTS_GLOB)):
        try:   # row filter at read time keeps memory low
            df = pd.read_parquet(pq, columns=cols,
                                 filters=[("document_id", "in", sorted(needed))])
        except Exception:  # noqa: BLE001 - fall back to a full read
            df = pd.read_parquet(pq, columns=cols)
        df = df[df["document_id"].astype(str).isin(needed)]
        if not df.empty:
            parts.append(df)
    texts = pd.concat(parts, ignore_index=True).drop_duplicates("document_id")
    texts["document_id"] = texts["document_id"].astype(str)
    print(f"Clustering {len(texts):,} addressing comments across "
          f"{texts['docket_id'].nunique()} dockets ...", file=sys.stderr)
    clusters = cluster_comments(texts)

    n_exact = clusters["cluster_exact"].nunique()
    n_near = clusters["cluster_near"].nunique()
    sizes = clusters["cluster_near"].value_counts()
    in_multi = int(sizes[sizes > 1].sum())
    print("\n=== Mass-comment structure among addressing comments ===")
    print(f"  addressing comments:            {len(clusters):,}")
    print(f"  exact-duplicate clusters:       {n_exact:,}")
    print(f"  near-duplicate clusters:        {n_near:,}")
    print(f"  comments in multi-member near clusters: {in_multi:,} "
          f"({100 * in_multi / max(len(clusters), 1):.1f}%)")
    top = sizes.head(5)
    for cid, size in top.items():
        print(f"    largest: {cid.split('::')[0]}  size {size:,}")

    kinds = kinds.copy()
    kinds["document_id"] = kinds["document_id"].astype(str)
    addr = addressed.copy()
    addr["comment_id"] = addr["comment_id"].astype(str)

    results = {"baseline": {"f1": base_f1, "f3_pre_audit": base_f3},
               "mass_comments": {"addressing_comments": int(len(clusters)),
                                 "exact_clusters": int(n_exact),
                                 "near_clusters": int(n_near),
                                 "comments_in_multi_member_near_clusters": in_multi}}
    if audit_available:
        results["baseline"]["f3_audit_corrected"] = base_f3a

    for rule, col in (("EXACT", "cluster_exact"), ("NEAR", "cluster_near")):
        out_d = dedup_aggregate(addr, df_outcomes, kinds, clusters, col,
                                c22._attach_docket_id)
        r = {"f1": f1_stats(out_d, c22), "f3_pre_audit": f3_stats(out_d, c22)}
        if audit_available:
            out_da = dedup_aggregate(addr, df_audit, kinds, clusters, col,
                                     c22._attach_docket_id)
            r["f3_audit_corrected"] = f3_stats(out_da, c22)
        results[rule] = r

    # --- Print comparison tables.
    print("\n=== Finding 1 (proposed-side obligations; high = >= 5 addressing units) ===")
    print(f"{'count':<22}{'n_high':>7}{'high %':>8}{'low %':>8}{'gap pp':>8}"
          f"{'chi2':>8}{'p':>10}{'V':>7}{'FE OR [95% CI]':>22}{'FE p':>8}")
    rows = [("comments (paper)", results["baseline"]["f1"]),
            ("exact clusters", results["EXACT"]["f1"]),
            ("near clusters", results["NEAR"]["f1"])]
    for label, s in rows:
        ci = f"{_f(s['fe_OR'])} [{_f(s['fe_ci'][0])}, {_f(s['fe_ci'][1])}]"
        print(f"{label:<22}{s['n_high']:>7}{s['rate_high']:>8.1f}{s['rate_low']:>8.1f}"
              f"{s['rate_high'] - s['rate_low']:>8.1f}{s['chi2']:>8.2f}{_p(s['p']):>10}"
              f"{s['V']:>7.3f}{ci:>22}{_p(s['fe_p']):>8}")

    for key, title in (("f3_pre_audit", "classifier-assigned outcomes"),
                       ("f3_audit_corrected", "audit-corrected outcomes")):
        if key not in results["baseline"]:
            continue
        print(f"\n=== Finding 3 ({title}) ===")
        print(f"{'count':<22}{'cells SEo/SEn/MOo/MOn':>24}{'n':>5}{'%org SE':>9}"
              f"{'%org MO':>9}{'Fisher OR':>11}{'p':>8}{'clust. OR':>11}{'p':>8}"
              f"{'CMH OR':>9}{'p':>8}")
        for label, src in (("comments (paper)", results["baseline"]),
                           ("exact clusters", results["EXACT"]),
                           ("near clusters", results["NEAR"])):
            s = src[key]
            cells = "/".join(str(x) for x in s["cells"])
            print(f"{label:<22}{cells:>24}{s['n']:>5}{s['pct_org_SE']:>9.1f}"
                  f"{s['pct_org_MO']:>9.1f}{_f(s['fisher_OR']):>11}{_p(s['fisher_p']):>8}"
                  f"{_f(s['clustered_OR']):>11}{_p(s['clustered_p']):>8}"
                  f"{_f(s['cmh']['OR']):>9}{_p(s['cmh']['p']):>8}")

    pathlib.Path(SUMMARY_JSON).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {SUMMARY_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
