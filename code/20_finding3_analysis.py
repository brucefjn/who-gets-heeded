"""
20_finding3_analysis.py — Canonical script for Findings 1, 2, and 3 of the EAAMO paper.

Reproduces:
- Finding 1: engagement × revision chi-square (§6.3)
- Finding 2: opposing vs supporting direction-of-engagement z-test (§6.3)
- Finding 3: organizational-majority × outcome Fisher's exact (§6.4)
  -> 2x2 cells (43, 15, 28, 30), OR = 3.07, p = 0.007, n = 116

Recovered from chat-history paste 2026-05-15 evening; the original analysis was
run on May 15 ~01:00 PDT but the script was never committed to the repo. This
file is the canonical reproduction artifact going forward.

IMPORTANT — submitter_kind classification convention used here:
  This script uses a SIMPLER title-pattern classifier than the canonical
  `code/lib/submitter_classification.py`. The classifier is just:
      title contains "submitted by" | "on behalf" | "written by" → individual
      otherwise                                                  → organizational

  This produces 28.6% organizational base rate (vs 14.4% under the canonical
  classifier in code/lib/), matching the §6.4 paper-reported analysis exactly
  and matching the PROJECT_FACTS §2 distribution to the decimal point.

  The canonical classifier in code/lib/submitter_classification.py is more
  conservative (has anonymous-detection and an "other" category) and was built
  for a different pipeline step (`code/05_stratified_sample.py`). The two
  classifiers disagree on roughly 14% of comments — predominantly comments
  whose titles don't match the three individual-phrases but also don't contain
  recognized organizational keywords (e.g., "Letter from XYZ Corp on Proposed
  Rule," generic-format industry comment titles, anonymous submissions).

  Future-work refinement using the canonical classifier may yield different
  cell counts. See §8 of the paper for the limitation framing.

Inputs:
  data/processed/stage4/*.parquet                  (Stage 4 obligation-comment pairs)
  data/processed/path_a_obligation_outcomes.parquet (Path A outcome states)
  data/processed/comments_augmented/EPA_*.parquet  (per-comment titles)

Usage:
  python3 code/20_finding3_analysis.py

Output: printed statistics for all three findings (no file outputs).

Required packages: pandas, pyarrow, scipy, statsmodels
"""
from __future__ import annotations

import argparse
import glob
import sys

try:
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.proportion import proportions_ztest
except ImportError as e:
    sys.exit(f"ERROR: {e}. Required: pip install pandas pyarrow scipy statsmodels")


DEFAULT_OUTCOMES = 'data/processed/path_a_obligation_outcomes.parquet'
CFR_RESTRICTED_OUTCOMES = 'data/processed/path_a_obligation_outcomes.parquet.cfr_restricted.bak'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outcomes-path',
        default=DEFAULT_OUTCOMES,
        help=(f"Path to outcomes parquet. Default: {DEFAULT_OUTCOMES} (text-only "
              f"classification used in §6.4). Alternative: {CFR_RESTRICTED_OUTCOMES} "
              f"(cfr_section-gated classification — Finding 3 sensitivity check)."),
    )
    args = parser.parse_args()

    # === 1. Load core data ===
    print(f"Loading Stage 4 matches and Path A outcomes from {args.outcomes_path}...")
    df_matches = pd.concat([pd.read_parquet(p) for p in glob.glob('data/processed/stage4/*.parquet')])
    df_outcomes = pd.read_parquet(args.outcomes_path)
    addressed = df_matches[df_matches['addressed'] == True]  # noqa: E712

    # Per-obligation aggregation: addressing-commenter counts + stance counts
    agg = addressed.groupby('obligation_id').agg(
        n_addressing=('comment_id', 'nunique'),
        n_opposing=('stance', lambda s: (s == 'OPPOSING').sum()),
        n_supporting=('stance', lambda s: (s == 'SUPPORTING').sum()),
    ).reset_index()
    out = df_outcomes.merge(agg, on='obligation_id', how='left').fillna(
        {'n_addressing': 0, 'n_opposing': 0, 'n_supporting': 0}
    )

    # === Finding 1: Engagement → revision ===
    print("\n=== Finding 1: Engagement → revision (high-engagement vs baseline) ===")
    out['high_engagement'] = out['n_addressing'] >= 5
    out['was_revised'] = out['outcome_state'].isin(['SURVIVED-edited', 'MODIFIED', 'DROPPED'])
    ct = pd.crosstab(out['high_engagement'], out['was_revised'])
    print(ct)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"Chi-square: chi2={chi2:.2f}, p={p:.2e}, dof={dof}")
    cramers_v = (chi2 / ct.values.sum()) ** 0.5
    print(f"Effect size (Cramér's V): {cramers_v:.3f}")
    print(f"  (paper §6.3 reports: chi^2=24.53, p<0.001, V=0.044, n=12,730)")

    # === Finding 2: Opposing vs Supporting (direction-of-engagement) ===
    print("\n=== Finding 2: Opposing vs Supporting (direction-of-engagement) ===")
    opp = out[out['n_opposing'] > 0]
    sup = out[out['n_supporting'] > 0]
    print(f"Opposing subset: {len(opp)} obs, {opp['was_revised'].mean():.1%} revised")
    print(f"Supporting subset: {len(sup)} obs, {sup['was_revised'].mean():.1%} revised")
    counts = [opp['was_revised'].sum(), sup['was_revised'].sum()]
    nobs = [len(opp), len(sup)]
    z, p = proportions_ztest(counts, nobs)
    print(f"Two-proportion z-test: z={z:.2f}, p={p:.3f}")
    print(f"  (paper §6.3 reports: z=-1.42, p=0.155)")
    print(f"  CAVEAT: the two subsets overlap (an obligation can have both opposing")
    print(f"          and supporting addressers); test treats them as independent.")

    # === Finding 3: Equity — org-majority × outcome ===
    print("\n=== Finding 3: Equity — org vs ind addressing on MODIFIED vs SURVIVED-edited ===")
    comments_dfs = []
    for year_pq in glob.glob('data/processed/comments_augmented/EPA_*.parquet'):
        cdf = pd.read_parquet(year_pq, columns=['document_id', 'title'])
        # SIMPLER CLASSIFIER — see docstring. This is the convention used to
        # produce the §6.4 paper-reported cells. Future-work refinement noted in §8.
        cdf['submitter_kind'] = cdf['title'].fillna('').str.contains(
            'submitted by|on behalf|written by', case=False, regex=True
        ).map({True: 'individual', False: 'organizational'})
        comments_dfs.append(cdf[['document_id', 'submitter_kind']])
    comments = pd.concat(comments_dfs)
    print("\n  Submitter_kind heuristic distribution:")
    print(f"  {comments['submitter_kind'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  Total comments classified: {len(comments):,}")
    print(f"  (PROJECT_FACTS §2 reports: 71.4% individual / 28.6% organizational — should match)")

    addr_with_kind = addressed.merge(comments, left_on='comment_id', right_on='document_id', how='left')
    agg_kind = addr_with_kind.groupby('obligation_id').agg(
        n_addr=('comment_id', 'nunique'),
        n_org=('submitter_kind', lambda s: (s == 'organizational').sum()),
        n_ind=('submitter_kind', lambda s: (s == 'individual').sum()),
    ).reset_index()
    agg_kind['pct_org'] = agg_kind['n_org'] / (agg_kind['n_org'] + agg_kind['n_ind']).replace(0, 1)
    out2 = df_outcomes.merge(agg_kind, on='obligation_id', how='left')
    high = out2[out2['n_addr'] >= 5].copy()
    high['org_majority'] = high['pct_org'] > 0.5
    mod = high[high['outcome_state'] == 'MODIFIED']
    edited = high[high['outcome_state'] == 'SURVIVED-edited']
    print(f"\n  MODIFIED: {len(mod)} obs, {mod['org_majority'].mean():.1%} org-majority")
    print(f"  SURVIVED-edited: {len(edited)} obs, {edited['org_majority'].mean():.1%} org-majority")
    counts = [edited['org_majority'].sum(), mod['org_majority'].sum()]
    nobs = [len(edited), len(mod)]
    z, p = proportions_ztest(counts, nobs)
    print(f"  Two-proportion z-test (edited > mod for org-majority): z={z:.2f}, p={p:.3f}")

    subset = high[high['outcome_state'].isin(['MODIFIED', 'SURVIVED-edited'])]
    ct_eq = pd.crosstab(subset['outcome_state'], subset['org_majority'])
    print("\n  Cross-tab (MODIFIED vs SURVIVED-edited × org-majority):")
    print(ct_eq)
    odds_ratio, fisher_p = stats.fisher_exact(ct_eq.values)
    print(f"  Fisher's exact: odds_ratio={odds_ratio:.2f}, p={fisher_p:.3f}")
    print(f"  (paper §6.4 reports: OR=3.07, 95% CI [1.41, 6.71], p=0.007, n=116)")
    print(f"  (paper §6.4 cells: SE-org=43, SE-not-org=15, MO-org=28, MO-not-org=30)")


if __name__ == "__main__":
    main()
