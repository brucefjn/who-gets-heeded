"""
34_descriptive_counts.py - Recompute the descriptive counts the paper quotes
in Section 5 (matching) and Section 8 (limitations) from the current Stage 4
outputs, i.e. after the 5,000-character re-run of truncated pairs.

Prints:
  - total comment-obligation pairs, addressing pairs (addressed == True) and
    their share, and pairs with a non-NONE stance;
  - distinct addressing comments;
  - obligations with >= 1 addressing commenter, split into proposed-side and
    final-only NEW, and their share of the 12,243 proposed-side obligations;
  - high-engagement obligations (>= 5 addressing commenters);
  - per-docket counts (proposed-side obligations, addressed, high-engagement).

Usage (repo root):
    python3 code/34_descriptive_counts.py
"""
from __future__ import annotations

import glob
import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "c22", HERE / "22_clustered_logistic_f1_f3.py")
    c22 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c22)
    pd = c22.pd

    files = sorted(glob.glob(c22.STAGE4_GLOB))
    m = pd.concat([pd.read_parquet(f, columns=["comment_id", "obligation_id",
                                                "addressed", "stance"])
                   for f in files], ignore_index=True)
    outcomes = pd.read_parquet(c22.DEFAULT_OUTCOMES)
    state = dict(zip(outcomes["obligation_id"], outcomes["outcome_state"]))

    total = len(m)
    uniq = m.drop_duplicates(["comment_id", "obligation_id"])
    addr = m[m["addressed"] == True]  # noqa: E712
    non_none = m[m["stance"].fillna("NONE") != "NONE"]
    print(f"Stage 4 files: {len(files)}")
    print(f"Total pairs: {total:,} (distinct comment-obligation pairs: {len(uniq):,})")
    print(f"Addressing pairs (addressed == True): {len(addr):,} "
          f"({100 * len(addr) / total:.2f}%); distinct: "
          f"{len(addr.drop_duplicates(['comment_id', 'obligation_id'])):,}")
    print(f"Pairs with non-NONE stance: {len(non_none):,} ({100 * len(non_none) / total:.2f}%)")
    print(f"Distinct addressing comments: {addr['comment_id'].nunique():,}")

    per = addr.groupby("obligation_id")["comment_id"].nunique()
    kinds = per.index.map(lambda o: state.get(o, "not in outcomes"))
    prop = per[[k not in ("NEW", "not in outcomes") for k in kinds]]
    n_prop_side = int((outcomes["outcome_state"] != "NEW").sum())
    print(f"Obligations with >= 1 addressing commenter: {len(per):,} "
          f"(proposed-side {len(prop):,}; NEW {int((kinds == 'NEW').sum())}; "
          f"not in outcomes {int((kinds == 'not in outcomes').sum())})")
    print(f"  share of {n_prop_side:,} proposed-side obligations: "
          f"{100 * len(prop) / n_prop_side:.2f}%")
    print(f"High-engagement proposed-side obligations (>= 5): {int((prop >= 5).sum())}")

    prop_df = prop.rename("n").reset_index()
    prop_df["docket"] = prop_df["obligation_id"].map(c22.docket_of_obligation)
    outc = outcomes[outcomes["outcome_state"] != "NEW"].copy()
    outc["docket"] = outc["obligation_id"].map(c22.docket_of_obligation)
    table = outc.groupby("docket").size().rename("proposed_side").to_frame()
    table["addressed"] = prop_df.groupby("docket").size()
    table["high"] = prop_df[prop_df["n"] >= 5].groupby("docket").size()
    table = table.fillna(0).astype(int).sort_values("addressed", ascending=False)
    print("\nPer docket (sorted by addressed obligations):")
    print(table.to_string())


if __name__ == "__main__":
    main()
