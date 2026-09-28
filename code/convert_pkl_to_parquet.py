"""
convert_pkl_to_parquet.py
One-off helper. Walks data/processed/{documents,comments}/, finds every
*.pkl file, writes a sibling *.parquet, then optionally deletes the pickle.

Run after pyarrow has been installed on Bruce's MacBook so the earlier
sandbox-fallback pickle outputs become real parquet.

Usage:
    python code/convert_pkl_to_parquet.py            # convert + keep pickles
    python code/convert_pkl_to_parquet.py --delete   # convert + delete pickles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--delete", action="store_true",
                   help="Delete the .pkl after a successful parquet write")
    args = p.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("ERROR: pyarrow not installed. Run `pip install pyarrow` first.", file=sys.stderr)
        return 1

    pkls = sorted(PROCESSED_DIR.rglob("*.pkl"))
    if not pkls:
        print(f"No .pkl files found under {PROCESSED_DIR}")
        return 0

    print(f"Converting {len(pkls)} pickle file(s) to parquet:")
    for pkl in pkls:
        parquet = pkl.with_suffix(".parquet")
        df = pd.read_pickle(pkl)
        df.to_parquet(parquet, index=False)
        size_pkl = pkl.stat().st_size / 1024 / 1024
        size_pq = parquet.stat().st_size / 1024 / 1024
        print(f"  {pkl.name}  ({size_pkl:.1f} MB) -> {parquet.name}  ({size_pq:.1f} MB) "
              f"[rows={len(df):,}, cols={df.shape[1]}]")
        if args.delete:
            pkl.unlink()
            print(f"    deleted {pkl.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
