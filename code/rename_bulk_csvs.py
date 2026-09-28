"""
Identify and rename regulations.gov bulk download CSVs based on their content.

Reads the Agency ID and Posted Date columns from each CSV, determines what it
contains, renames it to {AGENCY}_{YEAR}_full.csv, and moves it to the right
agency subfolder.

Usage:
    python rename_bulk_csvs.py /path/to/folder/with/raw/csvs

If no path is given, defaults to ~/Downloads/regulations_bulk/
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
from collections import Counter

# ----- CONFIG -----
PROJECT_DIR = Path.home() / "regulatory-comments-project" / "data" / "raw" / "regulations_gov"
EXPECTED_AGENCIES = {"EPA", "FCC", "SEC"}


def inspect_csv(path: Path) -> tuple[str | None, int | None, int]:
    """
    Return (agency_id, year, n_rows) from a regulations.gov bulk CSV.
    Reads up to ~5000 rows to determine the dominant agency and year range.
    Returns (None, None, 0) if file is empty or unreadable.
    """
    agency_counts: Counter[str] = Counter()
    years: list[int] = []
    n_rows = 0

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return (None, None, 0)
            # Verify the columns we need exist
            cols = set(reader.fieldnames)
            agency_col = "Agency ID" if "Agency ID" in cols else None
            date_col = "Posted Date" if "Posted Date" in cols else None
            if agency_col is None or date_col is None:
                print(f"  [WARN] {path.name}: missing required columns "
                      f"(have: {sorted(cols)[:5]}...)")
                return (None, None, 0)

            for row in reader:
                n_rows += 1
                agency = (row.get(agency_col) or "").strip()
                if agency:
                    agency_counts[agency] += 1
                date_str = (row.get(date_col) or "").strip()
                if len(date_str) >= 4 and date_str[:4].isdigit():
                    years.append(int(date_str[:4]))
                # Sample the first 5000 rows for speed
                if n_rows >= 5000 and len(years) >= 100:
                    # Continue counting rows but stop sampling
                    break
            # Finish counting total rows
            for _ in reader:
                n_rows += 1
    except Exception as e:
        print(f"  [ERROR] {path.name}: {e}")
        return (None, None, 0)

    if not agency_counts or not years:
        return (None, None, n_rows)

    dominant_agency = agency_counts.most_common(1)[0][0]
    # Year is the mode of the sampled years
    year_counts = Counter(years)
    dominant_year = year_counts.most_common(1)[0][0]
    return (dominant_agency, dominant_year, n_rows)


def main() -> int:
    if len(sys.argv) > 1:
        source_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        source_dir = Path.home() / "Downloads" / "regulations_bulk"

    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}")
        print(f"Usage: python {Path(__file__).name} /path/to/csvs")
        return 1

    csvs = sorted(source_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {source_dir}")
        return 1

    print(f"Found {len(csvs)} CSV(s) in {source_dir}\n")

    moves: list[tuple[Path, Path]] = []
    skipped: list[Path] = []

    for csv_path in csvs:
        print(f"Inspecting {csv_path.name}...")
        agency, year, n_rows = inspect_csv(csv_path)

        if agency is None or year is None:
            print(f"  -> Could not determine agency/year. Skipping.\n")
            skipped.append(csv_path)
            continue

        if agency not in EXPECTED_AGENCIES:
            print(f"  -> Unexpected agency '{agency}'. Skipping.\n")
            skipped.append(csv_path)
            continue

        target = PROJECT_DIR / agency / f"{agency}_{year}_full.csv"
        print(f"  -> {agency} {year}, {n_rows:,} rows")
        print(f"     -> {target}")

        if target.exists():
            print(f"     [WARN] Target already exists. Will not overwrite.\n")
            skipped.append(csv_path)
            continue

        moves.append((csv_path, target))
        print()

    if not moves:
        print("No files to move.")
        return 0

    # Confirm with user
    print(f"\n{'=' * 60}")
    print(f"READY TO MOVE {len(moves)} files:")
    for src, dst in moves:
        print(f"  {src.name} -> {dst.relative_to(Path.home())}")
    print(f"\nSkipped: {len(skipped)} files")
    print(f"{'=' * 60}\n")

    confirm = input("Proceed? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return 0

    # Make sure target directories exist
    for agency in EXPECTED_AGENCIES:
        (PROJECT_DIR / agency).mkdir(parents=True, exist_ok=True)

    # Execute moves
    moved = 0
    for src, dst in moves:
        try:
            src.rename(dst)
            moved += 1
        except Exception as e:
            print(f"  Failed to move {src.name}: {e}")

    print(f"\nMoved {moved} of {len(moves)} files.")
    if skipped:
        print(f"\n{len(skipped)} files were not moved (still in {source_dir}):")
        for path in skipped:
            print(f"  {path.name}")

    # Summary by agency
    print(f"\n{'=' * 60}")
    print("Final inventory by agency:")
    for agency in sorted(EXPECTED_AGENCIES):
        agency_dir = PROJECT_DIR / agency
        files = sorted(agency_dir.glob("*.csv"))
        years = [f.name.split("_")[1] for f in files if "_" in f.name]
        print(f"  {agency}: {len(files)} files (years: {', '.join(sorted(years))})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
