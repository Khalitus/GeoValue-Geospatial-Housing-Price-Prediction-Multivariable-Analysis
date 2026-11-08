"""Restore the public 1990 California housing CSV if your local dataset is missing.

Only downloads when data/raw/housing.csv does not exist; never overwrites the
user's original dataset. Source is Aurélien Géron's public housing dataset.
"""
from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://raw.githubusercontent.com/ageron/data/main/housing/housing.csv"


def restore(destination: Path = ROOT / "data/raw/housing.csv") -> str:
    if destination.is_file():
        return f"Existing dataset preserved: {destination}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(SOURCE_URL, temp)
        with temp.open("r", newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            expected = {"longitude", "latitude", "median_house_value", "ocean_proximity", "total_bedrooms"}
            if not expected.issubset(rows.fieldnames or []):
                raise ValueError("Unexpected dataset columns")
            count = sum(1 for _ in rows)
        if count != 20640:
            raise ValueError(f"Unexpected dataset size: {count}")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return f"Downloaded and validated {count} public housing rows to {destination}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/housing.csv")
    args = parser.parse_args()
    print(restore(args.output))
