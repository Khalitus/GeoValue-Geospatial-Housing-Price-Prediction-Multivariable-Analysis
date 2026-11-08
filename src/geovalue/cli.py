"""Command-line interface for GeoValue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geovalue.data import HousingDataError, load_housing_data, summarize_housing


DEFAULT_DATASET = Path("data/raw/housing.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geovalue",
        description="Explore and predict historical California census-area housing values.",
    )
    commands = parser.add_subparsers(dest="command")

    status = commands.add_parser("status", help="Check for the local dataset")
    status.add_argument("--data", type=Path, default=DEFAULT_DATASET)

    inspect = commands.add_parser("inspect", help="Validate and summarize the dataset")
    inspect.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    inspect.add_argument(
        "--output", type=Path, help="Optional location for a JSON summary file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "status":
        exists = args.data.is_file()
        print(f"Dataset {'found' if exists else 'missing'}: {args.data}")
        return 0 if exists else 1

    try:
        frame = load_housing_data(args.data)
        summary = summarize_housing(frame)
        document = json.dumps(summary, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(document + "\n", encoding="utf-8")
        print(document)
        return 0
    except (HousingDataError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
