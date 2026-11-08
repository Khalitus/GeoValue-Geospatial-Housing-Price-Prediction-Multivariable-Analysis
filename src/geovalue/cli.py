"""Command-line interface for GeoValue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geovalue.data import HousingDataError, load_housing_data, summarize_housing

DEFAULT_DATASET = Path("data/raw/housing.csv")
DEFAULT_MODEL = Path("artifacts/housing-model.joblib")
DEFAULT_TRAINING_REPORT = Path("reports/training-report.json")


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
    inspect.add_argument("--output", type=Path, help="Optional location for a JSON summary file")

    train = commands.add_parser(
        "train", help="Train and compare three regressors, then save the best model"
    )
    train.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--report-output", type=Path, default=DEFAULT_TRAINING_REPORT)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--forest-trees", type=int, default=120)
    train.add_argument("--xgboost-trees", type=int, default=220)

    info = commands.add_parser("model-info", help="Show metadata for a saved model")
    info.add_argument("--model", type=Path, default=DEFAULT_MODEL)
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
        if args.command == "inspect":
            frame = load_housing_data(args.data)
            summary = summarize_housing(frame)
            document = json.dumps(summary, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(document + "\n", encoding="utf-8")
            print(document)
            return 0

        if args.command == "train":
            from geovalue.training import TrainingConfig, run_training

            result = run_training(
                args.data,
                args.model_output,
                args.report_output,
                TrainingConfig(
                    random_state=args.seed,
                    forest_trees=args.forest_trees,
                    xgboost_trees=args.xgboost_trees,
                ),
            )
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0

        if args.command == "model-info":
            from geovalue.modeling import load_model

            artifact = load_model(args.model)
            info = {key: value for key, value in artifact.items() if key != "pipeline"}
            print(json.dumps(info, indent=2))
            return 0

    except (HousingDataError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
