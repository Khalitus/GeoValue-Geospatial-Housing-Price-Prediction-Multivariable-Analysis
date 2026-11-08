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
    predict = commands.add_parser(
        "predict", help="Predict a single historical census area or a CSV batch"
    )
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--input", type=Path, help="CSV of feature-only records for batch prediction")
    predict.add_argument("--output", type=Path, help="Optional output file (CSV for batch; JSON for single)")
    predict.add_argument("--longitude", type=float)
    predict.add_argument("--latitude", type=float)
    predict.add_argument("--housing-median-age", type=float)
    predict.add_argument("--total-rooms", type=float)
    predict.add_argument("--total-bedrooms", type=float, help="Optional; omitted value will be imputed")
    predict.add_argument("--population", type=float)
    predict.add_argument("--households", type=float)
    predict.add_argument("--median-income", type=float)
    predict.add_argument("--ocean-proximity")

    analyze = commands.add_parser(
        "analyze", help="Generate geographical maps, 3D contours and calculus outputs"
    )
    analyze.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    analyze.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    analyze.add_argument("--output-dir", type=Path, default=Path("reports/geospatial"))
    analyze.add_argument("--grid-size", type=int, default=90)
    analyze.add_argument("--sigma", type=float, default=1.6, help="Gaussian smoothing in grid cells")
    analyze.add_argument("--support-km", type=float, default=45.0)
    analyze.add_argument("--map-points", type=int, default=1250)

    gradient = commands.add_parser(
        "gradient", help="Query the local geographical gradient of a saved scenario surface"
    )
    gradient.add_argument("--surface", type=Path, default=Path("reports/geospatial/price-surface.npz"))
    gradient.add_argument("--longitude", type=float, required=True)
    gradient.add_argument("--latitude", type=float, required=True)
    gradient.add_argument("--bearing", type=float, default=0, help="Direction clockwise from north")
    gradient.add_argument("--max-distance-km", type=float, default=30)
    gradient.add_argument("--output", type=Path, help="Optional JSON report path")
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

        if args.command == "predict":
            import pandas as pd
            from geovalue.inference import predict_housing_csv, predict_housing_frame

            fields = {
                "longitude": args.longitude,
                "latitude": args.latitude,
                "housing_median_age": args.housing_median_age,
                "total_rooms": args.total_rooms,
                "total_bedrooms": args.total_bedrooms,
                "population": args.population,
                "households": args.households,
                "median_income": args.median_income,
                "ocean_proximity": args.ocean_proximity,
            }
            if args.input is not None:
                if any(value is not None for value in fields.values()):
                    raise ValueError("Use either --input CSV or single-record field flags, not both")
                destination = args.output or Path("reports/predictions.csv")
                predict_housing_csv(args.model, args.input, destination)
                print(json.dumps({"predictions_csv": str(destination)}, indent=2))
                return 0
            required = [key for key, value in fields.items() if value is None and key != "total_bedrooms"]
            if required:
                raise ValueError("Missing single-record fields: " + ", ".join(required))
            features = pd.DataFrame([fields])
            result = predict_housing_frame(args.model, features)
            payload = {
                "predicted_median_house_value": float(result["predicted_median_house_value"].iloc[0]),
                "units": "historical_1990_USD",
                "note": "Census-area median value estimate, not a current house-sale price",
            }
            document = json.dumps(payload, indent=2, allow_nan=False)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(document + "\n", encoding="utf-8")
            print(document)
            return 0

        if args.command == "analyze":
            from geovalue.analysis import run_geospatial_analysis

            report = run_geospatial_analysis(
                args.data, args.model, args.output_dir,
                grid_size=args.grid_size,
                smooth_sigma=args.sigma,
                max_support_distance_km=args.support_km,
                max_map_points=args.map_points,
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0

        if args.command == "gradient":
            from geovalue.calculus import gradient_at_location, load_price_surface

            result = gradient_at_location(
                load_price_surface(args.surface),
                longitude=args.longitude, latitude=args.latitude,
                bearing_degrees=args.bearing, max_distance_km=args.max_distance_km,
            )
            document = json.dumps(result, indent=2, allow_nan=False)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(document + "\n", encoding="utf-8")
            print(document)
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
