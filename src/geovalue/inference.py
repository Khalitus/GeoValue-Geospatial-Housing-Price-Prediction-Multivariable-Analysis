"""Validated single-record and CSV inference for the trained GeoValue pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geovalue.data import INPUT_COLUMNS, NUMERIC_INPUT_COLUMNS
from geovalue.modeling import predict_records

OCEAN_CATEGORIES = frozenset({"<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"})
PREDICTION_COLUMN = "predicted_median_house_value"


def validate_prediction_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a safe, correctly typed feature matrix; no target column is necessary."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Prediction input must contain at least one row")
    missing = sorted(set(INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")

    result = frame.loc[:, list(INPUT_COLUMNS)].copy()
    for column in NUMERIC_INPUT_COLUMNS:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric input: {column}") from exc
        series = result[column].to_numpy(dtype=float)
        if np.isinf(series).any() or (column != "total_bedrooms" and np.isnan(series).any()):
            raise ValueError(f"Missing or infinite input: {column}")

    for column in ("housing_median_age", "total_rooms", "total_bedrooms", "population", "median_income"):
        if (result[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot be negative")
    if (result["households"] <= 0).any():
        raise ValueError("households must be greater than zero")
    if not result["latitude"].between(-90, 90).all():
        raise ValueError("latitude must be between -90 and 90")
    if not result["longitude"].between(-180, 180).all():
        raise ValueError("longitude must be between -180 and 180")

    ocean = result["ocean_proximity"].astype("string").str.strip()
    if ocean.isna().any() or not ocean.isin(OCEAN_CATEGORIES).all():
        raise ValueError("ocean_proximity must match a known California housing category")
    result["ocean_proximity"] = ocean.astype(str)
    return result


def predict_housing_frame(model_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the supplied records with an additional prediction column."""
    typed = validate_prediction_inputs(frame)
    result = frame.copy()
    predictions = predict_records(model_path, typed)
    if len(predictions) != len(result):
        raise ValueError("Prediction count mismatch")
    result[PREDICTION_COLUMN] = predictions
    return result


def predict_housing_csv(model_path: str | Path, input_path: str | Path, output_path: str | Path) -> Path:
    """Make batch predictions and write the resulting CSV."""
    try:
        frame = pd.read_csv(input_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read prediction CSV: {input_path}") from exc
    result = predict_housing_frame(model_path, frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
