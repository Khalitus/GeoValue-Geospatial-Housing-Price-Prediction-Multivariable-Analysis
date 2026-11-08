"""Loading, validation and descriptive analysis for Kaggle housing.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "median_house_value"
NUMERIC_INPUT_COLUMNS = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
)
CATEGORICAL_INPUT_COLUMNS = ("ocean_proximity",)
INPUT_COLUMNS = NUMERIC_INPUT_COLUMNS + CATEGORICAL_INPUT_COLUMNS
REQUIRED_COLUMNS = INPUT_COLUMNS + (TARGET_COLUMN,)
NUMERIC_COLUMNS = NUMERIC_INPUT_COLUMNS + (TARGET_COLUMN,)
NULLABLE_COLUMNS = frozenset({"total_bedrooms"})


class HousingDataError(ValueError):
    """The dataset is missing or violates the expected housing schema."""


def validate_housing_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, validated copy of a California housing dataset.

    The original Kaggle dataset permits missing total_bedrooms values.
    Imputation belongs inside the ML pipeline and must not happen here.
    """
    if frame.empty:
        raise HousingDataError("Dataset contains no rows")

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise HousingDataError(f"Missing required columns: {', '.join(missing)}")

    result = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in NUMERIC_COLUMNS:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise HousingDataError(f"Non-numeric values in {column}") from exc

        series = result[column]
        if not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
            raise HousingDataError(f"Infinite values in {column}")
        if column not in NULLABLE_COLUMNS and series.isna().any():
            raise HousingDataError(f"Missing values in {column}")

    result["ocean_proximity"] = (
        result["ocean_proximity"].astype("string").str.strip()
    )
    if (
        result["ocean_proximity"].isna().any()
        or result["ocean_proximity"].eq("").any()
    ):
        raise HousingDataError("Missing values in ocean_proximity")

    for column, minimum in (
        ("housing_median_age", 0),
        ("total_rooms", 0),
        ("total_bedrooms", 0),
        ("population", 0),
        ("median_income", 0),
    ):
        if result[column].lt(minimum).any():
            raise HousingDataError(f"Negative values in {column}")

    if result["households"].le(0).any():
        raise HousingDataError("households must be greater than zero")
    if result[TARGET_COLUMN].le(0).any():
        raise HousingDataError("median_house_value must be greater than zero")
    if not result["latitude"].between(-90, 90).all():
        raise HousingDataError("latitude must be between -90 and 90")
    if not result["longitude"].between(-180, 180).all():
        raise HousingDataError("longitude must be between -180 and 180")

    return result


def load_housing_data(path: str | Path) -> pd.DataFrame:
    """Read and validate a local Kaggle California housing CSV."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise HousingDataError(f"Dataset not found: {dataset_path}")

    try:
        frame = pd.read_csv(dataset_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HousingDataError(f"Cannot read dataset: {dataset_path}") from exc

    return validate_housing_data(frame)


def summarize_housing(frame: pd.DataFrame) -> dict:
    """Produce a JSON-serializable snapshot, not fitted model parameters."""
    validated = validate_housing_data(frame)
    values = validated[TARGET_COLUMN]
    correlations = validated.loc[:, list(NUMERIC_COLUMNS)].corr(numeric_only=True)[
        TARGET_COLUMN
    ].drop(TARGET_COLUMN)

    def safe_float(value: float) -> float | None:
        return float(value) if pd.notna(value) and np.isfinite(value) else None

    return {
        "records": int(len(validated)),
        "columns": list(validated.columns),
        "missing_values": {
            col: int(count) for col, count in validated.isna().sum().items()
        },
        "ocean_proximity_counts": {
            str(category): int(count)
            for category, count in validated["ocean_proximity"].value_counts().items()
        },
        "median_house_value": {
            "mean": safe_float(values.mean()),
            "median": safe_float(values.median()),
            "minimum": safe_float(values.min()),
            "maximum": safe_float(values.max()),
        },
        "correlation_to_median_house_value": {
            col: safe_float(value) for col, value in correlations.items()
        },
    }
