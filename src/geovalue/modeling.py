"""Regression pipelines, evaluation, and trusted local model persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from geovalue.data import CATEGORICAL_INPUT_COLUMNS, INPUT_COLUMNS, NUMERIC_INPUT_COLUMNS
from geovalue.features import ENGINEERED_COLUMNS, HousingFeatureEngineer

MODEL_NAMES = ("linear", "random_forest", "xgboost")
MODEL_FORMAT_VERSION = 1
GEO_FEATURES = ("longitude", "latitude", "ocean_proximity")


def build_model(
    name: str,
    *,
    random_state: int = 42,
    include_geography: bool = True,
    forest_trees: int = 120,
    xgboost_trees: int = 220,
) -> Pipeline:
    """Build a fully fitted-at-train-time, serialization-ready regression pipeline."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {name}. Expected one of {MODEL_NAMES}")
    if forest_trees < 1 or xgboost_trees < 1:
        raise ValueError("Tree counts must be positive")

    numeric = list(NUMERIC_INPUT_COLUMNS + ENGINEERED_COLUMNS)
    categorical = list(CATEGORICAL_INPUT_COLUMNS)
    if not include_geography:
        numeric = [column for column in numeric if column not in GEO_FEATURES]
        categorical = [column for column in categorical if column not in GEO_FEATURES]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", numeric_pipeline, numeric),
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    if name == "linear":
        regressor = LinearRegression()
    elif name == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=forest_trees,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        regressor = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=xgboost_trees,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=4,
            random_state=random_state,
        )

    return Pipeline(
        steps=[
            ("features", HousingFeatureEngineer()),
            ("preprocess", preprocess),
            ("regressor", regressor),
        ]
    )


def regression_metrics(actual: np.ndarray | pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics in original historical USD units."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted values must be equal-length 1D arrays")
    if not len(y_true) or not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Metrics require nonempty, finite values")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_model(model: Pipeline, path: str | Path, *, model_name: str) -> Path:
    """Persist a fitted sklearn pipeline for inference. Only load trusted artifacts."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name: {model_name}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "model_name": model_name,
            "input_columns": list(INPUT_COLUMNS),
            "pipeline": model,
        },
        output,
        compress=3,
    )
    return output


def load_model(path: str | Path) -> dict:
    """Load a model saved by GeoValue (trusted local files only)."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("format_version") != MODEL_FORMAT_VERSION
        or artifact.get("model_name") not in MODEL_NAMES
        or artifact.get("input_columns") != list(INPUT_COLUMNS)
        or not isinstance(artifact.get("pipeline"), Pipeline)
    ):
        raise ValueError("Unsupported or malformed GeoValue model artifact")
    return artifact


def predict_records(model_path: str | Path, features: pd.DataFrame) -> np.ndarray:
    """Predict historical area-level median values from feature-only rows."""
    model = load_model(model_path)["pipeline"]
    result = np.asarray(model.predict(features), dtype=float)
    if result.ndim != 1 or len(result) != len(features) or not np.isfinite(result).all():
        raise ValueError("Model returned invalid predictions")
    return result
