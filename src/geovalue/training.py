"""Reproducible train/validation/test workflow and geographic feature experiment."""

from __future__ import annotations

import json
import math
import platform
from importlib.metadata import version
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from geovalue.data import INPUT_COLUMNS, TARGET_COLUMN, load_housing_data, validate_housing_data
from geovalue.modeling import MODEL_NAMES, build_model, regression_metrics, save_model


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    test_fraction: float = 0.20
    validation_fraction: float = 0.20
    forest_trees: int = 120
    xgboost_trees: int = 220

    def __post_init__(self) -> None:
        if not (0 < self.test_fraction < 1 and 0 < self.validation_fraction < 1):
            raise ValueError("Split fractions must be between 0 and 1")
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("Train split must contain some observations")
        if self.forest_trees < 1 or self.xgboost_trees < 1:
            raise ValueError("Tree counts must be positive")


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    stratified: bool


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    model_name: str
    report: dict


def _income_strata(frame: pd.DataFrame, fraction: float) -> pd.Series | None:
    """Use coarse income bins only when each split can hold every populated bin."""
    income = frame["median_income"]
    strata = pd.cut(
        income,
        bins=[-np.inf, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=False,
        include_lowest=True,
    )
    counts = strata.value_counts(dropna=False)
    held_out = math.ceil(len(frame) * fraction)
    if (
        strata.isna().any()
        or (counts < 2).any()
        or held_out < len(counts)
        or len(frame) - held_out < len(counts)
    ):
        return None
    return strata


def split_housing(frame: pd.DataFrame, config: TrainingConfig) -> DataSplits:
    """Make three reproducible, non-overlapping splits before any fitted preprocessing."""
    validated = validate_housing_data(frame).reset_index(drop=True)
    if len(validated) < 15:
        raise ValueError("At least 15 housing records are required for training")

    dev, test = train_test_split(
        validated,
        test_size=config.test_fraction,
        random_state=config.random_state,
        stratify=_income_strata(validated, config.test_fraction),
    )
    internal_validation_fraction = config.validation_fraction / (1 - config.test_fraction)
    dev_strata = _income_strata(dev, internal_validation_fraction)
    train, validation = train_test_split(
        dev,
        test_size=internal_validation_fraction,
        random_state=config.random_state,
        stratify=dev_strata,
    )
    first_stratified = _income_strata(validated, config.test_fraction) is not None
    return DataSplits(
        train=train.copy(),
        validation=validation.copy(),
        test=test.copy(),
        stratified=first_stratified and dev_strata is not None,
    )


def _features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.loc[:, list(INPUT_COLUMNS)], frame[TARGET_COLUMN]


def _build(name: str, config: TrainingConfig, *, include_geography: bool = True) -> Pipeline:
    return build_model(
        name,
        random_state=config.random_state,
        include_geography=include_geography,
        forest_trees=config.forest_trees,
        xgboost_trees=config.xgboost_trees,
    )


def train_and_evaluate(frame: pd.DataFrame, config: TrainingConfig | None = None) -> TrainingResult:
    """Select on validation RMSE; refit on development data; test exactly once.

    The test set is never used to choose a model or compare geographic features.
    """
    config = config or TrainingConfig()
    splits = split_housing(frame, config)
    x_train, y_train = _features_and_target(splits.train)
    x_val, y_val = _features_and_target(splits.validation)
    validation_results: dict[str, dict] = {}

    for name in MODEL_NAMES:
        model = _build(name, config)
        model.fit(x_train, y_train)
        validation_results[name] = {
            "train": regression_metrics(y_train, model.predict(x_train)),
            "validation": regression_metrics(y_val, model.predict(x_val)),
        }

    chosen_name = min(
        MODEL_NAMES, key=lambda name: validation_results[name]["validation"]["rmse"]
    )

    # Same selected estimator and train/validation partition, minus location fields.
    no_geo_model = _build(chosen_name, config, include_geography=False)
    no_geo_model.fit(x_train, y_train)
    without_geo = regression_metrics(y_val, no_geo_model.predict(x_val))
    with_geo = validation_results[chosen_name]["validation"]

    # Only now refit selected architecture on train + validation, then score the test once.
    development = pd.concat([splits.train, splits.validation], axis=0)
    x_dev, y_dev = _features_and_target(development)
    x_test, y_test = _features_and_target(splits.test)
    final_model = _build(chosen_name, config)
    final_model.fit(x_dev, y_dev)
    test_metrics = regression_metrics(y_test, final_model.predict(x_test))

    report = {
        "target": TARGET_COLUMN,
        "target_units": "historical_1990_USD",
        "random_state": config.random_state,
        "software_versions": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "joblib": version("joblib"),
        },
        "hyperparameters": {
            "random_forest_trees": config.forest_trees,
            "xgboost_trees": config.xgboost_trees,
        },
        "split": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
            "income_stratified": splits.stratified,
        },
        "input_columns": list(INPUT_COLUMNS),
        "validation_results": validation_results,
        "selected_model": chosen_name,
        "selection_metric": "validation_rmse",
        "geographic_feature_experiment": {
            "model": chosen_name,
            "comparison_split": "validation",
            "with_geography": with_geo,
            "without_geography": without_geo,
            "rmse_improvement_from_geography": float(without_geo["rmse"] - with_geo["rmse"]),
            "excluded_features": ["longitude", "latitude", "ocean_proximity"],
        },
        "held_out_test": test_metrics,
        "final_refit_rows": len(development),
        "notes": [
            "The target is 1990 census-area median house value, not current individual home price.",
            "The dataset has an upper-coded target value of 500001 USD.",
            "All fitted preprocessing is learned from training data only during selection.",
            "The selected model is refitted on train+validation before one test evaluation.",
            "The geographic comparison is descriptive, not causal or a spatially held-out test.",
        ],
    }
    return TrainingResult(final_model, chosen_name, report)


def run_training(
    data_path: str | Path,
    model_output: str | Path,
    report_output: str | Path,
    config: TrainingConfig | None = None,
) -> dict:
    """Load the housing CSV, train models and write both reproducible outputs."""
    data = load_housing_data(data_path)
    result = train_and_evaluate(data, config)
    model_path = save_model(result.pipeline, model_output, model_name=result.model_name)
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "selected_model": result.model_name,
        "held_out_test": result.report["held_out_test"],
        "model_path": str(model_path),
        "report_path": str(report_path),
    }
