"""Leakage-safe housing feature engineering for scikit-learn pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from geovalue.data import INPUT_COLUMNS

ENGINEERED_COLUMNS = (
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Preserve undefined ratios as NaN for downstream training-only imputation."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator.astype(float) / denominator.astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


class HousingFeatureEngineer(TransformerMixin, BaseEstimator):
    """Append useful household ratios without learning any dataset statistics.

    Imputation and scaling intentionally happen later inside the fitted pipeline.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> HousingFeatureEngineer:
        self._validate(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        result = X.loc[:, list(INPUT_COLUMNS)].copy()
        result["rooms_per_household"] = _safe_ratio(
            result["total_rooms"], result["households"]
        )
        result["bedrooms_per_room"] = _safe_ratio(
            result["total_bedrooms"], result["total_rooms"]
        )
        result["population_per_household"] = _safe_ratio(
            result["population"], result["households"]
        )
        return result

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Housing features must be supplied as a pandas DataFrame")
        missing = sorted(set(INPUT_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing housing features: {', '.join(missing)}")
