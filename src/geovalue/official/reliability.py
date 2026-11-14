"""Spatial Evidence Indicator: distance-to-training support + held-out error audit.

The indicator is NOT a probability, statistical confidence interval or guarantee.
Thresholds and empirical errors come from 2024 validation ONLY; a separate
2025 audit is reported after calibration without tuning the thresholds.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from geovalue.official.spatial import project_km

LEVELS = ("dense", "typical", "sparse")
K_NEIGHBORS = 10


def support_distance(reference: pd.DataFrame | np.ndarray, queries: pd.DataFrame | np.ndarray, *, k: int = K_NEIGHBORS) -> np.ndarray:
    """Kilometres to k-th closest reference location; k adapts for tiny fixtures."""
    if isinstance(reference, pd.DataFrame):
        rx, ry = project_km(reference.longitude.to_numpy(float), reference.latitude.to_numpy(float))
        reference = np.column_stack((rx, ry))
    if isinstance(queries, pd.DataFrame):
        qx, qy = project_km(queries.longitude.to_numpy(float), queries.latitude.to_numpy(float))
        queries = np.column_stack((qx, qy))
    reference, queries = np.asarray(reference, float), np.asarray(queries, float)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError("Reference locations require at least two x/y pairs")
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("Query locations must be x/y pairs")
    if not np.isfinite(reference).all() or not np.isfinite(queries).all():
        raise ValueError("Non-finite geographical coordinates")
    return np.asarray(cKDTree(reference).query(queries, k=min(k, len(reference)))[0], dtype=float)[:, -1] if min(k, len(reference)) > 1 else np.asarray(cKDTree(reference).query(queries, k=1)[0], dtype=float)


def support_buckets(distances: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    d = np.asarray(distances, float)
    if not np.isfinite(d).all() or not np.isfinite(limits).all() or limits[0] > limits[1]:
        raise ValueError("Invalid support distances or thresholds")
    return np.where(d <= limits[0], 0, np.where(d <= limits[1], 1, 2)).astype(int)


def error_by_support(distances: np.ndarray, actual: np.ndarray, predicted: np.ndarray, limits: tuple[float,float]) -> dict:
    from geovalue.official.modeling import metrics
    distances, actual, predicted = map(lambda x: np.asarray(x, float), (distances, actual, predicted))
    if not (distances.shape == actual.shape == predicted.shape and distances.ndim == 1):
        raise ValueError("Distance, target and prediction arrays must match")
    bins = support_buckets(distances, limits)
    result = {}
    for i, name in enumerate(LEVELS):
        mask = bins == i
        if not mask.any():
            result[name] = {"n": 0, "mae_usd": None, "rmse_usd": None, "p80_absolute_error_usd": None, "median_km": None}
            continue
        errors = np.abs(actual[mask] - predicted[mask])
        result[name] = {
            "n": int(mask.sum()),
            "mae_usd": float(np.mean(errors)),
            "rmse_usd": float(np.sqrt(np.mean(errors ** 2))),
            "p80_absolute_error_usd": float(np.quantile(errors, 0.8)),
            "median_km": float(np.median(distances[mask])),
        }
    return result


def calibrate_support(train: pd.DataFrame, validation: pd.DataFrame, validation_prediction: np.ndarray) -> tuple[dict, tuple[float,float]]:
    """Calibrate using pre-2024 fit locations + out-of-time 2024 errors."""
    distances = support_distance(train, validation)
    limits = tuple(float(v) for v in np.quantile(distances, [1/3, 2/3]))
    actual = validation.sale_price.to_numpy(float)
    predicted = np.asarray(validation_prediction, float)
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Invalid validation predictions for evidence calibration")
    absolute_error = np.abs(actual - predicted)
    rho = float(spearmanr(distances, absolute_error).statistic) if np.ptp(distances) > 0 and np.ptp(absolute_error) > 0 else None
    result = {
        "method": "10th-nearest-training-location distance; 2024 out-of-time validation error calibration",
        "reference_crs": "EPSG:26917",
        "distance_units": "km",
        "distance_quantile_boundaries_km": [*limits],
        "support_levels": list(LEVELS),
        "validation_rows": int(len(validation)),
        "validation_error_by_support": error_by_support(distances, actual, predicted, limits),
        "spearman_distance_vs_absolute_error": rho,
        "reliability_caveat": "Density categories describe spatial support, not calibrated prediction confidence; error figures are empirical historical validation statistics, not an individual prediction interval.",
    }
    return result, limits


def create_support_artifacts(data: pd.DataFrame, pipeline: object, training_report: dict, *, output_npz: Path, report_path: Path, sample_cap: int | None = None) -> dict:
    """Recreate selected pre-2024 model for leakage-safe validation calibration.

    The caller supplies the already-trained final refitted model for 2025 audit;
    it is never used to fit the 2024 calibration thresholds.
    """
    from geovalue.official.training import Config, split_sales
    from geovalue.official.modeling import build_model
    config = training_report.get("configuration", {})
    cfg = Config(seed=int(config.get("seed",42)), forest_trees=int(config.get("forest_trees",120)), xgboost_trees=int(config.get("xgboost_trees",220)), max_training_rows=config.get("max_training_rows"))
    train, validation, test, partition = split_sales(data)
    if cfg.max_training_rows is not None and len(train) > cfg.max_training_rows:
        train = train.sample(cfg.max_training_rows, random_state=cfg.seed).sort_values("sale_year")
    from geovalue.official.data import INPUT_COLUMNS
    chosen = training_report["model_selected"]
    old = build_model(chosen, seed=cfg.seed, forest_trees=cfg.forest_trees, xgboost_trees=cfg.xgboost_trees)
    old.fit(train.loc[:, list(INPUT_COLUMNS)], train.sale_price)
    validation_prediction = old.predict(validation.loc[:, list(INPUT_COLUMNS)])
    report, bounds = calibrate_support(train, validation, validation_prediction)
    # Inference support is computed against the final model's dev set, not 2025.
    development = pd.concat([train,validation],ignore_index=True)
    x, y = project_km(development.longitude.to_numpy(float),development.latitude.to_numpy(float))
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, x_km=x, y_km=y, boundaries=np.asarray(bounds, float))
    test_distances = support_distance(np.column_stack((x,y)), test)
    test_predictions = np.asarray(pipeline.predict(test.loc[:, list(INPUT_COLUMNS)]), float)
    report["held_out_2025_support_audit"] = error_by_support(test_distances, test.sale_price.to_numpy(float), test_predictions, bounds)
    report["held_out_2025_rows"] = len(test)
    report["reference_rows_for_inference"] = len(development)
    report["note"] = "2024 bins fitted on pre-2024 support, 2025 audit uses 2013-2024 support. Proportions may shift; 2025 data never sets thresholds."
    report["sample_cap"] = cfg.max_training_rows
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


@lru_cache(maxsize=4)
def _cached_reference(path: str):
    """Build a reusable nearest-neighbor index for repeated local GUI queries."""
    with np.load(path, allow_pickle=False) as saved:
        points = np.column_stack((saved["x_km"], saved["y_km"]))
        boundaries = tuple(float(v) for v in saved["boundaries"])
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Reference site artifact has invalid coordinates")
    return cKDTree(points), len(points), boundaries


def reliability_for_location(longitude: float, latitude: float, reference_path: Path, report: dict) -> dict:
    tree, size, boundaries = _cached_reference(str(reference_path))
    if not np.isfinite([longitude, latitude]).all():
        raise ValueError("Coordinates must be finite")
    x, y = project_km(np.asarray([longitude]),np.asarray([latitude]))
    distances = tree.query([float(x[0]),float(y[0])],k=min(K_NEIGHBORS,size))[0]
    kth_distance = float(np.asarray(distances).reshape(-1)[-1])
    bucket = int(support_buckets(np.asarray([kth_distance]),boundaries)[0])
    level = LEVELS[bucket]
    calibration = report["validation_error_by_support"][level]
    return {
        "level": level,
        "tenth_neighbor_km": kth_distance,
        "reference_sample_count": int(size),
        "validation_reference_count": calibration["n"],
        "validation_group_mae_usd": calibration["mae_usd"],
        "validation_group_p80_absolute_error_usd": calibration["p80_absolute_error_usd"],
        "is_confidence_interval": False,
        "interpretation": "Spatial support describes density of recorded training locations. Group errors summarize 2024 validation observations; they are not uncertainty guarantees for this property.",
    }
