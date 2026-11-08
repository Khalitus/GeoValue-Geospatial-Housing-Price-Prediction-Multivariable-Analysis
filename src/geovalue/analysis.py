"""Run all geographical and multivariable-calculus analyses for a trained model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geovalue.calculus import (
    build_price_surface,
    integrate_price_surface,
    save_price_surface,
)
from geovalue.data import INPUT_COLUMNS, load_housing_data
from geovalue.geospatial import make_housing_map
from geovalue.modeling import load_model
from geovalue.visualization import plot_gradient_field, plot_price_contours, plot_price_surface


def run_geospatial_analysis(
    data_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    grid_size: int = 90,
    smooth_sigma: float = 1.6,
    max_support_distance_km: float = 45.0,
    max_map_points: int = 1250,
) -> dict:
    """Create HTML visualizations, numeric surface and a JSON analysis report."""
    frame = load_housing_data(data_path)
    artifact = load_model(model_path)  # Only load locally trusted joblib files.
    model = artifact["pipeline"]
    outputs = Path(output_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    input_features = frame.loc[:, list(INPUT_COLUMNS)]
    all_predictions = np.asarray(model.predict(input_features), dtype=float)
    if not np.isfinite(all_predictions).all():
        raise ValueError("Non-finite model predictions for geographical map")
    map_path = make_housing_map(
        frame, outputs / "housing-map.html", predictions=all_predictions,
        max_points=max_map_points,
    )
    surface, reference = build_price_surface(
        frame, model, grid_size=grid_size,
        smooth_sigma=smooth_sigma, max_support_distance_km=max_support_distance_km,
    )
    integral = integrate_price_surface(surface)
    surface_npz = save_price_surface(surface, outputs / "price-surface.npz")
    surface_html = plot_price_surface(surface, outputs / "price-surface-3d.html")
    contours_html = plot_price_contours(surface, outputs / "price-contours.html")
    gradients_html = plot_gradient_field(surface, outputs / "price-gradients.html")

    magnitude = np.hypot(surface.d_dx, surface.d_dy)
    finite = magnitude[np.isfinite(magnitude)]
    report = {
        "model_name": artifact["model_name"],
        "crs": "EPSG:3310",
        "axis_units": "kilometres",
        "price_units": "historical_1990_USD",
        "gradient_units": "historical_1990_USD_per_km",
        "records": int(len(frame)),
        "grid_size": int(grid_size),
        "supported_grid_points": int(np.isfinite(surface.values).sum()),
        "smoothing_sigma_grid_cells": float(smooth_sigma),
        "max_support_distance_km": float(max_support_distance_km),
        "fixed_reference_features": reference,
        "numerical_double_integral": integral,
        "gradient_magnitude": {
            "mean_usd_per_km": float(finite.mean()) if len(finite) else None,
            "max_usd_per_km": float(finite.max()) if len(finite) else None,
        },
        "outputs": {
            "housing_map": str(map_path),
            "price_surface_3d": str(surface_html),
            "price_contours": str(contours_html),
            "price_gradients": str(gradients_html),
            "numerical_surface": str(surface_npz),
        },
        "limitations": [
            "Outputs describe area-level 1990 median values, not current individual-property prices.",
            "The price surface holds all non-geographical features (including ocean proximity) fixed.",
            "The original tree model is discontinuous; derivatives refer only to a smoothed scenario surface.",
            "Gaussian smoothing and grid resolution alter gradient magnitudes and directions.",
            "The integration domain is a grid region supported by observed locations, not official California land boundaries.",
            "Gradient arrows describe modelled local patterns, not causal price changes from moving a property.",
        ],
    }
    report_path = outputs / "geospatial-report.json"
    report["outputs"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
