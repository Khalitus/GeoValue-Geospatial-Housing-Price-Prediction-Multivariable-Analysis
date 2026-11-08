"""Smooth geographic price surfaces, spatial derivatives and numerical integration.

The underlying tree models are piecewise constant: do not differentiate their raw
predictions. A masked Gaussian-smoothed *scenario surface* is used instead.
Coordinates are California Albers kilometres (EPSG:3310), so spatial derivatives
have units of historical USD/km, and integrals have USD km² units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from geovalue.data import INPUT_COLUMNS, NUMERIC_INPUT_COLUMNS
from geovalue.geospatial import project_locations, unproject_locations


@dataclass(frozen=True)
class PriceSurface:
    """Masked projected grid and smooth price derivatives (USD/km)."""

    x_km: np.ndarray
    y_km: np.ndarray
    values: np.ndarray
    d_dx: np.ndarray
    d_dy: np.ndarray


def spatial_gradient(
    x_km: np.ndarray, y_km: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d(value)/dx and d(value)/dy; axes are projected kilometres."""
    x = np.asarray(x_km, dtype=float)
    y = np.asarray(y_km, dtype=float)
    z = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or z.shape != (len(y), len(x)):
        raise ValueError("Surface shape must be (len(y_km), len(x_km))")
    if len(x) < 3 or len(y) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Spatial gradient requires at least a 3x3 grid of finite coordinates")
    if not ((np.diff(x) > 0).all() and (np.diff(y) > 0).all()):
        raise ValueError("Surface coordinates must be strictly increasing")
    if np.isinf(z).any():
        raise ValueError("Surface prices must not be infinite")
    d_dy, d_dx = np.gradient(z, y, x, edge_order=2)
    d_dx[~np.isfinite(z)] = np.nan
    d_dy[~np.isfinite(z)] = np.nan
    return d_dx, d_dy


def integrate_price_surface(surface: PriceSurface) -> dict[str, float]:
    """Integrate over valid 4-corner grid cells with 2D trapezoid quadrature.

    Integrals cover the observed-data-supported part of the grid; this is not
    the official land area of California or an administrative boundary.
    """
    z = np.asarray(surface.values, dtype=float)
    x = np.asarray(surface.x_km, dtype=float)
    y = np.asarray(surface.y_km, dtype=float)
    if z.shape != (len(y), len(x)) or len(x) < 2 or len(y) < 2:
        raise ValueError("Integration needs a 2D grid with at least 2 points per axis")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Integration coordinates must be finite")
    dx, dy = np.diff(x), np.diff(y)
    if not ((dx > 0).all() and (dy > 0).all()):
        raise ValueError("Integration coordinates must be strictly increasing")
    if np.isinf(z).any():
        raise ValueError("Cannot integrate infinite surface values")

    corners = (z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:])
    valid = np.logical_and.reduce([np.isfinite(arr) for arr in corners])
    area_cells = dy[:, None] * dx[None, :]
    cell_means = sum(np.where(valid, arr, 0.0) for arr in corners) / 4.0
    area = float(np.sum(area_cells[valid]))
    if area <= 0:
        raise ValueError("No fully supported cells available for numerical integration")
    integral = float(np.sum(cell_means[valid] * area_cells[valid]))
    return {
        "valid_area_km2": area,
        "integral_usd_km2": integral,
        "mean_surface_value_usd": integral / area,
        "included_grid_cells": int(valid.sum()),
    }


def reference_housing_record(frame: pd.DataFrame) -> dict[str, float | str]:
    """Fix non-location inputs to dataset medians/mode for a controlled scenario."""
    if frame.empty:
        raise ValueError("Cannot build a scenario from an empty dataset")
    reference: dict[str, float | str] = {}
    for column in NUMERIC_INPUT_COLUMNS:
        column_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(column_values).sum() == 0:
            raise ValueError(f"No finite values for reference feature {column}")
        reference[column] = float(np.nanmedian(column_values))
    category_mode = frame["ocean_proximity"].mode(dropna=True)
    if category_mode.empty:
        raise ValueError("No ocean_proximity category for reference scenario")
    reference["ocean_proximity"] = str(category_mode.iloc[0])
    return reference


def build_price_surface(
    frame: pd.DataFrame,
    model: object,
    *,
    grid_size: int = 90,
    smooth_sigma: float = 1.6,
    max_support_distance_km: float = 45.0,
) -> tuple[PriceSurface, dict[str, float | str]]:
    """Predict a fixed-feature 2D location scenario, then smooth and mask it.

    Grid points farther than max_support_distance_km from any observed data
    are removed so the model is not presented as reliable across empty regions.
    The category 'ocean_proximity' stays fixed for all grid points by design.
    """
    if not 12 <= grid_size <= 250:
        raise ValueError("grid_size must be between 12 and 250")
    if not 0.5 <= smooth_sigma <= 10:
        raise ValueError("smooth_sigma must be between 0.5 and 10 grid cells")
    if not 1 <= max_support_distance_km <= 150:
        raise ValueError("max_support_distance_km must be between 1 and 150")
    if len(frame) < 4:
        raise ValueError("At least four observed areas are required to build a surface")

    lon = frame["longitude"].to_numpy(dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    x_m, y_m = project_locations(lon, lat)
    if not (np.isfinite(x_m).all() and np.isfinite(y_m).all()):
        raise ValueError("Non-finite projected geographical coordinates")
    if np.ptp(x_m) < 1 or np.ptp(y_m) < 1:
        raise ValueError("Observed coordinates must span a geographical region")

    x_km = np.linspace(float(x_m.min()) / 1000, float(x_m.max()) / 1000, grid_size)
    y_km = np.linspace(float(y_m.min()) / 1000, float(y_m.max()) / 1000, grid_size)
    grid_x, grid_y = np.meshgrid(x_km, y_km)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    nearest, _ = cKDTree(np.column_stack([x_m / 1000, y_m / 1000])).query(points, k=1)
    supported = nearest.reshape(grid_x.shape) <= max_support_distance_km
    if supported.sum() < 15:
        raise ValueError("Not enough geographically supported grid points; adjust resolution/support distance")

    g_lon, g_lat = unproject_locations(grid_x.ravel() * 1000, grid_y.ravel() * 1000)
    reference = reference_housing_record(frame)
    rows = pd.DataFrame({key: [reference[key]] * len(g_lon) for key in INPUT_COLUMNS})
    rows["longitude"] = g_lon
    rows["latitude"] = g_lat
    predictions = np.asarray(model.predict(rows), dtype=float).reshape(grid_x.shape)
    if not np.isfinite(predictions).all():
        raise ValueError("Prediction model returned non-finite surface values")

    numerator = gaussian_filter(np.where(supported, predictions, 0.0), smooth_sigma)
    denominator = gaussian_filter(supported.astype(float), smooth_sigma)
    smooth = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-8)
    smooth[~supported] = np.nan
    d_dx, d_dy = spatial_gradient(x_km, y_km, smooth)
    return PriceSurface(x_km=x_km, y_km=y_km, values=smooth, d_dx=d_dx, d_dy=d_dy), reference


def save_price_surface(surface: PriceSurface, path: str | Path) -> Path:
    """Persist numerical arrays without pickle/object arrays."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x_km=surface.x_km, y_km=surface.y_km, values=surface.values, d_dx=surface.d_dx, d_dy=surface.d_dy)
    return output


def load_price_surface(path: str | Path) -> PriceSurface:
    """Load a numerical surface artifact (NumPy pickle explicitly disabled)."""
    with np.load(path, allow_pickle=False) as arrays:
        surface = PriceSurface(**{key: np.asarray(arrays[key], dtype=float).copy() for key in ("x_km", "y_km", "values", "d_dx", "d_dy")})
    if (surface.values.shape != (len(surface.y_km), len(surface.x_km))
            or surface.d_dx.shape != surface.values.shape
            or surface.d_dy.shape != surface.values.shape):
        raise ValueError("Saved geographical surface has inconsistent shapes")
    return surface


def gradient_at_location(
    surface: PriceSurface, *, longitude: float, latitude: float,
    bearing_degrees: float = 0.0, max_distance_km: float = 30.0,
) -> dict[str, float]:
    """Find nearest supported derivative; bearing 0 is projected grid north, 90 is grid east.

    This is the derivative of a smoothed, fixed-feature scenario surface,
    not the derivative of the raw XGBoost prediction function.
    """
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError("Invalid longitude or latitude")
    if not np.isfinite([longitude, latitude, bearing_degrees, max_distance_km]).all() or max_distance_km <= 0:
        raise ValueError("Bearing and distance must be finite, with a positive distance")
    x, y = project_locations(np.asarray([longitude]), np.asarray([latitude]))
    px, py = x[0] / 1000, y[0] / 1000
    mesh_x, mesh_y = np.meshgrid(surface.x_km, surface.y_km)
    valid = np.isfinite(surface.values) & np.isfinite(surface.d_dx) & np.isfinite(surface.d_dy)
    if not valid.any():
        raise ValueError("The surface has no valid gradient locations")
    dist = np.hypot(mesh_x[valid] - px, mesh_y[valid] - py)
    nearest = int(np.argmin(dist))
    distance = float(dist[nearest])
    if distance > max_distance_km:
        raise ValueError(f"No calculated gradient within {max_distance_km:g} km of the requested point")
    gx = float(surface.d_dx[valid][nearest])
    gy = float(surface.d_dy[valid][nearest])
    theta = np.deg2rad(bearing_degrees)
    magnitude = float(np.hypot(gx, gy))
    direction = float((np.degrees(np.arctan2(gx, gy)) + 360) % 360) if magnitude > 0 else 0.0
    return {
        "surface_value_usd": float(surface.values[valid][nearest]),
        "gradient_east_usd_per_km": gx,
        "gradient_north_usd_per_km": gy,
        "gradient_magnitude_usd_per_km": magnitude,
        "steepest_increase_bearing_degrees": direction,  # relative to projected grid north
        "requested_bearing_degrees": float(bearing_degrees % 360),
        "directional_derivative_usd_per_km": float(gx * np.sin(theta) + gy * np.cos(theta)),
        "distance_to_sampled_gradient_km": distance,
    }
