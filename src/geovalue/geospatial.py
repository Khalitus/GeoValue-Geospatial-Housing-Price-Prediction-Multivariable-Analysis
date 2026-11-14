"""Geographical projection and interactive observed/predicted housing-value map."""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.colormap import linear
from pyproj import Transformer

from geovalue.data import TARGET_COLUMN

CRS_GEOGRAPHIC = "EPSG:4326"
CRS_CALIFORNIA = "EPSG:3310"  # NAD83 / California Albers; projected metres
_TO_CALIFORNIA = Transformer.from_crs(CRS_GEOGRAPHIC, CRS_CALIFORNIA, always_xy=True)
_TO_GEOGRAPHIC = Transformer.from_crs(CRS_CALIFORNIA, CRS_GEOGRAPHIC, always_xy=True)


def project_locations(longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude (degrees) to California Albers x/y (metres)."""
    x, y = _TO_CALIFORNIA.transform(longitude, latitude)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def unproject_locations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert California Albers x/y (metres) back to longitude/latitude."""
    lon, lat = _TO_GEOGRAPHIC.transform(x, y)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def make_housing_map(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    predictions: np.ndarray | None = None,
    max_points: int = 1250,
    random_state: int = 42,
) -> Path:
    """Export an interactive Folium map with observed values and optional model predictions."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if frame.empty or any(c not in frame for c in ("longitude", "latitude", TARGET_COLUMN)):
        raise ValueError("Map data must include coordinates and observed median house values")

    coords = frame.loc[:, ["longitude", "latitude", TARGET_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Map coordinates and observed prices must be finite")
    if predictions is not None:
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
            raise ValueError("Predictions must be a finite value for every record")

    # Select the same reproducible subset for observed and predicted layers.
    indexes = np.arange(len(frame))
    if len(indexes) > max_points:
        indexes = np.sort(np.random.default_rng(random_state).choice(indexes, max_points, replace=False))
    chosen = coords[indexes]
    center = [float(np.median(chosen[:, 1])), float(np.median(chosen[:, 0]))]
    chart = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap", control_scale=True)

    all_values = chosen[:, 2]
    if predictions is not None:
        all_values = np.concatenate((all_values, predictions[indexes]))
    low, high = np.percentile(all_values, [5, 95])
    if low >= high:
        high = low + 1.0
    colors = linear.YlOrRd_09.scale(float(low), float(high))
    colors.caption = "Housing value (historical 1990 USD)"
    colors.add_to(chart)

    layers = [("Observed median values", chosen[:, 2], True)]
    if predictions is not None:
        layers.append(("Model predictions", predictions[indexes], False))

    for title, values, shown in layers:
        layer = folium.FeatureGroup(name=title, show=shown)
        for (lon, lat, _), value in zip(chosen, values, strict=True):
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=3.4,
                color=colors(float(value)),
                fill=True,
                fill_color=colors(float(value)),
                fill_opacity=0.80,
                weight=0.4,
                tooltip=f"{title}: ${value:,.0f} (1990 USD)",
            ).add_to(layer)
        layer.add_to(chart)
    folium.LayerControl(collapsed=False).add_to(chart)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output))
    return output
