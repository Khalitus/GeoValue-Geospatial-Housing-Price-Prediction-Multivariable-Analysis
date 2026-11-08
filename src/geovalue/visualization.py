"""Interactive, browser-viewable 3D surface, contour and gradient plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from geovalue.calculus import PriceSurface


def _write_figure(fig: go.Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=30, b=35, t=65))
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    return output


def plot_price_surface(surface: PriceSurface, output_path: str | Path) -> Path:
    """Interactive 3D surface of modelled historical values in California Albers."""
    fig = go.Figure(data=[go.Surface(
        x=surface.x_km, y=surface.y_km, z=surface.values,
        colorscale="Viridis", colorbar=dict(title="1990 USD"),
        hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>modelled=$%{z:,.0f}<extra></extra>",
    )])
    fig.update_layout(
        title="GeoValue | Smoothed historical housing-value scenario",
        scene=dict(xaxis_title="CA Albers east (km)", yaxis_title="CA Albers north (km)", zaxis_title="Modelled median value (1990 USD)"),
    )
    return _write_figure(fig, output_path)


def plot_price_contours(surface: PriceSurface, output_path: str | Path) -> Path:
    """Interactive level curves for the same masked, smoothed price surface."""
    fig = go.Figure(data=[go.Contour(
        x=surface.x_km, y=surface.y_km, z=surface.values,
        colorscale="Viridis", contours=dict(showlabels=True),
        colorbar=dict(title="1990 USD"),
        hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>modelled=$%{z:,.0f}<extra></extra>",
    )])
    fig.update_layout(
        title="GeoValue | Level curves (contour map)",
        xaxis_title="CA Albers east (km)", yaxis_title="CA Albers north (km)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return _write_figure(fig, output_path)


def plot_gradient_field(surface: PriceSurface, output_path: str | Path, *, max_arrows_per_axis: int = 16) -> Path:
    """Overlay gradients on a price field, arrowheads indicating the steepest increase."""
    if max_arrows_per_axis < 2:
        raise ValueError("max_arrows_per_axis must be at least 2")
    magnitude = np.hypot(surface.d_dx, surface.d_dy)
    fig = go.Figure(data=[go.Heatmap(
        x=surface.x_km, y=surface.y_km, z=magnitude,
        colorscale="Blues", colorbar=dict(title="USD/km"),
        hovertemplate="x=%{x:.0f} km<br>y=%{y:.0f} km<br>|gradient|=%{z:,.0f} USD/km<extra></extra>",
    )])
    step = max(1, int(np.ceil(max(len(surface.x_km), len(surface.y_km)) / max_arrows_per_axis)))
    x_spacing = float(np.median(np.diff(surface.x_km)))
    y_spacing = float(np.median(np.diff(surface.y_km)))
    length = min(x_spacing, y_spacing) * step * 0.65
    for j in range(1, len(surface.y_km) - 1, step):
        for i in range(1, len(surface.x_km) - 1, step):
            gx, gy = surface.d_dx[j, i], surface.d_dy[j, i]
            m = float(np.hypot(gx, gy))
            if not np.isfinite(m) or m < 1e-8:
                continue
            px, py = float(surface.x_km[i]), float(surface.y_km[j])
            fig.add_annotation(
                x=px + length * gx / m, y=py + length * gy / m,
                ax=px, ay=py, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=0.65, arrowwidth=1.2,
                arrowcolor="#202D38", opacity=0.85,
            )
    fig.update_layout(
        title="GeoValue | Spatial price-gradient field (illustrative scenario)",
        xaxis_title="CA Albers east (km)", yaxis_title="CA Albers north (km)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return _write_figure(fig, output_path)
