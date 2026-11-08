import numpy as np
import pytest

from geovalue.calculus import (
    PriceSurface, build_price_surface, gradient_at_location,
    integrate_price_surface, load_price_surface, save_price_surface,
    spatial_gradient,
)
from geovalue.geospatial import project_locations


def test_spatial_gradient_exact_for_linear_function():
    x = np.linspace(-6, 6, 7)
    y = np.linspace(0, 10, 9)
    grid_x, grid_y = np.meshgrid(x, y)
    z = 100_000 + 3 * grid_x - 7 * grid_y
    gx, gy = spatial_gradient(x, y, z)
    np.testing.assert_allclose(gx, 3, atol=1e-10)
    np.testing.assert_allclose(gy, -7, atol=1e-10)


def test_spatial_gradient_preserves_invalid_mask():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 4, 5)
    z = np.add.outer(y, x)
    z[2, 2] = np.nan
    gx, gy = spatial_gradient(x, y, z)
    assert np.isnan(gx[2, 2]) and np.isnan(gy[2, 2])
    assert np.isfinite(gx[0, 0])


def test_spatial_gradient_rejects_invalid_spacing():
    with pytest.raises(ValueError, match="strictly increasing"):
        spatial_gradient(np.array([0, 0, 1]), np.arange(3), np.ones((3, 3)))


def test_double_integral_recovers_exact_affine_surface():
    x = np.linspace(0, 2, 5)
    y = np.linspace(0, 4, 9)
    X, Y = np.meshgrid(x, y)
    z = 100 + 2 * X + 3 * Y
    zeros = np.zeros_like(z)
    surface = PriceSurface(x, y, z, zeros, zeros)
    summary = integrate_price_surface(surface)
    assert summary["valid_area_km2"] == pytest.approx(8)
    assert summary["mean_surface_value_usd"] == pytest.approx(108)
    assert summary["integral_usd_km2"] == pytest.approx(864)
    assert summary["included_grid_cells"] == 32


def test_integral_excludes_cells_containing_missing_values():
    x = np.linspace(0, 2, 3)
    y = np.linspace(0, 2, 3)
    z = np.ones((3, 3)) * 5
    z[0, 0] = np.nan
    s = PriceSurface(x, y, z, z, z)
    summary = integrate_price_surface(s)
    assert summary["valid_area_km2"] == pytest.approx(3)
    assert summary["integral_usd_km2"] == pytest.approx(15)


def test_gradient_query_bearing_and_direction():
    xm, ym = project_locations(np.array([-120.0]), np.array([37.0]))
    cx, cy = float(xm[0] / 1000), float(ym[0] / 1000)
    x = np.linspace(cx - 12, cx + 12, 7)
    y = np.linspace(cy - 12, cy + 12, 7)
    X, Y = np.meshgrid(x, y)
    z = 130_000 + 2 * (X - cx) + 4 * (Y - cy)
    gx, gy = spatial_gradient(x, y, z)
    surface = PriceSurface(x, y, z, gx, gy)
    north = gradient_at_location(surface, longitude=-120, latitude=37, bearing_degrees=0)
    east = gradient_at_location(surface, longitude=-120, latitude=37, bearing_degrees=90)
    assert north["directional_derivative_usd_per_km"] == pytest.approx(4)
    assert east["directional_derivative_usd_per_km"] == pytest.approx(2)
    assert east["gradient_magnitude_usd_per_km"] == pytest.approx(np.hypot(2, 4))
    assert east["steepest_increase_bearing_degrees"] == pytest.approx(np.degrees(np.arctan2(2, 4)))


def test_gradient_query_rejects_locations_outside_support():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 4, 5)
    z = np.ones((5, 5)) * 50
    s = PriceSurface(x, y, z, z, z)
    with pytest.raises(ValueError, match="No calculated gradient"):
        gradient_at_location(s, longitude=-120, latitude=37, max_distance_km=1)


def test_surface_round_trip(tmp_path):
    x, y = np.arange(5, dtype=float), np.arange(4, dtype=float)
    z = np.ones((4, 5))
    original = PriceSurface(x, y, z, z * 2, z * 3)
    result = load_price_surface(save_price_surface(original, tmp_path / "surface.npz"))
    np.testing.assert_array_equal(result.values, original.values)
    np.testing.assert_array_equal(result.d_dx, original.d_dx)


def test_build_smoothed_scenario_surface(synthetic_housing):
    class DummyPredictor:
        def predict(self, frame):
            return 100_000 + 5_000 * (frame["longitude"].to_numpy() + 120)

    surface, ref = build_price_surface(
        synthetic_housing, DummyPredictor(), grid_size=35,
        smooth_sigma=1.5, max_support_distance_km=120,
    )
    assert surface.values.shape == (35, 35)
    assert np.isfinite(surface.values).sum() > 15
    assert len(ref) == 9
    assert "ocean_proximity" in ref
    assert np.isnan(surface.values).any()  # Mask the unsupported exterior of the point cloud
    assert integrate_price_surface(surface)["valid_area_km2"] > 0
