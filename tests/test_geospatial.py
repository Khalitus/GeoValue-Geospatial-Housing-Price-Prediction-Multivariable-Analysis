import numpy as np
import pytest

from geovalue.geospatial import (
    make_housing_map, project_locations, unproject_locations,
)


def test_california_projection_round_trip():
    lon = np.array([-122.23, -118.24, -121.48])
    lat = np.array([37.88, 34.05, 38.57])
    x, y = project_locations(lon, lat)
    lon_out, lat_out = unproject_locations(x, y)
    np.testing.assert_allclose(lon_out, lon, atol=1e-6)
    np.testing.assert_allclose(lat_out, lat, atol=1e-6)
    assert np.ptp(x) > 1000


def test_map_includes_observed_and_predicted_layers(housing_frame, tmp_path):
    path = make_housing_map(
        housing_frame, tmp_path / "map.html",
        predictions=np.array([120_000, 130_000, 140_000, 150_000]),
        max_points=3,
    )
    html = path.read_text()
    assert "Observed median values" in html
    assert "Model predictions" in html
    assert "1990 USD" in html


def test_map_rejects_bad_predictions(housing_frame, tmp_path):
    with pytest.raises(ValueError, match="Predictions"):
        make_housing_map(housing_frame, tmp_path / "map.html", predictions=np.array([42]))
    with pytest.raises(ValueError, match="max_points"):
        make_housing_map(housing_frame, tmp_path / "map.html", max_points=0)
