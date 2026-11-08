import json
from pathlib import Path

import numpy as np
import pytest

from geovalue.analysis import run_geospatial_analysis
from geovalue.calculus import gradient_at_location, load_price_surface
from geovalue.cli import main
from geovalue.data import INPUT_COLUMNS
from geovalue.geospatial import unproject_locations
from geovalue.modeling import build_model, save_model


@pytest.fixture
def analysis_files(tmp_path, synthetic_housing):
    data_path = tmp_path / "housing.csv"
    synthetic_housing.to_csv(data_path, index=False)
    model = build_model("linear").fit(
        synthetic_housing.loc[:, list(INPUT_COLUMNS)], synthetic_housing["median_house_value"]
    )
    model_path = save_model(model, tmp_path / "model.joblib", model_name="linear")
    return data_path, model_path


def test_complete_geographical_pipeline(analysis_files, tmp_path):
    data_path, model_path = analysis_files
    results = run_geospatial_analysis(
        data_path, model_path, tmp_path / "report", grid_size=34,
        smooth_sigma=1.3, max_support_distance_km=120, max_map_points=25,
    )
    assert results["model_name"] == "linear"
    assert results["crs"] == "EPSG:3310"
    assert results["numerical_double_integral"]["valid_area_km2"] > 0
    for name, path in results["outputs"].items():
        assert Path(path).is_file(), name
    assert results["gradient_magnitude"]["mean_usd_per_km"] is not None
    report = json.loads((tmp_path / "report" / "geospatial-report.json").read_text())
    assert len(report["limitations"]) >= 4

    surface = load_price_surface(results["outputs"]["numerical_surface"])
    valid = np.argwhere(np.isfinite(surface.d_dx) & np.isfinite(surface.d_dy))
    assert len(valid) > 0
    j, i = valid[len(valid) // 2]
    lon, lat = unproject_locations(np.array([surface.x_km[i] * 1000]), np.array([surface.y_km[j] * 1000]))
    gradient = gradient_at_location(surface, longitude=float(lon[0]), latitude=float(lat[0]))
    assert np.isfinite(gradient["directional_derivative_usd_per_km"])


def test_cli_analyze_and_gradient(analysis_files, tmp_path, capsys):
    data_path, model_path = analysis_files
    output_dir = tmp_path / "report"
    assert main([
        "analyze", "--data", str(data_path), "--model", str(model_path),
        "--output-dir", str(output_dir), "--grid-size", "26",
        "--support-km", "120", "--map-points", "20",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    surface = load_price_surface(report["outputs"]["numerical_surface"])
    valid = np.argwhere(np.isfinite(surface.d_dx) & np.isfinite(surface.d_dy))
    j, i = valid[len(valid)//2]
    lon, lat = unproject_locations(np.array([surface.x_km[i] * 1000]), np.array([surface.y_km[j] * 1000]))
    assert main([
        "gradient", "--surface", str(report["outputs"]["numerical_surface"]),
        "--longitude", str(float(lon[0])), "--latitude", str(float(lat[0])),
        "--bearing", "90",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "directional_derivative_usd_per_km" in payload
