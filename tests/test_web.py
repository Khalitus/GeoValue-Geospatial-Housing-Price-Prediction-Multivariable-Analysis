"""FastAPI integration checks; no remote services or user dataset required."""
import json

import numpy as np
from fastapi.testclient import TestClient

from geovalue import web
from geovalue.data import INPUT_COLUMNS
from geovalue.modeling import build_model, save_model

CLIENT = TestClient(web.app)
SAMPLE = {
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
    "total_rooms": 880, "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY",
}


def test_serves_dashboard_and_browser_assets():
    assert CLIENT.get("/").status_code == 200
    assert "Allegheny County" in CLIENT.get("/").text
    html = CLIENT.get("/legacy").text
    assert "California housing," in html
    assert "Select a supported region" in html
    assert "Gradient field" in html
    assert 'aria-label="Main navigation"' in html
    assert "GeoValue" in CLIENT.get("/app.js").text
    assert ".site-header" in CLIENT.get("/styles.css").text
    assert CLIENT.get("/favicon.svg").status_code == 200


def test_health_does_not_pretend_missing_model_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "model_path", lambda: tmp_path / "absent.joblib")
    payload = CLIENT.get("/api/health").json()
    assert payload["status"] == "model_missing"
    assert payload["model_available"] is False
    response = CLIENT.post("/api/predict", json=SAMPLE)
    assert response.status_code == 503


def test_api_rejects_bad_user_features():
    assert CLIENT.post("/api/predict", json={**SAMPLE, "latitude": 200}).status_code == 422
    assert CLIENT.post("/api/predict", json={**SAMPLE, "households": 0}).status_code == 422
    assert CLIENT.post("/api/predict", json={**SAMPLE, "ocean_proximity": "UNKNOWN"}).status_code == 422


def test_prediction_uses_fitted_pipeline(tmp_path, monkeypatch, synthetic_housing):
    x = synthetic_housing.loc[:, list(INPUT_COLUMNS)]
    model = build_model("linear").fit(x, synthetic_housing["median_house_value"])
    path = save_model(model, tmp_path / "model.joblib", model_name="linear")
    monkeypatch.setattr(web, "model_path", lambda: path)
    web._cached_pipeline.cache_clear()
    response = CLIENT.post("/api/predict", json=SAMPLE)
    assert response.status_code == 200, response.text
    assert np.isfinite(response.json()["predicted_median_house_value"])
    assert response.json()["price_year"] == 1990
    assert response.json()["warnings"] == []
    extreme = CLIENT.post("/api/predict", json={**SAMPLE, "longitude": -90.0})
    assert extreme.status_code == 200
    assert "outside the approximate California" in extreme.json()["warnings"][0]
    web._cached_pipeline.cache_clear()


def test_prepared_training_report_and_missing_report(tmp_path, monkeypatch):
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    report = {"selected_model": "xgboost", "held_out_test": {"r2": 0.84347}}
    (deploy / "training-report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(web, "DEPLOY", deploy)
    monkeypatch.setattr(web, "PUBLIC", tmp_path / "public")
    monkeypatch.setattr(web, "ROOT", tmp_path)

    response = CLIENT.get("/api/report")
    assert response.status_code == 200
    assert response.json() == report

    (deploy / "training-report.json").unlink()
    assert CLIENT.get("/api/report").status_code == 503


def test_gradient_query_uses_smooth_surface_only(tmp_path, monkeypatch):
    public = tmp_path / "public"
    (public / "data").mkdir(parents=True)
    payload = {
        "longitude": [[-122.0, -121.9],[-122.0,-121.9]],
        "latitude": [[37.0,37.0],[37.1,37.1]],
        "values": [[100,110],[120,130]],
        "d_dx": [[2,2],[2,2]],
        "d_dy": [[3,3],[3,3]],
    }
    (public / "data/surface.json").write_text(json.dumps(payload))
    monkeypatch.setattr(web, "PUBLIC", public)
    monkeypatch.setattr(web, "DEPLOY", tmp_path / "deploy")
    response = CLIENT.post("/api/gradient", json={"longitude":-122,"latitude":37,"bearing":90})
    assert response.status_code == 200, response.text
    assert abs(response.json()["directional_derivative_usd_per_km"] - 2) < 1e-9
    assert response.json()["sampled_grid_longitude"] == -122.0
    assert response.json()["sampled_grid_latitude"] == 37.0
    assert CLIENT.post("/api/gradient", json={"longitude":-117,"latitude":37,"bearing":90}).status_code == 422


def test_unavailable_surface_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "PUBLIC", tmp_path)
    monkeypatch.setattr(web, "DEPLOY", tmp_path / "deploy")
    assert CLIENT.get("/api/surface").status_code == 503
    assert CLIENT.post("/api/gradient", json={"longitude":-122,"latitude":37}).status_code == 503
