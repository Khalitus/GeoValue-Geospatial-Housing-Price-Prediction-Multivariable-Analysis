"""FastAPI endpoints for the GeoValue browser application.

No model training or large geospatial calculations happen in web requests.
Publication requires the trusted, locally generated deployment artifacts.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from geovalue.inference import OCEAN_CATEGORIES, validate_prediction_inputs
from geovalue.modeling import load_model

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public"
DEPLOY = ROOT / "deploy"

app = FastAPI(title="GeoValue API", version="0.4.1", description="Historical California housing-value predictions")


class HousingInput(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float | None = Field(default=None, ge=0)
    population: float = Field(ge=0)
    households: float = Field(gt=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


class GradientInput(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    bearing: float = Field(default=0, ge=0, lt=360)


def model_path() -> Path:
    configured = os.environ.get("GEOVALUE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (DEPLOY / "housing-model.joblib", ROOT / "artifacts/housing-model.joblib"):
        if candidate.is_file():
            return candidate
    return DEPLOY / "housing-model.joblib"


@lru_cache(maxsize=4)
def _cached_pipeline(path: str):
    # Trust only deployment artifacts created by the local GeoValue training pipeline.
    return load_model(path)["pipeline"]


def load_json_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PUBLIC / "index.html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(PUBLIC / "favicon.svg", media_type="image/svg+xml")


@app.get("/styles.css", include_in_schema=False)
def css():
    return FileResponse(PUBLIC / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript():
    return FileResponse(PUBLIC / "app.js", media_type="application/javascript")


@app.get("/data/{filename}", include_in_schema=False)
def public_data(filename: str):
    # A strict whitelist avoids serving arbitrary project files.
    if filename not in {"points.json", "surface.json", "training-report.json", "geospatial-report.json"}:
        raise HTTPException(status_code=404, detail="Not found")
    path = PUBLIC / "data" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not generated")
    return FileResponse(path, media_type="application/json")


@app.get("/api/health")
def health():
    return {
        "service": "GeoValue", "status": "ready" if model_path().is_file() else "model_missing",
        "model_available": model_path().is_file(),
        "geography_available": (PUBLIC / "data/points.json").is_file(),
        "surface_available": (PUBLIC / "data/surface.json").is_file(),
        "target": "1990 California census-area median house value",
    }


@app.get("/api/report")
def report():
    document = (load_json_file(DEPLOY / "training-report.json")
                or load_json_file(PUBLIC / "data/training-report.json")
                or load_json_file(ROOT / "reports/training-report.json"))
    if document is None:
        raise HTTPException(status_code=503, detail="Training report not prepared")
    return document


@app.get("/api/geo-report")
def geo_report():
    document = (load_json_file(DEPLOY / "geospatial-report.json")
                or load_json_file(PUBLIC / "data/geospatial-report.json")
                or load_json_file(ROOT / "reports/geospatial/geospatial-report.json"))
    if document is None:
        raise HTTPException(status_code=503, detail="Geospatial report not prepared")
    return document


@app.get("/api/map-points")
def map_points():
    path = PUBLIC / "data/points.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Generate map assets with prepare_deployment.py")
    return FileResponse(path, media_type="application/json")


@app.get("/api/surface")
def surface():
    path = PUBLIC / "data/surface.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Generate surface assets with prepare_deployment.py")
    return FileResponse(path, media_type="application/json")


@app.post("/api/predict")
def predict(record: HousingInput):
    if record.ocean_proximity not in OCEAN_CATEGORIES:
        raise HTTPException(status_code=422, detail="Invalid ocean_proximity category")
    path = model_path()
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Trained model unavailable; prepare deployment assets first")
    frame = pd.DataFrame([record.model_dump()])
    try:
        features = validate_prediction_inputs(frame)
        predicted = float(_cached_pipeline(str(path)).predict(features)[0])
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not math.isfinite(predicted):
        raise HTTPException(status_code=500, detail="Model generated a non-finite prediction")
    warnings = []
    # The historical training sample covers California, not arbitrary world coordinates.
    if not (-124.5 <= record.longitude <= -114.0 and 32.0 <= record.latitude <= 42.1):
        warnings.append("This location is outside the approximate California training region. The estimate may be unreliable.")
    if record.median_income > 15:
        warnings.append("Median income is unusually high for the 1990 dataset scale; interpret the prediction cautiously.")
    return {"predicted_median_house_value": predicted, "currency": "USD", "price_year": 1990,
            "interpretation": "Historical census-area median value, not an individual sale price.",
            "warnings": warnings}


@app.post("/api/gradient")
def gradient(req: GradientInput):
    """Nearest grid sample of precomputed projected gradients; no raw XGBoost differentiation."""
    payload = load_json_file(DEPLOY / "surface.json") or load_json_file(PUBLIC / "data/surface.json")
    if payload is None:
        raise HTTPException(status_code=503, detail="Surface asset unavailable")
    try:
        lon = np.asarray(payload["longitude"], dtype=float)
        lat = np.asarray(payload["latitude"], dtype=float)
        dx = np.asarray(payload["d_dx"], dtype=float)
        dy = np.asarray(payload["d_dy"], dtype=float)
        values = np.asarray(payload["values"], dtype=float)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Malformed surface asset") from exc
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(values)
    if not valid.any():
        raise HTTPException(status_code=503, detail="No supported locations on gradient surface")
    rad = np.pi / 180.0
    delta_lat = (lat[valid] - req.latitude) * 111.2
    delta_lon = (lon[valid] - req.longitude) * 111.2 * math.cos(req.latitude * rad)
    distances = np.hypot(delta_lat, delta_lon)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > 30:
        raise HTTPException(status_code=422, detail="No supported geographical gradient within 30 km")
    gx, gy = float(dx[valid][index]), float(dy[valid][index])
    theta = req.bearing * rad
    return {
        "nearest_distance_km": distance,
        "sampled_grid_longitude": float(lon[valid][index]),
        "sampled_grid_latitude": float(lat[valid][index]),
        "sampled_surface_value_usd": float(values[valid][index]),
        "gradient_east_usd_per_km": gx,
        "gradient_north_usd_per_km": gy,
        "magnitude_usd_per_km": math.hypot(gx, gy),
        "directional_derivative_usd_per_km": gx * math.sin(theta) + gy * math.cos(theta),
        "steepest_ascent_bearing_grid_degrees": (math.degrees(math.atan2(gx, gy)) + 360) % 360,
        "bearing_reference": "California Albers projected grid north, not true north",
        "disclaimer": "Gradient of smoothed fixed-feature modeled surface, not a causal effect.",
    }
