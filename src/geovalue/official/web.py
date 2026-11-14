"""Local FastAPI routes for official Allegheny County transaction predictions."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from geovalue.official.modeling import predict, load_model
from geovalue.official.reliability import reliability_for_location
from geovalue.official.spatial import gradient_query
from geovalue.calculus import load_price_surface

router = APIRouter(prefix="/api/official", tags=["Official residential sale analysis"])
ROOT = Path(__file__).resolve().parents[3]


class PropertyInput(BaseModel):
    latitude: float = Field(ge=39.5, le=41.0)
    longitude: float = Field(ge=-81.0, le=-79.0)
    bedrooms: float = Field(ge=0, le=20)
    full_baths: float | None = Field(default=None, ge=0)
    half_baths: float | None = Field(default=None, ge=0)
    total_rooms: float | None = Field(default=None, ge=0)
    living_area_sqft: float = Field(ge=250, le=30000)
    lot_area_sqft: float = Field(gt=0)
    year_built: float = Field(ge=1700, le=2025)
    stories: float | None = Field(default=None, ge=0)
    fireplaces: float | None = Field(default=None, ge=0)
    garage_spaces: float | None = Field(default=None, ge=0)
    sale_year: int = Field(ge=2013, le=2025)
    sale_month: int = Field(ge=1, le=12)
    construction_grade: str = "Unknown"
    condition_code: str = "Unknown"
    heating_cooling: str = "Unknown"
    municipality: str = "Unknown"
    zip_code: str = "Unknown"


class GradientInput(BaseModel):
    longitude: float = Field(ge=-81.0, le=-79.0)
    latitude: float = Field(ge=39.5, le=41.0)
    bearing: float = Field(default=90, ge=0, lt=360)


def official_model_path() -> Path:
    env = os.environ.get("GEOVALUE_OFFICIAL_MODEL_PATH")
    return Path(env).expanduser().resolve() if env else ROOT / "artifacts/official/residential-sale-model.joblib"


def evidence_path() -> Path:
    env = os.environ.get("GEOVALUE_OFFICIAL_EVIDENCE_PATH")
    return Path(env).expanduser().resolve() if env else ROOT / "artifacts/official/reliability-support.npz"


def official_report_path(name: str) -> Path:
    return ROOT / "reports/official" / name


def _json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=4)
def _model(path: str):
    return load_model(Path(path))["pipeline"]


@router.get("/health")
def health():
    available = official_model_path().is_file()
    return {
        "model_available": available,
        "status": "ready" if available else "model_missing",
        "surface_available": official_report_path("price-surface.npz").is_file(),
        "reliability_available": evidence_path().is_file() and official_report_path("reliability-report.json").is_file(),
        "scope": "official Allegheny County residential recorded sales through 2025",
    }


@router.get("/report")
def official_report():
    result = _json(official_report_path("training-report.json"))
    if result is None:
        raise HTTPException(status_code=503, detail="Train the official model to generate the training report")
    return result


@router.get("/geo-report")
def official_geo_report():
    result = _json(official_report_path("geospatial-report.json"))
    if result is None:
        raise HTTPException(status_code=503, detail="Run geovalue-official analyze to generate geospatial outputs")
    return result


@router.get("/reliability-report")
def official_reliability_report():
    result = _json(official_report_path("reliability-report.json"))
    if result is None:
        raise HTTPException(status_code=503, detail="Run geovalue-official analyze to calibrate spatial evidence")
    return result


@router.post("/predict")
def predict_property(record: PropertyInput):
    model_path = official_model_path()
    if not model_path.is_file():
        raise HTTPException(status_code=503, detail="Official model not trained; run geovalue-official train")
    try:
        pipeline = _model(str(model_path))
        from geovalue.official.modeling import validate_inputs
        frame = validate_inputs(pd.DataFrame([record.model_dump()]))
        predicted = float(pipeline.predict(frame)[0])
        if not 0 <= predicted < float("inf"):
            raise ValueError("Model returned an invalid price")
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    support = None
    calibration = _json(official_report_path("reliability-report.json"))
    if evidence_path().is_file() and calibration:
        try:
            support = reliability_for_location(record.longitude, record.latitude, evidence_path(), calibration)
        except (ValueError, OSError, KeyError):
            support = None
    return {
        "predicted_sale_price_usd": predicted,
        "price_basis": "recorded historical sale price; unadjusted for inflation",
        "spatial_evidence": support,
        "limitations": [
            "Current property assessments may differ from the attributes recorded at the historical sale date.",
            "Spatial-support level is not a prediction confidence interval or an appraisal.",
        ],
    }


@router.post("/gradient")
def official_gradient(request: GradientInput):
    path = official_report_path("price-surface.npz")
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Official surface missing; run geovalue-official analyze")
    try:
        return gradient_query(load_price_surface(path), request.longitude, request.latitude, request.bearing)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
