"""Optional local FastAPI endpoints; Task 7 will migrate the UI to these APIs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel, Field

from geovalue.official.modeling import predict, load_model

router=APIRouter(prefix='/api/official',tags=['Official residential sale predictions'])
ROOT=Path(__file__).resolve().parents[3]


class PropertyInput(BaseModel):
    latitude: float=Field(ge=39.5,le=41.0)
    longitude: float=Field(ge=-81.0,le=-79.0)
    bedrooms: float=Field(ge=0,le=20)
    full_baths: float | None=Field(default=None,ge=0)
    half_baths: float | None=Field(default=None,ge=0)
    total_rooms: float | None=Field(default=None,ge=0)
    living_area_sqft: float=Field(ge=250,le=30000)
    lot_area_sqft: float=Field(gt=0)
    year_built: float=Field(ge=1700,le=2025)
    stories: float | None=Field(default=None,ge=0)
    fireplaces: float | None=Field(default=None,ge=0)
    garage_spaces: float | None=Field(default=None,ge=0)
    sale_year: int=Field(ge=2013,le=2025)
    sale_month: int=Field(ge=1,le=12)
    construction_grade: str='Unknown'
    condition_code: str='Unknown'
    heating_cooling: str='Unknown'
    municipality: str='Unknown'
    zip_code: str='Unknown'


def official_model_path() -> Path:
    env=os.environ.get('GEOVALUE_OFFICIAL_MODEL_PATH')
    return Path(env).expanduser().resolve() if env else ROOT/'artifacts/official/residential-sale-model.joblib'


@router.get('/health')
def health():
    available=official_model_path().is_file()
    return {'model_available':available,'status':'ready' if available else 'model_missing',
            'scope':'official county residential transactions through 2025'}


@router.post('/predict')
def predict_property(record: PropertyInput):
    model=official_model_path()
    if not model.is_file():
        raise HTTPException(status_code=503,detail='Official model not trained; run geovalue-official train')
    try:
        output=predict(model,pd.DataFrame([record.model_dump()]))
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {'predicted_sale_price_usd':float(output['predicted_sale_price_usd'].iloc[0]),
            'price_basis':'recorded historical sale price; unadjusted for inflation',
            'limitations':['Current assessment snapshot may not reflect characteristics at the time of sale.',
                           'Estimate is not a current appraisal or confidence interval.']}
