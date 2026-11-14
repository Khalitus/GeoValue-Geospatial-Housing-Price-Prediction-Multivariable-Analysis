"""Local dashboard endpoints use real generated files and never invent missing data."""
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from geovalue.web import app
from geovalue.official import web
from geovalue.official.modeling import save_model,build_model
from geovalue.official.data import INPUT_COLUMNS
from geovalue.official.spatial import geographic_from_km
from geovalue.calculus import PriceSurface,save_price_surface

client=TestClient(app)


def test_root_is_official_local_gui_and_legacy_california_kept():
    homepage=client.get('/')
    assert homepage.status_code==200
    assert 'Allegheny County' in homepage.text
    assert 'Spatial evidence' in homepage.text
    assert client.get('/legacy').status_code==200
    assert 'California housing,' in client.get('/legacy').text
    assert client.get('/official/styles.css').status_code==200
    assert client.get('/official/app.js').status_code==200
    assert client.get('/official/plotly.min.js').status_code==200
    assert client.get('/official/favicon.svg').status_code==200
    assert client.get('/official/unknown.js').status_code==404


def test_missing_official_reports_return_503(tmp_path,monkeypatch):
    monkeypatch.setattr(web,'ROOT',tmp_path)
    assert client.get('/api/official/report').status_code==503
    assert client.get('/api/official/geo-report').status_code==503
    assert client.get('/api/official/reliability-report').status_code==503
    assert client.post('/api/official/gradient',json={'longitude':-80,'latitude':40.4}).status_code==503


def test_generated_official_reports_are_served(tmp_path,monkeypatch):
    (tmp_path/'reports/official').mkdir(parents=True)
    for name in ('training-report','geospatial-report','reliability-report'):
        (tmp_path/f'reports/official/{name}.json').write_text(json.dumps({'marker':name}))
    monkeypatch.setattr(web,'ROOT',tmp_path)
    assert client.get('/api/official/report').json()['marker']=='training-report'
    assert client.get('/api/official/geo-report').json()['marker']=='geospatial-report'
    assert client.get('/api/official/reliability-report').json()['marker']=='reliability-report'


def test_gradient_endpoint_uses_official_projection(tmp_path,monkeypatch):
    report=tmp_path/'reports/official';report.mkdir(parents=True)
    x=np.linspace(580,588,9);y=np.linspace(4474,4482,9)
    xx,yy=np.meshgrid(x,y)
    surface=PriceSurface(x,y,100+4*xx+5*yy,np.ones_like(xx)*4,np.ones_like(xx)*5)
    save_price_surface(surface,report/'price-surface.npz')
    monkeypatch.setattr(web,'ROOT',tmp_path)
    lon,lat=geographic_from_km(np.array([584]),np.array([4478]))
    response=client.post('/api/official/gradient',json={'longitude':float(lon[0]),'latitude':float(lat[0]),'bearing':90})
    assert response.status_code==200,response.text
    assert response.json()['directional_derivative_usd_per_km']==4


def test_predict_without_spatial_assets_returns_model_prediction(tmp_path,monkeypatch,official_data):
    model=build_model('xgboost',xgboost_trees=3).fit(official_data.loc[:,list(INPUT_COLUMNS)],official_data.sale_price)
    saved=save_model(model,'xgboost',tmp_path/'official.joblib')
    monkeypatch.setattr(web,'official_model_path',lambda:saved)
    monkeypatch.setattr(web,'ROOT',tmp_path)
    web._model.cache_clear()
    payload=official_data.iloc[0].loc[list(web.PropertyInput.model_fields)].to_dict()
    response=client.post('/api/official/predict',json=payload)
    assert response.status_code==200,response.text
    assert response.json()['spatial_evidence'] is None
    assert response.json()['predicted_sale_price_usd']>0
    web._model.cache_clear()


def test_official_static_data_restricted_to_expected_names():
    assert client.get('/official/data/secret.json').status_code==404
    assert client.get('/official/data/../../.gitignore').status_code in (404,422)


def test_predict_includes_empirical_spatial_evidence_when_prepared(tmp_path,monkeypatch,official_data):
    from geovalue.official.reliability import _cached_reference
    from geovalue.official.spatial import project_km
    frame=official_data.loc[:,list(INPUT_COLUMNS)]
    model=build_model('xgboost',xgboost_trees=3).fit(frame,official_data.sale_price)
    path=save_model(model,'xgboost',tmp_path/'model.joblib')
    site=tmp_path/'artifacts/official';site.mkdir(parents=True)
    x,y=project_km(official_data.longitude.to_numpy(float),official_data.latitude.to_numpy(float))
    np.savez_compressed(site/'reliability-support.npz',x_km=x,y_km=y,boundaries=np.array([1.0,3.0]))
    reports=tmp_path/'reports/official';reports.mkdir(parents=True)
    groups={k:{'n':250,'mae_usd':12345.0,'p80_absolute_error_usd':23456.0} for k in ('dense','typical','sparse')}
    (reports/'reliability-report.json').write_text(json.dumps({'validation_error_by_support':groups}))
    monkeypatch.setattr(web,'ROOT',tmp_path)
    monkeypatch.setattr(web,'official_model_path',lambda:path)
    web._model.cache_clear();_cached_reference.cache_clear()
    payload=official_data.iloc[0].loc[list(web.PropertyInput.model_fields)].to_dict()
    response=client.post('/api/official/predict',json=payload)
    assert response.status_code==200,response.text
    support=response.json()['spatial_evidence']
    assert support['level'] in ('dense','typical','sparse')
    assert support['validation_group_mae_usd']==12345.0
    assert support['is_confidence_interval'] is False
    web._model.cache_clear();_cached_reference.cache_clear()
