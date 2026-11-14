"""Task 7 mathematical tests: support calibration, projections, surfaces and integration."""
import json
import numpy as np
import pandas as pd
import pytest

from geovalue.official.spatial import build_surface, project_km, geographic_from_km, gradient_query
from geovalue.official.reliability import (
    support_distance,support_buckets,error_by_support,calibrate_support,
    create_support_artifacts,reliability_for_location,
)
from geovalue.official.analysis import audit_same_day_sales,build_analysis
from geovalue.calculus import integrate_price_surface, PriceSurface
from geovalue.official.modeling import build_model,save_model
from geovalue.official.training import Config,train_evaluate
from geovalue.official.data import INPUT_COLUMNS


def test_county_projected_coordinates_are_kilometres_and_roundtrip():
    lon=np.asarray([-80.0,-79.9]);lat=np.asarray([40.4,40.5]);x,y=project_km(lon,lat)
    assert 200 < x[0] < 900 and 4000 < y[0] < 5000
    a,b=geographic_from_km(x,y)
    assert np.allclose(a,lon,atol=1e-6)
    assert np.allclose(b,lat,atol=1e-6)


def test_support_distance_matches_known_geometry_and_rejects_nan():
    sites=np.asarray([[0,0],[2,0],[6,0],[10,0]],float)
    q=np.asarray([[1,0],[5,0]],float)
    assert np.allclose(support_distance(sites,q,k=2),[1,3])
    with pytest.raises(ValueError,match="Non-finite"):
        support_distance(sites,np.asarray([[np.nan,1]]))


def test_buckets_have_explicit_distance_cutoffs():
    out=support_buckets(np.asarray([0,1.1,2,4]),(1.0,3.0))
    assert out.tolist()==[0,1,1,2]
    with pytest.raises(ValueError):support_buckets(np.asarray([0]),(2,1))


def test_support_errors_are_actual_absolute_residuals():
    d=np.asarray([.5,.8,1.3,1.8,3.1,4.0])
    truth=np.asarray([100,100,100,100,100,100],float)
    predicted=np.asarray([90,110,80,120,60,140],float)
    out=error_by_support(d,truth,predicted,(1,2))
    assert out['dense']['mae_usd']==10
    assert out['typical']['mae_usd']==20
    assert out['sparse']['mae_usd']==40
    assert sum(x['n'] for x in out.values())==6


def test_same_day_conflicts_are_flagged_not_silently_removed():
    sample=pd.DataFrame({'parcel_id':['a','a','b','b'], 'sale_date':['2025-01-01']*2+['2025-02-01']*2,'sale_price':[100,110,150,150]})
    audit=audit_same_day_sales(sample)
    assert audit['conflicting_parcel_day_groups']==1
    assert audit['groups'][0]['records']==2
    assert audit['groups'][0]['minimum_price_usd']==100
    assert audit['policy'].startswith('flag_only')


def test_real_data_duplicate_audit_if_uploaded_dataset_present():
    from pathlib import Path
    p=Path('/mnt/data/residential_sales_2013_2025.csv')
    if not p.is_file():pytest.skip('Only inspect actual uploaded county CSV if mounted')
    d=pd.read_csv(p,dtype={'parcel_id':str})
    assert audit_same_day_sales(d)['conflicting_parcel_day_groups']==8


def test_surface_from_fitted_model_is_a_masked_numerical_scenario(official_data):
    model=build_model('xgboost',xgboost_trees=3).fit(official_data.loc[:,list(INPUT_COLUMNS)],official_data.sale_price)
    surface,reference=build_surface(official_data,model,grid_size=28,sigma=1.1,support_km=5)
    assert surface.values.shape==(28,28)
    assert np.isfinite(surface.values).any()
    assert reference['sale_year']==2025
    assert np.isfinite(integrate_price_surface(surface)['mean_surface_value_usd'])


def test_gradient_query_is_consistent_with_analytical_plane():
    xs=np.arange(0,8,dtype=float);ys=np.arange(0,8,dtype=float)
    xx,yy=np.meshgrid(xs,ys)
    values=2*xx+3*yy+100
    surface=PriceSurface(xs,ys,values,np.ones_like(xx)*2,np.ones_like(yy)*3)
    lon,lat=geographic_from_km(np.asarray([3]),np.asarray([3]))
    response=gradient_query(surface,float(lon[0]),float(lat[0]),90,max_distance_km=1)
    assert response['directional_derivative_usd_per_km']==pytest.approx(2)
    assert response['gradient_magnitude_usd_per_km']==pytest.approx(np.sqrt(13))
    offlon,offlat=geographic_from_km(np.asarray([3.4]),np.asarray([3.4]))
    with pytest.raises(ValueError,match='No supported'):
        gradient_query(surface,float(offlon[0]),float(offlat[0]),90,max_distance_km=.01)


def test_support_calibration_never_exposes_confidence_interval(official_data):
    train=official_data.iloc[:60];validation=official_data.iloc[60:85]
    predicted=validation.sale_price.to_numpy(float)+np.arange(len(validation))*300
    report,cut=calibrate_support(train,validation,predicted)
    assert len(cut)==2 and cut[0]<=cut[1]
    assert report['validation_rows']==25
    assert sum(v['n'] for v in report['validation_error_by_support'].values())==25
    assert 'not an individual prediction interval' in report['reliability_caveat']


def test_calibration_artifact_stays_numeric_and_no_pickle(tmp_path,official_data):
    model,name,report=train_evaluate(official_data,Config(forest_trees=3,xgboost_trees=3))
    reference=tmp_path/'support.npz';report_path=tmp_path/'evidence.json'
    evidence=create_support_artifacts(official_data,model,report,output_npz=reference,report_path=report_path)
    assert evidence['held_out_2025_rows']>0
    with np.load(reference,allow_pickle=False) as arrays:
        assert arrays['x_km'].ndim==1
        assert len(arrays['x_km'])==evidence['reference_rows_for_inference']
    sample=official_data.iloc[0]
    result=reliability_for_location(sample.longitude,sample.latitude,reference,evidence)
    assert result['level'] in ('dense','typical','sparse')
    assert result['is_confidence_interval'] is False


def test_complete_official_analysis_exports_true_test_map(tmp_path,official_data):
    model,name,report=train_evaluate(official_data,Config(forest_trees=3,xgboost_trees=3))
    model_path=save_model(model,name,tmp_path/'model.joblib')
    report_path=tmp_path/'training.json';report_path.write_text(json.dumps(report))
    dataset_path=tmp_path/'official.csv';official_data.to_csv(dataset_path,index=False)
    result=build_analysis(dataset_path,model_path,report_path,
                          report_dir=tmp_path/'report',public_dir=tmp_path/'public',
                          artifact_dir=tmp_path/'artifacts',grid_size=28,max_points=25,support_km=5)
    assert result['transactions']==len(official_data)
    assert result['reliability_calibrated_on']>0
    points=json.loads((tmp_path/'public/points.json').read_text())
    assert points['period']=='2025 held-out transactions'
    assert points['source_total']>=5
    assert len(points['latitude'])==len(points['observed'])
    surface=json.loads((tmp_path/'public/surface.json').read_text())
    assert len(surface['x_km'])==28
    assert (tmp_path/'report/price-surface.npz').is_file()
    assert (tmp_path/'artifacts/reliability-support.npz').is_file()
    assert json.loads((tmp_path/'public/reliability-report.json').read_text())['validation_rows']>0


def test_analysis_rejects_mismatched_model_and_training_report(tmp_path,official_data):
    model=build_model('linear').fit(official_data.loc[:,list(INPUT_COLUMNS)],official_data.sale_price)
    saved=save_model(model,'linear',tmp_path/'model.joblib')
    training=tmp_path/'report.json';training.write_text(json.dumps({'model_selected':'xgboost','partition':{'test_count':19}}))
    data=tmp_path/'data.csv';official_data.to_csv(data,index=False)
    with pytest.raises(ValueError,match='differ'):
        build_analysis(data,saved,training,report_dir=tmp_path/'reports',public_dir=tmp_path/'public',artifact_dir=tmp_path/'artifacts')


def test_linear_diagnostic_flags_outliers_without_mutating_data(tmp_path,official_data):
    from geovalue.official.analysis import diagnose_linear_baseline
    path=tmp_path/'source.csv';official_data.to_csv(path,index=False)
    original=path.read_bytes()
    result=diagnose_linear_baseline(path,tmp_path/'diagnose.json',top_n=3)
    assert result['validation_rows']>0
    assert len(result['worst_absolute_errors'])==3
    assert (tmp_path/'diagnose.json').is_file()
    assert path.read_bytes()==original


def test_linear_diagnostic_does_not_allow_unbounded_top_n(tmp_path,official_data):
    from geovalue.official.analysis import diagnose_linear_baseline
    path=tmp_path/'source.csv';official_data.to_csv(path,index=False)
    with pytest.raises(ValueError,match='top_n'):
        diagnose_linear_baseline(path,tmp_path/'diagnose.json',top_n=500)


def test_repeated_spatial_reference_queries_reuse_cached_tree(tmp_path, official_data):
    from geovalue.official.reliability import _cached_reference
    from geovalue.official.spatial import project_km
    coords=official_data.head(40)
    x,y=project_km(coords.longitude.to_numpy(float),coords.latitude.to_numpy(float))
    p=tmp_path/'support.npz';np.savez_compressed(p,x_km=x,y_km=y,boundaries=np.asarray([1.0,3.0]))
    report={'validation_error_by_support':{label:{'n':30,'mae_usd':40000.0,'p80_absolute_error_usd':70000.0} for label in ('dense','typical','sparse')}}
    _cached_reference.cache_clear()
    for _ in range(100):
        result=reliability_for_location(float(coords.iloc[0].longitude),float(coords.iloc[0].latitude),p,report)
        assert result['reference_sample_count']==40
        assert np.isfinite(result['tenth_neighbor_km'])
    assert _cached_reference.cache_info().misses==1
    assert _cached_reference.cache_info().hits>=99
    _cached_reference.cache_clear()
