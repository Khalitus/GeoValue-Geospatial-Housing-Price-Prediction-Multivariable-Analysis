import numpy as np
import pytest

from geovalue.official.training import Config,split_sales,train_evaluate,run_training


def test_chronological_group_disjoint(official_data):
    train,val,test,meta=split_sales(official_data)
    assert train.sale_year.max()<=2023
    assert set(val.sale_year)=={2024}
    assert set(test.sale_year)=={2025}
    assert not set(train.parcel_id)&set(val.parcel_id)
    assert not set(train.parcel_id)&set(test.parcel_id)
    assert not set(val.parcel_id)&set(test.parcel_id)
    assert meta['excluded_for_parcel_overlap']>=1


def test_complete_training_selects_and_never_uses_test_for_selection(official_data):
    _,model,report=train_evaluate(official_data,Config(forest_trees=3,xgboost_trees=5))
    assert model in ('linear','random_forest','xgboost')
    assert len(report['model_metrics'])==3
    assert report['partition']['test_count']>=5
    assert np.isfinite(report['held_out_2025_test']['mae'])
    assert report['selection_rule']=='lowest 2024 validation RMSE'
    assert report['geographical_ablation']['excluded_features']==['latitude','longitude','municipality','zip_code']
    assert 'temporal feature leakage' in report['major_limitations'][0]


def test_training_reports_and_model_are_saved(tmp_path,official_data):
    data_path=tmp_path/'data.csv';official_data.to_csv(data_path,index=False)
    saved=run_training(data_path,tmp_path/'model.joblib',tmp_path/'training.json',Config(forest_trees=3,xgboost_trees=3))
    assert (tmp_path/'model.joblib').is_file()
    assert (tmp_path/'training.json').is_file()
    assert saved['test_rows']>=5


def test_no_2025_sales_prevents_misleading_test(official_data):
    with pytest.raises(ValueError,match='Chronological'):
        split_sales(official_data.loc[official_data.sale_year<2025])


def test_sampling_cannot_hide_missing_test_period(official_data):
    with pytest.raises(ValueError):Config(max_training_rows=10)
