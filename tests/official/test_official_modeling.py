import numpy as np
import pandas as pd
import pytest

from geovalue.official.data import INPUT_COLUMNS, PROHIBITED_INPUTS
from geovalue.official.modeling import (MODELS, PropertyFeatureEngineer,build_model,
                                        load_model,metrics,predict,save_model,validate_inputs)


def test_engineered_property_age_baths_and_ratio(official_data):
    x=official_data.loc[:,list(INPUT_COLUMNS)].head(4)
    z=PropertyFeatureEngineer().fit_transform(x)
    expected=x['sale_year'].to_numpy()-x['year_built'].to_numpy()
    np.testing.assert_allclose(z['property_age_at_sale'],expected)
    np.testing.assert_allclose(z['bathroom_equivalents'],x['full_baths']+x['half_baths']*0.5)
    np.testing.assert_allclose(z['floor_area_ratio'],x['living_area_sqft']/x['lot_area_sqft'])


def test_all_model_families_predict_finite_values(official_data):
    x=official_data.loc[:,list(INPUT_COLUMNS)]
    y=official_data.sale_price
    for name in MODELS:
        model=build_model(name,forest_trees=3,xgboost_trees=3)
        model.fit(x,y)
        values=model.predict(x.iloc[:7])
        assert values.shape==(7,) and np.isfinite(values).all()


def test_no_geo_model_excludes_geo_encoder(official_data):
    model=build_model('linear',geography=False)
    x=official_data.loc[:,list(INPUT_COLUMNS)]
    model.fit(x,official_data.sale_price)
    cols=model.named_steps['preprocess'].transformers_
    assert all('municipality' not in t[2] and 'latitude' not in t[2] for t in cols if t[0] != 'remainder' and isinstance(t[2],list))


def test_saved_model_roundtrip(tmp_path,official_data):
    x=official_data.loc[:,list(INPUT_COLUMNS)]
    model=build_model('xgboost',xgboost_trees=4).fit(x,official_data.sale_price)
    path=save_model(model,'xgboost',tmp_path/'sample.joblib')
    loaded=load_model(path)
    np.testing.assert_allclose(model.predict(x[:5]), loaded['pipeline'].predict(x[:5]))
    assert predict(path,x.head(4)).columns[-1]=='predicted_sale_price_usd'


def test_reject_invalid_coordinate_and_post_sale_construction(official_data):
    x=official_data.loc[:,list(INPUT_COLUMNS)].head(1).copy()
    x['latitude']=0
    with pytest.raises(ValueError,match='location'):validate_inputs(x)
    x=official_data.loc[:,list(INPUT_COLUMNS)].head(1).copy()
    x['year_built']=2025
    x['sale_year']=2024
    with pytest.raises(ValueError,match='Construction'):validate_inputs(x)


def test_metrics_equal_to_hand_calculation():
    result=metrics([100,200,300],np.array([110,180,310]))
    assert result['mae']==pytest.approx(40/3)
    assert result['rmse']==pytest.approx((100+400+100)**0.5/(3**0.5))


def test_one_hot_accepts_unknown_category_at_inference(official_data):
    x=official_data.loc[:,list(INPUT_COLUMNS)]
    model=build_model('xgboost',xgboost_trees=3).fit(x,official_data.sale_price)
    unknown=x.head(1).copy()
    unknown['municipality']='NEVER_SEEN_BEFORE'
    assert np.isfinite(model.predict(unknown)).all()
