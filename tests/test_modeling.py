import joblib
import numpy as np
import pandas as pd
import pytest

from geovalue.data import INPUT_COLUMNS
from geovalue.modeling import (
    build_model,
    load_model,
    predict_records,
    regression_metrics,
    save_model,
)


def test_build_model_names_are_validated():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("neural_net")
    with pytest.raises(ValueError, match="positive"):
        build_model("xgboost", xgboost_trees=0)


def test_metrics_have_expected_units():
    scores = regression_metrics(np.array([100, 200]), np.array([110, 190]))
    assert scores["mae"] == pytest.approx(10)
    assert scores["rmse"] == pytest.approx(10)
    assert scores["r2"] == pytest.approx(0.96)
    with pytest.raises(ValueError, match="equal-length"):
        regression_metrics([1, 2], [1])


def test_pipeline_handles_missing_bedrooms_and_unknown_ocean_category(synthetic_housing):
    data = synthetic_housing.iloc[:110]
    x = data.loc[:, list(INPUT_COLUMNS)]
    y = data["median_house_value"]
    pipeline = build_model("linear")
    pipeline.fit(x, y)
    candidate = synthetic_housing.iloc[[120]].loc[:, list(INPUT_COLUMNS)].copy()
    candidate["ocean_proximity"] = "ISLAND"
    candidate["total_bedrooms"] = np.nan
    prediction = pipeline.predict(candidate)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_imputer_is_fitted_on_training_data_only(housing_frame):
    sample = housing_frame.copy()
    sample.loc[:, "total_bedrooms"] = [100.0, np.nan, 300.0, 1000.0]
    model = build_model("linear").fit(sample.loc[:, list(INPUT_COLUMNS)], sample["median_house_value"])
    numeric_imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    # Numeric transformer begins with columns from NUMERIC_INPUT_COLUMNS.
    assert numeric_imputer.statistics_[4] == pytest.approx(300.0)
    validation = sample.iloc[[0]].copy()
    validation["total_bedrooms"] = 9_000_000.0
    model.predict(validation.loc[:, list(INPUT_COLUMNS)])
    assert numeric_imputer.statistics_[4] == pytest.approx(300.0)


def test_geography_ablation_actually_drops_location(synthetic_housing):
    x = synthetic_housing.loc[:, list(INPUT_COLUMNS)]
    y = synthetic_housing["median_house_value"]
    model = build_model("linear", include_geography=False).fit(x, y)
    columns = model.named_steps["preprocess"].get_feature_names_out()
    assert "longitude" not in columns
    assert "latitude" not in columns
    assert not any("ocean_proximity" in item for item in columns)


def test_trusted_model_roundtrip(synthetic_housing, tmp_path):
    frame = synthetic_housing.iloc[:100]
    model = build_model("linear").fit(frame.loc[:, list(INPUT_COLUMNS)], frame["median_house_value"])
    target = tmp_path / "models" / "housing.joblib"
    save_model(model, target, model_name="linear")
    metadata = load_model(target)
    assert metadata["model_name"] == "linear"
    preds_before = model.predict(synthetic_housing.iloc[100:103].loc[:, list(INPUT_COLUMNS)])
    preds_after = predict_records(target, synthetic_housing.iloc[100:103].loc[:, list(INPUT_COLUMNS)])
    np.testing.assert_allclose(preds_before, preds_after)


def test_bad_model_artifact_is_rejected(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"version": 999}, path)
    with pytest.raises(ValueError, match="malformed"):
        load_model(path)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.joblib")
