import json

import numpy as np
import pandas as pd
import pytest

from geovalue.cli import main
from geovalue.data import INPUT_COLUMNS
from geovalue.inference import (
    PREDICTION_COLUMN, predict_housing_csv, predict_housing_frame,
    validate_prediction_inputs,
)
from geovalue.modeling import build_model, save_model


@pytest.fixture
def fitted_model(tmp_path, synthetic_housing):
    features = synthetic_housing.loc[:, list(INPUT_COLUMNS)]
    model = build_model("linear").fit(features, synthetic_housing["median_house_value"])
    return save_model(model, tmp_path / "model.joblib", model_name="linear")


def test_prediction_inputs_accept_missing_bedrooms(housing_frame):
    features = housing_frame.drop(columns="median_house_value")
    cleaned = validate_prediction_inputs(features)
    assert cleaned["total_bedrooms"].isna().sum() == 1
    assert list(cleaned.columns) == list(INPUT_COLUMNS)


@pytest.mark.parametrize("column,value", [
    ("households", 0), ("latitude", 91), ("longitude", 181),
    ("median_income", -1), ("total_rooms", float("inf")),
    ("ocean_proximity", "MARS"),
])
def test_prediction_inputs_reject_invalid_data(housing_frame, column, value):
    features = housing_frame.drop(columns="median_house_value")
    if column == "total_rooms":
        features[column] = features[column].astype(float)
    features.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_prediction_inputs(features)


def test_prediction_rejects_missing_columns(housing_frame):
    with pytest.raises(ValueError, match="Missing prediction columns"):
        validate_prediction_inputs(housing_frame.drop(columns=["latitude", "median_house_value"]))


def test_single_and_batch_predictions(fitted_model, housing_frame, tmp_path):
    features = housing_frame.drop(columns="median_house_value")
    result = predict_housing_frame(fitted_model, features)
    assert len(result) == len(features)
    assert np.isfinite(result[PREDICTION_COLUMN]).all()
    csv_in, csv_out = tmp_path / "input.csv", tmp_path / "output.csv"
    features.to_csv(csv_in, index=False)
    assert predict_housing_csv(fitted_model, csv_in, csv_out) == csv_out
    stored = pd.read_csv(csv_out)
    np.testing.assert_allclose(stored[PREDICTION_COLUMN], result[PREDICTION_COLUMN])


def test_cli_single_prediction(fitted_model, housing_frame, capsys):
    row = housing_frame.iloc[0]
    args = ["predict", "--model", str(fitted_model)]
    for field in INPUT_COLUMNS:
        if field == "total_bedrooms":
            continue  # optional; fitted imputer will use training median
        args += ["--" + field.replace("_", "-"), str(row[field])]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert np.isfinite(payload[PREDICTION_COLUMN])
    assert payload["units"] == "historical_1990_USD"


def test_cli_batch_prediction_and_invalid_flags(fitted_model, housing_frame, tmp_path, capsys):
    source = tmp_path / "input.csv"
    destination = tmp_path / "predictions.csv"
    housing_frame.drop(columns="median_house_value").to_csv(source, index=False)
    assert main(["predict", "--model", str(fitted_model), "--input", str(source), "--output", str(destination)]) == 0
    assert destination.is_file()
    assert PREDICTION_COLUMN in pd.read_csv(destination).columns
    assert main(["predict", "--model", str(fitted_model), "--input", str(source), "--longitude", "-120"]) == 2
    assert "either --input CSV" in capsys.readouterr().err
