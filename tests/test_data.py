import pandas as pd
import pytest

from geovalue.data import (
    HousingDataError,
    REQUIRED_COLUMNS,
    load_housing_data,
    summarize_housing,
    validate_housing_data,
)


def test_loads_valid_csv_with_required_schema(housing_csv):
    result = load_housing_data(housing_csv)
    assert len(result) == 4
    assert tuple(result.columns) == REQUIRED_COLUMNS


def test_missing_bedrooms_remain_missing_for_pipeline_imputation(housing_csv):
    result = load_housing_data(housing_csv)
    assert result["total_bedrooms"].isna().sum() == 1


def test_rejects_missing_csv(tmp_path):
    with pytest.raises(HousingDataError, match="Dataset not found"):
        load_housing_data(tmp_path / "missing.csv")


def test_rejects_missing_required_column(housing_frame):
    with pytest.raises(HousingDataError, match="median_income"):
        validate_housing_data(housing_frame.drop(columns="median_income"))


def test_rejects_non_numeric_feature(housing_frame):
    housing_frame["population"] = housing_frame["population"].astype(object)
    housing_frame.loc[1, "population"] = "not-a-number"
    with pytest.raises(HousingDataError, match="Non-numeric values in population"):
        validate_housing_data(housing_frame)


def test_rejects_missing_target(housing_frame):
    housing_frame.loc[1, "median_house_value"] = float("nan")
    with pytest.raises(HousingDataError, match="Missing values in median_house_value"):
        validate_housing_data(housing_frame)


def test_rejects_out_of_range_geography(housing_frame):
    housing_frame.loc[1, "latitude"] = 200
    with pytest.raises(HousingDataError, match="latitude"):
        validate_housing_data(housing_frame)


def test_rejects_zero_households(housing_frame):
    housing_frame.loc[1, "households"] = 0
    with pytest.raises(HousingDataError, match="households"):
        validate_housing_data(housing_frame)


def test_rejects_empty_ocean_proximity(housing_frame):
    housing_frame.loc[1, "ocean_proximity"] = "  "
    with pytest.raises(HousingDataError, match="ocean_proximity"):
        validate_housing_data(housing_frame)


def test_summarize_housing_is_json_safe(housing_frame):
    result = summarize_housing(housing_frame)
    assert result["records"] == 4
    assert result["missing_values"]["total_bedrooms"] == 1
    assert result["ocean_proximity_counts"]["NEAR BAY"] == 2
    assert result["median_house_value"]["minimum"] == 105000
    assert "median_income" in result["correlation_to_median_house_value"]
