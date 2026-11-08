import numpy as np
import pandas as pd
import pytest

from geovalue.data import INPUT_COLUMNS
from geovalue.features import HousingFeatureEngineer


def test_engineered_ratios_are_correct(housing_frame):
    features = HousingFeatureEngineer().fit_transform(housing_frame)
    assert features.iloc[0]["rooms_per_household"] == pytest.approx(880 / 126)
    assert features.iloc[0]["bedrooms_per_room"] == pytest.approx(129 / 880)
    assert features.iloc[0]["population_per_household"] == pytest.approx(322 / 126)
    assert pd.isna(features.iloc[1]["bedrooms_per_room"])
    assert set(INPUT_COLUMNS).issubset(features.columns)
    assert "median_house_value" not in features.columns


def test_zero_room_denominator_does_not_create_infinity(housing_frame):
    housing_frame.loc[0, "total_rooms"] = 0
    features = HousingFeatureEngineer().transform(housing_frame)
    assert np.isnan(features.loc[0, "bedrooms_per_room"])


def test_feature_engineer_rejects_invalid_inputs(housing_frame):
    with pytest.raises(TypeError, match="DataFrame"):
        HousingFeatureEngineer().transform(np.zeros((3, 9)))
    with pytest.raises(ValueError, match="longitude"):
        HousingFeatureEngineer().fit(housing_frame.drop(columns=["longitude"]))
