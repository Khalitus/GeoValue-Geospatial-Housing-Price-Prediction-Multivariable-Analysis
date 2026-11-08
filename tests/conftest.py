import pandas as pd
import pytest


@pytest.fixture
def housing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [-122.23, -118.31, -121.89, -119.58],
            "latitude": [37.88, 34.08, 37.29, 36.72],
            "housing_median_age": [41, 21, 35, 18],
            "total_rooms": [880, 600, 710, 1200],
            "total_bedrooms": [129, None, 136, 270],
            "population": [322, 550, 390, 701],
            "households": [126, 235, 146, 281],
            "median_income": [8.3252, 3.4, 5.8, 2.2],
            "ocean_proximity": ["NEAR BAY", "<1H OCEAN", "NEAR BAY", "INLAND"],
            "median_house_value": [452600, 245000, 310000, 105000],
        }
    )


@pytest.fixture
def housing_csv(tmp_path, housing_frame):
    path = tmp_path / "housing.csv"
    housing_frame.to_csv(path, index=False)
    return path
