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


@pytest.fixture
def synthetic_housing() -> pd.DataFrame:
    """Deterministic fake areas for testing, never used for reporting model accuracy."""
    import numpy as np

    rng = np.random.default_rng(2026)
    n = 180
    longitude = rng.uniform(-124, -114, n)
    latitude = rng.uniform(32, 42, n)
    households = rng.integers(50, 900, n)
    rooms = households * rng.uniform(3, 8, n)
    bedrooms = rooms * rng.uniform(0.15, 0.32, n)
    bedrooms[::13] = np.nan
    population = households * rng.uniform(1.5, 4.0, n)
    income = rng.uniform(1.0, 9.0, n)
    near_ocean = longitude < -120
    value = (
        50_000
        + income * 31_000
        + (longitude + 124) * 2_500
        + (latitude - 32) * 1_000
        + near_ocean * 19_000
        + rng.normal(0, 9_000, n)
    )
    return pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": rng.integers(1, 52, n),
            "total_rooms": rooms,
            "total_bedrooms": bedrooms,
            "population": population,
            "households": households,
            "median_income": income,
            "ocean_proximity": np.where(near_ocean, "NEAR OCEAN", "INLAND"),
            "median_house_value": np.clip(value, 15_000, 500_001),
        }
    )
