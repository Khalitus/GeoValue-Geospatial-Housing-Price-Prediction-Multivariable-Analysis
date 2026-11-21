# Data schema and eligibility rules

## Sources and joins

GeoValue uses three official Allegheny County resources from the Western Pennsylvania Regional Data Center (WPRDC): real-estate sales, property assessments and parcel centroids. Their resource identifiers and source pages are recorded in [REFERENCES.md](REFERENCES.md). The local downloader writes `assessments.csv`, `sales.csv`, `centroids.csv`, and a source-manifest JSON containing retrieval timestamps and SHA-256 hashes.

Sale and assessment records use `PARID`; centroid records use `PIN`. Parcel identifiers are normalized as strings to preserve leading zeroes. A valid sale is matched to a single residential assessment and a parcel-centroid coordinate. Source records can change after the retrieval date; the published results refer to the specific hashed source extract.

## Eligibility and quality control

| Rule | Implementation |
|---|---|
| Recorded sale description | `SALEDESC = VALID SALE` |
| Sale date | 2013-01-01 to 2025-12-31 inclusive |
| Transaction price | $10,000–$5,000,000 |
| Residential class | Assessment `CLASS = R` |
| Finished area | 250–30,000 sq. ft. |
| Lot area | Greater than zero |
| Bedrooms | 0–20 |
| Construction year | 1700–2025 and no later than sale year |
| Coordinates | Required; plausibility-checked for the study region |

Exact duplicate sale records are removed before joins; contradictory same-parcel/same-day prices are **flagged** in a separate audit rather than silently adjudicated. Historical assessment features may be inaccurate for earlier sale dates because the assessment source is a later snapshot.

## Processed dataset

`data/official/processed/residential_sales_2013_2025.csv` contains **22 columns**: two nonpredictive identifiers, nineteen original predictors and one target. The model's fitted feature transformer creates three further variables before encoding.

**Identifiers (not predictors):** `parcel_id`, `sale_date`.

**Numerical predictors (14):** `latitude`, `longitude`, `bedrooms`, `full_baths`, `half_baths`, `total_rooms`, `living_area_sqft`, `lot_area_sqft`, `year_built`, `stories`, `fireplaces`, `garage_spaces`, `sale_year`, `sale_month`.

**Categorical predictors (5):** `construction_grade`, `condition_code`, `heating_cooling`, `municipality`, `zip_code`.

**Target:** `sale_price`, nominal recorded USD for an eligible transaction.

**Engineered predictors:** `property_age_at_sale`, `bathroom_equivalents`, `floor_area_ratio`.

The engineered fields are calculated inside the pipeline, not supplied in prediction requests. Numerical features are imputed by training-set medians and standardized. Categorical features use most-frequent imputation and frequency-aware one-hot encoding. Unseen categories are tolerated by the fitted encoder.

The target, county appraisal and tax valuation fields, administrative sale validation labels and parcel IDs do not enter the feature matrix. See the canonical column definitions in [`src/geovalue/official/data.py`](../../src/geovalue/official/data.py) and the model transformation in [`modeling.py`](../../src/geovalue/official/modeling.py).

## Prediction contract

Batch inference accepts a CSV containing the 19 original predictor fields; see [`data/official/examples/property-inputs.csv`](../../data/official/examples/property-inputs.csv). Input validation rejects missing required numerical values, impossible geography, invalid area or dates and nonfinite numeric inputs as specified by the implementation. Model output is a historical recorded-sale-price estimate, not an appraisal of a present property or a forecast for years outside the supported transaction period.

Temporal provenance for all predictor groups is documented in [FEATURE_VALIDITY.md](FEATURE_VALIDITY.md).
