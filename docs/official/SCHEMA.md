# Official residential-sale model: data contract

This describes the `geovalue-official` research pipeline introduced in Task 6. It is separate from the historical California census-area model. The model is trained on an official county **recorded transaction price**, not a census median.

## Raw sources

See [Task 6](../TASK_6.md) for official datasets and resource identifiers. Only three raw CSVs are needed: `assessments.csv`, `sales.csv`, `centroids.csv`. County parcel identifiers are strings, not integers; `PARID` and centroid `PIN` are normalized and joined. Keep originals unmodified.

## Processed output

`data/official/processed/residential_sales_2013_2025.csv` contains the following columns:

- Metadata (excluded from ML): `parcel_id`, `sale_date`.
- Numerical ML inputs: `latitude`, `longitude`, `bedrooms`, `full_baths`, `half_baths`, `total_rooms`, `living_area_sqft`, `lot_area_sqft`, `year_built`, `stories`, `fireplaces`, `garage_spaces`, `sale_year`, `sale_month`.
- Categorical ML inputs: `construction_grade`, `condition_code`, `heating_cooling`, `municipality`, `zip_code`.
- Regression target (excluded from ML inputs): `sale_price`.

The fitted feature transformer adds: `property_age_at_sale`, `bathroom_equivalents`, `floor_area_ratio`. Do not input these derived features manually.

## Data-quality contract

1. **County validity:** only transfers explicitly described as `VALID SALE` are included. `MULTI-PARCEL SALE` and ambiguous/missing descriptions are excluded.
2. **Residential:** assessment `CLASS` must be `R` and have plausible physical characteristics. `YEARBLT` cannot exceed the sale year.
3. **Range:** year 2013–2025 inclusive, sale price $10,000–$5 million by default, area 250–30,000 square feet, lot area >0, bedrooms 0–20, and coordinates within broad Allegheny County bounds.
4. **Join:** each sale has one validated property assessment and one parcel centroid; assessment and centroid records are unique on normalized key.
5. **Target separation:** never use county appraisal values (`FAIRMARKET*`, `COUNTY*`, `LOCAL*`), current/prior transaction prices, sale validation description, sale ID, or record date as ML inputs.
6. **Record provenance:** SHA-256 hashes, retrieval dates and source URLs are written to `source-manifest.json` if downloaded automatically; current assessments can differ from sale-date characteristics.

## Prediction input schema

Prediction CSVs require the 19 original (14 numerical + 5 categorical) property predictors. The model checks required numeric inputs, coordinate bounds, construction date relative to sale year, positive square footage, and finite values. Optional amenity counts can be missing; fitted imputation handles them. The reported prediction is **nominal historical USD for the supplied sale year**; it is not a current appraisal.

See `data/official/examples/property-inputs.csv` for illustrative input shape and `src/geovalue/official/data.py` for the source of truth.
