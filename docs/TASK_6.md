# Task 6 — Official Property Dataset and Machine Learning Migration

## 1. Scope and release boundary

This task implements a separate **official residential sale-price research pipeline** for Allegheny County, Pennsylvania, USA. It does not overwrite the older California dataset, model or its browser interface. The extension is implemented in `src/geovalue/official/`, exposed as `geovalue-official` and through the optional local API endpoints under `/api/official/`.

**Included:** official-source downloading and provenance; parcel-key joins; county sale validation; residence/feature quality filters; source and prediction-schema audits; regression pipelines; chronological 2024/2025 evaluation; parcel-overlap prevention; an additional held-out-municipality experiment; geographic feature ablation; locally saved model and JSON reports; CLI and API inference; pytest verification.

**Not included until Task 7:** replacing the original UI with the new property form, adapting the 3D gradients/integrals to Pennsylvania coordinates, and implementing/calibrating the novel spatial reliability score.

## 2. Official sources

| File in `data/official/raw` | Source/resource | Primary fields |
|---|---|---|
| `assessments.csv` | [Allegheny County Property Assessments](https://data.wprdc.org/dataset/property-assessments/resource/9a1c60bd-f9f7-4aba-aeb7-af8c3aaa44e5) | `PARID`, `CLASS`, `YEARBLT`, `BEDROOMS`, `FULLBATHS`, `HALFBATHS`, `TOTALROOMS`, `FINISHEDLIVINGAREA`, `LOTAREA`, `STORIES`, `GRADE`, `CONDITION`, `HEATINGCOOLINGDESC`, `MUNIDESC`, `PROPERTYZIP` |
| `sales.csv` | [Allegheny County Property Sales Transactions](https://data.wprdc.org/dataset/real-estate-sales/resource/5bbe6c55-bce6-4edb-9d04-68edeb6bf7b1) | `PARID`, `SALEDATE`, `PRICE`, `SALECODE`, `SALEDESC` |
| `centroids.csv` | [2025 March Parcel Centroids](https://data.wprdc.org/dataset/parcel-centroids-in-allegheny-county-with-geographic-identifiers/resource/3fab7152-3f11-4788-8372-4c33f86ea813) | `PIN`, `LAT`, `LONG` |

The assessments are a **changing parcel snapshot**, NOT a historical sale-date snapshot. The sales dataset includes recorded transfers, not all valid market transactions. The centroid file is **March 2025**, not an annual historical geolocation series. The current assessment resource is large (roughly 415 MiB according to its resource page); the centroid resource is about 106 MiB. The raw source records have been published in much larger quantities than the old 20,640-row California dataset, but **the count of eligible joined sale records must be measured**, not assumed.

All sources are on the county/WPRDC government open-data portal. The published resources are CCZero; review the portal's data-use agreement. The downloader uses CKAN `resource_show` metadata to resolve each resource's download URL and writes an SHA-256 manifest. It **never overwrites existing source files** unless explicitly requested.

## 3. Installation on your existing Mac

Do not rebuild Python or your virtual environment. Keep the original repository and merge the Task 6 update ZIP from the project root:

```bash
source "$HOME/.venvs/geovalue/bin/activate"
unzip -o ~/Downloads/GeoValue-Task6-Update.zip -d .
uv pip install --link-mode copy -e ".[ml,geo,web,dev]"
python -m pytest -q
geovalue-official --help
```

Keep the Python environment on the internal Mac drive as previously configured; the code and CSVs can remain on your external SSD.

## 4. Download or place the official data

Automatic download (may be hundreds of MB per source; requires internet and WPRDC access):

```bash
geovalue-official sources
geovalue-official download --raw-dir data/official/raw
```

This creates `data/official/raw/{assessments,sales,centroids}.csv` and `source-manifest.json`, with source URLs, retrieval date, sizes and file SHA-256 hashes. The assessment and sales portals sometimes require accepting WPRDC terms; an HTTP 403 is an access problem, not a Python error. **If automated download is blocked:** visit the three official links above, accept the portal terms, download the CSVs through the site's Download controls, and rename them exactly `assessments.csv`, `sales.csv`, and `centroids.csv` under `data/official/raw`. Then run `build` normally. The tool does not invent a fake dataset as a fallback.

Retain the original raw CSVs. Their contents and exact retrieval timestamps matter for reproducibility because the county can revise records after publication.

## 5. Construct the final research dataset

```bash
geovalue-official build \
  --raw-dir data/official/raw \
  --output data/official/processed/residential_sales_2013_2025.csv \
  --report reports/official/build-report.json \
  --start-year 2013 --end-year 2025

geovalue-official profile
python -m json.tool reports/official/build-report.json
```

The builder reads the large sales CSV in 100,000-row chunks. Only `SALEDESC` equal to the explicit county description `VALID SALE` is retained. Sales with missing or ambiguous descriptions and `MULTI-PARCEL SALE` are rejected. Only sale dates 2013-01-01 through 2025-12-31 and nominal sale prices from $10,000 to $5,000,000 are retained (adjustable). These are conservative **research filtering choices**, not a claim that all other price levels are intrinsically illegitimate.

`PARID`/`PIN` are normalized **as strings**, preserving leading zeroes in 16-character identifiers. Assessment data is restricted to `CLASS == R` (residential parcels). Joins are explicitly many-to-one: multiple valid sales can belong to one parcel, but assessment and centroid keys are each de-duplicated before joining. Invalid coordinates and nonsensical physical characteristics are rejected. The code excludes properties for which `YEARBLT > sale_year` but **this does not solve time-dependent renovations or changes to current attributes**.

The generated audit report records: source-sales count, rejected sale-code rows, rejected date/price rows, unmatched assessments, unmatched centroids, rejected physical rows, final eligible sale rows, unique parcels and sales-per-year counts. The number of usable observations is *not known until you run this command on the real source files*.

**Never pass an assessment valuation into the model.** `SALEPRICE`, `PRICE`, `FAIRMARKETTOTAL`, `COUNTYTOTAL`, `LOCALTOTAL`, `PREVSALEPRICE`, and administrative sale-validation fields are excluded by a predefined allowlist of predictors.

## 6. Official property prediction schema

The target is **recorded residential sale price (USD)**, not median house value, and not inflation-adjusted. There are 14 base numeric predictors, five categorical predictors, and three engineered predictors.

| Numeric | Description |
|---|---|
| `latitude`, `longitude` | Parcel-centroid coordinates, degrees |
| `bedrooms` | Bedrooms reported in assessment |
| `full_baths`, `half_baths`, `total_rooms` | Room/bathroom characteristics |
| `living_area_sqft`, `lot_area_sqft` | Finished interior floor and parcel area in square feet |
| `year_built`, `stories` | Original construction year, stories |
| `fireplaces`, `garage_spaces` | Recorded amenities |
| `sale_year`, `sale_month` | Transaction context (nominal market conditions by date) |

| Categorical | Description |
|---|---|
| `construction_grade` | Assessment building-construction grade |
| `condition_code` | Assessment condition category, treated categorically |
| `heating_cooling` | Recorded heating and cooling description |
| `municipality` | Assessment municipality |
| `zip_code` | Assessment postal code, stored as a category/string |

Feature engineering: `property_age_at_sale = max(0, sale_year - year_built)`, `bathroom_equivalents = full_baths + 0.5*half_baths`, `floor_area_ratio = living_area_sqft / lot_area_sqft` (undefined ratios stay missing for fitted imputation).

Fitted preprocessing consists of numerical median imputation and scaling, categorical most-frequent imputation and one-hot encoding with safe unknown-category handling. The preprocessing pipeline is fitted **after** the training split. Like the original GeoValue project, XGBoost does not strictly require scaling, but the shared regression pipeline keeps the baseline comparison consistent.

## 7. Chronological, parcel-disjoint evaluation

| Partition | Sale dates | Purpose |
|---|---|---|
| Train | 2013–2023 | Fit candidate models |
| Validation | 2024 | Select model using lowest RMSE; geographical feature ablation |
| Held-out test | 2025 | Final performance evaluation, once after refit |

A parcel with a 2025 sale is removed from the earlier partitions. A parcel with a 2024 sale is removed from the training partition. This intentionally favors the newer partition when enforcing disjoint parcel identifiers, preventing the same parcel from being used as a training and holdout observation. The report records all resulting counts and exclusions. A second, exploratory experiment holds out approximately 15% of distinct municipalities *within the pre-2024 training period*; this evaluates location transfer, **not** prospective 2025 accuracy.

The selected estimator is refitted on 2013–2024 development records and tested on 2025 sales. Models: Linear Regression; 120-tree Random Forest; 220-tree XGBoost (`max_depth=6`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.9`). The report includes train/validation MAE, RMSE, R², the 2025 held-out metrics, and the geographical ablation excluding latitude, longitude, municipality, and ZIP.

**Methodological caveat:** all partitions attach the *current* assessment snapshot to sales from different years. The snapshot can reflect alterations made *after* the sale. Even a chronological, parcel-disjoint split cannot guarantee that input features were observable at the relevant sale date. The correct description is **retrospective reconstruction with time-split evaluation**, not a leakage-free future-price forecasting study. A true point-in-time feature store (historical assessment snapshots) is a recommended subsequent research enhancement.

## 8. Training and prediction commands

Optional small-scale run for debugging (do not use smoke-test results in the university report):

```bash
geovalue-official train \
  --forest-trees 15 --xgboost-trees 30 \
  --max-training-rows 15000 \
  --model-output artifacts/official/quick-model.joblib \
  --report-output reports/official/quick-training.json
```

Full experiment, using all eligible training records by default:

```bash
geovalue-official train \
  --data data/official/processed/residential_sales_2013_2025.csv \
  --model-output artifacts/official/residential-sale-model.joblib \
  --report-output reports/official/training-report.json \
  --seed 42
```

The default full run trains multiple tree ensembles, a geographic ablation, and a municipality holdout. On an Intel Mac this can be computationally intensive. You can use `--max-training-rows 100000` when necessary; **the report records the cap**. Do not quote a result without reporting whether a training cap was used.

Predict on example property records (illustrative inputs, NOT actual county properties):

```bash
geovalue-official predict \
  --model artifacts/official/residential-sale-model.joblib \
  --input data/official/examples/property-inputs.csv \
  --output reports/official/predictions.csv

geovalue-official model-info --model artifacts/official/residential-sale-model.joblib
```

For a local API, run `python -m uvicorn app:app --reload` and access `GET /api/official/health` or `POST /api/official/predict`. The **browser form itself still shows the California version** until Task 7; use the CLI or API for new property inputs in this task.

## 9. Tests and commit

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

The test suite uses synthetic parcel records and synthetic sale histories: correct joins and identifiers, invalid sale-code rejection, 100,000+ row CSV chunking, date/price screening, temporal construction checks, all three fitted regressors, leak-prone-field exclusions, model serialization, parcel-disjoint splits, 2025 chronology and local API validation. Synthetic model scores are not official county price results.

After tests and your actual source downloads/build are verified:

```bash
git add README.md pyproject.toml .gitignore src/geovalue/__init__.py src/geovalue/web.py \
  src/geovalue/official tests/official docs/TASK_6.md docs/official data/official/examples

git commit -m "feat(official): add county sales dataset and chronological ML pipeline"
git push origin main
```

The official raw/processed CSVs, derived reports and saved models remain excluded by `.gitignore`. Existing California records, reports, trained model and browser outputs are not overwritten.

## 10. Verified references

- [WPRDC: official property sales, including validity caveats](https://data.wprdc.org/dataset/real-estate-sales)
- [WPRDC: official assessment schema and snapshot limitations](https://data.wprdc.org/dataset/property-assessments)
- [WPRDC: March 2025 parcel centroid resource](https://data.wprdc.org/dataset/parcel-centroids-in-allegheny-county-with-geographic-identifiers/resource/3fab7152-3f11-4788-8372-4c33f86ea813)
- [WPRDC Property and Housing Guide: Sales](https://wiki.wprdc.org/wiki/Property_and_Housing_Guide%3AProperty_Sales)

Neither the public source-page row count nor synthetic tests justify claiming any specific number of eligible training sales. Use `reports/official/build-report.json` from the actual data run as the authoritative experimental sample size.
