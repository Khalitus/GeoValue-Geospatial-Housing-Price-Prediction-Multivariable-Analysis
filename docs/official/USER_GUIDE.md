# GeoValue official-data user guide

This guide covers the local Allegheny County residential-sale application. For the legacy California study, use the older CLI (`geovalue`) and browser route `/legacy`.

## Installation

GeoValue requires Python 3.12 and packages listed in `pyproject.toml`. On macOS, if the project files reside on an external SSD, use an internal-drive virtual environment:

```bash
uv python install 3.12
uv venv --python 3.12 "$HOME/.venvs/geovalue"
source "$HOME/.venvs/geovalue/bin/activate"
uv pip install --link-mode copy -e ".[ml,geo,web,dev]"
```

Do not copy an old `.venv` directory between computers; each machine should install the dependencies locally. Use `python -m uvicorn` to avoid accidentally launching a different system Python.

## Obtain and validate official data

```bash
geovalue-official sources
geovalue-official download
geovalue-official build
geovalue-official profile
```

The download saves official `assessments.csv`, `sales.csv`, `centroids.csv`, and a SHA-256 source manifest under `data/official/raw/`. The builder saves `data/official/processed/residential_sales_2013_2025.csv` and its JSON build report. WPRDC can revise recorded transactions after publication, so keep the retrieval date and hashes from your own download.

## Train and test

```bash
geovalue-official train \
  --data data/official/processed/residential_sales_2013_2025.csv \
  --model-output artifacts/official/residential-sale-model.joblib \
  --report-output reports/official/training-report.json --seed 42
python -m pytest -q
```

The saved model is a trusted fitted pipeline. Linear Regression, Random Forest and XGBoost are compared using 2024 validation RMSE; XGBoost won in the supplied full experiment. The 2025 partition is held out, with repeated parcel IDs excluded across periods.

## Analyze the county scenario and calibrate spatial support

```bash
geovalue-official audit
geovalue-official diagnose
geovalue-official analyze \
  --data data/official/processed/residential_sales_2013_2025.csv \
  --model artifacts/official/residential-sale-model.joblib \
  --training-report reports/official/training-report.json \
  --grid-size 86 --sigma 1.5 --support-km 4.5
```

The analysis saves `.npz` numerical arrays and JSON reports under `reports/official`, a spatial-support reference under `artifacts/official`, and browser JSON files under `public/official/data`. It calibrates the spatial evidence tiers on 2024 residuals using an architecture trained on pre-2024 records. The full-data run may require appreciable memory; do not replace it with artificially generated records.

If you retrain the model, **run `analyze` again and restart the browser server**. The support artifact and displayed report must correspond to the same fitted model and input data.

## Launch the local browser

```bash
python -m uvicorn app:app --reload
```

Navigate to `http://127.0.0.1:8000` on your own computer. The interface includes:

- **Overview:** official 2025 test MAE/RMSE/R², algorithm comparison and geographical ablation.
- **Estimate:** input coordinates, sale year/month, square footage, bedrooms, baths, year built, and optional categorical amenities; the result includes a spatial evidence tier and observed 2024 group errors when prepared.
- **Geography:** interactive 2025 **held-out** sale observations and predictions, absolute errors, controlled price surface and contours. Clicking a point transfers its location to forms; other property inputs remain unchanged.
- **Calculus:** supported coordinate selector, directional bearing, magnitude/steepest ascent, and an integral summary.
- **Method:** scope, evaluation boundaries and model limitations.

The official browser bundles its plotting library locally and does not need an external Plotly CDN. Your local API and files should still be kept private unless you deliberately share them.

## Batch prediction

```bash
geovalue-official predict \
  --model artifacts/official/residential-sale-model.joblib \
  --input data/official/examples/property-inputs.csv \
  --output reports/official/predictions.csv
```

Batch CSVs must provide the 19 official input fields listed in [SCHEMA.md](SCHEMA.md), but the target (`sale_price`) is **not** part of prediction input. `sale_year` must be within the historical training range; this application does not forecast year 2026 or beyond.

## Spatial evidence interpretation

The interface reports the query's distance to its 10th-nearest historical development location (kilometres) and the error observed for its *support category* in 2024. More sparse areas may show higher errors in an empirical audit, but category membership cannot guarantee a particular prediction error. An error percentile from historical groups is **not** a confidence interval.

## Errors and solutions

| Symptom | Meaning | Action |
|---|---|---|
| `/api/official/health` says `model_missing` | `artifacts/official/residential-sale-model.joblib` missing | Run official `train` or restore trusted model |
| Metrics show dashes | Official `training-report.json` not prepared | Run official `train`; check report path |
| Geography/calculus charts unavailable | Official `analyze` outputs absent | Run `geovalue-official analyze` |
| Prediction works but spatial evidence unavailable | Reference `.npz`/JSON not generated | Run `analyze`, restart server |
| API returns HTTP 422 | Invalid property input or unsupported derivative coordinate | Check ranges and use a supported surface location |
| Browser uses Python 3.14 unexpectedly | Terminal Uvicorn path points at system Python | Launch with `python -m uvicorn app:app --reload` |
| Prebuilt pyproj wheel missing on Intel Mac | Incompatible installation selection | Keep configured `pyproj==3.6.1` and Python 3.12 |
| CSV disappeared after replacing project folder | Directory was replaced instead of files merged | Restore local backup, then merge only the update ZIP |

Always keep raw and processed county CSV files, the source manifest, local model, and experimental JSON reports in a **private full backup** (for example, with `python scripts/package_full.py`). They are not part of a regular Git commit.
