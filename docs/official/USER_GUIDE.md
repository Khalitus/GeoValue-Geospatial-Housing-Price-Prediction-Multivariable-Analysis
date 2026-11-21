# Local application and command-line reference

GeoValue is a Python 3.12 research application with a command-line interface and a FastAPI browser. The official Pennsylvania study is served at `/`; the retained California census-area study is available at `/legacy`.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,geo,web,dev]'
python -m pytest -q
```

An activated virtual environment is required for consistent Python dependencies. Running Uvicorn as `python -m uvicorn` avoids using a different system Python. The browser uses locally bundled Plotly and does not require an external chart CDN.

## Data and model pipeline

```bash
geovalue-official sources
geovalue-official download
geovalue-official build
geovalue-official profile
geovalue-official train
geovalue-official analyze
```

The download retrieves county assessments, recorded sales and parcel centroids and stores a source manifest under `data/official/raw/`. The builder writes the processed CSV to `data/official/processed/residential_sales_2013_2025.csv`. The trainer compares the configured regression architectures and saves its selected pipeline to `artifacts/official/residential-sale-model.joblib`. The analyzer writes numerical and JSON reports under `reports/official/` and browser assets under `public/official/data/`.

The trained pipeline can be used directly; repeated training is not required for every browser session. The spatial analyzer uses the saved model but separately fits a pre-2024 calibration model for the validation-only spatial-support summaries. The 2025 support audit uses the final model.

## Browser

```bash
python -m uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> for the official study. The application provides an overview of model evaluation, a property estimate form, observed-sale geography, numerical calculus views, and a methodology summary.

The estimate form takes geographical coordinates, a historical transaction year and month, structural attributes and optional categorical amenities. Its output contains a recorded-sale-price estimate and a nearest-neighbor support category where the reliability artifact exists. The support summary is historical group evidence, not a guarantee about the individual estimate.

The calculus view evaluates directional derivatives of the supported, smoothed geographical scenario. Results are in nominal USD/km and describe mathematical sensitivity, not appreciation of an actual property.

## Batch inference and diagnostics

```bash
geovalue-official predict \
  --model artifacts/official/residential-sale-model.joblib \
  --input data/official/examples/property-inputs.csv \
  --output reports/official/predictions.csv

geovalue-official audit
geovalue-official diagnose
geovalue-official model-info
```

Batch inputs contain the 19 original predictor fields defined in [SCHEMA.md](SCHEMA.md). They do not contain the sale-price target or derived model features. The audit flags contradictory same-day sale records, while the diagnostic reports extreme Linear Regression validation predictions without modifying the data.

## Expected local artifacts

| Path | Purpose |
|---|---|
| `data/official/raw/source-manifest.json` | Official source IDs, retrieval timestamps and hashes |
| `reports/official/build-report.json` | Filtering and join counts |
| `reports/official/training-report.json` | Model selection and test metrics |
| `artifacts/official/residential-sale-model.joblib` | Saved fitted model and preprocessing pipeline |
| `reports/official/reliability-report.json` | Spatial-support calibration and 2025 audit |
| `reports/official/geospatial-report.json` | Supported geographical surface and integral summary |
| `reports/official/price-surface.npz` | Numerical grid, derivatives and mask |

Raw datasets, model artifacts and local analytical reports are normally excluded from Git. Joblib artifacts should only be loaded from a trusted source because deserialization can execute code. See [reproducibility](REPRODUCIBILITY.md), [architecture](ARCHITECTURE.md) and [methodology](METHODOLOGY.md).
