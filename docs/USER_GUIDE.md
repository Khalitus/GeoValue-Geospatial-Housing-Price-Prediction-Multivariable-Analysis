> **Legacy California baseline documentation.** This file describes the original 1990 census-area project. The primary GeoValue release now uses official Allegheny County transaction records through 2025; see [official project documentation](official/PROJECT_REPORT.md) and [current methodology](official/METHODOLOGY.md).

# User guide

## Audience and operating mode

GeoValue is a **local research application** with a CLI and a FastAPI browser dashboard. It does not require an account, hosting or a Vercel deployment. Python runs on your machine; the browser opens at `127.0.0.1:8000`. Some chart components still fetch Plotly from an external CDN and thus require internet access to render fully.

## 1. Install

Use Python 3.12 and run from the repository root. The following is the tested approach for a Mac with project files on an external SSD:

```bash
uv python install 3.12
uv venv --python 3.12 "$HOME/.venvs/geovalue"
source "$HOME/.venvs/geovalue/bin/activate"
uv pip install --link-mode copy -e ".[ml,geo,web,dev]"
```

For an existing internal-drive environment, **do not recreate it**. Simply activate `$HOME/.venvs/geovalue/bin/activate`. On another OS, use a normal Python 3.12 venv and install the same package extras using pip. Do not rely on a globally installed `uvicorn`; launch via `python -m uvicorn`.

Check:

```bash
python --version
python -m pytest -q
python -m geovalue --help
```

## 2. Data and saved artifacts

The original Kaggle CSV belongs in `data/raw/housing.csv` (20,640 rows). A prepared copy of model and browser data may already exist in `deploy/` and `public/data/`; if it does, the browser may work without the raw CSV. To retrain or rerun the full mathematical analysis, you need the original CSV and an appropriate trusted saved model.

The `data/examples/` files are **small illustrations**, not the full research dataset. `scripts/restore_dataset.py` can retrieve the matching public archived housing data when the CSV is absent, but never overwrites an existing CSV.

## 3. Run from the CLI

### Inspect dataset

```bash
geovalue status
geovalue inspect --data data/raw/housing.csv --output reports/dataset-summary.json
```

### Train and select a model

```bash
geovalue train \
  --data data/raw/housing.csv \
  --model-output artifacts/housing-model.joblib \
  --report-output reports/training-report.json \
  --seed 42
```

The report gives three models' train/validation scores and final held-out test score. All dollar-denominated outputs are historical 1990 values. Training time depends on the computer; re-running is not necessary for browsing a compatible already saved model.

### Predict one area

```bash
geovalue predict \
  --model artifacts/housing-model.joblib \
  --longitude -122.23 --latitude 37.88 \
  --housing-median-age 41 --total-rooms 880 \
  --total-bedrooms 129 --population 322 \
  --households 126 --median-income 8.3252 \
  --ocean-proximity "NEAR BAY"
```

### Predict a CSV batch

```bash
geovalue predict \
  --model artifacts/housing-model.joblib \
  --input data/examples/prediction-input.csv \
  --output reports/predictions.csv
```

Batch input must contain the nine features but not the target; `total_bedrooms` may be missing. Inspect `data/examples/prediction-input.csv` for its exact format.

### Generate geographic and calculus outputs

```bash
geovalue analyze \
  --data data/raw/housing.csv \
  --model artifacts/housing-model.joblib \
  --output-dir reports/geospatial \
  --grid-size 90 --sigma 1.6 --support-km 45
```

Expected outputs are `housing-map.html`, `price-surface-3d.html`, `price-contours.html`, `price-gradients.html`, `price-surface.npz`, and `geospatial-report.json` under `reports/geospatial/`. The HTML plots can be opened in a browser. Maps may use a third-party tile provider; the numerical results themselves do not require that map tile service.

### Query a directional gradient

```bash
geovalue gradient \
  --surface reports/geospatial/price-surface.npz \
  --longitude -122.23 --latitude 37.88 --bearing 90
```

The query selects the nearest supported numerical sample within the configured distance threshold. Some coordinates will be rejected because the surface is not supported there; use a supported preset in the dashboard instead. Bearing is clockwise from the **projected grid north** reference.

## 4. Use the browser interface

Start the local server with the active Python interpreter:

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` and use the five sections: **Overview** (evaluation), **Prediction** (single-area input), **Geography** (observed/predicted/error point layers), **Calculus** (surface, contours and gradient query), and **Method** (assumptions). On Geography, selecting a point can transfer coordinates to the prediction form; on Calculus, use a supported-location preset if an arbitrary coordinate is rejected.

If your original CSV/model/surface exist but the browser's JSON is missing, rebuild the browser assets:

```bash
python scripts/prepare_deployment.py
```

Despite its name, this script prepares **local** browser inputs; it does not deploy, upload or retrain anything. It preserves your source CSV and model. `/api/health` helps identify missing asset files.

## 5. Interpret responsibly

A predicted value represents a **1990 census-area median estimate**. A larger modeled gradient means that the smoothed fixed-feature scenario changes more rapidly with location, in historical USD/km; it is not a mortgage rate, future investment return or verified neighborhood effect. The numerical integration is over the grid-supported region, not California's official area.

## 6. Troubleshooting

| Symptom | Likely cause / resolution |
|---|---|
| `No module named geovalue` when starting server | Using system Python; activate the project venv and run `python -m uvicorn app:app --reload` |
| `Model not found` | Confirm `artifacts/housing-model.joblib` or prepared `deploy/housing-model.joblib`; retrain if neither exists |
| Dashboard returns 503 for reports | Re-run `geovalue train`/`geovalue analyze`, then `python scripts/prepare_deployment.py` |
| Gradient returns 422 | Coordinate is outside a supported grid neighborhood; select a supported preset |
| Charts appear blank but server returns 200 | Check access to `cdn.plot.ly`; the frontend is not fully offline yet |
| `pyproj` tries to compile from source on an Intel Mac | Keep the pinned `pyproj==3.6.1` in `pyproject.toml` and use a clean compatible wheel |
| `._*` metadata or wheel-copy errors on external SSD | Keep the venv on the internal drive and install with `uv pip --link-mode copy` |
| Folium warns about CartoDB tiles | Background tiles may require an API key; numerical output is unaffected |
| A Starlette/httpx deprecation warning appears | Nonfatal; track with upstream packages rather than suppressing tests |

## 7. Protect the project files

Run `python scripts/package_full.py` to create a local backup ZIP including your existing CSV, JSON, NPZ, HTML and Joblib files (if present). This archive may contain large or copyrighted dataset files and **should not be committed to the public repository**. Commit this documentation package separately. Use `git status` to confirm only intended files are staged.
