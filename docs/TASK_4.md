# Task 4 — Web dashboard and Vercel packaging

## Verified starting point

This project continues the Task 3 CLI, XGBoost model, geographical maps and numerical calculus. The Python runtime dependencies are pinned to the exact major ML package versions recorded in the user-provided Task 2 training report (scikit-learn 1.9.1, XGBoost 3.4.1, pandas 2.3.3, NumPy 2.5.4 and Joblib 1.6.0). The browser interface is a thin layer over the same trusted scikit-learn/XGBoost pipeline; it does not train models in response to requests.

## Files

- `app.py` — Vercel-recognized FastAPI entrypoint.
- `src/geovalue/web.py` — prediction, metrics and geographical-gradient API.
- `public/index.html`, `public/styles.css`, `public/app.js` — responsive dashboard, prediction form, model comparison charts, map, surface, contours and gradient visuals.
- `scripts/prepare_deployment.py` — locally exports the original trusted model and numerical grid to deployable static files and sampled map points; it NEVER changes the source dataset, model or reports.
- `scripts/package_full.py` — locally creates a ZIP INCLUDING ALL currently present CSV, JSON, NPZ and Joblib files, with only caches, Git internals and virtual environments excluded.
- `tests/test_web.py` and `tests/test_deployment_assets.py` — API, validation and packaging checks.
- `.github/workflows/tests.yml` — corrects GitHub Actions to install ML and geospatial dependencies.

## Prepare on your Mac (from the existing project directory)

Keep the existing project directory and merge the **Update ZIP** rather than replacing the folder:

```sh
unzip -o ~/Downloads/GeoValue-Task4-Update.zip -d .
source "$HOME/.venvs/geovalue/bin/activate"
uv pip install --link-mode copy -e ".[ml,geo,web,dev]"
python -m pytest -q
```

Verify original generated files:

```sh
ls -lh data/raw/housing.csv
ls -lh reports/training-report.json
ls -lh reports/geospatial/geospatial-report.json
ls -lh reports/geospatial/price-surface.npz
ls -lh artifacts/housing-model.joblib
```

If `price-surface.npz` is absent, run `geovalue analyze ...` as in Task 3.

```sh
python scripts/prepare_deployment.py
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000`. The prediction API uses the trusted local model file and the dashboard uses exported JSON samples of the full dataset and projected surface.

## A truly FULL ZIP containing your private local CSV, JSON and model

No assistant can access ignored files stored only on your Mac. The distributed full source ZIP includes the uploaded actual training JSON, an explicitly labeled example input CSV, and the geospatial report values shared in the conversation. It DOES NOT claim to contain your unretrieved 20,640-row `data/raw/housing.csv` or original `artifacts/housing-model.joblib`.

To make a genuinely complete snapshot from your own current files, **without deleting anything**, run:

```sh
python scripts/package_full.py
```

This creates `../GeoValue-Task4-Full-With-Local-Data.zip`, including the original CSV, all generated JSON files, the trained joblib file, NPZ surface, and generated HTML visualizations if they exist. This is the ZIP to retain as your full backup.

## Deploy to Vercel (Git integration)

Vercel detects `app.py` and the FastAPI dependency in `pyproject.toml`. Our interface is in `public/` and the API is handled by Vercel's Python runtime. The Vercel Python runtime supports Python 3.12 and typically allows 500 MB uncompressed; large Python functions beta can be enabled for eligible projects if the bundle exceeds that size.

Before deploying, run `python scripts/prepare_deployment.py`. The following files are normally ignored to protect against accidental commits and need to be explicitly staged **only if you are comfortable publishing this historical model and sampled data**:

```sh
git add README.md pyproject.toml .gitignore .vercelignore vercel.json app.py public/index.html public/styles.css public/app.js scripts docs src/geovalue/web.py src/geovalue/geospatial.py src/geovalue/__init__.py .github/workflows/tests.yml tests

git add -f deploy/housing-model.joblib deploy/surface.json deploy/training-report.json deploy/geospatial-report.json public/data/training-report.json public/data/geospatial-report.json public/data/points.json public/data/surface.json

git commit -m "feat(web): deploy GeoValue prediction and calculus dashboard"
git push origin main
```

Do NOT commit `data/raw/housing.csv`, your local `artifacts/` directory, credential files, or virtual environments. The public app uses an exported sample of points rather than redistributing every original record.

Import your GitHub repo into Vercel from https://vercel.com/new, choose root project directory, keep the detected FastAPI preset, and deploy. Or use the CLI: `npm install -g vercel`, `vercel`, then `vercel --prod` once preview is verified. If the build hits the standard Python bundle limit, enable Large Functions (Fluid compute) for the Vercel project and redeploy.

Vercel's deploy is NOT complete until you confirm a ready URL and test `/api/health` (model_available true), `/api/report`, `/api/predict` and map/surface charts. Our local pytest does not prove a Vercel production deployment succeeded.

## Security and scientific limitations

Joblib uses pickle internally: load only a trusted locally created model; never accept user-uploaded model files. The public API accepts only numeric/known categorical features. Surface derivatives are computed from the smoothed fixed-feature scenario, not raw tree predictions. The distance to the nearest gradient point in the web API uses an approximate geographical distance for convenience, not an official administrative boundary.

### If the original housing CSV was already deleted

If you cannot recover your original local Kaggle CSV, you can restore the publicly archived matching California housing dataset (20,640 rows) without using the Kaggle CLI:

```sh
python scripts/restore_dataset.py
```

This command refuses to overwrite an existing `data/raw/housing.csv`. The public reference is https://github.com/ageron/data/blob/main/housing/housing.csv. Confirm your model and reports remain aligned with the restored dataset if you retrain.
