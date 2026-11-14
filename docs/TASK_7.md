# Task 7 — Spatial Evidence Indicator, Pennsylvania calculus and local GUI

## Status and release scope

This task builds on the [official dataset and model pipeline](TASK_6.md), not the former California census-group dataset. The original `geovalue` CLI, baseline model and California browser remain available under the `geovalue` command and `/legacy`. The primary local browser route (`/`) now uses the **official Allegheny County residential sale** project.

The released code includes numerical calculus, 2024-calibrated spatial evidence, 2025 evidence-group audit, browser charts and interaction, data-quality diagnostics, and pytest integration coverage. The actual model and source files remain on the developer's Mac and are **not** embedded in the code update ZIP.

## 1. Install the merge-only update

From the existing GeoValue repository root, first run the existing full backup utility:

```bash
source "$HOME/.venvs/geovalue/bin/activate"
python scripts/package_full.py
unzip -o ~/Downloads/GeoValue-Task7-Update.zip -d .
uv pip install --link-mode copy -e ".[ml,geo,web,dev]"
python -m pytest -q
```

A ZIP filename in the command above is an example; use the actual downloaded archive filename if different. `unzip -o` merges only the code and documentation files in the update. It does **not** delete original CSV, JSON or Joblib data. Do **not** replace your whole repository directory or unzip a full-source snapshot over your existing local data.

## 2. Verify the official inputs and model

```bash
geovalue-official profile
geovalue-official model-info --model artifacts/official/residential-sale-model.joblib
python -m json.tool reports/official/training-report.json
```

The verified build report contains 82,169 eligible records/72,607 parcels. For training-report metrics see [EXPERIMENT_RESULTS.md](official/EXPERIMENT_RESULTS.md). `geovalue-official analyze` will stop if the report's selected model name or 2025 test-row count conflicts with the provided saved model/CSV.

## 3. Generate the new calculus and Spatial Evidence Indicator

```bash
geovalue-official audit
geovalue-official diagnose
geovalue-official analyze \
  --data data/official/processed/residential_sales_2013_2025.csv \
  --model artifacts/official/residential-sale-model.joblib \
  --training-report reports/official/training-report.json \
  --grid-size 86 --sigma 1.5 --support-km 4.5 \
  --map-points 1600
```

The `analyze` command generates the following, using the fitted official model and the actual county dataset:

| Path | Content |
|---|---|
| `reports/official/geospatial-report.json` | Surface summary, support mask, area quadrature, limits |
| `reports/official/price-surface.npz` | Numeric x/y grid, surface values and partial derivatives |
| `reports/official/reliability-report.json` | 2024 validation-group errors, 2025 holdout audit, thresholds |
| `reports/official/transaction-audit.json` | Same-parcel/same-day price conflicts; flagged only |
| `reports/official/linear-diagnostics.json` | Optional 2024 Linear Regression outlier diagnosis (`diagnose`) |
| `artifacts/official/reliability-support.npz` | Coordinates of training + validation reference locations and support thresholds |
| `public/official/data/*.json` | Sampled actual 2025 held-out points, surface and reports for the local UI |

No generated outputs are fabricated by the browser. If the model or precomputed files are missing, the corresponding API returns an explicit error and the interface explains the missing step.

### Spatial Evidence Indicator definition

1. Project training and query locations into **EPSG:26917** (UTM zone 17N) in kilometres.
2. From 2013–2023 training locations, calculate each 2024 validation property's **distance to its 10th closest training location**, denoted `d10`.
3. Use the **33rd and 67th percentiles of 2024 d10** to identify dense, typical and sparse geographical support tiers.
4. Compute observed 2024 residual MAE, RMSE and the 80th percentile of absolute errors within each tier. Do **not** use 2025 records to select thresholds or calibrate error groups.
5. For an interactive user prediction, calculate `d10` against the final model's **2013–2024 development locations** and display the 2024 validation-group error alongside the tier. The support base has grown, so the calibration is approximate rather than an exchangeable confidence statement.
6. After the thresholds have been fixed, publish the 2025 error audit stratified by the same thresholds. This tests whether sparsely represented locations show worse errors, but **does not** turn the indicator into a confidence interval.

A location density tier alone is not an uncertainty guarantee. Its relationship to actual errors is **empirical**, geography-specific, and may change if the data or model changes. The UI labels it “Spatial evidence” accordingly.

### Calculus method

- Project longitude/latitude into NAD83 / UTM zone 17N (**EPSG:26917**, metres; divide by 1,000 for km).
- Create a 2D grid (default **86 × 86**), retain only nodes within **4.5 km** of at least one observed location, and hold non-location inputs constant at empirical median/mode for a controlled **2025** scenario.
- Gaussian-filter the supported model predictions using `sigma=1.5` grid cells. The mask is re-applied after weighted smoothing.
- Differentiate numerically in projected x/y kilometres using finite differences; return `dS/dx`, `dS/dy`, magnitude, steepest-ascent grid bearing and bearing-specific directional derivative. Do not differentiate raw XGBoost trees.
- Integrate only grid cells whose **four** corner values are finite; report the area-integrated `USD·km²` and area-weighted mean `USD`. The domain is a support mask, **not** a true county area boundary.

A scenario fixes ZIP and municipality at their most frequent categories *even as locations vary*. These combinations need not describe actual local properties. The surface is for MVC analysis, not individual valuations or causal claims.

## 4. Run the local app

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. The top navigation includes **Overview, Estimate, Geography, Calculus, Method**. The charts use a locally bundled Plotly runtime, with no CDN. If WebGL is unavailable, the 3D scenario uses a 2D contour fallback. The 2025 point plot uses regular SVG-based scatter for compatibility with older Macs.

`/legacy` remains the original California UI. The old California API is still accessible under `/api/predict`, `/api/gradient`, etc. The official API lives under `/api/official`.

## 5. Verification and version caveats

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

The Task 7 developer environment passed **123 tests** (including the 101 earlier tests) and browser-interaction checks at desktop/mobile sizes. Browser screenshots are UI evidence; numerical research results must be taken from the recorded official reports. The original training used Python 3.12.16/scikit-learn 1.9.1/XGBoost 3.4.1; a local cross-environment reproduction may produce slightly different predictions. Re-run `analyze` on the Mac using the original trusted saved model before citing final calculus/reliability numbers.

### Linear Regression caveat

The 2024 full-data Linear Regression validation RMSE is anomalously large. A local diagnostic on the processed dataset revealed one parcel with a recorded `lot_area_sqft = 1` and an uncapped `floor_area_ratio = 1475`, associated with a very large negative linear extrapolation. `diagnose` documents this case rather than altering data or the already-trained XGBoost model. Consider robust ratio processing and retraining as a **future controlled experiment**. Do not describe the linear baseline as generally having such an RMSE on other datasets.

### Conflicting recorded sale events

The official processed CSV contains **eight** conflicting same-PARID/same-sale-date groups, each containing two different recorded prices. These remain in the historical experiment and are flagged for a later source-level check. No price is chosen without evidence of which county instrument/record is authoritative.

## 6. Suggested commit

```bash
git status
git add README.md pyproject.toml .gitignore docs/TASK_7.md docs/official \
  src/geovalue/web.py src/geovalue/official \
  public/official/index.html public/official/styles.css \
  public/official/app.js public/official/favicon.svg public/official/plotly.min.js public/official/plotly-LICENSE.txt \
  tests/test_web.py tests/official
python -m pytest -q
git commit -m "feat(official): complete spatial evidence and local calculus dashboard"
git push origin main
```

Never force-stage raw or processed county CSVs, private model artifacts, or generated JSON. Preserve the source manifest on the local machine for provenance. Verify GitHub Actions after the push.
