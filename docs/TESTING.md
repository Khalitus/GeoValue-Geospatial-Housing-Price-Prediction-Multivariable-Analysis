# Testing and verification

## 1. Test strategy

GeoValue uses `pytest` for synthetic-data unit/integration checks and GitHub Actions for repeatable CI. Separate ad-hoc browser and API stress checks were conducted before documentation. **Unit tests passing does not prove generalization to new geographical regions**; that limitation belongs to model evaluation, not software correctness.

## 2. Reproducible commands

From the project root and activated Python 3.12 environment:

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

The GitHub Actions workflow installs `.[ml,geo,web,dev]` and runs the full test suite. At the reviewed repository commit `5a855bf24f15b102ee5ad948943928dc8737d0f1`, workflow run [42475421269](https://github.com/Khalitus/GeoValue-Geospatial-Housing-Price-Prediction-Multivariable-Analysis/actions/runs/42475421269) passed **67 tests**. A separate audit of a code-matched local copy also recorded **67 passed, approximately 90% line coverage**, using Linux/Python 3.13.5. The author previously ran 67 local tests on macOS/Python 3.12; those results can be rerun with the commands above.

## 3. Areas covered

| Layer | Example checks |
|---|---|
| Data | Required schema, numeric validity, missing bedrooms, file errors, descriptive analysis |
| Feature engineering | Density ratios, missing/zero denominators, stable column selection |
| Modeling | Construction, fitting, metrics, serialization and reload |
| Training | Non-overlapping splits, model selection, geographical feature comparison |
| CLI/inference | Single and CSV prediction, malformed inputs, output files |
| Spatial mathematics | Projection, synthetic analytic gradients, supported-area integration, saved NPZ |
| Geospatial generation | HTML output generation and figure data |
| Web/API | Dashboard assets, prediction responses, validation, missing-model handling, gradient responses |
| Packaging | Generated-file handling and full-data backup behavior |

Numerical tests compare calculations against known functions/expected results. They should not be confused with the measured statistical accuracy of the actual 20,640-record dataset.

## 4. Supplemental stress checks

A separate audit generated a **synthetic 1,100-row fixture and temporary XGBoost model**. Requests were issued to an in-process ASGI client, not to a public service:

| Scenario | Result |
|---|---|
| Valid predictions | 400/400 HTTP 200 with finite values |
| Deliberately invalid inputs | 100/100 HTTP 422 |
| Total request volume | 500, concurrency limited to 12 |
| Supported gradient queries | 60/60 correct responses and derivative consistency |
| Unsupported gradient location | Correctly returned HTTP 422 |
| Suspicious path traversal | Returned HTTP 404 |
| In-process p50/p95/p99 latency | 60.31 / 109.11 / 133.14 ms |

Latency was measured in a shared audit environment and is **not a production performance guarantee** or a benchmark for the author's Mac. These are behavioral checks on a simulated dataset, not empirical housing market results.

## 5. Browser review and image evidence

A real browser exercised the committed UI at desktop **1440×900** and mobile **390×844** viewport sizes. The five primary sections, prediction form, location-layer switching, supported gradient queries and unsupported-location feedback were exercised; no uncaught JavaScript errors or horizontal overflow appeared during those runs. The screenshot set is in [`docs/images`](images/) and catalogued in [FIGURES.md](FIGURES.md).

For reproducibility, the browser harness used a synthetic data/model fixture but retained the user's separately recorded model-evaluation summary. Plotly was provided locally in the audit browser, because that environment did not have external CDN access. Thus **screenshots are UI testing evidence only**; plotted locations and example predictions are not the real empirical 1990 analysis.

## 6. Known warnings and limits

- **Starlette/httpx:** a deprecation warning in `fastapi.testclient` is nonfatal.
- **Folium/CartoDB:** map tiles may require an external key; the spatial calculus functions do not rely on those raster tiles.
- **Chart CDN:** the delivered frontend references `cdn.plot.ly`; offline chart behavior requires bundling that JavaScript locally.
- **Mac packages:** external-SSD file copying caused environment problems during development; a dedicated internal-drive venv solved the author's reported issue.
- **Joblib safety:** tests load only trusted locally generated model artifacts; never deserialize arbitrary user-submitted models.

## 7. Model-evaluation evidence is separate

The **real training report** records held-out test **R² 0.84347, RMSE $45,167.23, MAE $30,559.76**. It is reproduced in [METHODOLOGY.md](METHODOLOGY.md) from `public/data/training-report.json` and not inferred from test fixture runs. A random income-stratified split was used, so **spatial holdout validation remains a documented limitation**.
