# GeoValue

**Geospatial Housing Price Prediction with Machine Learning and Multivariable Calculus**

GeoValue is a university project using historical **1990 California census block-group** records. The prediction target is the **median house value of a census area**, not a specific home's current market price.

## Five-task roadmap

1. Foundation, Kaggle dataset validation and descriptive analysis — **complete**.
2. Feature engineering, Linear Regression, Random Forest, XGBoost, evaluation and saved ML pipeline — **complete**.
3. Geospatial visualization, mathematical price surfaces, gradients, numerical double integration and a complete inference CLI.
4. FastAPI web interface and Vercel deployment.
5. Final documentation, report and presentation materials.

## Requirements

- Python 3.12+ (the project was designed for 3.12)
- Kaggle's `housing.csv` from <https://www.kaggle.com/datasets/camnugent/california-housing-prices>
- Git

## Install on macOS (recommended: uv)

If your `uv` and Python 3.12 setup from Task 1 already works, **reuse your existing `.venv`**:

```sh
source .venv/bin/activate
uv pip install --link-mode copy -e ".[ml,dev]"
```

Alternatively, with an existing Python 3.12 virtual environment:

```sh
python -m pip install -e ".[ml,dev]"
```

XGBoost is a compiled dependency. Its PyPI distributions support Intel and Apple Silicon Macs, but some macOS installations may need an OpenMP runtime. Check `python -c "import xgboost; print(xgboost.__version__)"` before training. If the import complains specifically about missing `libomp.dylib`, see the XGBoost installation guide and install the OpenMP runtime; do not rebuild Python.

Keep your previously downloaded `data/raw/housing.csv` in place. The raw CSV and generated artifacts are excluded from Git.

## Task 1 — Explore the dataset

```sh
geovalue status
geovalue inspect --data data/raw/housing.csv --output reports/dataset-summary.json
```

## Task 2 — Train, compare, save and inspect models

```sh
geovalue train \
  --data data/raw/housing.csv \
  --model-output artifacts/housing-model.joblib \
  --report-output reports/training-report.json \
  --seed 42

geovalue model-info --model artifacts/housing-model.joblib
```

Model selection and test evaluation happen automatically inside `train`. It prints a short JSON summary and saves the complete report to `reports/training-report.json`.

Inspect your actual measured results:

```sh
python -m json.tool reports/training-report.json
```

Use fewer trees for a quick smoke test, **not your final report**:

```sh
geovalue train --forest-trees 8 --xgboost-trees 12 \
  --model-output artifacts/quick-model.joblib \
  --report-output reports/quick-training.json
```

### How we prevent data leakage

- First split input rows into train (60%), validation (20%), and held-out test (20%) sets using a fixed random seed. When sample counts allow, the split is stratified on coarse median-income bins.
- All numeric imputation, scaling and one-hot category learning happen **inside scikit-learn pipelines**, fitted on the training subset only.
- Train and compare all three model families on the same train/validation partitions.
- Select the model with the **lowest validation RMSE**. The test set is not used for model selection.
- Compare this selected model on validation data with and without **longitude, latitude and ocean_proximity**. This is a feature-ablation experiment, not a causal study.
- Fit a fresh instance of the chosen model on train+validation, then evaluate it **once** on the untouched test rows.
- Save the complete inference pipeline to `artifacts/housing-model.joblib`, including preprocessing and the chosen regressor.

### Three models

- `linear`: ordinary least-squares Linear Regression (baseline)
- `random_forest`: Random Forest regressor (120 trees by default)
- `xgboost`: gradient-boosted tree regression (220 trees by default)

### Feature engineering

`HousingFeatureEngineer` adds `rooms_per_household`, `bedrooms_per_room` and `population_per_household`. Undefined ratios, including missing bedroom counts, remain NaN and are then imputed using **training-only** median statistics. Numerical features are standardized; `ocean_proximity` is one-hot encoded with safe handling of unknown categories.

### Report fields

- `validation_results`: train and validation MAE, RMSE and R² per model.
- `selected_model` and `selection_metric`: the winning architecture and validation criterion.
- `geographic_feature_experiment`: validation metrics with/without location features and the RMSE difference.
- `held_out_test`: final test MAE, RMSE and R² of the refitted winner.
- `software_versions`: versions to support reproducibility and deployment compatibility.

All error metrics are in historical (1990) US dollars. The source data caps median house values at 500001, affecting error interpretation near the upper limit. A standard random split is not a spatial holdout; geographical generalization needs separate tests.

### Programmatic inference (used in Task 3)

```python
import pandas as pd
from geovalue.modeling import predict_records

rows = pd.read_csv("data/raw/housing.csv").drop(columns=["median_house_value"]).head(2)
values = predict_records("artifacts/housing-model.joblib", rows)
print(values)
```

Only load `.joblib` artifacts you created or trust: like other pickle-based formats, Joblib loading can execute arbitrary code.

## Tests

```sh
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

Automated tests use generated synthetic records to exercise data validation, feature engineering, trained preprocessing, reproducible splitting, three-model comparisons, geographic ablation, saved-model reloads and CLI integration. **Synthetic test metrics must not be used as university project accuracy results.** GitHub Actions runs the full test suite using `.[ml,dev]`.

## Project structure

```text
src/geovalue/
  __init__.py       package version
  __main__.py       python -m geovalue
  cli.py            status, inspect, train, model-info
  data.py           loading, schema validation, descriptive statistics
  features.py       leakage-safe derived housing ratios
  modeling.py       model factory, metrics, saving, loading, inference
  training.py       splits, comparison, ablation, final refit + test

tests/               pytest units and integration tests
docs/TASK_2.md       detailed explanation of training methodology
artifacts/           locally generated models (ignored by Git)
reports/             locally generated JSON reports (ignored by Git)
```

## Data/license notes

Download housing data through the Kaggle dataset page. Review the original data licensing and attribution conditions before redistributing. Do not commit the raw CSV, credentials or user-specific files.
