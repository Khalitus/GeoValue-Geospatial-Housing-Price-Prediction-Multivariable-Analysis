# Task 2 — Machine Learning Implementation Notes

## Objective

Predict `median_house_value` for **1990 California census block groups** using housing, demographic and location attributes. This is area-level regression, not individual residential-property valuation or current price forecasting.

## A. Data preparation and splitting

`geovalue.data.load_housing_data()` enforces the Task 1 schema. `split_housing()` makes 60% training, 20% validation and 20% test partitions using `random_state=42`. Split indices do not overlap. Income bins are used for stratification when each bin has enough records; smaller synthetic datasets can fall back to an ordinary seeded split.

We never impute before splitting: otherwise the imputation value could incorporate validation/test information. Fit preprocessing on training rows inside a scikit-learn `Pipeline` and a `ColumnTransformer`.

## B. Features and transformations

Let R be total rooms, B be bedrooms, H be households and N be population. We add:

- `rooms_per_household = R/H`
- `bedrooms_per_room = B/R`
- `population_per_household = N/H`

Missing B or undefined division produces NaN, which is handled by the training-fitted median imputer. Raw numerical features and engineered ratios are also standardized (mean zero, unit variance on training data). The ocean-proximity category becomes one-hot indicators; categories unseen during training are safely ignored.

No target-derived features are created.

## C. Model families

The same preprocessing is used for three regression algorithms:

1. **Linear Regression** — baseline estimates a weighted linear combination of transformed predictors.
2. **Random Forest** — ensembles decision trees trained on resampled observations. Repeated splits model nonlinearities and interactions.
3. **XGBoost** — gradient-boosted decision trees, with sequential optimization of squared-error loss.

Default tree settings trade modest runtime against a meaningful comparison; they are not a comprehensive hyperparameter search. `--forest-trees` and `--xgboost-trees` are configurable.

## D. Scoring

For actual values y_i, predicted values ŷ_i and n rows:

- MAE = (1/n) Σ |y_i − ŷ_i|
- RMSE = √((1/n) Σ (y_i − ŷ_i)²)
- R² = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − mean(y))²

MAE and RMSE have the same monetary unit as the target, **historical USD**. Larger errors are penalized more heavily by RMSE.

## E. Leakage-safe model selection

Each candidate is trained on 60% and evaluated on the same 20% validation split. The smallest validation RMSE determines the architecture. The comparison uses only train and validation predictions. The winning architecture is rebuilt from scratch and fitted on the combined 80% development set. **Only after selection and refitting** do we calculate the held-out test metrics using the untouched 20%.

## F. Geography experiment

After selecting the best model type, we repeat validation training with `longitude`, `latitude` and `ocean_proximity` excluded. Report the difference:

`RMSE_without_geography − RMSE_with_geography`

Positive values imply location information helped on this particular random validation split. Since the full version was among the models considered for selection, the difference is exploratory and should not be represented as a blind test or causal proof. Task 3 builds geographic maps and mathematical surfaces.

## G. Artifacts and interpretation

- `artifacts/housing-model.joblib` stores the fitted feature engineer, preprocessing and final selected estimator.
- `reports/training-report.json` stores exact validation/test scores, the split counts, package versions, the selected model and geography experiment.

Both are generated locally and gitignored; raw data is also gitignored. The model should be loaded only from trusted files. Saving a model to Joblib does not mean it is compatible with every future scikit-learn/XGBoost version: use `software_versions` when packaging the web app in Task 4.

## H. Tests

`tests/test_features.py` checks derived ratios and undefined values. `tests/test_modeling.py` tests pipeline leakage precautions, unknown categories, metrics and model reload. `tests/test_training.py` tests split disjointness, reproducibility and full model comparison. `tests/test_cli_ml.py` runs training through the public CLI.

Run the full suite after each change:

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

For final submitted accuracy, use `geovalue train` with the actual Kaggle CSV, not synthetic fixture data.
