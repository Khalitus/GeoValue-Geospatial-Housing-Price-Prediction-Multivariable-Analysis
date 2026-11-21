# GeoValue

**Retrospective residential sale-price prediction and spatial sensitivity analysis**

GeoValue is a local research application for studying recorded residential transactions in Allegheny County, Pennsylvania. It combines parcel-level property attributes, transaction records and geographical coordinates to estimate sale prices and examine the spatial behavior of a trained regression model.

The principal dataset contains **82,169 eligible transactions involving 72,607 parcels**, dated 2013–2025. Model selection uses a 2024 validation cohort; the selected XGBoost pipeline is evaluated on **8,691 parcel-disjoint sales recorded in 2025**. The geographical analysis and Spatial Evidence Indicator extend the evaluation beyond a single aggregate error score.

This is a **retrospective research system**. It is not a certified property appraisal, an estimate of current market value or a validated forecast for later years.

## Empirical results

| Measure | Result | Evaluation |
|---|---:|---|
| Test R² | **0.8412** | 2025 held-out sales |
| Test MAE | **$58,190.78** | 2025 held-out sales |
| Test RMSE | **$101,555.74** | 2025 held-out sales |
| Geography-associated RMSE reduction | **15.39%** | 2024 validation ablation |
| Municipality holdout R² | **0.7650** | Separate pre-2024 spatial holdout |

XGBoost was selected among Linear Regression, Random Forest and XGBoost by the lowest 2024 validation RMSE. The reported test results were subsequently reproduced using the saved fitted pipeline and the original evaluation partition. The [experiment record](docs/official/EXPERIMENT_RESULTS.md) gives model comparisons, study design and subgroup errors; the [evidence manifest](poster/evidence/original_model_verification.json) records input hashes and the reproduced metrics.

## Research scope

The application provides four closely related functions:

- **Property-level regression.** A scikit-learn preprocessing pipeline prepares structural, categorical, geographical and transaction-date inputs for XGBoost inference.
- **Spatial generalization assessment.** Geographic-feature ablation and a separate municipality holdout examine the contribution and transferability of location inputs.
- **Numerical multivariable calculus.** A fixed-feature, Gaussian-smoothed geographical scenario is used to calculate contour curves, partial derivatives, gradient vectors, directional derivatives and an area-weighted integral over supported grid cells.
- **Spatial Evidence Indicator.** Distance to the tenth-nearest development location is grouped into validation-derived support tiers and compared with historical absolute errors. The indicator describes geographical sample coverage; it is not a prediction interval or a guarantee of accuracy.

The primary application is available through the `geovalue-official` command-line interface and a FastAPI browser served at `/`. The earlier California census-area study is retained at `/legacy`, with its own feature schema and model. The two studies must not be interpreted as the same prediction target.

## Repository structure

| Path | Contents |
|---|---|
| `src/geovalue/official/` | County data ingestion, modeling, evaluation, spatial analysis and API |
| `src/geovalue/` | Application host and retained California study |
| `public/official/` | Local browser, styles, JavaScript and bundled Plotly runtime |
| `tests/` | Unit and integration tests |
| `docs/official/` | Research methodology, schema, results, validity and operation |
| `docs/legacy/` | Background on the historical California study |
| `poster/` | Research figures, publication assets and original-model evidence |

Raw source datasets, local training reports and the serialized fitted model are not committed. The repository contains their schema, source identifiers, associated evidence hashes and publication figures; their absence does not imply that they were used as input fixtures for CI tests.

## Local execution

Python 3.12 is the supported environment. From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,geo,web,dev]'
python -m pytest -q
```

The official-data pipeline can be run with:

```bash
geovalue-official sources
geovalue-official download
geovalue-official build
geovalue-official profile
geovalue-official train
geovalue-official analyze
python -m uvicorn app:app --reload
```

The local application is served at <http://127.0.0.1:8000>. Downloaded county files remain in `data/official/raw/`, the processed CSV in `data/official/processed/`, and the fitted model and analytical outputs in their respective local artifact and report directories. An existing trusted fitted model can be loaded without retraining.

For field definitions, detailed parameters and artifact locations, consult the [user guide](docs/official/USER_GUIDE.md), [data schema](docs/official/SCHEMA.md) and [reproducibility record](docs/official/REPRODUCIBILITY.md).

## Research documentation

| Document | Subject |
|---|---|
| [Project report](docs/official/PROJECT_REPORT.md) | Research questions, methods, empirical findings and discussion |
| [Methodology](docs/official/METHODOLOGY.md) | Regression design, spatial evidence and numerical calculus |
| [Experimental results](docs/official/EXPERIMENT_RESULTS.md) | Original-model validation, test metrics and diagnostics |
| [Data schema](docs/official/SCHEMA.md) | Official sources, eligibility rules and predictor definitions |
| [Feature timing and validity](docs/official/FEATURE_VALIDITY.md) | Historical availability and limitations of assessment variables |
| [Architecture](docs/official/ARCHITECTURE.md) | Package boundaries, data flow and API contracts |
| [Testing](docs/official/TESTING.md) | Automated, numerical and empirical validation |
| [Figures](docs/official/FIGURES.md) | Publication graphics and screenshot provenance |
| [References](docs/official/REFERENCES.md) | County and scholarly sources |

## Interpretation limits

Property-assessment attributes were obtained from a **November 2026 snapshot**, which may include changes made after earlier transaction dates. Chronological, parcel-disjoint evaluation reduces repeated-property overlap but does not resolve that temporal mismatch. Nominal transaction prices across 2013–2025 have not been inflation-adjusted. The Spatial Evidence Indicator measures training-location density and groupwise error, not individual confidence. The geographical gradient is calculated from a smoothed, controlled scenario, not directly from recorded price changes; the computational support region is not a cadastral or administrative land boundary.

The project is distributed for research and educational use. Only locally produced, trusted serialized model artifacts should be loaded through Joblib.

## Academic context

The research was developed for the Multivariable Calculus course in the Department of Computer Science, Salim Habib University, under **Ms. Uzma Javed Imam**. Contributors: **Khalitus Ali, Hubdar Ali, Eisa Masih and Vansh Raj**.
