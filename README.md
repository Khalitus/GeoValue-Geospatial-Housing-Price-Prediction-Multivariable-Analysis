# GeoValue

**Geospatial Housing Price Prediction with Machine Learning and Multivariable Calculus**

GeoValue studies the **1990 California census block-group** dataset. Its prediction target is the **median house value of a census area**, not a property's current market or sale price.

## Project plan

1. Foundation, Kaggle dataset validation, and exploratory data analysis (**implemented in this starter**).
2. Feature engineering, Linear Regression, Random Forest, XGBoost, evaluation, saved ML pipeline.
3. Geospatial visualization, mathematical price surfaces, gradients, numerical double integration, inference CLI.
4. FastAPI prediction interface and Vercel deployment.
5. Documentation, final university project report, and presentation material.

## Prerequisites

- Python 3.12+
- Git
- Kaggle account to obtain the original dataset

## Set up

```sh
python -m venv .venv
```

Activate the virtual environment: `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
geovalue --help
```

## Get the dataset

Source: https://www.kaggle.com/datasets/camnugent/california-housing-prices

Install the Kaggle CLI and authenticate with your own Kaggle credentials:

```sh
python -m pip install kaggle
kaggle datasets download -d camnugent/california-housing-prices -p data/raw --unzip
```

You should now have `data/raw/housing.csv`. If downloading manually, place `housing.csv` there yourself. Raw datasets are deliberately ignored by Git.

## Analyze housing records

```sh
geovalue status
geovalue inspect --data data/raw/housing.csv
geovalue inspect --data data/raw/housing.csv --output reports/dataset-summary.json
```

The `inspect` command verifies required columns, converts numeric features, permits the dataset's missing `total_bedrooms` values, and prints JSON containing record counts, missing-value counts, category counts, housing value statistics, and correlations with the target.

## Test and coverage

Tests use small synthetic records, **not the downloaded Kaggle CSV**, so they're runnable in CI:

```sh
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

A GitHub Actions workflow runs pytest on pushes and pull requests.

## Data integrity

- Do not impute missing values before train/test splitting. Task 2 will use training-only preprocessing inside sklearn pipelines.
- Do not use model evaluation results from the test set to tune hyperparameters.
- Do not describe a modeled spatial gradient as a causal effect of moving a property.
- Data, reports, local credentials, and trained artifacts are excluded from Git by default.

## License and dataset rights

Review the dataset's original license and attribution requirements on Kaggle before republishing its data.
