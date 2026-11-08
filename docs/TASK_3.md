# Task 3 — Geospatial analysis, numerical calculus and prediction CLI

Task 3 completes the core university project. It extends the existing trained GeoValue pipelines without re-training the final Kaggle model, and produces reusable numerical and HTML output files. Task 4 will add the hosted user interface; Task 5 will prepare the submission report.

## Setup

Reuse the existing Python 3.12 virtual environment and the trained model (`artifacts/housing-model.joblib`). On the user's external SSD, use `--link-mode copy`:

```bash
source .venv/bin/activate
uv pip install --link-mode copy -e ".[ml,geo,dev]"
python -m pytest -q
```

Do **not** extract any ZIP over your `data/raw/housing.csv` or existing saved training artifacts. The Task 3 update ZIP contains only source code, documentation, tests, and CI configuration.

## Predict one census area

```bash
geovalue predict \
  --model artifacts/housing-model.joblib \
  --longitude -122.23 --latitude 37.88 \
  --housing-median-age 41 \
  --total-rooms 880 --total-bedrooms 129 \
  --population 322 --households 126 \
  --median-income 8.3252 \
  --ocean-proximity "NEAR BAY"
```

The output is JSON containing a prediction of the **median house value for a census area, in historical 1990 USD**. It does not represent today's sale price of one property. The bedrooms field can be omitted, in which case the fitted pipeline uses its learned training-set median.

## Predict a CSV batch

The input file must have columns: `longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity`. The target column is not required. Rows are validated and retained in order.

```bash
geovalue predict --model artifacts/housing-model.joblib \
  --input data/my-input-areas.csv --output reports/predictions.csv
```

The output CSV contains the original columns and `predicted_median_house_value`. Do not submit current individual property attributes in place of block-group aggregates and describe the outputs as current home valuations.

## Generate interactive maps and mathematical analysis

```bash
geovalue analyze \
  --data data/raw/housing.csv \
  --model artifacts/housing-model.joblib \
  --output-dir reports/geospatial \
  --grid-size 90 --sigma 1.6 --support-km 45 --map-points 1250
```

Output files:

| File | Meaning |
| --- | --- |
| `housing-map.html` | Interactive observed price and trained-model prediction layers (Folium); sample of at most 1,250 areas |
| `price-surface-3d.html` | Interactive Plotly 3D surface of the **smoothed fixed-feature scenario** |
| `price-contours.html` | Contour/level-curve view of the same scenario |
| `price-gradients.html` | Local spatial gradient magnitude and arrows toward higher modelled value |
| `price-surface.npz` | Numerical price grid, coordinates and partial derivative arrays; NumPy pickle is disabled when loading |
| `geospatial-report.json` | Summary, numerical double integral, scenario assumptions and limitations |

The HTML files use hosted tiles or Plotly's CDN; a network connection is needed for all components to display in the browser. HTML files can be opened locally on a Mac in the default browser.

### Coordinate system

Longitude/latitude are expressed in `EPSG:4326` (degrees). Spatial calculations use **California Albers** `EPSG:3310`, whose native unit is metres. We convert to kilometres before differentiation or integration. Grid x/y directions are axes of the projection; an indicated gradient direction is clockwise from **projected grid north**, not an exact geodetic compass bearing.

### Constructing the price surface

The trained predictive model is a function of location and socioeconomic/housing covariates:

\[\widehat P=f(\lambda,\phi,I,A,R,B,N,H,O).\]

A 2D scenario varies longitude \(\lambda\) and latitude \(\phi\) while holding non-geographical inputs fixed at dataset medians and `ocean_proximity` at the dataset mode. The projected grid is limited by a nearest-observation distance threshold (45 km by default). The model prediction field is then smoothed with a normalized Gaussian filter; points outside the support mask are retained as missing data. We call this a **smoothed scenario surface** \(S(x,y)\), not a raw XGBoost surface.

A tree-based estimator is not smoothly differentiable. Raw tree predictions are piecewise constant and may jump at split boundaries; a nonzero spatial gradient of the smoothed scenario must **not** be interpreted as an exact derivative of the original fitted tree model.

### Partial derivatives, gradients and directional derivatives

For the 2D smooth field \(S(x,y)\), numerical derivatives use `numpy.gradient` with projected kilometre spacings:

\[S_x\approx \frac{S(x+\Delta x,y)-S(x-\Delta x,y)}{2\Delta x},\qquad S_y\approx\frac{S(x,y+\Delta y)-S(x,y-\Delta y)}{2\Delta y}.\]

The gradient vector and magnitude are:

\[\nabla S=(S_x,S_y),\qquad \|\nabla S\|=\sqrt{S_x^2+S_y^2}.\]

These quantities have units of historical **USD/km**. In a fixed scenario, the gradient direction points toward the fastest increasing value of the smoothed mathematical surface. For a user-chosen direction \(\theta\) clockwise from the projected north axis:

\[D_\theta S=S_x\sin\theta+S_y\cos\theta.\]

### Query a local price-gradient estimate

First generate `price-surface.npz` using `geovalue analyze`. Then:

```bash
geovalue gradient \
  --surface reports/geospatial/price-surface.npz \
  --longitude -122.23 --latitude 37.88 \
  --bearing 90 --max-distance-km 30
```

The command picks the nearest calculated valid gradient node, reports its distance from the requested point and rejects queries outside the distance tolerance. Use the distance output to judge approximation quality. Some points near the supported-region boundary will not have a defined central derivative.

### Double integration over supported grid cells

Let \(D\) denote the union of valid projected grid cells where all four corner values exist. Approximate the 2D integral with a trapezoid-style four-corner cell average:

\[\iint_D S(x,y)\,dA\approx\sum_{C\subseteq D} \frac{S_{00}+S_{01}+S_{10}+S_{11}}{4}\,\Delta A_C.\]

The **area-weighted average** of the scenario surface is:

\[\bar S=\frac{\iint_D S\,dA}{\iint_D 1\,dA}.\]

The integral has units of USD km², area is km² and mean value is USD. This is not total property value, transaction volume, or an official California land-area measure. Cell-based integration and nearest-data masking approximate a mathematical support domain, not county boundaries.

## Testing and evaluation

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

`test_calculus.py` checks exact analytical derivatives for linear fields and a known double integral. `test_geospatial.py` checks projection round-trip and interactive map layers. `test_inference.py` covers validation, single prediction, batch CSV, and CLI error behavior. `test_analysis.py` runs the complete analysis with a fitted synthetic small model and checks the output reports and HTML files. The existing Task 1 and Task 2 tests remain in place.

**Real Kaggle model accuracy belongs to Task 2**, from `reports/training-report.json`. Do not use the synthetic data from pytest fixtures to claim real-world prediction accuracy. Additional spatial holdout validation would be needed to claim transfer to unseen geographic regions.

## Reproducibility, safety and caution

- Version-controlled source should not contain Kaggle raw CSV data or model artifacts by default; these remain local and are ignored by Git.
- Joblib uses pickle-compatible serialization. Only load your own or trusted `.joblib` model artifacts.
- Model behaviour and the smoothed surface are sensitive to fixed reference attributes, smoothing strength, grid resolution and distance-to-data mask.
- Input records describe **area aggregates**, not listing-level attributes; the age, income, rooms and population refer to the 1990 census area.
- The dataset's target is upper-coded near $500,001, affecting training results and interpretations at the cap.
- Outputs are an educational multivariable calculus demonstration, not causal housing-market or financial advice.
