# Official residential-sale methodology and applied spatial calculus

## 1. Research object and temporal scope

GeoValue estimates **recorded nominal USD residential sale price** for eligible transactions in Allegheny County, Pennsylvania, from 2013 to 2025. It is a supervised tabular regression model, not a future-price forecast or an online appraisal service. The primary row unit is a **transaction**, with parcel ID metadata retained for leakage checks. A parcel may have multiple eligible transactions within one period.

Data lineage combines (a) recorded sales; (b) the county's **current** assessment snapshot; (c) March 2025 parcel centroids. All are distributed via WPRDC. Join keys are normalized 16-character identifiers. The assessment is not a point-in-time history and may describe a property after a historical transaction. That temporal mismatch remains even after using a chronological evaluation split.

The reproducible sample contains 82,169 eligible transactions from 72,607 parcels. The builder retains explicit `SALEDESC=VALID SALE` descriptions, checks sale dates and configured price bounds, joins residential assessment records and valid centroids, and screens impossible physical and construction-date attributes. Only a defined predictor allowlist enters the model; sale price, previous sale price, current assessed valuations and administrative sale descriptions are not predictors.

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for the measured build counts and 2025 test statistics.

## 2. Predictors, target and preprocessing

### Base predictors

The 14 numerical inputs are:

`latitude`, `longitude`, `bedrooms`, `full_baths`, `half_baths`, `total_rooms`, `living_area_sqft`, `lot_area_sqft`, `year_built`, `stories`, `fireplaces`, `garage_spaces`, `sale_year`, `sale_month`.

Five categorical inputs are:

`construction_grade`, `condition_code`, `heating_cooling`, `municipality`, `zip_code`.

The regression target `sale_price` is the recorded sale price in nominal dollars. We do not use `parcel_id` or previous/current property-value fields as predictors.

### Engineered features

For each input observation:

\[
\text{property age at sale} = \max(0,\text{sale year}-\text{year built})
\]

\[
\text{bathroom equivalents} = \text{full baths} + 0.5\times\text{half baths}
\]

\[
\text{floor-area ratio} = \frac{\text{living area (sq ft)}}{\text{lot area (sq ft)}}
\]

An undefined ratio is represented as missing and then imputed in the training-fitted preprocessing pipeline. Numerical features use median imputation and scaling. Categorical features use most-frequent imputation and one-hot encoding with a minimum-frequency threshold and unknown-category handling. The features and preprocessing are encapsulated in a scikit-learn Pipeline, fitted only after splitting.

**Known outlier:** very small reported land areas create unusually large ratios. One 2024 validation record reports a one-square-foot lot and generated a very large negative Linear Regression extrapolation. The `diagnose` command reports this case. The delivered code **does not silently modify or remove** that official record or the originally fitted XGBoost model; any ratio-capping scheme would require retraining and a new controlled comparison.

## 3. Train, validation, and test protocol

| Partition | Years | Count | Purpose |
|---|---|---:|---|
| Training | 2013–2023 | 60,767 | Fit candidate model and preprocessing |
| Validation | 2024 | 8,656 | Select architecture by lowest RMSE, fit support-tier thresholds |
| Test | 2025 | 8,691 | Final evaluation and *post-calibration* evidence audit |

The builder also rejects 4,055 potential cross-partition parcel overlaps by removing earlier observations for parcels appearing in newer partitions; this preserves parcel identity separation. The chosen algorithm is retrained on 69,423 development records (training + validation) before one 2025 test evaluation. A separate municipality holdout evaluates geographical transfer within the earlier years, not performance in a future year.

### Model candidates and objective

Candidates: Linear Regression, 120-tree Random Forest and 220-tree XGBoost. XGBoost uses `reg:squarederror`, `max_depth=6`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.9`, and histogram split selection. The model-selection criterion is 2024 validation RMSE. This is a deliberately bounded project comparison, not an exhaustive hyperparameter search.

For actual target \(y_i\), predicted \(\hat y_i\) and \(n\) examples:

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|,
\quad
\mathrm{RMSE}=\sqrt{\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]

### Observed outcomes

XGBoost had the lowest validation RMSE ($98,697.19) and reached 2025 test **MAE $58,190.78**, **RMSE $101,555.74**, **R² 0.84120**. Excluding location variables (latitude, longitude, municipality, ZIP) increased validation RMSE to $116,653.97, a reduction of $17,956.78 when geography was retained. These values are sourced from the user's actual full model-training report. A causal interpretation is not supported.

## 4. Small methodological contribution: spatial evidence and validation error

### Research gap in the delivered application

A single point estimate does not explain whether the proposed property location has many nearby *development* observations, or how the model performed in locations with similar geographical support during validation. GeoValue adds a **spatial evidence indicator**. It is an auditable descriptive support layer, **not** a calibrated probability, a confidence interval or an individual prediction-error guarantee.

### Projected nearest-neighbor support

Let \(z_i=(x_i,y_i)\) be a 2024 validation property's point in kilometre-based UTM coordinates. Let \(T\) be the 2013–2023 training-locations set. Define:

\[
d_{10}(z_i,T)=\text{distance from }z_i\text{ to its 10th-nearest location in }T\]

Larger values mean fewer recorded development locations in the immediate vicinity. We estimate thresholds from the validation distance distribution:

\[
q_1=Q_{1/3}\big(d_{10}(z_i,T)\big),\qquad
q_2=Q_{2/3}\big(d_{10}(z_i,T)\big).
\]

Tiers are **dense** for \(d_{10}\le q_1\), **typical** for \(q_1<d_{10}\le q_2\), and **sparse** for \(d_{10}>q_2\). The implementation uses an indexed nearest-neighbor search (`scipy.spatial.cKDTree`) and a fixed neighbour count of ten for the real dataset.

### Linking support to error

For each of the three 2024 tiers, using model predictions from a model fitted on *pre-2024* records, compute:

\[
\mathrm{MAE}_{b}=\frac{1}{|V_b|}\sum_{i\in V_b}|y_i-\hat y_i|,
\quad Q_{0.8,b}=\operatorname{quantile}_{0.8}\{|y_i-\hat y_i|:i\in V_b\}
\]

The group sample count, RMSE, median neighbor distance, and the Spearman correlation between \(d_{10}\) and absolute errors are stored. The label is not assumed to predict accuracy automatically; the generated results must be examined.

For predictions from the **final** model, distances are computed against the expanded 2013–2024 development reference set. The display includes the tier and its historical 2024 error summary. Since the reference set grew, matching support tier to validation errors is **approximate** and explicitly not an exchangeable confidence guarantee.

Finally, the already fixed 2024 boundaries are applied to a **separate 2025 support audit**, using the final refitted model's 2025 residuals. The 2025 audit measures whether the stratification remains informative without tuning the cutoffs from test targets. No 2025 observations enter the training-location support set.

This modest, measurable addition is GeoValue's proposed **incremental novelty**, rather than a claim of inventing uncertainty estimation.

## 5. County-projected geographical surface

The model is a multivariable function \(f(x,y,\mathbf z)\) of position and property features. Fix non-location predictors \(\mathbf z=\mathbf z_0\) at medians/modes, except the transaction year and month set to **2025 / June**, to get the controlled geographical scenario:

\[
P(x,y)=f(x,y,\mathbf z_0).
\]

Longitude and latitude are transformed from EPSG:4326 into **NAD83 / UTM zone 17N (EPSG:26917)**, whose coordinates are expressed in metres; we divide by 1,000 to use kilometres. This projected CRS is suitable for the western Pennsylvania study area. See [EPSG:26917 reference](https://epsg.io/26917).

The default 86×86 grid is evaluated across the range of observed projected parcel coordinates. Nodes more than **4.5 km** from any observed property centroid are marked unsupported. A masked, normalized Gaussian smoothing with default standard deviation **1.5 grid cells** produces \(S(x,y)\). Unsupported cells remain invalid.

**Why smooth?** A finite boosted-tree ensemble is piecewise constant with jumps, so differentiating raw tree outputs gives uninformative derivatives within leaves and undefined derivatives at discontinuities. The calculus operates on the **smoothed numerical scenario** \(S\), not on the raw regressor \(f\).

### Partial derivatives, gradient and directional derivatives

Finite differences approximate the two partial derivatives on the projected grid:

\[
S_x(x,y)\approx\frac{S(x+h,y)-S(x-h,y)}{2h},\quad
S_y(x,y)\approx\frac{S(x,y+k)-S(x,y-k)}{2k}
\]

\[
\nabla S=(S_x,S_y),\quad \|\nabla S\|=\sqrt{S_x^2+S_y^2}.
\]

For a requested angle \(\theta\) clockwise from **projected grid north**:

\[
D_{\theta}S=S_x\sin(\theta)+S_y\cos(\theta).
\]

The stored partial derivatives and magnitude are in **nominal USD per kilometre**. A query samples the nearest supported grid derivative within 5 km. The gradient direction is relative to the UTM grid, not true geographical north. Neither magnitude nor direction is evidence that relocating a real property would cause the predicted change.

### Numerical double integration

The computational supported domain \(D\) consists of grid cells whose four corner values are finite. We use each cell's average of four corner values as a two-dimensional trapezoidal quadrature:

\[
I\approx\sum_{c\in D}A_c\,\frac{S_{c,1}+S_{c,2}+S_{c,3}+S_{c,4}}{4},\quad
\bar S_D=\frac{I}{\sum_{c\in D}A_c}.
\]

The integral has units **nominal USD·km²**; the area-weighted mean is in USD. Crucially, \(D\) is not a true county polygon and may include areas outside county land boundaries. It is not a sum of property values or an empirical house-weighted county price average.

## 6. Threats to validity

- **Temporal feature leakage:** later assessment snapshots can describe renovations after early sale dates. Chronological splitting cannot undo it.
- **Selection bias:** only county-described valid sales within price limits are retained.
- **Transaction ambiguity:** eight same-parcel/same-day groups have conflicting recorded prices. They remain flagged until source-level adjudication, not silently corrected.
- **Nominal prices:** dollars across 2013–2025 are not inflation-adjusted.
- **Sparse-support ambiguity:** geographical density alone may correlate only weakly with absolute errors, and reliability tiers are not guaranteed uncertainty intervals.
- **Fixed reference features:** municipality and ZIP remain fixed even when longitude/latitude vary, creating physically unrealistic hypothetical combinations.
- **Geometry and smoothing:** projection, grid size, support radius and Gaussian bandwidth affect the calculus results; no official county polygon is used for area integration.
- **Generalization:** separate municipal holdout covers distinct municipalities in pre-2024 data, but not unseen future markets or neighboring counties.

For practical commands and artifact paths see [Task 7](../TASK_7.md). For audit and test results see [Testing](TESTING.md) and [Experiment results](EXPERIMENT_RESULTS.md).
