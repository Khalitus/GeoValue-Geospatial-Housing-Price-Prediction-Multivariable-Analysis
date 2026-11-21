# GeoValue: Retrospective Sale-Price Prediction and Spatial Sensitivity Analysis

## Abstract

GeoValue investigates recorded residential sale prices in Allegheny County, Pennsylvania, using a combination of official transaction data, a later property-assessment extract and parcel centroids. Eligibility filtering and parcel-key joins produced **82,169 transactions associated with 72,607 parcels** from 2013–2025. A supervised modeling pipeline compares Linear Regression, Random Forest and XGBoost, using 2024 validation data for model selection and a parcel-disjoint 2025 cohort for evaluation. The selected XGBoost pipeline achieved **MAE $58,190.78**, **RMSE $101,555.74** and **R² 0.8412** on 8,691 held-out transactions. In a separate validation ablation, geographical features reduced RMSE by 15.39% compared with the same model configuration without geographical inputs.

A supplementary Spatial Evidence Indicator describes the density of development observations near a prediction location. Support tiers are defined using 2024 validation distances and compared against a separate 2025 audit. The sparse-support cohort had a mean absolute error of $69,389.57, compared with $54,145.08 in the dense-support cohort. The association between support distance and individual validation error was weak (Spearman ρ = 0.09687), and the indicator is not a confidence interval. The project further evaluates a Gaussian-smoothed, fixed-property geographical scenario using partial derivatives, directional derivatives and numerical double integration. Those derivatives describe the model's constructed field, not observed price appreciation or causal effects. The application provides a local command-line interface, analytical API and browser dashboard. Since its property-characteristic extract postdates many transactions, its evaluation is retrospective rather than a validated point-in-time forecast.

**Keywords:** residential sale prices; hedonic characteristics; XGBoost; geospatial regression; gradient field; spatial evidence; numerical integration; retrospective evaluation.

## 1. Problem and research questions

Residential property prices vary with building characteristics, recorded sale conditions and location. Regression models can summarize these relationships as point predictions, but a single estimate gives limited information about variation in the model's geographical response or the distribution of nearby training records. GeoValue examines three connected questions:

1. How do specified regression models perform on parcel-disjoint, later-year sales?
2. How much do geographical inputs contribute to validation accuracy and transfer across municipalities?
3. What do numerical spatial derivatives and training-location support reveal about a controlled model scenario and its observed errors?

The analysis is constrained to eligible recorded sales from Allegheny County and does not infer causal changes from moving a property. It extends established methods by integrating prediction, supported numerical analysis and descriptive local evidence in one reproducible research workflow.

## 2. Data and sampling

The data were retrieved through the Western Pennsylvania Regional Data Center on **14 November 2026**. The sources comprise recorded sales, current assessments and parcel centroids. The source identifiers, eligibility rules and processed input schema are documented in [SCHEMA.md](SCHEMA.md) and [REFERENCES.md](REFERENCES.md).

Starting from **508,013 recorded sale rows**, the build restricts county sale descriptions to `VALID SALE`, applies configured date and price criteria, joins parcel attributes and centroid positions, and rejects implausible or temporally inconsistent structural fields. The resulting research dataset contains **82,169 eligible transactions** and **72,607 distinct parcels**. The transaction is the unit of prediction; a parcel can appear in more than one sale record. Sale price is the nominal-dollar target. Assessment valuations and previous sale prices are excluded from the predictor set.

The assessment file represents a **November 2026 property snapshot**, not a historical record of each property as it existed on every sale date. Renovations, subdivision or reassessment may affect what characteristics the records attach to older sales. This feature-timing issue is central to the interpretation of all evaluation metrics, as examined in [FEATURE_VALIDITY.md](FEATURE_VALIDITY.md).

## 3. Modeling framework

Fourteen numerical features represent coordinates, property size, room and bathroom counts, construction vintage, amenities and transaction timing. Five categorical fields represent construction grade, condition, heating/cooling, municipality and ZIP code. A fitted feature transformer adds property age at sale, bathroom equivalents and floor-area ratio. Preprocessing uses numerical median imputation and scaling, categorical most-frequent imputation, and one-hot encoding. The entire transformation is fitted only on the relevant training partition.

The chronological evaluation is parcel-disjoint: **60,767 observations from 2013–2023** form the training set, **8,656 observations from 2024** form validation, and **8,691 observations from 2025** form the test set. Excluding 4,055 earlier repeat-sale observations avoids parcel identity crossing these partitions. Model selection minimizes validation RMSE; the selected architecture is then refitted on 69,423 development observations. The held-out 2025 targets do not choose the model.

| Candidate | 2024 validation RMSE (USD) | Validation R² |
|---|---:|---:|
| Linear Regression | 956,526.74 | -14.1316 |
| Random Forest | 111,735.19 | 0.7935 |
| **XGBoost** | **98,697.19** | **0.8389** |

The Linear Regression baseline is sensitive to an extreme floor-area-ratio observation and produces a very large error. The study preserves that diagnostic rather than silently removing observations to improve the baseline score. XGBoost uses squared-error regression, 220 trees, depth six, learning rate 0.06, observation subsampling and histogram split selection. The comparison does not establish that these hyperparameters or this algorithm are globally optimal.

## 4. Prediction and geographic generalization

The selected XGBoost model achieved **R² 0.84119967**, **MAE $58,190.78** and **RMSE $101,555.74** on the untouched 2025 cohort. Its saved pipeline was subsequently applied again to the same partition; the metrics matched the original training report to numerical precision. The verification and file hashes are published in [poster/evidence/original_model_verification.json](../../poster/evidence/original_model_verification.json).

Excluding longitude, latitude, municipality and ZIP code increased 2024 validation RMSE from $98,697.19 to $116,653.97. The $17,956.78 difference corresponds to a **15.39% reduction** relative to the no-geography condition. A separate municipality holdout conducted among pre-2024 transactions yielded **R² 0.76503** on 14,259 records. The experimental populations and temporal structures differ; these outcomes cannot be interpreted as direct evidence of future-market forecasting.

Errors vary across sale-price segments. In the below-$150,000 group, the median absolute percentage error was **36.45%**, whereas transactions at or above $750,000 had a mean absolute error of **$218,529.69**. Absolute errors do not establish the direction of bias. [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) reports all five sale-price groups and sample sizes.

## 5. Spatial Evidence Indicator

For a location in projected kilometre coordinates, the indicator measures distance to its **tenth-nearest development-location record**. Two tercile thresholds of the 2024 validation distance distribution define dense, typical and sparse sample-coverage categories. A pre-2024 historical calibration model produces validation residual summaries within those categories, including MAE, RMSE and 80th-percentile absolute error. The final saved XGBoost model is audited separately against the 2025 held-out outcomes using the previously established thresholds.

| Tier | 2025 observations | Original-model MAE (USD) |
|---|---:|---:|
| Dense | 3,559 | 54,145.08 |
| Typical | 2,669 | 53,251.12 |
| Sparse | 2,463 | 69,389.57 |

The sparse-minus-dense difference was **$15,244.49**, approximately 28.2% of dense MAE. The validation Spearman coefficient between tenth-neighbor distance and absolute error was only **0.09687**. The comparison is a group-level diagnostic rather than a calibrated individual prediction interval. A composition-adjusted or geographically clustered uncertainty estimate has not been established by the published original-model evidence.

## 6. Multivariable calculus and geographical scenarios

Let a trained price predictor be written as \(f(x,y,\mathbf{z})\), where \((x,y)\) denotes location and \(\mathbf{z}\) represents other predictors. Fixed reference features \(\mathbf{z}_0\) define a two-dimensional model scenario \(P(x,y)=f(x,y,\mathbf{z}_0)\). Coordinates are transformed to **EPSG:26917** and converted from metres to kilometres before sampling an 86×86 grid. Only nodes within 4.5 km of an observed parcel position are supported.

The tree ensemble is piecewise constant. Therefore, a masked Gaussian-smoothed surface \(S(x,y)\) is used for finite-difference calculus. The spatial gradient is \(\nabla S=(\partial S/\partial x,\partial S/\partial y)\), expressed in **nominal USD/km**. The directional derivative is the gradient's dot product with a unit projected direction. The double integral approximates \(\iint_D S\,dA\) over valid computational grid cells, and division by the supported area yields an area-weighted scenario mean.

The original-model analysis identified **6,117 supported grid nodes**, **5,945 integration cells** and an area-weighted modeled scenario mean of **$228,831.68**. None of these quantities represents total property wealth or the observed county sale-price mean. The location of strong gradient bands depends on choices such as smoothing and on fixing municipality and ZIP while coordinates move. At Gaussian σ=1.0, the mean magnitude was approximately **$2,092.49/km**, compared with **$1,690.37/km** at σ=2.5. Stability under multiple geographically coherent reference profiles has not been established.

## 7. Software and reproducibility

The `geovalue-official` CLI exposes source retrieval, dataset construction, profiling, model training, prediction, outlier/transaction diagnostics and spatial analysis. FastAPI provides validated property and gradient endpoints. The primary browser interface uses locally bundled Plotly and includes overview, estimate, geography, calculus and methodology views. The California census-area study remains available separately under `/legacy`.

Automated tests cover data validation and joins, split isolation, preprocessing, numerical finite differences and integration, support calibration and HTTP behavior. GitHub Actions tests are designed to run without county raw CSVs or a saved trained model, using controlled synthetic fixtures. The empirical results above come instead from full official-data evaluation and the original saved-model verification. See [TESTING.md](TESTING.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 8. Limitations and conclusions

GeoValue provides a documented association study across recorded property-sale prices and a controlled spatial interpretation layer. The assessment data may postdate transactions, and the evaluation does not establish performance under true historical information availability. Sale prices were not inflation-adjusted; county-valid sales are a selected sample. Local sample density does not provide calibrated individual error bounds, and the numerical surface holds some geographically dependent categorical features constant. The integration domain is a supported grid rather than the legal county boundary.

Under these constraints, the principal conclusion is that geographical features materially contributed to retrospective validation performance, XGBoost had the lowest RMSE among the tested architectures, and sparse training-location coverage corresponded to higher groupwise 2025 errors. The spatial surface permits numerical MVC analysis but cannot support claims about causal price changes.

## References

[Official data and scholarly references](REFERENCES.md) · [Data and model schema](SCHEMA.md) · [Experimental results](EXPERIMENT_RESULTS.md)
