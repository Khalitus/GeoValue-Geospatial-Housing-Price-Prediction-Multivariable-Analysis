# Experimental results

The findings below refer to recorded residential sales in Allegheny County and the original fitted GeoValue XGBoost pipeline. The source extract was retrieved on **14 November 2026**. The sale period ends on 31 December 2025; its property-assessment fields are from the later extract and are not asserted to be historical snapshots.

Machine-readable evidence: [original-model verification](../../poster/evidence/original_model_verification.json), [figure-generation code](../../poster/scripts/generate_original_figures.py), and [scientific figures](../../poster/figures/). The verification record contains SHA-256 hashes of the processed dataset, fitted model, training report, numerical surface and reliability report. The private source CSVs and serialized model are not distributed in this repository.

## Sample construction

| Processing measure | Count |
|---|---:|
| Raw sale records | 508,013 |
| Sale-description exclusions | 407,162 |
| Date and price exclusions | 9,972 |
| Eligible records before joins | 90,851 |
| Residential assessment records in source | 523,752 |
| Matched assessment records | 89,717 |
| Matched parcel centroids | 89,711 |
| Physical or temporal consistency exclusions | 7,542 |
| **Eligible transactions after processing** | **82,169** |
| **Distinct parcels** | **72,607** |

Twenty-eight exact duplicate sale rows were removed during the source-cleaning sequence before the main eligibility and join accounting. Counts in this table describe successive processing checkpoints; they are **not** independent rejection categories to be added together. The source includes repeated sales of some parcels.

## Chronological and parcel-disjoint evaluation

| Partition | Sale years | Observations |
|---|---|---:|
| Training | 2013–2023 | 60,767 |
| Validation | 2024 | 8,656 |
| Held-out test | 2025 | 8,691 |

The partitioning procedure excludes 4,055 earlier observations to ensure no parcel identifier appears in more than one partition. The chosen XGBoost architecture is refitted on 69,423 development observations (training plus validation) before evaluation against the 2025 test set. All performance measures use recorded **nominal USD** prices.

### 2024 validation model selection

| Candidate model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Linear Regression | 71,249.66 | 956,526.74 | -14.1316 |
| Random Forest | 63,044.69 | 111,735.19 | 0.7935 |
| **XGBoost** | **57,551.40** | **98,697.19** | **0.8389** |

The very large Linear Regression RMSE reflects an unstable extrapolation associated with an anomalous floor-area ratio. The comparison is among the specified candidate configurations, not a global algorithm search. The original training procedure selected XGBoost by **validation RMSE**, without consulting 2025 test targets.

### 2025 held-out test

| Metric | Reported | Recomputed from saved XGBoost model |
|---|---:|---:|
| Observations | 8,691 | 8,691 |
| MAE (USD) | 58,190.78 | 58,190.7843 |
| RMSE (USD) | 101,555.74 | 101,555.7448 |
| R² | 0.84120 | 0.84119967 |

The recomputed results agree with the original training report to numerical precision. This establishes internal artifact consistency; public reproduction remains limited by the fact that the fitted model and underlying private data files are not hosted in GitHub.

## Contribution of geographical features

| XGBoost configuration | 2024 validation RMSE (USD) | R² |
|---|---:|---:|
| With longitude, latitude, municipality and ZIP | 98,697.19 | 0.83890 |
| Without these four fields | 116,653.97 | 0.77494 |

The observed RMSE reduction is **$17,956.78**, or **15.39% relative to the model without geographic inputs**. This experiment measures predictive association, not the causal value of changing a property's location.

A separate municipality holdout among pre-2024 observations assessed the model's transfer to municipalities excluded from that experiment's fitting sample: **14,259 observations**, MAE **$49,961.24**, RMSE **$95,885.20**, R² **0.76503**. Its population and split design differ from the chronological 2025 test; the metrics are not interchangeable.

## Spatial Evidence Indicator

The **2024 calibration** computes ten-nearest-location support distances and error summaries using a model trained on pre-2024 observations. The **2025 audit** uses the saved final XGBoost pipeline, with the previously defined distance thresholds. The two model identities are deliberately distinct.

| Geographical support | 2024 records | 2024 MAE (USD) | 2025 records | 2025 MAE (USD) | 2025 RMSE (USD) |
|---|---:|---:|---:|---:|---:|
| Dense | 2,886 | 51,819.50 | 3,559 | 54,145.08 | 85,131.29 |
| Typical | 2,885 | 53,349.62 | 2,669 | 53,251.12 | 89,158.94 |
| Sparse | 2,885 | 67,487.07 | 2,463 | 69,389.57 | 131,553.13 |

The 2025 sparse-minus-dense MAE difference is **$15,244.49** (28.16% of the dense MAE). The 2024 distance-error Spearman correlation is **ρ = 0.09687**, a weak individual association. The dense and typical groups have similar audit errors. The support categories should therefore be interpreted as **descriptive sample coverage**, not as monotonic individual uncertainty estimates or calibrated prediction intervals.

The 2024 distance-tercile boundaries were **0.13863 km** and **0.19986 km**. The 80th percentile of 2025 absolute errors was **$76,375.56** in dense areas and **$94,635.09** in sparse areas. These are group summaries; they do not provide coverage probabilities for a new property.

## Error by recorded sale-price bracket

| 2025 recorded price | Records | MAE (USD) | Median absolute percentage error |
|---|---:|---:|---:|
| Below $150,000 | 1,098 | 47,425.75 | 36.45% |
| $150,000–$299,999 | 3,774 | 33,867.49 | 12.11% |
| $300,000–$499,999 | 2,413 | 53,204.69 | 11.86% |
| $500,000–$749,999 | 884 | 94,334.09 | 12.62% |
| $750,000 and above | 522 | 218,529.69 | 17.39% |

Lower-priced sales exhibit the largest median proportional errors; high-priced sales have the largest mean dollar errors. These are **absolute**, not signed, residual statistics. Systematic over- or underprediction has not been established by the published table.

## Numerical geographical scenario

| Parameter or result | Value |
|---|---:|
| Projection | NAD83 / UTM zone 17N (EPSG:26917) |
| Candidate grid | 86 × 86 |
| Supported grid nodes | 6,117 |
| Gaussian smoothing | σ = 1.5 grid cells |
| Maximum support distance | 4.5 km |
| Integration cells | 5,945 |
| Supported computational area | 2,331.44 km² |
| Area integral | 533,507,923.29 USD·km² |
| Area-weighted scenario mean | 228,831.68 USD |

The integrated region is a data-supported numerical grid, not Allegheny County's official boundary. The area-weighted mean is a **model scenario statistic**, not the observed county sale-price mean or a sum of property values. A strong vertical gradient band is present under the fixed-reference scenario; its geographical stability has not been established under coherent category assignments and multiple reference-property profiles.

The mean modeled gradient magnitude changes with smoothing: **$2,092.49/km at σ = 1.0**, **$1,922.49/km at σ = 1.5** and **$1,690.37/km at σ = 2.5**. These figures quantify numerical sensitivity, not causal geographical price appreciation.

## Data-quality and validity notes

Eight parcel-and-sale-day groups containing contradictory prices were identified by an audit. The source records were flagged without automatically selecting a preferred price. The assessment snapshot may contain post-sale changes to floor area, rooms, conditions or amenities. The evaluation also uses nominal transaction prices across years. These constraints limit interpretation as historical point-in-time or prospective valuation.

See [feature timing and validity](FEATURE_VALIDITY.md), [methodology](METHODOLOGY.md) and [figure provenance](FIGURES.md).
