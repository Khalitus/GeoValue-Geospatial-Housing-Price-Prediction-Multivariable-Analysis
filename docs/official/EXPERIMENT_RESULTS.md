# Official property-sale experiment results

**Recorded experiment:** User-supplied `training-report.json` generated using WPRDC / Allegheny County transaction, assessment and centroid records. Source download occurred **14 November 2026**; the chosen transaction period ends **31 December 2025**. Do not refer to the assessment snapshot as an as-of-2025 record.

## Dataset filtering

| Audit quantity | Value |
|---|---:|
| Raw recorded sale rows | 508,013 |
| Rejected invalid or uncertain sale description | 407,162 |
| Rejected date/price bounds | 9,972 |
| Eligible sale rows before property/location joins | 90,851 |
| Residential assessments | 523,752 |
| Sales matched to assessment | 89,717 |
| Matched to centroid | 89,711 |
| Rejected physical/temporal inconsistencies | 7,542 |
| **Final transactions** | **82,169** |
| **Unique parcels** | **72,607** |

The source tables contain many more parcels than eligible *valid sales*. The final regression count, not the county's total parcel count, is the relevant sample size.

## Chronological evaluation

Training-period **2013–2023**: 60,767 records; **2024** validation: 8,656; **2025** held-out test: 8,691. The split excludes **4,055** repeated-parcel observations to avoid parcel identity crossing partitions. The selected architecture is refitted on **69,423** development rows before 2025 testing.

### 2024 validation comparison

| Algorithm | Validation MAE (USD) | Validation RMSE (USD) | Validation R² |
|---|---:|---:|---:|
| Linear Regression | $71,249.66 | $956,526.74 | -14.1316 |
| Random Forest | $63,044.69 | $111,735.19 | 0.7935 |
| **XGBoost** | **$57,551.40** | **$98,697.19** | **0.8389** |

The unregularized linear model has an extreme RMSE outlier; it should not be treated as a typical predictor. The `diagnose` command provides a reproducible validation-only anomaly report. This comparison is **not** a claim of globally optimal hyperparameters.

### 2025 held-out test

| Metric | Value |
|---|---:|
| Selected architecture | XGBoost |
| MAE | **$58,190.78** |
| RMSE | **$101,555.74** |
| R² | **0.84120** |

Errors are nominal US dollars, not inflation-adjusted. The model predicts recorded residential sale prices, not current asking prices.

## Geographical-feature ablation

Using the selected XGBoost model and unchanged 2024 validation partition:

| Condition | RMSE (USD) | R² |
|---|---:|---:|
| With longitude, latitude, municipality and ZIP | $98,697.19 | 0.83890 |
| Without those four location variables | $116,653.97 | 0.77494 |

**RMSE difference:** $17,956.78, or approximately **15.39%** relative to the no-geography model. This is a descriptive feature-ablation result; it does **not** identify causal effects of location.

## Municipality generalization experiment

A separate *pre-2024* municipality holdout yielded **MAE $49,961.24**, **RMSE $95,885.20**, and **R² 0.76503** across **14,259** held-out records. This is not a future-2025 test and is not the same comparison as the chronological test above.

## Spatial Evidence Indicator (Task 7)

The Task 7 code calibrates support tiers on 2024 validation data and independently audits 2025 test error across the same distance boundaries. The authoritative new results are written to:

- `reports/official/reliability-report.json`
- `reports/official/geospatial-report.json`
- `reports/official/transaction-audit.json`

The attached user files do **not** yet contain the author's Task 7 analysis outputs. Therefore this document deliberately does not claim specific final density thresholds, tier-wise errors or calculus integral values. They should be recorded after `geovalue-official analyze` is run with the trusted, previously trained model on the developer's Mac.

A developer **reproduction check** using the uploaded 82,169-row file and a freshly fitted XGBoost architecture (under different installed library versions) generated 8,656 calibration observations and 6,117 supported surface grid nodes, but those figures are verification evidence, *not a substitute for the author's original saved-model report*.

## Important limitations

1. Assessment attributes are a newer property snapshot and may reflect changes made after the historical sale date. The chrono split does not eliminate this temporal information mismatch.
2. Repeated sales of one parcel are removed across partitions, but the presence of transaction corrections or multiple instruments within a day remains a data-quality issue.
3. County-valid recorded sales are a selected sample of all properties and all transfers.
4. Rare categories, extreme property ratios and luxury transactions can challenge especially the linear baseline.
5. Geographic support does not guarantee low error; the indicator is not a confidence interval.
6. The modeled spatial surface fixes non-location features, including location-dependent categories; its integral is over grid-supported cells rather than Allegheny County's land boundary.

See [Methodology](METHODOLOGY.md) and [Testing](TESTING.md).
