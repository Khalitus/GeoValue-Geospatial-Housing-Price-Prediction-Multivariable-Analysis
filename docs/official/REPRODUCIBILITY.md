# Experiment reproducibility and provenance

## Study identity

The primary GeoValue experiment uses Allegheny County recorded sales from 2013–2025 combined with an assessment extract retrieved in November 2026 and parcel centroids. The input file identities are recorded as SHA-256 digests, and the selected XGBoost pipeline was evaluated against a parcel-disjoint 2025 test partition. The original training report and the later saved-model verification agree to numerical precision.

The public repository contains the research code, the official source identifiers, publication graphics, a model verification record and tests. The full raw county inputs and the original serialized trained pipeline remain separate local artifacts and are **not represented as publicly downloadable files**.

## Evidence chain

1. County resource metadata and download hashes are stored locally in `data/official/raw/source-manifest.json`.
2. Eligibility and parcel joining produce `residential_sales_2013_2025.csv` and a JSON build report.
3. A chronological, parcel-disjoint split supplies training (2013–2023), validation (2024) and held-out test (2025) cohorts.
4. The candidate with lowest 2024 validation RMSE is refitted on the 2013–2024 development partition and stored as a Joblib pipeline.
5. Predictions against all 8,691 held-out 2025 observations reproduce the reported MAE, RMSE and R².
6. The local geographical analysis generates a numerical surface, gradient samples, support-calibration report and 2025 audit. Scientific figures are exported from these data and labeled by their actual interpretation.

The public [verification record](../../poster/evidence/original_model_verification.json) lists the SHA-256 digests of the processed dataset, model, training report, reliability report, geospatial report and numerical surface. Those hashes establish identities of the inputs used in the recorded run; they cannot substitute for third-party access to the original bytes.

## Reference workflow

```bash
geovalue-official download
geovalue-official build
geovalue-official train --seed 42
geovalue-official analyze --grid-size 86 --sigma 1.5 --support-km 4.5
python poster/scripts/generate_original_figures.py --sensitivity
python -m pytest -q
```

The scripts use paths documented in [USER_GUIDE.md](USER_GUIDE.md). Repeating a study against a later county extract may change counts or quality labels; comparison requires matching source hashes. Regenerating figures with a new model is a different model experiment and must be labeled accordingly.

## Saved results

| Result | Source |
|---|---|
| 2024 validation candidates and geographical ablation | Original training report |
| 2025 test metrics and price-bracket errors | Original saved pipeline and held-out records |
| 2024 spatial-support calibration | Separate model fitted on pre-2024 records |
| 2025 spatial-support audit | Saved final XGBoost pipeline |
| Scenario, gradients and area-weighted integration | Saved final pipeline, fixed reference features and smoothed supported grid |
| Raw 2025 transaction map | Recorded official test transactions |

The 2025 support audit is independent of 2024 tier selection in the sense that 2025 residuals do not define support thresholds. This does not imply that the indicator is a statistically calibrated confidence measure.

## Limitations of reproduction

Current assessments may contain property characteristics acquired after historical sales. The fixed-feature scenario also holds municipality and ZIP constant while latitude and longitude vary, which may create geographically inconsistent input combinations. These limitations are part of the design and should remain explicit in downstream analyses. The poster [scientific figure catalogue](FIGURES.md) identifies which illustrations represent observed records, tested predictions and modeled derivatives.
