# GeoValue research poster and figures

The poster presents the GeoValue retrospective Allegheny County sale-price study, including its original-model evaluation, observed sale records and numerical spatial sensitivity analysis.

## Publication assets

- [Research poster, PDF](GeoValue_Research_Poster.pdf)
- [Research poster, editable PowerPoint](GeoValue_Research_Poster.pptx)
- [Original-model evidence manifest](evidence/original_model_verification.json)
- [Research figures](figures/)

The poster is a presentation of the documented study. It is not an additional model-training experiment. Its scientific figures and results are traceable to the recorded original fitted-model run; the 2024 support-calibration fit is explicitly separate from the 2025 final-model audit.

## Scientific figures

| File prefix | Source and subject |
|---|---|
| `01_observed_2025_sales` | Recorded 2025 transaction prices and density by hexagon |
| `02_original_model_surface_and_gradient` | Smoothed fixed-feature original-model scenario and partial derivatives |
| `03_original_2025_actual_vs_predicted` | Original-model prediction pairs on 8,691 held-out 2025 sales |
| `04_original_2025_price_bracket_errors` | Absolute and proportional errors grouped by recorded price |
| `05_original_model_spatial_support_errors` | 2024 historical calibration and 2025 original-model support audit |
| `06_reported_validation_comparisons` | Model selection and geographic-feature ablation |
| `07_original_model_gradient_sensitivity` | Numerical sensitivity to smoothing bandwidth |

Each figure is provided in a raster PNG and vector PDF format. The plotting definitions are retained in [`scripts/generate_original_figures.py`](scripts/generate_original_figures.py). The JSON evidence record contains the SHA-256 digests of the original processed data, fitted model and analysis reports, together with recomputed 2025 performance and spatial results.

## Interpretation

Observed transaction maps and model-derived gradients represent different objects. The gradients are numerical derivatives of a smoothed fixed-property scenario, not directly observed appreciation or a causal geographical effect. Local training-data coverage is descriptive and does not establish calibrated individual prediction confidence. Historical assessment fields may postdate older transactions. Full explanations appear in [methodology](../docs/official/METHODOLOGY.md), [experimental results](../docs/official/EXPERIMENT_RESULTS.md) and [feature validity](../docs/official/FEATURE_VALIDITY.md).
