# Figure catalogue and provenance

Scientific results and interface illustrations have different evidentiary roles. The published poster figures under [`poster/figures/`](../../poster/figures/) are derived from the original-model verification run recorded in [`poster/evidence/original_model_verification.json`](../../poster/evidence/original_model_verification.json). The screenshots under `docs/official/images/` document browser layout and interaction; their displayed record-level examples should not be used as proof of the original fitted model's numeric output.

## Original-model research figures

| Figure | File | Interpretation |
|---|---|---|
| Observed sales | [`01_observed_2025_sales`](../../poster/figures/01_observed_2025_sales.pdf) | Hexagon-aggregated 2025 recorded prices and sale counts |
| Surface and gradient | [`02_original_model_surface_and_gradient`](../../poster/figures/02_original_model_surface_and_gradient.pdf) | Fixed-property smoothed model field and numerical derivative; not observed appreciation |
| Prediction diagnostic | [`03_original_2025_actual_vs_predicted`](../../poster/figures/03_original_2025_actual_vs_predicted.pdf) | All 8,691 2025 test pairs |
| Price-stratum errors | [`04_original_2025_price_bracket_errors`](../../poster/figures/04_original_2025_price_bracket_errors.pdf) | MAE and median absolute percentage error for observed sale-price groups |
| Geographic support | [`05_original_model_spatial_support_errors`](../../poster/figures/05_original_model_spatial_support_errors.pdf) | Separate 2024 historical calibration and 2025 final-model audit |
| Validation comparisons | [`06_reported_validation_comparisons`](../../poster/figures/06_reported_validation_comparisons.pdf) | Model candidates and geographic-feature ablation |
| Gradient smoothing | [`07_original_model_gradient_sensitivity`](../../poster/figures/07_original_model_gradient_sensitivity.pdf) | Scenario derivative variation with Gaussian bandwidth |

Each PDF also has a PNG counterpart. The script [`generate_original_figures.py`](../../poster/scripts/generate_original_figures.py) contains the plotting definitions and verifies agreement between the saved original model and its reported 2025 scores. The source-matching inputs are identified by hashes in the evidence record. The 2024 support calibration is fitted on pre-2024 records; the 2025 audit uses the final saved XGBoost model.

## Interface screenshots

| View | Desktop | Mobile |
|---|---|---|
| Overview | [desktop](images/overview-desktop.png) | [mobile](images/overview-mobile.png) |
| Estimate | [desktop](images/estimate-desktop.png) | [mobile](images/estimate-mobile.png) |
| Geography | [desktop](images/geography-desktop.png) | [mobile](images/geography-mobile.png) |
| Calculus | [desktop](images/calculus-desktop.png) | [mobile](images/calculus-mobile.png) |
| Method | [desktop](images/method-desktop.png) | [mobile](images/method-mobile.png) |

These images were captured from the working local browser at desktop and mobile viewport sizes using an engineering evaluation fixture fitted independently of the final model. Although the overview displayed the original recorded evaluation metrics, screenshot-specific example prices and surfaces do not replace the published original-model evidence. Figures intended to support quantitative claims are listed in the first table.

## Poster

The completed research poster is available in [PDF](../../poster/GeoValue_Research_Poster.pdf) and [editable PowerPoint](../../poster/GeoValue_Research_Poster.pptx) formats. Poster numeric claims are drawn from the original-model evidence record; the poster remains a concise presentation of findings rather than an additional experiment.
