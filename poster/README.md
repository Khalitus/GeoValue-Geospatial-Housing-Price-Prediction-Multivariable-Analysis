# GeoValue | TechXhibit 8.0 poster evidence

This folder contains poster-ready figures regenerated from the authors' trusted **original fitted model** and their official Allegheny County data. No model retraining is performed by the figure script.

## Running from the repository root

```bash
source "$HOME/.venvs/geovalue/bin/activate"
uv pip install --link-mode copy matplotlib
geovalue-official analyze \
  --data data/official/processed/residential_sales_2013_2025.csv \
  --model artifacts/official/residential-sale-model.joblib \
  --training-report reports/official/training-report.json \
  --grid-size 86 --sigma 1.5 --support-km 4.5 --map-points 1600
python poster/scripts/generate_original_figures.py --sensitivity
```

The `analyze` command regenerates the surface and reliability reports from the trusted original local model. It also **refits a pre-2024 calibration model** only to construct 2024 validation support tiers; the 2025 audit uses the original final model. The figure script itself never fits or changes any model.

## Committed versus local files

- Commit: `poster/scripts/`, `poster/figures/` (`.png` and vector `.pdf`), `poster/evidence/original_model_verification.json`, and this README.
- **Do not commit:** `data/official/raw/`, `data/official/processed/`, the private model under `artifacts/official/`, large numerical intermediate files, or the entire local reports directory.
- The evidence manifest stores SHA-256 **hashes**, not the private source files or fitted model.

## Figures

| Figure | Evidence | Meaning |
|---|---|---|
| `01_observed_2025_sales` | Official 2025 held-out transactions | Median observed sale price and sale counts per hexagon; a hexagon requires at least 8 sales |
| `02_original_model_surface_and_gradient` | Original model + `price-surface.npz` | Fixed-feature, Gaussian-smoothed scenario and its numerical gradient; **not observed prices** |
| `03_original_2025_actual_vs_predicted` | Original model predictions for every 2025 test transaction | All 8,691 points on symmetric-log axes, including any negative predictions |
| `04_original_2025_price_bracket_errors` | Original model predictions against 2025 actuals | MAE and median absolute percentage error by recorded-price bracket |
| `05_original_model_spatial_support_errors` | Original-model Task 7 evidence report | 2024 support calibration errors and independent 2025 audit errors |
| `06_reported_validation_comparisons` | Original 2024 training report | Candidate-model and geography-feature ablation RMSE |
| `07_original_model_gradient_sensitivity` | Optional original model re-evaluation | Gradient magnitude at sigma 1.0, 1.5, and 2.5; changing smoothing affects magnitudes |

## Verification and limitations

`poster/evidence/original_model_verification.json` records the model and data SHA-256 hashes, software versions, and *recomputed* 2025 MAE/RMSE/R². The figure generator **aborts** if these metrics differ materially from the original training report or if the surface/reliability cohort sizes are inconsistent.

Original analysis and poster figures have two distinct types of evidence:

1. Recorded property sale prices and their official 2025 spatial distribution are observations.
2. Gradient maps describe a numerically smoothed *model scenario* where non-spatial property characteristics, including municipality and ZIP, are held fixed. They are not observed geographical price gradients, prospective appreciation rates, or causal effects.

The official dataset is retrospective and recent property assessment snapshots can include characteristics acquired after historical sale dates. The Spatial Evidence Indicator is descriptive, not a calibrated confidence interval.

### Updating the poster PDF/PPTX

Import the generated PNG or PDF figures into the editable poster, replacing the earlier **independent-model** graphics. Use the numbers from `original_model_verification.json`, not the independently refitted poster package. Do not blindly reuse supplementary figures until their original-model equivalents have been generated and verified.
