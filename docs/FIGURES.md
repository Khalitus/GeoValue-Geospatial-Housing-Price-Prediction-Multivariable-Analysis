# Figures and screenshot provenance

## Scientific figures — actual recorded experimental results

| Figure | File | Interpretation |
|---|---|---|
| Figure 1 | [Validation RMSE](images/validation-rmse.png) | Three regression models on the real 4,128-row validation partition |
| Figure 2 | [Geographic ablation](images/geography-ablation.png) | Same selected XGBoost model with and without three geographical features |

The values in these charts were transcribed directly from the recorded `public/data/training-report.json` and verified against its numeric fields. Both figures use historical 1990 US dollars.

## Interface screenshots — synthetic browser-test fixture

The following screenshots were captured from the **real committed GeoValue web frontend**, using synthetic 1,100-row geography/model fixtures for the chart data. These images are suitable for a **software implementation or user-interface appendix**, but **not** for asserting that the depicted geographical points or specific example predictions are observations from California's real housing dataset.

| View | Desktop, 1440 px | Mobile, 390 px |
|---|---|---|
| Overview | [overview-desktop.png](images/overview-desktop.png) | [overview-mobile.png](images/overview-mobile.png) |
| Prediction | [prediction-desktop.png](images/prediction-desktop.png) | [prediction-mobile.png](images/prediction-mobile.png) |
| Geography | [geography-desktop.png](images/geography-desktop.png) | [geography-mobile.png](images/geography-mobile.png) |
| Calculus | [calculus-desktop.png](images/calculus-desktop.png) | [calculus-mobile.png](images/calculus-mobile.png) |
| Method | [method-desktop.png](images/method-desktop.png) | [method-mobile.png](images/method-mobile.png) |

[Contact sheet with all five desktop pages](images/desktop-contact-sheet.png)

## Using figures in the university report

- Cite scientific Figures 1 and 2 for measured model comparisons.
- Use screenshots to document the implemented interface, labeling them **"Synthetic fixture; interface verification only"**.
- For actual geographical housing distribution plots in the final report, run `geovalue analyze` on the original Kaggle CSV, then capture the resulting `reports/geospatial/*.html` figures from your own computer. Do not substitute fixture screenshots for those plots.
- All figures should have numbered captions, units, and a source note that distinguishes actual results from demonstrations.
