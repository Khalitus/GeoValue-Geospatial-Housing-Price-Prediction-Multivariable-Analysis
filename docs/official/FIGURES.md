# Official local dashboard screenshot catalog

The figures in `docs/official/images/` document the delivered UI at **1440×900 desktop** and **390×844 mobile** viewport sizes. They were created in a local browser against the 82,169-row processed official dataset, using a **separately refitted model in the audit environment**, not the author's original Joblib artifact. The displayed overview metrics were read from the user's recorded full training report. These figures demonstrate layout, navigation and interactive behavior only; they are **not independent model accuracy evidence**.

| Page | Desktop | Mobile | Primary check |
|---|---|---|---|
| Overview | [overview-desktop.png](images/overview-desktop.png) | [overview-mobile.png](images/overview-mobile.png) | Actual provided evaluation summary and model comparison |
| Estimate | [estimate-desktop.png](images/estimate-desktop.png) | [estimate-mobile.png](images/estimate-mobile.png) | Property form, inferred prediction, spatial support display |
| Geography | [geography-desktop.png](images/geography-desktop.png) | [geography-mobile.png](images/geography-mobile.png) | 2025 held-out recorded and predicted sample, contours |
| Calculus | [calculus-desktop.png](images/calculus-desktop.png) | [calculus-mobile.png](images/calculus-mobile.png) | Gradient location, bearing, magnitude and numeric surface |
| Method | [method-desktop.png](images/method-desktop.png) | [method-mobile.png](images/method-mobile.png) | Assumptions, evidence boundaries and limitations |

The original uploaded training report is the source for the authoritative held-out MAE/RMSE/R². When using the project for presentation, regenerate and capture the GUI from the original trusted model on the Mac if screenshot-specific numerical values will be discussed.
