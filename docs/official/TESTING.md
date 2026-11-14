# GeoValue Task 7 testing and verification

## Automated validation

The complete code-matched Task 7 source passed **121 pytest tests** in the local audit environment, including all 101 preexisting Task 6 tests. The original Task 6 Mac terminal log also recorded **101 passed**; the Mac must run the 123-test update after installation. CI test totals after a new push are not yet claimed.

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

Tests exercise source CSV ingestion, parcel-key joins, sale validity, chronological splits, model comparisons, input validation, existing California routes, new official API routes, projection, price surface, known synthetic gradients/integrals, spatial support thresholds, stored calibration artifacts, missing-asset behavior and numerical report exports.

### New Task 7 mathematical checks

| Test case | Expected property |
|---|---|
| Coordinate round-trip | EPSG:4326 → EPSG:26917 → EPSG:4326 recovers location |
| Nearest-neighbor distances | 2nd-nearest distance matches known numerical geometry |
| Reliability buckets | Threshold boundaries reproducibly assign support tiers |
| Synthetic calibration | Group MAE matches known absolute errors; all samples counted |
| Support artifact | `.npz` contains only numeric arrays and loads with pickle disabled |
| Plane gradient | For S=2x+3y+100, east derivative equals 2 and gradient magnitude √13 |
| Surface integration | Scenario numerical quadrature produces a finite supported-region mean |
| Source conflict audit | Contradictory parcel/date pairs are flagged without data mutation |
| End-to-end official analysis | Generates genuine test-cohort map, report, surface and support outputs |
| API behavior | Proper 422/503 errors, model response and projection-based gradient |
| Backward compatibility | Official UI at `/`, California UI at `/legacy` |

The repository's CI tests work without county data and without real trained artifacts by using synthetic fixtures. The real official dataset itself is not committed to GitHub.

## Actual-data integration test

The uploaded official processed CSV containing **82,169** records was inspected and processed with a **separately refitted XGBoost** architecture in the audit environment. The integration generated a valid geographical surface, support-calibration JSON, 2025 point JSON, and transaction conflict audit. It identified **eight** same-parcel/same-day conflicting-price groups. The original saved user-trained XGBoost Joblib file was **not uploaded**; therefore numerical surface/reliability figures from this separate model must not be silently substituted for the author's original model. Run `geovalue-official analyze` with the trusted local Joblib file on the Mac before using the new Task 7 numbers in academic results.

The original training report and supplied CSV are an appropriate basis for model-selection and temporal holdout claims; the interactive browser screenshots are *engineering verification*, not substitutes for regression evaluation.

## Local browser checks

Headless Chromium was exercised against a FastAPI TestClient of the committed application source at **1440×900 desktop** and **390×844 mobile**, using inline local chart assets and API responses from the audit environment. This avoids an external CDN or public deployment. Tested navigation, prediction submission, gradient form, valid sample locations, chart rendering, no JavaScript errors and no horizontal overflow. The 2025 chart uses a normal scatter plot to avoid mandatory WebGL; the 3D chart uses a contour fallback when WebGL is unavailable.

The screenshots in `images/` display the **real interface against the uploaded county dataset with the audit environment's separately fitted model**. The summary model metrics shown in the browser were loaded from the uploaded original training report, while point/surface values were derived from the audit model. Screenshots are marked as **UI evidence only**, not an independent model accuracy experiment.

## Scientific limits of testing

- Software unit tests cannot prove real-world model accuracy.
- A chronological test reduces some leakage, but contemporary assessment features remain potentially post-sale.
- The municipality holdout is distinct from the 2025 test.
- The spatial evidence indicator needs its actual 2024/2025 audit to establish whether density correlates with error; the code does not guarantee a confidence interval.
- Outlier and contradictory-record diagnostics flag issues but do not justify arbitrary data deletion.
- The local test environment may have different library patch versions than the author's trained model.
- Source updates after download may alter rows, valid-sale labels, assessment attributes and transaction corrections.

## Optional regression checks on the Mac

After running the official analysis, verify:

```bash
python -m json.tool reports/official/geospatial-report.json
python -m json.tool reports/official/reliability-report.json
python -m json.tool reports/official/transaction-audit.json
geovalue-official model-info --model artifacts/official/residential-sale-model.joblib
```

Inspect `validation_error_by_support` and `held_out_2025_support_audit` counts. Tiers must be selected from 2024 data and audited separately on 2025. A sparse location with good predicted performance is not a contradiction; the indicator describes **sample density** and empirical group errors, not deterministic correctness.
