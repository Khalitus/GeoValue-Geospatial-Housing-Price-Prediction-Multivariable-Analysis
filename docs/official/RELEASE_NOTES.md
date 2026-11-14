# GeoValue 0.7.0 — Local official-data research release

## Included

- Official Allegheny County dataset ingestion, documented sample-size filtering and source hashes (Task 6).
- Property-level XGBoost model trained and evaluated using 2024 validation and 2025 held-out sales (Task 6).
- Pennsylvania projected price scenario (`EPSG:26917`) and numerical partial derivatives, gradients, 3D/2D contour charts and double integration (Task 7).
- Spatial Evidence Indicator from pre-2024 training-site density and 2024 residual statistics, with 2025 post-calibration audit (Task 7).
- Flag-only same-day conflicting-sale audit and linear-baseline outlier diagnostic (Task 7).
- Local responsive browser with offline Plotly bundle and preserved `/legacy` California view (Task 7).
- Synthetic numerical tests, API tests and complete local integration with actual official processed records.

## Local distribution

Use a **merge-only update**, not a directory replacement. User datasets, downloaded official CSV files, source manifest, trained artifacts and machine-generated reports must remain on the original Mac. The source ZIP does not and cannot include the user's trusted previously trained Joblib artifact unless the user explicitly exports it.

## Scientific boundary

The spatial indicator is not a confidence interval; the surface is not raw XGBoost or a land-polygon integral; current assessment data can postdate historical sales. Official Task 7 report values must be generated from the trusted saved model on the user's computer before being quoted in an academic defense.

## Scope decision

GeoValue remains **local-first**. Vercel deployment is abandoned. Existing legacy deployment configuration files are not required for normal operation and may be removed as a separate cleanup commit if desired; the Task 7 update does not depend on them.
