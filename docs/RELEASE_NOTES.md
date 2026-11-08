# Final local release notes

**Project:** GeoValue — Geospatial Housing Price Prediction and Multivariable Analysis  
**Repository:** `Khalitus/GeoValue-Geospatial-Housing-Price-Prediction-Multivariable-Analysis`  
**Documentation closeout:** 8 November 2026  
**Reviewed implementation commit:** `5a855bf24f15b102ee5ad948943928dc8737d0f1`  
**Python package version at review:** `0.4.1`

## Delivery scope

- Local CLI and FastAPI dashboard with a responsive HTML/CSS/JavaScript interface.
- California Housing Prices CSV validation, exploratory summaries and three regression models.
- Reproducible splitting, real-data model-evaluation report and saved XGBoost pipeline.
- Observed-versus-predicted geography, fixed-feature smoothed price surface, contours, numerical partial/directional derivatives, gradient field and double integral.
- A passing GitHub Actions pytest workflow (67 tests) and documented synthetic stress/browser checks.
- A documented local installation, operation guide, architecture, scientific methodology and academic report.

## Cancelled, not delivered

- **Vercel production deployment:** dropped in favor of a simpler local-first scope after the remote Python build exceeded platform function-size limits. Files left over from that attempt are legacy conveniences, not evidence of a live app.
- **Fully offline browser charts:** Plotly still loads via an external CDN.
- **Spatial holdout evaluation:** not performed; the reported R² and error values are for the income-stratified random split only.
- **Current-market property appraisals:** not part of the dataset or model.

## Known constraints

The dataset dates from 1990, targets censal-area medians, and upper-codes higher values. The projected scenario holds non-geographic features fixed and is smoothed before differentiation. Gradient magnitudes vary with grid resolution, smoothing and masking. Integration is over a computationally supported grid, not an official administrative boundary. Model artifacts must be trusted to load safely.

## Commit and review provenance

- Actual model figures: `public/data/training-report.json`.
- Actual mathematical summary: `public/data/geospatial-report.json`.
- GitHub Actions: https://github.com/Khalitus/GeoValue-Geospatial-Housing-Price-Prediction-Multivariable-Analysis/actions/runs/42475421269
- Audit details: [TESTING.md](TESTING.md).
- Figure attribution: [FIGURES.md](FIGURES.md).

The documentation package intentionally contains Markdown and PNG artifacts only. It does **not** overwrite the source code, original CSV, trusted model, generated reports or Git history when extracted into the existing project folder.
