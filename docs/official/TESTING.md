# Software and experimental verification

GeoValue separates software correctness, original-model verification and empirical predictive validity. Automated tests use small controlled fixtures to check source processing and mathematical behavior; the published model scores refer to a full evaluation on historical county transactions.

## Automated software checks

| Component | Tested behavior |
|---|---|
| County ingestion | Required source fields, parcel normalization, joins and valid-sale filtering |
| Feature pipeline | Engineered fields, missing values, unknown categories and persisted estimators |
| Experimental design | Chronology, parcel-disjoint splits, validation selection and geography ablation |
| Numerical calculus | Projected coordinate round trips, finite differences, mask handling and integration |
| Spatial evidence | Nearest-neighbor support distances, threshold tiers and historical group errors |
| Web application | Input validation, API error codes, prediction and gradient responses |
| Legacy compatibility | California CLI and `/legacy` route |

For the test field \(S(x,y)=2x+3y+100\), the expected partial derivatives are 2 and 3 and the gradient norm is \(\sqrt{13}\). Other fixtures test known numerical integrals, support-mask exclusion, and preservation of parcel IDs during joins.

The completed local run recorded **122 tests passed and one optional real-data test skipped**. A GitHub Actions run against repository commit `0972e19` completed successfully. These results describe their respective code revisions; CI executes again when repository content changes.

```bash
python -m pytest -q
python -m pytest --cov=geovalue --cov-report=term-missing
```

GitHub CI does not require the original large county CSV files or the private trained model. A successful CI run is therefore a software test, not a claim of independent empirical model replication.

## Saved-model verification

The original saved XGBoost pipeline was reapplied to all **8,691 parcel-disjoint 2025 observations**. The recomputed metrics (MAE **$58,190.7843**, RMSE **$101,555.7448**, R² **0.84119967**) match the training report to numerical precision. The [verification manifest](../../poster/evidence/original_model_verification.json) records the model and dataset SHA-256 hashes, metrics, cohorts, spatial support and numerical surface summaries.

The original Joblib artifact is not publicly distributed. The manifest is an internal consistency and provenance record, not a claim of third-party replication with independently acquired original inputs.

The raw-data audit identified eight conflicting same-parcel/same-sale-date price groups. They were reported without automatic adjudication. A separate baseline diagnostic traces the unusually high Linear Regression validation RMSE to a large floor-area-ratio extrapolation.

## Interface verification

The local browser was checked at desktop and mobile viewport sizes with navigation, property-form submission, gradient queries, static chart loading and invalid-input behavior. Documentation screenshots illustrate the interface. Some screenshots used a separate engineering fixture; research figures and numerical model claims instead use the [original-model plot set](../../poster/figures/).

## Validity boundaries

Numerical unit tests do not establish the causal interpretation of geographical price gradients. A chronological, parcel-disjoint design does not remove historical information mismatch from a later assessment snapshot. County-valid sales form a selected set of transfers, and subgroup error comparisons are descriptive. The Spatial Evidence Indicator does not yield calibrated individual prediction intervals. See [feature validity](FEATURE_VALIDITY.md) and [experimental results](EXPERIMENT_RESULTS.md).
