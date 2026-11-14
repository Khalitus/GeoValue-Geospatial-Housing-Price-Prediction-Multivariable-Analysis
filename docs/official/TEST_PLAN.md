# Task 6 verification and remaining scientific validation

## Automated suite

`python -m pytest -q` includes the legacy project's tests and the new official tests under `tests/official/`. New tests cover: source joins, invalid sale labels, preservation of 16-character parcel IDs, missing/invalid schema, date and price screening, 100,000+ row chunked sales ingestion, impossible construction dates, engineered ratios, three regression families, serialization, unknown categories, parcel-disjoint chronology, 2024 selection, 2025 test isolation, CLI output and both local API missing-model and successful-prediction paths.

The test records are **synthetically generated**. Passing tests means the code behaved as specified on those fixtures; it does not measure real Allegheny County training accuracy or the actual number of eligible records. Performance claims for Task 6 require running the full real-source pipeline and publishing `reports/official/build-report.json` and `reports/official/training-report.json`.

## Manual acceptance

1. Download the three official resources; verify the manifest and file sizes.
2. Run `geovalue-official build` and verify that all years through 2025 are represented, that the output has substantially more eligible training sales than the old 20,640-row source, and that sale-validity rejection counts are plausible. Do **not** alter filtering merely to inflate row counts.
3. Open the generated audit JSON and verify that the final `unique_parcels` count cannot exceed the final transaction count.
4. Run `geovalue-official profile`; inspect medians, missing features, and per-year sample size.
5. Run a small training smoke test, then the real full-capacity or explicitly capped experiment; record configuration and metrics.
6. Validate 2025 holdout results, municipal holdout metrics (if available) and geographical ablation. Do not quote previous California model accuracy as official-data accuracy.
7. Verify CLI and `/api/official/predict` against a locally created trusted official model. Old California GUI is unchanged until Task 7.

## Limitations to report

- The source assessment table is a current snapshot even for older sale dates, and property condition or area may have changed; this is a potential point-in-time feature leak.
- Sales may be recorded late or revised and validation codes can change.
- Classification by `SALEDESC=VALID SALE` is conservative and may omit some legitimate market transactions.
- Unadjusted USD across 2013–2025 reflects inflation and changing markets.
- Distinct parcels across chronological partitions reduce repeated-property leakage but do not remove spatial correlation or snapshot drift.
- A holdout of entire municipalities is exploratory and different from an official prospective market generalization test.
