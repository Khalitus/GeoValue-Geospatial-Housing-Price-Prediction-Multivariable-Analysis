# Temporal feature validity

## Evidence boundary

The study joins recorded residential transactions from **2013–2025** with an assessment extract retrieved in **November 2026** and parcel coordinates from a 2025 centroid source. It is a retrospective association and prediction study. The available sources do not establish that all model inputs describe each property as it existed on the date of its earlier sale.

The evaluation partitions are chronological and parcel-disjoint. That design prevents the same parcel from appearing across training, validation and test partitions, but it does not create historical feature snapshots. Neither the high 2025 R² nor the spatial-support audit resolves this limitation.

## Feature-level provenance

| Predictor group | Fields | Available timing evidence | Main risk |
|---|---|---|---|
| Recorded transaction date | `sale_year`, `sale_month` | Derived from `SALEDATE` in the transaction record | Later record corrections remain possible |
| Position | `latitude`, `longitude` | Parcel centroid extract dated 2025 | Parcel boundaries or identifiers may have changed since an earlier sale |
| Local geography | `municipality`, `zip_code` | Assessment extract and parcel context | Boundaries, labels or parcel attribution may differ historically |
| Building size and layout | `living_area_sqft`, `total_rooms`, `bedrooms`, `full_baths`, `half_baths`, `stories` | November 2026 assessment snapshot | Renovations or additions after sale can alter recorded features |
| Parcel area | `lot_area_sqft` | November 2026 assessment snapshot | Subdivision, parcel consolidation and recording corrections |
| Construction and amenities | `year_built`, `fireplaces`, `garage_spaces`, `heating_cooling` | November 2026 assessment snapshot | Later renovation or changes in property amenities |
| Assessment condition and grade | `condition_code`, `construction_grade` | November 2026 assessment snapshot | Rating updates after historical transactions |
| Derived attributes | Property age at sale, bathroom equivalents, floor-area ratio | Calculated from the input fields above | Inherits temporal mismatch from its source fields |

The prediction target, `sale_price`, is recorded in the historical sale resource, but transaction qualification (`VALID SALE`) and processing rules may themselves be revised by the publisher.

## Interpretation and sensitivity

The principal result answers how well a specified feature representation can explain **historically recorded sale prices under the available assessment snapshot**. It does not establish the predictive error that would have been achieved using only information known at each transaction date.

A timing-sensitive experiment would require (i) archived assessment values effective at each sale date, or (ii) a controlled exclusion or replacement of attributes likely to have changed after sale, refitting and re-evaluating models using the same established partitions. The public evidence package contains neither a complete historical assessment panel nor a finished, original-model comparison of these alternative feature sets. No numerical improvement or absence of leakage is claimed here.

In the geographical calculus experiment, municipality and ZIP are fixed when longitude and latitude move across the grid. Some location–category combinations can therefore be inconsistent with actual geography. The resulting field describes a controlled model sensitivity scenario rather than a market-price response to physical relocation.

## Other sources of uncertainty

County-valid recorded sales are a selected subset of property transfers; nominal dollars across 2013–2025 are not inflation-adjusted. Property conditions, renovation history, proximity to amenities and broader market changes are incompletely represented. Spatial density categories describe historical training-location coverage rather than an individual prediction interval.

See [data schema](SCHEMA.md), [experimental results](EXPERIMENT_RESULTS.md) and [methodology](METHODOLOGY.md).
