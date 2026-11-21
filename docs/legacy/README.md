# Historical California study

GeoValue retains an earlier experiment based on the **1990 California housing census-area dataset**. It predicts median house values for geographic census areas, not the sale price of an individual home. The original application remains accessible through the `geovalue` command-line interface and the `/legacy` browser route.

This experiment is separate from the primary Allegheny County residential-sale research. Its source, target, feature schema, spatial projection and evaluation design differ. Reported California values should not be compared directly with the Pennsylvania sale-price estimates.

## Data and model

The historical source comprises 20,640 area-level observations with geographical coordinates, housing-age statistics, housing and population totals, median income, and ocean proximity. The target is `median_house_value`. A scikit-learn preprocessing pipeline and regression estimators are used to estimate area-level median values. Geographical analysis uses EPSG:3310 and numerical smoothing for contour, derivative and integration calculations.

A small example CSV is provided under `data/examples/`. The historical training source and fitted artifact are separate from the official Pennsylvania model. The legacy CLI offers dataset inspection, training, inference and geographical analysis commands; run `geovalue --help` for the current command contract.

## Mathematical interpretation

As in the principal study, finite differences are applied to a **smoothed, fixed-feature scenario** rather than analytically differentiating the raw tree ensemble. The spatial integral is over cells supported by observed census-area positions, not an official administrative polygon. Median census-area values are not individual current house prices.

For the primary project, see [official methodology](../official/METHODOLOGY.md), [evaluation](../official/EXPERIMENT_RESULTS.md) and [application guide](../official/USER_GUIDE.md).
