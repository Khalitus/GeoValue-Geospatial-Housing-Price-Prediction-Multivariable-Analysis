"""Post-training official county surface, chart assets, support/error calibration.

No cloud services, database or synthetic data in the production outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geovalue.calculus import integrate_price_surface, save_price_surface
from geovalue.official.data import INPUT_COLUMNS, load_processed_data
from geovalue.official.modeling import load_model
from geovalue.official.spatial import build_surface, geographic_from_km, CRS
from geovalue.official.reliability import create_support_artifacts


def _serialize_grid(array: np.ndarray) -> list:
    """Finite floats become JSON numbers; unsupported (NaN) becomes null."""
    values = np.asarray(array, float)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional grid")
    return [[float(v) if np.isfinite(v) else None for v in row] for row in values]


def _write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def audit_same_day_sales(data: pd.DataFrame) -> dict:
    """Flag unresolved conflicting recorded prices, without silently removing sales."""
    required={"parcel_id","sale_date","sale_price"}
    if missing:=required-set(data.columns):
        raise ValueError(f"Audit requires {sorted(missing)}")
    keys=["parcel_id","sale_date"]
    groups=data.groupby(keys,dropna=False).sale_price.agg(["size","nunique","min","max"]).reset_index()
    inconsistent=groups.loc[(groups["size"]>1)&(groups["nunique"]>1)].copy()
    details=[{"parcel_id":str(r.parcel_id),"sale_date":str(r.sale_date),"records":int(r["size"]),
              "minimum_price_usd":float(r["min"]),"maximum_price_usd":float(r["max"])}
             for _,r in inconsistent.iterrows()]
    return {"conflicting_parcel_day_groups":len(details),"groups":details,
            "policy":"flag_only: preserve official distinct sale records; potential multiple instruments or amended transfers require source-level adjudication"}


def build_analysis(
    data_path: Path, model_path: Path, training_report_path: Path,
    *, report_dir: Path = Path("reports/official"),
    public_dir: Path = Path("public/official/data"),
    artifact_dir: Path = Path("artifacts/official"),
    grid_size: int = 86, sigma: float = 1.5, support_km: float = 4.5,
    max_points: int = 1600, seed: int = 42,
) -> dict:
    if max_points < 25 or max_points > 10000:
        raise ValueError("Map sample size must be between 25 and 10000")
    data=load_processed_data(data_path)
    saved=load_model(model_path)
    pipeline=saved["pipeline"]
    training_report=json.loads(Path(training_report_path).read_text(encoding="utf-8"))
    if training_report["model_selected"] != saved["name"]:
        raise ValueError("Training report selected model and supplied model artifact differ")
    if training_report.get("partition",{}).get("test_count") != int((data.sale_year==2025).sum()):
        raise ValueError("Training report and processed dataset have different 2025 row counts")
    report_dir=Path(report_dir);public_dir=Path(public_dir);artifact_dir=Path(artifact_dir)
    report_dir.mkdir(parents=True,exist_ok=True)
    public_dir.mkdir(parents=True,exist_ok=True)
    artifact_dir.mkdir(parents=True,exist_ok=True)

    duplicate_audit=audit_same_day_sales(data)
    _write_json(report_dir/"transaction-audit.json",duplicate_audit)

    surface,reference=build_surface(data,pipeline,grid_size=grid_size,sigma=sigma,support_km=support_km)
    integration=integrate_price_surface(surface)
    save_price_surface(surface,report_dir/"price-surface.npz")
    x,y=np.meshgrid(surface.x_km,surface.y_km)
    lon,lat=geographic_from_km(x.ravel(),y.ravel())
    geo={"crs":CRS,"axis_units":"km","target":"recorded residential sale price (nominal USD)",
         "records":int(len(data)),"grid_shape":list(surface.values.shape),"supported_grid_points":int(np.isfinite(surface.values).sum()),
         "smoothing_sigma_grid_cells":sigma,"support_distance_km":support_km,
         "fixed_reference_features":reference,"numerical_double_integral":integration,
         "mean_gradient_usd_per_km":float(np.nanmean(np.hypot(surface.d_dx,surface.d_dy))),
         "warnings":[
             "The surface fixes every non-location property input including municipality and ZIP; combinations can be unrealistic.",
             "The differentiated object is Gaussian-smoothed, not the raw XGBoost tree ensemble.",
             "The integrated supported grid region is not an official county land polygon; area/mean are scenario-only.",
             "Current assessment attributes can reflect later renovations even for older recorded sales."
         ]}
    _write_json(report_dir/"geospatial-report.json",geo)
    _write_json(public_dir/"geospatial-report.json",geo)
    _write_json(public_dir/"surface.json",{
        "crs":CRS,"x_km":[float(v) for v in surface.x_km],"y_km":[float(v) for v in surface.y_km],
        "longitude":_serialize_grid(lon.reshape(surface.values.shape)),
        "latitude":_serialize_grid(lat.reshape(surface.values.shape)),
        "values":_serialize_grid(surface.values),
        "d_dx":_serialize_grid(surface.d_dx),"d_dy":_serialize_grid(surface.d_dy),
    })

    # The map uses the untouched 2025 test cohort, not the development training rows.
    test=data.loc[data.sale_year==2025].copy()
    sampled=test.sample(n=min(max_points,len(test)),random_state=seed).sort_values("sale_date")
    fitted=np.asarray(pipeline.predict(sampled.loc[:,list(INPUT_COLUMNS)]),float)
    _write_json(public_dir/"points.json",{
        "period":"2025 held-out transactions",
        "count":len(sampled),"source_total":len(test),
        "latitude":[float(v) for v in sampled.latitude],
        "longitude":[float(v) for v in sampled.longitude],
        "observed":[float(v) for v in sampled.sale_price],
        "predicted":[float(v) for v in fitted],
        "absolute_error":[float(v) for v in np.abs(fitted-sampled.sale_price.to_numpy(float))],
        "note":"Observed transactions and frozen final model predictions for sampled 2025 held-out properties; not current listings.",
    })
    _write_json(public_dir/"training-report.json",training_report)

    reliability=create_support_artifacts(
        data,pipeline,training_report,
        output_npz=artifact_dir/"reliability-support.npz",
        report_path=report_dir/"reliability-report.json",
    )
    _write_json(public_dir/"reliability-report.json",reliability)
    return {
        "selected_model":saved["name"],"transactions":len(data),"sampled_2025_map_points":len(sampled),
        "supported_grid_points":geo["supported_grid_points"],
        "scenario_surface_mean_usd":integration["mean_surface_value_usd"],
        "reliability_calibrated_on":reliability["validation_rows"],
        "duplicate_conflict_groups":duplicate_audit["conflicting_parcel_day_groups"],
        "outputs":{
            "geospatial_report":str(report_dir/"geospatial-report.json"),
            "reliability_report":str(report_dir/"reliability-report.json"),
            "transaction_audit":str(report_dir/"transaction-audit.json"),
            "support_reference":str(artifact_dir/"reliability-support.npz"),
            "public_json":str(public_dir),
        },
    }


def diagnose_linear_baseline(data_path: Path, output_path: Path, *, seed: int = 42, top_n: int = 5) -> dict:
    """Explain extreme 2024 linear validation predictions without changing model selection.

    The fitted linear baseline is only a diagnostic; the saved XGBoost model is
    not modified or retrained. The diagnosis reads no 2025 sale prices.
    """
    from geovalue.official.training import split_sales
    from geovalue.official.modeling import build_model, metrics
    if not 1 <= top_n <= 30:
        raise ValueError("top_n must be between 1 and 30")
    data = load_processed_data(data_path)
    train, validation, _, _ = split_sales(data)
    predictor = build_model("linear", seed=seed)
    predictor.fit(train.loc[:, list(INPUT_COLUMNS)], train.sale_price)
    predictions = np.asarray(predictor.predict(validation.loc[:, list(INPUT_COLUMNS)]), dtype=float)
    truth = validation.sale_price.to_numpy(float)
    if not np.isfinite(predictions).all():
        raise ValueError("Linear baseline produced non-finite predictions")
    errors = np.abs(predictions-truth)
    ordered = np.argsort(errors)[-top_n:][::-1]
    worst = []
    for index in ordered:
        r=validation.iloc[index]
        worst.append({"parcel_id":str(r.parcel_id),"sale_date":str(r.sale_date),
                      "sale_price_usd":float(truth[index]),
                      "predicted_usd":float(predictions[index]),
                      "absolute_error_usd":float(errors[index]),
                      "lot_area_sqft":float(r.lot_area_sqft),
                      "living_area_sqft":float(r.living_area_sqft),
                      "floor_area_ratio":float(r.living_area_sqft/r.lot_area_sqft)})
    summary = {
        "experiment":"Linear Regression diagnostic on 2024 validation partition only",
        "validation_rows": len(validation),
        "metrics": metrics(truth,predictions),
        "negative_price_predictions": int(np.sum(predictions<0)),
        "predictions_above_5m_usd":int(np.sum(predictions>5_000_000)),
        "minimum_prediction_usd":float(predictions.min()),
        "maximum_prediction_usd":float(predictions.max()),
        "worst_absolute_errors":worst,
        "interpretation":"Unregularized linear extrapolation can become unstable for rare combinations and extreme engineered ratios. This diagnostic does not change XGBoost, the sale records, or the published model comparison.",
    }
    _write_json(Path(output_path), summary)
    return summary
