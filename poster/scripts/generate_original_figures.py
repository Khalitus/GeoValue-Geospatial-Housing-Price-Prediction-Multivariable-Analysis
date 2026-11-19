"""Generate competition figures from GeoValue's trusted, ORIGINAL local model.

Usage from repository root:
    python poster/scripts/generate_original_figures.py
    python poster/scripts/generate_original_figures.py --sensitivity

No model training, no source-data edits, no synthetic observations.
Only load an original joblib artifact you created yourself on this machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from geovalue.calculus import load_price_surface
from geovalue.official.data import INPUT_COLUMNS, load_processed_data
from geovalue.official.modeling import load_model, metrics
from geovalue.official.training import split_sales
from geovalue.official.spatial import build_surface, project_km

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "poster" / "figures"
EVIDENCE = ROOT / "poster" / "evidence"
DATA = ROOT / "data/official/processed/residential_sales_2013_2025.csv"
MODEL = ROOT / "artifacts/official/residential-sale-model.joblib"
REPORT = ROOT / "reports/official/training-report.json"
GEO = ROOT / "reports/official/geospatial-report.json"
RELIABILITY = ROOT / "reports/official/reliability-report.json"
SURFACE = ROOT / "reports/official/price-surface.npz"

RED = "#A41B22"
DARK = "#1C262D"
BLUE = "#52748A"
GRAY = "#6B7280"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 230})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_figure(fig, name: str) -> None:
    fig.savefig(OUT / (name + ".png"), dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_required():
    missing = [str(p.relative_to(ROOT)) for p in [DATA, MODEL, REPORT, GEO, RELIABILITY, SURFACE] if not p.is_file()]
    if missing:
        raise SystemExit("Missing original local project files; run `geovalue-official analyze` first if needed:\n  " + "\n  ".join(missing))


def check_original_metrics(pipeline, data, report):
    train, validation, test, partition = split_sales(data)
    for part, column in [("train_count","train_count"), ("validation_count","validation_count"), ("test_count","test_count")]:
        if int(partition[column]) != int(report["partition"][part]):
            raise RuntimeError(f"Data and training report disagree on {part}")
    actual = test.sale_price.to_numpy(dtype=float)
    predicted = np.asarray(pipeline.predict(test.loc[:, list(INPUT_COLUMNS)]), dtype=float)
    got = metrics(actual, predicted)
    expected = report["held_out_2025_test"]
    # For a genuine saved model + its own data, metric differences should be numerical noise only.
    # Stop rather than silently publishing diagnostics from a different fitted model.
    if (abs(got["mae"] - expected["mae"]) > 2.0
        or abs(got["rmse"] - expected["rmse"]) > 2.0
        or abs(got["r2"] - expected["r2"]) > 0.0001):
        raise RuntimeError("The saved model does NOT reproduce the published 2025 report. "
                           f"Computed={got}; Reported={expected}. "
                           "Check model/report pairing and environment before publishing.")
    return test, actual, predicted, got


def format_usd(v):
    return f"${v/1000:,.0f}k"


def figure_observed_sales(test):
    x, y = project_km(test.longitude.to_numpy(float), test.latitude.to_numpy(float))
    p = test.sale_price.to_numpy(float)
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.1), layout="constrained")
    h1 = axs[0].hexbin(x, y, C=p, gridsize=52, mincnt=8, reduce_C_function=np.median, cmap="YlOrRd", linewidths=0.0)
    h2 = axs[1].hexbin(x, y, gridsize=52, mincnt=8, cmap="Blues", linewidths=0.0)
    fig.colorbar(h1, ax=axs[0], label="Observed median sale price (USD)", shrink=0.86)
    fig.colorbar(h2, ax=axs[1], label="Transactions per displayed hexagon", shrink=0.86)
    for ax in axs:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (km, EPSG:26917)")
        ax.set_ylabel("Northing (km, EPSG:26917)")
    axs[0].set_title("A. 2025 recorded sale prices")
    axs[1].set_title("B. 2025 recorded sale density")
    fig.suptitle(f"Official held-out 2025 sales | {len(test):,} observed transactions | minimum 8 per hexagon", color=DARK, fontsize=12)
    write_figure(fig,"01_observed_2025_sales")


def figure_original_surface(surface, geo):
    xx, yy = np.meshgrid(surface.x_km, surface.y_km)
    values = np.ma.masked_invalid(surface.values)
    mag_raw = np.hypot(surface.d_dx, surface.d_dy)
    mag = np.ma.masked_invalid(mag_raw)
    cap = float(np.nanpercentile(mag_raw,95))
    fig, axs = plt.subplots(1,2, figsize=(13.2,5.2), layout="constrained")
    ax=axs[0]
    cf=ax.contourf(xx,yy,values, levels=20, cmap="viridis")
    if values.count() > 10 and float(values.max()) > float(values.min()):
        cs = ax.contour(xx,yy,values, colors="#111827", linewidths=0.45, alpha=0.55, levels=8)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    fig.colorbar(cf,ax=ax,label="Smoothed modeled price (nominal USD)",shrink=0.85)
    ax.set_title("C. Controlled-property price scenario")
    ax=axs[1]
    cf2=ax.pcolormesh(xx,yy,mag,vmin=0,vmax=cap,cmap="magma",shading="auto",rasterized=True)
    fig.colorbar(cf2,ax=ax,label="Gradient magnitude (USD/km; scale capped at P95)",shrink=0.85)
    stride=max(1,len(surface.x_km)//19)
    grad_x=surface.d_dx[::stride,::stride]
    grad_y=surface.d_dy[::stride,::stride]
    norm=np.hypot(grad_x,grad_y)
    ux=np.divide(grad_x,norm,out=np.zeros_like(grad_x),where=np.isfinite(norm)&(norm>0))
    uy=np.divide(grad_y,norm,out=np.zeros_like(grad_y),where=np.isfinite(norm)&(norm>0))
    good=np.isfinite(grad_x)&np.isfinite(grad_y)&(norm>0)
    ax.quiver(xx[::stride,::stride][good], yy[::stride,::stride][good], ux[good],uy[good],
              color="white",alpha=0.85,scale=24,width=0.0045,headwidth=3,headlength=4)
    ax.set_title("D. Derived spatial gradient field")
    for ax in axs:
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel("Easting (km, EPSG:26917)")
        ax.set_ylabel("Northing (km, EPSG:26917)")
    sig=geo.get("smoothing_sigma_grid_cells")
    fig.suptitle(f"Original saved XGBoost model | sigma={sig} grid cells | NOT observed prices or a causal change",color=DARK,fontsize=11)
    write_figure(fig,"02_original_model_surface_and_gradient")
    return {"gradient_display_cap_p95_usd_per_km":cap,"original_gradient_max_usd_per_km":float(np.nanmax(mag_raw))}


def figure_prediction_diagnostic(test, actual, predicted):
    fig,ax=plt.subplots(figsize=(6.9,5.8),layout="constrained")
    ax.scatter(actual,predicted,s=5,alpha=0.18,c=BLUE,rasterized=True)
    lo=min(float(actual.min()),float(predicted.min()))
    hi=max(float(actual.max()),float(predicted.max()))
    ax.plot([lo,hi],[lo,hi],linestyle="--",lw=1,color=RED,label="Exact prediction")
    # Symmetric log retains negative predictions instead of silently dropping any records.
    ax.set_xscale("symlog",linthresh=10000)
    ax.set_yscale("symlog",linthresh=10000)
    ax.set_xlabel("Observed sale price (USD)")
    ax.set_ylabel("Original-model prediction (USD)")
    ax.set_title(f"2025 held-out actual vs. predicted | all {len(test):,} rows")
    ax.legend(frameon=False)
    write_figure(fig,"03_original_2025_actual_vs_predicted")


def figure_price_bracket(actual, predicted):
    bins=[0,150000,300000,500000,750000,np.inf]
    names=["<$150k","$150-300k","$300-500k","$500-750k","$750k+"]
    frame=pd.DataFrame({"observed":actual,"predicted":predicted})
    frame["bracket"]=pd.cut(frame.observed,bins=bins,labels=names,include_lowest=True,right=False)
    rows=[]
    for name in names:
        subset=frame[frame.bracket==name]
        if len(subset)==0:continue
        ae=(subset.predicted-subset.observed).abs()
        rows.append({"bracket":name,"n":len(subset),"mae_usd":float(ae.mean()),"median_ape_percent":float((ae/subset.observed).median()*100)})
    bars=pd.DataFrame(rows)
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(11.8,4.5),layout="constrained")
    bars.plot.bar(x="bracket",y="mae_usd",ax=ax,color=RED,legend=False,rot=0)
    bars.plot.bar(x="bracket",y="median_ape_percent",ax=ax2,color=BLUE,legend=False,rot=0)
    ax.set_title("A. Mean absolute error by sale-price bracket")
    ax.set_ylabel("MAE (USD)")
    ax2.set_title("B. Median absolute percentage error")
    ax2.set_ylabel("Median absolute percentage error (%)")
    for a in (ax,ax2):
        a.set_xlabel("Observed sale price bracket")
    for patch,n in zip(ax.patches,bars.n):
        ax.text(patch.get_x()+patch.get_width()/2,patch.get_height(),f"n={n:,}",fontsize=8,ha="center",va="bottom")
    fig.suptitle("Original saved XGBoost model | 2025 test cohort",fontsize=12)
    write_figure(fig,"04_original_2025_price_bracket_errors")
    return rows


def figure_support(reliability):
    validation=reliability["validation_error_by_support"]
    audit=reliability["held_out_2025_support_audit"]
    tiers=["dense","typical","sparse"]
    vs=[validation[t]["mae_usd"] for t in tiers]
    ts=[audit[t]["mae_usd"] for t in tiers]
    if any(v is None for v in vs+ts):
        raise ValueError("Support groups missing; cannot plot misleading support comparisons")
    ix=np.arange(3)
    fig,ax=plt.subplots(figsize=(8,4.7),layout="constrained")
    b1=ax.bar(ix-.18,vs,width=.36,color=BLUE,label="2024 validation")
    b2=ax.bar(ix+.18,ts,width=.36,color=RED,label="2025 original-model audit")
    for bar,year in [(b1,validation),(b2,audit)]:
        for k,patch in enumerate(bar):
            key=tiers[k]
            ax.text(patch.get_x()+patch.get_width()/2,patch.get_height(),f"n={year[key]['n']:,}",ha="center",va="bottom",fontsize=8)
    ax.set_xticks(ix, [v.capitalize() for v in tiers])
    ax.set_ylabel("MAE (nominal USD)")
    ax.set_title("Spatial support: observed errors by geographical density tier")
    ax.legend(frameon=False)
    write_figure(fig,"05_original_model_spatial_support_errors")


def figure_validation(report):
    labels=["Random Forest", "XGBoost"]
    values=[report["model_metrics"]["random_forest"]["validation"]["rmse"],
            report["model_metrics"]["xgboost"]["validation"]["rmse"]]
    geo=report["geographical_ablation"]
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axs[0].bar(labels,values,color=[BLUE,RED]);axs[0].set_title("2024 model-selection RMSE");axs[0].set_ylabel("RMSE (USD)")
    axs[1].bar(["Without geography","With geography"],
               [geo["validation_without_geo"]["rmse"],geo["validation_with_geo"]["rmse"]],color=[BLUE,RED])
    axs[1].set_title("2024 geography-feature ablation")
    axs[1].set_ylabel("RMSE (USD)")
    write_figure(fig,"06_reported_validation_comparisons")


def figure_sensitivity(data,pipeline):
    sigmas=(1.0,1.5,2.5)
    fig,axs=plt.subplots(1,3,figsize=(13,4.3),layout="constrained")
    cases=[]
    for ax,sigma in zip(axs,sigmas):
        result,_=build_surface(data,pipeline,grid_size=86,sigma=sigma,support_km=4.5)
        mag=np.hypot(result.d_dx,result.d_dy)
        cap=float(np.nanpercentile(mag,95))
        xx,yy=np.meshgrid(result.x_km,result.y_km)
        pc=ax.pcolormesh(xx,yy,np.ma.masked_invalid(mag),shading="auto",cmap="magma",vmin=0,vmax=cap,rasterized=True)
        ax.set_aspect("equal",adjustable="box")
        ax.set_title(f"Sigma {sigma} grid cells")
        ax.set_xlabel("Easting (km)")
        if sigma==1.0:ax.set_ylabel("Northing (km)")
        cases.append({"sigma":sigma,"mean_usd_per_km":float(np.nanmean(mag)),"p95_usd_per_km":cap,
                      "max_usd_per_km":float(np.nanmax(mag))})
    fig.suptitle("Original model: numerical gradient sensitivity to Gaussian smoothing")
    write_figure(fig,"07_original_model_gradient_sensitivity")
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity",action="store_true",help="Also evaluate smoothing sigma = 1.0, 1.5, 2.5 (slower)")
    args=parser.parse_args()
    assert_required()
    OUT.mkdir(parents=True,exist_ok=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    data=load_processed_data(DATA)
    artifact=load_model(MODEL)
    if artifact["name"]!="xgboost":raise SystemExit("Original artifact is not the selected XGBoost model")
    pipeline=artifact["pipeline"]
    report=load_json(REPORT)
    geo=load_json(GEO)
    reliability=load_json(RELIABILITY)
    if report["model_selected"] != artifact["name"]:raise RuntimeError("Model and report names disagree")
    test, actual, predicted, observed_metrics=check_original_metrics(pipeline,data,report)
    surface=load_price_surface(SURFACE)
    if list(surface.values.shape)!=geo["grid_shape"]:raise RuntimeError("Surface shape and geographical report disagree")
    if int(np.isfinite(surface.values).sum())!=int(geo["supported_grid_points"]):
        raise RuntimeError("Surface finite-node count and geographical report disagree")
    if int(reliability["held_out_2025_rows"])!=len(test):raise RuntimeError("2025 support audit size does not match test data")
    figure_observed_sales(test)
    gradient_info=figure_original_surface(surface,geo)
    figure_prediction_diagnostic(test,actual,predicted)
    bracket_info=figure_price_bracket(actual,predicted)
    figure_support(reliability)
    figure_validation(report)
    cases=figure_sensitivity(data,pipeline) if args.sensitivity else None
    provenance={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source":"Author's trusted original saved XGBoost artifact (no retraining)",
        "model_selected":artifact["name"],
        "records_total":len(data),"test_rows":len(test),
        "saved_model_verified_against_original_2025_report":True,
        "recomputed_2025_test_metrics":observed_metrics,
        "reported_2025_test_metrics":report["held_out_2025_test"],
        "grid_shape":list(surface.values.shape),
        "supported_grid_points":int(np.isfinite(surface.values).sum()),
        "area_integral_and_mean":geo.get("numerical_double_integral"),
        "gaussian_sigma_grid_cells":geo.get("smoothing_sigma_grid_cells"),
        "gradient_display":gradient_info,
        "spatial_evidence":{
            "validation_rows":reliability.get("validation_rows"),
            "2025_rows":reliability.get("held_out_2025_rows"),
            "validation_error_by_support":reliability["validation_error_by_support"],
            "held_out_2025_support_audit":reliability["held_out_2025_support_audit"],
            "spearman_validation":reliability.get("spearman_distance_vs_absolute_error"),
            "distance_boundaries_km":reliability.get("distance_quantile_boundaries_km")
        },
        "2025_price_brackets":bracket_info,
        "sensitivity":cases,
        "input_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [DATA,MODEL,REPORT,GEO,RELIABILITY,SURFACE]},
        "library_versions":{p:version(p) for p in ["numpy","pandas","matplotlib","scipy","scikit-learn","xgboost","joblib","pyproj"]},
        "caveats":[
            "2025 observed-sale map uses actual 2025 recorded transactions.",
            "Numerical gradient map is the derivative of a Gaussian-smoothed fixed-feature scenario produced with the original fitted XGBoost model; not an observed or causal price change.",
            "Support tier 2024 calibration refits the historical pre-2024 candidate model, whereas 2025 audit uses the original final saved XGBoost model.",
            "Present-day assessment attributes may postdate historical sales."
        ],
    }
    dest=EVIDENCE/"original_model_verification.json"
    dest.write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("VERIFIED: original model predicts the published 2025 test metrics")
    print(json.dumps({"metrics":observed_metrics,"figures":sorted(p.name for p in OUT.glob("*.png")),"provenance":str(dest)},indent=2))

if __name__=="__main__":
    main()
