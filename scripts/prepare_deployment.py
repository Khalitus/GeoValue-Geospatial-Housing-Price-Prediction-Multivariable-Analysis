"""Export the user's local trusted model, dataset, and Task 3 surface for Vercel.

Run locally after `geovalue train` and `geovalue analyze`.
Does not retrain or modify the original CSV/model/reports.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def clean_array(array: np.ndarray) -> list:
    """Replace non-finite values with JSON null, preserving the original grid shape."""
    a = np.asarray(array, dtype=float)
    return np.where(np.isfinite(a), a, np.nan).tolist()  # cleaned by dump below


def _json_safe(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(v) for key, v in value.items()}
    return value


def prepare(
    *, data: Path, model: Path, surface: Path,
    train_report: Path, geo_report: Path, max_points: int = 1200,
    root: Path = ROOT,
) -> dict:
    from geovalue.data import INPUT_COLUMNS, load_housing_data
    from geovalue.geospatial import unproject_locations
    from geovalue.modeling import load_model

    if not 1 <= max_points <= 3000:
        raise ValueError("max_points must be from 1 to 3000")
    for path in (data, model, surface, train_report, geo_report):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required local artifact: {path}")

    frame = load_housing_data(data)
    pipeline = load_model(model)["pipeline"]
    selected = np.sort(np.random.default_rng(42).choice(
        len(frame), size=min(max_points, len(frame)), replace=False
    ))
    sample = frame.iloc[selected]
    results = pipeline.predict(sample[list(INPUT_COLUMNS)])
    points = [
        {"lon": float(lon), "lat": float(lat), "observed": float(actual),
         "predicted": float(pred)}
        for lon, lat, actual, pred in zip(
            sample["longitude"], sample["latitude"],
            sample["median_house_value"], results, strict=True
        )
    ]
    with np.load(surface, allow_pickle=False) as content:
        x = np.asarray(content["x_km"], dtype=float)
        y = np.asarray(content["y_km"], dtype=float)
        values = np.asarray(content["values"], dtype=float)
        dx = np.asarray(content["d_dx"], dtype=float)
        dy = np.asarray(content["d_dy"], dtype=float)
    if values.shape != (len(y), len(x)) or dx.shape != values.shape or dy.shape != values.shape:
        raise ValueError("Invalid geographical surface grid")
    mesh_x, mesh_y = np.meshgrid(x, y)
    lon, lat = unproject_locations(mesh_x.ravel()*1000, mesh_y.ravel()*1000)
    payload = _json_safe({
        "x_km": x.tolist(), "y_km": y.tolist(),
        "values": values.tolist(), "d_dx": dx.tolist(), "d_dy": dy.tolist(),
        "longitude": lon.reshape(values.shape).tolist(),
        "latitude": lat.reshape(values.shape).tolist(),
    })
    # Read/validate all source files before copying or exporting anything.
    reports = {}
    for key, path in (("training-report.json", train_report), ("geospatial-report.json", geo_report)):
        reports[key] = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(reports[key], dict):
            raise ValueError(f"Report is not a JSON object: {path}")

    dest = root / "public" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    deployment = root / "deploy"
    deployment.mkdir(parents=True, exist_ok=True)
    (dest / "points.json").write_text(json.dumps({"points": points}, separators=(",", ":")), encoding="utf-8")
    (dest / "surface.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    for name, document in reports.items():
        (dest / name).write_text(json.dumps(document, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    shutil.copyfile(model, deployment / "housing-model.joblib")
    shutil.copyfile(dest / "surface.json", deployment / "surface.json")
    for name in reports:
        shutil.copyfile(dest / name, deployment / name)
    return {"map_records": len(points), "grid_shape": list(values.shape),
            "exported": [str(dest / name) for name in ("points.json", "surface.json", "training-report.json", "geospatial-report.json")]
                         + [str(deployment / name) for name in ("housing-model.joblib", "surface.json", "training-report.json", "geospatial-report.json")]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/housing.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/housing-model.joblib")
    parser.add_argument("--surface", type=Path, default=ROOT / "reports/geospatial/price-surface.npz")
    parser.add_argument("--training-report", type=Path, default=ROOT / "reports/training-report.json")
    parser.add_argument("--geo-report", type=Path, default=ROOT / "reports/geospatial/geospatial-report.json")
    parser.add_argument("--max-points", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(prepare(data=args.data, model=args.model, surface=args.surface,
                            train_report=args.training_report, geo_report=args.geo_report,
                            max_points=args.max_points), indent=2))


if __name__ == "__main__":
    main()
