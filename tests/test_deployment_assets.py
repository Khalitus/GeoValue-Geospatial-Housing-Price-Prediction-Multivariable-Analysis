"""Packaging/export checks, using synthetic data and a small trusted model."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from geovalue.data import INPUT_COLUMNS
from geovalue.modeling import build_model, save_model

ROOT = Path(__file__).resolve().parents[1]


def load_script(file: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_exports_real_synthetic_predictions_and_json(tmp_path, synthetic_housing):
    prepare = load_script("prepare_deployment.py", "prepare_web_for_test").prepare
    data = tmp_path / "housing.csv"
    synthetic_housing.to_csv(data, index=False)
    model = build_model("linear").fit(synthetic_housing[list(INPUT_COLUMNS)], synthetic_housing["median_house_value"])
    saved = save_model(model, tmp_path / "model.joblib", model_name="linear")
    x = np.linspace(-200, 200, 5)
    y = np.linspace(-100, 300, 5)
    mx, my = np.meshgrid(x, y)
    surface = tmp_path / "price-surface.npz"
    np.savez_compressed(surface, x_km=x,y_km=y,values=1000+mx+2*my,d_dx=np.ones_like(mx),d_dy=np.ones_like(mx)*2)
    train = tmp_path / "train.json";geo=tmp_path / "geo.json"
    train.write_text(json.dumps({"selected_model":"linear"}))
    geo.write_text(json.dumps({"crs":"EPSG:3310"}))
    result=prepare(data=data,model=saved,surface=surface,train_report=train,geo_report=geo,max_points=20,root=tmp_path / "deployment")
    assert result["map_records"] == 20
    assert result["grid_shape"] == [5,5]
    points=json.loads((tmp_path / "deployment/public/data/points.json").read_text())
    assert len(points["points"]) == 20
    assert np.isfinite([x["predicted"] for x in points["points"]]).all()
    assert (tmp_path / "deployment/deploy/housing-model.joblib").exists()
    assert json.loads((tmp_path / "deployment/public/data/surface.json").read_text())["values"][0][0] == 600
    assert data.exists() and saved.exists()  # source never deleted


def test_packager_includes_local_csv_json_and_models(tmp_path):
    pkg = load_script("package_full.py", "package_for_test")
    project = tmp_path / "project"; project.mkdir()
    (project / "data/raw").mkdir(parents=True)
    (project / "artifacts").mkdir()
    (project / "reports").mkdir()
    (project / ".venv").mkdir()
    (project / ".venv/bogus.py").write_text("ignore")
    (project / "data/raw/housing.csv").write_text("real,data\n1,2\n")
    (project / "artifacts/housing-model.joblib").write_bytes(b"demo")
    (project / "reports/training-report.json").write_text('{"ok":true}')
    destination = tmp_path / "all.zip"
    count,_ = pkg.create_archive(project,destination)
    assert count == 3
    with ZipFile(destination) as archive:
        names=archive.namelist()
        assert "GeoValue/data/raw/housing.csv" in names
        assert "GeoValue/artifacts/housing-model.joblib" in names
        assert "GeoValue/reports/training-report.json" in names
        assert not any(".venv" in n for n in names)
