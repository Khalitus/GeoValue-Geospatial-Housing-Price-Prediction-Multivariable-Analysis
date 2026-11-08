import json

from geovalue.cli import build_parser, main


def test_help_includes_training_commands():
    description = build_parser().format_help()
    assert "train" in description
    assert "model-info" in description


def test_training_cli_and_model_info(synthetic_housing, tmp_path, capsys):
    csv_path = tmp_path / "housing.csv"
    synthetic_housing.to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "results.json"
    assert main([
        "train", "--data", str(csv_path),
        "--model-output", str(model_path),
        "--report-output", str(report_path),
        "--forest-trees", "5", "--xgboost-trees", "8",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected_model"] in {"linear", "random_forest", "xgboost"}
    assert result["model_path"] == str(model_path)
    assert main(["model-info", "--model", str(model_path)]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["model_name"] == result["selected_model"]


def test_model_info_reports_missing_artifact(tmp_path, capsys):
    assert main(["model-info", "--model", str(tmp_path / "missing.joblib")]) == 2
    assert "Model not found" in capsys.readouterr().err
