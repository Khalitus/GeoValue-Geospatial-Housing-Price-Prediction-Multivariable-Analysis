import json

from geovalue.cli import build_parser, main


def test_no_command_displays_help(capsys):
    assert main([]) == 0
    assert "usage: geovalue" in capsys.readouterr().out


def test_status_reports_present_dataset(housing_csv, capsys):
    assert main(["status", "--data", str(housing_csv)]) == 0
    assert "Dataset found" in capsys.readouterr().out


def test_status_reports_missing_dataset(tmp_path, capsys):
    assert main(["status", "--data", str(tmp_path / "none.csv")]) == 1
    assert "Dataset missing" in capsys.readouterr().out


def test_inspect_prints_json_and_writes_report(housing_csv, tmp_path, capsys):
    report = tmp_path / "nested" / "summary.json"
    assert main(["inspect", "--data", str(housing_csv), "--output", str(report)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["records"] == 4


def test_inspect_missing_dataset_is_clean_error(tmp_path, capsys):
    assert main(["inspect", "--data", str(tmp_path / "none.csv")]) == 2
    assert "Error: Dataset not found" in capsys.readouterr().err


def test_help_mentions_inspect():
    parser = build_parser()
    assert "inspect" in parser.format_help()
