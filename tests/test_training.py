import json

import numpy as np
import pytest

from geovalue.training import (
    TrainingConfig,
    run_training,
    split_housing,
    train_and_evaluate,
)


@pytest.fixture
def quick_config():
    return TrainingConfig(forest_trees=6, xgboost_trees=10)


def test_split_is_reproducible_and_disjoint(synthetic_housing, quick_config):
    a = split_housing(synthetic_housing, quick_config)
    b = split_housing(synthetic_housing, quick_config)
    assert (len(a.train), len(a.validation), len(a.test)) == (108, 36, 36)
    assert a.train.index.tolist() == b.train.index.tolist()
    assert a.validation.index.tolist() == b.validation.index.tolist()
    assert a.test.index.tolist() == b.test.index.tolist()
    assert not (set(a.train.index) & set(a.validation.index))
    assert not (set(a.train.index) & set(a.test.index))
    assert not (set(a.validation.index) & set(a.test.index))
    assert len(set(a.train.index) | set(a.validation.index) | set(a.test.index)) == 180


def test_split_config_validation():
    with pytest.raises(ValueError, match="Train split"):
        TrainingConfig(test_fraction=0.7, validation_fraction=0.4)
    with pytest.raises(ValueError, match="Tree counts"):
        TrainingConfig(forest_trees=0)


def test_training_selects_and_refits_one_model(synthetic_housing, quick_config):
    result = train_and_evaluate(synthetic_housing, quick_config)
    report = result.report
    assert report["selected_model"] in ("linear", "random_forest", "xgboost")
    assert set(report["validation_results"]) == {"linear", "random_forest", "xgboost"}
    assert report["selection_metric"] == "validation_rmse"
    assert report["final_refit_rows"] == 144
    assert report["geographic_feature_experiment"]["comparison_split"] == "validation"
    assert set(report["held_out_test"]) == {"mae", "rmse", "r2"}
    assert all(np.isfinite(list(report["held_out_test"].values())))
    assert report["held_out_test"]["rmse"] >= 0


def test_training_files_are_written(synthetic_housing, quick_config, tmp_path):
    csv_path = tmp_path / "housing.csv"
    synthetic_housing.to_csv(csv_path, index=False)
    model_path = tmp_path / "artifacts" / "housing.joblib"
    report_path = tmp_path / "reports" / "training.json"
    summary = run_training(csv_path, model_path, report_path, quick_config)
    assert model_path.exists()
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["selected_model"] == saved["selected_model"]
    assert summary["held_out_test"] == saved["held_out_test"]
