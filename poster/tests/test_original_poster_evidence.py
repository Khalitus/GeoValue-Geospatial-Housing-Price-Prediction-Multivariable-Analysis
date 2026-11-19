"""Invariants for the committed original-model poster evidence (no model loading)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'poster'


def test_original_model_evaluation_verified():
    report = json.loads((P/'evidence/original_model_verification.json').read_text())
    assert report['source'].startswith("Author's trusted original")
    assert report['saved_model_verified_against_original_2025_report'] is True
    assert report['model_selected'] == 'xgboost'
    assert report['test_rows'] == 8691
    got = report['recomputed_2025_test_metrics']
    published = report['reported_2025_test_metrics']
    for metric in ('mae','rmse','r2'):
        assert abs(got[metric]-published[metric]) < 1e-8


def test_spatial_evidence_has_expected_cohorts():
    report=json.loads((P/'evidence/original_model_verification.json').read_text())
    support=report['spatial_evidence']
    assert support['validation_rows'] == 8656
    assert support['2025_rows'] == 8691
    for name,expected in [('validation_error_by_support',8656),('held_out_2025_support_audit',8691)]:
        assert sum(v['n'] for v in support[name].values()) == expected
    assert support['held_out_2025_support_audit']['sparse']['mae_usd'] > support['held_out_2025_support_audit']['dense']['mae_usd']


def test_all_original_figures_are_present():
    for i in range(1,8):
        matches = list((P/'figures').glob(f'{i:02d}_*.png'))
        assert len(matches) == 1
        assert matches[0].stat().st_size > 1000
