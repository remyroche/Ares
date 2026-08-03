from pathlib import Path


def test_challenger_is_hourly_strict_and_does_not_use_transition_fields():
    source = Path("scripts/run_strict_forward_sticky_fullcov_regime_challenger.py").read_text()
    assert '"transition" not in name.lower()' in source
    assert '"model_sample_cadence": "1h"' in source
    assert '"exact_replay_bar_cadence": "1m_labels_only"' in source
    assert "all feature selection, preprocessing, geometry, persistence" in source


def test_challenger_is_full_covariance_with_train_only_blocked_selection():
    source = Path("scripts/run_strict_forward_sticky_fullcov_regime_challenger.py").read_text()
    assert 'covariance_type="full"' in source
    assert "selection, blocked = train.iloc[:split], train.iloc[split:]" in source
    assert "select_features(selection, candidates)" in source
    assert "COMPONENTS = (3, 4, 5, 6)" in source
    assert "STICKY_PRIORS" in source


def test_persistence_gate_identity_uncertainty_and_diagonal_comparison_are_materialized():
    source = Path("scripts/run_strict_forward_sticky_fullcov_regime_challenger.py").read_text()
    assert "median dwell >=6h and hourly temporal switching <=10%" in source
    assert "regime_model_identity" in source and "regime_is_ood" in source
    assert "frozen_regime_model.joblib" in source
    assert "direct_rejected_diagonal_comparison.csv" in source
    assert 'manifest["test_temporal_switch_rate"]' in source
