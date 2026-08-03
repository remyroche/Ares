from pathlib import Path


def test_dae_gmm_challenger_is_strict_hourly_and_keeps_transition_separate():
    source = Path("scripts/run_strict_forward_dae_gmm_regime_challenger.py").read_text()
    assert '"transition" not in name.lower()' in source
    assert '"model_sample_cadence": "1h"' in source
    assert '"exact_replay_bar_cadence": "1m_labels_only"' in source
    assert "DAE representation/HPO, GMM geometry, persistence, OOD thresholds" in source


def test_dae_representation_hpo_and_frozen_uncertainty_are_materialized():
    source = Path("scripts/run_strict_forward_dae_gmm_regime_challenger.py").read_text()
    assert "DAE_HPO = ((4, 0.05), (8, 0.05), (12, 0.05))" in source
    assert "fit_dae(x_selection" in source and "fit_dae(x_train" in source
    assert "regime_is_density_ood" in source and "regime_is_reconstruction_ood" in source
    assert "frozen_dae_state_dict.pt" in source and "regime_model_identity" in source


def test_three_arm_stability_and_economic_comparisons_are_written():
    source = Path("scripts/run_strict_forward_dae_gmm_regime_challenger.py").read_text()
    assert "three_arm_structural_comparison.csv" in source
    assert "three_arm_monthly_stability_comparison.csv" in source
    assert "three_arm_exact_side_economic_attribution.parquet" in source
    assert "median dwell >=6h and hourly temporal switching <=10%" in source
