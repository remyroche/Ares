from pathlib import Path


SOURCE = Path("scripts/materialize_july2025_common30_final_base_residual_oof_bridge.py").read_text()


def test_bridge_declares_hourly_clock_and_strict_july_blocked_oof_cutoff():
    assert '"decision_timeframe": "1h candidate clock and 1h model features/scores"' in SOURCE
    assert '"native decision+24h < 2025-07-01T00:00:00+00:00"' in SOURCE
    assert '"native label resolution < 2025-07-01T00:00:00+00:00"' in SOURCE
    assert '"no_2026_outcomes": True' in SOURCE


def test_bridge_has_separate_resumable_base_and_residual_stages():
    assert 'stage must be base, residual, full, or finalize' in SOURCE
    assert '"manifest.sha256"' in SOURCE
    assert 'BASE_STAGE_COMPLETE_RESIDUAL_PENDING' in SOURCE
    assert 'base_oof_predictions.parquet' in SOURCE
    assert 'score_base_alpha' in SOURCE
    assert 'score_residual_expected_ev' in SOURCE


def test_bridge_does_not_change_the_frozen_model_parameters_for_resource_control():
    assert 'The accepted constructor already pins n_jobs=1' in SOURCE
    assert 'params["n_jobs"]' not in SOURCE
