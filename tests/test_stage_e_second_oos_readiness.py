from pathlib import Path

from scripts.audit_stage_e_second_oos_readiness import V9, V10, model_candidates


def test_frozen_model_is_not_refit_on_second_oos() -> None:
    source = Path("scripts/audit_stage_e_second_oos_readiness.py").read_text()
    assert "model_refit\": False" in source
    assert all(token not in source for token in ("LGBMRegressor(", ".fit(", "train("))


def test_second_oos_was_not_used_by_prior_stages() -> None:
    assert not model_candidates(V9)
    assert not model_candidates(V10)
    source = Path("scripts/run_stage_d_compact_action_model.py").read_text()
    assert "FINAL_END=pd.Timestamp('2024-12-01" in source
    assert "2025-" not in source
