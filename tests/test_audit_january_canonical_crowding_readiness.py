from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_january_canonical_crowding_readiness.py"
SPEC = importlib.util.spec_from_file_location("january_canonical_crowding_readiness", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_canonical_score_fails_closed_even_if_other_inputs_exist() -> None:
    result = MODULE.assess_readiness(
        canonical_score_present=False, exact_policy_labels_present=True, covariates_present=True
    )
    assert not result["materialization_legal"]
    assert result["status"] == "NOT_READY_FAIL_CLOSED_NO_CANONICAL_JANUARY_SCORE_BRIDGE"
    assert result["blocking_prerequisites"] == ["canonical_base_score_same_recipe_and_replayable_fitted_state"]


def test_crowding_quartile_is_outcome_free_and_deterministic() -> None:
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    first = MODULE._quantile_code(values)
    second = MODULE._quantile_code(values)
    assert first.equals(second)
    assert set(first) == {"q0", "q1", "q2", "q3"}


def test_runner_explicitly_prohibits_historical_score_bridge() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"historical_base_soft_oof calibration bridge"' in text
    assert "outcome_access\": False" in text


def test_correction_withdraws_cardinality_as_crowding() -> None:
    correction = Path(__file__).resolve().parents[1] / "scripts" / "write_january_canonical_crowding_readiness_v1_correction.py"
    payload = correction.read_text(encoding="utf-8")
    assert "UNIVERSE_CARDINALITY_NOT_SIGNAL_CROWDING" in payload
    assert "signal_density_or_crowding_support\": \"NOT_QUANTIFIED\"" in payload
    assert "january_requirement_for_q2_asset_count\": \"NOT_REQUIRED" in payload
