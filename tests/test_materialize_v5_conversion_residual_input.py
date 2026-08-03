from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_v5_conversion_residual_input.py"
)
SPEC = importlib.util.spec_from_file_location("v5_conversion_input", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_contract_keeps_actions_and_outcomes_out_of_model_features() -> None:
    roles = MODULE.context_feature_contract()
    model = set(roles["baseline_model_features"]) | set(
        roles["optional_adverse_risk_ablation_only"]
    )
    assert not model.intersection(roles["target_only_never_features"])
    assert not model.intersection(roles["evaluation_only_never_features"])
    assert not any("time_to_first" in column for column in model)
    assert not any("bars_before" in column for column in model)


def test_join_uses_candidate_side_and_asserts_timestamp_not_raw_symbol() -> None:
    timestamp = pd.Timestamp("2025-03-01T00:00:00Z")
    left = pd.DataFrame(
        {
            "candidate_id": ["raw/symbol|t|long"],
            "side_name": ["long"],
            "__symbol__": ["NORMALISED"],
            "__ts__": [timestamp],
        }
    )
    right = pd.DataFrame(
        {
            "candidate_id": ["raw/symbol|t|long"],
            "side_name": ["long"],
            "__symbol__": ["RAW/SYMBOL"],
            "__ts__": [timestamp],
            "value": [1.0],
        }
    )
    joined = MODULE._join(left, right, name="source", columns=["value"])
    assert joined.value.iloc[0] == 1.0
    right.loc[0, "__ts__"] = timestamp + pd.Timedelta(hours=1)
    with pytest.raises(MODULE.MaterializationError, match="timestamp parity"):
        MODULE._join(left, right, name="source", columns=["value"])


def test_period_readiness_fails_closed_on_february_and_january() -> None:
    panel = pd.DataFrame(
        {
            "model_development_eligible": [True, False],
            "forward_diagnostic_only": [False, True],
        }
    )
    result = MODULE.period_readiness(panel).set_index("period")
    assert result.loc["2025-02", "status"] == "PARTIAL_NOT_JOINABLE_TO_V5_CANDIDATE_HEAD"
    assert result.loc["2025-01", "status"] == "MISSING_FAIL_CLOSED"
    assert result.loc["broader_history", "status"] == "FORBIDDEN_BRIDGE"


def test_successor_contract_has_explicit_calibration_and_selection_populations() -> None:
    assert MODULE.EXPECTED_ROWS == 110_730
    assert MODULE.EXPECTED_MARCH_ROWS == 41_472
    assert MODULE.EXPECTED_APRIL_ROWS == 69_258
    assert MODULE.DEFAULT_OUTPUT.name.endswith("_v3")


def test_full_sl_and_timeout_targets_use_canonical_exit_flags() -> None:
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [-0.02, -0.01, 0.01],
            "raw_score": [0.0, 0.0, 0.0],
            "execution_exit_reason": ["full_sl", "timeout", "trailing"],
            "exit_is_full_stop": [True, False, False],
            "exit_is_timeout": [False, True, False],
        }
    )
    result = MODULE.add_execution_targets(frame)
    assert result.target_stop_exit.tolist() == [1, 0, 0]
    assert result.target_timeout_exit.tolist() == [0, 1, 0]
    assert result.target_stop_exit.sum() == 1
