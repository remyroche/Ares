from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import materialize_canonical_execution_reliability_input as materialize


def test_targets_are_exact_cost_aware_and_competing_classes_are_preserved() -> None:
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [0.01, -0.02, -0.005],
            "execution_gross_ev_12h": [0.02, -0.01, 0.005],
            "execution_cost_return": [0.01, 0.01, 0.01],
            "execution_mfe_return_12h": [0.03, 0.005, 0.02],
            "__soft_tb_first_event__": [
                "favorable_first",
                "adverse_first_or_conflict",
                "timeout",
            ],
            "__soft_tb_order_ambiguous__": [0, 1, 0],
            "__meaningful_mfe_reached_12h__": [1, 0, 1],
        }
    )
    result = materialize.add_targets(frame)
    assert result.target_meaningful_mfe.tolist() == [1, 0, 1]
    assert result.target_clean_favorable_first.tolist() == [1, 0, 0]
    assert result.target_economic_opportunity_hard.tolist() == [1, 0, 1]
    assert result.target_net_positive.tolist() == [1, 0, 0]
    assert result.target_severe_loss_100bps.tolist() == [0, 1, 0]
    assert result.target_competing_class.tolist() == [
        "favorable_first",
        "adverse_first_or_conflict",
        "timeout",
    ]


def test_feature_roles_keep_targets_and_action_fields_out() -> None:
    configs = [f"target{i}__S0__fixed" for i in range(8)]
    roles = materialize.feature_roles(configs)
    model = set(roles["default_ev_inputs"])
    assert not model.intersection(roles["target_only_never_features"])
    assert "target_net_positive_given_opportunity_valid" in roles["target_only_never_features"]
    assert not any("time_to" in name.lower() for name in model)
    assert not any("mae" in name.lower() for name in model)
    assert not any("mapped" in name.lower() for name in model)
    assert len(roles["transition_interaction_sources"]) == 5


def test_exact_join_uses_candidate_side_and_asserts_timestamp() -> None:
    timestamp = pd.Timestamp("2025-03-01T00:00:00Z")
    left = pd.DataFrame(
        {
            "candidate_id": ["c"],
            "side_name": ["long"],
            "__ts__": [timestamp],
        }
    )
    right = pd.DataFrame(
        {
            "candidate_id": ["c"],
            "side_name": ["long"],
            "__ts__": [timestamp],
            "value": [1],
        }
    )
    assert materialize.join_exact(left, right, "x").value.iloc[0] == 1
    right["__ts__"] = timestamp + pd.Timedelta(hours=1)
    with pytest.raises(materialize.MaterializationError, match="timestamp parity"):
        materialize.join_exact(left, right, "x")


def test_ambiguous_event_cannot_escape_adverse_conflict_class() -> None:
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [0.0],
            "execution_gross_ev_12h": [0.01],
            "execution_cost_return": [0.01],
            "execution_mfe_return_12h": [0.02],
            "__soft_tb_first_event__": ["favorable_first"],
            "__soft_tb_order_ambiguous__": [1],
            "__meaningful_mfe_reached_12h__": [1],
        }
    )
    with pytest.raises(materialize.MaterializationError, match="ambiguous"):
        materialize.add_targets(frame)
