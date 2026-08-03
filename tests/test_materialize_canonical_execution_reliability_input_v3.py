from __future__ import annotations

import numpy as np

from scripts import materialize_canonical_execution_reliability_input_v3 as v3


def test_updated_roles_keep_capture_targets_out_of_default_inputs() -> None:
    roles = v3.updated_roles(
        {
            "default_ev_inputs": ["score"],
            "target_only_never_features": ["old_target"],
            "explicitly_unavailable": [
                "proper pre-exit capture target/head",
                "archetype z unavailable",
            ],
        }
    )
    assert not set(v3.CAPTURE_COLUMNS).intersection(roles["default_ev_inputs"])
    assert set(v3.CAPTURE_COLUMNS).issubset(
        roles["target_only_never_features"]
    )
    assert roles["explicitly_unavailable"] == ["archetype z unavailable"]


def test_capture_conditional_columns_are_declared() -> None:
    roles = v3.updated_roles(
        {
            "default_ev_inputs": ["score"],
            "target_only_never_features": [],
            "explicitly_unavailable": [],
        }
    )
    assert (
        roles["capture_target_contract"]["capture_validity_mask"]
        == "target_pre_exit_capture_valid"
    )
    assert "target_pre_exit_economic_capture_ratio" in roles[
        "capture_target_contract"
    ]["conditional_magnitudes"]
