from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_frozen_older_data_wait10_action_ablation import (
    FEATURE_SETS,
    choose_weighted_threshold,
    route,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wait_delta": [0.01, -0.02],
            "pred_direct_delta": [0.01, -0.01],
            "pred_expected_delta": [-0.01, 0.01],
            "pred_q25_delta": [0.001, -0.001],
            "pred_weighted_event_score": [0.8, 0.2],
            "pred_soft_score": [0.6, 0.4],
        }
    )


def test_predeclared_action_routes_are_candidate_local() -> None:
    rows = _rows()
    assert route("enter_now", rows, 0.7).tolist() == [False, False]
    assert route("always_wait10", rows, 0.7).tolist() == [True, True]
    assert route("oracle_wait10", rows, 0.7).tolist() == [True, False]
    assert route("direct_delta", rows, 0.7).tolist() == [True, False]
    assert route("expected_delta", rows, 0.7).tolist() == [False, True]
    assert route("q25_guard", rows, 0.7).tolist() == [True, False]
    assert route("weighted_q25_fixed", rows, 0.7).tolist() == [True, False]
    assert route("weighted_q25_calibrated", rows, 0.7).tolist() == [True, False]
    assert route("soft_q25", rows, 0.7).tolist() == [True, False]


def test_calibration_abstains_without_positive_day_cluster_lower_bound() -> None:
    rows = pd.DataFrame(
        {
            "execution_decision_utc": pd.date_range(
                "2025-02-01", periods=400, freq="h", tz="UTC"
            ),
            "wait_delta": np.full(400, -0.001),
            "pred_weighted_event_score": np.linspace(0.0, 1.0, 400),
            "pred_q25_delta": np.full(400, 0.001),
        }
    )
    threshold, audit = choose_weighted_threshold(rows)
    assert np.isinf(threshold)
    assert audit["selection"] == "ABSTAIN_NO_POSITIVE_CALIBRATION_LOWER_BOUND"


def test_transition_feature_ablation_is_nested() -> None:
    assert set(FEATURE_SETS["base_only"]).issubset(
        FEATURE_SETS["base_plus_transition"]
    )
    assert set(FEATURE_SETS["base_plus_transition"]).issubset(
        FEATURE_SETS["all_state_transition"]
    )
