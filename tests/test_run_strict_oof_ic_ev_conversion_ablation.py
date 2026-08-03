from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_strict_oof_ic_ev_conversion_ablation import (
    compose_conversion_scores,
    prepare_conversion_frame,
)


def test_complete_exit_policy_ev_is_a_simplex_and_uses_net_payoffs_once() -> None:
    scores = compose_conversion_scores(
        direct_net=np.array([0.01]),
        p_incidence=np.array([0.8]),
        p_capture_given_incidence=np.array([0.5]),
        positive_net_given_capture=np.array([0.03]),
        p_adverse_first=np.array([0.2]),
        adverse_net=np.array([-0.04]),
        p_favorable_first=np.array([0.6]),
        p_capture_given_favorable=np.array([0.5]),
        p_timeout=np.array([0.2]),
        favorable_positive_net=np.array([0.05]),
        favorable_nonpositive_net=np.array([-0.01]),
        timeout_net=np.array([-0.02]),
    )
    # (0.6*0.5)*5% + (0.6*0.5)*-1% + 0.2*-4% + 0.2*-2%
    assert np.isclose(scores["complete_exit_policy_ev"][0], 0.015 - 0.003 - 0.008 - 0.004)
    assert np.isclose(
        scores["p_exit_favorable_positive"][0]
        + scores["p_exit_favorable_nonpositive"][0]
        + scores["p_exit_adverse"][0]
        + scores["p_exit_timeout"][0],
        1.0,
    )
    # Exact-net payoff inputs have no deterministic fee subtraction here.
    assert np.isclose(
        scores["meaningful_mfe_capture_minus_adverse_diagnostic"][0],
        0.8 * 0.5 * 0.03 - 0.2 * 0.04,
    )


def test_composition_preserves_unscored_oof_rows_and_normalizes_catboost_drift() -> None:
    nan = np.nan
    scores = compose_conversion_scores(
        direct_net=np.array([nan, 0.007]),
        p_incidence=np.array([nan, 0.7123456789012345]),
        p_capture_given_incidence=np.array([nan, 0.6135792468013579]),
        positive_net_given_capture=np.array([nan, 0.028]),
        # Independent calibrated CatBoost path heads need not sum to exactly
        # one before explicit state normalization.
        p_adverse_first=np.array([nan, 0.3180049467201430]),
        adverse_net=np.array([nan, -0.035]),
        p_favorable_first=np.array([nan, 0.4662022009750090]),
        p_capture_given_favorable=np.array([nan, 0.9999999999999998]),
        p_timeout=np.array([nan, 0.2157928523048470]),
        favorable_positive_net=np.array([nan, 0.044]),
        favorable_nonpositive_net=np.array([nan, -0.006]),
        timeout_net=np.array([nan, -0.012]),
    )
    state = np.column_stack(
        [
            scores["p_exit_favorable_positive"],
            scores["p_exit_favorable_nonpositive"],
            scores["p_exit_adverse"],
            scores["p_exit_timeout"],
        ]
    )
    assert np.isnan(state[0]).all()
    assert np.isnan(scores["complete_exit_policy_ev"][0])
    assert np.isfinite(state[1]).all()
    assert np.isclose(state[1].sum(), 1.0, atol=1e-15)
    assert np.isfinite(scores["complete_exit_policy_ev"][1])


def test_composition_rejects_partial_or_zero_mass_probability_rows() -> None:
    common = {
        "direct_net": np.array([0.01]),
        "p_incidence": np.array([0.8]),
        "p_capture_given_incidence": np.array([0.5]),
        "positive_net_given_capture": np.array([0.03]),
        "adverse_net": np.array([-0.04]),
        "p_capture_given_favorable": np.array([0.5]),
        "favorable_positive_net": np.array([0.05]),
        "favorable_nonpositive_net": np.array([-0.01]),
        "timeout_net": np.array([-0.02]),
    }
    with pytest.raises(ValueError, match="partially missing"):
        compose_conversion_scores(
            **common,
            p_adverse_first=np.array([0.2]),
            p_favorable_first=np.array([np.nan]),
            p_timeout=np.array([0.2]),
        )
    with pytest.raises(ValueError, match="zero or negligible"):
        compose_conversion_scores(
            **common,
            p_adverse_first=np.array([0.0]),
            p_favorable_first=np.array([0.0]),
            p_timeout=np.array([0.0]),
        )


def test_prepare_frame_uses_maximum_execution_and_path_label_availability() -> None:
    identity = {
        "__ts__": [1],
        "__symbol__": ["BTC"],
        "side_name": ["long"],
        "candidate_id": ["candidate"],
    }
    frame = pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": ["2026-06-01T00:00:00Z"],
            "execution_label_end_utc": ["2026-06-01T10:00:00Z"],
            "execution_net_ev_12h": [0.01],
        }
    )
    grid = pd.DataFrame(
        {
            **identity,
            "grid_name": ["h12_u1p5atr"],
            "label_valid": [True],
            "label_resolution_utc": ["2026-06-01T12:00:00Z"],
            "peak_mfe_atr": [1.6],
            "upper_atr": [1.5],
            "favorable_first": [True],
            "adverse_first": [False],
            "timeout": [False],
            "early_3bar_adverse_atr": [0.1],
        }
    )
    prepared = prepare_conversion_frame(frame, grid, grid_name="h12_u1p5atr")
    assert prepared.loc[0, "support_label_available_utc"] == pd.Timestamp("2026-06-01T12:00:00Z")
    assert prepared.loc[0, "target_meaningful_mfe_incidence"] == 1
    assert prepared.loc[0, "target_exit_state"] == 0


def test_prepare_frame_rejects_label_available_before_decision() -> None:
    identity = {
        "__ts__": [1],
        "__symbol__": ["BTC"],
        "side_name": ["long"],
        "candidate_id": ["candidate"],
    }
    frame = pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": ["2026-06-01T00:00:00Z"],
            "execution_label_end_utc": ["2026-06-01T10:00:00Z"],
            "execution_net_ev_12h": [0.01],
        }
    )
    grid = pd.DataFrame(
        {
            **identity,
            "grid_name": ["h12_u1p5atr"],
            "label_valid": [True],
            "label_resolution_utc": ["2026-05-31T23:00:00Z"],
            "peak_mfe_atr": [1.6],
            "upper_atr": [1.5],
            "favorable_first": [True],
            "adverse_first": [False],
            "timeout": [False],
            "early_3bar_adverse_atr": [0.1],
        }
    )
    with pytest.raises(ValueError, match="precedes decision"):
        prepare_conversion_frame(frame, grid, grid_name="h12_u1p5atr")
