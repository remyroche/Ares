from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_two_stage_absolute_net_conversion_challenger import (
    compose_scores,
    derive_targets,
    economics,
    select_blend,
    stage2_folds,
)


def test_derive_targets_keeps_loss_states_economically_signed() -> None:
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [0.02, -0.03, -0.01],
            "execution_exit_reason": ["trailing", "full_sl", "timeout"],
            "adverse_1atr_reached": [0, 1, 0],
            "existing_alpha_ev": [0.01, -0.01, 0.00],
        }
    )
    result = derive_targets(frame)
    assert result["positive_net"].tolist() == [1, 0, 0]
    assert result["adverse_negative"].tolist() == [0, 1, 0]
    assert result["timeout_negative"].tolist() == [0, 0, 1]
    assert np.allclose(result["direct_residual"], [0.01, -0.02, -0.01])


def test_stage2_folds_enforce_label_resolution_before_validation() -> None:
    ts = pd.date_range("2026-05-01", "2026-07-19 23:00", freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": np.repeat(ts, 8),
            "label_resolution_utc": np.repeat(ts + pd.Timedelta(hours=12), 8),
        }
    )
    folds = stage2_folds(frame)
    for _, train, validation in folds:
        start = frame.iloc[validation]["__ts__"].min()
        assert (frame.iloc[train]["label_resolution_utc"] < start).all()


def test_compose_scores_uses_absolute_payoff_not_peak_product() -> None:
    frame = pd.DataFrame(
        {
            "pred_positive_probability": [0.4],
            "pred_timeout_probability": [0.2],
            "catboost_adverse_1atr_gate__probability": [0.3],
            "pred_positive_payoff": [0.05],
            "pred_adverse_loss": [-0.04],
            "pred_timeout_loss": [-0.02],
            "pred_other_loss": [-0.01],
            "existing_alpha_ev": [0.01],
            "pred_direct_residual": [-0.005],
        }
    )
    result = compose_scores(frame)
    assert np.isclose(result.loc[0, "hurdle_ev"], 0.003)
    assert np.isclose(result.loc[0, "direct_ev"], 0.005)


def test_blend_and_economics_are_global_not_per_timestamp() -> None:
    n = 100
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC"),
            "direct_ev": np.arange(n),
            "hurdle_ev": np.arange(n)[::-1],
            "execution_net_ev_12h": np.arange(n) / 10_000,
        }
    )
    winner, trials = select_blend(frame)
    assert len(trials) == 5
    assert winner["weight_direct"] >= 0.75

    current = pd.DataFrame(
        {
            "selected_ev": np.arange(n),
            "eligible": np.ones(n, dtype=bool),
            "execution_net_ev_12h": np.arange(n) / 10_000,
            "side_name": ["long"] * n,
        }
    )
    result = economics(current)
    top = result.loc[result["arm"].eq("unrestricted_global_top10")].iloc[0]
    assert top["rows"] == 10
    assert np.isclose(top["net_ev_bps"], np.mean(np.arange(90, 100)))
