from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_timeout_action_head_ablation import (
    FEATURES,
    action_flags,
    action_metrics,
    join_causal_features,
    mixed_horizon_frame,
    router_gate_summary,
    select_thresholds,
)


def _paired(rows: int = 240) -> pd.DataFrame:
    decision = pd.date_range("2026-05-01", periods=rows, freq="h", tz="UTC")
    delta = np.where(np.arange(rows) % 5 == 0, 0.02, -0.005)
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(rows)],
            "__ts__": decision - pd.Timedelta(hours=1),
            "__symbol__": [f"A{i % 20}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2, "long", "short"),
            "execution_decision_utc": decision,
            "mapped_execution_ev": np.linspace(-0.01, 0.01, rows),
            "global_top10_capacity_member": True,
            "globally_admitted_floor_0bps": np.arange(rows) % 3 == 0,
            "globally_admitted_floor_25bps": np.arange(rows) % 7 == 0,
            "globally_admitted_floor_50bps": np.arange(rows) % 11 == 0,
            "paired_delta_net_24h_minus_12h": delta,
            "execution_label_end_utc__24h": decision + pd.Timedelta(hours=24),
            "execution_exit_reason__12h": "timeout",
            "execution_exit_reason__24h": np.where(
                delta > 0, "trailing", "full_sl"
            ),
            "execution_net_ev_12h__12h": -0.01,
            "execution_net_ev_12h__24h": -0.01 + delta,
        }
    )
    for horizon in ("12h", "24h"):
        for column in (
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_exit_hour",
            "execution_mfe_return_12h",
            "execution_mae_return_12h",
            "execution_entry_price",
            "execution_exit_price",
            "execution_expected_spread_bps",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
            "execution_label_end_utc",
            "execution_label_available_at",
            "policy_archetype",
            "execution_geometry_key",
            "execution_geometry_source",
        ):
            if f"{column}__{horizon}" in frame:
                continue
            frame[f"{column}__{horizon}"] = (
                decision + pd.Timedelta(hours=int(horizon[:-1]))
                if "utc" in column
                else "x"
                if column in ("policy_archetype", "execution_geometry_key", "execution_geometry_source")
                else 1.0
            )
    return frame


def _features(paired: pd.DataFrame) -> pd.DataFrame:
    frame = paired[
        ["candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc"]
    ].copy()
    frame["catboost_archetype"] = "slow_grinder"
    for index, column in enumerate(FEATURES[1:]):
        frame[column] = np.linspace(0.0, 1.0, len(frame)) + index
    for column in (
        "feature_available_at",
        "residual_available_at",
        "peak_mfe_available_at",
        "path_catboost_available_at",
    ):
        frame[column] = frame["execution_decision_utc"]
    return frame


def test_causal_join_rejects_post_decision_feature() -> None:
    paired = _paired()
    features = _features(paired)
    features.loc[0, "peak_mfe_available_at"] += pd.Timedelta(minutes=1)
    try:
        join_causal_features(paired, features, historical=False)
    except ValueError as exc:
        assert "post-decision" in str(exc)
    else:
        raise AssertionError("post-decision action feature was accepted")


def test_threshold_selection_and_action_metrics_do_not_reselect_outcomes() -> None:
    frame = join_causal_features(_paired(), _features(_paired()), historical=False)
    frame["action_oof_fold"] = 0
    frame["classifier_action_score"] = np.linspace(-1, 1, len(frame))
    frame["regression_action_score"] = np.linspace(-0.02, 0.02, len(frame))
    frame["blend_action_score"] = np.linspace(-0.5, 0.5, len(frame))
    thresholds, grid = select_thresholds(frame)
    assert set(thresholds) == {
        "classifier_action",
        "regression_action",
        "blend_action",
    }
    assert not grid.empty
    gates = router_gate_summary(grid, thresholds)
    assert set(gates) == set(thresholds)
    acted = action_flags(frame, thresholds)
    metrics = action_metrics(acted, evaluation="test")
    assert set(metrics["policy"]) == {
        "no_action_12h",
        "always_24h",
        "classifier_action",
        "regression_action",
        "blend_action",
    }


def test_mixed_horizon_uses_stored_net_and_cost_once() -> None:
    frame = join_causal_features(_paired(), _features(_paired()), historical=False)
    frame["action__regression_action"] = np.arange(len(frame)) % 2 == 0
    mixed = mixed_horizon_frame(frame, "regression_action")
    action = mixed["action__regression_action"].to_numpy()
    expected = np.where(
        action,
        frame["execution_net_ev_12h__24h"],
        frame["execution_net_ev_12h__12h"],
    )
    assert np.allclose(mixed["execution_net_ev_12h"], expected)
    expected_cost = np.where(
        action,
        frame["execution_cost_return__24h"],
        frame["execution_cost_return__12h"],
    )
    assert np.allclose(mixed["execution_cost_return"], expected_cost)
