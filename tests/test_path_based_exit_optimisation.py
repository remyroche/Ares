from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.path_based_exit_optimisation import (
    AdaptiveExitConfig,
    BayesianActionSurface,
    BinContract,
    build_action_grid,
    build_decision_states,
    causal_hourly_asof_join,
    chronological_purged_folds,
    effective_support,
    evaluate_adaptive_exit,
    materialize_action_params,
    replay_static_action_grid,
)


BASELINE = {
    "sl_mult": 4.0,
    "trailing_activation_mult": 2.0,
    "trailing_power": 1.5,
    "giveback_beta": 0.25,
}


def test_action_grid_has_exact_baseline_and_625_joint_actions() -> None:
    actions = build_action_grid()
    assert len(actions) == 625
    assert actions[0].is_baseline
    assert sum(action.is_baseline for action in actions) == 1
    assert len({action.action_id for action in actions}) == 625
    assert {action.intensity() for action in actions} == {"mild", "medium", "aggressive"}


def test_action_mapping_preserves_zero_action_and_never_widens_stop() -> None:
    for action in build_action_grid():
        params = materialize_action_params(BASELINE, action)
        assert params["sl_mult"] <= BASELINE["sl_mult"]
        if action.is_baseline:
            for field in BASELINE:
                assert params[field] == BASELINE[field]


def test_hourly_join_is_backward_available_at_and_staleness_is_explicit() -> None:
    states = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "decision_ts": pd.to_datetime(
                ["2026-01-01 01:15", "2026-01-01 02:15", "2026-01-01 05:00"], utc=True
            ),
        }
    )
    hourly = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "available_at": pd.to_datetime(
                ["2026-01-01 01:05", "2026-01-01 02:05", "2026-01-01 05:05"], utc=True
            ),
            "trend": [1.0, 2.0, 999.0],
        }
    )
    joined = causal_hourly_asof_join(
        states,
        hourly,
        config=AdaptiveExitConfig(max_hourly_age_minutes=60.0),
    )
    assert joined["trend"].iloc[:2].tolist() == [1.0, 2.0]
    assert pd.isna(joined["trend"].iloc[2])
    assert (joined["available_at"].dropna() <= joined.loc[joined["available_at"].notna(), "decision_ts"]).all()


def test_decision_states_use_only_current_and_prior_path_bars() -> None:
    rows = pd.DataFrame(
        {
            "candidate_id": ["x"],
            "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
            "symbol": ["A"],
            "side": [1.0],
            "barrier_pct": [0.01],
        }
    )
    close = np.array([[100.0, 101.0, 102.0, 99.0]], dtype=np.float32)
    high = close + 0.5
    low = close - 0.5
    states = build_decision_states(
        rows,
        (close, high, low, close),
        entry_prices=np.array([100.0]),
        baseline_exit_bars=np.array([3]),
        bar_minutes=15,
    )
    changed = close.copy()
    changed[0, 3] = 1_000.0
    changed_states = build_decision_states(
        rows,
        (changed, changed + 0.5, changed - 0.5, changed),
        entry_prices=np.array([100.0]),
        baseline_exit_bars=np.array([3]),
        bar_minutes=15,
    )
    pd.testing.assert_frame_equal(
        states.iloc[:3].reset_index(drop=True),
        changed_states.iloc[:3].reset_index(drop=True),
    )


def test_bin_edges_are_frozen_from_training_rows() -> None:
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    contract = BinContract.fit(train, ["x"])
    original = contract.transform(pd.DataFrame({"x": [5.0, 95.0]}), "x")
    shifted_future = contract.transform(pd.DataFrame({"x": [-1e9, 1e9]}), "x")
    assert original[0] <= original[1]
    assert shifted_future.tolist() == [0, len(contract.edges["x"])]


def test_effective_support_does_not_scale_with_duplicate_path_states() -> None:
    timestamps = pd.date_range("2026-01-01", periods=20, freq="12h", tz="UTC")
    base = effective_support([f"t{i}" for i in range(20)], timestamps, [f"a{i % 4}" for i in range(20)])
    repeated = effective_support(
        np.repeat([f"t{i}" for i in range(20)], 10),
        np.repeat(timestamps, 10),
        np.repeat([f"a{i % 4}" for i in range(20)], 10),
    )
    assert repeated["unique_trades"] == base["unique_trades"]
    assert repeated["kish_ess"] == base["kish_ess"]
    assert repeated["effective_support"] == base["effective_support"]


def test_bayesian_surface_is_zero_anchored_and_support_reduces_uncertainty() -> None:
    actions = build_action_grid()[:25]
    # build_action_grid sorts baseline first, so a compact test grid keeps it.
    state = pd.DataFrame(
        {
            "candidate_id": [f"t{i}" for i in range(80)],
            "decision_ts": pd.date_range("2025-01-01", periods=80, freq="12h", tz="UTC"),
            "symbol": [f"A{i % 8}" for i in range(80)],
            "x": np.linspace(-1.0, 1.0, 80),
        }
    )
    delta = np.column_stack(
        [np.zeros(len(state)) if action.is_baseline else state["x"].to_numpy() * (1 + index)
         for index, action in enumerate(actions)]
    )
    small = BayesianActionSurface.fit(state.iloc[:20], actions, delta[:20], feature_names=["x"])
    large = BayesianActionSurface.fit(state, actions, delta, feature_names=["x"])
    small_mean, small_sd = small.predict(state.iloc[:5], actions)
    large_mean, large_sd = large.predict(state.iloc[:5], actions)
    assert np.all(small_mean[:, 0] == 0.0)
    assert np.all(large_mean[:, 0] == 0.0)
    assert np.all(small_sd[:, 0] == 0.0)
    assert np.nanmedian(large_sd[:, 1:]) <= np.nanmedian(small_sd[:, 1:])


def test_purged_folds_are_chronological_and_horizon_separated() -> None:
    rows = pd.DataFrame(
        {"timestamp": pd.date_range("2025-01-01", "2025-06-30", freq="12h", tz="UTC")}
    )
    for train, validation in chronological_purged_folds(rows, n_folds=3, purge_hours=12):
        assert rows.iloc[train]["timestamp"].max() < rows.iloc[validation]["timestamp"].min() - pd.Timedelta(hours=11)


def test_static_grid_replay_baseline_parity_is_exact() -> None:
    actions = build_action_grid()[:7]
    rows = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "symbol": ["A", "B"],
            "side": [1.0, 1.0],
            "barrier_pct": [0.01, 0.01],
        }
    )
    paths = tuple(np.ones((2, 3), dtype=np.float32) * 100.0 for _ in range(4))

    def simulator(frame, *_paths, **params):
        value = (
            params["sl_mult"]
            + params["trailing_activation_mult"]
            + params["trailing_power"]
            + params["giveback_beta"]
        )
        return {
            "selected_mask": np.ones(len(frame), dtype=bool),
            "net_returns": np.repeat(value / 10_000.0, len(frame)),
            "exit_bars": np.repeat(2, len(frame)),
            "exit_reason": ["timeout"] * len(frame),
        }

    delta, replay, _ = replay_static_action_grid(
        rows, paths, BASELINE, cost_pct=0.005, size_power=1.0,
        actions=actions, simulator=simulator,
    )
    assert np.all(delta[:, 0] == 0.0)
    assert np.allclose(replay["net_bps"][:, 0], sum(BASELINE.values()))


def test_metrics_include_relative_stability_and_requested_attribution() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=120, freq="7D", tz="UTC"),
            "baseline_net_bps": np.tile([-100.0, 150.0, 50.0], 40),
            "adaptive_net_bps": np.tile([-80.0, 160.0, 55.0], 40),
            "mfe_bps": np.repeat(300.0, 120),
            "conservative_delta_q_bps": np.linspace(-5.0, 20.0, 120),
            "action_distance": np.repeat(0.25, 120),
            "intervened": np.repeat(True, 120),
            "action_intensity": np.tile(["mild", "medium", "aggressive"], 40),
        }
    )
    metrics, monthly, weekly = evaluate_adaptive_exit(frame)
    required = {
        "delta_pnl_over_abs_delta_maxdd",
        "delta_sortino",
        "delta_trade_cvar_5_bps",
        "winner_retention",
        "delta_mfe_capture",
        "delta_winners_bps",
        "delta_losers_bps",
        "action_efficiency_bps",
        "worst_month_delta_ev_bps",
        "month_delta_ev_mad_bps",
        "relative_month_stability_ratio",
        "relative_week_stability_ratio",
        "month_cv_abs",
        "week_cv_abs",
    }
    assert required.issubset(metrics)
    assert not monthly.empty and not weekly.empty
    assert metrics["delta_ev_bps"] > 0.0
