import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_joint_objective import (
    evaluate_joint_wallet_objective,
    ev_bayesian_requested_fractions,
    priority_order,
)


def test_corrected_ev_controls_simultaneous_priority():
    ts = np.array([1, 1, 2], dtype=np.int64)
    order = priority_order(ts, np.array([0.01, 0.03, 0.02]), np.array([0.8, 0.9, 0.7]))
    assert order.tolist() == [1, 0, 2]


def test_ev_bayesian_sizing_is_base_then_overlay():
    result = ev_bayesian_requested_fractions(np.array([0.0, 1.0]), np.array([1.0, 1.2]))
    assert np.allclose(result, [0.075, 0.18])


def test_joint_objective_applies_wallet_cap_and_holding_efficiency():
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-01", "2026-07-01"], utc=True)})
    timestamps = rows["timestamp"].astype("int64").to_numpy()
    close = np.full((2, 2), 100.0)
    score, metrics, detail = evaluate_joint_wallet_objective(
        rows=rows,
        timestamps_ns=timestamps,
        symbol_codes=np.array([0, 1], dtype=np.int32),
        side=np.ones(2),
        raw_entry_prices=np.full(2, 100.0),
        entry_half_spread_bps=np.zeros(2),
        close_paths=close,
        exit_bars=np.array([0, 1], dtype=np.int32),
        net_returns=np.array([0.10, 0.10]),
        corrected_ev=np.array([0.03, 0.02]),
        corrected_ev_rank=np.ones(2),
        bayesian_multiplier=np.array([4.0, 4.0]),
        holding_efficiency_weight=0.1,
    )
    assert np.isclose(detail["admitted_notional"].sum(), 0.8)
    assert metrics["holding_efficiency_pnl"] > metrics["net_pnl_bankroll"]
    assert score > metrics["stability_component"]
