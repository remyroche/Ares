from __future__ import annotations

import pandas as pd
import pytest

from scripts.execution.build_execution_oracle import build_execution_oracle


def _states() -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="min")
    costs = [20., 20., 20., 20., 10., 50.]
    return pd.DataFrame({
        "symbol": ["BTC/USD"] * len(times), "available_ts": times, "mid": [100.] * len(times),
        "spread_bps": [10.] * len(times), "book_valid": [True] * len(times),
        "sell_book_cost_bps_n100": costs, "sell_insufficient_depth_n100": [False] * len(times),
        "buy_book_cost_bps_n100": costs, "buy_insufficient_depth_n100": [False] * len(times),
    })


def test_oracle_uses_predeclared_earlier_exit_and_charges_parent_cost_once() -> None:
    t_star = pd.Timestamp("2026-01-01T00:05:00Z")
    exits = pd.DataFrame({
        "exit_id": ["x"], "symbol": ["BTC/USD"], "side": ["long"], "exit_ts": [t_star],
        "entry_price": [100.], "position_notional": [100.], "exit_reason": ["trailing"],
    })
    prices = pd.DataFrame({
        "symbol": ["BTC/USD"] * 6, "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="min"),
        "close": [100., 100., 100., 100., 105., 104.],
    })
    first = build_execution_oracle(exits=exits, states=_states(), prices=prices, notionals=(100,), policy_cost_bps=100.)
    row = first.loc[first["preempt_minutes"].eq(1)].iloc[0]
    # Earlier close adds 100 bps and book execution saves another 40 bps.
    assert row["preemption_gain_bps"] == pytest.approx(140.0)
    second = build_execution_oracle(exits=exits, states=_states(), prices=prices, notionals=(100,), policy_cost_bps=250.)
    assert second.loc[second["preempt_minutes"].eq(1), "preemption_gain_bps"].iloc[0] == pytest.approx(140.0)
