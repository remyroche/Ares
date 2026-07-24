from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    replay_candidates,
)
from scripts.report_long_policy_replay_daily import (
    complete_daily_calendar,
    summarize,
    to_portfolio_candidates,
)


def test_summarize_keeps_prediction_provenance_separate() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z", "2026-07-11T00:00:00Z"]
            ),
            "prediction_provenance": ["oos", "oos", "forward"],
            "symbol": ["A", "B", "C"],
            "net_return_notional": [0.01, -0.02, 0.03],
            "gross_return_notional": [0.02, -0.01, 0.04],
            "fee_return": [0.01, 0.01, 0.01],
            "position_size": [0.1, 0.1, 0.1],
            "bankroll_pnl": [0.001, -0.002, 0.003],
            "exit_reason": ["trailing", "full_sl", "timeout"],
        }
    )

    daily = summarize(frame, "day")

    assert set(daily["prediction_provenance"]) == {"oos", "forward"}
    oos = daily.loc[daily["prediction_provenance"].eq("oos")].iloc[0]
    assert oos["trades"] == 2
    assert oos["avg_net_return_notional"] == -0.005
    assert oos["full_stop_rate"] == 0.5


def test_complete_daily_calendar_materializes_no_trade_days() -> None:
    daily = pd.DataFrame(
        {
            "prediction_provenance": ["oos"],
            "period": ["2026-07-02"],
            "trades": [3],
            "symbols": [2],
            "avg_net_return_notional": [0.01],
            "bankroll_pnl": [0.02],
        }
    )
    out = complete_daily_calendar(
        daily,
        start=pd.Timestamp("2026-07-01", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-07-04", tz="UTC"),
        prediction_provenance="oos",
    )

    assert out["period"].tolist() == ["2026-07-01", "2026-07-02", "2026-07-03"]
    assert out["trades"].tolist() == [0, 3, 0]
    assert out["avg_net_return_notional"].tolist() == [0.0, 0.01, 0.0]


def _simulated_trade() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-01T00:00:00Z"]),
            "decision_timestamp": pd.to_datetime(["2026-07-01T01:00:00Z"]),
            "exit_timestamp": pd.to_datetime(["2026-07-01T02:00:00Z"]),
            "symbol": ["BTC/USD"],
            "side_name": ["long"],
            "strategy_id": ["long_btc"],
            "rank_pct": [0.97],
            "entry_price": [100.0],
            "exit_price": [102.0],
            "gross_return_notional": [0.02],
            "net_return_notional": [0.01],
            "fee_return": [0.01],
            "exit_reason": ["trailing"],
            "policy_archetype": ["breakout"],
        }
    )


def test_to_portfolio_candidates_maps_simulated_trade_contract() -> None:
    candidates = to_portfolio_candidates(
        _simulated_trade(), base_strategy_threshold=0.90
    )
    row = candidates.iloc[0]

    assert row["timestamp"] == pd.Timestamp("2026-07-01T01:00:00Z")
    assert row["signal_timestamp"] == pd.Timestamp("2026-07-01T00:00:00Z")
    assert row["side"] == "long"
    assert row["normalized_rank_score"] == pytest.approx(0.97)
    assert row["base_strategy_threshold"] == pytest.approx(0.90)
    assert row["simple_policy_exit_reason"] == "trailing"
    assert row["holding_bars"] == 4


def test_portfolio_adapter_preserves_already_costed_returns() -> None:
    candidates = to_portfolio_candidates(
        _simulated_trade(), base_strategy_threshold=0.50
    )
    params = PortfolioPolicyParams(
        max_new_entries_per_bar=1,
        global_threshold_floor=0.50,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
        cooldown_hours_after_loss=0.0,
    )

    decisions, _, metrics = replay_candidates(candidates, params, market_mode="perps")
    accepted = decisions.loc[decisions["accepted"]].iloc[0]

    assert accepted["position_net_return"] == pytest.approx(0.01)
    assert accepted["position_gross_return"] == pytest.approx(0.02)
    assert metrics["net_pnl"] == pytest.approx(accepted["position_size"] * 0.01)
    assert metrics["gross_pnl"] == pytest.approx(accepted["position_size"] * 0.02)
