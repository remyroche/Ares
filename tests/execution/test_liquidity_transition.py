from __future__ import annotations

import pandas as pd

from scripts.execution.evaluate_liquidity_transition import _causal_feature_groups, build_lambdarank_spread_target
from src.execution.liquidity_transition import (
    add_actual_position_book_cost,
    add_causal_orderbook_flow_features,
    add_causal_trade_activity_features,
    join_causal_btc_benchmark_context,
    join_causal_context,
    join_causal_trade_activity_recap,
)


def test_actual_position_cost_interpolates_but_never_extrapolates() -> None:
    frame = pd.DataFrame({
        "position_notional": [100.0, 300.0, 600.0],
        "sell_book_cost_bps_n100": [10.0, 10.0, 10.0],
        "sell_book_cost_bps_n500": [30.0, 30.0, 30.0],
    })
    result = add_actual_position_book_cost(frame)
    assert result.loc[0, "book_cost_for_actual_position_bps"] == 10.0
    assert result.loc[1, "book_cost_for_actual_position_bps"] == 20.0
    assert pd.isna(result.loc[2, "book_cost_for_actual_position_bps"])
    assert not bool(result.loc[2, "book_cost_actual_position_in_grid"])


def test_context_join_is_backward_only_and_stale_values_are_null() -> None:
    panel = pd.DataFrame({
        "symbol": ["ETH/USD", "ETH/USD"],
        "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:03:00Z"]),
    })
    context = pd.DataFrame({
        "symbol": ["ETH/USD", "ETH/USD"],
        "available_ts": pd.to_datetime(["2026-01-01T00:02:00Z", "2026-01-01T00:03:00Z"]),
        "oi_context": [1.0, 2.0],
    })
    result = join_causal_context(panel, context, max_age=pd.Timedelta(minutes=2))
    assert pd.isna(result.loc[0, "oi_context"])
    assert result.loc[1, "oi_context"] == 2.0


def test_feature_groups_exclude_every_future_label_family() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD"], "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "spread_bps": [12.0], "spread_widening_5m": [100.0],
        "spread_bps_future_5m": [112.0], "deterioration_sell_5m_n100": [50.0],
    })
    groups = _causal_feature_groups(frame)
    selected = {column for columns in groups.values() for column in columns}
    assert "spread_bps" in selected
    assert not {"spread_widening_5m", "spread_bps_future_5m", "deterioration_sell_5m_n100"}.intersection(selected)


def test_retained_book_flow_rates_are_a_separate_grouped_mda_family() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD"], "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "book_flow_imbalance_50bps": [.1], "bid_cancel_rate_50bps": [.2],
        "spread_bps_change_1m": [1.0],
    })
    groups = _causal_feature_groups(frame)
    assert {"book_flow_imbalance_50bps", "bid_cancel_rate_50bps"}.issubset(groups["book_flow_rates"])
    assert "spread_bps_change_1m" in groups["book_transition"]


def test_btc_benchmark_is_its_own_grouped_mda_family() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD"], "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "btc_ret_5m": [.01], "market_return_5m": [.02],
    })
    groups = _causal_feature_groups(frame)
    assert groups["btc_benchmark"] == ["btc_ret_5m"]
    assert groups["market_cross_asset"] == ["market_return_5m"]


def test_executed_activity_fields_are_not_mixed_with_book_transition_mda() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD"], "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "trade_intensity_change_1m": [1.0],
        "sell_order_flow_imbalance_change_1m": [.1],
        "volume_ratio_5m": [2.0],
    })
    groups = _causal_feature_groups(frame)
    assert set(groups["trade_flow"]) == {
        "trade_intensity_change_1m", "sell_order_flow_imbalance_change_1m", "volume_ratio_5m",
    }


def test_book_flow_rates_use_only_prior_contiguous_displayed_depth() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD", "ETH/USD", "ETH/USD"],
        "state_minute": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "2026-01-01T00:03:00Z",
        ]),
        "book_valid": [True, True, True],
        "bid_depth_50bps": [100.0, 150.0, 200.0],
        "ask_depth_50bps": [100.0, 100.0, 100.0],
        "bid_cancel_notional": [0.0, 20.0, 99.0],
        "ask_cancel_notional": [0.0, 10.0, 99.0],
        "bid_replenish_notional": [0.0, 5.0, 99.0],
        "ask_replenish_notional": [0.0, 25.0, 99.0],
    })
    result = add_causal_orderbook_flow_features(frame)
    assert result.loc[1, "bid_cancel_rate_50bps"] == 0.20
    assert result.loc[1, "ask_replenishment_rate_50bps"] == 0.25
    # Bid cancellations plus ask replenishment dominate at this minute.
    assert result.loc[1, "book_flow_imbalance_50bps"] > 0.0
    # The missing 00:02 minute must not be bridged.
    assert pd.isna(result.loc[2, "bid_cancel_rate_50bps"])


def test_btc_benchmark_is_joined_only_after_completed_candle_close() -> None:
    panel = pd.DataFrame({
        "symbol": ["ETH/USD", "ETH/USD"],
        "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"]),
    })
    benchmark = pd.DataFrame({
        "available_ts": pd.to_datetime(["2026-01-01T00:01:00.001Z"]),
        "btc_ret_1m": [.01], "btc_ret_5m": [.02], "btc_ret_15m": [.03],
    })
    result = join_causal_btc_benchmark_context(panel, benchmark)
    assert pd.isna(result.loc[0, "btc_ret_1m"])
    assert pd.isna(result.loc[1, "btc_ret_1m"])
    panel.loc[1, "state_minute"] = pd.Timestamp("2026-01-01T00:02:00Z")
    result = join_causal_btc_benchmark_context(panel, benchmark)
    assert result.loc[1, "btc_ret_1m"] == .01


def test_btc_benchmark_replaces_null_native_placeholder_columns() -> None:
    panel = pd.DataFrame({
        "symbol": ["ETH/USD"],
        "state_minute": pd.to_datetime(["2026-01-01T00:02:00Z"]),
        "btc_ret_1m": [float("nan")],
        "btc_ret_5m": [float("nan")],
        "btc_ret_15m": [float("nan")],
    })
    benchmark = pd.DataFrame({
        "available_ts": pd.to_datetime(["2026-01-01T00:01:00Z"]),
        "btc_ret_1m": [.01], "btc_ret_5m": [.02], "btc_ret_15m": [.03],
    })
    result = join_causal_btc_benchmark_context(panel, benchmark)
    assert result.loc[0, "btc_ret_1m"] == .01
    assert "btc_ret_1m_benchmark" not in result.columns


def test_trade_activity_recap_requires_availability_before_next_bar_decision() -> None:
    panel = pd.DataFrame({
        "symbol": ["ETH/USD", "ETH/USD"],
        "state_minute": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"]),
        "decision_ts": pd.to_datetime(["2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z"]),
    })
    activity = pd.DataFrame({
        "symbol": ["ETH/USD", "ETH/USD"],
        "state_minute": panel["state_minute"],
        "activity_available_ts": pd.to_datetime(["2026-01-01T00:00:59Z", "2026-01-01T00:02:01Z"]),
        "trade_quote_volume": [10.0, 20.0],
        "trade_intensity": [1.0, 2.0],
        "sell_order_flow_imbalance": [.1, -.1],
    })
    result = join_causal_trade_activity_recap(panel, activity)
    assert result.loc[0, "trade_quote_volume"] == 10.0
    assert pd.isna(result.loc[1, "trade_quote_volume"])
    assert bool(result.loc[1, "activity_stale_or_unavailable"])


def test_trade_activity_rolling_features_do_not_bridge_a_minute_gap() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD"] * 7,
        "state_minute": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z",
            "2026-01-01T00:03:00Z", "2026-01-01T00:04:00Z", "2026-01-01T00:05:00Z",
            "2026-01-01T00:07:00Z",
        ]),
        "trade_quote_volume": [10.0] * 7,
        "position_notional": [100.0] * 7,
    })
    result = add_causal_trade_activity_features(frame)
    assert result.loc[5, "volume_ratio_5m"] == 1.0
    assert result.loc[5, "position_to_quote_volume_5m"] == 2.0
    assert pd.isna(result.loc[6, "volume_ratio_5m"])
    assert pd.isna(result.loc[6, "position_to_quote_volume_5m"])


def test_zero_trade_minute_keeps_activity_state_and_finite_position_pressure() -> None:
    frame = pd.DataFrame({
        "symbol": ["ETH/USD"] * 6,
        "state_minute": pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="min"),
        "trade_quote_volume": [10.0, 10.0, 10.0, 10.0, 10.0, 0.0],
        "position_notional": [100.0] * 6,
    })
    result = add_causal_trade_activity_features(frame)
    assert bool(result.loc[5, "trade_quote_volume_zero_1m"])
    assert result.loc[5, "position_to_quote_volume_1m"] == 100.0
    assert bool(result.loc[5, "position_to_quote_volume_1m_capped"])


def test_lambdarank_targets_normalize_by_strictly_prior_asset_spread_state() -> None:
    rows: list[dict[str, object]] = []
    for offset, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="5min")):
        for symbol, spread, future in (
            ("BTC/USD", 2.0, 3.0 + offset), ("ETH/USD", 4.0, 6.0 + offset),
            ("SOL/USD", 8.0, 11.0 + offset), ("DOT/USD", 16.0, 20.0 + offset),
            ("ALT/USD", 40.0, 44.0 + offset),
        ):
            rows.append({
                "symbol": symbol, "state_minute": timestamp, "book_valid": True,
                "spread_bps": spread, "spread_bps_future_5m": future,
            })
    panel = build_lambdarank_spread_target(
        pd.DataFrame(rows), horizon_minutes=5, target_kind="spread_delta", asset_baseline_observations=4,
    )
    assert set(panel["rank_grade"].unique()).issubset({0, 1, 2, 3, 4})
    selected = {column for values in _causal_feature_groups(panel).values() for column in values}
    assert not {"rank_target_raw_bps", "rank_target_normalized", "rank_grade", "asset_spread_baseline_bps"}.intersection(selected)
    # Causal liquidity-profile context is usable at inference and is distinct
    # from the target-construction baseline itself.
    assert "spread_bps_to_asset_prior_median" in selected


def test_absolute_lambdarank_target_is_literal_future_spread_not_relative_spread() -> None:
    rows: list[dict[str, object]] = []
    for offset, timestamp in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="5min")):
        for symbol, spread, future in (
            ("BTC/USD", 2.0, 6.0 + offset), ("ETH/USD", 4.0, 12.0 + offset),
            ("SOL/USD", 8.0, 24.0 + offset), ("DOT/USD", 16.0, 48.0 + offset),
            ("ALT/USD", 40.0, 96.0 + offset),
        ):
            rows.append({
                "symbol": symbol, "state_minute": timestamp, "book_valid": True,
                "spread_bps": spread, "spread_bps_future_5m": future,
                "bid_depth_50bps": 1000.0 + future, "ask_depth_50bps": 900.0 + future,
            })
    panel = build_lambdarank_spread_target(
        pd.DataFrame(rows), horizon_minutes=5, target_kind="absolute_future_spread", asset_baseline_observations=4,
    )
    assert (panel["rank_target_normalized"] == panel["spread_bps_future_5m"]).all()
    assert (panel["rank_target_raw_bps"] == panel["spread_bps_future_5m"]).all()
    assert panel.loc[panel["symbol"].eq("ALT/USD"), "rank_grade"].gt(
        panel.loc[panel["symbol"].eq("BTC/USD"), "rank_grade"].iloc[0]
    ).all()
