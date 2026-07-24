import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.l2_execution_confirmation import (
    L2ConfirmationConfig,
    apply_confirmed_l2_cost,
    confirm_l2_execution,
    summarise_l2_confirmation,
)


def _snapshot(ts, symbol="BTC/USD:USD"):
    rows = []
    for side, levels in (
        ("bid", [(99.0, 2.0), (98.5, 5.0)]),
        ("ask", [(101.0, 1.0), (101.5, 5.0)]),
    ):
        for level, (price, qty) in enumerate(levels, 1):
            rows.append(
                {
                    "observed_ts": pd.Timestamp(ts),
                    "symbol": symbol,
                    "side": side,
                    "level": level,
                    "price": price,
                    "qty": qty,
                }
            )
    return rows


def test_confirmation_uses_latest_causally_prior_snapshot_and_walks_exact_depth():
    books = pd.DataFrame(
        _snapshot("2026-07-01T10:00:00Z")
        + _snapshot("2026-07-01T10:30:00Z")
        + _snapshot("2026-07-01T11:01:00Z")  # future for exit; must not leak
    )
    trades = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD"],
            "side": ["long"],
            "entry_ts": [pd.Timestamp("2026-07-01T10:05:00Z")],
            "exit_ts": [pd.Timestamp("2026-07-01T11:00:00Z")],
            "admitted_quote_notional": [202.5],
        }
    )
    out = confirm_l2_execution(
        trades,
        books,
        config=L2ConfirmationConfig(max_walk_slippage_bps=100.0),
    ).iloc[0]

    assert out["l2_entry_snapshot_observed_ts"] == pd.Timestamp("2026-07-01T10:00:00Z")
    assert out["l2_exit_snapshot_observed_ts"] == pd.Timestamp("2026-07-01T10:30:00Z")
    assert out["l2_roundtrip_covered"]
    # Entry consumes 101 quote at 101 and 101.5 quote at 101.5.
    expected_entry = 202.5 / (1.0 + 101.5 / 101.5)
    assert out["l2_entry_fill_price"] == pytest.approx(expected_entry)
    assert out["l2_entry_capacity_quote"] == pytest.approx(608.5)
    assert out["l2_exit_capacity_quote"] == pytest.approx(690.5)
    assert out["l2_roundtrip_depth_slippage_bps"] == pytest.approx(
        out["l2_entry_depth_slippage_bps"] + out["l2_exit_depth_slippage_bps"]
    )


def test_uncovered_or_stale_trades_receive_no_extrapolated_cost():
    books = pd.DataFrame(_snapshot("2026-07-01T10:00:00Z"))
    trades = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": ["short", "long"],
            "entry_ts": pd.to_datetime(["2026-07-01T12:00:01Z", "2026-07-01T10:01:00Z"]),
            "exit_ts": pd.to_datetime(["2026-07-01T12:01:00Z", "2026-07-01T10:02:00Z"]),
            "admitted_quote_notional": [100.0, 100.0],
        }
    )
    out = confirm_l2_execution(trades, books)

    assert not out["l2_roundtrip_covered"].any()
    assert not out["l2_roundtrip_snapshot_covered"].any()
    assert out["l2_roundtrip_depth_slippage_bps"].isna().all()
    assert "stale_snapshot" in out.iloc[0]["l2_confirmation_reason"]
    assert "no_causally_prior_snapshot" in out.iloc[1]["l2_confirmation_reason"]


def test_actual_quantity_is_supported_and_insufficient_depth_is_not_covered():
    books = pd.DataFrame(_snapshot("2026-07-01T10:00:00Z"))
    trades = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD"],
            "side": ["long"],
            "entry_ts": [pd.Timestamp("2026-07-01T10:01:00Z")],
            "exit_ts": [pd.Timestamp("2026-07-01T10:02:00Z")],
            "admitted_quantity": [100.0],
        }
    )
    out = confirm_l2_execution(trades, books).iloc[0]

    assert out["l2_admitted_quantity"] == pytest.approx(100.0)
    assert not out["l2_entry_covered"]
    assert not out["l2_exit_covered"]
    assert out["l2_roundtrip_snapshot_covered"]
    assert out["l2_confirmation_reason"] == "insufficient_capacity:entry,exit"
    assert np.isnan(out["l2_roundtrip_depth_slippage_bps"])


def test_exit_uses_held_quantity_when_quote_and_quantity_are_both_available():
    books = pd.DataFrame(_snapshot("2026-07-01T10:00:00Z"))
    trades = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD"],
            "side": ["long"],
            "entry_ts": [pd.Timestamp("2026-07-01T10:01:00Z")],
            "exit_ts": [pd.Timestamp("2026-07-01T10:02:00Z")],
            "admitted_quote_notional": [100.0],
            "admitted_quantity": [3.0],
        }
    )
    out = confirm_l2_execution(trades, books).iloc[0]

    assert out["l2_entry_capacity_ratio"] == pytest.approx(608.5 / 100.0)
    assert out["l2_exit_capacity_ratio"] == pytest.approx(690.5 / (3.0 * 99.0))


def test_summary_uses_confirmed_subset_only():
    books = pd.DataFrame(_snapshot("2026-07-01T10:00:00Z"))
    trades = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "UNKNOWN"],
            "side": ["long", "long"],
            "entry_ts": pd.to_datetime(["2026-07-01T10:01:00Z"] * 2),
            "exit_ts": pd.to_datetime(["2026-07-01T10:02:00Z"] * 2),
            "admitted_quote_notional": [100.0, 100.0],
        }
    )
    diagnostics = confirm_l2_execution(trades, books)
    summary = summarise_l2_confirmation(diagnostics)

    assert summary["trade_count"] == 2
    assert summary["confirmed_trade_count"] == 1
    assert summary["roundtrip_coverage_rate"] == pytest.approx(0.5)
    assert np.isfinite(summary["mean_roundtrip_depth_slippage_bps"])


def test_cost_application_changes_only_roundtrip_confirmed_rows():
    books = pd.DataFrame(_snapshot("2026-07-01T10:00:00Z"))
    trades = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "UNKNOWN"],
            "side": ["long", "long"],
            "entry_ts": pd.to_datetime(["2026-07-01T10:01:00Z"] * 2),
            "exit_ts": pd.to_datetime(["2026-07-01T10:02:00Z"] * 2),
            "admitted_quote_notional": [150.0, 150.0],
        },
        index=[10, 20],
    )
    diagnostics = confirm_l2_execution(trades, books)
    baseline = pd.Series([0.02, 0.03], index=trades.index)
    adjusted = apply_confirmed_l2_cost(baseline, diagnostics)

    expected = 0.02 - diagnostics.loc[10, "l2_roundtrip_depth_slippage_bps"] / 10000.0
    assert adjusted.loc[10, "l2_adjusted_net_return"] == pytest.approx(expected)
    assert adjusted.loc[10, "l2_cost_applied"]
    assert adjusted.loc[20, "l2_adjusted_net_return"] == 0.03
    assert not adjusted.loc[20, "l2_cost_applied"]
    assert np.isnan(adjusted.loc[20, "l2_incremental_cost_bps"])
