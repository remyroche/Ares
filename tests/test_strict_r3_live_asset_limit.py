from __future__ import annotations

import inspect

import pandas as pd
import pytest

import extreme_price_movements.inference.strict_r3_live_execution as live_execution

from extreme_price_movements.inference.strict_r3_live_execution import (
    _exclude_open_symbol_admissions,
)


def test_open_asset_is_removed_before_live_execution_preflight():
    admitted = pd.DataFrame([
        {
            "candidate_id": "PUMP/USD:USD|long|new",
            "__symbol__": "PUMP/USD:USD",
            "mc1_d2_expected_net_bps": 150.0,
        },
        {
            "candidate_id": "SUSHI/USD:USD|long|new",
            "__symbol__": "SUSHI/USD:USD",
            "mc1_d2_expected_net_bps": 140.0,
        },
    ])
    retained, rejected = _exclude_open_symbol_admissions(
        admitted,
        open_symbols={"PUMP/USD:USD"},
    )
    assert retained["candidate_id"].tolist() == ["SUSHI/USD:USD|long|new"]
    assert rejected == [{
        "action": "entry_rejected_portfolio_asset_limit",
        "candidate_id": "PUMP/USD:USD|long|new",
        "symbol": "PUMP/USD:USD",
        "reason": "symbol_already_open",
        "mapped_expected_net_bps": 150.0,
    }]


def test_duplicate_admitted_symbol_is_retained_only_once():
    admitted = pd.DataFrame([
        {"candidate_id": "A", "__symbol__": "A/USD:USD", "mc1_d2_expected_net_bps": 120.0},
        {"candidate_id": "B", "__symbol__": "A/USD:USD", "mc1_d2_expected_net_bps": 110.0},
    ])
    retained, rejected = _exclude_open_symbol_admissions(admitted, open_symbols=set())
    assert retained["candidate_id"].tolist() == ["A"]
    assert rejected[0]["candidate_id"] == "B"
    assert rejected[0]["reason"] == "symbol_already_open"


def test_minute_monitor_is_entry_free():
    """The asset admission gate belongs to the entry executor, never the monitor."""
    source = inspect.getsource(live_execution.monitor_live_positions_once)
    assert "admitted_count" not in source
    assert "_exclude_open_symbol_admissions" not in source


def _tracked_position():
    return {
        "exchange_symbol": "PORTAL/USD:USD",
        "side": "long",
        "amount": 10.0,
        "entry_ts": "2026-08-16T14:00:00Z",
    }


def test_full_private_liquidation_sequence_is_confirmed_not_mislabelled_stop():
    class Exchange:
        def fetch_my_trades(self, symbol, *, since, limit):
            assert symbol == "PORTAL/USD:USD"
            assert limit == 100
            return [
                {
                    "side": "sell", "amount": 4.0, "price": 0.0138,
                    "timestamp": 1786889100000,
                    "order": "liq-1",
                    "info": {"fillType": "partialLiquidation"},
                },
                {
                    "side": "sell", "amount": 6.0, "price": 0.0136,
                    "timestamp": 1786889160000,
                    "order": "liq-2",
                    "info": {"fillType": "partialLiquidation"},
                },
            ]

    result = live_execution._confirmed_exchange_liquidation(
        Exchange(), position=_tracked_position(),
    )

    assert result["resolved_via"] == "fetch_my_trades_full_liquidation"
    assert result["filled_amount"] == 10.0
    assert result["fill_price"] == pytest.approx(0.01368)
    assert result["liquidation_fill_count"] == 2
    assert result["order"]["status"] == "closed"


def test_ordinary_opposite_side_fill_never_reconciles_as_liquidation():
    class Exchange:
        def fetch_my_trades(self, symbol, *, since, limit):
            return [{
                "side": "sell", "amount": 10.0, "price": 0.0138,
                "timestamp": 1786889100000,
                "info": {"fillType": "taker"},
            }]

    with pytest.raises(ValueError, match="no complete opposite-side liquidation"):
        live_execution._confirmed_exchange_liquidation(
            Exchange(), position=_tracked_position(),
        )


def test_exchange_absent_exit_prefers_protective_then_verified_liquidation(monkeypatch):
    protective = {
        "order_id": "stop-1", "fill_price": 0.0137, "filled_amount": 10.0,
        "order": {"status": "closed", "filled": 10.0, "average": 0.0137},
    }
    monkeypatch.setattr(
        live_execution,
        "_confirmed_filled_protective_order",
        lambda _exchange, *, position: protective,
    )
    result, kind = live_execution._confirmed_exchange_absent_exit(
        object(), position=_tracked_position(),
    )

    assert kind == "protective"
    assert result == protective


def test_mixed_ordinary_and_liquidation_fills_reconcile_only_when_exact():
    class Exchange:
        def fetch_my_trades(self, symbol, *, since, limit):
            assert symbol == "PORTAL/USD:USD"
            return [
                {
                    "side": "sell", "amount": 2.0, "price": 0.0139,
                    "timestamp": 1786889100000,
                    "info": {"fillType": "taker"},
                },
                {
                    "side": "sell", "amount": 8.0, "price": 0.0135,
                    "timestamp": 1786889160000,
                    "info": {"fillType": "partialLiquidation"},
                },
            ]

    result, kind = live_execution._confirmed_exchange_absent_exit(
        Exchange(), position=_tracked_position(),
    )

    assert kind == "mixed_liquidation_external_exit"
    assert result["resolved_via"] == "fetch_my_trades_full_mixed_liquidation_external_exit"
    assert result["filled_amount"] == 10.0
    assert result["mixed_exit_fill_types"] == ["partialliquidation", "taker"]
