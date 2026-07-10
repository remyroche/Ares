import pandas as pd

from extreme_price_movements.scripts.live_closed_trade_exit_replay import (
    _logged_live_exchange_stop_fill,
    _logged_live_software_handoff_exit,
    _select_closed_trade_rows,
)


def test_select_closed_trade_rows_keeps_latest_rows_after_symbol_filter() -> None:
    closed = pd.DataFrame(
        [
            {
                "symbol": "AAVE/USD:USD",
                "entry_time": "2026-07-10T10:00:00Z",
                "exit_time": "2026-07-10T10:15:00Z",
            },
            {
                "symbol": "FIL/USD:USD",
                "entry_time": "2026-07-10T11:00:00Z",
                "exit_time": "2026-07-10T11:15:00Z",
            },
            {
                "symbol": "HOODX/USD:USD",
                "entry_time": "2026-07-10T12:00:00Z",
                "exit_time": "2026-07-10T12:15:00Z",
            },
            {
                "symbol": "AAVE/USD:USD",
                "entry_time": "2026-07-10T13:00:00Z",
                "exit_time": "2026-07-10T13:15:00Z",
            },
        ]
    )

    selected = _select_closed_trade_rows(
        closed,
        symbols="AAVE/USD:USD,FIL/USD:USD,HOODX/USD:USD",
        limit=2,
    )

    assert selected["symbol"].tolist() == ["HOODX/USD:USD", "AAVE/USD:USD"]
    assert selected["exit_time"].tolist() == [
        "2026-07-10T12:15:00Z",
        "2026-07-10T13:15:00Z",
    ]


def test_logged_live_software_handoff_exit_is_replayable() -> None:
    row = {
        "reason": (
            "software_executable_stop_breach_pretrigger:"
            "exchange_valid_giveback_fallback_handoff"
        ),
        "close_trigger_type": "software_bid_ask_sentinel",
        "close_execution_method": "ask_bid_software_close",
        "exit_time": "2026-07-10T16:16:17Z",
        "exit_price": "0.804",
    }

    event = _logged_live_software_handoff_exit(row)

    assert event is not None
    assert event["status"] == "logged_live_software_handoff"
    assert event["price"] == 0.804
    assert event["ts"].isoformat() == "2026-07-10T16:16:17+00:00"


def test_logged_live_exchange_stop_fill_is_replayable_from_closed_trade() -> None:
    row = {
        "reason": "stop_loss_filled:trailing_risk_reduction",
        "close_trigger_type": "exchange_stop_order",
        "close_price_source": "exchange_stop_order_fill",
        "exit_time": "2026-07-10T11:49:20Z",
        "exit_price": "68.983",
    }

    event = _logged_live_exchange_stop_fill(row)

    assert event is not None
    assert event["status"] == "logged_live_exchange_stop_fill_from_closed_trade"
    assert event["reason"] == "stop_loss_filled:trailing_risk_reduction"
    assert event["price"] == 68.983
