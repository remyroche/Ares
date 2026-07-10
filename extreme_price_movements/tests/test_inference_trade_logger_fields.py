import json
import sqlite3

import pandas as pd

from extreme_price_movements.inference.trade_logger import TradeLogger


def test_trade_logger_includes_required_parity_fields(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    decision = {
        "symbol": "BTC/USDT",
        "side": "long",
        "action": "enter",
        "strategy_id": "long_mr",
        "calibrated_score": 0.72,
        "rank_threshold": 0.60,
        "stop_price": 98.0,
        "stop_price_updated": 99.0,
        "orderbook_snapshot": "top5",
        "net_pnl": 0.01,
        "entry_time": "2026-05-12T09:00:00Z",
        "exit_time": "2026-05-12T10:30:00Z",
        "holding_time_hours": 1.5,
        "fee_source": "verified_order_fees",
        "fees_verified": True,
        "net_pnl_estimated": 0.009,
        "estimated_fee_source": "configured_entry_market_fee_bps",
        "net_pnl_verification_status": "verified_exchange_fees",
    }
    model_results = {"meta_pred": 0.8, "position_size": 0.5, "alpha_preds": {}}
    market_data = {"close": 100.0, "volume": 1234.0}
    cfg = {"run_id": "r1", "mode": "shadow"}

    logger.log_trade(decision, model_results, market_data, cfg)
    df = pd.read_csv(path)

    required = {
        "trade_id",
        "position_id",
        "lifecycle_event",
        "strategy_id",
        "calibrated_score",
        "rank_threshold",
        "stop_price",
        "stop_price_updated",
        "orderbook_snapshot",
        "net_pnl",
        "entry_time",
        "exit_time",
        "holding_time_hours",
        "fee_source",
        "fees_verified",
        "net_pnl_estimated",
        "estimated_fee_source",
        "net_pnl_verification_status",
    }
    assert required.issubset(df.columns)
    assert df.loc[0, "strategy_id"] == "long_mr"
    assert df.loc[0, "position_id"]
    assert df.loc[0, "lifecycle_event"] == "entry_recorded"

    db_path = path.with_suffix(".sqlite")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT strategy_id, orderbook_snapshot, net_pnl, lifecycle_event, "
            "entry_time, exit_time, holding_time_hours, fee_source, fees_verified, "
            "net_pnl_estimated, estimated_fee_source, net_pnl_verification_status "
            "FROM trades"
        ).fetchall()
    assert rows == [
        (
            "long_mr",
            "top5",
            "0.01",
            "entry_recorded",
            "2026-05-12T09:00:00Z",
            "2026-05-12T10:30:00Z",
            "1.5",
            "verified_order_fees",
            "True",
            "0.009",
            "configured_entry_market_fee_bps",
            "verified_exchange_fees",
        )
    ]


def test_trade_logger_persists_policy_and_exchange_stop_observability(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    adjustment = {
        "status": "ok",
        "policy_stop_price": 0.17,
        "exchange_stop_price": 0.1696,
        "position_side": "short",
        "bid": 0.1617,
        "ask": 0.1621,
        "last": 0.1617,
        "spread": 0.0004,
        "last_to_executable_gap": 0.0004,
        "gap_source": "ask_minus_last",
        "trigger_signal": "last",
    }
    logger.log_trade(
        {
            "symbol": "DYDX/USD:USD",
            "side": "short",
            "action": "enter",
            "strategy_id": "short_asset",
            "stop_price": 0.17,
            "policy_stop_price": 0.17,
            "requested_policy_stop": 0.17,
            "exchange_stop_price": 0.1696,
            "final_placed_stop": 0.1696,
            "exchange_stop_trigger_reference_source": "last",
            "exchange_stop_adjustment": adjustment,
            "stop_trigger_signal": "last",
            "stop_order_id": "stop-1",
        },
        {"meta_pred": 0.8, "position_size": 7.0, "alpha_preds": {}},
        {"close": 0.1621},
        {"run_id": "r1", "mode": "live"},
    )

    df = pd.read_csv(path, dtype=str)
    assert df.loc[0, "stop_price"] == "0.17"
    assert df.loc[0, "policy_stop_price"] == "0.17"
    assert df.loc[0, "requested_policy_stop"] == "0.17"
    assert df.loc[0, "exchange_stop_price"] == "0.1696"
    assert df.loc[0, "final_placed_stop"] == "0.1696"
    assert df.loc[0, "exchange_stop_trigger_reference_source"] == "last"
    assert json.loads(df.loc[0, "exchange_stop_adjustment"]) == adjustment

    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        rows = conn.execute(
            "SELECT policy_stop_price, requested_policy_stop, exchange_stop_price, "
            "final_placed_stop, exchange_stop_trigger_reference_source, "
            "exchange_stop_adjustment FROM trades"
        ).fetchall()
    assert rows[0][:5] == ("0.17", "0.17", "0.1696", "0.1696", "last")
    assert json.loads(rows[0][5]) == adjustment


def test_trade_logger_derives_entry_notional_for_fee_audit(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    decision = {
        "symbol": "PUMP/USD:USD",
        "side": "long",
        "action": "enter",
        "strategy_id": "long_bars",
        "intended_quote_size": 7.0,
        "realized_entry_price": 0.001483,
        "base_amount": 4900.0,
        "entry_fee_estimate_quote": 0.0035,
        "entry_fee_estimate_bps": 5.0,
        "entry_fee_estimate_source": "env_EPM_LIVE_FEE_FALLBACK_BPS_entry_limit",
    }
    logger.log_trade(
        decision,
        {"meta_pred": 0.8, "position_size": 7.0, "alpha_preds": {}},
        {"close": 0.001483},
        {"run_id": "r1", "mode": "live-test"},
    )

    df = pd.read_csv(path)
    assert float(df.loc[0, "entry_notional_quote"]) == 7.0
    assert float(df.loc[0, "entry_fee_estimate_quote"]) == 0.0035

    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        rows = conn.execute(
            "SELECT entry_notional_quote, entry_fee_estimate_quote, "
            "entry_fee_estimate_bps FROM trades"
        ).fetchall()
    assert rows == [("7.0", "0.0035", "5.0")]


def test_trade_logger_reads_sqlite_and_repairs_csv_schema(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")
    logger.log_entry(
        symbol="BTC/USDT",
        side="long",
        size=10.0,
        price=100.0,
        predictions={"meta_pred": 0.8},
        features={"strategy_id": "long_mr"},
        mode="live-test",
    )

    # Simulate historical CSV schema drift/corruption. The sqlite record remains
    # canonical and the next logger init should regenerate the CSV from it.
    path.write_text("timestamp,run_id,symbol\n,0.8,0.1\n")
    logger = TradeLogger(output_path=str(path), run_id="r1")
    df = logger.read_logs()

    assert list(pd.read_csv(path).columns) == logger.columns
    assert df.loc[0, "symbol"] == "BTC/USDT"
    assert df.loc[0, "meta_pred"] == "0.8"
    assert df.loc[0, "trade_id"]
    assert pd.notna(pd.to_datetime(df.loc[0, "timestamp"], errors="coerce"))


def test_trade_logger_persists_holding_time_to_csv_and_sqlite(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    logger.log_trade_legacy(
        symbol="ETH/USDT",
        side="short",
        action="exit",
        size=2.0,
        price=2000.0,
        status="closed",
        context={
            "lifecycle_event": "exit_filled",
            "entry_time": "2026-05-12T09:00:00Z",
            "exit_time": "2026-05-12T11:30:00Z",
            "holding_time_hours": 2.5,
            "net_pnl_amount": 10.0,
        },
    )

    df = pd.read_csv(path)
    assert "holding_time_hours" in df.columns
    assert float(df.loc[0, "holding_time_hours"]) == 2.5

    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        rows = conn.execute(
            'SELECT holding_time_hours FROM trades WHERE symbol = "ETH/USDT"'
        ).fetchall()
    assert rows == [("2.5",)]


def test_trade_logger_promotes_estimated_net_fields_for_exit_rows(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    logger.log_trade_legacy(
        symbol="UNI/USD:USD",
        side="short",
        action="exit",
        size=3.0,
        price=3.528,
        status="closed",
        context={
            "lifecycle_event": "exit_filled",
            "entry_time": "2026-07-10T07:12:00Z",
            "exit_time": "2026-07-10T07:30:00Z",
            "gross_pnl_pct": 0.001984,
            "net_pnl_estimated": 0.010427,
            "net_pnl_pct_estimated": 0.000985,
            "estimated_fees_amount": 0.010573,
            "estimated_fee_source": (
                "default_live_perp_fee_bps_entry_market"
                "+default_live_perp_fee_bps_exit_market"
            ),
            "gross_to_estimated_net_cost_quote": 0.010573,
            "gross_to_estimated_net_cost_pct": 0.000999,
            "gross_to_estimated_net_friction_drag_bps": 9.99,
            "net_pnl_verification_status": "estimated_missing_exchange_fees",
        },
    )

    df = pd.read_csv(path)
    assert float(df.loc[0, "net_pnl"]) == 0.010427
    assert float(df.loc[0, "net_pnl_amount"]) == 0.010427
    assert float(df.loc[0, "net_pnl_pct"]) == 0.000985
    assert float(df.loc[0, "fees_amount"]) == 0.010573
    assert str(df.loc[0, "fees_estimated"]) == "True"
    assert df.loc[0, "net_pnl_verification_status"] == (
        "estimated_missing_exchange_fees"
    )

    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        rows = conn.execute(
            'SELECT net_pnl, net_pnl_amount, net_pnl_pct, fees_amount '
            'FROM trades WHERE symbol = "UNI/USD:USD"'
        ).fetchall()
    assert rows == [("0.010427", "0.010427", "0.000985", "0.010573")]


def test_trade_logger_derives_holding_time_from_entry_and_exit_times(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    logger.log_trade_legacy(
        symbol="ETH/USDT",
        side="short",
        action="exit",
        size=2.0,
        price=2000.0,
        status="closed",
        context={
            "lifecycle_event": "exit_filled",
            "entry_time": "2026-05-12T09:00:00Z",
            "exit_time": "2026-05-12T11:30:00Z",
            "net_pnl_amount": 10.0,
        },
    )

    df = pd.read_csv(path)
    assert float(df.loc[0, "holding_time_hours"]) == 2.5

    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        rows = conn.execute(
            'SELECT holding_time_hours FROM trades WHERE symbol = "ETH/USDT"'
        ).fetchall()
    assert rows == [("2.5",)]
