"""Place a tiny Binance margin live-test order and replace its stop loss.

This script uses the inference exchange and executor stack so API plumbing is
verified through the same code path used by deployment.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from extreme_price_movements.inference.config import (
    DEFAULT_EXECUTION_ACCOUNT,
    DEFAULT_MARGIN_MODE,
)
from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.trade_executor import TradeExecutor
from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.utils import tprint


def _jsonable(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str, sort_keys=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDC")
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--quote-size", type=float, default=10.0)
    parser.add_argument("--initial-stop-pct", type=float, default=0.003)
    parser.add_argument("--replacement-stop-pct", type=float, default=0.005)
    parser.add_argument("--quote-currency", default="USDC")
    parser.add_argument(
        "--execution-account",
        default=DEFAULT_EXECUTION_ACCOUNT,
        choices=["spot", "margin"],
    )
    parser.add_argument(
        "--margin-mode",
        default=DEFAULT_MARGIN_MODE,
        choices=["cross", "isolated"],
    )
    args = parser.parse_args()

    config = {
        "execution_account": args.execution_account,
        "margin_mode": args.margin_mode,
        "monitor_interval_seconds": 300,
        "live_quote_currency": args.quote_currency.upper(),
    }
    bucket_key = "manual_live_test"
    bucket_params = {
        bucket_key: {
            "fixed_stop_loss_pct": float(args.initial_stop_pct),
            "enable_trailing": False,
            "cooldown_hours": 0.0,
        }
    }

    exchange = make_exchange()
    portfolio = PortfolioManager(max_positions=4)
    logger = TradeLogger()
    before = portfolio.fetch_exchange_snapshot(
        exchange,
        quote_currency=args.quote_currency,
        execution_account=args.execution_account,
        margin_mode=args.margin_mode,
    )
    tprint("[live_margin_stop_smoke] balance_before=" + _jsonable(before))

    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params=bucket_params,
        config=config,
    )
    try:
        entry = executor.execute_trade(
            symbol=args.symbol,
            side=args.side,
            size=float(args.quote_size),
            price=None,
            bucket_key=bucket_key,
        )
        tprint("[live_margin_stop_smoke] entry_result=" + _jsonable(entry))
        if not bool(entry.get("success")):
            raise SystemExit(2)

        statuses = executor.monitor_orders_once()
        tprint("[live_margin_stop_smoke] order_statuses=" + _jsonable(statuses))

        replacement = executor.replace_stop_loss_pct(
            args.symbol,
            float(args.replacement_stop_pct),
        )
        tprint("[live_margin_stop_smoke] stop_replacement=" + _jsonable(replacement))
        if not bool(replacement.get("success")):
            raise SystemExit(3)

        logger.log_entry(
            symbol=args.symbol,
            side=args.side,
            size=float(args.quote_size),
            price=entry.get("realized_entry_price"),
            predictions={
                "position_size": float(args.quote_size),
                "sizing_source": "manual_live_margin_stop_smoke",
            },
            mode="live-test",
            strategy_id=bucket_key,
            expected_entry_price=entry.get("expected_entry_price"),
            realized_entry_price=entry.get("realized_entry_price"),
            actual_entry_price=entry.get("realized_entry_price"),
            stop_price=replacement.get("stop_price"),
            stop_order_id=replacement.get("stop_order_id"),
            exchange_order_id=(entry.get("order") or {}).get("id"),
            order_error_category=entry.get("error_category", ""),
            status="pending",
            error=entry.get("error", ""),
        )

        after = portfolio.fetch_exchange_snapshot(
            exchange,
            quote_currency=args.quote_currency,
            execution_account=args.execution_account,
            margin_mode=args.margin_mode,
        )
        tprint("[live_margin_stop_smoke] balance_after=" + _jsonable(after))
    finally:
        if executor.oco_executor is not None:
            executor.oco_executor.stop_monitoring()


if __name__ == "__main__":
    main()
