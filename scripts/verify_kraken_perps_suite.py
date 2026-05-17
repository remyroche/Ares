#!/usr/bin/env python3
"""Verify Kraken Futures perp readiness.

Default mode is read-only. Live order lifecycle is opt-in with
--execute-trading-tests --confirm-live-orders.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    _load_local_env_if_present,
    make_perp_exchange,
    normalize_orderbook_proxy_frame,
)
from extreme_price_movements.inference.trade_executor import (
    _cancel_open_protective_stop_orders,
    _create_reduce_market_order,
    _create_reduce_stop_loss_order,
    _fetch_open_protective_stop_orders,
)


ORDERBOOK_PROXY_COLUMNS = [
    "best_bid",
    "best_ask",
    "mid",
    "bid_qty_1",
    "ask_qty_1",
    "cum_bid_qty_l10",
    "cum_ask_qty_l10",
    "cum_bid_qty_l20",
    "cum_ask_qty_l20",
    "snapshot_ts",
    "trade_count_1h",
    "buy_qty_1h",
    "sell_qty_1h",
    "notional_1h",
    "buy_notional_1h",
    "sell_notional_1h",
    "vwap_1h",
    "mean_trade_qty_1h",
    "signed_flow_imbalance_1h",
    "source",
]


def _ok(result: Dict[str, Any], name: str, **details: Any) -> None:
    result[name] = {"ok": True, **details}


def _fail(result: Dict[str, Any], name: str, exc: Exception) -> None:
    result[name] = {
        "ok": False,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _records(df: pd.DataFrame, columns: Iterable[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    sample = df[list(columns)].tail(1).reset_index()
    return json.loads(sample.to_json(orient="records", date_format="iso"))


def _order_id(order: Any) -> Any:
    if not isinstance(order, dict):
        return None
    return order.get("id") or order.get("orderId") or (order.get("info") or {}).get("order_id")


def _normalize_orderbook(orderbook: Dict[str, Any]) -> pd.DataFrame:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        raise ValueError("orderbook has no bids or asks")
    bid = float(bids[0][0])
    ask = float(asks[0][0])
    mid = (bid + ask) / 2.0
    bid_sizes = [float(level[1]) for level in bids if len(level) > 1]
    ask_sizes = [float(level[1]) for level in asks if len(level) > 1]
    ts = (
        pd.to_datetime(orderbook.get("timestamp"), unit="ms", utc=True)
        if orderbook.get("timestamp")
        else pd.Timestamp.now(tz="UTC")
    )
    row = {
        "best_bid": bid,
        "best_ask": ask,
        "mid": mid,
        "bid_qty_1": bid_sizes[0] if bid_sizes else 0.0,
        "ask_qty_1": ask_sizes[0] if ask_sizes else 0.0,
        "cum_bid_qty_l10": float(sum(bid_sizes[:10])),
        "cum_ask_qty_l10": float(sum(ask_sizes[:10])),
        "cum_bid_qty_l20": float(sum(bid_sizes[:20])),
        "cum_ask_qty_l20": float(sum(ask_sizes[:20])),
        "snapshot_ts": ts,
        "source": "live_l2_orderbook",
    }
    return normalize_orderbook_proxy_frame(pd.DataFrame([row], index=[ts.floor("1h")]))


def _wallet_from_balance(balance: Dict[str, Any]) -> Dict[str, Any]:
    info = balance.get("info") if isinstance(balance.get("info"), dict) else {}
    accounts = info.get("accounts") if isinstance(info.get("accounts"), dict) else {}
    flex = accounts.get("flex") if isinstance(accounts.get("flex"), dict) else {}
    return {
        "full_wallet_usd": _safe_float(
            flex.get("marginEquity"),
            _safe_float(flex.get("portfolioValue"), _safe_float(flex.get("balanceValue"), 0.0)),
        ),
        "available_wallet_usd": _safe_float(flex.get("availableMargin"), 0.0),
        "collateral_value_usd": _safe_float(flex.get("collateralValue"), 0.0),
        "total": balance.get("total"),
        "free": balance.get("free"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH/USD:USD")
    parser.add_argument("--quote-notional", type=float, default=5.0)
    parser.add_argument("--stop-pct", type=float, default=0.03)
    parser.add_argument("--replacement-stop-pct", type=float, default=0.02)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--execute-trading-tests", action="store_true")
    parser.add_argument("--confirm-live-orders", action="store_true")
    args = parser.parse_args()

    _load_local_env_if_present()
    os.environ["EPM_EXCHANGE"] = "kraken"
    exchange = make_perp_exchange()
    symbol = args.symbol
    result: Dict[str, Any] = {
        "exchange_id": getattr(exchange, "id", None),
        "symbol_requested": args.symbol,
        "trading_tests_executed": False,
    }

    try:
        market = exchange.market(symbol)
        result["symbol"] = symbol
        result["market"] = {
            key: market.get(key)
            for key in ("id", "symbol", "type", "swap", "linear", "inverse", "settle", "quote", "base")
        }
        result["market"]["tradeable"] = (market.get("info") or {}).get("tradeable")
        _ok(result, "load_markets", market_count=len(getattr(exchange, "markets", {}) or {}))
    except Exception as exc:
        _fail(result, "load_markets", exc)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 1

    try:
        ticker = exchange.fetch_ticker(symbol)
        last = _safe_float(ticker.get("last"), _safe_float(ticker.get("close")))
        amount_raw = float(args.quote_notional) / last
        amount = float(exchange.amount_to_precision(symbol, amount_raw))
        notional = amount * last
        if amount <= 0.0:
            raise ValueError(f"amount rounded to zero for {symbol}")
        _ok(
            result,
            "ticker_and_amount",
            last=last,
            requested_quote_notional=float(args.quote_notional),
            amount=amount,
            notional_estimate=notional,
            amount_precision=(market.get("precision") or {}).get("amount"),
        )
    except Exception as exc:
        _fail(result, "ticker_and_amount", exc)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 1

    try:
        proxy = _normalize_orderbook(exchange.fetch_order_book(symbol, limit=20))
        _ok(
            result,
            "orderbook_proxy",
            missing_expected_columns=[c for c in ORDERBOOK_PROXY_COLUMNS if c not in proxy.columns],
            sample=_records(proxy, ORDERBOOK_PROXY_COLUMNS),
        )
    except Exception as exc:
        _fail(result, "orderbook_proxy", exc)
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=5)
        _ok(
            result,
            "ohlcv",
            rows=len(ohlcv),
            latest_ts=pd.to_datetime(ohlcv[-1][0], unit="ms", utc=True).isoformat() if ohlcv else None,
        )
    except Exception as exc:
        _fail(result, "ohlcv", exc)
    try:
        fr = exchange.fetch_funding_rate(symbol)
        _ok(result, "funding_rate", funding_rate=fr.get("fundingRate"), keys=sorted(fr.keys()))
    except Exception as exc:
        _fail(result, "funding_rate", exc)
    try:
        balance = exchange.fetch_balance()
        _ok(result, "balance", **_wallet_from_balance(balance))
    except Exception as exc:
        _fail(result, "balance", exc)
    try:
        positions = exchange.fetch_positions([symbol])
        _ok(
            result,
            "positions_before",
            rows=len(positions),
            sample=[
                {key: pos.get(key) for key in ("symbol", "side", "contracts", "notional", "entryPrice")}
                for pos in positions[:3]
                if isinstance(pos, dict)
            ],
        )
    except Exception as exc:
        _fail(result, "positions_before", exc)
    try:
        _ok(result, "open_orders_before", rows=len(exchange.fetch_open_orders(symbol)))
    except Exception as exc:
        _fail(result, "open_orders_before", exc)

    stop_initial = float(last * (1.0 - float(args.stop_pct)))
    stop_replacement = float(last * (1.0 - float(args.replacement_stop_pct)))
    result["stop_loss_dry_run_payload"] = {
        "entry": {"symbol": symbol, "type": "market", "side": "buy", "amount": amount},
        "stop": {
            "symbol": symbol,
            "type": "market",
            "side": "sell",
            "amount": amount,
            "stopLossPrice": stop_initial,
            "params": {"reduceOnly": True, "triggerSignal": "mark"},
        },
        "replacement_stop": stop_replacement,
    }

    if args.execute_trading_tests:
        if not args.confirm_live_orders:
            result["trading_tests"] = {
                "ok": False,
                "skipped": True,
                "reason": "--confirm-live-orders is required with --execute-trading-tests.",
            }
        else:
            cfg = {"execution_account": "perps", "exchange": "kraken", "market_mode": "perps"}
            entry_order: Dict[str, Any] | None = None
            stop_order: Dict[str, Any] | None = None
            replacement_order: Dict[str, Any] | None = None
            close_order: Dict[str, Any] | None = None
            lifecycle: Dict[str, Any] = {}
            try:
                entry_order = exchange.create_order(symbol, "market", "buy", amount)
                lifecycle["entry_order_id"] = _order_id(entry_order)
                time.sleep(max(0.0, float(args.hold_seconds)))
                stop_order = _create_reduce_stop_loss_order(
                    exchange,
                    symbol=symbol,
                    side="sell",
                    amount=amount,
                    stop_price=stop_initial,
                    config=cfg,
                )
                lifecycle["initial_stop_order_id"] = _order_id(stop_order)
                time.sleep(max(0.0, float(args.hold_seconds)))
                cancelled = _cancel_open_protective_stop_orders(
                    exchange,
                    symbol=symbol,
                    position_side="long",
                    config=cfg,
                )
                replacement_order = _create_reduce_stop_loss_order(
                    exchange,
                    symbol=symbol,
                    side="sell",
                    amount=amount,
                    stop_price=stop_replacement,
                    config=cfg,
                )
                lifecycle["cancelled_stops_for_replacement"] = cancelled
                lifecycle["replacement_stop_order_id"] = _order_id(replacement_order)
                time.sleep(max(0.0, float(args.hold_seconds)))
                stops = _fetch_open_protective_stop_orders(
                    exchange,
                    symbol=symbol,
                    position_side="long",
                    config=cfg,
                )
                positions = exchange.fetch_positions([symbol])
                lifecycle["open_protective_stops_after_replace"] = len(stops)
                lifecycle["positions_after_replace"] = [
                    {key: pos.get(key) for key in ("symbol", "side", "contracts", "notional", "entryPrice")}
                    for pos in positions
                    if isinstance(pos, dict)
                ]
                cleanup_cancelled = _cancel_open_protective_stop_orders(
                    exchange,
                    symbol=symbol,
                    position_side="long",
                    config=cfg,
                )
                close_order = _create_reduce_market_order(
                    exchange,
                    symbol=symbol,
                    side="sell",
                    amount=amount,
                    config=cfg,
                )
                time.sleep(max(0.0, float(args.hold_seconds)))
                lifecycle["cleanup_cancelled_stops"] = cleanup_cancelled
                lifecycle["close_order_id"] = _order_id(close_order)
                lifecycle["positions_after_close"] = [
                    {key: pos.get(key) for key in ("symbol", "side", "contracts", "notional", "entryPrice")}
                    for pos in exchange.fetch_positions([symbol])
                    if isinstance(pos, dict)
                ]
                lifecycle["open_orders_after_close"] = len(exchange.fetch_open_orders(symbol))
                result["trading_tests_executed"] = True
                _ok(result, "trading_tests", **lifecycle)
            except Exception as exc:
                _fail(result, "trading_tests", exc)
                result["trading_tests_cleanup_note"] = (
                    "If an entry succeeded before this error, check Kraken Futures manually."
                )

    result["sell_dust"] = {
        "ok": True,
        "status": "not_applicable",
        "reason": "Kraken Futures multi-collateral account has no Binance-style dust conversion endpoint.",
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    failed = any(isinstance(v, dict) and v.get("ok") is False for v in result.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
