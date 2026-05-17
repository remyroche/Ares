#!/usr/bin/env python3
"""Verify the OKX perp exchange path without mutating the account by default."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    _load_local_env_if_present,
    _resolve_perp_symbol,
    make_perp_exchange,
    normalize_orderbook_proxy_frame,
)
from extreme_price_movements.inference.trade_executor import (
    _cancel_open_protective_stop_orders,
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


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _ok(result: Dict[str, Any], name: str, **details: Any) -> None:
    result[name] = {"ok": True, **details}


def _fail(result: Dict[str, Any], name: str, exc: Exception) -> None:
    result[name] = {
        "ok": False,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _normalize_orderbook_proxy(orderbook: Dict[str, Any]) -> pd.DataFrame:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        raise ValueError("orderbook has no bids or asks")

    bid_px = float(bids[0][0])
    ask_px = float(asks[0][0])
    mid = (bid_px + ask_px) / 2.0
    bid_sizes = [float(level[1]) for level in bids if len(level) > 1]
    ask_sizes = [float(level[1]) for level in asks if len(level) > 1]
    ts_ms = orderbook.get("timestamp")
    ts = (
        pd.to_datetime(ts_ms, unit="ms", utc=True)
        if ts_ms is not None
        else pd.Timestamp.now(tz="UTC")
    )

    row = {
        "best_bid": bid_px,
        "best_ask": ask_px,
        "mid": mid,
        "bid_qty_1": bid_sizes[0] if bid_sizes else 0.0,
        "ask_qty_1": ask_sizes[0] if ask_sizes else 0.0,
        "cum_bid_qty_l10": float(sum(bid_sizes[:10])),
        "cum_ask_qty_l10": float(sum(ask_sizes[:10])),
        "cum_bid_qty_l20": float(sum(bid_sizes[:20])),
        "cum_ask_qty_l20": float(sum(ask_sizes[:20])),
        "snapshot_ts": ts,
        "trade_count_1h": 0.0,
        "buy_qty_1h": 0.0,
        "sell_qty_1h": 0.0,
        "notional_1h": 0.0,
        "buy_notional_1h": 0.0,
        "sell_notional_1h": 0.0,
        "vwap_1h": mid,
        "mean_trade_qty_1h": 0.0,
        "signed_flow_imbalance_1h": 0.0,
        "source": "live_l2_orderbook",
    }
    return normalize_orderbook_proxy_frame(pd.DataFrame([row], index=[ts.floor("1h")]))


def _safe_sample_records(df: pd.DataFrame, columns: Iterable[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    sample = df[list(columns)].tail(1).reset_index()
    return json.loads(sample.to_json(orient="records", date_format="iso"))


def _auth_status() -> Dict[str, Any]:
    return {
        "api_key_present": bool(_env_first("OKX_API_KEY")),
        "secret_present": bool(_env_first("OKX_API_SECRET", "OKX_SECRET_KEY")),
        "passphrase_present": bool(
            _env_first("OKX_API_PASSPHRASE", "OKX_PASSPHRASE", "OKX_PASSWORD")
        ),
    }


def _requires_private_auth(result: Dict[str, Any]) -> bool:
    auth = _auth_status()
    result["auth"] = auth
    return bool(
        auth["api_key_present"] and auth["secret_present"] and auth["passphrase_present"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--stop-side", choices=["long", "short"], default="long")
    parser.add_argument("--stop-amount", type=float, default=0.001)
    parser.add_argument("--stop-price", type=float, default=0.0)
    parser.add_argument("--margin-mode", choices=["cross", "isolated"], default="cross")
    parser.add_argument(
        "--execute-trading-tests",
        action="store_true",
        help="Actually create/cancel OKX stop-loss orders. Default is dry-run only.",
    )
    parser.add_argument(
        "--confirm-live-orders",
        action="store_true",
        help="Required with --execute-trading-tests to mutate the OKX account.",
    )
    args = parser.parse_args()

    _load_local_env_if_present()
    os.environ["EPM_EXCHANGE"] = "okx"

    result: Dict[str, Any] = {
        "exchange_requested": "okx",
        "symbol_requested": args.symbol,
        "trading_tests_executed": False,
    }

    try:
        exchange = make_perp_exchange()
        symbol = _resolve_perp_symbol(exchange, args.symbol) or args.symbol
        result["exchange_id"] = getattr(exchange, "id", None)
        result["symbol"] = symbol
        result["market"] = {
            k: exchange.market(symbol).get(k)
            for k in ("id", "symbol", "type", "spot", "swap", "future", "linear")
            if k in exchange.market(symbol)
        }
        _ok(result, "load_markets", market_count=len(getattr(exchange, "markets", {}) or {}))
    except Exception as exc:
        _fail(result, "load_markets", exc)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 1

    try:
        orderbook = exchange.fetch_order_book(symbol, limit=20)
        proxy = _normalize_orderbook_proxy(orderbook)
        missing = [col for col in ORDERBOOK_PROXY_COLUMNS if col not in proxy.columns]
        numeric_cols = [col for col in ORDERBOOK_PROXY_COLUMNS if col not in {"snapshot_ts", "source"}]
        finite_numeric = bool(proxy[numeric_cols].map(_finite).all().all())
        _ok(
            result,
            "orderbook_proxy",
            columns=list(proxy.columns),
            missing_expected_columns=missing,
            finite_numeric=finite_numeric,
            sample=_safe_sample_records(proxy, ORDERBOOK_PROXY_COLUMNS),
        )
    except Exception as exc:
        _fail(result, "orderbook_proxy", exc)

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=5)
        frame = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        _ok(
            result,
            "ohlcv",
            rows=len(frame),
            columns=list(frame.columns),
            latest_ts=(
                pd.to_datetime(frame["timestamp"].iloc[-1], unit="ms", utc=True).isoformat()
                if not frame.empty
                else None
            ),
        )
    except Exception as exc:
        _fail(result, "ohlcv", exc)

    try:
        funding = exchange.fetch_funding_rate(symbol)
        _ok(
            result,
            "funding_rate",
            keys=sorted(list(funding.keys())) if isinstance(funding, dict) else [],
            funding_rate=funding.get("fundingRate") if isinstance(funding, dict) else None,
            timestamp=funding.get("timestamp") if isinstance(funding, dict) else None,
        )
    except Exception as exc:
        _fail(result, "funding_rate", exc)

    private_ready = _requires_private_auth(result)
    private_params = {"type": "swap"}
    if private_ready:
        try:
            balance = exchange.fetch_balance(private_params)
            _ok(
                result,
                "balance",
                currencies=len(balance.get("total", {}) if isinstance(balance, dict) else {}),
            )
        except Exception as exc:
            _fail(result, "balance", exc)
        try:
            positions = exchange.fetch_positions([symbol], private_params)
            open_positions = [
                pos
                for pos in positions
                if _finite(pos.get("contracts") or pos.get("contractSize") or 0)
                and abs(float(pos.get("contracts") or 0)) > 0
            ]
            _ok(
                result,
                "positions",
                rows=len(positions),
                open_rows=len(open_positions),
                sample=[
                    {
                        "symbol": pos.get("symbol"),
                        "side": pos.get("side"),
                        "contracts": pos.get("contracts"),
                        "notional": pos.get("notional"),
                        "entryPrice": pos.get("entryPrice"),
                    }
                    for pos in positions[:3]
                    if isinstance(pos, dict)
                ],
            )
        except Exception as exc:
            _fail(result, "positions", exc)
        try:
            stops = _fetch_open_protective_stop_orders(
                exchange,
                symbol=symbol,
                position_side=args.stop_side,
                config={
                    "execution_account": "perps",
                    "exchange": "okx",
                    "margin_mode": args.margin_mode,
                },
            )
            _ok(result, "open_protective_stops", rows=len(stops))
        except Exception as exc:
            _fail(result, "open_protective_stops", exc)
    else:
        result["private_checks"] = {
            "ok": False,
            "skipped": True,
            "reason": "OKX private API requires OKX_API_KEY, OKX_API_SECRET, and OKX_API_PASSPHRASE/OKX_PASSWORD.",
        }

    stop_payload = {
        "method": "create_stop_loss_order",
        "symbol": symbol,
        "type": "market",
        "side": "sell" if args.stop_side == "long" else "buy",
        "amount": args.stop_amount,
        "stopLossPrice": args.stop_price if args.stop_price > 0 else "<required>",
        "params": {"reduceOnly": True, "tdMode": args.margin_mode},
    }
    result["stop_loss_dry_run_payload"] = stop_payload
    result["replace_stop_loss_model"] = "cancel current protective stop, then create replacement stop-loss order"
    result["sell_dust"] = {
        "ok": True,
        "status": "not_applicable",
        "reason": "OKX swap/perp trading does not use the Binance margin dust conversion endpoints.",
    }

    if args.execute_trading_tests:
        if not args.confirm_live_orders:
            result["trading_tests"] = {
                "ok": False,
                "skipped": True,
                "reason": "--confirm-live-orders is required with --execute-trading-tests.",
            }
        elif not private_ready:
            result["trading_tests"] = {
                "ok": False,
                "skipped": True,
                "reason": "private OKX auth is incomplete.",
            }
        elif args.stop_price <= 0:
            result["trading_tests"] = {
                "ok": False,
                "skipped": True,
                "reason": "--stop-price must be positive for live stop-loss tests.",
            }
        else:
            config = {
                "execution_account": "perps",
                "exchange": "okx",
                "margin_mode": args.margin_mode,
            }
            try:
                before_cancelled = _cancel_open_protective_stop_orders(
                    exchange,
                    symbol=symbol,
                    position_side=args.stop_side,
                    config=config,
                )
                order = _create_reduce_stop_loss_order(
                    exchange,
                    symbol=symbol,
                    side=stop_payload["side"],
                    amount=args.stop_amount,
                    stop_price=args.stop_price,
                    config=config,
                )
                after_cancelled = _cancel_open_protective_stop_orders(
                    exchange,
                    symbol=symbol,
                    position_side=args.stop_side,
                    config=config,
                    keep_order_id=None,
                )
                result["trading_tests_executed"] = True
                _ok(
                    result,
                    "trading_tests",
                    created_order_id=order.get("id") if isinstance(order, dict) else None,
                    cancelled_before=before_cancelled,
                    cancelled_after=after_cancelled,
                )
            except Exception as exc:
                _fail(result, "trading_tests", exc)

    success = all(
        not isinstance(value, dict) or value.get("ok", True)
        for key, value in result.items()
        if key
        in {
            "load_markets",
            "orderbook_proxy",
            "ohlcv",
            "funding_rate",
            "balance",
            "positions",
            "open_protective_stops",
            "trading_tests",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
