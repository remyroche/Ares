"""Trade executor for inference.

Live mode places an entry order plus a STOP_LOSS order and updates that stop
with cancel-replace. Shadow mode records the same lifecycle without sending
exchange orders. The inference caller passes USDT quote notional; live orders
convert that notional to base quantity before touching the exchange.
"""

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.utils import tprint

LIVE_EXECUTION_MODES = {"live", "live-test", "live_test", "livetest"}
TRIGGER_PRICE_REJECT_TOKENS = (
    "order_would_immediately_trigger",
    "order would immediately trigger",
    "orderimmediatelyfillable",
    "immediately fillable",
    "strategy_invalid_trigger_price",
    "invalid trigger price",
    "conditional_order_trigger_reject",
    "trigger reject",
    "trigger price",
)


def _is_live_execution_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in LIVE_EXECUTION_MODES


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Convert exchange payload values to float without raising."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _extract_policy_barrier_frac(
    trade_context: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
) -> float:
    """Return the optimiser barrier fraction without ATR-style fallbacks."""
    for source in (trade_context, params):
        if not isinstance(source, dict):
            continue
        for key in ("barrier_frac", "barrier_pct"):
            value = _safe_float(source.get(key), default=np.nan)
            if np.isfinite(value) and value > 0.0:
                return float(value)
    return np.nan


def _exchange_error_text(exc: Exception) -> str:
    return f"{exc.__class__.__name__} {exc}".lower()


def _exchange_reject_reason(exc: Exception) -> str:
    """Extract the most useful stable reject reason from a Binance/ccxt error."""
    text = _exchange_error_text(exc)
    for token in TRIGGER_PRICE_REJECT_TOKENS:
        if token in text:
            return token.upper().replace(" ", "_")
    for token in (
        "insufficient",
        "balance",
        "min_notional",
        "lot_size",
        "precision",
        "rate limit",
        "too many requests",
        "timeout",
        "network",
        "permission",
        "unauthorized",
    ):
        if token in text:
            return token.upper().replace(" ", "_")
    return exc.__class__.__name__


def _is_trigger_price_reject(exc: Exception) -> bool:
    text = _exchange_error_text(exc)
    return any(token in text for token in TRIGGER_PRICE_REJECT_TOKENS)


def _stop_side_is_valid(side: str, stop_price: float, current_price: float) -> bool:
    if not (np.isfinite(stop_price) and np.isfinite(current_price)):
        return False
    if current_price <= 0.0 or stop_price <= 0.0:
        return False
    return stop_price < current_price if side == "long" else stop_price > current_price


def _stop_gap_fraction(stop_price: float, current_price: float) -> float:
    if not (np.isfinite(stop_price) and np.isfinite(current_price)):
        return np.nan
    if current_price <= 0.0:
        return np.nan
    return float(
        abs(float(current_price) - float(stop_price)) / abs(float(current_price))
    )


def _widen_stop_away_from_market(
    *,
    side: str,
    stop_price: float,
    current_price: float,
    round_fee_buffer_pct: float,
    retry_gap_growth: float,
    immediate_buffer_pct: float,
) -> float:
    """Move a stop away from current price while preserving the stop side.

    The retry schedule keeps at least the round-trip fee buffer between current
    price and stop, then widens by 10/20/30% of the distance between the
    requested gap and that fee buffer.
    """
    raw_gap = _stop_gap_fraction(stop_price, current_price)
    if not np.isfinite(raw_gap):
        return float(stop_price)
    min_gap = max(float(round_fee_buffer_pct), float(immediate_buffer_pct), 1e-9)
    target_gap = max(float(raw_gap), min_gap)
    target_gap += max(float(retry_gap_growth), 0.0) * abs(float(raw_gap) - min_gap)
    if str(side).lower() == "long":
        return float(current_price) * (1.0 - target_gap)
    return float(current_price) * (1.0 + target_gap)


def _append_position_event(
    state: Dict[str, Any],
    event: str,
    **payload: Any,
) -> None:
    """Append a compact per-position audit event for close recaps."""
    events = state.setdefault("trade_recap_events", [])
    if not isinstance(events, list):
        events = []
        state["trade_recap_events"] = events
    if len(events) >= 500:
        del events[: len(events) - 499]
    clean_payload: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        clean_payload[key] = value
    events.append(
        {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "event": str(event),
            **clean_payload,
        }
    )


def _format_trade_recap(events: Any) -> str:
    """Return a readable one-line-per-event recap for email/reporting."""
    if not isinstance(events, list) or not events:
        return ""
    lines: List[str] = []
    for event in events[-120:]:
        if not isinstance(event, dict):
            continue
        ts = event.get("ts", "")
        name = event.get("event", "event")
        rest = " ".join(
            f"{k}={v}"
            for k, v in event.items()
            if k not in {"ts", "event"} and v is not None
        )
        lines.append(f"{ts} {name} {rest}".strip())
    return "\n".join(lines)


def _classify_exchange_error(exc: Exception) -> str:
    """Return a stable exchange error category for logs and risk gates."""
    text = _exchange_error_text(exc)
    if _is_trigger_price_reject(exc):
        return "trigger_price_rejected"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "authentication" in text or "unauthorized" in text or "permission" in text:
        return "auth_or_permission"
    if "max pledged collateral" in text or "max transfer in quantity is 0" in text:
        return "asset_collateral_limit"
    if "insufficient" in text or "balance" in text or "margin" in text:
        return "insufficient_balance"
    if (
        "precision" in text
        or "lot_size" in text
        or "min_notional" in text
        or "notional below" in text
        or "exchange minimum" in text
    ):
        return "invalid_precision_or_filter"
    if "halt" in text or "suspend" in text or "inactive" in text or "closed" in text:
        return "symbol_halted"
    if "duplicate" in text or "clientorderid" in text or "client order id" in text:
        return "duplicate_client_order_id"
    if "timeout" in text or "timed out" in text or "network" in text:
        return "network_timeout"
    if "cancel" in text:
        return "cancel_failed"
    if "reject" in text or "invalidorder" in text:
        return "order_rejected"
    return "exchange_error"


def _load_market(exchange: Any, symbol: str) -> Dict[str, Any]:
    """Load a ccxt market definition if available."""
    if exchange is None:
        return {}
    try:
        if hasattr(exchange, "load_markets") and not getattr(exchange, "markets", None):
            exchange.load_markets()
    except Exception as exc:
        tprint(f"Warning: could not load markets before trading {symbol}: {exc}")
    try:
        if hasattr(exchange, "market"):
            market = exchange.market(symbol)
            if isinstance(market, dict):
                return market
    except Exception:
        pass
    markets = getattr(exchange, "markets", None)
    if isinstance(markets, dict):
        market = markets.get(symbol, {})
        if isinstance(market, dict):
            return market
    return {}


def _market_is_active(market: Dict[str, Any]) -> bool:
    """Return whether a market definition looks tradeable."""
    if not market:
        return True
    if market.get("active") is False:
        return False
    info = market.get("info", {})
    if isinstance(info, dict):
        status = str(info.get("status", "")).upper()
        if status and status not in {"TRADING", "ENABLED"}:
            return False
    return True


def _exchange_precision(
    exchange: Any,
    symbol: str,
    value: float,
    *,
    kind: str,
) -> float:
    """Apply exchange amount or price precision when supported."""
    if not np.isfinite(value):
        return value
    method_name = f"{kind}_to_precision"
    method = getattr(exchange, method_name, None)
    if callable(method):
        try:
            return float(method(symbol, value))
        except Exception:
            return float(value)
    return float(value)


def _validate_order_filters(
    symbol: str,
    market: Dict[str, Any],
    *,
    amount: float,
    price: float,
) -> None:
    """Raise if amount/cost violates exchange market limits."""
    if not _market_is_active(market):
        raise ValueError(f"symbol halted or inactive: {symbol}")
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError(f"invalid base amount for {symbol}: {amount}")
    if not np.isfinite(price) or price <= 0:
        raise ValueError(f"invalid reference price for {symbol}: {price}")
    limits = market.get("limits", {}) if isinstance(market, dict) else {}
    amount_limits = limits.get("amount", {}) if isinstance(limits, dict) else {}
    cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
    min_amount = _safe_float(amount_limits.get("min"), default=np.nan)
    max_amount = _safe_float(amount_limits.get("max"), default=np.nan)
    min_cost = _safe_float(cost_limits.get("min"), default=np.nan)
    max_cost = _safe_float(cost_limits.get("max"), default=np.nan)
    cost = amount * price
    if np.isfinite(min_amount) and amount < min_amount:
        raise ValueError(
            f"amount below exchange minimum for {symbol}: {amount} < {min_amount}"
        )
    if np.isfinite(max_amount) and amount > max_amount:
        raise ValueError(
            f"amount above exchange maximum for {symbol}: {amount} > {max_amount}"
        )
    if np.isfinite(min_cost) and cost < min_cost:
        raise ValueError(
            f"notional below exchange minimum for {symbol}: {cost} < {min_cost}"
        )
    if np.isfinite(max_cost) and cost > max_cost:
        raise ValueError(
            f"notional above exchange maximum for {symbol}: {cost} > {max_cost}"
        )


def _extract_order_fill(
    order: Dict[str, Any], fallback_price: float
) -> Tuple[float, float, bool]:
    """Return average price, filled amount, and partial-fill flag."""
    avg = _safe_float(order.get("average"), default=np.nan)
    price = _safe_float(order.get("price"), default=np.nan)
    filled = _safe_float(order.get("filled"), default=np.nan)
    amount = _safe_float(order.get("amount"), default=np.nan)
    realized_price = avg if np.isfinite(avg) and avg > 0 else price
    if not np.isfinite(realized_price) or realized_price <= 0:
        realized_price = fallback_price
    partial = bool(
        np.isfinite(filled) and np.isfinite(amount) and filled > 0 and filled < amount
    )
    return float(realized_price), float(filled), partial


def _filled_base_fee(order: Dict[str, Any], symbol: str) -> float:
    """Return base-asset fees charged by the entry fill."""
    base_asset = str(symbol).split("/", 1)[0].upper()
    fees = order.get("fees")
    if not isinstance(fees, list) or not fees:
        fee = order.get("fee")
        fees = [fee] if isinstance(fee, dict) else []
    total = 0.0
    for fee in fees:
        if not isinstance(fee, dict):
            continue
        if str(fee.get("currency", "")).upper() != base_asset:
            continue
        total += _safe_float(fee.get("cost"), default=0.0)
    return float(max(total, 0.0))


def _make_legacy_order_manager(exchange: Any, enabled: bool) -> Optional[Any]:
    """Optionally construct the legacy live_trading OrderManager.

    Importing that stack pulls in heavyweight explainability modules, so keep it
    out of normal inference unless explicitly requested.
    """
    if not enabled:
        return None
    try:
        from live_trading.order_manager import OrderManager

        return OrderManager(exchange=exchange)
    except Exception as exc:
        tprint(f"Warning: legacy OrderManager unavailable: {exc}")
        return None


def _execution_account(config: Optional[Dict[str, Any]]) -> str:
    """Return configured execution account type."""
    cfg = config or {}
    raw = str(
        cfg.get("execution_account", cfg.get("account_type", "margin")) or "margin"
    ).lower()
    if raw in {"margin", "cross_margin", "isolated_margin"}:
        return "margin"
    return "spot"


def _margin_mode(config: Optional[Dict[str, Any]]) -> str:
    """Return configured margin mode for ccxt order params."""
    cfg = config or {}
    raw = str(cfg.get("margin_mode", "cross") or "cross").lower()
    return "isolated" if raw == "isolated" else "cross"


def _margin_side_effect(config: Optional[Dict[str, Any]]) -> str:
    """Return Binance margin sideEffectType for opening orders."""
    cfg = config or {}
    raw = str(cfg.get("margin_side_effect_type", "AUTO_BORROW_REPAY") or "")
    allowed = {"NO_SIDE_EFFECT", "MARGIN_BUY", "AUTO_REPAY", "AUTO_BORROW_REPAY"}
    return raw if raw in allowed else "AUTO_BORROW_REPAY"


def _order_params(
    config: Optional[Dict[str, Any]],
    *,
    reduce_only: bool = False,
) -> Dict[str, Any]:
    """Build ccxt params for spot or margin order placement."""
    if _execution_account(config) != "margin":
        return {}
    params: Dict[str, Any] = {"marginMode": _margin_mode(config)}
    if reduce_only:
        params["sideEffectType"] = "AUTO_REPAY"
    else:
        params["sideEffectType"] = _margin_side_effect(config)
    return params


def _cancel_params(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build ccxt params for spot or margin order cancellation."""
    if _execution_account(config) != "margin":
        return {}
    return {"marginMode": _margin_mode(config)}


def _symbol_from_asset_quote(asset: str, quote: str) -> str:
    """Return the canonical ccxt symbol for an asset/quote pair."""
    return f"{str(asset).upper()}/{str(quote).upper()}"


def _closed_trade_metrics(
    symbol: str,
    state: Dict[str, Any],
    order: Optional[Dict[str, Any]],
    *,
    reason: str,
) -> Dict[str, Any]:
    """Build close metrics from tracked state and an exchange close order."""
    order = order or {}
    side = str(state.get("side", "") or "").lower()
    entry_price = _safe_float(state.get("entry_price"))
    exit_price = _safe_float(
        order.get("average"),
        _safe_float(order.get("price"), _safe_float(order.get("stopPrice"))),
    )
    filled = _safe_float(order.get("filled"), _safe_float(order.get("amount")))
    if not np.isfinite(filled) or filled <= 0:
        filled = _safe_float(state.get("size"))
    gross_pnl = np.nan
    gross_pnl_pct = np.nan
    notional = np.nan
    if (
        np.isfinite(entry_price)
        and entry_price > 0
        and np.isfinite(exit_price)
        and np.isfinite(filled)
        and filled > 0
    ):
        direction = 1.0 if side == "long" else -1.0
        notional = entry_price * filled
        gross_pnl = direction * (exit_price - entry_price) * filled
        gross_pnl_pct = gross_pnl / max(notional, 1e-12)
    fee = order.get("fee") if isinstance(order.get("fee"), dict) else {}
    fee_cost = _safe_float(fee.get("cost"), 0.0)
    fee_currency = str(fee.get("currency") or "").upper()
    base_asset = str(symbol).split("/", 1)[0].upper()
    quote_asset = (
        str(symbol).split("/", 1)[1].split(":", 1)[0].upper()
        if "/" in str(symbol)
        else ""
    )
    fee_quote = 0.0
    if np.isfinite(fee_cost) and fee_cost > 0.0:
        if fee_currency == quote_asset or not fee_currency:
            fee_quote = fee_cost
        elif fee_currency == base_asset and np.isfinite(exit_price):
            fee_quote = fee_cost * exit_price
    net_pnl = gross_pnl - fee_quote if np.isfinite(gross_pnl) else np.nan
    net_pnl_pct = net_pnl / max(notional, 1e-12) if np.isfinite(notional) else np.nan
    stop_origin = str(state.get("stop_reason") or "original_stop_loss")
    reason_detail = state.get("stop_reason_detail") or stop_origin
    return {
        "symbol": symbol,
        "side": side,
        "strategy_id": state.get("bucket_key"),
        "reason": reason,
        "exit_reason_detail": reason_detail,
        "stop_origin": stop_origin,
        "entry_time": state.get("entry_time"),
        "exit_time": pd.Timestamp.now(tz="UTC"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_order_type": state.get("entry_order_type"),
        "ohlcv_entry_price": state.get("ohlcv_entry_price"),
        "entry_price_delta_vs_ohlcv": state.get("entry_price_delta_vs_ohlcv"),
        "entry_price_delta_vs_ohlcv_pct": state.get("entry_price_delta_vs_ohlcv_pct"),
        "base_pred": state.get("base_pred"),
        "base_rank_pct": state.get("base_rank_pct"),
        "base_train_rank_pct": state.get("base_train_rank_pct"),
        "base_gate_top_frac": state.get("base_gate_top_frac"),
        "meta_pred": state.get("meta_pred"),
        "meta_train_rank_pct": state.get("meta_train_rank_pct"),
        "rank_score_source": state.get("rank_score_source"),
        "calibrated_score": state.get("calibrated_score"),
        "rank_percentile": state.get("rank_percentile"),
        "effective_threshold": state.get("effective_threshold"),
        "deployment_rank_threshold": state.get("deployment_rank_threshold"),
        "filled": filled,
        "gross_pnl": gross_pnl,
        "gross_pnl_amount": gross_pnl,
        "gross_pnl_pct": gross_pnl_pct,
        "net_pnl": net_pnl,
        "net_pnl_amount": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "pnl_scope": "position_notional_excluding_wallet_equity_borrow_interest",
        "mfe": _safe_float(state.get("mfe"), 0.0),
        "mae": _safe_float(state.get("mae"), 0.0),
        "stop_price": state.get("stop_price"),
        "stop_order_id": state.get("stop_order_id"),
        "close_order_id": order.get("id"),
        "close_order_status": order.get("status"),
        "close_order_type": order.get("type"),
        "close_order_cost": order.get("cost"),
        "fee_cost": fee_cost,
        "fee_currency": fee_currency,
        "fees_amount": fee_quote,
        "trade_recap": _format_trade_recap(state.get("trade_recap_events")),
    }


class OCOExecutor:
    """STOP_LOSS executor with dynamic cancel-replace updates.

    Legacy names are kept because inference callers already reference them.
    Features:
    - Place stop-loss orders after entry
    - Giveback: Move SL up as price increases (lock in profits)
    - Trailing profit: Adjust SL based on trailing parameters
    - Monitor prices every configured interval during live trading
    """

    def __init__(
        self,
        exchange: Any,
        bucket_params: Dict[str, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OCOExecutor.

        Args:
            exchange: ccxt exchange instance
            bucket_params: Parameters from ridge position sizer per bucket
            config: Additional configuration options
        """
        self.exchange = exchange
        self.bucket_params = bucket_params
        self.config = config or {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self._positions_lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._monitor_interval = config.get("monitor_interval_seconds", 900)

        # Default parameters (fallback when bucket not found)
        self._default_params = {
            "sl_mult": 1.0,
            "trail_mult": 0.25,
            "giveback_pct": 0.005,
            "profit_lock_amount": 0.003,
            "max_hold_hours": 48,
        }

    def _get_bucket_params(self, bucket_key: str) -> Dict[str, Any]:
        """Get parameters for a specific bucket, with fallback to defaults."""
        raw_key = str(bucket_key or "")
        core_key = strategy_core_id(raw_key)
        key_lower = raw_key.lower()
        key_upper = raw_key.upper()
        bucket = (
            self.bucket_params.get(raw_key, {})
            or self.bucket_params.get(core_key, {})
            or self.bucket_params.get(key_lower, {})
            or self.bucket_params.get(key_upper, {})
        )
        params = self._default_params.copy()
        params.update(
            {k: v for k, v in self.bucket_params.items() if not isinstance(v, dict)}
        )
        if not bucket:
            # Try to find in nested structure
            for key in self.bucket_params:
                if isinstance(self.bucket_params[key], dict):
                    if raw_key in self.bucket_params[key]:
                        bucket = self.bucket_params[key][raw_key]
                        break
                    if key_lower in self.bucket_params[key]:
                        bucket = self.bucket_params[key][key_lower]
                        break
                    if key_upper in self.bucket_params[key]:
                        bucket = self.bucket_params[key][key_upper]
                        break
                    if key.upper() == key_upper.split("_")[0]:
                        inner = self.bucket_params[key]
                        # Check for tp_sl or similar nested structure
                        if "tp_sl" in inner:
                            bucket = inner["tp_sl"]
                            break
        if isinstance(bucket, dict):
            params.update(bucket)
        return params

    def get_bucket_params(self, bucket_key: Optional[str] = None) -> Dict[str, Any]:
        """Public accessor for bucket params, including top-level defaults."""
        params = self._default_params.copy()
        if bucket_key is None:
            params.update(
                {k: v for k, v in self.bucket_params.items() if not isinstance(v, dict)}
            )
            return params
        params.update(self._get_bucket_params(bucket_key))
        return params

    def get_cooldown_hours(self, bucket_key: Optional[str] = None) -> float:
        """Return zero: live inference cooldown is handled on losing closes only."""
        return 0.0

    def _fetch_aggtrades(
        self, symbol: str, since: int = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """Fetch aggregated trades for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            since: Start time in milliseconds (optional)
            limit: Number of trades to fetch (optional)

        Returns:
            List of aggtrade dictionaries
        """
        try:
            # Try different method names for aggtrades
            if hasattr(self.exchange, "fetch_aggregated_trades"):
                return self.exchange.fetch_aggregated_trades(
                    symbol, since=since, limit=limit
                )
            elif hasattr(self.exchange, "fetch_agg_trades"):
                return self.exchange.fetch_agg_trades(symbol, since=since, limit=limit)
            elif hasattr(self.exchange, "fetch_trades"):
                # Fallback to regular trades
                return self.exchange.fetch_trades(symbol, since=since, limit=limit)
            else:
                tprint(f"Exchange does not support fetching trades for {symbol}")
                return []
        except Exception as e:
            tprint(
                f"Error fetching aggtrades for {symbol}: "
                f"{_classify_exchange_error(e)}: {e}"
            )
            return []

    def _get_aggtrades_at_entry(
        self, symbol: str, entry_time: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """Fetch aggtrades around the entry time.

        Fetches aggtrades for the minute surrounding the entry time.

        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp

        Returns:
            List of aggtrade dictionaries
        """
        # Convert entry_time to milliseconds
        if entry_time.tz is not None:
            entry_time = entry_time.tz_convert("UTC")

        # Get trades for 1 minute around entry time
        start_ms = int(entry_time.timestamp() * 1000) - 60000  # 1 minute before
        end_ms = int(entry_time.timestamp() * 1000) + 60000  # 1 minute after

        return self._fetch_aggtrades(symbol, since=start_ms, limit=1000)

    def place_oco_order(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        bucket_key: str,
        atr_frac: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place the initial STOP_LOSS order after entry.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "long" or "short"
            entry_price: Entry price for the position
            size: Position size in base asset units
            bucket_key: Bucket key for getting stop parameters

        Returns:
            Dictionary with order details
        """
        params = self._get_bucket_params(bucket_key)
        barrier_frac = _safe_float(atr_frac, default=np.nan)
        if not np.isfinite(barrier_frac) or barrier_frac <= 0.0:
            error = "missing policy barrier_pct/barrier_frac for stop placement"
            tprint(f"Refusing STOP_LOSS for {symbol}: {error}")
            return {
                "success": False,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "size": size,
                "bucket_key": bucket_key,
                "error": error,
                "error_category": "missing_policy_barrier_pct",
            }

        sl_mult = params["sl_mult"]
        fixed_stop_loss_pct = _safe_float(
            params.get(
                "fixed_stop_loss_pct",
                params.get("stop_loss_pct", params.get("stop_loss_frac", np.nan)),
            ),
            default=np.nan,
        )

        if np.isfinite(fixed_stop_loss_pct) and fixed_stop_loss_pct > 0.0:
            if side == "long":
                stop_price = entry_price * (1.0 - fixed_stop_loss_pct)
            else:
                stop_price = entry_price * (1.0 + fixed_stop_loss_pct)
        elif side == "long":
            stop_price = entry_price * (1 - sl_mult * barrier_frac)
        else:  # short
            stop_price = entry_price * (1 + sl_mult * barrier_frac)
        limit_price = None

        # Track position state
        position_state = {
            "side": side,
            "entry_price": entry_price,
            "size": size,
            "bucket_key": bucket_key,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "barrier_frac": barrier_frac,
            "barrier_pct": barrier_frac,
            "sl_mult": sl_mult,
            "initial_stop_price": stop_price,
            "stop_reason": "original_stop_loss",
            "stop_reason_detail": (
                f"original_stop_loss: sl_mult={sl_mult:.6g} "
                f"barrier_frac={barrier_frac:.6g}"
            ),
            "peak_price": entry_price,
            "mfe": 0.0,
            "mae": 0.0,
            "entry_time": pd.Timestamp.now(tz="UTC"),
            "last_update": pd.Timestamp.now(tz="UTC"),
            "oco_order_id": None,
            "stop_order_id": None,
            "take_profit_order_id": None,
        }
        _append_position_event(
            position_state,
            "entry_stop_created",
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_dev_pct=(
                (float(stop_price) - float(entry_price))
                / max(float(entry_price), 1e-12)
            ),
            sl_mult=float(sl_mult),
            barrier_frac=float(barrier_frac),
            stop_reason="original_stop_loss",
        )

        market = _load_market(self.exchange, symbol)
        amount = _exchange_precision(self.exchange, symbol, float(size), kind="amount")
        stop_price = _exchange_precision(
            self.exchange, symbol, float(stop_price), kind="price"
        )
        position_state["size"] = amount
        position_state["stop_price"] = stop_price

        stop_order_error = None
        stop_order_error_category = None
        try:
            _validate_order_filters(
                symbol, market, amount=amount, price=float(entry_price)
            )
            stop_order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_LOSS",
                side="sell" if side == "long" else "buy",
                amount=amount,
                price=stop_price,
                params={
                    **_order_params(self.config, reduce_only=True),
                    "stopPrice": stop_price,
                },
            )
            position_state["stop_order_id"] = stop_order.get("id")

        except Exception as e:
            stop_order_error = str(e)
            stop_order_error_category = _classify_exchange_error(e)
            stop_order_reject_reason = _exchange_reject_reason(e)
            position_state["stop_order_error"] = stop_order_error
            position_state["stop_order_error_category"] = stop_order_error_category
            position_state["stop_order_reject_reason"] = stop_order_reject_reason
            _append_position_event(
                position_state,
                "entry_stop_failed",
                error_category=stop_order_error_category,
                reject_reason=stop_order_reject_reason,
                error=stop_order_error,
            )
            tprint(
                f"Error placing STOP_LOSS for {symbol}: "
                f"{stop_order_error_category} reason={stop_order_reject_reason}: {e}"
            )
            # Continue with tracking even if order placement fails
            # This allows for manual intervention or retry

        # Store position
        with self._positions_lock:
            self.active_positions[symbol] = position_state

        # Fetch aggtrades and 5m OHLCV for analysis (OCOExecutor is only used in live mode)
        aggtrades_data = None
        ohlcv_5m_data = None

        # Only fetch data if order was placed successfully
        if position_state.get("stop_order_id"):
            try:
                # Fetch aggtrades around entry time
                entry_time = pd.Timestamp.now(tz="UTC")
                aggtrades_data = self._get_aggtrades_at_entry(symbol, entry_time)
                position_state["aggtrades"] = aggtrades_data

                # Fetch 5m OHLCV data using hf_data_loader
                try:
                    ohlcv_5m_data = hf_data_loader.fetch_ohlcv_5m(
                        self.exchange,
                        symbol,
                        entry_time - pd.Timedelta(hours=1),  # 1 hour before
                        entry_time + pd.Timedelta(hours=12),  # 12 hours after
                    )
                    position_state["ohlcv_5m"] = ohlcv_5m_data
                except Exception as e:
                    tprint(
                        f"Error fetching 5m OHLCV for {symbol}: "
                        f"{_classify_exchange_error(e)}: {e}"
                    )

            except Exception as e:
                tprint(
                    f"Error fetching entry diagnostics for {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

        if position_state.get("stop_order_id"):
            tprint(
                f"Placed STOP_LOSS for {symbol}: SL={stop_price:.8g} "
                f"barrier_frac={barrier_frac:.6g} sl_mult={sl_mult:.6g}"
            )

        return {
            "success": position_state.get("stop_order_id") is not None,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "size": amount,
            "bucket_key": bucket_key,
            "stop_order_id": position_state.get("stop_order_id"),
            "fixed_stop_loss_pct": (
                float(fixed_stop_loss_pct) if np.isfinite(fixed_stop_loss_pct) else None
            ),
            "barrier_frac": float(barrier_frac),
            "barrier_pct": float(barrier_frac),
            "error": stop_order_error,
            "error_category": stop_order_error_category,
            "aggtrades": aggtrades_data,
            "ohlcv_5m": ohlcv_5m_data,
        }

    def start_monitoring(self):
        """Start monitoring thread for position updates."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        tprint("OCOExecutor monitoring started")

    def stop_monitoring(self):
        """Stop monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        tprint("OCOExecutor monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop - runs every _monitor_interval seconds."""
        while self._monitoring:
            try:
                self.monitor_positions()
            except Exception as e:
                tprint(
                    f"Error in OCO monitoring loop: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

            # Sleep in small increments to allow quick shutdown
            for _ in range(int(self._monitor_interval)):
                if not self._monitoring:
                    break
                time.sleep(1)

    def monitor_positions(self):
        """Monitor all active positions every 15 minutes.

        Checks for:
        - Giveback: Move SL up as price increases
        - Trailing profit: Adjust based on trailing params
        - MFE exit: Exit if price reaches target
        """
        current_time = pd.Timestamp.now(tz="UTC")

        with self._positions_lock:
            items = list(self.active_positions.items())
        for symbol, state in items:
            # Skip if not enough time passed.
            time_since_update = current_time - state["last_update"]
            if time_since_update < pd.Timedelta(seconds=float(self._monitor_interval)):
                continue

            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])

                # Update position based on price movement
                self._update_oco(symbol, state, current_price)

            except Exception as e:
                tprint(
                    f"Error monitoring {symbol}: " f"{_classify_exchange_error(e)}: {e}"
                )
                continue

    def _update_oco(self, symbol: str, state: Dict[str, Any], current_price: float):
        """Update STOP_LOSS based on price movement.

        1. Giveback: Move SL up as price increases
        2. Trailing profit: Adjust based on trailing params

        Args:
            symbol: Trading symbol
            state: Position state dictionary
            current_price: Current market price
        """
        side = state["side"]
        entry_price = state["entry_price"]
        params = self._get_bucket_params(state["bucket_key"])

        # Update peak price and MFE
        if side == "long":
            if current_price > state["peak_price"]:
                state["peak_price"] = current_price
                state["mfe"] = (current_price - entry_price) / entry_price
        else:  # short
            if current_price < state["peak_price"]:
                state["peak_price"] = current_price
                state["mfe"] = (entry_price - current_price) / entry_price

        # Calculate new stop price (giveback + trailing)
        giveback_pct = params.get("giveback_pct", 0.005)
        trail_mult = params.get("trail_mult", 0.25)
        profit_lock = params.get("profit_lock_amount", 0.003)

        if side == "long":
            # Giveback: move SL up as price increases
            max_giveback = giveback_pct * state["peak_price"]
            new_stop = state["peak_price"] - max_giveback

            # Trailing: also apply trailing multiplier
            if state["peak_price"] > entry_price:
                trail_distance = trail_mult * (state["peak_price"] - entry_price)
                new_stop = max(new_stop, entry_price + trail_distance)

            # Don't go below entry + profit lock
            min_stop = entry_price + profit_lock * entry_price
            new_stop = max(new_stop, min_stop)

        else:  # short
            max_giveback = giveback_pct * state["peak_price"]
            new_stop = state["peak_price"] + max_giveback

            if state["peak_price"] < entry_price:
                trail_distance = trail_mult * (entry_price - state["peak_price"])
                new_stop = min(new_stop, entry_price - trail_distance)

            max_stop = entry_price - profit_lock * entry_price
            new_stop = min(new_stop, max_stop)

        # If new stop is better than current, update
        should_update = False
        if side == "long" and new_stop > state["stop_price"]:
            should_update = True
        elif side == "short" and new_stop < state["stop_price"]:
            should_update = True

        if should_update:
            self._update_stop_loss(symbol, state, new_stop)

        state["last_update"] = pd.Timestamp.now(tz="UTC")

    def _update_stop_loss(
        self, symbol: str, state: Dict[str, Any], new_stop_price: float
    ):
        """Update the stop loss order.

        Args:
            symbol: Trading symbol
            state: Position state dictionary
            new_stop_price: New stop loss price
        """
        old_stop_price = _safe_float(state.get("stop_price"), default=np.nan)
        side = str(state.get("side", "long")).lower()
        canceled_existing = False
        last_error: Optional[Exception] = None
        try:
            stop_price_candidate = _exchange_precision(
                self.exchange, symbol, float(new_stop_price), kind="price"
            )
            immediate_buffer = (
                float((self.config or {}).get("stop_replace_immediate_buffer_bps", 5.0))
                / 1e4
            )
            round_fee_buffer = float(
                (self.config or {}).get("stop_replace_round_fee_buffer_pct", 0.003)
            )
            round_fee_buffer = max(round_fee_buffer, 0.0)
            retry_growths = (self.config or {}).get(
                "stop_replace_retry_gap_growths", [0.0, 0.10, 0.20, 0.30]
            )
            if not isinstance(retry_growths, (list, tuple)):
                retry_growths = [0.0, 0.10, 0.20, 0.30]
            retry_growths = [float(x) for x in retry_growths]
            max_attempts = max(
                1,
                int(
                    (self.config or {}).get(
                        "stop_replace_max_attempts", len(retry_growths)
                    )
                ),
            )
            if len(retry_growths) < max_attempts:
                retry_growths.extend(
                    [retry_growths[-1] if retry_growths else 0.30]
                    * (max_attempts - len(retry_growths))
                )
            backoff_base = max(
                0.0,
                float(
                    (self.config or {}).get("stop_replace_retry_backoff_seconds", 0.25)
                ),
            )
            backoff_max = max(
                backoff_base,
                float(
                    (self.config or {}).get(
                        "stop_replace_retry_backoff_max_seconds", 2.0
                    )
                ),
            )

            # Stop replacement is authoritative: cancel any tracked exit
            # orders before placing the new protective stop.
            cancel_ids = []
            for key in (
                "stop_order_id",
                "take_profit_order_id",
                "limit_order_id",
                "oco_order_id",
            ):
                order_id = state.get(key)
                if order_id and order_id not in cancel_ids:
                    cancel_ids.append(order_id)
            for order_id in cancel_ids:
                try:
                    self.exchange.cancel_order(
                        order_id, symbol, _cancel_params(self.config)
                    )
                except Exception as exc:
                    category = _classify_exchange_error(exc)
                    text = str(exc).lower()
                    already_done = any(
                        token in text
                        for token in ("not found", "unknown order", "filled", "closed")
                    )
                    if not already_done:
                        raise RuntimeError(
                            f"cancel failed before stop replace for {symbol}: "
                            f"{category}: {exc}"
                        ) from exc

            for key in ("stop_order_id", "take_profit_order_id", "limit_order_id"):
                state[key] = None
            canceled_existing = bool(cancel_ids)

            market = _load_market(self.exchange, symbol)
            amount = _exchange_precision(
                self.exchange, symbol, float(state["size"]), kind="amount"
            )
            _validate_order_filters(
                symbol,
                market,
                amount=amount,
                price=max(
                    float(state.get("entry_price", stop_price_candidate)),
                    stop_price_candidate,
                ),
            )

            new_stop_order = None
            stop_price = stop_price_candidate
            entry_price = _safe_float(state.get("entry_price"), default=np.nan)
            pending_reason = state.get("_pending_stop_reason") or state.get(
                "stop_reason", "stop_replaced"
            )

            for attempt_idx in range(max_attempts):
                retry_growth = retry_growths[min(attempt_idx, len(retry_growths) - 1)]
                current_price = np.nan
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = _safe_float(ticker.get("last"), default=np.nan)
                except Exception as price_exc:
                    _append_position_event(
                        state,
                        "stop_replace_price_fetch_failed",
                        attempt=attempt_idx + 1,
                        error_category=_classify_exchange_error(price_exc),
                        error=str(price_exc),
                    )

                attempt_stop = stop_price_candidate
                if np.isfinite(current_price) and current_price > 0.0:
                    adjusted = _widen_stop_away_from_market(
                        side=side,
                        stop_price=attempt_stop,
                        current_price=current_price,
                        round_fee_buffer_pct=round_fee_buffer,
                        retry_gap_growth=retry_growth,
                        immediate_buffer_pct=immediate_buffer,
                    )
                    adjusted = _exchange_precision(
                        self.exchange, symbol, float(adjusted), kind="price"
                    )
                    if abs(float(adjusted) - float(attempt_stop)) > 1e-12:
                        _append_position_event(
                            state,
                            "stop_replace_candidate_adjusted",
                            attempt=attempt_idx + 1,
                            previous_candidate=float(attempt_stop),
                            adjusted_candidate=float(adjusted),
                            current_price=float(current_price),
                            retry_gap_growth=float(retry_growth),
                            round_fee_buffer_pct=float(round_fee_buffer),
                        )
                    attempt_stop = adjusted

                improves_existing = True
                if np.isfinite(old_stop_price):
                    improves_existing = (
                        attempt_stop > old_stop_price
                        if side == "long"
                        else attempt_stop < old_stop_price
                    )
                if not improves_existing:
                    _append_position_event(
                        state,
                        "stop_replace_skipped",
                        reason="adjusted_candidate_not_better_than_existing_stop",
                        current_price=(
                            float(current_price) if np.isfinite(current_price) else None
                        ),
                        candidate_stop=float(attempt_stop),
                        previous_stop=old_stop_price,
                    )
                    tprint(
                        f"Skipping SL update for {symbol}: adjusted candidate "
                        f"{attempt_stop:.8g} is not better than existing "
                        f"{old_stop_price:.8g}"
                    )
                    return

                if (
                    str(pending_reason) in {"capital_preservation", "trailing_profit"}
                    and np.isfinite(entry_price)
                    and entry_price > 0.0
                ):
                    breakeven_stop = (
                        entry_price * (1.0 + round_fee_buffer)
                        if side == "long"
                        else entry_price * (1.0 - round_fee_buffer)
                    )
                    non_loss_ok = (
                        attempt_stop >= breakeven_stop
                        if side == "long"
                        else attempt_stop <= breakeven_stop
                    )
                    if not non_loss_ok:
                        _append_position_event(
                            state,
                            "stop_replace_skipped",
                            reason="adjusted_candidate_below_fee_breakeven",
                            candidate_stop=float(attempt_stop),
                            fee_breakeven_stop=float(breakeven_stop),
                            current_price=(
                                float(current_price)
                                if np.isfinite(current_price)
                                else None
                            ),
                            round_fee_buffer_pct=float(round_fee_buffer),
                        )
                        tprint(
                            f"Skipping SL update for {symbol}: adjusted stop "
                            f"{attempt_stop:.8g} would not clear fee breakeven "
                            f"{breakeven_stop:.8g}"
                        )
                        return

                if (
                    np.isfinite(current_price)
                    and current_price > 0.0
                    and not _stop_side_is_valid(side, attempt_stop, current_price)
                ):
                    _append_position_event(
                        state,
                        "stop_replace_attempt_failed",
                        attempt=attempt_idx + 1,
                        error_category="trigger_price_rejected",
                        reject_reason="LOCAL_STOP_SIDE_INVALID",
                        current_price=float(current_price),
                        candidate_stop=float(attempt_stop),
                    )
                    continue

                try:
                    stop_price = attempt_stop
                    new_stop_order = self.exchange.create_order(
                        symbol=symbol,
                        type="STOP_LOSS",
                        side="sell" if state["side"] == "long" else "buy",
                        amount=amount,
                        price=stop_price,
                        params={
                            **_order_params(self.config, reduce_only=True),
                            "stopPrice": stop_price,
                        },
                    )
                    break
                except Exception as create_exc:
                    last_error = create_exc
                    category = _classify_exchange_error(create_exc)
                    reject_reason = _exchange_reject_reason(create_exc)
                    _append_position_event(
                        state,
                        "stop_replace_attempt_failed",
                        attempt=attempt_idx + 1,
                        error_category=category,
                        reject_reason=reject_reason,
                        current_price=(
                            float(current_price) if np.isfinite(current_price) else None
                        ),
                        candidate_stop=float(attempt_stop),
                        retry_gap_growth=float(retry_growth),
                        error=str(create_exc),
                    )
                    tprint(
                        f"Stop replace attempt {attempt_idx + 1}/{max_attempts} "
                        f"failed for {symbol}: category={category} "
                        f"reason={reject_reason} candidate={attempt_stop:.8g} "
                        f"current={current_price if np.isfinite(current_price) else 'n/a'} "
                        f"error={create_exc}"
                    )
                    retryable = category in {
                        "trigger_price_rejected",
                        "rate_limited",
                        "network_timeout",
                    }
                    if not retryable or attempt_idx >= max_attempts - 1:
                        raise
                    sleep_s = min(backoff_max, backoff_base * (2**attempt_idx))
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)

            if new_stop_order is None:
                raise RuntimeError(
                    f"stop replace did not create an order for {symbol}"
                ) from last_error

            state["stop_price"] = stop_price
            state["stop_order_id"] = new_stop_order.get("id")
            state["size"] = amount
            stop_reason = state.pop("_pending_stop_reason", None) or state.get(
                "stop_reason", "stop_replaced"
            )
            stop_detail = state.pop("_pending_stop_reason_detail", None) or state.get(
                "stop_reason_detail", stop_reason
            )
            state["stop_reason"] = stop_reason
            state["stop_reason_detail"] = stop_detail
            state.pop("stop_update_error", None)
            state.pop("stop_update_error_category", None)
            _append_position_event(
                state,
                "stop_replaced",
                stop_reason=stop_reason,
                previous_stop=old_stop_price,
                new_stop=float(stop_price),
                stop_order_id=state.get("stop_order_id"),
                entry_price=entry_price,
                stop_dev_pct=(
                    (float(stop_price) - entry_price) / max(abs(entry_price), 1e-12)
                    if np.isfinite(entry_price)
                    else np.nan
                ),
                mfe=_safe_float(state.get("mfe"), 0.0),
                mae=_safe_float(state.get("mae"), 0.0),
            )
            tprint(
                f"Updated SL for {symbol} to {stop_price:.8g} "
                f"reason={stop_reason} detail={stop_detail}"
            )

        except Exception as e:
            category = _classify_exchange_error(e)
            reject_reason = _exchange_reject_reason(e)
            state["stop_update_error"] = str(e)
            state["stop_update_error_category"] = category
            state["stop_update_reject_reason"] = reject_reason
            _append_position_event(
                state,
                "stop_replace_failed",
                previous_stop=old_stop_price,
                candidate_stop=float(new_stop_price),
                error_category=category,
                reject_reason=reject_reason,
                error=str(e),
            )
            tprint(
                f"Error updating SL for {symbol}: category={category} "
                f"reason={reject_reason}: {e}"
            )
            if canceled_existing and np.isfinite(old_stop_price):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = _safe_float(ticker.get("last"), default=np.nan)
                    restore_ok = not np.isfinite(current_price)
                    if side == "long" and np.isfinite(current_price):
                        restore_ok = old_stop_price < current_price
                    if side == "short" and np.isfinite(current_price):
                        restore_ok = old_stop_price > current_price
                    if restore_ok:
                        restore_order = self.exchange.create_order(
                            symbol=symbol,
                            type="STOP_LOSS",
                            side="sell" if side == "long" else "buy",
                            amount=_exchange_precision(
                                self.exchange,
                                symbol,
                                float(state["size"]),
                                kind="amount",
                            ),
                            price=_exchange_precision(
                                self.exchange,
                                symbol,
                                old_stop_price,
                                kind="price",
                            ),
                            params={
                                **_order_params(self.config, reduce_only=True),
                                "stopPrice": _exchange_precision(
                                    self.exchange,
                                    symbol,
                                    old_stop_price,
                                    kind="price",
                                ),
                            },
                        )
                        state["stop_price"] = old_stop_price
                        state["stop_order_id"] = restore_order.get("id")
                        state["stop_reason"] = "original_stop_loss"
                        state["stop_reason_detail"] = (
                            "restored_previous_stop_after_replace_failure"
                        )
                        _append_position_event(
                            state,
                            "stop_restored",
                            restored_stop=float(old_stop_price),
                            stop_order_id=state.get("stop_order_id"),
                        )
                        tprint(
                            f"Restored previous SL for {symbol}: "
                            f"{old_stop_price:.4f}"
                        )
                except Exception as restore_exc:
                    state["stop_restore_error"] = str(restore_exc)
                    state["stop_restore_error_category"] = _classify_exchange_error(
                        restore_exc
                    )
                    tprint(
                        f"CRITICAL: failed to restore previous SL for {symbol}: "
                        f"{state['stop_restore_error_category']}: {restore_exc}"
                    )

    def _close_position(
        self, symbol: str, state: Dict[str, Any], current_price: float, reason: str
    ):
        """Close position and remove from tracking.

        Args:
            symbol: Trading symbol
            state: Position state dictionary
            current_price: Current market price
            reason: Reason for closing (e.g., "stop_loss")
        """
        try:
            # Cancel any existing orders
            if state.get("stop_order_id"):
                try:
                    self.exchange.cancel_order(
                        state["stop_order_id"], symbol, _cancel_params(self.config)
                    )
                except Exception:
                    pass

            # Market close
            close_order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="sell" if state["side"] == "long" else "buy",
                amount=state["size"],
                params=_order_params(self.config, reduce_only=True),
            )
            close_order.setdefault("average", current_price)
            close_order.setdefault("price", current_price)

            tprint(f"Closed {symbol} at {current_price:.4f}, reason: {reason}")

            # Log trade result
            pnl = 0
            if state["side"] == "long":
                pnl = (current_price - state["entry_price"]) * state["size"]
            else:
                pnl = (state["entry_price"] - current_price) * state["size"]

            tprint(f"  PnL: {pnl:.2f}, MFE: {state['mfe']*100:.2f}%")
            _append_position_event(
                state,
                "position_closed",
                reason=reason,
                close_price=float(current_price),
                gross_pnl=float(pnl),
                mfe=_safe_float(state.get("mfe"), 0.0),
                mae=_safe_float(state.get("mae"), 0.0),
                stop_reason=state.get("stop_reason"),
            )
            state["last_close_metrics"] = _closed_trade_metrics(
                symbol, state, close_order, reason=reason
            )

        except Exception as e:
            category = _classify_exchange_error(e)
            state["close_error"] = str(e)
            state["close_error_category"] = category
            tprint(f"Error closing {symbol}: {category}: {e}")
        finally:
            # Remove from active positions
            with self._positions_lock:
                if symbol in self.active_positions:
                    del self.active_positions[symbol]

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        with self._positions_lock:
            return self.active_positions.copy()

    def fetch_5m_ohlcv_for_positions(self) -> Dict[str, pd.DataFrame]:
        """Fetch 5m OHLCV data for all active positions.

        Uses hf_data_loader.fetch_ohlcv_5m() to get high-frequency
        data for trailing profit analysis.

        Returns:
            Dictionary mapping symbol to 5m OHLCV DataFrame
        """
        results = {}
        current_time = pd.Timestamp.now(tz="UTC")

        with self._positions_lock:
            items = list(self.active_positions.items())
        for symbol, state in items:
            try:
                entry_time = state.get("entry_time", current_time)

                # Fetch 5m OHLCV from 1 hour before entry to now + 1 hour
                ohlcv_5m = hf_data_loader.fetch_ohlcv_5m(
                    self.exchange,
                    symbol,
                    entry_time - pd.Timedelta(hours=1),
                    current_time + pd.Timedelta(hours=1),
                )

                # Safely check ohlcv_5m
                try:
                    ohlcv_not_empty = (
                        ohlcv_5m is not None
                        and isinstance(ohlcv_5m, (pd.DataFrame, pd.Series))
                        and not (hasattr(ohlcv_5m, "empty") and ohlcv_5m.empty)
                    )
                except Exception:
                    ohlcv_not_empty = False

                if ohlcv_not_empty:
                    results[symbol] = ohlcv_5m
                    # Also store in position state
                    state["ohlcv_5m"] = ohlcv_5m
                else:
                    tprint(f"Warning: No 5m OHLCV data fetched for {symbol}")

            except Exception as e:
                tprint(
                    f"Error fetching 5m OHLCV for {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

        return results

    def monitor_order_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Fetch current stop-order status for all tracked positions."""
        statuses: Dict[str, Dict[str, Any]] = {}
        with self._positions_lock:
            items = list(self.active_positions.items())
        for symbol, state in items:
            stop_order_id = state.get("stop_order_id")
            if not stop_order_id:
                statuses[symbol] = {"status": "missing_stop_order"}
                continue
            try:
                fetch_order = getattr(self.exchange, "fetch_order", None)
                if not callable(fetch_order):
                    statuses[symbol] = {"status": "fetch_order_unavailable"}
                    continue
                order = fetch_order(stop_order_id, symbol, _cancel_params(self.config))
                status = str(order.get("status", "") or "").lower()
                state["last_order_status"] = status
                state["last_order_check_ts"] = pd.Timestamp.now(tz="UTC")
                statuses[symbol] = {
                    "status": status or "unknown",
                    "order": order,
                }
                if status in {"closed", "filled"}:
                    state["exit_reason"] = "stop_loss_filled"
                    _append_position_event(
                        state,
                        "stop_order_filled",
                        stop_order_id=stop_order_id,
                        stop_price=state.get("stop_price"),
                        stop_reason=state.get("stop_reason"),
                        order_status=status,
                    )
                    statuses[symbol]["closed_trade"] = _closed_trade_metrics(
                        symbol, state, order, reason="stop_loss_filled"
                    )
                    with self._positions_lock:
                        self.active_positions.pop(symbol, None)
                elif status in {"canceled", "cancelled", "expired", "rejected"}:
                    state["stop_order_error"] = f"stop_order_{status}"
            except Exception as exc:
                category = _classify_exchange_error(exc)
                reconciled_order: Optional[Dict[str, Any]] = None
                try:
                    fetch_orders = getattr(self.exchange, "fetch_orders", None)
                    if callable(fetch_orders):
                        recent_orders = fetch_orders(
                            symbol, None, 20, _cancel_params(self.config)
                        )
                        stop_id = str(stop_order_id)
                        for candidate in reversed(list(recent_orders or [])):
                            if str(candidate.get("id")) == stop_id:
                                reconciled_order = candidate
                                break
                except Exception as reconcile_exc:
                    tprint(
                        f"Error reconciling stop order for {symbol}: "
                        f"{_classify_exchange_error(reconcile_exc)}: {reconcile_exc}"
                    )
                if reconciled_order is not None:
                    status = str(reconciled_order.get("status", "") or "").lower()
                    state["last_order_status"] = status
                    state["last_order_check_ts"] = pd.Timestamp.now(tz="UTC")
                    statuses[symbol] = {
                        "status": status or "unknown",
                        "order": reconciled_order,
                        "reconciled_after_error": True,
                        "fetch_order_error_category": category,
                        "fetch_order_error": str(exc),
                    }
                    if status in {"closed", "filled"}:
                        state["exit_reason"] = "stop_loss_filled"
                        _append_position_event(
                            state,
                            "stop_order_filled",
                            stop_order_id=stop_order_id,
                            stop_price=state.get("stop_price"),
                            stop_reason=state.get("stop_reason"),
                            order_status=status,
                            reconciled_after_error=True,
                        )
                        statuses[symbol]["closed_trade"] = _closed_trade_metrics(
                            symbol, state, reconciled_order, reason="stop_loss_filled"
                        )
                        with self._positions_lock:
                            self.active_positions.pop(symbol, None)
                        tprint(
                            f"Reconciled filled stop order for {symbol} after "
                            f"fetch_order error: order_id={stop_order_id}"
                        )
                    elif status in {"canceled", "cancelled", "expired", "rejected"}:
                        state["stop_order_error"] = f"stop_order_{status}"
                    continue
                statuses[symbol] = {
                    "status": "error",
                    "error_category": category,
                    "error": str(exc),
                }
                tprint(f"Error monitoring stop order for {symbol}: {category}: {exc}")
        return statuses

    def close_all_positions(self):
        """Close all active positions."""
        with self._positions_lock:
            symbols = list(self.active_positions.keys())
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
                with self._positions_lock:
                    state = self.active_positions.get(symbol)
                if state is not None:
                    self._close_position(
                        symbol, state, current_price, "emergency_close"
                    )
            except Exception as e:
                tprint(
                    f"Error emergency closing {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )


class TradeExecutor:
    """Handles trade execution in live or shadow mode with OCO support."""

    def __init__(
        self,
        mode: str = "shadow",
        exchange: Optional[Any] = None,
        capital: float = 10000.0,
        max_position_size: float = 0.1,
        bucket_params: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trade executor.

        Args:
            mode: "live" or "shadow"
            exchange: ccxt exchange instance (required for live mode)
            capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            bucket_params: Parameters from ridge position sizer for SL/TP
            config: Additional configuration options
        """
        self.mode = mode
        self.exchange = exchange
        self.capital = capital
        self.max_position_size = max_position_size
        self.bucket_params = bucket_params or {}
        self.config = config or {}

        # Track positions
        self.positions = {}
        self._state_lock = threading.RLock()
        self._last_trade_timestamps: Dict[str, pd.Timestamp] = {}

        # Initialize OCO executor for live mode
        self.oco_executor: Optional[OCOExecutor] = None
        if _is_live_execution_mode(mode) and exchange is not None:
            self.oco_executor = OCOExecutor(
                exchange=exchange, bucket_params=self.bucket_params, config=self.config
            )
            if bool(self.config.get("enable_background_oco_monitor", False)):
                self.oco_executor.start_monitoring()
            else:
                tprint(
                    "OCOExecutor background monitor disabled; inference loop owns "
                    "15m stop-policy updates"
                )

        self.order_manager = _make_legacy_order_manager(
            exchange,
            bool(
                _is_live_execution_mode(mode)
                and exchange is not None
                and self.config.get("enable_legacy_order_manager", False)
            ),
        )

        tprint(f"TradeExecutor initialized in {mode} mode")

    def _fetch_margin_balance(self) -> Dict[str, Any]:
        """Fetch cross-margin balances through ccxt with Binance fallbacks."""
        if self.exchange is None:
            return {}
        attempts = (
            {"type": "margin", "marginMode": "cross"},
            {"type": "margin"},
            {},
        )
        last_error: Optional[Exception] = None
        for params in attempts:
            try:
                balance = self.exchange.fetch_balance(params)
                if isinstance(balance, dict):
                    return balance
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return {}

    def _asset_quote_value(
        self,
        asset: str,
        amount: float,
        quote: str,
    ) -> Tuple[float, Optional[float], Optional[str]]:
        """Convert an asset quantity to quote value using current tickers."""
        asset_u = str(asset or "").upper()
        quote_u = str(quote or "").upper()
        amount_f = abs(float(amount))
        if amount_f <= 0.0:
            return 0.0, None, None
        if asset_u == quote_u:
            return amount_f, 1.0, quote_u
        symbol = _symbol_from_asset_quote(asset_u, quote_u)
        try:
            _load_market(self.exchange, symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            px = _safe_float(ticker.get("last"), default=np.nan)
            if np.isfinite(px) and px > 0.0:
                return amount_f * float(px), float(px), symbol
        except Exception:
            pass
        return np.nan, None, symbol

    def _parse_margin_assets(
        self, balance: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Normalize ccxt and Binance raw margin balance payloads by asset."""
        assets: Dict[str, Dict[str, float]] = {}

        def _slot(asset: str) -> Dict[str, float]:
            asset_u = str(asset or "").upper()
            return assets.setdefault(
                asset_u,
                {"free": 0.0, "used": 0.0, "total": 0.0, "debt": 0.0, "interest": 0.0},
            )

        for asset, payload in balance.items():
            if asset in {"info", "free", "used", "total"} or not isinstance(
                payload, dict
            ):
                continue
            slot = _slot(asset)
            slot["free"] = max(slot["free"], _safe_float(payload.get("free"), 0.0))
            slot["used"] = max(slot["used"], _safe_float(payload.get("used"), 0.0))
            slot["total"] = max(slot["total"], _safe_float(payload.get("total"), 0.0))
            slot["debt"] = max(
                slot["debt"],
                _safe_float(
                    payload.get("debt", payload.get("borrowed", payload.get("loan"))),
                    0.0,
                ),
            )
            slot["interest"] = max(
                slot["interest"], _safe_float(payload.get("interest"), 0.0)
            )

        info = balance.get("info")
        user_assets = None
        if isinstance(info, dict):
            user_assets = info.get("userAssets") or info.get("userassets")
        if isinstance(user_assets, list):
            for row in user_assets:
                if not isinstance(row, dict):
                    continue
                asset = row.get("asset")
                if not asset:
                    continue
                slot = _slot(asset)
                free = _safe_float(row.get("free"), 0.0)
                locked = _safe_float(row.get("locked"), 0.0)
                borrowed = _safe_float(row.get("borrowed"), 0.0)
                interest = _safe_float(row.get("interest"), 0.0)
                net_asset = _safe_float(row.get("netAsset"), np.nan)
                slot["free"] = max(slot["free"], free)
                slot["used"] = max(slot["used"], locked)
                slot["total"] = max(
                    slot["total"],
                    (
                        free + locked
                        if not np.isfinite(net_asset)
                        else max(free + locked, 0.0)
                    ),
                )
                slot["debt"] = max(slot["debt"], borrowed + interest)
                slot["interest"] = max(slot["interest"], interest)
        return assets

    def _track_external_margin_position(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        quote_value: float,
        reason: str,
    ) -> bool:
        """Import a real external margin position for monitoring-only tracking."""
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            return False
        params = self.get_bucket_params(None)
        stop_frac = _safe_float(
            params.get("fixed_stop_loss_pct", params.get("stop_loss_pct")),
            default=0.03,
        )
        if not np.isfinite(stop_frac) or stop_frac <= 0.0:
            stop_frac = 0.03
        sl_mult = _safe_float(params.get("sl_mult"), default=1.0)
        if not np.isfinite(sl_mult) or sl_mult <= 0.0:
            sl_mult = 1.0
        attach_barrier_frac = float(stop_frac) / float(sl_mult)
        stop_price = (
            entry_price * (1.0 - stop_frac)
            if side == "long"
            else entry_price * (1.0 + stop_frac)
        )
        now = pd.Timestamp.now(tz="UTC")
        state = {
            "side": side,
            "size": float(amount),
            "quote_size": float(quote_value),
            "entry_price": float(entry_price),
            "timestamp": datetime.now(),
            "entry_time": now,
            "bucket_key": "external_margin_reconciliation",
            "stop_price": float(stop_price),
            "limit_price": None,
            "peak_price": float(entry_price),
            "mfe": 0.0,
            "last_update": now,
            "last_5m_eval_ts": None,
            "external_position": True,
            "monitoring_only": True,
            "reconciliation_reason": reason,
        }
        if self.oco_executor is not None:
            oco_state = state.copy()
            oco_state.update(
                {
                    "oco_order_id": None,
                    "stop_order_id": None,
                            "take_profit_order_id": None,
                            "atr": float(attach_barrier_frac),
                            "sl_mult": float(sl_mult),
                        }
                    )
            adopted_stop_order = None
            try:
                fetch_open_orders = getattr(self.exchange, "fetch_open_orders", None)
                if callable(fetch_open_orders):
                    for order in (
                        fetch_open_orders(symbol, _cancel_params(self.config)) or []
                    ):
                        if not isinstance(order, dict):
                            continue
                        order_side = str(order.get("side") or "").lower()
                        order_type = str(order.get("type") or "").upper()
                        info = (
                            order.get("info")
                            if isinstance(order.get("info"), dict)
                            else {}
                        )
                        stop_price_raw = (
                            order.get("stopPrice")
                            or order.get("stop_price")
                            or info.get("stopPrice")
                            or order.get("price")
                        )
                        stop_px = _safe_float(stop_price_raw, default=np.nan)
                        expected_side = "sell" if side == "long" else "buy"
                        if (
                            order_side == expected_side
                            and "STOP" in order_type
                            and np.isfinite(stop_px)
                            and stop_px > 0.0
                        ):
                            adopted_stop_order = order
                            break
            except Exception as exc:
                tprint(
                    f"External position stop-order adoption failed for {symbol}: "
                    f"{_classify_exchange_error(exc)}: {exc}"
                )

            if isinstance(adopted_stop_order, dict):
                oco_state["stop_order_id"] = adopted_stop_order.get("id")
                adopted_stop = _safe_float(
                    adopted_stop_order.get("stopPrice")
                    or adopted_stop_order.get("stop_price")
                    or (
                        adopted_stop_order.get("info", {}).get("stopPrice")
                        if isinstance(adopted_stop_order.get("info"), dict)
                        else None
                    )
                    or adopted_stop_order.get("price"),
                    default=np.nan,
                )
                if np.isfinite(adopted_stop) and adopted_stop > 0.0:
                    oco_state["stop_price"] = float(adopted_stop)
                with self.oco_executor._positions_lock:
                    self.oco_executor.active_positions[symbol] = oco_state
                with self._state_lock:
                    self.positions[symbol] = state.copy()
                tprint(
                    f"Imported external margin position with existing STOP_LOSS: "
                    f"{symbol} side={side} stop_order_id={oco_state.get('stop_order_id')} "
                    f"stop={oco_state.get('stop_price')}"
                )
                return True

            protect_external = bool(
                self.config.get("protect_external_margin_positions", True)
            )
            if protect_external and _is_live_execution_mode(self.mode):
                try:
                    stop_result = self.oco_executor.place_oco_order(
                        symbol=symbol,
                        side=side,
                        entry_price=float(entry_price),
                        size=float(amount),
                        bucket_key=None,
                        atr_frac=float(attach_barrier_frac),
                    )
                    with self.oco_executor._positions_lock:
                        tracked = self.oco_executor.active_positions.get(symbol)
                        if isinstance(tracked, dict):
                            tracked.update(oco_state)
                            tracked["stop_order_id"] = stop_result.get("stop_order_id")
                            tracked["stop_price"] = stop_result.get(
                                "stop_price", tracked.get("stop_price")
                            )
                            tracked["external_stop_attached"] = bool(
                                stop_result.get("success")
                            )
                    if stop_result.get("success"):
                        tprint(
                            f"Imported external margin position and attached STOP_LOSS: "
                            f"{symbol} side={side} stop={stop_result.get('stop_price')} "
                            f"stop_order_id={stop_result.get('stop_order_id')}"
                        )
                        with self._state_lock:
                            self.positions[symbol] = state.copy()
                        return True
                    else:
                        with self.oco_executor._positions_lock:
                            self.oco_executor.active_positions.pop(symbol, None)
                        with self._state_lock:
                            self.positions.pop(symbol, None)
                        tprint(
                            f"External margin position was not imported because "
                            f"STOP_LOSS attach failed: "
                            f"{symbol} category={stop_result.get('error_category')} "
                            f"error={stop_result.get('error')}"
                        )
                    return False
                except Exception as exc:
                    with self.oco_executor._positions_lock:
                        self.oco_executor.active_positions.pop(symbol, None)
                    with self._state_lock:
                        self.positions.pop(symbol, None)
                    tprint(
                        f"External position STOP_LOSS attach failed for {symbol}: "
                        f"{_classify_exchange_error(exc)}: {exc}"
                    )
            tprint(
                f"External margin position not imported for monitoring without "
                f"STOP_LOSS protection: {symbol} side={side}"
            )
            return False
        return False

    def reconcile_cross_margin_account(self) -> Dict[str, Any]:
        """Classify existing cross-margin balances/debts before live trading."""
        report: Dict[str, Any] = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "mode": self.mode,
            "execution_account": _execution_account(self.config),
            "margin_mode": _margin_mode(self.config),
            "quote_currency": str(
                self.config.get("live_quote_currency")
                or self.config.get("quote_currency")
                or "USDC"
            ).upper(),
            "dust_quote_threshold": float(
                self.config.get("cross_margin_dust_quote_threshold", 10.0)
            ),
            "cross_margin_residual_assets": list(
                self.config.get("cross_margin_residual_assets", ["BNB"])
            ),
            "items": [],
            "summary": {},
        }
        if (
            not _is_live_execution_mode(self.mode)
            or _execution_account(self.config) != "margin"
            or _margin_mode(self.config) != "cross"
            or self.exchange is None
        ):
            report["summary"] = {"skipped": True, "reason": "not_cross_margin_live"}
            return report

        try:
            balance = self._fetch_margin_balance()
        except Exception as exc:
            category = _classify_exchange_error(exc)
            tprint(f"Cross-margin reconciliation failed: {category}: {exc}")
            report["summary"] = {
                "skipped": True,
                "reason": "fetch_balance_failed",
                "error_category": category,
                "error": str(exc),
            }
            return report

        quote = str(report["quote_currency"])
        threshold = float(report["dust_quote_threshold"])
        residual_assets = {
            str(asset).upper()
            for asset in self.config.get("cross_margin_residual_assets", ["BNB"])
        }
        import_non_dust = bool(
            self.config.get("import_external_margin_positions", True)
        )
        assets = self._parse_margin_assets(balance)
        counts: Dict[str, int] = {}

        for asset, vals in sorted(assets.items()):
            if not asset:
                continue
            total = max(_safe_float(vals.get("total"), 0.0), 0.0)
            debt = max(_safe_float(vals.get("debt"), 0.0), 0.0)
            interest = max(_safe_float(vals.get("interest"), 0.0), 0.0)
            free = max(_safe_float(vals.get("free"), 0.0), 0.0)
            used = max(_safe_float(vals.get("used"), 0.0), 0.0)

            for kind, amount, side in (
                ("debt", debt, "short"),
                ("balance", total, "long"),
            ):
                if amount <= 0.0:
                    continue
                quote_value, px, symbol = self._asset_quote_value(asset, amount, quote)
                if kind == "balance" and asset in residual_assets:
                    classification = "fee_token_or_collateral_residual"
                elif not np.isfinite(quote_value) or quote_value <= 0.0:
                    classification = f"unpriced_{kind}"
                elif quote_value < threshold:
                    classification = (
                        "dust_to_repay" if kind == "debt" else "dust_residual"
                    )
                elif asset == quote and kind == "balance":
                    classification = "quote_balance_available"
                else:
                    classification = f"external_{side}_position"

                item = {
                    "asset": asset,
                    "kind": kind,
                    "side": side,
                    "amount": float(amount),
                    "free": float(free),
                    "used": float(used),
                    "debt": float(debt),
                    "interest": float(interest),
                    "quote_value": (
                        float(quote_value) if np.isfinite(quote_value) else None
                    ),
                    "price": float(px) if px is not None else None,
                    "symbol": symbol,
                    "classification": classification,
                    "imported_for_monitoring": False,
                }
                if (
                    import_non_dust
                    and classification == f"external_{side}_position"
                    and symbol
                    and px is not None
                ):
                    imported = self._track_external_margin_position(
                        symbol=symbol,
                        side=side,
                        amount=float(amount),
                        entry_price=float(px),
                        quote_value=float(quote_value),
                        reason=classification,
                    )
                    item["imported_for_monitoring"] = bool(imported)
                    if not imported:
                        item["monitoring_skip_reason"] = "stop_loss_not_attached"
                report["items"].append(item)
                counts[classification] = counts.get(classification, 0) + 1

        report["summary"] = {
            "skipped": False,
            "item_count": len(report["items"]),
            "counts": counts,
            "active_positions_after_reconcile": len(self.get_active_positions()),
        }
        tprint(
            "Cross-margin reconciliation complete: "
            f"items={len(report['items'])} counts={counts} "
            f"active_positions={report['summary']['active_positions_after_reconcile']}"
        )
        return report

    def get_bucket_params(self, bucket_key: Optional[str] = None) -> Dict[str, Any]:
        """Return normalized bucket params for live or shadow execution."""
        if self.oco_executor is not None:
            return self.oco_executor.get_bucket_params(bucket_key)
        params = {
            "sl_mult": 1.0,
            "tp_mult": 3.0,
            "trail_mult": 0.25,
            "giveback_pct": 0.005,
            "profit_lock_amount": 0.003,
            "mfe_early_exit_threshold": 0.02,
            "max_hold_hours": 48,
        }
        params.update(
            {k: v for k, v in self.bucket_params.items() if not isinstance(v, dict)}
        )
        if bucket_key is None:
            return params
        raw_key = str(bucket_key or "")
        core_key = strategy_core_id(raw_key)
        key_lower = raw_key.lower()
        key_upper = raw_key.upper()
        bucket = (
            self.bucket_params.get(raw_key, {})
            or self.bucket_params.get(core_key, {})
            or self.bucket_params.get(key_lower, {})
            or self.bucket_params.get(key_upper, {})
        )
        if (
            not bucket
            and "buckets" in self.bucket_params
            and isinstance(self.bucket_params["buckets"], dict)
        ):
            bucket = (
                self.bucket_params["buckets"].get(key_upper, {})
                or self.bucket_params["buckets"].get(raw_key, {})
                or self.bucket_params["buckets"].get(key_lower, {})
            )
        if isinstance(bucket, dict):
            params.update(bucket)
        return params

    def get_cooldown_hours(self, bucket_key: Optional[str] = None) -> float:
        """Return zero: live inference cooldown is handled on losing closes only."""
        return 0.0

    def execute_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
        ohlcv_reference_price: Optional[float] = None,
        trade_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a trade with OCO order support.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "long" or "short"
            size: Position size in quote currency
            price: Limit price (None for market order)
            bucket_key: Bucket key for getting SL/TP parameters (required for OCO)

        Returns:
            Dictionary with execution results
        """
        # The live inference path passes quote-currency USDT after portfolio
        # gating. Keep legacy fractional behavior for direct callers/tests.
        size_f = abs(float(size))
        if size_f <= 1.0:
            position_value = size_f * self.capital * self.max_position_size
        else:
            position_value = size_f

        cooldown_hours = self.get_cooldown_hours(bucket_key)
        now_utc = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            active_oco = bool(
                self.oco_executor and symbol in self.oco_executor.get_active_positions()
            )
            if symbol in self.positions or active_oco:
                return {"success": False, "error": f"symbol {symbol} already active"}
            last_ts = self._last_trade_timestamps.get(symbol)
            if (
                last_ts is not None
                and cooldown_hours > 0.0
                and now_utc
                < (pd.Timestamp(last_ts) + pd.Timedelta(hours=float(cooldown_hours)))
            ):
                return {
                    "success": False,
                    "error": f"symbol {symbol} in cooldown for {cooldown_hours:.1f}h",
                }

        if _is_live_execution_mode(self.mode):
            return self._execute_live(
                symbol,
                side,
                position_value,
                price,
                bucket_key,
                ohlcv_reference_price=ohlcv_reference_price,
                trade_context=trade_context,
            )
        else:
            return self._record_shadow_trade(
                symbol,
                side,
                position_value,
                price,
                bucket_key=bucket_key,
                ohlcv_reference_price=ohlcv_reference_price,
                trade_context=trade_context,
            )

    def _execute_live(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
        ohlcv_reference_price: Optional[float] = None,
        trade_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute live trade with STOP_LOSS support.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in quote currency
            price: Limit price
            bucket_key: Bucket key for OCO parameters

        Returns:
            Execution result dictionary
        """
        order_side = "buy" if side == "long" else "sell"
        market = _load_market(self.exchange, symbol)
        expected_entry_price = float(price) if price is not None else np.nan

        try:
            barrier_override = np.nan
            if bucket_key and self.oco_executor:
                barrier_override = _extract_policy_barrier_frac(
                    trade_context,
                    self.oco_executor.get_bucket_params(bucket_key),
                )
                if not np.isfinite(barrier_override) or barrier_override <= 0.0:
                    error = (
                        "missing policy barrier_pct/barrier_frac before live entry; "
                        "refusing to place unprotected order"
                    )
                    tprint(f"Refusing live entry for {symbol}: {error}")
                    return {
                        "success": False,
                        "error": error,
                        "error_category": "missing_policy_barrier_pct",
                        "symbol": symbol,
                        "side": side,
                        "size": float(size),
                    }

            if not np.isfinite(expected_entry_price):
                ticker = self.exchange.fetch_ticker(symbol)
                expected_entry_price = _safe_float(ticker.get("last"), default=np.nan)
            amount_base = self._quote_to_base_amount(
                symbol,
                quote_size=float(size),
                reference_price=expected_entry_price,
                market=market,
            )

            force_market_entries = bool(
                self.config.get("force_market_entry_orders", True)
            )
            if force_market_entries and price is not None:
                tprint(
                    f"Live entry for {symbol}: forcing market order; "
                    f"caller reference price={float(price):.8g}"
                )

            if price is not None and not force_market_entries:
                entry_price_for_order = _exchange_precision(
                    self.exchange, symbol, float(price), kind="price"
                )
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=order_side,
                    amount=amount_base,
                    price=entry_price_for_order,
                    params=_order_params(self.config, reduce_only=False),
                )
                entry_order_type = "limit"
            else:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order_side,
                    amount=amount_base,
                    params=_order_params(self.config, reduce_only=False),
                )
                entry_order_type = "market"
            fallback_price = float(expected_entry_price)
            entry_price, filled_amount, partial_fill = _extract_order_fill(
                order, fallback_price
            )
            ohlcv_entry_price = _safe_float(
                ohlcv_reference_price,
                default=_safe_float(price, default=np.nan),
            )
            entry_price_delta = (
                float(entry_price) - float(ohlcv_entry_price)
                if np.isfinite(ohlcv_entry_price)
                else np.nan
            )
            entry_price_delta_pct = (
                entry_price_delta / max(abs(float(ohlcv_entry_price)), 1e-12)
                if np.isfinite(entry_price_delta) and np.isfinite(ohlcv_entry_price)
                else np.nan
            )
            base_fee = _filled_base_fee(order, symbol)
            stop_amount_source = filled_amount
            if (
                side == "long"
                and np.isfinite(stop_amount_source)
                and stop_amount_source > base_fee
            ):
                stop_amount_source = max((stop_amount_source - base_fee) * 0.999, 0.0)
            stop_amount = (
                _exchange_precision(
                    self.exchange, symbol, stop_amount_source, kind="amount"
                )
                if np.isfinite(stop_amount_source) and stop_amount_source > 0
                else amount_base
            )

            oco_result = None
            if bucket_key and self.oco_executor:
                oco_result = self.oco_executor.place_oco_order(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    size=stop_amount,
                    bucket_key=bucket_key,
                    atr_frac=barrier_override,
                )
                if isinstance(trade_context, dict):
                    with self.oco_executor._positions_lock:
                        state = self.oco_executor.active_positions.get(symbol)
                        if isinstance(state, dict):
                            state.update(
                                {
                                    **trade_context,
                                    "quote_size": float(size),
                                    "requested_base_amount": amount_base,
                                    "base_fee_amount": base_fee,
                                    "entry_order_type": entry_order_type,
                                    "ohlcv_entry_price": ohlcv_entry_price,
                                    "entry_price_delta_vs_ohlcv": entry_price_delta,
                                    "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                                }
                            )

            context = dict(trade_context or {})
            with self._state_lock:
                self.positions[symbol] = {
                    "side": side,
                    "size": stop_amount,
                    "quote_size": float(size),
                    "requested_base_amount": amount_base,
                    "base_fee_amount": base_fee,
                    "entry_price": entry_price,
                    "ohlcv_entry_price": ohlcv_entry_price,
                    "entry_price_delta_vs_ohlcv": entry_price_delta,
                    "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                    "timestamp": datetime.now(),
                    "bucket_key": bucket_key,
                    "partial_fill": partial_fill,
                    "entry_order_type": entry_order_type,
                    **context,
                }
                self._last_trade_timestamps[symbol] = pd.Timestamp.now(tz="UTC")

            if oco_result is not None and not oco_result.get("success", False):
                error_category = oco_result.get("error_category") or "stop_loss_failed"
                return {
                    "success": False,
                    "error": oco_result.get("error") or "stop_loss_not_placed",
                    "error_category": error_category,
                    "order": order,
                    "oco_result": oco_result,
                    "symbol": symbol,
                    "side": side,
                    "size": float(size),
                    "base_amount": stop_amount,
                    "base_fee_amount": base_fee,
                    "expected_entry_price": expected_entry_price,
                    "realized_entry_price": entry_price,
                    "ohlcv_entry_price": ohlcv_entry_price,
                    "entry_price_delta_vs_ohlcv": entry_price_delta,
                    "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                    "partial_fill": partial_fill,
                    "entry_order_type": entry_order_type,
                }

            return {
                "success": True,
                "order": order,
                "oco_result": oco_result,
                "symbol": symbol,
                "side": side,
                "size": float(size),
                "base_amount": stop_amount,
                "base_fee_amount": base_fee,
                "price": entry_price,
                "expected_entry_price": expected_entry_price,
                "realized_entry_price": entry_price,
                "ohlcv_entry_price": ohlcv_entry_price,
                "entry_price_delta_vs_ohlcv": entry_price_delta,
                "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                "stop_price": (
                    oco_result.get("stop_price")
                    if isinstance(oco_result, dict)
                    else None
                ),
                "stop_order_id": (
                    oco_result.get("stop_order_id")
                    if isinstance(oco_result, dict)
                    else None
                ),
                "barrier_frac": (
                    oco_result.get("barrier_frac")
                    if isinstance(oco_result, dict)
                    else None
                ),
                "partial_fill": partial_fill,
                "entry_order_type": entry_order_type,
                "price_slippage_pct": (
                    (entry_price - expected_entry_price)
                    / max(abs(expected_entry_price), 1e-12)
                    if np.isfinite(expected_entry_price)
                    else 0.0
                ),
            }

        except Exception as e:
            category = _classify_exchange_error(e)
            tprint(f"Error executing live trade for {symbol}: {category}: {e}")
            return {"success": False, "error": str(e), "error_category": category}

    def _quote_to_base_amount(
        self,
        symbol: str,
        *,
        quote_size: float,
        reference_price: float,
        market: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Convert quote-currency notional into exchange-ready base quantity."""
        if not np.isfinite(quote_size) or quote_size <= 0:
            raise ValueError(f"invalid quote size for {symbol}: {quote_size}")
        if not np.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"invalid reference price for {symbol}: {reference_price}")
        market_info = (
            market if market is not None else _load_market(self.exchange, symbol)
        )
        raw_amount = quote_size / reference_price
        amount = _exchange_precision(self.exchange, symbol, raw_amount, kind="amount")
        _validate_order_filters(
            symbol, market_info, amount=amount, price=float(reference_price)
        )
        return amount

    def _record_shadow_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
        ohlcv_reference_price: Optional[float] = None,
        trade_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record shadow trade decision.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in quote currency
            price: Entry price

        Returns:
            Shadow trade record
        """
        params = self.get_bucket_params(bucket_key)
        expected_entry_price = float(price) if price is not None else 0.0
        entry_price = expected_entry_price
        ohlcv_entry_price = _safe_float(
            ohlcv_reference_price,
            default=_safe_float(price, default=np.nan),
        )
        entry_price_delta = (
            float(entry_price) - float(ohlcv_entry_price)
            if np.isfinite(ohlcv_entry_price)
            else np.nan
        )
        entry_price_delta_pct = (
            entry_price_delta / max(abs(float(ohlcv_entry_price)), 1e-12)
            if np.isfinite(entry_price_delta) and np.isfinite(ohlcv_entry_price)
            else np.nan
        )
        barrier_frac = _extract_policy_barrier_frac(trade_context, params)
        if not np.isfinite(barrier_frac) or barrier_frac <= 0.0:
            return {
                "timestamp": datetime.now().isoformat(),
                "mode": "shadow",
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "status": "failed",
                "success": False,
                "bucket_key": bucket_key,
                "error": "missing policy barrier_pct/barrier_frac",
                "error_category": "missing_policy_barrier_pct",
                **dict(trade_context or {}),
            }
        sl_mult = float(params.get("sl_mult", 1.0))
        if side == "long":
            stop_price = (
                entry_price * (1.0 - sl_mult * barrier_frac)
                if entry_price > 0
                else None
            )
        else:
            stop_price = (
                entry_price * (1.0 + sl_mult * barrier_frac)
                if entry_price > 0
                else None
            )
        limit_price = None
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": "shadow",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "expected_entry_price": expected_entry_price,
            "realized_entry_price": entry_price,
            "ohlcv_entry_price": ohlcv_entry_price,
            "entry_price_delta_vs_ohlcv": entry_price_delta,
            "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
            "price_slippage_pct": 0.0,
            "spread_proxy_pct": 0.0,
            "status": "recorded",
            "bucket_key": bucket_key,
            **dict(trade_context or {}),
        }

        # Update positions
        with self._state_lock:
            self.positions[symbol] = {
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "ohlcv_entry_price": ohlcv_entry_price,
                "entry_price_delta_vs_ohlcv": entry_price_delta,
                "entry_price_delta_vs_ohlcv_pct": entry_price_delta_pct,
                "timestamp": datetime.now(),
                "entry_time": pd.Timestamp.now(tz="UTC"),
                "bucket_key": bucket_key,
                "stop_price": stop_price,
                "initial_stop_price": stop_price,
                "limit_price": limit_price,
                "barrier_frac": barrier_frac,
                "barrier_pct": barrier_frac,
                "sl_mult": sl_mult,
                "stop_reason": "original_stop_loss",
                "stop_reason_detail": (
                    f"original_stop_loss: sl_mult={sl_mult:.6g} "
                    f"barrier_frac={barrier_frac:.6g}"
                ),
                "peak_price": entry_price,
                "mfe": 0.0,
                "mae": 0.0,
                "last_update": pd.Timestamp.now(tz="UTC"),
                "last_5m_eval_ts": None,
                **dict(trade_context or {}),
            }
            _append_position_event(
                self.positions[symbol],
                "entry_stop_created",
                entry_price=float(entry_price),
                stop_price=stop_price,
                stop_dev_pct=(
                    (float(stop_price) - float(entry_price))
                    / max(float(entry_price), 1e-12)
                    if stop_price is not None
                    else np.nan
                ),
                sl_mult=float(sl_mult),
                barrier_frac=float(barrier_frac),
                stop_reason="original_stop_loss",
            )
            self._last_trade_timestamps[symbol] = pd.Timestamp.now(tz="UTC")

        return record

    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None,
        reason: str = "manual_close",
    ) -> Dict[str, Any]:
        """Close an existing position.

        Args:
            symbol: Trading symbol
            price: Exit price (None for market)

        Returns:
            Close result
        """
        if symbol not in self.positions:
            return {"success": False, "error": "No position for symbol"}

        position = self.positions[symbol]
        side = position["side"]
        size = position["size"]

        if _is_live_execution_mode(self.mode):
            return self._close_live(symbol, side, size, price, reason=reason)
        else:
            return self._record_shadow_close(symbol, side, size, price, reason=reason)

    def _close_live(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        reason: str = "manual_close",
    ) -> Dict[str, Any]:
        """Close live position."""
        # Cancel OCO orders first if using OCO executor
        oco_state: Optional[Dict[str, Any]] = None
        if self.oco_executor:
            with self.oco_executor._positions_lock:
                raw_state = self.oco_executor.active_positions.get(symbol)
                if isinstance(raw_state, dict):
                    oco_state = raw_state
        if self.oco_executor and oco_state is not None:
            try:
                current_price = _safe_float(price, default=np.nan)
                if not np.isfinite(current_price) or current_price <= 0.0:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = float(ticker["last"])
                state = oco_state
                self.oco_executor._close_position(symbol, state, current_price, reason)
                with self._state_lock:
                    self.positions.pop(symbol, None)
                close_metrics = state.get("last_close_metrics")
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": "closed",
                    "size": size,
                    "reason": reason,
                    "closed_trade": close_metrics,
                }
            except Exception as e:
                tprint(
                    f"Error canceling OCO for {symbol}: "
                    f"{_classify_exchange_error(e)}: {e}"
                )

        try:
            order_side = "sell" if side == "long" else "buy"

            if price is not None:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=order_side,
                    amount=size,
                    price=price,
                    params=_order_params(self.config, reduce_only=True),
                )
            else:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order_side,
                    amount=size,
                    params=_order_params(self.config, reduce_only=True),
                )

            with self._state_lock:
                if symbol in self.positions:
                    del self.positions[symbol]

            return {
                "success": True,
                "order": order,
                "symbol": symbol,
                "side": "closed",
                "size": size,
            }
        except Exception as e:
            category = _classify_exchange_error(e)
            tprint(f"Error closing live trade for {symbol}: {category}: {e}")
            return {"success": False, "error": str(e), "error_category": category}

    def _record_shadow_close(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        reason: str = "manual_close",
    ) -> Dict[str, Any]:
        """Record shadow position close."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": "shadow",
            "symbol": symbol,
            "side": "closed",
            "original_side": side,
            "size": size,
            "price": price,
            "status": "closed",
            "reason": reason,
        }

        with self._state_lock:
            if symbol in self.positions:
                del self.positions[symbol]

        return record

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        with self._state_lock:
            return self.positions.copy()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        if self.oco_executor is not None:
            with self.oco_executor._positions_lock:
                state = self.oco_executor.active_positions.get(symbol)
                if isinstance(state, dict):
                    return dict(state)
        with self._state_lock:
            pos = self.positions.get(symbol)
            return dict(pos) if isinstance(pos, dict) else None

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return currently active positions for live or shadow mode."""
        if self.oco_executor is not None:
            return self.oco_executor.get_active_positions()
        with self._state_lock:
            return {
                sym: dict(state)
                for sym, state in self.positions.items()
                if isinstance(state, dict)
            }

    def replace_stop_loss_pct(
        self,
        symbol: str,
        stop_loss_pct: float,
    ) -> Dict[str, Any]:
        """Cancel/replace the tracked stop order at a fixed entry-distance.

        Canceling and creating a replacement stop should not create trading
        fees unless an order fills; the exchange response remains the source of
        truth for any venue-specific charges.
        """
        pct = abs(float(stop_loss_pct))
        if not np.isfinite(pct) or pct <= 0.0:
            return {
                "success": False,
                "error": f"invalid stop_loss_pct={stop_loss_pct}",
                "error_category": "invalid_stop_loss_pct",
            }

        state: Optional[Dict[str, Any]] = None
        if self.oco_executor is not None:
            with self.oco_executor._positions_lock:
                raw_state = self.oco_executor.active_positions.get(symbol)
                state = raw_state if isinstance(raw_state, dict) else None
        else:
            with self._state_lock:
                raw_state = self.positions.get(symbol)
                state = raw_state if isinstance(raw_state, dict) else None

        if state is None:
            return {
                "success": False,
                "error": f"no tracked active position for {symbol}",
                "error_category": "missing_active_position",
            }

        entry_price = _safe_float(state.get("entry_price"), default=np.nan)
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            return {
                "success": False,
                "error": f"missing entry price for {symbol}",
                "error_category": "missing_entry_price",
            }

        side = str(state.get("side", "long")).lower()
        new_stop = (
            entry_price * (1.0 - pct) if side == "long" else entry_price * (1.0 + pct)
        )

        if self.oco_executor is not None:
            before_order_id = state.get("stop_order_id")
            state.pop("stop_update_error", None)
            state.pop("stop_update_error_category", None)
            self.oco_executor._update_stop_loss(symbol, state, float(new_stop))
            success = "stop_update_error" not in state
            return {
                "success": bool(success),
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "stop_loss_pct": pct,
                "stop_price": state.get("stop_price"),
                "previous_stop_order_id": before_order_id,
                "stop_order_id": state.get("stop_order_id"),
                "error": state.get("stop_update_error"),
                "error_category": state.get("stop_update_error_category"),
            }

        with self._state_lock:
            state["stop_price"] = float(new_stop)
        return {
            "success": True,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_loss_pct": pct,
            "stop_price": float(new_stop),
            "mode": "shadow",
        }

    def monitor_orders_once(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live order statuses for active stop-loss orders once."""
        if self.oco_executor is not None:
            statuses = self.oco_executor.monitor_order_statuses()
            closed_symbols = [
                symbol
                for symbol, status in statuses.items()
                if isinstance(status, dict)
                and str(status.get("status", "")).lower() in {"closed", "filled"}
            ]
            if closed_symbols:
                with self._state_lock:
                    for symbol in closed_symbols:
                        self.positions.pop(symbol, None)
            return statuses
        statuses: Dict[str, Dict[str, Any]] = {}
        now_utc = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            items = list(self.positions.items())
            for symbol, state in items:
                if not isinstance(state, dict):
                    continue
                status = {
                    "status": "open",
                    "mode": "shadow",
                    "symbol": symbol,
                    "side": state.get("side"),
                    "size": state.get("size"),
                    "entry_price": state.get("entry_price"),
                    "stop_price": state.get("stop_price"),
                    "bucket_key": state.get("bucket_key"),
                    "last_order_check_ts": now_utc,
                }
                state["last_order_status"] = "open"
                state["last_order_check_ts"] = now_utc
                statuses[symbol] = status
        return statuses

    def update_position_policy_state(
        self,
        symbol: str,
        *,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        peak_price: Optional[float] = None,
        mfe: Optional[float] = None,
        mae: Optional[float] = None,
        stop_reason: Optional[str] = None,
        stop_reason_detail: Optional[str] = None,
        last_5m_eval_ts: Optional[pd.Timestamp] = None,
    ) -> None:
        """Persist live/shadow threshold updates from the 15m monitor."""
        if self.oco_executor is not None:
            with self.oco_executor._positions_lock:
                state = self.oco_executor.active_positions.get(symbol)
                if state is None:
                    return
                if stop_price is not None and np.isfinite(stop_price):
                    current_stop = float(state.get("stop_price", stop_price))
                    improved = (
                        state.get("side") == "long" and float(stop_price) > current_stop
                    ) or (
                        state.get("side") == "short"
                        and float(stop_price) < current_stop
                    )
                    if improved:
                        if stop_reason:
                            state["_pending_stop_reason"] = str(stop_reason)
                        if stop_reason_detail:
                            state["_pending_stop_reason_detail"] = str(
                                stop_reason_detail
                            )
                        before_replace_stop = _safe_float(
                            state.get("stop_price"), default=np.nan
                        )
                        self.oco_executor._update_stop_loss(
                            symbol, state, float(stop_price)
                        )
                        after_replace_stop = _safe_float(
                            state.get("stop_price"), default=np.nan
                        )
                        if (
                            np.isfinite(before_replace_stop)
                            and np.isfinite(after_replace_stop)
                            and abs(after_replace_stop - before_replace_stop) < 1e-12
                        ):
                            state.pop("_pending_stop_reason", None)
                            state.pop("_pending_stop_reason_detail", None)
                if limit_price is not None and np.isfinite(limit_price):
                    state["limit_price"] = float(limit_price)
                if peak_price is not None and np.isfinite(peak_price):
                    state["peak_price"] = float(peak_price)
                if mfe is not None and np.isfinite(mfe):
                    state["mfe"] = float(mfe)
                if mae is not None and np.isfinite(mae):
                    state["mae"] = float(mae)
                if last_5m_eval_ts is not None:
                    state["last_5m_eval_ts"] = pd.Timestamp(last_5m_eval_ts)
                    state["last_update"] = pd.Timestamp.now(tz="UTC")
            return
        with self._state_lock:
            state = self.positions.get(symbol)
            if state is None:
                return
            if stop_price is not None and np.isfinite(stop_price):
                state["stop_price"] = float(stop_price)
            if limit_price is not None and np.isfinite(limit_price):
                state["limit_price"] = float(limit_price)
            if peak_price is not None and np.isfinite(peak_price):
                state["peak_price"] = float(peak_price)
            if mfe is not None and np.isfinite(mfe):
                state["mfe"] = float(mfe)
            if mae is not None and np.isfinite(mae):
                state["mae"] = float(mae)
            if stop_reason:
                state["stop_reason"] = str(stop_reason)
            if stop_reason_detail:
                state["stop_reason_detail"] = str(stop_reason_detail)
            if last_5m_eval_ts is not None:
                state["last_5m_eval_ts"] = pd.Timestamp(last_5m_eval_ts)
                state["last_update"] = pd.Timestamp.now(tz="UTC")

    def get_oco_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current OCO positions."""
        if self.oco_executor:
            return self.oco_executor.get_active_positions()
        return self.get_active_positions()

    def fetch_5m_ohlcv_for_positions(self) -> Dict[str, pd.DataFrame]:
        """Fetch 5m OHLCV for all active OCO positions.

        Uses hf_data_loader.fetch_ohlcv_5m() to get high-frequency
        data for trailing profit analysis.

        Returns:
            Dictionary mapping symbol to 5m OHLCV DataFrame
        """
        if self.oco_executor:
            return self.oco_executor.fetch_5m_ohlcv_for_positions()
        results: Dict[str, pd.DataFrame] = {}
        if self.exchange is None:
            with self._state_lock:
                items = list(self.positions.items())
            for symbol, state in items:
                if not isinstance(state, dict):
                    continue
                bars = state.get("ohlcv_5m_latest", state.get("ohlcv_5m"))
                if (
                    bars is not None
                    and isinstance(bars, (pd.DataFrame, pd.Series))
                    and not (hasattr(bars, "empty") and bars.empty)
                ):
                    results[symbol] = pd.DataFrame(bars)
            return results

        current_time = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            items = list(self.positions.items())
        for symbol, state in items:
            if not isinstance(state, dict):
                continue
            try:
                entry_time = pd.Timestamp(state.get("entry_time", current_time))
                bars = hf_data_loader.fetch_ohlcv_5m(
                    self.exchange,
                    symbol,
                    entry_time - pd.Timedelta(hours=1),
                    current_time + pd.Timedelta(hours=1),
                )
                if (
                    bars is not None
                    and isinstance(bars, (pd.DataFrame, pd.Series))
                    and not (hasattr(bars, "empty") and bars.empty)
                ):
                    frame = pd.DataFrame(bars)
                    results[symbol] = frame
                    with self._state_lock:
                        if symbol in self.positions:
                            self.positions[symbol]["ohlcv_5m"] = frame
            except Exception as exc:
                tprint(
                    f"Error fetching shadow 5m OHLCV for {symbol}: "
                    f"{_classify_exchange_error(exc)}: {exc}"
                )
        return results

    def shutdown(self):
        """Shutdown executor and cleanup resources."""
        if self.oco_executor:
            self.oco_executor.stop_monitoring()
            self.oco_executor.close_all_positions()
        tprint("TradeExecutor shutdown complete")


def execute_live_trade(
    executor: TradeExecutor,
    symbol: str,
    side: str,
    size: float,
    price: Optional[float] = None,
    bucket_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a live trade.

    Args:
        executor: TradeExecutor instance
        symbol: Trading symbol
        side: "long" or "short"
        size: Position size
        price: Limit price
        bucket_key: Bucket key for OCO parameters

    Returns:
        Execution result
    """
    if not _is_live_execution_mode(executor.mode):
        return {"success": False, "error": "Not in live mode"}

    return executor.execute_trade(symbol, side, size, price, bucket_key)


def record_shadow_trade(
    executor: TradeExecutor,
    symbol: str,
    side: str,
    size: float,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """Record a shadow trade.

    Args:
        executor: TradeExecutor instance
        symbol: Trading symbol
        side: "long" or "short"
        size: Position size
        price: Entry price

    Returns:
        Shadow record
    """
    if executor.mode != "shadow":
        return {"success": False, "error": "Not in shadow mode"}

    return executor._record_shadow_trade(symbol, side, size, price)
