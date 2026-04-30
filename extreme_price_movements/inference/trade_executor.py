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


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Convert exchange payload values to float without raising."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _classify_exchange_error(exc: Exception) -> str:
    """Return a stable exchange error category for logs and risk gates."""
    message = str(exc).lower()
    name = exc.__class__.__name__.lower()
    text = f"{name} {message}"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "authentication" in text or "unauthorized" in text or "permission" in text:
        return "auth_or_permission"
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
        params["reduceOnly"] = True
    else:
        params["sideEffectType"] = _margin_side_effect(config)
    return params


def _cancel_params(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build ccxt params for spot or margin order cancellation."""
    if _execution_account(config) != "margin":
        return {}
    return {"marginMode": _margin_mode(config)}


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
        self._monitor_interval = config.get("monitor_interval_seconds", 300)

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
        """Return cooldown hours for a bucket or global fallback."""
        params = self.get_bucket_params(bucket_key)
        return float(
            params.get("cooldown_hours", self.config.get("cooldown_hours", 0.0)) or 0.0
        )

    def _get_current_atr(self, symbol: str) -> float:
        """Get current ATR for the symbol.

        In production, this would fetch from exchange or calculate from recent data.
        Returns ATR as a fraction of price (e.g., 0.01 = 1%).
        """
        try:
            # Try to fetch recent ohlcv to calculate ATR
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=14)
            if ohlcv and len(ohlcv) >= 14:
                highs = [x[2] for x in ohlcv]
                lows = [x[3] for x in ohlcv]
                closes = [x[4] for x in ohlcv]

                # Calculate True Range
                trs = []
                for i in range(1, len(ohlcv)):
                    high = highs[i]
                    low = lows[i]
                    prev_close = closes[i - 1]
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    trs.append(tr)

                atr = sum(trs) / len(trs) if trs else 0.01
                current_price = closes[-1]
                return atr / current_price  # Return as fraction
        except Exception as e:
            tprint(
                f"Could not fetch ATR for {symbol}: "
                f"{_classify_exchange_error(e)}: {e}"
            )

        return 0.01  # Default 1% ATR

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
        self, symbol: str, side: str, entry_price: float, size: float, bucket_key: str
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
        atr = self._get_current_atr(symbol)

        sl_mult = params["sl_mult"]

        if side == "long":
            stop_price = entry_price * (1 - sl_mult * atr)
        else:  # short
            stop_price = entry_price * (1 + sl_mult * atr)
        limit_price = None

        # Track position state
        position_state = {
            "side": side,
            "entry_price": entry_price,
            "size": size,
            "bucket_key": bucket_key,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "atr": atr,
            "sl_mult": sl_mult,
            "peak_price": entry_price,
            "mfe": 0.0,
            "entry_time": pd.Timestamp.now(tz="UTC"),
            "last_update": pd.Timestamp.now(tz="UTC"),
            "oco_order_id": None,
            "stop_order_id": None,
            "take_profit_order_id": None,
        }

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
            position_state["stop_order_error"] = stop_order_error
            position_state["stop_order_error_category"] = stop_order_error_category
            tprint(
                f"Error placing STOP_LOSS for {symbol}: "
                f"{stop_order_error_category}: {e}"
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

        tprint(f"Placed STOP_LOSS for {symbol}: SL={stop_price:.4f}")

        return {
            "success": position_state.get("stop_order_id") is not None,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "size": amount,
            "bucket_key": bucket_key,
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
        try:
            # Cancel existing stop order if exists
            if state.get("stop_order_id"):
                try:
                    self.exchange.cancel_order(
                        state["stop_order_id"], symbol, _cancel_params(self.config)
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

            market = _load_market(self.exchange, symbol)
            amount = _exchange_precision(
                self.exchange, symbol, float(state["size"]), kind="amount"
            )
            stop_price = _exchange_precision(
                self.exchange, symbol, float(new_stop_price), kind="price"
            )
            _validate_order_filters(
                symbol,
                market,
                amount=amount,
                price=max(float(state.get("entry_price", stop_price)), stop_price),
            )

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

            state["stop_price"] = stop_price
            state["stop_order_id"] = new_stop_order.get("id")
            state["size"] = amount
            tprint(f"Updated SL for {symbol} to {stop_price:.4f}")

        except Exception as e:
            category = _classify_exchange_error(e)
            state["stop_update_error"] = str(e)
            state["stop_update_error_category"] = category
            tprint(f"Error updating SL for {symbol}: {category}: {e}")

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

            tprint(f"Closed {symbol} at {current_price:.4f}, reason: {reason}")

            # Log trade result
            pnl = 0
            if state["side"] == "long":
                pnl = (current_price - state["entry_price"]) * state["size"]
            else:
                pnl = (state["entry_price"] - current_price) * state["size"]

            tprint(f"  PnL: {pnl:.2f}, MFE: {state['mfe']*100:.2f}%")

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
                    with self._positions_lock:
                        self.active_positions.pop(symbol, None)
                elif status in {"canceled", "cancelled", "expired", "rejected"}:
                    state["stop_order_error"] = f"stop_order_{status}"
            except Exception as exc:
                category = _classify_exchange_error(exc)
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
        if mode == "live" and exchange is not None:
            self.oco_executor = OCOExecutor(
                exchange=exchange, bucket_params=self.bucket_params, config=self.config
            )
            self.oco_executor.start_monitoring()

        self.order_manager = _make_legacy_order_manager(
            exchange,
            bool(
                mode == "live"
                and exchange is not None
                and self.config.get("enable_legacy_order_manager", False)
            ),
        )

        tprint(f"TradeExecutor initialized in {mode} mode")

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
        """Return cooldown hours for a bucket or global fallback."""
        params = self.get_bucket_params(bucket_key)
        return float(
            params.get("cooldown_hours", self.config.get("cooldown_hours", 0.0)) or 0.0
        )

    def execute_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
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

        if self.mode == "live":
            return self._execute_live(symbol, side, position_value, price, bucket_key)
        else:
            return self._record_shadow_trade(
                symbol, side, position_value, price, bucket_key=bucket_key
            )

    def _execute_live(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None,
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
            if not np.isfinite(expected_entry_price):
                ticker = self.exchange.fetch_ticker(symbol)
                expected_entry_price = _safe_float(ticker.get("last"), default=np.nan)
            amount_base = self._quote_to_base_amount(
                symbol,
                quote_size=float(size),
                reference_price=expected_entry_price,
                market=market,
            )

            if price is not None:
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
            else:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order_side,
                    amount=amount_base,
                    params=_order_params(self.config, reduce_only=False),
                )
            fallback_price = float(expected_entry_price)
            entry_price, filled_amount, partial_fill = _extract_order_fill(
                order, fallback_price
            )
            stop_amount = (
                _exchange_precision(self.exchange, symbol, filled_amount, kind="amount")
                if np.isfinite(filled_amount) and filled_amount > 0
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
                )

            with self._state_lock:
                self.positions[symbol] = {
                    "side": side,
                    "size": stop_amount,
                    "quote_size": float(size),
                    "requested_base_amount": amount_base,
                    "entry_price": entry_price,
                    "timestamp": datetime.now(),
                    "bucket_key": bucket_key,
                    "partial_fill": partial_fill,
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
                    "expected_entry_price": expected_entry_price,
                    "realized_entry_price": entry_price,
                    "partial_fill": partial_fill,
                }

            return {
                "success": True,
                "order": order,
                "oco_result": oco_result,
                "symbol": symbol,
                "side": side,
                "size": float(size),
                "base_amount": stop_amount,
                "price": entry_price,
                "expected_entry_price": expected_entry_price,
                "realized_entry_price": entry_price,
                "partial_fill": partial_fill,
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
        atr_frac = float(params.get("atr", 0.01) or 0.01)
        sl_mult = float(params.get("sl_mult", 1.0))
        if side == "long":
            stop_price = (
                entry_price * (1.0 - sl_mult * atr_frac) if entry_price > 0 else None
            )
        else:
            stop_price = (
                entry_price * (1.0 + sl_mult * atr_frac) if entry_price > 0 else None
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
            "price_slippage_pct": 0.0,
            "spread_proxy_pct": 0.0,
            "status": "recorded",
            "bucket_key": bucket_key,
        }

        # Update positions
        with self._state_lock:
            self.positions[symbol] = {
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "timestamp": datetime.now(),
                "entry_time": pd.Timestamp.now(tz="UTC"),
                "bucket_key": bucket_key,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "peak_price": entry_price,
                "mfe": 0.0,
                "last_update": pd.Timestamp.now(tz="UTC"),
                "last_5m_eval_ts": None,
            }
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

        if self.mode == "live":
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
        if self.oco_executor and symbol in self.oco_executor.active_positions:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
                state = self.oco_executor.active_positions[symbol]
                self.oco_executor._close_position(symbol, state, current_price, reason)
                with self._state_lock:
                    self.positions.pop(symbol, None)
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": "closed",
                    "size": size,
                    "reason": reason,
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

    def monitor_orders_once(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live order statuses for active stop-loss orders once."""
        if self.oco_executor is not None:
            return self.oco_executor.monitor_order_statuses()
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
        last_5m_eval_ts: Optional[pd.Timestamp] = None,
    ) -> None:
        """Persist live/shadow threshold updates from the 5m monitor."""
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
                        self.oco_executor._update_stop_loss(
                            symbol, state, float(stop_price)
                        )
                if limit_price is not None and np.isfinite(limit_price):
                    state["limit_price"] = float(limit_price)
                if peak_price is not None and np.isfinite(peak_price):
                    state["peak_price"] = float(peak_price)
                if mfe is not None and np.isfinite(mfe):
                    state["mfe"] = float(mfe)
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
    if executor.mode != "live":
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
