import numpy as np
"""
Trade Executor for Inference.

This module handles trade execution in two modes:
- Live mode: Execute real trades using OrderManager from live_trading
- Shadow mode: Record decisions to CSV

Includes OCO (One Cancel the Other) order handling with dynamic updates:
- Place OCO orders: Stop Loss + Take Profit together
- Update OCO as price changes:
  - Giveback: Move stop loss up as price increases (lock in profits)
  - Trailing profit: Adjust stop loss based on trailing parameters
  - MFE (Max Favorably Excursed) early exit: Exit when price reaches target
- Monitor prices every 3 minutes during live trading
- Use bucket_params from Ridge position sizer for SL/TP parameters
"""

import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd

from extreme_price_movements.utils import tprint, log_error
from extreme_price_movements import hf_data_loader

# Try to import OrderManager from live_trading
try:
    from live_trading.order_manager import OrderManager
    ORDER_MANAGER_AVAILABLE = True
except Exception:
    ORDER_MANAGER_AVAILABLE = False
    tprint("Warning: OrderManager not available from live_trading")


class OCOExecutor:
    """
    OCO (One Cancel the Other) order executor with dynamic updates.
    Monitors positions and updates SL/TP based on price movement.
    
    Features:
    - Place OCO orders: Stop Loss + Take Profit together
    - Giveback: Move SL up as price increases (lock in profits)
    - Trailing profit: Adjust SL based on trailing parameters
    - MFE early exit: Exit when price reaches target
    - Monitor prices every 3 minutes during live trading
    """
    
    def __init__(
        self,
        exchange: Any,
        bucket_params: Dict[str, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
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
        self._monitor_interval = config.get("monitor_interval_seconds", 300)  # 5 minutes
        
        # Default parameters (fallback when bucket not found)
        self._default_params = {
            "sl_mult": 1.0,
            "tp_mult": 3.0,
            "trail_mult": 0.25,
            "giveback_pct": 0.005,
            "profit_lock_amount": 0.003,
            "mfe_early_exit_threshold": 0.02,
            "max_hold_hours": 48
        }
    
    def _get_bucket_params(self, bucket_key: str) -> Dict[str, Any]:
        """Get parameters for a specific bucket, with fallback to defaults."""
        raw_key = str(bucket_key or "")
        key_lower = raw_key.lower()
        key_upper = raw_key.upper()
        bucket = (
            self.bucket_params.get(raw_key, {})
            or self.bucket_params.get(key_lower, {})
            or self.bucket_params.get(key_upper, {})
        )
        params = self._default_params.copy()
        params.update({
            k: v
            for k, v in self.bucket_params.items()
            if not isinstance(v, dict)
        })
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
            params.update({k: v for k, v in self.bucket_params.items() if not isinstance(v, dict)})
            return params
        params.update(self._get_bucket_params(bucket_key))
        return params

    def get_cooldown_hours(self, bucket_key: Optional[str] = None) -> float:
        """Return cooldown hours for a bucket or global fallback."""
        params = self.get_bucket_params(bucket_key)
        return float(params.get("cooldown_hours", self.config.get("cooldown_hours", 0.0)) or 0.0)
    
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
                    prev_close = closes[i-1]
                    tr = max(
                        high - low,
                        abs(high - prev_close),
                        abs(low - prev_close)
                    )
                    trs.append(tr)
                
                atr = sum(trs) / len(trs) if trs else 0.01
                current_price = closes[-1]
                return atr / current_price  # Return as fraction
        except Exception as e:
            log_error(f"Could not fetch ATR for {symbol}: {e}", exc=e)
        
        return 0.01  # Default 1% ATR
    
    def _fetch_aggtrades(self, symbol: str, since: int = None, limit: int = None) -> List[Dict[str, Any]]:
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
            if hasattr(self.exchange, 'fetch_aggregated_trades'):
                return self.exchange.fetch_aggregated_trades(symbol, since=since, limit=limit)
            elif hasattr(self.exchange, 'fetch_agg_trades'):
                return self.exchange.fetch_agg_trades(symbol, since=since, limit=limit)
            elif hasattr(self.exchange, 'fetch_trades'):
                # Fallback to regular trades
                return self.exchange.fetch_trades(symbol, since=since, limit=limit)
            else:
                tprint(f"Exchange does not support fetching trades for {symbol}")
                return []
        except Exception as e:
            log_error(f"Error fetching aggtrades for {symbol}: {e}", exc=e)
            return []
    
    def _get_aggtrades_at_entry(self, symbol: str, entry_time: pd.Timestamp) -> List[Dict[str, Any]]:
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
            entry_time = entry_time.tz_convert('UTC')
        
        # Get trades for 1 minute around entry time
        start_ms = int(entry_time.timestamp() * 1000) - 60000  # 1 minute before
        end_ms = int(entry_time.timestamp() * 1000) + 60000   # 1 minute after
        
        return self._fetch_aggtrades(symbol, since=start_ms, limit=1000)
    
    def place_oco_order(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        bucket_key: str
    ) -> Dict[str, Any]:
        """Place OCO orders: Stop Loss + Take Profit.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "long" or "short"
            entry_price: Entry price for the position
            size: Position size
            bucket_key: Bucket key for getting SL/TP parameters
            
        Returns:
            Dictionary with order details
        """
        params = self._get_bucket_params(bucket_key)
        atr = self._get_current_atr(symbol)
        
        # Get parameters
        sl_mult = params["sl_mult"]
        tp_mult = params["tp_mult"]
        
        # Calculate prices
        if side == "long":
            stop_price = entry_price * (1 - sl_mult * atr)
            limit_price = entry_price * (1 + tp_mult * atr)
        else:  # short
            stop_price = entry_price * (1 + sl_mult * atr)
            limit_price = entry_price * (1 - tp_mult * atr)
        
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
            "tp_mult": tp_mult,
            "peak_price": entry_price,
            "mfe": 0.0,
            "entry_time": pd.Timestamp.now(tz="UTC"),
            "last_update": pd.Timestamp.now(tz="UTC"),
            "oco_order_id": None,
            "stop_order_id": None,
            "take_profit_order_id": None
        }
        
        # Try to place OCO via exchange API
        try:
            # Try Binance-style OCO first
            if hasattr(self.exchange, 'create_oco_order'):
                oco_result = self.exchange.create_oco_order(
                    symbol=symbol,
                    side="sell" if side == "long" else "buy",
                    quantity=size,
                    price=limit_price,
                    stopPrice=stop_price,
                    stopLimitPrice=stop_price,
                    stopLimitTimeInForce="GTC"
                )
                position_state["oco_order_id"] = oco_result.get("orderListId") or oco_result.get("listOrderSynomyms")
            else:
                # Fallback: place stop loss and take profit separately
                # First place the stop loss
                stop_order = self.exchange.create_order(
                    symbol=symbol,
                    type="stop_loss_limit" if hasattr(self.exchange, 'create_order') else "stop",
                    side="sell" if side == "long" else "buy",
                    amount=size,
                    price=stop_price,
                    stopPrice=stop_price
                )
                position_state["stop_order_id"] = stop_order.get("id")
                
                # Then place take profit limit
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side="sell" if side == "long" else "buy",
                    amount=size,
                    price=limit_price
                )
                position_state["take_profit_order_id"] = tp_order.get("id")
                
        except Exception as e:
            log_error(f"Error placing OCO orders for {symbol}: {e}", exc=e)
            # Continue with tracking even if order placement fails
            # This allows for manual intervention or retry
        
        # Store position
        with self._positions_lock:
            self.active_positions[symbol] = position_state
        
        # Fetch aggtrades and 5m OHLCV for analysis (OCOExecutor is only used in live mode)
        aggtrades_data = None
        ohlcv_5m_data = None
        
        # Only fetch data if order was placed successfully
        if position_state.get("oco_order_id") or position_state.get("stop_order_id"):
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
                        entry_time + pd.Timedelta(hours=12)  # 12 hours after
                    )
                    position_state["ohlcv_5m"] = ohlcv_5m_data
                except Exception as e:
                    log_error(f"Error fetching 5m OHLCV for {symbol}: {e}", exc=e)
                    
            except Exception as e:
                log_error(f"Error fetching aggtrades for {symbol}: {e}", exc=e)
        
        tprint(f"Placed OCO for {symbol}: SL={stop_price:.4f}, TP={limit_price:.4f}")
        
        return {
            "success": True,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "limit_price": limit_price,
            "size": size,
            "bucket_key": bucket_key,
            "aggtrades": aggtrades_data,
            "ohlcv_5m": ohlcv_5m_data
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
                log_error(f"Error in OCO monitoring loop: {e}", exc=e)
            
            # Sleep in small increments to allow quick shutdown
            for _ in range(int(self._monitor_interval)):
                if not self._monitoring:
                    break
                time.sleep(1)
    
    def monitor_positions(self):
        """Monitor all active positions every 3 minutes.
        
        Checks for:
        - Giveback: Move SL up as price increases
        - Trailing profit: Adjust based on trailing params
        - MFE exit: Exit if price reaches target
        """
        current_time = pd.Timestamp.now(tz="UTC")
        
        with self._positions_lock:
            items = list(self.active_positions.items())
        for symbol, state in items:
            # Skip if not enough time passed (3 min)
            time_since_update = current_time - state["last_update"]
            if time_since_update < pd.Timedelta(minutes=5):
                continue
            
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])
                
                # Update position based on price movement
                self._update_oco(symbol, state, current_price)
                
            except Exception as e:
                log_error(f"Error monitoring {symbol}: {e}", exc=e)
                continue
    
    def _update_oco(
        self,
        symbol: str,
        state: Dict[str, Any],
        current_price: float
    ):
        """Update OCO based on price movement.
        
        1. Giveback: Move SL up as price increases
        2. Trailing profit: Adjust based on trailing params
        3. MFE exit: Exit if price reaches target
        
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
        
        # Check MFE early exit
        mfe_threshold = params.get("mfe_early_exit_threshold", 0.02)
        if state["mfe"] >= mfe_threshold:
            # Close position early - MFE target reached
            self._close_position(symbol, state, current_price, "mfe_exit")
            return
        
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
        self,
        symbol: str,
        state: Dict[str, Any],
        new_stop_price: float
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
                    self.exchange.cancel_order(state["stop_order_id"], symbol)
                except Exception:
                    pass  # Order may have already filled
            
            # Place new stop limit order
            new_stop_order = self.exchange.create_order(
                symbol=symbol,
                type="stop_loss_limit" if hasattr(self.exchange, 'create_order') else "stop",
                side="sell" if state["side"] == "long" else "buy",
                amount=state["size"],
                price=new_stop_price,
                stopPrice=new_stop_price
            )
            
            state["stop_price"] = new_stop_price
            state["stop_order_id"] = new_stop_order.get("id")
            tprint(f"Updated SL for {symbol} to {new_stop_price:.4f}")
            
        except Exception as e:
            log_error(f"Error updating SL for {symbol}: {e}", exc=e)
    
    def _close_position(
        self,
        symbol: str,
        state: Dict[str, Any],
        current_price: float,
        reason: str
    ):
        """Close position and remove from tracking.
        
        Args:
            symbol: Trading symbol
            state: Position state dictionary
            current_price: Current market price
            reason: Reason for closing (e.g., "mfe_exit", "stop_loss", "take_profit")
        """
        try:
            # Cancel any existing orders
            if state.get("stop_order_id"):
                try:
                    self.exchange.cancel_order(state["stop_order_id"], symbol)
                except Exception:
                    pass
            
            if state.get("take_profit_order_id"):
                try:
                    self.exchange.cancel_order(state["take_profit_order_id"], symbol)
                except Exception:
                    pass
            
            # Market close
            close_order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="sell" if state["side"] == "long" else "buy",
                amount=state["size"]
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
            log_error(f"Error closing {symbol}: {e}", exc=e)
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
                    current_time + pd.Timedelta(hours=1)
                )
                
                # Safely check ohlcv_5m
                try:
                    ohlcv_not_empty = ohlcv_5m is not None and isinstance(ohlcv_5m, (pd.DataFrame, pd.Series)) and not (hasattr(ohlcv_5m, 'empty') and ohlcv_5m.empty)
                except Exception:
                    ohlcv_not_empty = False
                
                if ohlcv_not_empty:
                    results[symbol] = ohlcv_5m
                    # Also store in position state
                    state["ohlcv_5m"] = ohlcv_5m
                else:
                    tprint(f"Warning: No 5m OHLCV data fetched for {symbol}")
                    
            except Exception as e:
                log_error(f"Error fetching 5m OHLCV for {symbol}: {e}", exc=e)
        
        return results
    
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
                    self._close_position(symbol, state, current_price, "emergency_close")
            except Exception as e:
                log_error(f"Error emergency closing {symbol}: {e}", exc=e)


class TradeExecutor:
    """Handles trade execution in live or shadow mode with OCO support."""
    
    def __init__(
        self,
        mode: str = "shadow",
        exchange: Optional[Any] = None,
        capital: float = 10000.0,
        max_position_size: float = 0.1,
        bucket_params: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
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
                exchange=exchange,
                bucket_params=self.bucket_params,
                config=self.config
            )
            self.oco_executor.start_monitoring()
        
        # Initialize OrderManager for live mode (legacy support)
        if mode == "live" and ORDER_MANAGER_AVAILABLE and exchange is not None:
            self.order_manager = OrderManager(exchange=exchange)
        else:
            self.order_manager = None
        
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
        params.update({k: v for k, v in self.bucket_params.items() if not isinstance(v, dict)})
        if bucket_key is None:
            return params
        raw_key = str(bucket_key or "")
        key_lower = raw_key.lower()
        key_upper = raw_key.upper()
        bucket = (
            self.bucket_params.get(raw_key, {})
            or self.bucket_params.get(key_lower, {})
            or self.bucket_params.get(key_upper, {})
        )
        if not bucket and "buckets" in self.bucket_params and isinstance(self.bucket_params["buckets"], dict):
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
        return float(params.get("cooldown_hours", self.config.get("cooldown_hours", 0.0)) or 0.0)
    
    def execute_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None
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
        # Calculate actual size
        position_value = size * self.capital * self.max_position_size

        cooldown_hours = self.get_cooldown_hours(bucket_key)
        now_utc = pd.Timestamp.now(tz="UTC")
        with self._state_lock:
            active_oco = bool(self.oco_executor and symbol in self.oco_executor.get_active_positions())
            if symbol in self.positions or active_oco:
                return {"success": False, "error": f"symbol {symbol} already active"}
            last_ts = self._last_trade_timestamps.get(symbol)
            if (
                last_ts is not None
                and cooldown_hours > 0.0
                and now_utc < (pd.Timestamp(last_ts) + pd.Timedelta(hours=float(cooldown_hours)))
            ):
                return {"success": False, "error": f"symbol {symbol} in cooldown for {cooldown_hours:.1f}h"}
        
        if self.mode == "live":
            return self._execute_live(
                symbol, side, position_value, price, bucket_key
            )
        else:
            return self._record_shadow_trade(symbol, side, position_value, price, bucket_key=bucket_key)
    
    def _execute_live(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        bucket_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute live trade with OCO support.
        
        Args:
            symbol: Trading symbol
            side: "long" or "short"
            size: Position size in quote currency
            price: Limit price
            bucket_key: Bucket key for OCO parameters
            
        Returns:
            Execution result dictionary
        """
        # Determine order side
        order_side = "buy" if side == "long" else "sell"
        
        # Place entry order first
        entry_price = price
        
        try:
            if price is not None:
                # Limit order for entry
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=order_side,
                    amount=size,
                    price=price
                )
                entry_price = float(price)
            else:
                # Market order for entry
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order_side,
                    amount=size
                )
                # Get fill price
                ticker = self.exchange.fetch_ticker(symbol)
                entry_price = float(ticker["last"])
            
            # If bucket_key provided, place OCO orders
            oco_result = None
            if bucket_key and self.oco_executor:
                oco_result = self.oco_executor.place_oco_order(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    size=size,
                    bucket_key=bucket_key
                )
            
            # Update positions
            with self._state_lock:
                self.positions[symbol] = {
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "timestamp": datetime.now(),
                    "bucket_key": bucket_key
                }
                self._last_trade_timestamps[symbol] = pd.Timestamp.now(tz="UTC")
            
            return {
                "success": True,
                "order": order,
                "oco_result": oco_result,
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": entry_price,
            }
            
        except Exception as e:
            log_error(f"Error executing live trade: {e}", exc=e)
            return {"success": False, "error": str(e)}
    
    def _record_shadow_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None
        ,
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
        entry_price = float(price) if price is not None else 0.0
        atr_frac = float(params.get("atr", 0.01) or 0.01)
        sl_mult = float(params.get("sl_mult", 1.0))
        tp_mult = float(params.get("tp_mult", 3.0))
        if side == "long":
            stop_price = entry_price * (1.0 - sl_mult * atr_frac) if entry_price > 0 else None
            limit_price = entry_price * (1.0 + tp_mult * atr_frac) if entry_price > 0 else None
        else:
            stop_price = entry_price * (1.0 + sl_mult * atr_frac) if entry_price > 0 else None
            limit_price = entry_price * (1.0 - tp_mult * atr_frac) if entry_price > 0 else None
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": "shadow",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
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
        price: Optional[float] = None
        ,
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
            except Exception as e:
                log_error(f"Error canceling OCO for {symbol}: {e}", exc=e)
        
        try:
            order_side = "sell" if side == "long" else "buy"
            
            if price is not None:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=order_side,
                    amount=size,
                    price=price
                )
            else:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=order_side,
                    amount=size
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
            return {"success": False, "error": str(e)}
    
    def _record_shadow_close(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None
        ,
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
                        (state.get("side") == "long" and float(stop_price) > current_stop)
                        or (state.get("side") == "short" and float(stop_price) < current_stop)
                    )
                    if improved:
                        self.oco_executor._update_stop_loss(symbol, state, float(stop_price))
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
        return {}
    
    def fetch_5m_ohlcv_for_positions(self) -> Dict[str, pd.DataFrame]:
        """Fetch 5m OHLCV for all active OCO positions.
        
        Uses hf_data_loader.fetch_ohlcv_5m() to get high-frequency
        data for trailing profit analysis.
        
        Returns:
            Dictionary mapping symbol to 5m OHLCV DataFrame
        """
        if self.oco_executor:
            return self.oco_executor.fetch_5m_ohlcv_for_positions()
        return {}
    
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
    bucket_key: Optional[str] = None
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
    price: Optional[float] = None
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
