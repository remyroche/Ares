"""Order Manager V2 for Live Trading.

Integrates with PortfolioManager for position constraints and provides:
- OCO order placement (entry + SL + TP)
- Position monitoring and updates
- Wallet monitoring for dynamic sizing
- Order status tracking
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum

import pandas as pd
import numpy as np

try:
    from extreme_price_movements.portfolio_manager import PortfolioManager
    from extreme_price_movements.utils import tprint
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extreme_price_movements.portfolio_manager import PortfolioManager
    from extreme_price_movements.utils import tprint


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(Enum):
    """Position status enumeration."""
    PENDING_ENTRY = "pending_entry"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    EMERGENCY_CLOSE = "emergency_close"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "limit", "market", "stop_loss", etc.
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    exchange_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a trading position."""
    position_id: str
    symbol: str
    side: str  # "long" or "short"
    strategy_id: str
    entry_price: float
    size: float  # In quote currency (USDT)
    quantity: float  # In base asset units
    status: PositionStatus = PositionStatus.PENDING_ENTRY
    entry_order: Optional[Order] = None
    stop_loss_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = field(default_factory=lambda: datetime.utcnow())
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagerV2:
    """Advanced order manager with PortfolioManager integration.
    
    Features:
    - Wallet monitoring for dynamic position sizing
    - OCO order placement (entry + SL + TP)
    - Position tracking with PortfolioManager sync
    - Real-time order status monitoring
    - Automatic SL updates (giveback, trailing)
    """
    
    def __init__(
        self,
        api_client: Any,
        portfolio_manager: PortfolioManager,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.api_client = api_client
        self.portfolio_mgr = portfolio_manager
        self.config = config or {}
        
        # Default configuration
        self.default_sl_mult = self.config.get("sl_mult", 1.0)
        self.default_tp_mult = self.config.get("tp_mult", 3.0)
        self.default_trail_mult = self.config.get("trail_mult", 0.25)
        self.monitor_interval = self.config.get("monitor_interval_seconds", 60)
        self.fee_rate = self.config.get("fee_rate", 0.0003)  # 0.03% per trade
        
        # State tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []
        
        # Async/threading
        self._lock = threading.RLock()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_position_open: List[Callable[[Position], None]] = []
        self._on_position_close: List[Callable[[Position], None]] = []
        self._on_order_update: List[Callable[[Order], None]] = []
    
    async def start(self) -> None:
        """Start the order manager monitoring loop."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        tprint("[OrderManagerV2] Started monitoring loop")
    
    async def stop(self) -> None:
        """Stop the order manager monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        tprint("[OrderManagerV2] Stopped monitoring loop")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop for positions and orders."""
        while self._running:
            try:
                await self._sync_positions_with_portfolio()
                await self._update_order_statuses()
                await self._monitor_active_positions()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                tprint(f"[OrderManagerV2] Monitor loop error: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _sync_positions_with_portfolio(self) -> None:
        """Sync position state with PortfolioManager."""
        portfolio_state = self.portfolio_mgr.get_portfolio_state()
        
        # Ensure PortfolioManager knows about our positions
        with self._lock:
            for symbol, position in self.positions.items():
                if position.status == PositionStatus.OPEN:
                    # Check if PortfolioManager has this position
                    if symbol not in portfolio_state.get("open_symbols", []):
                        # Position exists locally but not in PM - re-register
                        self.portfolio_mgr.record_position_open(
                            symbol=symbol,
                            side=position.side,
                            strategy_id=position.strategy_id,
                            position_size=position.size,
                            entry_price=position.entry_price,
                            entry_time=pd.Timestamp(position.created_at, tz="UTC"),
                        )
    
    async def _update_order_statuses(self) -> None:
        """Update status of all pending/open orders."""
        with self._lock:
            orders_to_check = [
                order for order in self.orders.values()
                if order.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
            ]
        
        for order in orders_to_check:
            try:
                response = await self.api_client.get_order_status(
                    symbol=order.symbol.replace("/", ""),
                    order_id=order.order_id
                )
                
                if response.success:
                    self._update_order_from_exchange(order, response.data)
                    
            except Exception as e:
                tprint(f"[OrderManagerV2] Error checking order {order.order_id}: {e}")
    
    def _update_order_from_exchange(self, order: Order, exchange_data: Dict[str, Any]) -> None:
        """Update local order from exchange data."""
        # Map exchange status to our status
        exchange_status = exchange_data.get("status", "").lower()
        
        status_map = {
            "new": OrderStatus.OPEN,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        
        old_status = order.status
        order.status = status_map.get(exchange_status, OrderStatus.OPEN)
        order.filled_quantity = float(exchange_data.get("executedQty", 0))
        order.remaining_quantity = order.quantity - order.filled_quantity
        order.exchange_data = exchange_data
        
        if "avgPrice" in exchange_data:
            order.average_price = float(exchange_data["avgPrice"])
        
        # Notify if status changed
        if old_status != order.status:
            tprint(f"[OrderManagerV2] Order {order.order_id} status: {old_status.value} -> {order.status.value}")
            for callback in self._on_order_update:
                callback(order)
        
        # Update associated position
        self._update_position_from_order(order)
    
    def _update_position_from_order(self, order: Order) -> None:
        """Update position state based on order updates."""
        with self._lock:
            # Find position associated with this order
            for symbol, position in self.positions.items():
                if (position.entry_order and position.entry_order.order_id == order.order_id) or \
                   (position.stop_loss_order and position.stop_loss_order.order_id == order.order_id) or \
                   (position.take_profit_order and position.take_profit_order.order_id == order.order_id):
                    
                    # Update position state
                    if order.status == OrderStatus.FILLED:
                        if position.entry_order and position.entry_order.order_id == order.order_id:
                            # Entry filled
                            position.status = PositionStatus.OPEN
                            position.entry_price = order.average_price or order.price or position.entry_price
                            
                            # Notify position open
                            for callback in self._on_position_open:
                                callback(position)
                            
                            tprint(f"[OrderManagerV2] Position opened: {symbol} at {position.entry_price}")
                            
                        elif (position.stop_loss_order and position.stop_loss_order.order_id == order.order_id) or \
                             (position.take_profit_order and position.take_profit_order.order_id == order.order_id):
                            # SL or TP filled - position closed
                            self._close_position(symbol, order.average_price or order.price, "sl_tp")
                    
                    position.updated_at = datetime.utcnow()
                    break
    
    async def _monitor_active_positions(self) -> None:
        """Monitor active positions for SL updates (giveback, trailing)."""
        with self._lock:
            active_positions = [
                (symbol, pos) for symbol, pos in self.positions.items()
                if pos.status == PositionStatus.OPEN
            ]
        
        for symbol, position in active_positions:
            try:
                # Get current price
                ticker_response = await self.api_client.get_ticker(symbol.replace("/", ""))
                if not ticker_response.success:
                    continue
                
                current_price = float(ticker_response.data.get("lastPrice", 0))
                if current_price <= 0:
                    continue
                
                # Check if position should be updated
                await self._check_position_updates(symbol, position, current_price)
                
            except Exception as e:
                tprint(f"[OrderManagerV2] Error monitoring {symbol}: {e}")
    
    async def _check_position_updates(
        self, 
        symbol: str, 
        position: Position, 
        current_price: float
    ) -> None:
        """Check and apply position updates (giveback, trailing)."""
        # Get position metadata for parameters
        metadata = position.metadata
        peak_price = metadata.get("peak_price", position.entry_price)
        
        # Update peak price
        if position.side == "long":
            if current_price > peak_price:
                peak_price = current_price
        else:
            if current_price < peak_price:
                peak_price = current_price
        
        metadata["peak_price"] = peak_price
        
        # Calculate unrealized PnL
        if position.side == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Check for giveback/trailing update
        params = metadata.get("params", {})
        giveback_pct = params.get("giveback_pct", 0.005)
        trail_mult = params.get("trail_mult", 0.25)
        
        # Calculate new stop price
        if position.side == "long":
            # Giveback stop
            new_stop = peak_price * (1 - giveback_pct)
            
            # Trailing stop (if in profit)
            if peak_price > position.entry_price:
                trail_stop = position.entry_price + trail_mult * (peak_price - position.entry_price)
                new_stop = max(new_stop, trail_stop)
            
            # Don't move stop below entry if profitable
            if pnl_pct > 0:
                new_stop = max(new_stop, position.entry_price * 1.001)
        else:  # short
            new_stop = peak_price * (1 + giveback_pct)
            
            if peak_price < position.entry_price:
                trail_stop = position.entry_price - trail_mult * (position.entry_price - peak_price)
                new_stop = min(new_stop, trail_stop)
            
            if pnl_pct > 0:
                new_stop = min(new_stop, position.entry_price * 0.999)
        
        # Check if we should update SL
        current_sl = position.stop_loss_order.stop_price if position.stop_loss_order else None
        
        should_update = False
        if position.side == "long":
            should_update = current_sl is None or new_stop > current_sl
        else:
            should_update = current_sl is None or new_stop < current_sl
        
        if should_update and position.stop_loss_order:
            await self._update_stop_loss(symbol, position, new_stop)
    
    async def _update_stop_loss(
        self, 
        symbol: str, 
        position: Position, 
        new_stop: float
    ) -> None:
        """Update stop loss order for a position."""
        try:
            # Cancel existing stop order
            if position.stop_loss_order:
                await self.api_client.cancel_order(
                    symbol=symbol.replace("/", ""),
                    order_id=position.stop_loss_order.order_id
                )
            
            # Place new stop order
            side = "sell" if position.side == "long" else "buy"
            response = await self.api_client.create_order(
                symbol=symbol.replace("/", ""),
                side=side,
                order_type="STOP_LOSS_LIMIT",
                quantity=position.quantity,
                price=new_stop,
                stopPrice=new_stop,
                timeInForce="GTC"
            )
            
            if response.success:
                # Update position
                new_order = Order(
                    order_id=response.data.get("orderId", str(time.time())),
                    symbol=symbol,
                    side=side,
                    order_type="STOP_LOSS_LIMIT",
                    quantity=position.quantity,
                    price=new_stop,
                    stop_price=new_stop,
                    status=OrderStatus.OPEN,
                    exchange_data=response.data
                )
                
                position.stop_loss_order = new_order
                position.updated_at = datetime.utcnow()
                
                with self._lock:
                    self.orders[new_order.order_id] = new_order
                
                tprint(f"[OrderManagerV2] Updated SL for {symbol} to {new_stop:.4f}")
                
        except Exception as e:
            tprint(f"[OrderManagerV2] Error updating SL for {symbol}: {e}")
    
    async def place_oco_order(
        self,
        symbol: str,
        side: str,  # "long" or "short"
        strategy_id: str,
        entry_price: float,
        confidence_score: float,
        initial_threshold: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Place OCO order with PortfolioManager integration.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: "long" or "short"
            strategy_id: Strategy identifier
            entry_price: Entry price
            confidence_score: Model confidence score
            initial_threshold: Initial entry threshold
            params: Optional SL/TP parameters
            
        Returns:
            Dict with order result
        """
        params = params or {}
        
        # Check PortfolioManager constraints
        can_enter, pm_info = self.portfolio_mgr.can_enter_position(
            symbol=symbol,
            side=side,
            strategy_id=strategy_id,
            confidence_score=confidence_score,
            initial_threshold=initial_threshold,
            current_time=pd.Timestamp.now(tz="UTC"),
        )
        
        if not can_enter:
            return {
                "success": False,
                "error": f"Portfolio constraint: {pm_info.get('reason', 'unknown')}",
                "portfolio_info": pm_info
            }
        
        # Get position size from PortfolioManager
        position_size_usdt = pm_info.get("position_size_cap", 1000.0)
        
        # Fetch wallet balance for sizing
        wallet_balance = await self._get_wallet_balance()
        available_balance = wallet_balance.get("USDT", 0.0)
        
        # Ensure we have enough balance
        if position_size_usdt > available_balance * 0.95:  # 5% buffer
            position_size_usdt = available_balance * 0.95
        
        if position_size_usdt < 10.0:  # Minimum order size
            return {
                "success": False,
                "error": f"Insufficient balance: {available_balance:.2f} USDT",
                "portfolio_info": pm_info
            }
        
        # Calculate quantity
        quantity = position_size_usdt / entry_price
        
        # Calculate SL/TP prices
        sl_mult = params.get("sl_mult", self.default_sl_mult)
        tp_mult = params.get("tp_mult", self.default_tp_mult)
        
        # Estimate ATR (can be improved with real-time calculation)
        atr_estimate = params.get("atr", 0.01)  # Default 1%
        
        if side == "long":
            stop_price = entry_price * (1 - sl_mult * atr_estimate)
            limit_price = entry_price * (1 + tp_mult * atr_estimate)
        else:
            stop_price = entry_price * (1 + sl_mult * atr_estimate)
            limit_price = entry_price * (1 - tp_mult * atr_estimate)
        
        # Place entry order
        entry_side = "buy" if side == "long" else "sell"
        
        try:
            entry_response = await self.api_client.create_order(
                symbol=symbol.replace("/", ""),
                side=entry_side,
                order_type="LIMIT",
                quantity=quantity,
                price=entry_price,
                timeInForce="GTC"
            )
            
            if not entry_response.success:
                return {
                    "success": False,
                    "error": f"Entry order failed: {entry_response.error}",
                    "portfolio_info": pm_info
                }
            
            entry_order = Order(
                order_id=entry_response.data.get("orderId", str(time.time())),
                symbol=symbol,
                side=entry_side,
                order_type="LIMIT",
                quantity=quantity,
                price=entry_price,
                status=OrderStatus.OPEN,
                exchange_data=entry_response.data
            )
            
            with self._lock:
                self.orders[entry_order.order_id] = entry_order
            
            # Place OCO for SL and TP (Binance-style)
            exit_side = "sell" if side == "long" else "buy"
            
            try:
                oco_response = await self._place_oco_sl_tp(
                    symbol=symbol.replace("/", ""),
                    side=exit_side,
                    quantity=quantity,
                    stop_price=stop_price,
                    limit_price=limit_price
                )
                
                if oco_response.success:
                    # Create SL and TP orders from OCO response
                    sl_order = Order(
                        order_id=oco_response.data.get("orders", [{}])[0].get("orderId", str(time.time()) + "_sl"),
                        symbol=symbol,
                        side=exit_side,
                        order_type="STOP_LOSS_LIMIT",
                        quantity=quantity,
                        price=stop_price,
                        stop_price=stop_price,
                        status=OrderStatus.OPEN,
                        exchange_data=oco_response.data
                    )
                    
                    tp_order = Order(
                        order_id=oco_response.data.get("orders", [{}, {}])[1].get("orderId", str(time.time()) + "_tp"),
                        symbol=symbol,
                        side=exit_side,
                        order_type="LIMIT",
                        quantity=quantity,
                        price=limit_price,
                        status=OrderStatus.OPEN,
                        exchange_data=oco_response.data
                    )
                    
                    with self._lock:
                        self.orders[sl_order.order_id] = sl_order
                        self.orders[tp_order.order_id] = tp_order
                else:
                    # OCO failed, place individually
                    sl_order, tp_order = await self._place_sl_tp_individually(
                        symbol, exit_side, quantity, stop_price, limit_price
                    )
                
            except Exception as e:
                tprint(f"[OrderManagerV2] OCO placement failed, using individual orders: {e}")
                sl_order, tp_order = await self._place_sl_tp_individually(
                    symbol, exit_side, quantity, stop_price, limit_price
                )
            
            # Create position
            position = Position(
                position_id=f"{symbol}_{int(time.time())}",
                symbol=symbol,
                side=side,
                strategy_id=strategy_id,
                entry_price=entry_price,
                size=position_size_usdt,
                quantity=quantity,
                status=PositionStatus.PENDING_ENTRY,
                entry_order=entry_order,
                stop_loss_order=sl_order,
                take_profit_order=tp_order,
                metadata={
                    "params": params,
                    "peak_price": entry_price,
                    "portfolio_info": pm_info,
                    "confidence_score": confidence_score,
                }
            )
            
            with self._lock:
                self.positions[symbol] = position
            
            # Record in PortfolioManager
            self.portfolio_mgr.record_position_open(
                symbol=symbol,
                side=side,
                strategy_id=strategy_id,
                position_size=position_size_usdt,
                entry_price=entry_price,
                entry_time=pd.Timestamp.now(tz="UTC"),
            )
            
            tprint(f"[OrderManagerV2] Placed OCO for {symbol}: size={position_size_usdt:.2f} USDT, "
                   f"SL={stop_price:.4f}, TP={limit_price:.4f}")
            
            return {
                "success": True,
                "position": position,
                "portfolio_info": pm_info,
                "entry_order": entry_order,
                "sl_order": sl_order,
                "tp_order": tp_order,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Order placement failed: {e}",
                "portfolio_info": pm_info
            }
    
    async def _place_oco_sl_tp(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: float
    ) -> Any:
        """Place OCO order for SL and TP (Binance-specific)."""
        # This is exchange-specific; for Binance use OCO endpoint
        # For now, return failure to trigger individual placement
        # TODO: Implement exchange-specific OCO
        class FakeResponse:
            success = False
            error = "OCO not implemented"
            data = {}
        return FakeResponse()
    
    async def _place_sl_tp_individually(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: float
    ) -> Tuple[Order, Order]:
        """Place SL and TP orders individually."""
        # Place stop loss
        sl_response = await self.api_client.create_order(
            symbol=symbol,
            side=side,
            order_type="STOP_LOSS_LIMIT",
            quantity=quantity,
            price=stop_price,
            stopPrice=stop_price,
            timeInForce="GTC"
        )
        
        sl_order = Order(
            order_id=sl_response.data.get("orderId", str(time.time()) + "_sl") if sl_response.success else str(time.time()) + "_sl",
            symbol=symbol,
            side=side,
            order_type="STOP_LOSS_LIMIT",
            quantity=quantity,
            price=stop_price,
            stop_price=stop_price,
            status=OrderStatus.OPEN if sl_response.success else OrderStatus.REJECTED,
            exchange_data=sl_response.data if sl_response.success else {}
        )
        
        with self._lock:
            self.orders[sl_order.order_id] = sl_order
        
        # Place take profit
        tp_response = await self.api_client.create_order(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=limit_price,
            timeInForce="GTC"
        )
        
        tp_order = Order(
            order_id=tp_response.data.get("orderId", str(time.time()) + "_tp") if tp_response.success else str(time.time()) + "_tp",
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=limit_price,
            status=OrderStatus.OPEN if tp_response.success else OrderStatus.REJECTED,
            exchange_data=tp_response.data if tp_response.success else {}
        )
        
        with self._lock:
            self.orders[tp_order.order_id] = tp_order
        
        return sl_order, tp_order
    
    async def _get_wallet_balance(self) -> Dict[str, float]:
        """Get wallet balance from exchange."""
        try:
            response = await self.api_client.get_account_info()
            if response.success:
                balances = {}
                # Parse Binance-style balances
                for balance in response.data.get("balances", []):
                    asset = balance.get("asset", "")
                    free = float(balance.get("free", 0))
                    locked = float(balance.get("locked", 0))
                    balances[asset] = free + locked
                return balances
        except Exception as e:
            tprint(f"[OrderManagerV2] Error fetching wallet balance: {e}")
        
        return {"USDT": 10000.0}  # Fallback
    
    def _close_position(self, symbol: str, exit_price: Optional[float], reason: str) -> None:
        """Close a position and update PortfolioManager."""
        with self._lock:
            position = self.positions.get(symbol)
            if not position:
                return
            
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.utcnow()
            position.exit_price = exit_price
            
            # Calculate PnL
            if position.side == "long":
                position.pnl = (exit_price - position.entry_price) * position.quantity if exit_price else 0
                position.pnl_pct = (exit_price - position.entry_price) / position.entry_price if exit_price else 0
            else:
                position.pnl = (position.entry_price - exit_price) * position.quantity if exit_price else 0
                position.pnl_pct = (position.entry_price - exit_price) / position.entry_price if exit_price else 0
            
            # Update PortfolioManager
            self.portfolio_mgr.record_position_close(
                symbol=symbol,
                exit_price=exit_price or position.entry_price,
                exit_time=pd.Timestamp(position.closed_at, tz="UTC"),
            )
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            tprint(f"[OrderManagerV2] Position closed: {symbol}, PnL: {position.pnl:.2f} ({reason})")
            
            # Notify listeners
            for callback in self._on_position_close:
                callback(position)
    
    async def emergency_close_all(self) -> None:
        """Emergency close all positions."""
        tprint("[OrderManagerV2] EMERGENCY CLOSE ALL POSITIONS")
        
        with self._lock:
            symbols = list(self.positions.keys())
        
        for symbol in symbols:
            try:
                # Cancel all orders
                position = self.positions.get(symbol)
                if position:
                    for order in [position.entry_order, position.stop_loss_order, position.take_profit_order]:
                        if order:
                            try:
                                await self.api_client.cancel_order(
                                    symbol=symbol.replace("/", ""),
                                    order_id=order.order_id
                                )
                            except Exception:
                                pass
                
                # Market close
                # Get current price
                ticker_response = await self.api_client.get_ticker(symbol.replace("/", ""))
                current_price = float(ticker_response.data.get("lastPrice", 0)) if ticker_response.success else None
                
                self._close_position(symbol, current_price, "emergency_close")
                
            except Exception as e:
                tprint(f"[OrderManagerV2] Error emergency closing {symbol}: {e}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        with self._lock:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions."""
        with self._lock:
            return self.positions.copy()
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state from PortfolioManager."""
        return self.portfolio_mgr.get_portfolio_state()
    
    def register_position_open_callback(self, callback: Callable[[Position], None]) -> None:
        """Register callback for position open events."""
        self._on_position_open.append(callback)
    
    def register_position_close_callback(self, callback: Callable[[Position], None]) -> None:
        """Register callback for position close events."""
        self._on_position_close.append(callback)


__all__ = ["OrderManagerV2", "Order", "Position", "OrderStatus", "PositionStatus"]
