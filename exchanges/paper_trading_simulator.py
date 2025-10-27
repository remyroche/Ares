"""
Paper Trading Simulator

This module provides a comprehensive trading simulator for paper trading mode.
It simulates real trading conditions including slippage, fees, and market impact.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
import math

from .base_exchange.exchange_interface import OrderSide, OrderType, OrderStatus


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


@dataclass
class SimulatedOrder:
    """Simulated order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    executed_price: float
    executed_quantity: float
    remaining_quantity: float
    status: OrderStatus
    commission: float
    commission_asset: str
    slippage: float
    executed_at: datetime
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_commission: float
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    """Configuration for the trading simulator"""
    # Fee structure
    maker_fee_rate: float = 0.001  # 0.1% maker fee
    taker_fee_rate: float = 0.001  # 0.1% taker fee
    
    # Slippage settings
    base_slippage_rate: float = 0.0001  # 0.01% base slippage
    slippage_volatility_factor: float = 0.5  # Volatility impact on slippage
    max_slippage_rate: float = 0.01  # 1% maximum slippage
    
    # Market impact simulation
    market_impact_factor: float = 0.0001  # Market impact per unit volume
    min_volume_threshold: float = 1000.0  # Minimum volume for market impact
    
    # Order execution simulation
    execution_delay_ms: Tuple[int, int] = (10, 100)  # Min/max execution delay
    partial_fill_probability: float = 0.1  # 10% chance of partial fill
    rejection_probability: float = 0.001  # 0.1% chance of order rejection
    
    # Risk management
    max_position_size: float = 1000000.0  # Maximum position size
    max_daily_trades: int = 1000  # Maximum trades per day


class PaperTradingSimulator:
    """
    Comprehensive paper trading simulator that mimics real exchange behavior.
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        """
        Initialize the paper trading simulator.
        
        Args:
            config: Simulator configuration
        """
        self.config = config or SimulatorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Simulator state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, SimulatedOrder] = {}
        self.balance: Dict[str, float] = {"USDT": 100000.0}  # Starting balance
        self.trade_history: List[SimulatedOrder] = []
        self.daily_trade_count: int = 0
        self.last_reset_date: str = datetime.now().strftime("%Y-%m-%d")
        
        # Market data cache for price simulation
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Paper Trading Simulator initialized")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"SIM_{uuid.uuid4().hex[:12].upper()}"
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        In a real implementation, this would fetch from market data.
        For simulation, we'll use cached prices or generate realistic prices.
        """
        if symbol in self.price_cache:
            return self.price_cache[symbol].get("price", 1.0)
        
        # Generate a realistic price based on symbol
        if "BTC" in symbol:
            base_price = 50000.0
        elif "ETH" in symbol:
            base_price = 3000.0
        else:
            base_price = 1.0
        
        # Add some random variation
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        price = base_price * (1 + variation)
        
        # Cache the price
        self.price_cache[symbol] = {
            "price": price,
            "timestamp": datetime.now(timezone.utc)
        }
        
        return price
    
    def _calculate_slippage(self, symbol: str, side: OrderSide, quantity: float, price: float) -> float:
        """
        Calculate slippage based on order size and market conditions.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price
            
        Returns:
            Slippage amount in price units
        """
        # Base slippage
        base_slippage = self.config.base_slippage_rate
        
        # Volume-based slippage (larger orders = more slippage)
        volume_impact = min(quantity * price / self.config.min_volume_threshold, 1.0)
        volume_slippage = base_slippage * volume_impact * self.config.market_impact_factor
        
        # Random volatility factor
        volatility_factor = random.uniform(0.5, 1.5) * self.config.slippage_volatility_factor
        
        # Total slippage
        total_slippage_rate = min(
            base_slippage + volume_slippage * volatility_factor,
            self.config.max_slippage_rate
        )
        
        # Apply slippage (buy orders get worse price, sell orders get better price)
        slippage_amount = price * total_slippage_rate
        if side == OrderSide.BUY:
            return slippage_amount  # Higher price for buys
        else:
            return -slippage_amount  # Lower price for sells
    
    def _calculate_commission(self, quantity: float, price: float, order_type: OrderType) -> float:
        """
        Calculate commission based on order type and size.
        
        Args:
            quantity: Order quantity
            price: Order price
            order_type: Order type (market/limit)
            
        Returns:
            Commission amount
        """
        notional_value = quantity * price
        
        # Use maker fee for limit orders, taker fee for market orders
        if order_type == OrderType.LIMIT:
            fee_rate = self.config.maker_fee_rate
        else:
            fee_rate = self.config.taker_fee_rate
        
        return notional_value * fee_rate
    
    def _simulate_execution_delay(self) -> float:
        """Simulate realistic order execution delay"""
        delay_ms = random.randint(
            self.config.execution_delay_ms[0],
            self.config.execution_delay_ms[1]
        )
        return delay_ms / 1000.0  # Convert to seconds
    
    async def _simulate_order_execution(self, order: SimulatedOrder) -> SimulatedOrder:
        """
        Simulate order execution with realistic delays and conditions.
        
        Args:
            order: Order to execute
            
        Returns:
            Updated order with execution details
        """
        # Simulate execution delay
        delay = self._simulate_execution_delay()
        await asyncio.sleep(delay)
        
        # Check for order rejection
        if random.random() < self.config.rejection_probability:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "Simulated rejection"
            return order
        
        # Get current market price
        current_price = self._get_current_price(order.symbol)
        
        # Calculate slippage
        slippage = self._calculate_slippage(
            order.symbol, order.side, order.quantity, current_price
        )
        
        # Calculate executed price
        if order.order_type == OrderType.MARKET:
            order.executed_price = current_price + slippage
        else:  # Limit order
            if order.side == OrderSide.BUY:
                order.executed_price = min(order.price, current_price + slippage)
            else:
                order.executed_price = max(order.price, current_price + slippage)
        
        # Check if limit order can be filled
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.executed_price > order.price:
                order.status = OrderStatus.CANCELLED
                order.metadata["cancellation_reason"] = "Price not reached"
                return order
            elif order.side == OrderSide.SELL and order.executed_price < order.price:
                order.status = OrderStatus.CANCELLED
                order.metadata["cancellation_reason"] = "Price not reached"
                return order
        
        # Simulate partial fills
        if random.random() < self.config.partial_fill_probability:
            fill_ratio = random.uniform(0.3, 0.9)
            order.executed_quantity = order.quantity * fill_ratio
            order.remaining_quantity = order.quantity - order.executed_quantity
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.executed_quantity = order.quantity
            order.remaining_quantity = 0.0
            order.status = OrderStatus.FILLED
        
        # Calculate commission
        order.commission = self._calculate_commission(
            order.executed_quantity, order.executed_price, order.order_type
        )
        order.commission_asset = "USDT"  # Assume USDT for commission
        
        # Record execution time
        order.executed_at = datetime.now(timezone.utc)
        order.slippage = slippage
        
        return order
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a simulated order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            quantity: Order quantity
            price: Order price (required for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order response dictionary
        """
        try:
            # Validate inputs
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            if order_type == OrderType.LIMIT and price is None:
                raise ValueError("Price required for limit orders")
            
            # Check daily trade limit
            if self.daily_trade_count >= self.config.max_daily_trades:
                raise ValueError("Daily trade limit exceeded")
            
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Set price for market orders
            if order_type == OrderType.MARKET:
                price = self._get_current_price(symbol)
            
            # Create order
            order = SimulatedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                executed_price=0.0,
                executed_quantity=0.0,
                remaining_quantity=quantity,
                status=OrderStatus.PENDING,
                commission=0.0,
                commission_asset="USDT",
                slippage=0.0,
                executed_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                metadata=kwargs
            )
            
            # Store order
            self.orders[order_id] = order
            
            # Simulate execution
            executed_order = await self._simulate_order_execution(order)
            
            # Update order in storage
            self.orders[order_id] = executed_order
            
            # If order was filled, update positions and balance
            if executed_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                await self._update_position(executed_order)
                await self._update_balance(executed_order)
                
                # Add to trade history
                self.trade_history.append(executed_order)
                self.daily_trade_count += 1
            
            # Convert to response format
            return self._order_to_response(executed_order)
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise
    
    async def _update_position(self, order: SimulatedOrder) -> None:
        """
        Update position based on executed order.
        
        Args:
            order: Executed order
        """
        symbol = order.symbol
        position_key = f"{symbol}_{order.side.value}"
        
        if position_key not in self.positions:
            # Create new position
            position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
            self.positions[position_key] = Position(
                symbol=symbol,
                side=position_side,
                quantity=0.0,
                entry_price=0.0,
                current_price=order.executed_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                total_commission=0.0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
        
        position = self.positions[position_key]
        
        # Update position based on order side
        if order.side == OrderSide.BUY:
            # Opening or adding to long position
            if position.quantity >= 0:  # Long position or no position
                # Calculate new average entry price
                total_cost = (position.quantity * position.entry_price + 
                            order.executed_quantity * order.executed_price)
                total_quantity = position.quantity + order.executed_quantity
                position.entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
            else:  # Short position - closing or reducing
                close_quantity = min(abs(position.quantity), order.executed_quantity)
                remaining_quantity = order.executed_quantity - close_quantity
                
                # Calculate realized PnL for closed portion
                realized_pnl = close_quantity * (position.entry_price - order.executed_price)
                position.realized_pnl += realized_pnl
                position.quantity += close_quantity  # Reduce short position
                
                # If there's remaining quantity, open long position
                if remaining_quantity > 0:
                    position.entry_price = order.executed_price
                    position.quantity = remaining_quantity
        else:  # SELL order
            # Opening or adding to short position
            if position.quantity <= 0:  # Short position or no position
                # Calculate new average entry price
                total_cost = (abs(position.quantity) * position.entry_price + 
                            order.executed_quantity * order.executed_price)
                total_quantity = abs(position.quantity) + order.executed_quantity
                position.entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = -total_quantity  # Negative for short
            else:  # Long position - closing or reducing
                close_quantity = min(position.quantity, order.executed_quantity)
                remaining_quantity = order.executed_quantity - close_quantity
                
                # Calculate realized PnL for closed portion
                realized_pnl = close_quantity * (order.executed_price - position.entry_price)
                position.realized_pnl += realized_pnl
                position.quantity -= close_quantity  # Reduce long position
                
                # If there's remaining quantity, open short position
                if remaining_quantity > 0:
                    position.entry_price = order.executed_price
                    position.quantity = -remaining_quantity  # Negative for short
        
        # Update commission and timestamp
        position.total_commission += order.commission
        position.updated_at = datetime.now(timezone.utc)
        
        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-8:
            del self.positions[position_key]
    
    async def _update_balance(self, order: SimulatedOrder) -> None:
        """
        Update balance based on executed order.
        
        Args:
            order: Executed order
        """
        # Calculate total cost (including commission)
        total_cost = order.executed_quantity * order.executed_price + order.commission
        
        if order.side == OrderSide.BUY:
            # Deduct cost from USDT balance
            self.balance["USDT"] -= total_cost
        else:  # SELL order
            # Add proceeds to USDT balance
            self.balance["USDT"] += total_cost - order.commission
    
    def _order_to_response(self, order: SimulatedOrder) -> Dict[str, Any]:
        """Convert SimulatedOrder to response dictionary"""
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "executed_price": order.executed_price,
            "executed_quantity": order.executed_quantity,
            "remaining_quantity": order.remaining_quantity,
            "status": order.status.value,
            "commission": order.commission,
            "commission_asset": order.commission_asset,
            "slippage": order.slippage,
            "executed_at": order.executed_at.isoformat(),
            "created_at": order.created_at.isoformat(),
            "metadata": order.metadata
        }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in status: {order.status.value}")
        
        # Cancel the order
        order.status = OrderStatus.CANCELLED
        order.remaining_quantity = 0.0
        order.metadata["cancelled_at"] = datetime.now(timezone.utc).isoformat()
        
        return self._order_to_response(order)
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status information
        """
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        return self._order_to_response(self.orders[order_id])
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open orders
        """
        open_orders = []
        
        for order in self.orders.values():
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(self._order_to_response(order))
        
        return open_orders
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of current positions
        """
        positions = []
        
        for position in self.positions.values():
            # Calculate current price and unrealized PnL
            current_price = self._get_current_price(position.symbol)
            position.current_price = current_price
            
            if position.quantity > 0:  # Long position
                position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
            elif position.quantity < 0:  # Short position
                position.unrealized_pnl = abs(position.quantity) * (position.entry_price - current_price)
            else:
                position.unrealized_pnl = 0.0
            
            positions.append({
                "symbol": position.symbol,
                "side": position.side.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "total_commission": position.total_commission,
                "created_at": position.created_at.isoformat(),
                "updated_at": position.updated_at.isoformat(),
                "metadata": position.metadata
            })
        
        return positions
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get current balance.
        
        Returns:
            Current balance dictionary
        """
        return self.balance.copy()
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of trades to return
            
        Returns:
            List of trade history
        """
        trades = self.trade_history.copy()
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Sort by execution time (newest first)
        trades.sort(key=lambda x: x.executed_at, reverse=True)
        
        # Limit results
        trades = trades[:limit]
        
        return [self._order_to_response(trade) for trade in trades]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        total_trades = len(self.trade_history)
        total_volume = sum(trade.executed_quantity * trade.executed_price for trade in self.trade_history)
        total_commission = sum(trade.commission for trade in self.trade_history)
        
        # Calculate win/loss statistics
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        
        for position in self.positions.values():
            total_pnl += position.realized_pnl
            if position.realized_pnl > 0:
                winning_trades += 1
            elif position.realized_pnl < 0:
                losing_trades += 1
        
        # Calculate current portfolio value
        portfolio_value = self.balance["USDT"]
        for position in self.positions.values():
            portfolio_value += position.quantity * position.current_price
        
        return {
            "total_trades": total_trades,
            "total_volume": total_volume,
            "total_commission": total_commission,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / max(total_trades, 1) * 100,
            "total_realized_pnl": total_pnl,
            "current_portfolio_value": portfolio_value,
            "current_balance": self.balance.copy(),
            "active_positions": len(self.positions),
            "daily_trade_count": self.daily_trade_count
        }
    
    def reset_daily_counters(self) -> None:
        """Reset daily trade counters"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
    
    def update_market_price(self, symbol: str, price: float) -> None:
        """
        Update market price for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self.price_cache[symbol] = {
            "price": price,
            "timestamp": datetime.now(timezone.utc)
        }
    
    def get_simulator_status(self) -> Dict[str, Any]:
        """Get simulator status"""
        return {
            "is_active": True,
            "total_orders": len(self.orders),
            "active_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "daily_trade_count": self.daily_trade_count,
            "balance": self.balance.copy(),
            "config": {
                "maker_fee_rate": self.config.maker_fee_rate,
                "taker_fee_rate": self.config.taker_fee_rate,
                "base_slippage_rate": self.config.base_slippage_rate,
                "max_slippage_rate": self.config.max_slippage_rate,
                "max_position_size": self.config.max_position_size,
                "max_daily_trades": self.config.max_daily_trades
            }
        }