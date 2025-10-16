"""
Paper Trading Engine with Realistic Market Simulation

This module provides a comprehensive paper trading engine that simulates
realistic market conditions for A/B/C testing of multiple models.

Key Features:
- Realistic market simulation with slippage, fees, and latency
- Order book simulation with bid-ask spreads
- Market impact modeling
- Liquidity constraints and partial fills
- Real-time position tracking and P&L calculation
- Risk management and position sizing
- Performance metrics calculation
- Trade execution simulation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path
import copy
import uuid
import random
from collections import defaultdict, deque

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose, validate_data_quality,
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for paper trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class MarketCondition(Enum):
    """Market conditions affecting execution."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    HALTED = "halted"
    GAPPING = "gapping"

@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    fees: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0

@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Trade:
    """Trade execution record."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fees: float
    slippage: float
    market_impact: float
    executed_at: datetime = field(default_factory=datetime.now)

@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    volatility: float
    spread: float
    market_condition: MarketCondition

@dataclass
class PaperTradingConfig:
    """Paper trading configuration."""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital
    risk_per_trade: float = 0.02  # 2% risk per trade
    commission_rate: float = 0.001  # 0.1% commission
    slippage_model: str = "linear"  # linear, sqrt, constant
    market_impact_model: str = "sqrt"  # linear, sqrt, constant
    enable_slippage: bool = True
    enable_market_impact: bool = True
    enable_partial_fills: bool = True
    max_slippage_bps: float = 50.0  # 50 basis points
    latency_ms: Tuple[int, int] = (10, 100)  # Min, max latency
    volatility_multiplier: float = 1.0
    liquidity_factor: float = 1.0

class MarketSimulator:
    """Realistic market simulation engine."""

    def __init__(self, config: PaperTradingConfig):
        """Initialize market simulator."""
        self.config = config
        self.logger = logger.getChild('MarketSimulator')

        # Market state
        self.current_prices: Dict[str, float] = {}
        self.order_books: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self.market_conditions: Dict[str, MarketCondition] = {}
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Simulation parameters
        self.slippage_models = {
            "linear": self._linear_slippage,
            "sqrt": self._sqrt_slippage,
            "constant": self._constant_slippage
        }

        self.impact_models = {
            "linear": self._linear_impact,
            "sqrt": self._sqrt_impact,
            "constant": self._constant_impact
        }

        self.logger.info("🚀 MarketSimulator initialized")
        self.logger.info(f"📊 Slippage model: {config.slippage_model}")
        self.logger.info(f"📊 Market impact model: {config.market_impact_model}")

    def update_market_data(self, symbol: str, market_data: MarketData) -> None:
        """Update market data for a symbol."""
        self.current_prices[symbol] = market_data.last_price
        self.market_conditions[symbol] = market_data.market_condition

        # Update volatility history
        if market_data.volatility > 0:
            self.volatility_history[symbol].append(market_data.volatility)

        # Update order book
        self.order_books[symbol] = {
            "bids": [(market_data.bid_price, market_data.bid_size)],
            "asks": [(market_data.ask_price, market_data.ask_size)]
        }

    def simulate_order_execution(self, order: Order, market_data: MarketData) -> Tuple[Order, List[Trade]]:
        """Simulate order execution with realistic market conditions."""
        try:
            # Check if order can be executed
            if not self._can_execute_order(order, market_data):
                order.status = OrderStatus.REJECTED
                return order, []

            # Calculate execution parameters
            execution_price = self._calculate_execution_price(order, market_data)
            slippage = self._calculate_slippage(order, market_data)
            market_impact = self._calculate_market_impact(order, market_data)

            # Apply slippage and market impact
            if order.side == OrderSide.BUY:
                final_price = execution_price + slippage + market_impact
            else:
                final_price = execution_price - slippage - market_impact

            # Calculate fees
            fees = self._calculate_fees(order, final_price)

            # Simulate partial fills if enabled
            if self.config.enable_partial_fills:
                fills = self._simulate_partial_fills(order, final_price, market_data)
            else:
                fills = [self._create_fill(order, order.quantity, final_price, fees, slippage, market_impact)]

            # Update order status
            total_filled = sum(fill.quantity for fill in fills)
            if total_filled >= order.quantity:
                order.status = OrderStatus.FILLED
            elif total_filled > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.REJECTED

            order.filled_quantity = total_filled
            order.average_fill_price = sum(fill.price * fill.quantity for fill in fills) / total_filled if total_filled > 0 else 0
            order.fees = sum(fill.fees for fill in fills)
            order.slippage = slippage
            order.market_impact = market_impact
            order.updated_at = datetime.now()

            return order, fills

        except Exception as e:
            self.logger.error(f"❌ Error simulating order execution: {e}")
            order.status = OrderStatus.REJECTED
            return order, []

    def _can_execute_order(self, order: Order, market_data: MarketData) -> bool:
        """Check if order can be executed."""
        # Check market condition
        if market_data.market_condition == MarketCondition.HALTED:
            return False

        # Check if we have market data
        if not market_data.bid_price or not market_data.ask_price:
            return False

        # Check order validity
        if order.quantity <= 0:
            return False

        # Check limit order conditions
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price < market_data.ask_price:
                return False
            elif order.side == OrderSide.SELL and order.price > market_data.bid_price:
                return False

        return True

    def _calculate_execution_price(self, order: Order, market_data: MarketData) -> float:
        """Calculate base execution price."""
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                return market_data.ask_price
            else:
                return market_data.bid_price
        elif order.order_type == OrderType.LIMIT:
            return order.price
        else:
            return market_data.last_price

    def _calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """Calculate slippage based on order size and market conditions."""
        if not self.config.enable_slippage:
            return 0.0

        # Get volatility
        volatility = market_data.volatility
        if order.symbol in self.volatility_history and len(self.volatility_history[order.symbol]) > 0:
            volatility = np.mean(list(self.volatility_history[order.symbol]))

        # Calculate base slippage
        slippage_model = self.slippage_models.get(self.config.slippage_model, self._linear_slippage)
        base_slippage = slippage_model(order, market_data, volatility)

        # Apply market condition multiplier
        condition_multiplier = self._get_condition_multiplier(market_data.market_condition)

        # Apply volatility multiplier
        volatility_multiplier = 1.0 + (volatility * self.config.volatility_multiplier)

        # Cap slippage
        max_slippage = market_data.last_price * (self.config.max_slippage_bps / 10000)
        final_slippage = min(base_slippage * condition_multiplier * volatility_multiplier, max_slippage)

        return final_slippage

    def _linear_slippage(self, order: Order, market_data: MarketData, volatility: float) -> float:
        """Linear slippage model."""
        size_ratio = order.quantity / market_data.volume if market_data.volume > 0 else 0.01
        return market_data.last_price * size_ratio * 0.001  # 0.1% per 1% of volume

    def _sqrt_slippage(self, order: Order, market_data: MarketData, volatility: float) -> float:
        """Square root slippage model."""
        size_ratio = order.quantity / market_data.volume if market_data.volume > 0 else 0.01
        return market_data.last_price * np.sqrt(size_ratio) * 0.002  # 0.2% per sqrt(1% of volume)

    def _constant_slippage(self, order: Order, market_data: MarketData, volatility: float) -> float:
        """Constant slippage model."""
        return market_data.last_price * 0.0005  # 0.05% constant slippage

    def _calculate_market_impact(self, order: Order, market_data: MarketData) -> float:
        """Calculate market impact."""
        if not self.config.enable_market_impact:
            return 0.0

        impact_model = self.impact_models.get(self.config.market_impact_model, self._sqrt_impact)
        return impact_model(order, market_data)

    def _linear_impact(self, order: Order, market_data: MarketData) -> float:
        """Linear market impact model."""
        size_ratio = order.quantity / market_data.volume if market_data.volume > 0 else 0.01
        return market_data.last_price * size_ratio * 0.0005  # 0.05% per 1% of volume

    def _sqrt_impact(self, order: Order, market_data: MarketData) -> float:
        """Square root market impact model."""
        size_ratio = order.quantity / market_data.volume if market_data.volume > 0 else 0.01
        return market_data.last_price * np.sqrt(size_ratio) * 0.001  # 0.1% per sqrt(1% of volume)

    def _constant_impact(self, order: Order, market_data: MarketData) -> float:
        """Constant market impact model."""
        return market_data.last_price * 0.0002  # 0.02% constant impact

    def _calculate_fees(self, order: Order, price: float) -> float:
        """Calculate trading fees."""
        notional_value = order.quantity * price
        return notional_value * self.config.commission_rate

    def _simulate_partial_fills(self, order: Order, price: float, market_data: MarketData) -> List[Trade]:
        """Simulate partial fills based on market liquidity."""
        fills = []
        remaining_quantity = order.quantity

        # Calculate available liquidity
        if order.side == OrderSide.BUY:
            available_liquidity = market_data.ask_size
        else:
            available_liquidity = market_data.bid_size

        # Apply liquidity factor
        available_liquidity *= self.config.liquidity_factor

        while remaining_quantity > 0 and available_liquidity > 0:
            # Calculate fill size
            fill_size = min(remaining_quantity, available_liquidity * random.uniform(0.1, 0.8))

            if fill_size > 0:
                # Add some price variation for partial fills
                price_variation = price * random.uniform(-0.0001, 0.0001)  # ±0.01%
                fill_price = price + price_variation

                fees = self._calculate_fees(order, fill_price)
                slippage = abs(fill_price - price)
                market_impact = 0.0  # Already included in price

                fill = self._create_fill(order, fill_size, fill_price, fees, slippage, market_impact)
                fills.append(fill)

                remaining_quantity -= fill_size
                available_liquidity -= fill_size
            else:
                break

        return fills

    def _create_fill(self, order: Order, quantity: float, price: float,
                    fees: float, slippage: float, market_impact: float) -> Trade:
        """Create a trade fill."""
        return Trade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            fees=fees,
            slippage=slippage,
            market_impact=market_impact
        )

    def _get_condition_multiplier(self, condition: MarketCondition) -> float:
        """Get multiplier based on market condition."""
        multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 1.5,
            MarketCondition.ILLIQUID: 2.0,
            MarketCondition.HALTED: 0.0,
            MarketCondition.GAPPING: 3.0
        }
        return multipliers.get(condition, 1.0)

class PaperTradingEngine:
    """Comprehensive paper trading engine."""

    def __init__(self, config: PaperTradingConfig):
        """Initialize paper trading engine."""
        self.config = config
        self.logger = logger.getChild('PaperTradingEngine')

        # Core components
        self.market_simulator = MarketSimulator(config)

        # Trading state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.cash_balance: float = config.initial_capital
        self.total_fees: float = 0.0

        # Performance tracking
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.max_drawdown: float = 0.0
        self.peak_value: float = config.initial_capital

        # Risk management
        self.daily_pnl: float = 0.0
        self.max_daily_loss: float = config.initial_capital * 0.05  # 5% max daily loss

        self.logger.info("🚀 PaperTradingEngine initialized")
        self.logger.info(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"📊 Max position size: {config.max_position_size:.1%}")
        self.logger.info(f"⚠️ Risk per trade: {config.risk_per_trade:.1%}")

    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """Place a new order."""
        try:
            # Validate order
            if not self._validate_order(symbol, side, quantity, price):
                return ""

            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )

            # Store order
            self.orders[order.order_id] = order

            self.logger.info(f"📝 Order placed: {order.side.value} {order.quantity} {order.symbol} @ {order.price or 'MARKET'}")
            return order.order_id

        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return ""

    def execute_orders(self, market_data: Dict[str, MarketData]) -> Dict[str, List[Trade]]:
        """Execute pending orders against market data."""
        executed_trades = {}

        try:
            for order_id, order in list(self.orders.items()):
                if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                    continue

                if order.symbol not in market_data:
                    continue

                # Execute order
                updated_order, trades = self.market_simulator.simulate_order_execution(
                    order, market_data[order.symbol]
                )

                # Update order
                self.orders[order_id] = updated_order

                # Process trades
                if trades:
                    executed_trades[order_id] = trades
                    for trade in trades:
                        self._process_trade(trade)

                # Remove filled orders
                if updated_order.status == OrderStatus.FILLED:
                    del self.orders[order_id]

            return executed_trades

        except Exception as e:
            self.logger.error(f"❌ Error executing orders: {e}")
            return {}

    def _validate_order(self, symbol: str, side: OrderSide, quantity: float, price: Optional[float]) -> bool:
        """Validate order parameters."""
        # Check quantity
        if quantity <= 0:
            self.logger.warning("❌ Invalid quantity")
            return False

        # Check price for limit orders
        if price is not None and price <= 0:
            self.logger.warning("❌ Invalid price")
            return False

        # Check available capital for buy orders
        if side == OrderSide.BUY:
            required_capital = quantity * (price or self.current_prices.get(symbol, 0))
            if required_capital > self.cash_balance:
                self.logger.warning(f"❌ Insufficient capital: ${required_capital:,.2f} > ${self.cash_balance:,.2f}")
                return False

        # Check available position for sell orders
        elif side == OrderSide.SELL:
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                self.logger.warning(f"❌ Insufficient position: {quantity} > {self.positions.get(symbol, Position(symbol, 0, 0)).quantity}")
                return False

        return True

    def _process_trade(self, trade: Trade) -> None:
        """Process a completed trade."""
        try:
            # Add to trades list
            self.trades.append(trade)

            # Update cash balance
            trade_value = trade.quantity * trade.price
            if trade.side == OrderSide.BUY:
                self.cash_balance -= trade_value + trade.fees
            else:
                self.cash_balance += trade_value - trade.fees

            # Update total fees
            self.total_fees += trade.fees

            # Update position
            self._update_position(trade)

            self.logger.info(f"✅ Trade executed: {trade.side.value} {trade.quantity} {trade.symbol} @ ${trade.price:.4f}")

        except Exception as e:
            self.logger.error(f"❌ Error processing trade: {e}")

    def _update_position(self, trade: Trade) -> None:
        """Update position after trade execution."""
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=0.0,
                average_price=0.0
            )

        position = self.positions[trade.symbol]

        if trade.side == OrderSide.BUY:
            # Add to position
            total_cost = (position.quantity * position.average_price) + (trade.quantity * trade.price)
            total_quantity = position.quantity + trade.quantity

            if total_quantity > 0:
                position.average_price = total_cost / total_quantity
                position.quantity = total_quantity
        else:
            # Reduce position
            if position.quantity >= trade.quantity:
                # Calculate realized P&L
                realized_pnl = trade.quantity * (trade.price - position.average_price)
                position.realized_pnl += realized_pnl
                position.quantity -= trade.quantity

                # Remove position if fully closed
                if position.quantity <= 0:
                    del self.positions[trade.symbol]

        position.updated_at = datetime.now()

    def update_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Update portfolio value with current prices."""
        try:
            # Calculate position values
            position_value = 0.0
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position_value += position.quantity * current_prices[symbol]
                    # Update unrealized P&L
                    position.unrealized_pnl = position.quantity * (current_prices[symbol] - position.average_price)

            # Total portfolio value
            total_value = self.cash_balance + position_value

            # Update history
            self.portfolio_value_history.append((datetime.now(), total_value))

            # Update peak and drawdown
            if total_value > self.peak_value:
                self.peak_value = total_value

            current_drawdown = (self.peak_value - total_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            # Calculate daily return
            if len(self.portfolio_value_history) > 1:
                prev_value = self.portfolio_value_history[-2][1]
                daily_return = (total_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)

            return total_value

        except Exception as e:
            self.logger.error(f"❌ Error updating portfolio value: {e}")
            return self.cash_balance

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.portfolio_value_history:
                return {}

            # Basic metrics
            initial_value = self.config.initial_capital
            current_value = self.portfolio_value_history[-1][1]
            total_return = (current_value - initial_value) / initial_value

            # Calculate returns
            returns = np.array(self.daily_returns) if self.daily_returns else np.array([0.0])

            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0.0
            sortino_ratio = (np.mean(returns) * 252) / downside_volatility if downside_volatility > 0 else 0.0

            # Win rate
            winning_trades = len([r for r in returns if r > 0])
            win_rate = winning_trades / len(returns) if returns.size > 0 else 0.0

            # Calmar ratio
            annual_return = np.mean(returns) * 252
            calmar_ratio = annual_return / self.max_drawdown if self.max_drawdown > 0 else 0.0

            return {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": win_rate,
                "total_fees": self.total_fees,
                "current_value": current_value,
                "cash_balance": self.cash_balance,
                "position_count": len(self.positions),
                "trade_count": len(self.trades)
            }

        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            return {}

    def get_position_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all positions."""
        summary = {}

        for symbol, position in self.positions.items():
            summary[symbol] = {
                "quantity": position.quantity,
                "average_price": position.average_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "total_fees": position.total_fees
            }

        return summary

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self.logger.info(f"❌ Order cancelled: {order_id}")
                return True

        return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None

    def reset_portfolio(self) -> None:
        """Reset portfolio to initial state."""
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.cash_balance = self.config.initial_capital
        self.total_fees = 0.0
        self.portfolio_value_history.clear()
        self.daily_returns.clear()
        self.max_drawdown = 0.0
        self.peak_value = self.config.initial_capital
        self.daily_pnl = 0.0

        self.logger.info("🔄 Portfolio reset to initial state")

# Convenience function for easy integration
def create_paper_trading_engine(config: PaperTradingConfig) -> PaperTradingEngine:
    """Create a paper trading engine instance."""
    return PaperTradingEngine(config)
