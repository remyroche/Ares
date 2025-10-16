"""
Order Manager

Unified order management system for handling order creation, execution,
tracking, and management across different exchanges and trading modes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..config.trading_config import TradingConfig
from ..config.execution_config import ExecutionConfig
from ..utils.error_handling import (
    ExecutionError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config, validate_order_params

logger = system_logger.getChild('OrderManager')

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"

class TimeInForce(Enum):
    """Time in force options."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order
    GTD = "GTD"  # Good Till Date

@dataclass
class Order:
    """Order information."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    trailing_stop: Optional[float] = None  # For trailing stop orders
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fees: float = 0.0
    exchange_order_id: Optional[str] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity

@dataclass
class OrderExecution:
    """Order execution information."""
    execution_id: str
    order_id: str
    timestamp: datetime
    quantity: float
    price: float
    fees: float
    exchange_execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderBook:
    """Order book information."""
    symbol: str
    bids: List[Tuple[float, float]]  # List of (price, quantity) pairs
    asks: List[Tuple[float, float]]  # List of (price, quantity) pairs
    timestamp: datetime
    exchange: str

class OrderManager:
    """
    Unified Order Management System

    Handles order creation, execution, tracking, and management across
    different exchanges and trading modes (paper/live).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the order manager.

        Args:
            config: Configuration dictionary containing trading and execution settings
        """
        self.config = config
        self.logger = logger.getChild('OrderManager')

        # Validate configuration
        self.trading_config = TradingConfig(**config.get('trading_config', {}))
        self.execution_config = ExecutionConfig(**config.get('execution_config', {}))

        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: Dict[str, Order] = {}
        self.executions: Dict[str, List[OrderExecution]] = {}
        self.order_books: Dict[str, OrderBook] = {}

        # Exchange interfaces
        self.exchange_interfaces: Dict[str, Any] = {}

        # Performance tracking
        self.order_count = 0
        self.execution_count = 0
        self.total_fees = 0.0

        tprint_info("🚀 Initializing Order Manager...")

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def initialize(self) -> None:
        """Initialize order manager components."""
        try:
            # Initialize exchange interfaces
            await self._initialize_exchange_interfaces()

            # Load order history if resuming
            await self._load_order_history()

            tprint_success("✅ Order Manager initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Order Manager: {str(e)}")
            raise

    async def _initialize_exchange_interfaces(self) -> None:
        """Initialize exchange interfaces."""
        # This would typically initialize different exchange clients
        # For now, we'll use a generic interface
        self.exchange_interfaces['default'] = None  # Placeholder
        self.logger.info("Exchange interfaces initialized")

    async def _load_order_history(self) -> None:
        """Load order history from storage."""
        # Placeholder for loading order history
        self.logger.info("Order history loaded")

    @critical_operation
    @require_no_fallback
    @handles_errors
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            trailing_stop: Trailing stop percentage
            time_in_force: Time in force
            expires_at: Expiration time
            metadata: Additional order metadata

        Returns:
            Created order object
        """
        # Validate order parameters
        await validate_order_params(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )

        # Create order object
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trailing_stop=trailing_stop,
            time_in_force=time_in_force,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        # Store order
        self.active_orders[order.order_id] = order
        self.order_count += 1

        tprint_info(f"📝 Created {side.value} order for {symbol}: {quantity} @ {price}")

        # Submit order to exchange
        await self._submit_order(order)

        return order

    async def _submit_order(self, order: Order) -> None:
        """Submit order to exchange."""
        try:
            order.status = OrderStatus.SUBMITTED

            if self.trading_config.mode == TradingMode.PAPER:
                # Paper trading - simulate order
                await self._simulate_order_execution(order)
            else:
                # Live trading - submit to exchange
                await self._execute_live_order(order)

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            tprint_error(f"❌ Failed to submit order {order.order_id}: {str(e)}")
            raise

    async def _simulate_order_execution(self, order: Order) -> None:
        """Simulate order execution for paper trading."""
        # Simulate immediate execution for market orders
        if order.order_type == OrderType.MARKET:
            # Get current market price (simulated)
            current_price = await self._get_current_price(order.symbol)

            # Create execution
            execution = OrderExecution(
                execution_id=str(uuid.uuid4()),
                order_id=order.order_id,
                timestamp=datetime.now(),
                quantity=order.quantity,
                price=current_price,
                fees=order.quantity * current_price * 0.001  # 0.1% fee
            )

            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.remaining_quantity = 0.0
            order.average_fill_price = current_price
            order.fees = execution.fees

            # Store execution
            if order.order_id not in self.executions:
                self.executions[order.order_id] = []
            self.executions[order.order_id].append(execution)

            self.execution_count += 1
            self.total_fees += execution.fees

            tprint_success(f"✅ Simulated order {order.order_id} filled @ {current_price}")

        else:
            # For limit orders, we'd need more complex simulation
            order.status = OrderStatus.PENDING
            tprint_warning(f"⚠️ Limit orders not yet implemented in paper trading mode")

    async def _execute_live_order(self, order: Order) -> None:
        """Execute order on live exchange."""
        # Placeholder for live order execution
        # This would integrate with actual exchange APIs
        tprint_warning(f"⚠️ Live order execution not yet implemented")
        order.status = OrderStatus.PENDING

    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        # Placeholder - in real implementation, this would fetch from exchange
        # For now, return a simulated price
        return 3000.0 if symbol.startswith('ETH') else 50000.0

    @handles_errors
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        if order_id not in self.active_orders:
            tprint_warning(f"⚠️ Order {order_id} not found")
            return False

        order = self.active_orders[order_id]

        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            tprint_warning(f"⚠️ Cannot cancel order {order_id} with status {order.status.value}")
            return False

        try:
            # Cancel on exchange
            if self.trading_config.mode == TradingMode.PAPER:
                order.status = OrderStatus.CANCELLED
            else:
                await self._cancel_live_order(order)

            tprint_info(f"❌ Cancelled order {order_id}")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to cancel order {order_id}: {str(e)}")
            return False

    async def _cancel_live_order(self, order: Order) -> None:
        """Cancel order on live exchange."""
        # Placeholder for live order cancellation
        tprint_warning(f"⚠️ Live order cancellation not yet implemented")

    @handles_errors
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get the status of an order.

        Args:
            order_id: Order ID

        Returns:
            Order status or None if not found
        """
        if order_id in self.active_orders:
            return self.active_orders[order_id].status
        elif order_id in self.order_history:
            return self.order_history[order_id].status
        else:
            return None

    @handles_errors
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        return self.active_orders.get(order_id) or self.order_history.get(order_id)

    @handles_errors
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all active orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())

        if symbol:
            orders = [order for order in orders if order.symbol == symbol]

        return orders

    @handles_errors
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Order]:
        """
        Get order history with optional filters.

        Args:
            symbol: Optional symbol filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of historical orders
        """
        orders = list(self.order_history.values())

        if symbol:
            orders = [order for order in orders if order.symbol == symbol]

        if start_time:
            orders = [order for order in orders if order.timestamp >= start_time]

        if end_time:
            orders = [order for order in orders if order.timestamp <= end_time]

        return orders

    @handles_errors
    async def get_executions(self, order_id: str) -> List[OrderExecution]:
        """
        Get executions for a specific order.

        Args:
            order_id: Order ID

        Returns:
            List of executions
        """
        return self.executions.get(order_id, [])

    async def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """
        Get order book for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Order book or None if not available
        """
        return self.order_books.get(symbol)

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel all active orders
            active_orders = list(self.active_orders.keys())
            for order_id in active_orders:
                await self.cancel_order(order_id)

            # Move active orders to history
            self.order_history.update(self.active_orders)
            self.active_orders.clear()

            tprint_info("🧹 Order Manager cleaned up successfully")

        except Exception as e:
            tprint_error(f"❌ Error during Order Manager cleanup: {str(e)}")

# Factory functions for easy instantiation
async def create_order_manager(config: Dict[str, Any]) -> OrderManager:
    """Create and initialize an order manager."""
    manager = OrderManager(config)
    await manager.initialize()
    return manager

def get_order_manager() -> OrderManager:
    """Get the global order manager instance."""
    # This would typically return a singleton instance
    # For now, return None as placeholder
    return None
