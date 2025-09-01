# src/tactician/enhanced_order_manager.py

"""
Enhanced Order Manager for Tactician
Handles sophisticated order management including stop-limit orders and leveraged limit orders
with partial fill management.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
# from src.utils.prometheus_metrics import metrics  # Temporarily commented due to syntax errors
from src.utils.warning_symbols import (
    failed,
    missing,
)

class OrderType(Enum):
    """Order types supported by the enhanced order manager."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderRequest:
    """Order request data structure."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    leverage: float | None = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    reduce_only: bool = False
    close_on_trigger: bool = False
    order_link_id: str | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    trailing_stop: float | None = None
    iceberg_qty: float | None = None
    strategy_id: str | None = None
    strategy_type: str | None = None  # "CHASE_MICRO_BREAKOUT", "LIMIT_ORDER_RETURN", etc.
    post_only: bool | None = None

@dataclass
class OrderFill:
    """Order fill data structure."""

    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    commission_asset: str
    trade_time: datetime
    is_maker: bool = False

@dataclass
class OrderState:
    """Order state tracking."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    original_quantity: float
    executed_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fills: List[OrderFill] = field(default_factory=list)
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    strategy_id: str | None = None
    strategy_type: str | None = None

class EnhancedOrderManager:
    """
    Enhanced order manager for sophisticated order handling.

    Features:
    - Stop-limit order management
    - Leveraged limit order handling
    - Partial fill tracking
    - Order state management
    - Strategy-specific order handling
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the enhanced order manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedOrderManager")

        # Order tracking
        self.active_orders: Dict[str, OrderState] = {}
        self.order_history: List[OrderState] = []

        # Configuration
        self.max_active_orders = config.get("max_active_orders", 100)
        self.max_history_size = config.get("max_history_size", 1000)
        self.default_timeout = config.get("order_timeout", 300)  # 5 minutes

        # Metrics
        self.metrics = metrics

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order manager initialization"
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order creation"
    )
    def _validate_order_request(self, order_request: OrderRequest) -> bool:
        """
        Validate order request parameters.

        Args:
            order_request: Order request to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not order_request.symbol:
                self.logger.error(missing("Symbol is required"))
                return False

            if order_request.quantity <= 0:
                self.logger.error(invalid("Quantity must be positive"))
                return False

            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order_request.price is None or order_request.price <= 0:
                    self.logger.error(missing("Price is required for limit orders"))
                    return False

            if order_request.order_type in [OrderType.STOP_LIMIT, OrderType.STOP_MARKET]:
                if order_request.stop_price is None or order_request.stop_price <= 0:
                    self.logger.error(missing("Stop price is required for stop orders"))
                    return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Order validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order update"
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order cancellation"
    )
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if cancellation successful
        """
        try:
            if order_id not in self.active_orders:
                self.logger.error(missing(f"Order {order_id} not found"))
                return False

            order_state = self.active_orders[order_id]
            order_state.status = OrderStatus.CANCELLED
            order_state.updated_time = datetime.now()

            # Move to history
            self.order_history.append(order_state)
            del self.active_orders[order_id]

            # Update metrics
            self.metrics.increment_counter("orders_cancelled_total")

            self.logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Order cancellation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order fill processing"
    )
    async def process_fill(self, order_id: str, fill: OrderFill) -> Optional[OrderState]:
        """
        Process an order fill.

        Args:
            order_id: Order ID
            fill: Fill details

        Returns:
            OrderState: Updated order state or None if failed
        """
        try:
            if order_id not in self.active_orders:
                self.logger.error(missing(f"Order {order_id} not found"))
                return None

            order_state = self.active_orders[order_id]

            # Add fill to order
            order_state.fills.append(fill)

            # Update quantities
            order_state.executed_quantity += fill.quantity
            order_state.remaining_quantity = order_state.original_quantity - order_state.executed_quantity

            # Calculate average price
            total_value = sum(f.price * f.quantity for f in order_state.fills)
            order_state.average_price = total_value / order_state.executed_quantity

            # Update status
            if order_state.remaining_quantity <= 0:
                order_state.status = OrderStatus.FILLED
                # Move to history
                self.order_history.append(order_state)
                del self.active_orders[order_id]
            else:
                order_state.status = OrderStatus.PARTIALLY_FILLED

            order_state.updated_time = datetime.now()

            # Update metrics
            self.metrics.increment_counter("order_fills_total")

            self.logger.info(f"Processed fill for order {order_id}: {fill.quantity} @ {fill.price}")
            return order_state

        except Exception as e:
            self.logger.error(failed(f"❌ Fill processing failed: {e}"))
            return None
