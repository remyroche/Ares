# src/tactician/enhanced_order_manager.py


"""
Enhanced Order Manager for Tactician
Handles sophisticated order management including stop-limit orders and leveraged limit orders
with partial fill management.
"""

# from src.utils.prometheus_metrics import metrics  # Temporarily commented due to syntax errors
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Removed handle_errors import - not used in this file
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    missing,
)
from copy import copy
import asyncio
from src.core.decorators import handles_errors

# Optional Prometheus metrics integration with safe fallback
try:  # pragma: no cover - optional dependency
    from src.utils.prometheus_metrics import metrics  # type: ignore
from src.core.decorators.errors import handles_errors
except Exception:  # pragma: no cover - metrics not available
    class _MetricsStub:
        def increment_counter(self, *_args, **_kwargs) -> None:
            return None

    metrics = _MetricsStub()  # type: ignore


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
    fills: list[OrderFill] = field(default_factory=list)
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
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the enhanced order manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedOrderManager")

        # Order tracking
        self.active_orders: dict[str, OrderState] = {}
        self.order_history: list[OrderState] = []

        # Configuration
        self.max_active_orders = config.get("max_active_orders", 100)
        self.max_history_size = config.get("max_history_size", 1000)
        self.default_timeout = config.get("order_timeout", 300)  # 5 minutes

        # Metrics (falls back to no-op stub if Prometheus metrics are unavailable)
        self.metrics = metrics

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the order manager.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Enhanced Order Manager...")

            # Clear any existing state
            self.active_orders.clear()
            self.order_history.clear()

            self.logger.info("✅ Enhanced Order Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(failed(f"❌ Enhanced Order Manager initialization failed: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order creation",
    )
    async def create_order(self, order_request: OrderRequest) -> OrderState | None:
        """
        Create a new order.

        Args:
            order_request: Order request details

        Returns:
            OrderState: Created order state or None if failed
        """
        try:
            # Validate order request
            if not self._validate_order_request(order_request):
                self.logger.error(invalid("Invalid order request"))
                return None

            # Create order state
            order_id = str(uuid.uuid4())
            order_state = OrderState(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                original_quantity=order_request.quantity,
                remaining_quantity=order_request.quantity,
                strategy_id=order_request.strategy_id,
                strategy_type=order_request.strategy_type,
            )

            # Add to active orders
            self.active_orders[order_id] = order_state

            # Update metrics
            self.metrics.increment_counter("orders_created_total")

            self.logger.info(f"Created order {order_id} for {order_request.symbol}")
            return order_state

        except Exception as e:
            self.logger.exception(failed(f"❌ Order creation failed: {e}"))
            return None

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
            self.logger.exception(failed(f"❌ Order validation failed: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order update",
    )
    async def update_order(self, order_id: str, updates: dict[str, Any]) -> OrderState | None:
        """
        Update an existing order.

        Args:
            order_id: Order ID to update
            updates: Dictionary of updates to apply

        Returns:
            OrderState: Updated order state or None if failed
        """
        try:
            if order_id not in self.active_orders:
                self.logger.error(missing(f"Order {order_id} not found"))
                return None

            order_state = self.active_orders[order_id]

            # Apply updates
            for key, value in updates.items():
                if hasattr(order_state, key):
                    setattr(order_state, key, value)

            order_state.updated_time = datetime.now()

            self.logger.info(f"Updated order {order_id}")
            return order_state

        except Exception as e:
            self.logger.exception(failed(f"❌ Order update failed: {e}"))
            return None

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order cancellation",
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
            self.logger.exception(failed(f"❌ Order cancellation failed: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="order fill processing",
    )
    async def process_fill(self, order_id: str, fill: OrderFill) -> OrderState | None:
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
            self.logger.exception(failed(f"❌ Fill processing failed: {e}"))
            return None

    def get_active_orders(self) -> dict[str, OrderState]:
        """
        Get all active orders.

        Returns:
            Dict[str, OrderState]: Active orders
        """
        return self.active_orders.copy()

    def get_order_history(self) -> list[OrderState]:
        """
        Get order history.

        Returns:
            List[OrderState]: Order history
        """
        return self.order_history.copy()

    def get_order(self, order_id: str) -> OrderState | None:
        """
        Get a specific order.

        Args:
            order_id: Order ID

        Returns:
            OrderState: Order state or None if not found
        """
        return self.active_orders.get(order_id)

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info("Cleaning up Enhanced Order Manager...")

            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)

            self.logger.info("✅ Enhanced Order Manager cleanup completed")

        except Exception as e:
            self.logger.exception(failed(f"❌ Enhanced Order Manager cleanup failed: {e}"))

    # ---- Lightweight helpers used by higher-level components ----

    @handles_errors(exceptions=(Exception,), default_return=None, context="chase micro breakout placement")
    async def place_chase_micro_breakout_order(
        self,
        *,
        symbol: str,
        side: "OrderSide",
        quantity: float,
        current_price: float | None = None,
        breakout_price: float | None = None,
        strategy_id: str | None = None,
        **_kwargs: dict[str, Any],
    ) -> "OrderState" | None:
        """Place a minimal STOP_LIMIT order used by CHASE_MICRO_BREAKOUT strategy."""
        stop_price = breakout_price or current_price or 0.0
        price = stop_price
        req = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LIMIT,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            strategy_id=strategy_id,
            strategy_type="CHASE_MICRO_BREAKOUT",
        )
        return await self.create_order(req)

    @handles_errors(exceptions=(Exception,), default_return=None, context="limit order return placement")
    async def place_limit_order_return(
        self,
        *,
        symbol: str,
        side: "OrderSide",
        quantity: float,
        price: float,
        leverage: float | None = None,
        strategy_id: str | None = None,
        **_kwargs: dict[str, Any],
    ) -> "OrderState" | None:
        """Place a leveraged LIMIT order used by LIMIT_ORDER_RETURN strategy."""
        req = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            leverage=leverage,
            strategy_id=strategy_id,
            strategy_type="LIMIT_ORDER_RETURN",
        )
        return await self.create_order(req)

    def get_order_status(self, order_id: str) -> "OrderState" | None:
        """Return current status for an order by id from active set or history."""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        for o in reversed(self.order_history):
            if o.order_id == order_id:
                return o
        return None

    def get_strategy_orders(self, strategy_id: str) -> list["OrderState"]:
        """Return all orders associated with a given strategy id (active + history)."""
        matches: list[OrderState] = []
        for o in self.active_orders.values():
            if o.strategy_id == strategy_id:
                matches.append(o)
        for o in self.order_history:
            if o.strategy_id == strategy_id:
                matches.append(o)
        return matches

    def get_performance_metrics(self) -> dict[str, Any]:
        """Summarize basic order manager performance metrics."""
        try:
            filled = sum(1 for o in self.order_history if o.status == OrderStatus.FILLED)
            cancelled = sum(1 for o in self.order_history if o.status == OrderStatus.CANCELLED)
            rejected = sum(1 for o in self.order_history if o.status == OrderStatus.REJECTED)
            return {
                "active_orders": len(self.active_orders),
                "history_orders": len(self.order_history),
                "filled": filled,
                "cancelled": cancelled,
                "rejected": rejected,
            }
        except Exception:
            return {}


@handles_errors(exceptions=(Exception,), default_return=None, context="enhanced order manager setup")
async def setup_enhanced_order_manager(
    config: dict[str, Any] | None = None,
) -> EnhancedOrderManager | None:
    """Factory to create and initialize an EnhancedOrderManager instance."""
    mgr = EnhancedOrderManager(config or {})
    ok = await mgr.initialize()
    return mgr if ok else None