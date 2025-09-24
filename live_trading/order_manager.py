"""
Order Manager

Handles order placement, tracking, and routing to the appropriate exchange.
Provides exchange-agnostic order management capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

from src.interfaces.base_interfaces import TradeDecision
from exchange.base_exchange import BaseExchange
from dataclasses import dataclass
from typing import List


@dataclass
class TradingConfig:
    """Configuration for live trading operations"""
    exchange_name: str
    symbols: List[str]
    max_position_size: float
    max_daily_trades: int
    risk_per_trade: float
    enable_data_streaming: bool = True
    enable_order_execution: bool = True
    api_key: str = ""
    api_secret: str = ""


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILLED = "partial_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order:
    """Represents a trading order"""

    def __init__(self, trade_decision: TradeDecision):
        self.id = f"order_{datetime.now().timestamp()}"
        self.trade_decision = trade_decision
        self.status = OrderStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.exchange_order_id = None
        self.filled_quantity = 0.0
        self.remaining_quantity = trade_decision.quantity
        self.execution_price = 0.0
        self.fees = 0.0
        self.error_message = None

    def update_status(self, status: OrderStatus, exchange_order_id: Optional[str] = None,
                     filled_quantity: float = 0.0, execution_price: float = 0.0,
                     fees: float = 0.0, error_message: Optional[str] = None):
        """Update order status and details."""
        self.status = status
        self.updated_at = datetime.now()

        if exchange_order_id:
            self.exchange_order_id = exchange_order_id

        if filled_quantity > 0:
            self.filled_quantity = filled_quantity
            self.remaining_quantity = self.trade_decision.quantity - filled_quantity

        if execution_price > 0:
            self.execution_price = execution_price

        if fees > 0:
            self.fees = fees

        if error_message:
            self.error_message = error_message


class OrderManager:
    """
    Manages trading orders and provides exchange-agnostic order handling.

    This class is responsible for:
    - Order placement and routing
    - Order status tracking
    - Order validation
    - Risk management checks
    """

    def __init__(self, exchange: BaseExchange, config: TradingConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Order storage
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}

        # Order tracking
        self.order_counter = 0
        self.is_running = False

        # Event callbacks
        self.on_order_update: Optional[Callable] = None

        # Background tasks
        self.monitor_task = None
        self.retry_task = None

        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.order_timeout = 30.0  # seconds

    async def start(self) -> None:
        """Start the order manager."""
        try:
            if self.is_running:
                self.logger.warning("OrderManager is already running")
                return

            self.logger.info("Starting OrderManager...")

            # Start background monitoring
            self.monitor_task = asyncio.create_task(self._monitor_orders())
            self.retry_task = asyncio.create_task(self._retry_failed_orders())

            self.is_running = True
            self.logger.info("✅ OrderManager started successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to start OrderManager: {e}")
            raise

    async def stop(self) -> None:
        """Stop the order manager."""
        try:
            self.logger.info("Stopping OrderManager...")

            self.is_running = False

            # Cancel background tasks
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            if self.retry_task:
                self.retry_task.cancel()
                try:
                    await self.retry_task
                except asyncio.CancelledError:
                    pass

            # Cancel all active orders
            await self._cancel_all_orders()

            self.logger.info("✅ OrderManager stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping OrderManager: {e}")

    async def place_order(self, trade_decision: TradeDecision) -> Optional[Order]:
        """
        Place a new trading order.

        Args:
            trade_decision: The trade decision to execute

        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Validate order
            validation_result = self._validate_order(trade_decision)
            if not validation_result['valid']:
                self.logger.error(f"Order validation failed: {validation_result['message']}")
                return None

            # Create order object
            order = Order(trade_decision)
            order.id = f"order_{self.order_counter}_{datetime.now().timestamp()}"
            self.order_counter += 1

            # Store order
            self.active_orders[order.id] = order

            self.logger.info(f"Created order: {order.id} for {trade_decision.symbol} {trade_decision.action}")

            # Execute order
            success = await self._execute_order(order)

            if success:
                self.logger.info(f"✅ Order placed successfully: {order.id}")
                return order
            else:
                # Mark as failed
                order.update_status(
                    OrderStatus.REJECTED,
                    error_message="Failed to execute order"
                )
                self.logger.error(f"❌ Failed to execute order: {order.id}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: Any) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID (can be internal or exchange ID)

        Returns:
            True if cancelled successfully
        """
        try:
            # Find order
            order = None
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
            else:
                # Try to find by exchange order ID
                for o in self.active_orders.values():
                    if o.exchange_order_id == str(order_id):
                        order = o
                        break

            if not order:
                self.logger.warning(f"Order not found: {order_id}")
                return False

            # Cancel on exchange
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
                result = await self.exchange.cancel_order(symbol, order.exchange_order_id or order_id)

                if result:
                    order.update_status(OrderStatus.CANCELLED)
                    self.logger.info(f"✅ Order cancelled: {order.id}")
                    return True
                else:
                    self.logger.error(f"❌ Failed to cancel order: {order.id}")
                    return False
            else:
                self.logger.warning(f"Cannot cancel order {order.id} with status {order.status}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error cancelling order: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get order status.

        Args:
            order_id: Internal order ID

        Returns:
            Order object if found, None otherwise
        """
        return self.active_orders.get(order_id) or self.completed_orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        open_orders = []

        for order in self.active_orders.values():
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
                if not symbol or order.trade_decision.symbol == symbol:
                    open_orders.append(order)

        return open_orders

    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history."""
        orders = list(self.completed_orders.values())

        # Filter by symbol if specified
        if symbol:
            orders = [o for o in orders if o.trade_decision.symbol == symbol]

        # Sort by creation time (newest first)
        orders.sort(key=lambda x: x.created_at, reverse=True)

        return orders[:limit]

    def _validate_order(self, trade_decision: TradeDecision) -> Dict[str, Any]:
        """
        Validate a trade decision.

        Args:
            trade_decision: The trade decision to validate

        Returns:
            Validation result dictionary
        """
        try:
            # Basic validation
            if not trade_decision.symbol:
                return {'valid': False, 'message': 'Symbol is required'}

            if trade_decision.quantity <= 0:
                return {'valid': False, 'message': 'Quantity must be positive'}

            if trade_decision.price < 0:
                return {'valid': False, 'message': 'Price cannot be negative'}

            if trade_decision.action not in ['BUY', 'SELL']:
                return {'valid': False, 'message': 'Invalid action (must be BUY or SELL)'}

            # Symbol-specific validation
            if trade_decision.symbol not in self.config.symbols:
                return {'valid': False, 'message': f'Symbol {trade_decision.symbol} not in allowed symbols'}

            # Risk validation
            risk_check = self._validate_risk_parameters(trade_decision)
            if not risk_check['valid']:
                return risk_check

            return {'valid': True, 'message': 'Order is valid'}

        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {str(e)}'}

    def _validate_risk_parameters(self, trade_decision: TradeDecision) -> Dict[str, Any]:
        """
        Validate risk parameters for a trade decision.

        Args:
            trade_decision: The trade decision to validate

        Returns:
            Validation result dictionary
        """
        try:
            # Check position size limits
            position_value = trade_decision.quantity * trade_decision.price

            if position_value > self.config.max_position_size:
                return {
                    'valid': False,
                    'message': f'Position value {position_value} exceeds maximum {self.config.max_position_size}'
                }

            # Check risk per trade
            if position_value * self.config.risk_per_trade > self.config.max_position_size:
                return {
                    'valid': False,
                    'message': f'Trade risk exceeds position size limit'
                }

            return {'valid': True, 'message': 'Risk parameters are valid'}

        except Exception as e:
            return {'valid': False, 'message': f'Risk validation error: {str(e)}'}

    async def _execute_order(self, order: Order) -> bool:
        """
        Execute an order on the exchange.

        Args:
            order: The order to execute

        Returns:
            True if successful
        """
        try:
            # Convert trade decision to exchange format
            exchange_order = await self.exchange.create_order(
                symbol=order.trade_decision.symbol,
                side=order.trade_decision.action.lower(),
                quantity=order.trade_decision.quantity,
                price=order.trade_decision.price if order.trade_decision.price > 0 else None,
                order_type="MARKET" if order.trade_decision.price <= 0 else "LIMIT"
            )

            if exchange_order:
                order.update_status(
                    OrderStatus.PENDING,
                    exchange_order_id=exchange_order.get('orderId'),
                    execution_price=exchange_order.get('avgPrice', 0)
                )

                # Notify callback
                if self.on_order_update:
                    await self.on_order_update({
                        'order_id': order.id,
                        'status': order.status.value,
                        'symbol': order.trade_decision.symbol,
                        'action': order.trade_decision.action,
                        'quantity': order.trade_decision.quantity,
                        'price': order.trade_decision.price,
                        'exchange_order_id': order.exchange_order_id
                    })

                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"❌ Error executing order {order.id}: {e}")
            return False

    async def _monitor_orders(self) -> None:
        """Background task to monitor order status."""
        while self.is_running:
            try:
                # Check each active order
                for order_id, order in list(self.active_orders.items()):
                    if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
                        await self._check_order_status(order)

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in order monitoring: {e}")
                await asyncio.sleep(5)

    async def _check_order_status(self, order: Order) -> None:
        """
        Check the status of an order on the exchange.

        Args:
            order: The order to check
        """
        try:
            # Get order status from exchange
            if order.exchange_order_id:
                status_result = await self.exchange.get_order_status(
                    order.trade_decision.symbol,
                    order.exchange_order_id
                )

                if status_result:
                    # Update order based on exchange status
                    exchange_status = status_result.get('status', '').upper()
                    filled_qty = float(status_result.get('executedQty', 0))
                    avg_price = float(status_result.get('avgPrice', 0))

                    if exchange_status == 'FILLED':
                        order.update_status(
                            OrderStatus.FILLED,
                            filled_quantity=filled_qty,
                            execution_price=avg_price
                        )
                        # Move to completed orders
                        self.completed_orders[order.id] = order
                        del self.active_orders[order.id]

                    elif exchange_status == 'PARTIAL_FILLED':
                        order.update_status(
                            OrderStatus.PARTIAL_FILLED,
                            filled_quantity=filled_qty,
                            execution_price=avg_price
                        )

                    elif exchange_status in ['CANCELLED', 'CANCELED']:
                        order.update_status(OrderStatus.CANCELLED)
                        self.completed_orders[order.id] = order
                        del self.active_orders[order.id]

                    elif exchange_status == 'REJECTED':
                        order.update_status(OrderStatus.REJECTED)
                        self.completed_orders[order.id] = order
                        del self.active_orders[order.id]

                    # Notify callback
                    if self.on_order_update:
                        await self.on_order_update({
                            'order_id': order.id,
                            'status': order.status.value,
                            'symbol': order.trade_decision.symbol,
                            'exchange_order_id': order.exchange_order_id,
                            'filled_quantity': order.filled_quantity,
                            'execution_price': order.execution_price
                        })

        except Exception as e:
            self.logger.error(f"❌ Error checking order status {order.id}: {e}")

    async def _retry_failed_orders(self) -> None:
        """Background task to retry failed orders."""
        while self.is_running:
            try:
                # Retry failed orders (max 3 retries)
                for order_id, order in list(self.active_orders.items()):
                    if (order.status == OrderStatus.REJECTED and
                        hasattr(order, 'retry_count') and
                        order.retry_count < self.max_retries):

                        order.retry_count = getattr(order, 'retry_count', 0) + 1
                        self.logger.info(f"Retrying order {order.id} (attempt {order.retry_count})")

                        # Wait before retry
                        await asyncio.sleep(self.retry_delay * order.retry_count)

                        # Retry execution
                        await self._execute_order(order)

                await asyncio.sleep(10)  # Retry every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in retry task: {e}")
                await asyncio.sleep(5)

    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        try:
            open_orders = await self.get_open_orders()
            for order in open_orders:
                await self.cancel_order(order.trade_decision.symbol, order.id)

        except Exception as e:
            self.logger.error(f"❌ Error cancelling all orders: {e}")

    # Configuration methods
    def set_order_update_callback(self, callback: Callable):
        """Set callback for order updates."""
        self.on_order_update = callback