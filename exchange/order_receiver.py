"""
Exchange Order Receiver

Exchange-agnostic receiver that handles incoming orders and routes them
to the appropriate centralized exchange (CEX).
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.interfaces.base_interfaces import TradeDecision, MarketData
from .base_exchange import BaseExchange
from .factory import ExchangeFactory


class ReceiverStatus(Enum):
    """Receiver status enumeration"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    ERROR = "error"


@dataclass
class OrderRequest:
    """Order request structure"""
    id: str
    trade_decision: TradeDecision
    exchange_name: str
    timestamp: datetime
    priority: int = 1  # 1 = normal, 2 = high, 3 = critical
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class DataRequest:
    """Data request structure"""
    id: str
    symbol: str
    interval: str
    limit: int
    exchange_name: str
    timestamp: datetime


class ExchangeOrderReceiver:
    """
    Exchange-agnostic receiver for handling orders and data requests.

    This class acts as a central hub that:
    - Receives orders from various sources
    - Routes orders to appropriate exchanges
    - Handles data requests
    - Manages exchange connections
    - Provides load balancing and failover
    """

    def __init__(self, supported_exchanges: List[str]):
        self.supported_exchanges = supported_exchanges
        self.logger = logging.getLogger(__name__)

        # Exchange connections
        self.exchanges: Dict[str, BaseExchange] = {}
        self.exchange_status: Dict[str, ReceiverStatus] = {}

        # Order handling
        self.order_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_orders: Dict[str, OrderRequest] = {}
        self.completed_orders: Dict[str, OrderRequest] = {}

        # Data handling
        self.data_requests: asyncio.Queue = asyncio.Queue()
        self.data_subscriptions: Dict[str, Dict[str, Any]] = {}

        # Background tasks
        self.order_processor_task = None
        self.data_processor_task = None
        self.status_monitor_task = None
        self.is_running = False

        # Event callbacks
        self.on_order_received: Optional[Callable] = None
        self.on_order_executed: Optional[Callable] = None
        self.on_data_received: Optional[Callable] = None
        self.on_exchange_status_change: Optional[Callable] = None

        # Configuration
        self.max_concurrent_orders = 10
        self.order_timeout = 30  # seconds
        self.retry_delay = 1.0  # seconds
        self.max_retries = 3

    async def start(self) -> bool:
        """Start the exchange receiver."""
        try:
            if self.is_running:
                self.logger.warning("ExchangeOrderReceiver is already running")
                return True

            self.logger.info("Starting ExchangeOrderReceiver...")

            # Initialize exchange connections
            await self._initialize_exchanges()

            # Start background processors
            self.order_processor_task = asyncio.create_task(self._process_orders())
            self.data_processor_task = asyncio.create_task(self._process_data_requests())
            self.status_monitor_task = asyncio.create_task(self._monitor_exchange_status())

            self.is_running = True
            self.logger.info("✅ ExchangeOrderReceiver started successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start ExchangeOrderReceiver: {e}")
            return False

    async def stop(self) -> None:
        """Stop the exchange receiver."""
        try:
            self.logger.info("Stopping ExchangeOrderReceiver...")

            self.is_running = False

            # Cancel background tasks
            if self.order_processor_task:
                self.order_processor_task.cancel()
                try:
                    await self.order_processor_task
                except asyncio.CancelledError:
                    pass

            if self.data_processor_task:
                self.data_processor_task.cancel()
                try:
                    await self.data_processor_task
                except asyncio.CancelledError:
                    pass

            if self.status_monitor_task:
                self.status_monitor_task.cancel()
                try:
                    await self.status_monitor_task
                except asyncio.CancelledError:
                    pass

            # Close exchange connections
            for exchange in self.exchanges.values():
                await exchange.close()

            self.exchanges.clear()
            self.exchange_status.clear()

            self.logger.info("✅ ExchangeOrderReceiver stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping ExchangeOrderReceiver: {e}")

    async def submit_order(self, trade_decision: TradeDecision, exchange_name: str,
                          priority: int = 1) -> Optional[str]:
        """
        Submit an order for execution.

        Args:
            trade_decision: The trade decision to execute
            exchange_name: Target exchange name
            priority: Order priority (1=normal, 2=high, 3=critical)

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Validate exchange
            if exchange_name not in self.supported_exchanges:
                self.logger.error(f"Unsupported exchange: {exchange_name}")
                return None

            # Create order request
            order_request = OrderRequest(
                id=f"order_{datetime.now().timestamp()}_{id(trade_decision)}",
                trade_decision=trade_decision,
                exchange_name=exchange_name,
                timestamp=datetime.now(),
                priority=priority
            )

            # Add to queue
            await self.order_queue.put((priority, order_request.id, order_request))
            self.active_orders[order_request.id] = order_request

            # Notify callback
            if self.on_order_received:
                await self.on_order_received(order_request)

            self.logger.info(f"✅ Order submitted: {order_request.id} for {exchange_name}")
            return order_request.id

        except Exception as e:
            self.logger.error(f"❌ Error submitting order: {e}")
            return None

    async def request_data(self, symbol: str, interval: str = "1m",
                          limit: int = 100, exchange_name: str = "auto") -> Optional[List[MarketData]]:
        """
        Request market data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            limit: Number of data points
            exchange_name: Exchange name or "auto" for automatic selection

        Returns:
            List of MarketData objects or None if failed
        """
        try:
            # Determine target exchange
            target_exchange = self._select_best_exchange(exchange_name)

            if not target_exchange:
                self.logger.error("No suitable exchange available")
                return None

            # Create data request
            data_request = DataRequest(
                id=f"data_{datetime.now().timestamp()}_{id(symbol)}",
                symbol=symbol,
                interval=interval,
                limit=limit,
                exchange_name=target_exchange,
                timestamp=datetime.now()
            )

            # Process immediately (or queue for background processing)
            data = await self._process_data_request(data_request)

            if data and self.on_data_received:
                await self.on_data_received(data_request, data)

            return data

        except Exception as e:
            self.logger.error(f"❌ Error requesting data: {e}")
            return None

    async def subscribe_data(self, symbol: str, interval: str = "1m",
                           callback: Optional[Callable] = None, exchange_name: str = "auto") -> Optional[str]:
        """
        Subscribe to real-time data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            callback: Optional callback for data updates
            exchange_name: Exchange name or "auto" for automatic selection

        Returns:
            Subscription ID if successful, None otherwise
        """
        try:
            # Determine target exchange
            target_exchange = self._select_best_exchange(exchange_name)

            if not target_exchange:
                self.logger.error("No suitable exchange available")
                return None

            subscription_id = f"sub_{symbol}_{interval}_{datetime.now().timestamp()}"

            self.data_subscriptions[subscription_id] = {
                'symbol': symbol,
                'interval': interval,
                'exchange': target_exchange,
                'callback': callback,
                'is_active': True
            }

            self.logger.info(f"✅ Data subscription created: {subscription_id}")
            return subscription_id

        except Exception as e:
            self.logger.error(f"❌ Error subscribing to data: {e}")
            return None

    async def unsubscribe_data(self, subscription_id: str) -> bool:
        """
        Unsubscribe from data.

        Args:
            subscription_id: Subscription ID to cancel

        Returns:
            True if successful
        """
        try:
            if subscription_id in self.data_subscriptions:
                self.data_subscriptions[subscription_id]['is_active'] = False
                del self.data_subscriptions[subscription_id]
                self.logger.info(f"✅ Data subscription cancelled: {subscription_id}")
                return True
            else:
                self.logger.warning(f"Subscription not found: {subscription_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error unsubscribing from data: {e}")
            return False

    async def get_exchange_status(self, exchange_name: str) -> Optional[ReceiverStatus]:
        """Get the status of an exchange."""
        return self.exchange_status.get(exchange_name)

    async def get_all_exchange_status(self) -> Dict[str, ReceiverStatus]:
        """Get status of all exchanges."""
        return self.exchange_status.copy()

    async def get_order_status(self, order_id: str) -> Optional[OrderRequest]:
        """Get the status of an order."""
        return self.active_orders.get(order_id) or self.completed_orders.get(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            order = self.active_orders.get(order_id)
            if not order:
                self.logger.warning(f"Order not found: {order_id}")
                return False

            # Remove from queue if still there
            # Note: This is a simplified implementation
            # In practice, you'd need more sophisticated queue management

            # Mark as cancelled
            order.trade_decision.action = "CANCEL"
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]

            self.logger.info(f"✅ Order cancelled: {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error cancelling order: {e}")
            return False

    def _select_best_exchange(self, exchange_name: str) -> Optional[str]:
        """
        Select the best available exchange.

        Args:
            exchange_name: Preferred exchange name or "auto"

        Returns:
            Selected exchange name or None
        """
        try:
            if exchange_name != "auto" and exchange_name in self.supported_exchanges:
                # Check if preferred exchange is available
                if self.exchange_status.get(exchange_name) == ReceiverStatus.CONNECTED:
                    return exchange_name

            # Find the best available exchange
            for exchange in self.supported_exchanges:
                if self.exchange_status.get(exchange) == ReceiverStatus.CONNECTED:
                    return exchange

            # No exchanges available
            return None

        except Exception as e:
            self.logger.error(f"❌ Error selecting exchange: {e}")
            return None

    async def _initialize_exchanges(self) -> None:
        """Initialize all supported exchanges."""
        try:
            for exchange_name in self.supported_exchanges:
                try:
                    self.logger.info(f"Initializing exchange: {exchange_name}")
                    exchange = ExchangeFactory.get_exchange(exchange_name)

                    # Test connection
                    await exchange._initialize_exchange()
                    await exchange.get_klines("BTCUSDT", "1m", 1)

                    self.exchanges[exchange_name] = exchange
                    self.exchange_status[exchange_name] = ReceiverStatus.CONNECTED

                    self.logger.info(f"✅ Exchange initialized: {exchange_name}")

                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
                    self.exchange_status[exchange_name] = ReceiverStatus.ERROR

        except Exception as e:
            self.logger.error(f"❌ Error initializing exchanges: {e}")

    async def _process_orders(self) -> None:
        """Background task to process orders."""
        while self.is_running:
            try:
                # Get order from queue
                try:
                    priority, order_id, order_request = await asyncio.wait_for(
                        self.order_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                self.logger.info(f"Processing order: {order_request.id}")

                # Process the order
                success = await self._execute_order_request(order_request)

                if success:
                    self.completed_orders[order_request.id] = order_request
                    del self.active_orders[order_request.id]

                    if self.on_order_executed:
                        await self.on_order_executed(order_request, True)
                else:
                    # Handle retry logic
                    order_request.retry_count += 1

                    if order_request.retry_count >= order_request.max_retries:
                        self.logger.error(f"Order failed after {order_request.retry_count} retries: {order_request.id}")
                        self.completed_orders[order_request.id] = order_request
                        del self.active_orders[order_request.id]

                        if self.on_order_executed:
                            await self.on_order_executed(order_request, False)
                    else:
                        # Re-queue for retry
                        await self.order_queue.put((order_request.priority, order_request.id, order_request))

                self.order_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error processing orders: {e}")
                await asyncio.sleep(1)

    async def _execute_order_request(self, order_request: OrderRequest) -> bool:
        """
        Execute an order request.

        Args:
            order_request: The order request to execute

        Returns:
            True if successful
        """
        try:
            exchange = self.exchanges.get(order_request.exchange_name)
            if not exchange:
                self.logger.error(f"Exchange not available: {order_request.exchange_name}")
                return False

            # Execute the order
            result = await exchange.create_order(
                symbol=order_request.trade_decision.symbol,
                side=order_request.trade_decision.action.lower(),
                quantity=order_request.trade_decision.quantity,
                price=order_request.trade_decision.price,
                order_type="MARKET" if order_request.trade_decision.price <= 0 else "LIMIT"
            )

            if result:
                self.logger.info(f"✅ Order executed: {order_request.id}")
                return True
            else:
                self.logger.error(f"❌ Order execution failed: {order_request.id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error executing order {order_request.id}: {e}")
            return False

    async def _process_data_requests(self) -> None:
        """Background task to process data requests."""
        while self.is_running:
            try:
                # Process data subscriptions
                for sub_id, subscription in list(self.data_subscriptions.items()):
                    if subscription['is_active']:
                        await self._update_data_subscription(subscription)

                await asyncio.sleep(1)  # Update every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error processing data requests: {e}")
                await asyncio.sleep(5)

    async def _process_data_request(self, data_request: DataRequest) -> Optional[List[MarketData]]:
        """
        Process a data request.

        Args:
            data_request: The data request to process

        Returns:
            List of MarketData objects or None
        """
        try:
            exchange = self.exchanges.get(data_request.exchange_name)
            if not exchange:
                self.logger.error(f"Exchange not available: {data_request.exchange_name}")
                return None

            data = await exchange.get_klines(
                data_request.symbol,
                data_request.interval,
                data_request.limit
            )

            return data

        except Exception as e:
            self.logger.error(f"❌ Error processing data request: {e}")
            return None

    async def _update_data_subscription(self, subscription: Dict[str, Any]) -> None:
        """
        Update a data subscription.

        Args:
            subscription: The subscription to update
        """
        try:
            exchange = self.exchanges.get(subscription['exchange'])
            if not exchange:
                return

            # Get latest data
            data = await exchange.get_klines(
                subscription['symbol'],
                subscription['interval'],
                1
            )

            if data and subscription['callback']:
                await subscription['callback'](data)

        except Exception as e:
            self.logger.error(f"❌ Error updating data subscription: {e}")

    async def _monitor_exchange_status(self) -> None:
        """Background task to monitor exchange status."""
        while self.is_running:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        # Test connection
                        await exchange.get_klines("BTCUSDT", "1m", 1)
                        if self.exchange_status[exchange_name] != ReceiverStatus.CONNECTED:
                            self.exchange_status[exchange_name] = ReceiverStatus.CONNECTED
                            if self.on_exchange_status_change:
                                await self.on_exchange_status_change(exchange_name, ReceiverStatus.CONNECTED)
                    except Exception:
                        if self.exchange_status[exchange_name] != ReceiverStatus.ERROR:
                            self.exchange_status[exchange_name] = ReceiverStatus.ERROR
                            if self.on_exchange_status_change:
                                await self.on_exchange_status_change(exchange_name, ReceiverStatus.ERROR)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error monitoring exchange status: {e}")
                await asyncio.sleep(60)

    # Configuration methods
    def set_order_received_callback(self, callback: Callable):
        """Set callback for order received events."""
        self.on_order_received = callback

    def set_order_executed_callback(self, callback: Callable):
        """Set callback for order executed events."""
        self.on_order_executed = callback

    def set_data_received_callback(self, callback: Callable):
        """Set callback for data received events."""
        self.on_data_received = callback

    def set_exchange_status_callback(self, callback: Callable):
        """Set callback for exchange status change events."""
        self.on_exchange_status_change = callback

    def set_max_concurrent_orders(self, max_orders: int):
        """Set maximum concurrent orders."""
        self.max_concurrent_orders = max_orders

    def set_order_timeout(self, timeout: int):
        """Set order timeout in seconds."""
        self.order_timeout = timeout