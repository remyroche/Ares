"""
Exchange Message Handler

Handles messages between the trading system and exchanges.
Provides routing, queuing, and processing capabilities.
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.utils.tprint import tprint


class MessageType(Enum):
    """Types of messages handled by the system"""
    ORDER = "order"
    DATA_REQUEST = "data_request"
    ACCOUNT_INFO = "account_info"
    POSITION_INFO = "position_info"
    CANCEL_ORDER = "cancel_order"
    HEARTBEAT = "heartbeat"
    SYSTEM_STATUS = "system_status"
    BATCH_ORDER = "batch_order"
    BROADCAST = "broadcast"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    IMMEDIATE = 5


@dataclass
class ExchangeMessage:
    """Base message structure for exchange communication"""
    id: str
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "trading_system"
    destination: str = "exchange"
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[float] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderMessage(ExchangeMessage):
    """Order-specific message"""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: Optional[float] = None
    exchange_specific_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRequestMessage(ExchangeMessage):
    """Data request message"""
    data_type: str = ""
    symbol: str = ""
    interval: Optional[str] = None
    limit: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class BatchOrderMessage(ExchangeMessage):
    """Batch order message for multiple orders"""
    orders: List[OrderMessage] = field(default_factory=list)
    execution_strategy: str = "parallel"  # "parallel", "sequential", "atomic"


@dataclass
class MessageResponse:
    """Response to a message"""
    message_id: str
    correlation_id: Optional[str] = None
    success: bool = True
    response_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0  # Time taken to process in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageQueue:
    """Priority queue for messages"""

    def __init__(self):
        self._queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        self._processing = False
        self._logger = logging.getLogger("MessageQueue")

    async def enqueue(self, message: ExchangeMessage) -> None:
        """Add a message to the appropriate priority queue"""
        if message.ttl and (datetime.now() - message.timestamp).total_seconds() > message.ttl:
            self._logger.warning(f"Message {message.id} has expired, not enqueuing")
            return

        queue = self._queues[message.priority]
        await queue.put(message)
        self._logger.debug(f"Enqueued message {message.id} with priority {message.priority.value}")

    async def dequeue(self, priority: MessagePriority = MessagePriority.NORMAL) -> Optional[ExchangeMessage]:
        """Remove and return the next message from the queue"""
        queue = self._queues[priority]

        try:
            message = await queue.get()
            self._logger.debug(f"Dequeued message {message.id}")
            return message
        except asyncio.QueueEmpty:
            return None

    async def peek(self, priority: MessagePriority = MessagePriority.NORMAL) -> Optional[ExchangeMessage]:
        """Peek at the next message without removing it"""
        queue = self._queues[priority]

        if queue.empty():
            return None

        # Since asyncio.Queue doesn't support peeking, we'll need to get and put back
        try:
            message = queue.get_nowait()
            await queue.put(message)
            return message
        except asyncio.QueueEmpty:
            return None

    def get_queue_size(self, priority: MessagePriority = MessagePriority.NORMAL) -> int:
        """Get the size of a specific priority queue"""
        return self._queues[priority].qsize()

    def get_all_queue_sizes(self) -> Dict[int, int]:
        """Get sizes of all priority queues"""
        return {priority.value: self.get_queue_size(priority) for priority in MessagePriority}


class ExchangeMessageHandler:
    """
    Handles messages between the trading system and exchanges.
    Provides routing, queuing, and processing capabilities.
    """

    def __init__(self, exchange_registry: Optional[Any] = None):
        self.logger = logging.getLogger("ExchangeMessageHandler")
        self.message_queue = MessageQueue()
        self.pending_messages: Dict[str, ExchangeMessage] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.response_handlers: Dict[str, Callable] = {}
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self.exchange_registry = exchange_registry
        self.default_target_exchanges = ["binance", "okx"]

    async def start(self) -> None:
        """Start the message handler"""
        tprint(f"🔧 ExchangeMessageHandler.start called", "INFO")
        if self._running:
            tprint(f"⚠️ Message handler already running", "WARNING")
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("Exchange message handler started")
        tprint(f"✅ Exchange message handler started successfully", "SUCCESS")

    async def stop(self) -> None:
        """Stop the message handler"""
        tprint(f"🔧 ExchangeMessageHandler.stop called", "INFO")
        if not self._running:
            tprint(f"⚠️ Message handler not running", "WARNING")
            return

        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Exchange message handler stopped")
        tprint(f"✅ Exchange message handler stopped successfully", "SUCCESS")

    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[ExchangeMessage, str], Awaitable[None]]
    ) -> None:
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async function that takes (message, exchange_name) and returns None
        """
        self.message_handlers[message_type].append(handler)
        self.logger.info(f"Registered handler for message type {message_type.value}")

    def register_response_handler(
        self,
        message_id: str,
        handler: Callable[[MessageResponse], Awaitable[None]]
    ) -> None:
        """
        Register a handler for a specific message response.

        Args:
            message_id: ID of the message to handle response for
            handler: Async function that takes MessageResponse and returns None
        """
        self.response_handlers[message_id] = handler
        self.logger.debug(f"Registered response handler for message {message_id}")

    async def send_message(
        self,
        message: ExchangeMessage,
        target_exchanges: List[str],
        wait_for_response: bool = True,
        timeout: float = 30.0
    ) -> Dict[str, MessageResponse]:
        """
        Send a message to target exchanges.

        Args:
            message: Message to send
            target_exchanges: List of exchange names to send to
            wait_for_response: Whether to wait for responses
            timeout: Timeout for responses in seconds

        Returns:
            Dictionary mapping exchange names to their responses
        """
        tprint(f"🔧 ExchangeMessageHandler.send_message called with message_id={message.id}, type={message.message_type.value}, targets={target_exchanges}", "INFO")
        if not target_exchanges:
            tprint(f"❌ No target exchanges specified", "ERROR")
            raise ValueError("No target exchanges specified")

        # Enqueue the message
        await self.message_queue.enqueue(message)
        self.pending_messages[message.id] = message

        responses: Dict[str, MessageResponse] = {}

        # Process the message for each target exchange
        for exchange_name in target_exchanges:
            try:
                # Route the message to the appropriate handler
                await self._route_message_to_exchange(message, exchange_name)

                if wait_for_response:
                    # Wait for response or timeout
                    response = await self._wait_for_response(message.id, exchange_name, timeout)
                    responses[exchange_name] = response
                else:
                    responses[exchange_name] = MessageResponse(
                        message_id=message.id,
                        success=True,
                        response_data={"status": "sent"},
                        timestamp=datetime.now()
                    )

            except Exception as e:
                self.logger.error(f"Error sending message to {exchange_name}: {e}")
                tprint(f"❌ Error sending message to {exchange_name}: {e}", "ERROR")
                responses[exchange_name] = MessageResponse(
                    message_id=message.id,
                    success=False,
                    error_message=str(e),
                    timestamp=datetime.now()
                )

        tprint(f"✅ Message sent to {len([r for r in responses.values() if r.success])}/{len(responses)} exchanges", "SUCCESS")
        return responses

    async def send_order_message(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        target_exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, MessageResponse]:
        """
        Send an order message to exchanges.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            order_type: Order type ("market", "limit", etc.)
            quantity: Order quantity
            price: Order price (optional)
            target_exchanges: Target exchanges (all if None)
            **kwargs: Additional order parameters

        Returns:
            Responses from exchanges
        """
        tprint(f"🔧 ExchangeMessageHandler.send_order_message called with symbol={symbol}, side={side}, quantity={quantity}", "INFO")
        order_message = OrderMessage(
            id=str(uuid.uuid4()),
            message_type=MessageType.ORDER,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            exchange_specific_params=kwargs
        )

        # If no target exchanges specified, use default target exchanges
        if not target_exchanges:
            target_exchanges = self.default_target_exchanges

        return await self.send_message(order_message, target_exchanges)

    async def send_data_request(
        self,
        data_type: str,
        symbol: str,
        target_exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, MessageResponse]:
        """
        Send a data request to exchanges.

        Args:
            data_type: Type of data requested
            symbol: Trading symbol
            target_exchanges: Target exchanges (all if None)
            **kwargs: Additional parameters

        Returns:
            Responses from exchanges
        """
        tprint(f"🔧 ExchangeMessageHandler.send_data_request called with data_type={data_type}, symbol={symbol}", "INFO")
        data_message = DataRequestMessage(
            id=str(uuid.uuid4()),
            message_type=MessageType.DATA_REQUEST,
            data_type=data_type,
            symbol=symbol,
            **kwargs
        )

        if not target_exchanges:
            target_exchanges = self.default_target_exchanges

        return await self.send_message(data_message, target_exchanges)

    async def send_batch_orders(
        self,
        orders: List[OrderMessage],
        execution_strategy: str = "parallel",
        target_exchanges: Optional[List[str]] = None
    ) -> Dict[str, MessageResponse]:
        """
        Send multiple orders as a batch.

        Args:
            orders: List of order messages
            execution_strategy: How to execute the batch
            target_exchanges: Target exchanges

        Returns:
            Responses from exchanges
        """
        tprint(f"🔧 ExchangeMessageHandler.send_batch_orders called with {len(orders)} orders, strategy={execution_strategy}", "INFO")
        batch_message = BatchOrderMessage(
            id=str(uuid.uuid4()),
            message_type=MessageType.BATCH_ORDER,
            orders=orders,
            execution_strategy=execution_strategy
        )

        if not target_exchanges:
            target_exchanges = self.default_target_exchanges

        return await self.send_message(batch_message, target_exchanges)

    async def _route_message_to_exchange(
        self,
        message: ExchangeMessage,
        exchange_name: str
    ) -> None:
        """
        Route a message to a specific exchange.

        Args:
            message: Message to route
            exchange_name: Target exchange name
        """
        # Get handlers for this message type
        handlers = self.message_handlers.get(message.message_type, [])

        for handler in handlers:
            try:
                await handler(message, exchange_name)
            except Exception as e:
                self.logger.error(f"Error in message handler for {message.message_type.value}: {e}")

    async def _wait_for_response(
        self,
        message_id: str,
        exchange_name: str,
        timeout: float
    ) -> MessageResponse:
        """
        Wait for a response to a message.

        Args:
            message_id: ID of the message waiting for response
            exchange_name: Name of the exchange
            timeout: Timeout in seconds

        Returns:
            Response from the exchange
        """
        response_key = f"{message_id}_{exchange_name}"

        # Create a future to wait for the response
        response_future: asyncio.Future[MessageResponse] = asyncio.Future()

        def response_callback(response: MessageResponse) -> None:
            if not response_future.done():
                response_future.set_result(response)

        # Register the callback
        self.response_handlers[response_key] = response_callback

        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)

            # Clean up
            self.response_handlers.pop(response_key, None)
            self.pending_messages.pop(message_id, None)

            return response

        except asyncio.TimeoutError:
            # Clean up on timeout
            self.response_handlers.pop(response_key, None)
            self.pending_messages.pop(message_id, None)

            return MessageResponse(
                message_id=message_id,
                success=False,
                error_message=f"Timeout waiting for response from {exchange_name}",
                timestamp=datetime.now()
            )

    async def _process_messages(self) -> None:
        """Process messages from the queue"""
        while self._running:
            try:
                # Process messages in priority order
                for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
                    message = await self.message_queue.dequeue(priority)
                    if message:
                        await self._process_single_message(message)
                        break  # Process one message at a time

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing messages: {e}")
                await asyncio.sleep(1)  # Back off on errors

    async def _process_single_message(self, message: ExchangeMessage) -> None:
        """
        Process a single message.

        Args:
            message: Message to process
        """
        try:
            self.logger.debug(f"Processing message {message.id} of type {message.message_type.value}")

            # Route to appropriate handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    await handler(message, "all")  # "all" for broadcast messages
                except Exception as e:
                    self.logger.error(f"Error in handler for {message.message_type.value}: {e}")

        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of message queues"""
        queue_sizes = self.message_queue.get_all_queue_sizes()
        return {
            "total_pending_messages": sum(queue_sizes.values()),
            "queue_sizes_by_priority": queue_sizes,
            "pending_messages_count": len(self.pending_messages),
            "active_handlers": len(self.message_handlers),
            "running": self._running
        }
    
    async def get_available_target_exchanges(self) -> List[str]:
        """Get list of available target exchanges"""
        if self.exchange_registry and hasattr(self.exchange_registry, 'get_active_exchanges'):
            try:
                active_exchanges = await self.exchange_registry.get_active_exchanges()
                # Filter to only include our default target exchanges
                return [ex for ex in active_exchanges if ex in self.default_target_exchanges]
            except Exception as e:
                self.logger.warning(f"Error getting active exchanges: {e}")
        
        return self.default_target_exchanges
    
    def set_default_target_exchanges(self, exchanges: List[str]) -> None:
        """Set the default target exchanges"""
        self.default_target_exchanges = exchanges
        self.logger.info(f"Set default target exchanges: {exchanges}")
    
    async def update_target_exchanges_from_registry(self) -> None:
        """Update target exchanges based on registry status"""
        if self.exchange_registry and hasattr(self.exchange_registry, 'get_active_exchanges'):
            try:
                active_exchanges = await self.exchange_registry.get_active_exchanges()
                # Update to only include active exchanges from our default list
                self.default_target_exchanges = [ex for ex in self.default_target_exchanges if ex in active_exchanges]
                self.logger.info(f"Updated target exchanges from registry: {self.default_target_exchanges}")
            except Exception as e:
                self.logger.warning(f"Error updating target exchanges from registry: {e}")

    async def clear_queue(self, priority: Optional[MessagePriority] = None) -> int:
        """
        Clear messages from the queue.

        Args:
            priority: Specific priority to clear, or all if None

        Returns:
            Number of messages cleared
        """
        tprint(f"🔧 ExchangeMessageHandler.clear_queue called with priority={priority.value if priority else 'all'}", "INFO")
        cleared_count = 0

        if priority:
            # Clear specific priority queue
            while True:
                message = await self.message_queue.dequeue(priority)
                if message:
                    cleared_count += 1
                else:
                    break
        else:
            # Clear all queues
            for p in MessagePriority:
                while True:
                    message = await self.message_queue.dequeue(p)
                    if message:
                        cleared_count += 1
                    else:
                        break

        self.logger.info(f"Cleared {cleared_count} messages from queue")
        tprint(f"✅ Cleared {cleared_count} messages from queue", "SUCCESS")
        return cleared_count


class MessageRouter:
    """
    Routes messages to appropriate exchanges based on various strategies.
    """

    def __init__(self, exchange_registry: Any):
        self.exchange_registry = exchange_registry
        self.logger = logging.getLogger("MessageRouter")
        self.routing_strategies: Dict[str, Callable] = {
            "round_robin": self._round_robin_routing,
            "least_loaded": self._least_loaded_routing,
            "broadcast": self._broadcast_routing,
            "primary_only": self._primary_only_routing,
            "failover": self._failover_routing
        }

    async def route_message(
        self,
        message: ExchangeMessage,
        strategy: str = "broadcast",
        **strategy_params
    ) -> Dict[str, MessageResponse]:
        """
        Route a message using the specified strategy.

        Args:
            message: Message to route
            strategy: Routing strategy to use
            **strategy_params: Parameters for the routing strategy

        Returns:
            Responses from exchanges
        """
        if strategy not in self.routing_strategies:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        routing_func = self.routing_strategies[strategy]
        target_exchanges = await routing_func(message, **strategy_params)

        # Create message handler and send message
        message_handler = ExchangeMessageHandler()

        return await message_handler.send_message(message, target_exchanges)

    async def _round_robin_routing(
        self,
        message: ExchangeMessage,
        **kwargs
    ) -> List[str]:
        """Route to exchanges in round-robin fashion"""
        available_exchanges = await self.exchange_registry.get_active_exchanges()
        if not available_exchanges:
            raise RuntimeError("No active exchanges available")

        # Simple round-robin: return first exchange for now
        # In a real implementation, this would track round-robin state
        return [available_exchanges[0]]

    async def _least_loaded_routing(
        self,
        message: ExchangeMessage,
        **kwargs
    ) -> List[str]:
        """Route to the least loaded exchange"""
        active_exchanges = await self.exchange_registry.get_active_exchanges()
        if not active_exchanges:
            raise RuntimeError("No active exchanges available")

        # Return the first active exchange (simplified)
        return [active_exchanges[0]]

    async def _broadcast_routing(
        self,
        message: ExchangeMessage,
        **kwargs
    ) -> List[str]:
        """Broadcast to all available exchanges"""
        return await self.exchange_registry.get_active_exchanges()

    async def _primary_only_routing(
        self,
        message: ExchangeMessage,
        primary_exchange: str = "binance",
        **kwargs
    ) -> List[str]:
        """Route only to primary exchange"""
        return [primary_exchange]

    async def _failover_routing(
        self,
        message: ExchangeMessage,
        primary_exchange: str = "binance",
        failover_exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Route with failover support"""
        if failover_exchanges is None:
            failover_exchanges = ["okx", "gateio"]

        all_exchanges = [primary_exchange] + [ex for ex in failover_exchanges if ex != primary_exchange]

        # Filter to only active exchanges
        active_exchanges = await self.exchange_registry.get_active_exchanges()
        return [ex for ex in all_exchanges if ex in active_exchanges]


class MessageBroker:
    """
    Central message broker for the trading system.
    Manages message routing, queuing, and delivery.
    """

    def __init__(self):
        self.logger = logging.getLogger("MessageBroker")
        self.message_handler = ExchangeMessageHandler()
        self.message_router = MessageRouter(None)  # Will be set by trading receiver
        self._started = False

    async def start(self) -> None:
        """Start the message broker"""
        await self.message_handler.start()
        self._started = True
        self.logger.info("Message broker started")

    async def stop(self) -> None:
        """Stop the message broker"""
        await self.message_handler.stop()
        self._started = False
        self.logger.info("Message broker stopped")

    async def publish(
        self,
        message: ExchangeMessage,
        routing_strategy: str = "broadcast",
        **routing_params
    ) -> Dict[str, MessageResponse]:
        """
        Publish a message using the specified routing strategy.

        Args:
            message: Message to publish
            routing_strategy: Strategy for routing the message
            **routing_params: Parameters for the routing strategy

        Returns:
            Responses from exchanges
        """
        return await self.message_router.route_message(message, routing_strategy, **routing_params)

    def get_status(self) -> Dict[str, Any]:
        """Get message broker status"""
        return {
            "started": self._started,
            "queue_status": asyncio.run(self.message_handler.get_queue_status())
        }