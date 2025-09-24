"""
Event System for Live Trading

Provides an event-driven architecture for handling trading events,
order updates, data streams, and system notifications.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class EventType(Enum):
    """Event types for the trading system"""
    # Order events
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILLED = "order_partial_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Data events
    MARKET_DATA_UPDATE = "market_data_update"
    TICKER_UPDATE = "ticker_update"
    ORDER_BOOK_UPDATE = "order_book_update"

    # Position events
    POSITION_UPDATE = "position_update"
    POSITION_CLOSED = "position_closed"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"

    # Risk events
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    POSITION_SIZE_WARNING = "position_size_warning"
    DAILY_TRADE_LIMIT_EXCEEDED = "daily_trade_limit_exceeded"

    # Custom events
    CUSTOM_EVENT = "custom_event"


@dataclass
class Event:
    """Event structure"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    event_id: Optional[str] = None


@dataclass
class EventHandler:
    """Event handler configuration"""
    event_type: EventType
    handler: Callable
    priority: int = 1
    async_handler: bool = False
    filter_func: Optional[Callable] = None


class EventBus:
    """
    Event bus for handling trading system events.

    Provides:
    - Event publishing and subscription
    - Event filtering and routing
    - Priority-based event handling
    - Async and sync event handlers
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Event handlers storage
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False

        # Event history
        self.event_history: List[Event] = []
        self.max_history_size = 10000

        # Background tasks
        self.processor_task = None

    async def start(self) -> None:
        """Start the event bus."""
        if self.is_processing:
            self.logger.warning("EventBus is already running")
            return

        self.logger.info("Starting EventBus...")
        self.is_processing = True
        self.processor_task = asyncio.create_task(self._process_events())
        self.logger.info("✅ EventBus started successfully")

    async def stop(self) -> None:
        """Stop the event bus."""
        self.logger.info("Stopping EventBus...")
        self.is_processing = False

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        # Clear queues
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.logger.info("✅ EventBus stopped successfully")

    def subscribe(self, event_type: EventType, handler: Callable,
                 priority: int = 1, async_handler: bool = False,
                 filter_func: Optional[Callable] = None) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Handler function
            priority: Handler priority (higher = processed first)
            async_handler: Whether handler is async
            filter_func: Optional filter function

        Returns:
            Handler ID for unsubscribing
        """
        handler_id = f"handler_{event_type.value}_{id(handler)}_{datetime.now().timestamp()}"

        event_handler = EventHandler(
            event_type=event_type,
            handler=handler,
            priority=priority,
            async_handler=async_handler,
            filter_func=filter_func
        )

        self.handlers[event_type].append(event_handler)

        # Sort handlers by priority (highest first)
        self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)

        self.logger.info(f"✅ Subscribed handler {handler_id} to {event_type.value}")
        return handler_id

    def unsubscribe(self, event_type: EventType, handler: Callable) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler function to remove

        Returns:
            True if handler was removed
        """
        if event_type not in self.handlers:
            return False

        # Find and remove handler
        for i, event_handler in enumerate(self.handlers[event_type]):
            if event_handler.handler == handler:
                del self.handlers[event_type][i]
                self.logger.info(f"✅ Unsubscribed handler from {event_type.value}")
                return True

        return False

    async def publish(self, event_type: EventType, data: Dict[str, Any],
                     source: str = "system", priority: int = 1) -> str:
        """
        Publish an event.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            priority: Event priority

        Returns:
            Event ID
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source,
            priority=priority,
            event_id=f"event_{event_type.value}_{datetime.now().timestamp()}"
        )

        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]

        # Add to queue for processing
        await self.event_queue.put(event)

        self.logger.debug(f"📨 Published event: {event.event_type.value} from {event.source}")
        return event.event_id

    async def _process_events(self) -> None:
        """Background task to process events."""
        while self.is_processing:
            try:
                # Get event from queue
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self._handle_event(event)
                self.event_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error processing events: {e}")
                await asyncio.sleep(1)

    async def _handle_event(self, event: Event) -> None:
        """
        Handle an event by calling registered handlers.

        Args:
            event: Event to handle
        """
        try:
            handlers = self.handlers.get(event.event_type, [])

            for handler in handlers:
                # Check filter function
                if handler.filter_func and not handler.filter_func(event):
                    continue

                try:
                    if handler.async_handler:
                        # Async handler
                        await handler.handler(event)
                    else:
                        # Sync handler (run in executor)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, handler.handler, event)

                except Exception as e:
                    self.logger.error(f"❌ Error in event handler for {event.event_type.value}: {e}")

        except Exception as e:
            self.logger.error(f"❌ Error handling event {event.event_type.value}: {e}")

    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional event type filter
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        events = self.event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return most recent events first
        events.reverse()
        return events[:limit]

    def clear_event_history(self) -> None:
        """Clear event history."""
        self.event_history.clear()
        self.logger.info("Event history cleared")


# Convenience functions for common events
class TradingEventPublisher:
    """Helper class for publishing common trading events"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    async def publish_order_created(self, order_data: Dict[str, Any]) -> str:
        """Publish order created event."""
        return await self.event_bus.publish(
            EventType.ORDER_CREATED,
            order_data,
            source="order_manager"
        )

    async def publish_order_filled(self, order_data: Dict[str, Any]) -> str:
        """Publish order filled event."""
        return await self.event_bus.publish(
            EventType.ORDER_FILLED,
            order_data,
            source="exchange"
        )

    async def publish_market_data_update(self, market_data: Dict[str, Any]) -> str:
        """Publish market data update event."""
        return await self.event_bus.publish(
            EventType.MARKET_DATA_UPDATE,
            market_data,
            source="data_receiver"
        )

    async def publish_position_update(self, position_data: Dict[str, Any]) -> str:
        """Publish position update event."""
        return await self.event_bus.publish(
            EventType.POSITION_UPDATE,
            position_data,
            source="trade_executor"
        )

    async def publish_risk_warning(self, warning_data: Dict[str, Any]) -> str:
        """Publish risk warning event."""
        return await self.event_bus.publish(
            EventType.RISK_LIMIT_EXCEEDED,
            warning_data,
            source="risk_manager",
            priority=3
        )

    async def publish_system_startup(self, system_data: Dict[str, Any]) -> str:
        """Publish system startup event."""
        return await self.event_bus.publish(
            EventType.SYSTEM_STARTUP,
            system_data,
            source="system",
            priority=2
        )

    async def publish_system_shutdown(self, system_data: Dict[str, Any]) -> str:
        """Publish system shutdown event."""
        return await self.event_bus.publish(
            EventType.SYSTEM_SHUTDOWN,
            system_data,
            source="system",
            priority=2
        )


# Example event handlers
class EventHandlers:
    """Common event handlers for trading system"""

    @staticmethod
    def log_event_handler(event: Event):
        """Simple logging event handler."""
        logging.info(f"Event: {event.event_type.value} - {event.data}")

    @staticmethod
    async def async_log_event_handler(event: Event):
        """Async logging event handler."""
        logging.info(f"Async Event: {event.event_type.value} - {event.data}")

    @staticmethod
    def order_filter(event: Event) -> bool:
        """Filter for order events."""
        return event.event_type in [
            EventType.ORDER_CREATED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED
        ]

    @staticmethod
    def high_priority_filter(event: Event) -> bool:
        """Filter for high priority events."""
        return event.priority >= 3

    @staticmethod
    def risk_event_filter(event: Event) -> bool:
        """Filter for risk-related events."""
        return event.event_type in [
            EventType.RISK_LIMIT_EXCEEDED,
            EventType.POSITION_SIZE_WARNING,
            EventType.DAILY_TRADE_LIMIT_EXCEEDED
        ]