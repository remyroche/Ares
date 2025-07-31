# src/interfaces/event_bus.py

import asyncio
from typing import Dict, Any, List, Callable, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import weakref

from src.utils.logger import system_logger


class EventType(Enum):
    """Event types for the trading system"""

    MARKET_DATA_RECEIVED = "market_data_received"
    ANALYSIS_COMPLETED = "analysis_completed"
    STRATEGY_FORMULATED = "strategy_formulated"
    TRADE_DECISION_MADE = "trade_decision_made"
    TRADE_EXECUTED = "trade_executed"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_UPDATE = "performance_update"
    MODEL_UPDATED = "model_updated"
    SYSTEM_ERROR = "system_error"
    COMPONENT_STARTED = "component_started"
    COMPONENT_STOPPED = "component_stopped"


@dataclass
class Event:
    """Event structure"""

    event_type: EventType
    data: Any
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None


class EventBus:
    """
    Event bus for decoupled communication between trading components.
    Implements publish-subscribe pattern with async support.
    """

    def __init__(self):
        self.logger = system_logger.getChild("EventBus")
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    async def start(self):
        """Start the event bus"""
        self._running = True
        self.logger.info("Event bus started")

    async def stop(self):
        """Stop the event bus"""
        self._running = False
        self.logger.info("Event bus stopped")

    async def publish(
        self,
        event_type: EventType,
        data: Any,
        source: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Publish an event to all subscribers

        Args:
            event_type: Type of event
            data: Event data
            source: Source component name
            correlation_id: Optional correlation ID for tracking
        """
        if not self._running:
            self.logger.warning("Event bus not running, cannot publish event")
            return

        event = Event(
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source,
            correlation_id=correlation_id,
        )

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

        # Queue for async processing
        try:
            await self._event_queue.put(event)
        except asyncio.QueueFull:
            self.logger.warning("Event queue full, dropping event")

        self.logger.debug(f"Published event: {event_type.value} from {source}")

    async def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Subscribe to an event type

        Args:
            event_type: Event type to subscribe to
            callback: Async callback function to handle the event
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Use weakref to avoid memory leaks
        weak_callback = (
            weakref.WeakMethod(callback) if hasattr(callback, "__self__") else callback
        )
        self._subscribers[event_type].append(weak_callback)

        self.logger.debug(f"Subscribed to event: {event_type.value}")

    async def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Unsubscribe from an event type

        Args:
            event_type: Event type to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]
            self.logger.debug(f"Unsubscribed from event: {event_type.value}")

    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}", exc_info=True)

    async def _handle_event(self, event: Event):
        """Handle a single event"""
        if event.event_type in self._subscribers:
            callbacks = self._subscribers[event.event_type].copy()

            # Execute callbacks concurrently
            tasks = []
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        task = asyncio.create_task(callback(event))
                        tasks.append(task)
                    else:
                        # For non-async callbacks, run in executor
                        loop = asyncio.get_event_loop()
                        task = loop.run_in_executor(None, callback, event)
                        tasks.append(task)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}", exc_info=True)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def get_event_history(
        self, event_type: Optional[EventType] = None, limit: int = 100
    ) -> List[Event]:
        """Get event history"""
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type][
                -limit:
            ]
        return self._event_history[-limit:]

    async def clear_history(self):
        """Clear event history"""
        self._event_history.clear()
        self.logger.info("Event history cleared")

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type"""
        return len(self._subscribers.get(event_type, []))

    async def wait_for_event(
        self, event_type: EventType, timeout: float = 30.0
    ) -> Optional[Event]:
        """
        Wait for a specific event type

        Args:
            event_type: Event type to wait for
            timeout: Timeout in seconds

        Returns:
            Event if received, None if timeout
        """
        future = asyncio.Future()

        async def wait_callback(event: Event):
            if not future.done():
                future.set_result(event)

        await self.subscribe(event_type, wait_callback)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            await self.unsubscribe(event_type, wait_callback)
            return None
        finally:
            await self.unsubscribe(event_type, wait_callback)
