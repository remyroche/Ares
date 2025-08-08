# src/interfaces/event_bus.py

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Union

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
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
    correlation_id: str | None = None


class EventBus:
    """
    Enhanced Event Bus component with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EventBus")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.event_bus_config: dict[str, Any] = self.config.get("event_bus", {})
        self.processing_interval: int = self.event_bus_config.get(
            "processing_interval",
            10,
        )
        self.max_history: int = self.event_bus_config.get("max_history", 100)
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid event bus configuration"),
            AttributeError: (False, "Missing required event bus parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="event bus initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Event Bus...")
            await self._load_event_bus_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for event bus")
                return False
            await self._initialize_event_processing()
            self.logger.info("✅ Event Bus initialization completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Event Bus initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="event bus configuration loading",
    )
    async def _load_event_bus_configuration(self) -> None:
        try:
            self.event_bus_config.setdefault("processing_interval", 10)
            self.event_bus_config.setdefault("max_history", 100)
            self.processing_interval = self.event_bus_config["processing_interval"]
            self.max_history = self.event_bus_config["max_history"]
            self.logger.info("Event bus configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading event bus configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.processing_interval <= 0:
                self.logger.error("Invalid processing interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event processing initialization",
    )
    async def _initialize_event_processing(self) -> None:
        try:
            # Initialize event processing components
            self.event_queue = asyncio.Queue()
            self.event_history = []
            self.logger.info("Event processing initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing event processing: {e}")

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Event bus run failed"),
        },
        default_return=False,
        context="event bus run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Event Bus started.")
            while self.is_running:
                await self._process_events()
                await asyncio.sleep(self.processing_interval)
            return True
        except Exception as e:
            self.logger.error(f"Error in event bus run: {e}")
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event processing",
    )
    async def _process_events(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)

            # Process events from queue
            while not self.event_queue.empty():
                event = await self.event_queue.get()
                await self._dispatch_event(event)

            self.logger.info(f"Event processing tick at {now}")
        except Exception as e:
            self.logger.error(f"Error in event processing: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event dispatch",
    )
    async def _dispatch_event(self, event: dict[str, Any]) -> None:
        try:
            event_type = event.get("type", "unknown")
            subscribers = self.subscribers.get(event_type, [])
            payload = event.get("data")

            for subscriber in subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        try:
                            await subscriber(payload)
                        except TypeError:
                            await subscriber()
                    else:
                        try:
                            subscriber(payload)
                        except TypeError:
                            subscriber()
                except Exception as e:
                    self.logger.error(
                        f"Error in event subscriber {getattr(subscriber, '__name__', str(subscriber))}: {e}",
                    )

            # Add to event history
            self.event_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "subscribers_count": len(subscribers),
                },
            )

            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)

            self.logger.info(
                f"Event '{event_type}' dispatched to {len(subscribers)} subscribers",
            )
        except Exception as e:
            self.logger.error(f"Error dispatching event: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event bus stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Event Bus...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Event Bus stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping event bus: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event subscription",
    )
    def subscribe(self, event_type: Union[EventType, str], callback: Callable) -> None:
        """Subscribe to an event type."""
        try:
            event_key = event_type.value if isinstance(event_type, EventType) else str(event_type)
            self.subscribers[event_key].append(callback)
            self.logger.info(f"Subscriber added for event type: {event_key}")
        except Exception as e:
            self.logger.error(f"Error subscribing to event: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event unsubscription",
    )
    def unsubscribe(self, event_type: Union[EventType, str], callback: Callable) -> None:
        """Unsubscribe from an event type."""
        try:
            event_key = event_type.value if isinstance(event_type, EventType) else str(event_type)
            if event_key in self.subscribers:
                self.subscribers[event_key] = [
                    sub for sub in self.subscribers[event_key] if sub != callback
                ]
                self.logger.info(f"Subscriber removed for event type: {event_key}")
        except Exception as e:
            self.logger.error(f"Error unsubscribing from event: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event publishing",
    )
    async def publish(self, event_type: Union[EventType, str], data: Any) -> None:
        """Publish an event to the bus."""
        try:
            event_key = event_type.value if isinstance(event_type, EventType) else str(event_type)
            event = {
                "type": event_key,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }
            await self.event_queue.put(event)
            self.logger.info(f"Event '{event_key}' published to queue")
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_event_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.event_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_subscribers(self) -> dict[str, list[Callable]]:
        return dict(self.subscribers)


event_bus: EventBus | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="event bus setup",
)
async def setup_event_bus(config: dict[str, Any] | None = None) -> EventBus | None:
    try:
        global event_bus
        if config is None:
            config = {"event_bus": {"processing_interval": 10, "max_history": 100}}
        event_bus = EventBus(config)
        success = await event_bus.initialize()
        if success:
            return event_bus
        return None
    except Exception as e:
        print(f"Error setting up event bus: {e}")
        return None
