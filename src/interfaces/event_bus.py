# src/interfaces/event_bus.py

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
)


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

    def print(self, message: str) -> None:
        """Lightweight print wrapper to ensure class uses logger and stdout consistently."""
        try:
            self.logger.info(message)
        finally:
            try:
                builtins_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print  # type: ignore
                builtins_print(message)
            except Exception:
                pass

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
                self.print(invalid("Invalid configuration for event bus"))
                return False
            await self._initialize_event_processing()
            self.logger.info("✅ Event Bus initialization completed successfully")
            return True
        except Exception:
            self.print(failed("❌ Event Bus initialization failed: {e}"))
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
        except Exception:
            self.print(error("Error loading event bus configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.processing_interval <= 0:
                self.print(invalid("Invalid processing interval"))
                return False
            if self.max_history <= 0:
                self.print(invalid("Invalid max history"))
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception:
            self.print(error("Error validating configuration: {e}"))
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
        except Exception:
            self.print(initialization_error("Error initializing event processing: {e}"))

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
            self.logger.info("🚀 Event Bus started")

            while self.is_running:
                await self._process_events()

            self.logger.info("🛑 Event Bus stopped")
            return True
        except Exception:
            self.print(error("Error in event bus run: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event processing",
    )
    async def _process_events(self) -> None:
        try:
            # Process events from queue
            while not self.event_queue.empty():
                event = await self.event_queue.get()
                await self._dispatch_event(event)
        except Exception:
            self.print(error("Error in event processing: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event dispatching",
    )
    async def _dispatch_event(self, event: Event) -> None:
        try:
            subscribers = self.subscribers.get(event.event_type.value, [])
            for callback in subscribers:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
        except Exception:
            self.print(error("Error dispatching event: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="event bus cleanup",
    )
    async def stop(self) -> None:
        try:
            self.is_running = False
            self.logger.info("🛑 Event Bus stopping...")
        except Exception:
            self.print(error("Error stopping event bus: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="event subscription",
    )
    async def subscribe(self, event_type: EventType, callback: Callable) -> bool:
        try:
            self.subscribers[event_type.value].append(callback)
            return True
        except Exception:
            self.print(error("Error subscribing to event: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="event unsubscription",
    )
    async def unsubscribe(self, event_type: EventType, callback: Callable) -> bool:
        try:
            if callback in self.subscribers[event_type.value]:
                self.subscribers[event_type.value].remove(callback)
            return True
        except Exception:
            self.print(error("Error unsubscribing from event: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="event publishing",
    )
    async def publish(self, event: Event) -> bool:
        try:
            await self.event_queue.put(event)
            return True
        except Exception:
            self.print(error("Error publishing event: {e}"))
            return False


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
