# src/interfaces/event_bus.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict


class EventType(Enum):
    """Event types for the trading system."""
    MARKET_DATA = "market_data"
    TRADE_SIGNAL = "trade_signal"
    ORDER_UPDATE = "order_update"
    SYSTEM_ALERT = "system_alert"
    MODEL_UPDATE = "model_update"
    PERFORMANCE_UPDATE = "performance_update"


@dataclass
class Event:
    """Event structure."""
    event_type: EventType
    data: Any
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None


class EventBus:
    """
    Enhanced Event Bus component with DI, type hints, and robust error handling.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.is_running: bool = False
        self.status: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.event_bus_config: Dict[str, Any] = self.config.get("event_bus", {})
        self.processing_interval: int = self.event_bus_config.get("processing_interval", 10)
        self.max_history: int = self.event_bus_config.get("max_history", 100)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize the event bus."""
        try:
            await self._load_event_bus_configuration()
            if not self._validate_configuration():
                return False
            await self._initialize_event_processing()
            return True
        except Exception as e:
            return False

    async def _load_event_bus_configuration(self) -> None:
        """Load event bus configuration."""
        self.event_bus_config.setdefault("processing_interval", 10)
        self.event_bus_config.setdefault("max_history", 100)
        self.processing_interval = self.event_bus_config["processing_interval"]
        self.max_history = self.event_bus_config["max_history"]

    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        if self.processing_interval <= 0:
            return False
        if self.max_history <= 0:
            return False
        return True

    async def _initialize_event_processing(self) -> None:
        """Initialize event processing."""
        self.is_running = True

    async def publish(self, event_type: str, data: Any, source: str = "system", correlation_id: Optional[str] = None) -> bool:
        """Publish an event."""
        try:
            event = Event(
                event_type=EventType(event_type),
                data=data,
                timestamp=datetime.now(),
                source=source,
                correlation_id=correlation_id
            )
            await self.event_queue.put(event)
            return True
        except Exception:
            return False

    async def subscribe(self, event_type: str, callback: Callable) -> bool:
        """Subscribe to an event type."""
        try:
            self.subscribers[event_type].append(callback)
            return True
        except Exception:
            return False

    async def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """Unsubscribe from an event type."""
        try:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(callback)
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        """Shutdown the event bus."""
        self.is_running = False
        # Clear the queue
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
