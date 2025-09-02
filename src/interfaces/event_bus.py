# src/interfaces/event_bus.py

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio
import json
import uuid

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
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
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id
        }

    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        event_type = EventType(data["event_type"])
        timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            event_type=event_type,
            data=data["data"],
            timestamp=timestamp,
            source=data["source"],
            correlation_id=data.get("correlation_id")
        )


class EventBus:
    """
    Enhanced Event Bus component with DI, type hints, and robust error handling.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("EventBus")
        self.is_running: bool = False
        self.status: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Event bus configuration
        self.event_bus_config: Dict[str, Any] = self.config.get("event_bus", {})
        self.processing_interval: int = self.event_bus_config.get("processing_interval", 10)
        self.max_history: int = self.event_bus_config.get("max_history", 100)
        
        # Event handling
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Dict[str, Any]] = []
        
        # Statistics and monitoring
        self.stats: Dict[str, Any] = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "subscribers_active": 0
        }

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
        """Initialize the event bus"""
        try:
            self.logger.info("Initializing Event Bus...")
            
            # Load event bus configuration
            await self._load_event_bus_configuration()
            
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for event bus"))
                return False
            
            # Initialize event processing
            await self._initialize_event_processing()
            
            self.is_running = True
            self.logger.info("✅ Event Bus initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Event Bus: {e}")
            return False

    async def _load_event_bus_configuration(self) -> None:
        """Load event bus configuration"""
        try:
            # Set default values if not present
            self.event_bus_config.setdefault("processing_interval", 10)
            self.event_bus_config.setdefault("max_history", 100)
            
            self.processing_interval = self.event_bus_config["processing_interval"]
            self.max_history = self.event_bus_config["max_history"]
            
            self.logger.info("Event bus configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading event bus configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Validate event bus configuration"""
        try:
            if self.processing_interval <= 0:
                self.logger.error("Invalid processing interval")
                return False
            
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            
            self.logger.info("Event bus configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    async def _initialize_event_processing(self) -> None:
        """Initialize event processing infrastructure"""
        try:
            # Start event processing task
            asyncio.create_task(self._event_processor())
            self.logger.info("Event processing task started")
            
        except Exception as e:
            self.logger.error(f"Error initializing event processing: {e}")
            raise

    async def _event_processor(self) -> None:
        """Main event processing loop"""
        while self.is_running:
            try:
                # Process events from queue
                if not self.event_queue.empty():
                    event = await self.event_queue.get()
                    await self._process_event(event)
                    self.event_queue.task_done()
                
                # Update statistics
                self.stats["queue_size"] = self.event_queue.qsize()
                self.stats["subscribers_active"] = sum(len(subs) for subs in self.subscribers.values())
                
                await asyncio.sleep(self.processing_interval / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1.0)

    async def _process_event(self, event: Event) -> None:
        """Process a single event"""
        try:
            self.logger.debug(f"Processing event: {event.event_type.value}")
            
            # Find subscribers for this event type
            event_type_str = event.event_type.value
            if event_type_str in self.subscribers:
                subscribers = self.subscribers[event_type_str]
                
                # Execute callbacks for all subscribers
                for subscriber in subscribers:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(event)
                        else:
                            subscriber(event)
                    except Exception as e:
                        self.logger.error(f"Error in subscriber callback: {e}")
                        self.stats["events_failed"] += 1
            
            # Add to history
            self.event_history.append(event.to_dict())
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
            
            self.stats["events_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            self.stats["events_failed"] += 1

    async def publish(self, event_type: EventType, data: Any, source: str, 
                     correlation_id: Optional[str] = None) -> str:
        """Publish an event to the bus"""
        try:
            if not self.is_running:
                self.logger.error("Event bus is not running")
                return ""
            
            # Create event
            event = Event(
                event_type=event_type,
                data=data,
                timestamp=datetime.now(),
                source=source,
                correlation_id=correlation_id
            )
            
            # Add to processing queue
            await self.event_queue.put(event)
            
            self.stats["events_published"] += 1
            self.logger.debug(f"Event published: {event_type.value}")
            
            return str(uuid.uuid4())  # Return event ID
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            return ""

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to events of a specific type"""
        try:
            event_type_str = event_type.value
            self.subscribers[event_type_str].append(callback)
            
            self.logger.info(f"Subscriber added for event type: {event_type_str}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to event: {e}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe from events of a specific type"""
        try:
            event_type_str = event_type.value
            if event_type_str in self.subscribers:
                self.subscribers[event_type_str] = [
                    sub for sub in self.subscribers[event_type_str] 
                    if sub != callback
                ]
                
                self.logger.info(f"Subscriber removed for event type: {event_type_str}")
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from event: {e}")

    async def run(self) -> bool:
        """Run the event bus"""
        try:
            self.is_running = True
            self.logger.info("🚦 Event Bus started")
            
            while self.is_running:
                await self._process_events()
                await asyncio.sleep(self.processing_interval)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in event bus run: {e}")
            self.is_running = False
            return False

    async def _process_events(self) -> None:
        """Process events from the queue"""
        try:
            now = datetime.now()
            self.status = {"timestamp": now.isoformat(), "status": "running"}
            
            # Update history
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Process events from queue
            events_processed = 0
            while not self.event_queue.empty():
                event = await self.event_queue.get()
                await self._process_event(event)
                events_processed += 1
            
            if events_processed > 0:
                self.logger.debug(f"Processed {events_processed} events")
                
        except Exception as e:
            self.logger.error(f"Error in event processing: {e}")

    async def stop(self) -> None:
        """Stop the event bus"""
        try:
            self.logger.info("🛑 Stopping Event Bus...")
            self.is_running = False
            
            # Wait for event queue to empty
            await self.event_queue.join()
            
            self.status = {
                "timestamp": datetime.now().isoformat(),
                "status": "stopped"
            }
            
            self.logger.info("✅ Event Bus stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping event bus: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the event bus"""
        return {
            **self.status,
            "stats": self.stats.copy(),
            "queue_size": self.event_queue.qsize(),
            "subscribers_count": {k: len(v) for k, v in self.subscribers.items()}
        }

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event bus history"""
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_event_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event history"""
        history = self.event_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return self.stats.copy()


# Global instance
event_bus: Optional[EventBus] = None


async def setup_event_bus(config: Optional[Dict[str, Any]] = None) -> Optional[EventBus]:
    """Setup and initialize the event bus"""
    try:
        global event_bus
        
        if config is None:
            config = {
                "event_bus": {
                    "processing_interval": 10,
                    "max_history": 100,
                }
            }
        
        event_bus = EventBus(config)
        success = await event_bus.initialize()
        
        if success:
            return event_bus
        
        return None
        
    except Exception as e:
        print(f"Error setting up event bus: {e}")
        return None
