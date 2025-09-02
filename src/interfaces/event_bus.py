# src/interfaces/event_bus.py

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio
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
    """Event types for the trading system."""

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
    """Event structure for the event bus."""
    

    event_type: EventType
    data: Any
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate event after initialization."""
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType enum")
        if self.timestamp > datetime.now():
            raise ValueError("Timestamp cannot be in the future")
        if not self.source:
            raise ValueError("Source cannot be empty")


class EventBus:
    """Enhanced Event Bus component with DI, type hints, and robust error handling."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize EventBus."""

        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("EventBus")
        self.is_running: bool = False
        self.status: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Event bus configuration
        self.event_bus_config: Dict[str, Any] = self.config.get("event_bus", {})
        self.processing_interval: int = self.event_bus_config.get("processing_interval", 10)
        self.max_history: int = self.event_bus_config.get("max_history", 100)
        
        # Event management

        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_events_processed: int = 0
        self.total_events_published: int = 0
        self.start_time: Optional[datetime] = None
    
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
        """Initialize the Event Bus."""
        try:
            self.logger.info("Initializing Event Bus...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for event bus"))
                return False
            
            # Initialize event processing
            await self._initialize_event_processing()
            
            self.start_time = datetime.now()
            self.is_running = True
            

            self.logger.info("✅ Event Bus initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error initializing Event Bus: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate event bus configuration."""
        try:
            required_keys = ["processing_interval", "max_history"]
            for key in required_keys:
                if key not in self.event_bus_config:
                    self.logger.warning(f"Missing configuration key: {key}")
                    return False
            
            if self.processing_interval <= 0:
                self.logger.warning("Processing interval must be positive")
                return False
            
            if self.max_history <= 0:
                self.logger.warning("Max history must be positive")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    async def _initialize_event_processing(self) -> None:
        """Initialize event processing loop."""
        try:
            # Start event processing task
            asyncio.create_task(self._process_events())
            self.logger.info("Event processing task started")
            
        except Exception as e:
            self.logger.error(f"Error starting event processing: {e}")
            raise
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self.is_running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), 
                        timeout=self.processing_interval
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the event
                await self._handle_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        try:
            event_type = event.event_type.value
            
            # Log event
            self.logger.debug(f"Processing event: {event_type} from {event.source}")
            
            # Notify subscribers
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    try:
                        await callback(event)
                    except Exception as e:
                        self.logger.error(f"Error in event callback: {e}")
            
            # Update statistics
            self.total_events_processed += 1
            
            # Add to history
            self._add_to_history(event)
            
        except Exception as e:
            self.logger.error(f"Error handling event: {e}")
    
    def _add_to_history(self, event: Event) -> None:
        """Add event to history with size limit."""
        try:
            event_record = {
                "event_type": event.event_type.value,
                "source": event.source,
                "timestamp": event.timestamp.isoformat(),
                "correlation_id": event.correlation_id,
                "data_summary": str(event.data)[:100]  # Truncate for storage
            }
            
            self.event_history.append(event_record)
            
            # Maintain history size limit
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding event to history: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="event publishing",
    )
    async def publish_event(self, event_type: EventType, data: Any, source: str, 
                          correlation_id: Optional[str] = None) -> bool:
        """Publish an event to the bus."""
        try:
            # Generate correlation ID if not provided
            if not correlation_id:
                correlation_id = str(uuid.uuid4())

            
            # Create event
            event = Event(
                event_type=event_type,
                data=data,
                timestamp=datetime.now(),
                source=source,
                correlation_id=correlation_id
            )
            
            # Add to queue
            await self.event_queue.put(event)
            
            # Update statistics
            self.total_events_published += 1
            
            self.logger.debug(f"Event published: {event_type.value} from {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="event subscription",
    )
    async def subscribe(self, event_type: EventType, callback: Callable) -> bool:
        """Subscribe to events of a specific type."""
        try:
            event_type_str = event_type.value
            
            if callback not in self.subscribers[event_type_str]:
                self.subscribers[event_type_str].append(callback)
                self.logger.info(f"Subscribed to {event_type_str}")
                return True
            else:
                self.logger.warning(f"Callback already subscribed to {event_type_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="event unsubscription",
    )
    async def unsubscribe(self, event_type: EventType, callback: Callable) -> bool:
        """Unsubscribe from events of a specific type."""
        try:
            event_type_str = event_type.value
            
            if event_type_str in self.subscribers:
                if callback in self.subscribers[event_type_str]:
                    self.subscribers[event_type_str].remove(callback)
                    self.logger.info(f"Unsubscribed from {event_type_str}")
                    return True
                else:
                    self.logger.warning(f"Callback not found in {event_type_str} subscribers")
                    return False
            else:
                self.logger.warning(f"No subscribers for {event_type_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from events: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                "is_running": self.is_running,
                "total_events_published": self.total_events_published,
                "total_events_processed": self.total_events_processed,
                "queue_size": self.event_queue.qsize(),
                "subscriber_count": sum(len(callbacks) for callbacks in self.subscribers.values()),
                "history_size": len(self.event_history),
                "uptime_seconds": uptime,
                "events_per_second": self.total_events_processed / uptime if uptime > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def get_event_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event history."""
        try:
            if limit is None:
                limit = self.max_history
            
            return self.event_history[-limit:]
            
        except Exception as e:
            self.logger.error(f"Error getting event history: {e}")
            return []
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="event bus cleanup",
    )
    async def cleanup(self) -> bool:
        """Cleanup Event Bus resources."""
        try:
            self.logger.info("Cleaning up Event Bus...")
            
            # Stop processing
            self.is_running = False
            
            # Wait for queue to be processed
            if not self.event_queue.empty():
                await self.event_queue.join()
            
            # Clear subscribers
            self.subscribers.clear()
            
            # Clear history
            self.event_history.clear()
            
            self.logger.info("✅ Event Bus cleanup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up Event Bus: {e}")
            return False
