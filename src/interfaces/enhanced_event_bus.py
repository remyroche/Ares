# src/interfaces/enhanced_event_bus.py

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import uuid
import logging

from src.utils.logger import system_logger
from src.utils.advanced_decorators import performance_monitor, PerformanceLevel
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    validation_error,
    warning,
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
    SYSTEM_HEALTH_CHECK = "system_health_check"
    CONFIGURATION_CHANGED = "configuration_changed"
    SNAPSHOT_CREATED = "snapshot_created"


class EventStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class EventMetadata:
    """Metadata for event tracking and versioning"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.2.3"
    schema_version: str = "1.2.3"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    aggregate_id: Optional[str] = None
    sequence_number: int = 0
    retry_count: int = 0
    status: EventStatus = EventStatus.PENDING
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    """Enhanced event structure with versioning and metadata"""
    event_type: EventType
    data: Any
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        md = asdict(self.metadata)
        # Ensure timestamp is serialized to ISO format
        if isinstance(self.metadata.timestamp, datetime):
            md["timestamp"] = self.metadata.timestamp.isoformat()
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "metadata": md
        }

    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        event_type = EventType(data["event_type"])
        metadata_data = data.get("metadata", {})
        
        # Parse timestamp back to datetime
        if "timestamp" in metadata_data:
            metadata_data["timestamp"] = datetime.fromisoformat(metadata_data["timestamp"])
        
        metadata = EventMetadata(**metadata_data)
        return cls(event_type=event_type, data=data["data"], metadata=metadata)


@dataclass
class EventSubscription:
    """Event subscription configuration"""
    event_types: List[EventType]
    callback: Callable[[Event], Any]
    priority: int = 0
    filter_func: Optional[Callable[[Event], bool]] = None
    max_retries: int = 3
    timeout: Optional[float] = None


class EventBusInterface(ABC):
    """Abstract interface for event bus implementations"""
    
    @abstractmethod
    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus"""
        pass
    
    @abstractmethod
    async def subscribe(self, subscription: EventSubscription) -> str:
        """Subscribe to events"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        pass
    
    @abstractmethod
    async def get_event_history(self, event_type: Optional[EventType] = None, 
                              limit: int = 100) -> List[Event]:
        """Get event history"""
        pass


class EnhancedEventBus(EventBusInterface):
    """
    Enhanced event bus with advanced features:
    - Event persistence and replay
    - Priority-based processing
    - Retry mechanisms
    - Event filtering
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedEventBus")
        self.is_running: bool = False
        self.status: Dict[str, Any] = {}
        
        # Event bus configuration
        self.event_bus_config: Dict[str, Any] = self.config.get("event_bus", {})
        self.processing_interval: int = self.event_bus_config.get("processing_interval", 10)
        self.max_history: int = self.event_bus_config.get("max_history", 1000)
        self.max_retries: int = self.event_bus_config.get("max_retries", 3)
        self.retry_delay: float = self.event_bus_config.get("retry_delay", 1.0)
        
        # Event storage and processing
        self.subscribers: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.event_history: List[Event] = []
        self.subscription_counter: int = 0
        self.subscription_map: Dict[str, EventSubscription] = {}
        
        # Performance monitoring
        self.stats: Dict[str, Any] = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "subscribers_active": 0,
            "queue_size": 0
        }
        
        # Error handling
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        
        # Event persistence
        self.persistence_enabled: bool = self.event_bus_config.get("persistence_enabled", True)
        self.persistence_path: Optional[Path] = None
        if self.persistence_enabled:
            self.persistence_path = Path(self.event_bus_config.get("persistence_path", "data/events"))
            self.persistence_path.mkdir(parents=True, exist_ok=True)

    @performance_monitor(level=PerformanceLevel.HIGH)
    async def initialize(self) -> bool:
        """Initialize the enhanced event bus"""
        try:
            self.logger.info("Initializing Enhanced Event Bus...")
            
            # Load configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for enhanced event bus"))
                return False
            
            # Initialize event processing
            await self._initialize_event_processing()
            
            # Load persisted events if enabled
            if self.persistence_enabled:
                await self._load_persisted_events()
            
            self.is_running = True
            self.logger.info("✅ Enhanced Event Bus initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Event Bus: {e}")
            self.last_error = str(e)
            return False

    def _validate_configuration(self) -> bool:
        """Validate event bus configuration"""
        required_keys = ["processing_interval", "max_history"]
        for key in required_keys:
            if key not in self.event_bus_config:
                self.logger.error(f"Missing required configuration key: {key}")
                return False
        return True

    async def _initialize_event_processing(self) -> None:
        """Initialize event processing infrastructure"""
        # Start event processing task
        asyncio.create_task(self._event_processor())
        self.logger.info("Event processing task started")

    async def _event_processor(self) -> None:
        """Main event processing loop"""
        while self.is_running:
            try:
                # Process events from queue
                if not self.event_queue.empty():
                    priority, timestamp, event = await self.event_queue.get()
                    await self._process_event(event)
                    self.event_queue.task_done()
                
                # Update statistics
                self.stats["queue_size"] = self.event_queue.qsize()
                self.stats["subscribers_active"] = len(self.subscription_map)
                
                await asyncio.sleep(self.processing_interval / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in event processor: {e}")
                self.error_count += 1
                self.last_error = str(e)
                await asyncio.sleep(1.0)

    async def _process_event(self, event: Event) -> None:
        """Process a single event"""
        try:
            self.logger.debug(f"Processing event: {event.event_type.value}")
            
            # Update event status
            event.metadata.status = EventStatus.PROCESSING
            
            # Find subscribers for this event type
            event_type_str = event.event_type.value
            if event_type_str in self.subscribers:
                # Process subscribers by priority
                sorted_subscribers = sorted(
                    self.subscribers[event_type_str],
                    key=lambda s: s.priority,
                    reverse=True
                )
                
                for subscription in sorted_subscribers:
                    try:
                        # Apply filter if specified
                        if subscription.filter_func and not subscription.filter_func(event):
                            continue
                        
                        # Execute callback with timeout
                        if subscription.timeout:
                            await asyncio.wait_for(
                                self._execute_callback(subscription, event),
                                timeout=subscription.timeout
                            )
                        else:
                            await self._execute_callback(subscription, event)
                            
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Callback timeout for subscription {id(subscription)}")
                        await self._handle_callback_error(subscription, event, "Timeout")
                    except Exception as e:
                        await self._handle_callback_error(subscription, event, str(e))
            
            # Update event status
            event.metadata.status = EventStatus.PROCESSED
            
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
            
            # Persist event if enabled
            if self.persistence_enabled:
                await self._persist_event(event)
            
            self.stats["events_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing event {event.metadata.event_id}: {e}")
            event.metadata.status = EventStatus.FAILED
            self.stats["events_failed"] += 1
            self.error_count += 1

    async def _execute_callback(self, subscription: EventSubscription, event: Event) -> None:
        """Execute a subscription callback"""
        if asyncio.iscoroutinefunction(subscription.callback):
            await subscription.callback(event)
        else:
            subscription.callback(event)

    async def _handle_callback_error(self, subscription: EventSubscription, event: Event, error_msg: str) -> None:
        """Handle callback execution errors"""
        self.logger.error(f"Callback error for event {event.metadata.event_id}: {error_msg}")
        
        # Implement retry logic
        if event.metadata.retry_count < subscription.max_retries:
            event.metadata.retry_count += 1
            event.metadata.status = EventStatus.RETRYING
            
            # Re-queue event with delay
            await asyncio.sleep(self.retry_delay * event.metadata.retry_count)
            await self.event_queue.put((0, datetime.now().timestamp(), event))
        else:
            event.metadata.status = EventStatus.FAILED
            self.stats["events_failed"] += 1

    @performance_monitor(level=PerformanceLevel.MEDIUM)
    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus"""
        try:
            if not self.is_running:
                self.logger.error("Event bus is not running")
                return False
            
            # Validate event
            if not self._validate_event(event):
                self.logger.error(f"Invalid event: {event}")
                return False
            
            # Add to processing queue with priority
            priority = self._calculate_event_priority(event)
            timestamp = datetime.now().timestamp()
            await self.event_queue.put((priority, timestamp, event))
            
            self.stats["events_published"] += 1
            self.logger.debug(f"Event published: {event.event_type.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            self.error_count += 1
            return False

    def _validate_event(self, event: Event) -> bool:
        """Validate event structure"""
        if not isinstance(event, Event):
            return False
        if not isinstance(event.event_type, EventType):
            return False
        if event.metadata is None:
            return False
        return True

    def _calculate_event_priority(self, event: Event) -> int:
        """Calculate event processing priority"""
        # Higher priority for critical events
        critical_events = [
            EventType.RISK_ALERT,
            EventType.SYSTEM_ERROR,
            EventType.TRADE_DECISION_MADE
        ]
        
        if event.event_type in critical_events:
            return 10
        elif event.event_type == EventType.MARKET_DATA_RECEIVED:
            return 5
        else:
            return 1

    async def subscribe(self, subscription: EventSubscription) -> str:
        """Subscribe to events"""
        try:
            subscription_id = str(self.subscription_counter)
            self.subscription_counter += 1
            
            # Add to subscribers for each event type
            for event_type in subscription.event_types:
                event_type_str = event_type.value
                self.subscribers[event_type_str].append(subscription)
            
            # Store subscription mapping
            self.subscription_map[subscription_id] = subscription
            
            self.logger.info(f"Subscription {subscription_id} created for {len(subscription.event_types)} event types")
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Error creating subscription: {e}")
            raise

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        try:
            if subscription_id not in self.subscription_map:
                return False
            
            subscription = self.subscription_map[subscription_id]
            
            # Remove from subscribers for each event type
            for event_type in subscription.event_types:
                event_type_str = event_type.value
                if event_type_str in self.subscribers:
                    self.subscribers[event_type_str] = [
                        s for s in self.subscribers[event_type_str] 
                        if s != subscription
                    ]
            
            # Remove from subscription map
            del self.subscription_map[subscription_id]
            
            self.logger.info(f"Subscription {subscription_id} removed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing subscription: {e}")
            return False

    async def get_event_history(self, event_type: Optional[EventType] = None, 
                              limit: int = 100) -> List[Event]:
        """Get event history"""
        try:
            if event_type is None:
                return self.event_history[-limit:]
            else:
                filtered_events = [
                    event for event in self.event_history 
                    if event.event_type == event_type
                ]
                return filtered_events[-limit:]
                
        except Exception as e:
            self.logger.error(f"Error retrieving event history: {e}")
            return []

    async def _persist_event(self, event: Event) -> None:
        """Persist event to storage"""
        try:
            if not self.persistence_path:
                return
            
            # Create filename based on event ID and timestamp
            filename = f"{event.metadata.event_id}_{event.metadata.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.persistence_path / filename
            
            # Write event to file
            with open(filepath, 'w') as f:
                json.dump(event.to_dict(), f, default=str, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error persisting event: {e}")

    async def _load_persisted_events(self) -> None:
        """Load persisted events from storage"""
        try:
            if not self.persistence_path or not self.persistence_path.exists():
                return
            
            # Load events from persistence directory
            for filepath in self.persistence_path.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        event_data = json.load(f)
                    
                    event = Event.from_dict(event_data)
                    self.event_history.append(event)
                    
                except Exception as e:
                    self.logger.warning(f"Error loading persisted event from {filepath}: {e}")
            
            self.logger.info(f"Loaded {len(self.event_history)} persisted events")
            
        except Exception as e:
            self.logger.error(f"Error loading persisted events: {e}")

    async def shutdown(self) -> None:
        """Shutdown the event bus"""
        try:
            self.logger.info("Shutting down Enhanced Event Bus...")
            self.is_running = False
            
            # Wait for event queue to empty
            await self.event_queue.join()
            
            # Clear subscribers
            self.subscribers.clear()
            self.subscription_map.clear()
            
            self.logger.info("Enhanced Event Bus shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the event bus"""
        return {
            "is_running": self.is_running,
            "stats": self.stats.copy(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "queue_size": self.event_queue.qsize(),
            "subscriber_count": len(self.subscription_map),
            "history_size": len(self.event_history)
        }

    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information"""
        return {
            "status": "healthy" if self.is_running and self.error_count < 10 else "degraded",
            "is_running": self.is_running,
            "error_count": self.error_count,
            "queue_size": self.event_queue.qsize(),
            "last_error": self.last_error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
