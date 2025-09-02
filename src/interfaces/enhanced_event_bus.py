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
from src.utils.simple_decorators import performance_monitor, PerformanceLevel
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    validation_error,
    warning,
)


class EventType(Enum):
    """Enhanced event types for the trading system."""

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
    """Event processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class EventMetadata:
    """Metadata for event tracking and versioning."""
    

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
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.event_id:
            raise ValueError("Event ID cannot be empty")
        if self.retry_count < 0:
            raise ValueError("Retry count cannot be negative")
        if self.sequence_number < 0:
            raise ValueError("Sequence number cannot be negative")



@dataclass
class Event:
    """Enhanced event structure with versioning and metadata."""
    
    event_type: EventType
    data: Any
    metadata: EventMetadata = field(default_factory=EventMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        try:
            md = asdict(self.metadata)
            # Ensure timestamp is serialized to ISO format
            if isinstance(self.metadata.timestamp, datetime):
                md["timestamp"] = self.metadata.timestamp.isoformat()
            
            return {
                "event_type": self.event_type.value,
                "data": self.data,
                "metadata": md
            }
        except Exception as e:
            raise ValueError(f"Error serializing event: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        try:
            event_type = EventType(data["event_type"])
            event_data = data["data"]
            
            # Parse metadata
            md_data = data.get("metadata", {})
            if "timestamp" in md_data and isinstance(md_data["timestamp"], str):
                md_data["timestamp"] = datetime.fromisoformat(md_data["timestamp"])
            if "status" in md_data and isinstance(md_data["status"], str):
                md_data["status"] = EventStatus(md_data["status"])
            
            metadata = EventMetadata(**md_data)
            
            return cls(event_type=event_type, data=event_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Error deserializing event: {e}")


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize event handler."""
        self.name = name
        self.config = config or {}
        self.logger = system_logger.getChild(f"EventHandler.{name}")
        self.is_initialized = False
    
    @abstractmethod
    async def handle_event(self, event: Event) -> bool:
        """Handle an event."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the handler."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class EnhancedEventBus:
    """Enhanced Event Bus with advanced features like event persistence, retry logic, and performance monitoring."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize EnhancedEventBus."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedEventBus")
        self.is_running = False
        
        # Configuration
        self.event_bus_config = self.config.get("enhanced_event_bus", {})
        self.processing_interval = self.event_bus_config.get("processing_interval", 10)
        self.max_history = self.event_bus_config.get("max_history", 1000)
        self.max_retries = self.event_bus_config.get("max_retries", 3)
        self.retry_delay = self.event_bus_config.get("retry_delay", 1.0)
        self.enable_persistence = self.event_bus_config.get("enable_persistence", False)
        self.persistence_path = Path(self.event_bus_config.get("persistence_path", "data/events"))
        
        # Event management
        self.subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Event] = []
        self.failed_events: List[Event] = []
        
        # Performance tracking
        self.total_events_processed = 0
        self.total_events_published = 0
        self.start_time: Optional[datetime] = None
        
        # Event processing
        self.processing_task: Optional[asyncio.Task] = None
        self.retry_task: Optional[asyncio.Task] = None
    
    @performance_monitor(level=PerformanceLevel.HIGH)
    async def initialize(self) -> bool:
        """Initialize the Enhanced Event Bus."""
        try:
            self.logger.info("Initializing Enhanced Event Bus...")
            
            # Validate configuration

            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for enhanced event bus"))
                return False
            
            # Setup persistence if enabled
            if self.enable_persistence:
                await self._setup_persistence()
            
            # Initialize event processing
            await self._initialize_event_processing()
            
            self.start_time = datetime.now()
            self.is_running = True
            
            self.logger.info("✅ Enhanced Event Bus initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error initializing Enhanced Event Bus: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate enhanced event bus configuration."""
        try:
            required_keys = ["processing_interval", "max_history", "max_retries"]
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
            
            if self.max_retries < 0:
                self.logger.warning("Max retries cannot be negative")
                return False
            
            if self.retry_delay < 0:
                self.logger.warning("Retry delay cannot be negative")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    async def _setup_persistence(self) -> None:
        """Setup event persistence."""
        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Event persistence enabled at: {self.persistence_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up persistence: {e}")
            raise
    
    async def _initialize_event_processing(self) -> None:
        """Initialize event processing."""
        try:
            # Start event processing task
            self.processing_task = asyncio.create_task(self._process_events())
            
            # Start retry processing task
            self.retry_task = asyncio.create_task(self._process_retries())
            
            self.logger.info("Event processing tasks started")
            
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
                await asyncio.sleep(1)
    
    async def _process_retries(self) -> None:
        """Process failed events for retry."""
        while self.is_running:
            try:
                # Check for events that need retrying
                events_to_retry = [
                    event for event in self.failed_events
                    if event.metadata.retry_count < self.max_retries
                ]
                
                for event in events_to_retry:
                    if await self._should_retry_event(event):
                        await self._retry_event(event)
                
                # Wait before next retry cycle
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                self.logger.error(f"Error processing retries: {e}")
                await asyncio.sleep(1)
    
    async def _should_retry_event(self, event: Event) -> bool:
        """Check if an event should be retried."""
        try:
            # Check retry count
            if event.metadata.retry_count >= self.max_retries:
                return False
            
            # Check if enough time has passed since last retry
            time_since_last_retry = (datetime.now(timezone.utc) - event.metadata.timestamp).total_seconds()
            return time_since_last_retry >= self.retry_delay
            
        except Exception as e:
            self.logger.error(f"Error checking retry condition: {e}")
            return False
    
    async def _retry_event(self, event: Event) -> None:
        """Retry a failed event."""
        try:
            # Update retry count and timestamp
            event.metadata.retry_count += 1
            event.metadata.timestamp = datetime.now(timezone.utc)
            event.metadata.status = EventStatus.RETRYING
            
            # Remove from failed events
            self.failed_events.remove(event)
            
            # Add back to processing queue
            await self.event_queue.put(event)
            
            self.logger.info(f"Retrying event {event.metadata.event_id} (attempt {event.metadata.retry_count})")
            
        except Exception as e:
            self.logger.error(f"Error retrying event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        try:
            event_type = event.event_type.value
            
            # Update status
            event.metadata.status = EventStatus.PROCESSING
            
            # Log event
            self.logger.debug(f"Processing event: {event_type} (ID: {event.metadata.event_id})")
            
            # Notify subscribers
            success = await self._notify_subscribers(event)
            
            if success:
                event.metadata.status = EventStatus.PROCESSED
                self.total_events_processed += 1
                self._add_to_history(event)
                
                # Persist if enabled
                if self.enable_persistence:
                    await self._persist_event(event)
                    
            else:
                event.metadata.status = EventStatus.FAILED
                self.failed_events.append(event)
                
        except Exception as e:
            self.logger.error(f"Error handling event: {e}")
            event.metadata.status = EventStatus.FAILED
            self.failed_events.append(event)
    
    async def _notify_subscribers(self, event: Event) -> bool:
        """Notify all subscribers of an event."""
        try:
            event_type = event.event_type.value
            subscribers = self.subscribers.get(event_type, [])
            
            if not subscribers:
                self.logger.debug(f"No subscribers for event type: {event_type}")
                return True
            
            success_count = 0
            for handler in subscribers:
                try:
                    if await handler.handle_event(event):
                        success_count += 1
                    else:
                        self.logger.warning(f"Handler {handler.name} failed to process event")
                        
                except Exception as e:
                    self.logger.error(f"Error in event handler {handler.name}: {e}")
            
            # Consider successful if at least one handler succeeded
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error notifying subscribers: {e}")
            return False
    
    def _add_to_history(self, event: Event) -> None:
        """Add event to history with size limit."""
        try:
            self.event_history.append(event)
            
            # Maintain history size limit
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding event to history: {e}")
    
    async def _persist_event(self, event: Event) -> None:
        """Persist event to storage."""
        try:
            if not self.enable_persistence:
                return
            
            # Create filename with timestamp and event ID
            timestamp_str = event.metadata.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp_str}_{event.metadata.event_id}.json"
            filepath = self.persistence_path / filename
            
            # Serialize and save
            event_dict = event.to_dict()
            with open(filepath, 'w') as f:
                json.dump(event_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error persisting event: {e}")
    
    @performance_monitor(level=PerformanceLevel.MEDIUM)
    async def publish_event(self, event_type: EventType, data: Any, source: str, 
                          correlation_id: Optional[str] = None, 
                          causation_id: Optional[str] = None,
                          tags: Optional[Dict[str, str]] = None) -> bool:
        """Publish an event to the bus."""
        try:
            # Create metadata
            metadata = EventMetadata(
                source=source,
                correlation_id=correlation_id,
                causation_id=causation_id,
                tags=tags or {}
            )
            
            # Create event
            event = Event(
                event_type=event_type,
                data=data,
                metadata=metadata
            )
            
            # Add to queue
            await self.event_queue.put(event)
            
            # Update statistics
            self.total_events_published += 1
            
            self.logger.debug(f"Event published: {event_type.value} (ID: {metadata.event_id}) from {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            return False
    
    async def subscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Subscribe to events of a specific type."""
        try:
            event_type_str = event_type.value
            
            if handler not in self.subscribers[event_type_str]:
                self.subscribers[event_type_str].append(handler)
                self.logger.info(f"Handler {handler.name} subscribed to {event_type_str}")
                return True
            else:
                self.logger.warning(f"Handler {handler.name} already subscribed to {event_type_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error subscribing to events: {e}")
            return False
    
    async def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Unsubscribe from events of a specific type."""
        try:
            event_type_str = event_type.value
            
            if event_type_str in self.subscribers:
                if handler in self.subscribers[event_type_str]:
                    self.subscribers[event_type_str].remove(handler)
                    self.logger.info(f"Handler {handler.name} unsubscribed from {event_type_str}")
                    return True
                else:
                    self.logger.warning(f"Handler {handler.name} not found in {event_type_str} subscribers")
                    return False
            else:
                self.logger.warning(f"No subscribers for {event_type_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from events: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced event bus statistics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                "is_running": self.is_running,
                "total_events_published": self.total_events_published,
                "total_events_processed": self.total_events_processed,
                "queue_size": self.event_queue.qsize(),
                "subscriber_count": sum(len(handlers) for handlers in self.subscribers.values()),
                "history_size": len(self.event_history),
                "failed_events_count": len(self.failed_events),
                "uptime_seconds": uptime,
                "events_per_second": self.total_events_processed / uptime if uptime > 0 else 0,
                "persistence_enabled": self.enable_persistence,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def get_event_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event history."""
        try:
            if limit is None:
                limit = self.max_history
            
            events = self.event_history[-limit:]
            return [event.to_dict() for event in events]
            
        except Exception as e:
            self.logger.error(f"Error getting event history: {e}")
            return []
    
    async def get_failed_events(self) -> List[Dict[str, Any]]:
        """Get list of failed events."""
        try:
            return [event.to_dict() for event in self.failed_events]
            
        except Exception as e:
            self.logger.error(f"Error getting failed events: {e}")
            return []
    
    async def retry_failed_event(self, event_id: str) -> bool:
        """Manually retry a specific failed event."""
        try:
            # Find the event
            event = next((e for e in self.failed_events if e.metadata.event_id == event_id), None)
            
            if not event:
                self.logger.warning(f"Failed event {event_id} not found")
                return False
            
            # Remove from failed events
            self.failed_events.remove(event)
            
            # Reset retry count and add to queue
            event.metadata.retry_count = 0
            event.metadata.status = EventStatus.PENDING
            await self.event_queue.put(event)
            
            self.logger.info(f"Manually retrying failed event {event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error retrying failed event: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup Enhanced Event Bus resources."""
        try:
            self.logger.info("Cleaning up Enhanced Event Bus...")
            
            # Stop processing
            self.is_running = False
            
            # Cancel tasks
            if self.processing_task:
                self.processing_task.cancel()
            if self.retry_task:
                self.retry_task.cancel()
            
            # Wait for queue to be processed
            if not self.event_queue.empty():
                await self.event_queue.join()
            
            # Clear subscribers
            self.subscribers.clear()
            
            # Clear history
            self.event_history.clear()
            self.failed_events.clear()
            
            self.logger.info("✅ Enhanced Event Bus cleanup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up Enhanced Event Bus: {e}")
            return False
