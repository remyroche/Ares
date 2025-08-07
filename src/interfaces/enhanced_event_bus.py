# src/interfaces/enhanced_event_bus.py

import asyncio
import json
import pickle
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from src.utils.error_handler import handle_errors, handle_specific_errors
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
    version: str = "1.0.0"
    schema_version: str = "1.0.0"
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
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "metadata": asdict(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary"""
        metadata_dict = data.get("metadata", {})
        metadata = EventMetadata(
            event_id=metadata_dict.get("event_id", str(uuid.uuid4())),
            version=metadata_dict.get("version", "1.0.0"),
            schema_version=metadata_dict.get("schema_version", "1.0.0"),
            timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now(timezone.utc).isoformat())),
            source=metadata_dict.get("source", ""),
            correlation_id=metadata_dict.get("correlation_id"),
            causation_id=metadata_dict.get("causation_id"),
            aggregate_id=metadata_dict.get("aggregate_id"),
            sequence_number=metadata_dict.get("sequence_number", 0),
            retry_count=metadata_dict.get("retry_count", 0),
            status=EventStatus(metadata_dict.get("status", "pending")),
            tags=metadata_dict.get("tags", {})
        )
        
        return cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            metadata=metadata
        )


@dataclass
class EventSnapshot:
    """Snapshot of system state at a point in time"""
    
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: str = ""
    sequence_number: int = 0
    state_data: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


class IEventStore(ABC):
    """Interface for event storage implementations"""
    
    @abstractmethod
    async def save_event(self, event: Event) -> bool:
        """Save an event to the store"""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        aggregate_id: Optional[str] = None,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[Event]:
        """Retrieve events from the store"""
        pass
    
    @abstractmethod
    async def save_snapshot(self, snapshot: EventSnapshot) -> bool:
        """Save a snapshot to the store"""
        pass
    
    @abstractmethod
    async def get_latest_snapshot(self, aggregate_id: str) -> Optional[EventSnapshot]:
        """Get the latest snapshot for an aggregate"""
        pass


class FileEventStore(IEventStore):
    """File-based event store implementation"""
    
    def __init__(self, storage_path: str = "event_store"):
        self.storage_path = Path(storage_path)
        self.events_path = self.storage_path / "events"
        self.snapshots_path = self.storage_path / "snapshots"
        self.logger = system_logger.getChild("FileEventStore")
        
        # Create directories
        self.events_path.mkdir(parents=True, exist_ok=True)
        self.snapshots_path.mkdir(parents=True, exist_ok=True)
    
    async def save_event(self, event: Event) -> bool:
        """Save an event to file storage"""
        try:
            event_date = event.metadata.timestamp.strftime("%Y-%m-%d")
            event_file = self.events_path / f"events_{event_date}.jsonl"
            
            event_data = event.to_dict()
            event_line = json.dumps(event_data, default=str) + "\n"
            
            # Append to file
            with open(event_file, "a", encoding="utf-8") as f:
                f.write(event_line)
            
            self.logger.debug(f"Saved event {event.metadata.event_id} to {event_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save event: {e}")
            return False
    
    async def get_events(
        self,
        aggregate_id: Optional[str] = None,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[Event]:
        """Retrieve events from file storage"""
        try:
            events = []
            
            # Read all event files
            for event_file in self.events_path.glob("events_*.jsonl"):
                with open(event_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line.strip())
                            event = Event.from_dict(event_data)
                            
                            # Apply filters
                            if aggregate_id and event.metadata.aggregate_id != aggregate_id:
                                continue
                            
                            if event.metadata.sequence_number < from_sequence:
                                continue
                            
                            if to_sequence and event.metadata.sequence_number > to_sequence:
                                continue
                            
                            if event_types and event.event_type not in event_types:
                                continue
                            
                            events.append(event)
            
            # Sort by sequence number
            events.sort(key=lambda e: e.metadata.sequence_number)
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve events: {e}")
            return []
    
    async def save_snapshot(self, snapshot: EventSnapshot) -> bool:
        """Save a snapshot to file storage"""
        try:
            snapshot_file = self.snapshots_path / f"snapshot_{snapshot.aggregate_id}_{snapshot.sequence_number}.json"
            
            snapshot_data = asdict(snapshot)
            # Convert datetime to string for JSON serialization
            snapshot_data["timestamp"] = snapshot.timestamp.isoformat()
            
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            self.logger.debug(f"Saved snapshot {snapshot.snapshot_id} to {snapshot_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
            return False
    
    async def get_latest_snapshot(self, aggregate_id: str) -> Optional[EventSnapshot]:
        """Get the latest snapshot for an aggregate"""
        try:
            latest_snapshot = None
            latest_sequence = -1
            
            # Find the latest snapshot file for the aggregate
            for snapshot_file in self.snapshots_path.glob(f"snapshot_{aggregate_id}_*.json"):
                with open(snapshot_file, "r", encoding="utf-8") as f:
                    snapshot_data = json.load(f)
                    
                    sequence_number = snapshot_data.get("sequence_number", 0)
                    if sequence_number > latest_sequence:
                        latest_sequence = sequence_number
                        snapshot_data["timestamp"] = datetime.fromisoformat(snapshot_data["timestamp"])
                        latest_snapshot = EventSnapshot(**snapshot_data)
            
            return latest_snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve latest snapshot: {e}")
            return None


class EventVersionManager:
    """Manages event schema versioning and migration"""
    
    def __init__(self):
        self.logger = system_logger.getChild("EventVersionManager")
        self.version_mappings: Dict[str, Dict[str, Any]] = {}
        self._register_default_versions()
    
    def _register_default_versions(self):
        """Register default version mappings"""
        # Example version mappings for backward compatibility
        self.version_mappings = {
            "1.0.0": {
                "market_data_received": {
                    "required_fields": ["symbol", "price", "volume"],
                    "optional_fields": ["timestamp", "bid", "ask"]
                },
                "trade_executed": {
                    "required_fields": ["symbol", "side", "quantity", "price"],
                    "optional_fields": ["order_id", "commission"]
                }
            },
            "1.1.0": {
                "market_data_received": {
                    "required_fields": ["symbol", "price", "volume", "timestamp"],
                    "optional_fields": ["bid", "ask", "spread"]
                },
                "trade_executed": {
                    "required_fields": ["symbol", "side", "quantity", "price", "order_id"],
                    "optional_fields": ["commission", "fees"]
                }
            }
        }
    
    def validate_event_schema(self, event: Event) -> bool:
        """Validate event against its schema version"""
        try:
            version = event.metadata.schema_version
            event_type = event.event_type.value
            
            if version not in self.version_mappings:
                self.logger.warning(f"Unknown schema version: {version}")
                return True  # Allow unknown versions for forward compatibility
            
            schema = self.version_mappings[version].get(event_type)
            if not schema:
                self.logger.warning(f"No schema defined for event type: {event_type}")
                return True
            
            # Validate required fields
            required_fields = schema.get("required_fields", [])
            if isinstance(event.data, dict):
                for field in required_fields:
                    if field not in event.data:
                        self.logger.error(f"Missing required field '{field}' in event {event.metadata.event_id}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False
    
    def migrate_event(self, event: Event, target_version: str) -> Event:
        """Migrate event to target schema version"""
        try:
            current_version = event.metadata.schema_version
            
            if current_version == target_version:
                return event
            
            # Create a copy of the event for migration
            migrated_event = Event(
                event_type=event.event_type,
                data=event.data.copy() if isinstance(event.data, dict) else event.data,
                metadata=EventMetadata(
                    event_id=event.metadata.event_id,
                    version=event.metadata.version,
                    schema_version=target_version,
                    timestamp=event.metadata.timestamp,
                    source=event.metadata.source,
                    correlation_id=event.metadata.correlation_id,
                    causation_id=event.metadata.causation_id,
                    aggregate_id=event.metadata.aggregate_id,
                    sequence_number=event.metadata.sequence_number,
                    retry_count=event.metadata.retry_count,
                    status=event.metadata.status,
                    tags=event.metadata.tags.copy()
                )
            )
            
            # Apply migration logic based on version differences
            # This is a simplified example - real migration would be more complex
            if current_version == "1.0.0" and target_version == "1.1.0":
                if isinstance(migrated_event.data, dict):
                    # Add timestamp if missing
                    if "timestamp" not in migrated_event.data:
                        migrated_event.data["timestamp"] = migrated_event.metadata.timestamp.isoformat()
            
            self.logger.info(f"Migrated event {event.metadata.event_id} from {current_version} to {target_version}")
            return migrated_event
            
        except Exception as e:
            self.logger.error(f"Event migration error: {e}")
            return event


class EnhancedEventBus:
    """
    Enhanced Event Bus with event sourcing, versioning, and persistence capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedEventBus")
        self.is_running = False
        self.status: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Configuration
        self.event_bus_config = self.config.get("event_bus", {})
        self.processing_interval = self.event_bus_config.get("processing_interval", 1)
        self.max_history = self.event_bus_config.get("max_history", 1000)
        self.enable_persistence = self.event_bus_config.get("enable_persistence", True)
        self.enable_snapshots = self.event_bus_config.get("enable_snapshots", True)
        self.snapshot_frequency = self.event_bus_config.get("snapshot_frequency", 100)
        self.storage_path = self.event_bus_config.get("storage_path", "event_store")
        
        # Core components
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Event] = []
        self.sequence_counter = 0
        
        # Event sourcing components
        self.event_store: IEventStore = FileEventStore(self.storage_path)
        self.version_manager = EventVersionManager()
        self.snapshots: Dict[str, EventSnapshot] = {}
        
        # Metrics
        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "snapshots_created": 0,
            "replays_performed": 0
        }
    
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid event bus configuration"),
            AttributeError: (False, "Missing required event bus parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="enhanced event bus initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced event bus"""
        try:
            self.logger.info("Initializing Enhanced Event Bus...")
            
            await self._load_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for enhanced event bus")
                return False
            
            await self._initialize_event_processing()
            await self._load_event_history()
            
            self.logger.info("✅ Enhanced Event Bus initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced Event Bus initialization failed: {e}")
            return False
    
    async def _load_configuration(self) -> None:
        """Load event bus configuration"""
        try:
            self.event_bus_config.setdefault("processing_interval", 1)
            self.event_bus_config.setdefault("max_history", 1000)
            self.event_bus_config.setdefault("enable_persistence", True)
            self.event_bus_config.setdefault("enable_snapshots", True)
            self.event_bus_config.setdefault("snapshot_frequency", 100)
            self.event_bus_config.setdefault("storage_path", "event_store")
            
            self.processing_interval = self.event_bus_config["processing_interval"]
            self.max_history = self.event_bus_config["max_history"]
            self.enable_persistence = self.event_bus_config["enable_persistence"]
            self.enable_snapshots = self.event_bus_config["enable_snapshots"]
            self.snapshot_frequency = self.event_bus_config["snapshot_frequency"]
            self.storage_path = self.event_bus_config["storage_path"]
            
            self.logger.info("Enhanced event bus configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading enhanced event bus configuration: {e}")
    
    def _validate_configuration(self) -> bool:
        """Validate event bus configuration"""
        try:
            if self.processing_interval <= 0:
                self.logger.error("Invalid processing interval")
                return False
            
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            
            if self.snapshot_frequency <= 0:
                self.logger.error("Invalid snapshot frequency")
                return False
            
            self.logger.info("Enhanced event bus configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False
    
    async def _initialize_event_processing(self) -> None:
        """Initialize event processing components"""
        try:
            self.event_queue = asyncio.Queue()
            self.event_history = []
            self.sequence_counter = 0
            
            # Initialize event store if persistence is enabled
            if self.enable_persistence:
                # Load the latest sequence number from storage
                events = await self.event_store.get_events()
                if events:
                    self.sequence_counter = max(event.metadata.sequence_number for event in events) + 1
            
            self.logger.info("Enhanced event processing initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced event processing: {e}")
    
    async def _load_event_history(self) -> None:
        """Load recent event history from storage"""
        try:
            if self.enable_persistence:
                # Load recent events into memory
                events = await self.event_store.get_events()
                self.event_history = events[-self.max_history:] if events else []
                
                self.logger.info(f"Loaded {len(self.event_history)} events from storage")
            
        except Exception as e:
            self.logger.error(f"Error loading event history: {e}")
    
    async def run(self) -> bool:
        """Run the enhanced event bus"""
        try:
            self.is_running = True
            self.logger.info("🚦 Enhanced Event Bus started")
            
            while self.is_running:
                await self._process_events()
                await asyncio.sleep(self.processing_interval)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in enhanced event bus run: {e}")
            self.is_running = False
            return False
    
    async def _process_events(self) -> None:
        """Process events from the queue"""
        try:
            now = datetime.now(timezone.utc)
            self.status = {"timestamp": now.isoformat(), "status": "running"}
            
            # Update history
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Process events from queue
            events_processed = 0
            while not self.event_queue.empty() and events_processed < 100:  # Limit batch size
                event = await self.event_queue.get()
                success = await self._dispatch_event(event)
                
                if success:
                    self.metrics["events_processed"] += 1
                else:
                    self.metrics["events_failed"] += 1
                
                events_processed += 1
            
            # Create snapshot if needed
            if (self.enable_snapshots and 
                self.metrics["events_processed"] % self.snapshot_frequency == 0 and
                self.metrics["events_processed"] > 0):
                await self._create_snapshot()
            
        except Exception as e:
            self.logger.error(f"Error in enhanced event processing: {e}")
    
    async def _dispatch_event(self, event: Event) -> bool:
        """Dispatch event to subscribers"""
        try:
            # Update event status
            event.metadata.status = EventStatus.PROCESSING
            
            # Validate event schema
            if not self.version_manager.validate_event_schema(event):
                self.logger.error(f"Event {event.metadata.event_id} failed schema validation")
                event.metadata.status = EventStatus.FAILED
                return False
            
            # Get subscribers
            event_type_str = event.event_type.value
            subscribers = self.subscribers.get(event_type_str, [])
            
            # Dispatch to subscribers
            for subscriber in subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event)
                    else:
                        subscriber(event)
                except Exception as e:
                    self.logger.error(f"Error in event subscriber {subscriber.__name__}: {e}")
                    event.metadata.retry_count += 1
            
            # Update event status
            event.metadata.status = EventStatus.PROCESSED
            
            # Persist event if enabled
            if self.enable_persistence:
                await self.event_store.save_event(event)
            
            # Add to event history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
            
            self.logger.debug(f"Event '{event_type_str}' dispatched to {len(subscribers)} subscribers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error dispatching event: {e}")
            event.metadata.status = EventStatus.FAILED
            return False
    
    async def _create_snapshot(self) -> None:
        """Create a system snapshot"""
        try:
            snapshot = EventSnapshot(
                aggregate_id="system",
                sequence_number=self.sequence_counter,
                state_data={
                    "metrics": self.metrics.copy(),
                    "subscribers_count": {k: len(v) for k, v in self.subscribers.items()},
                    "queue_size": self.event_queue.qsize(),
                    "last_processed": datetime.now(timezone.utc).isoformat()
                }
            )
            
            if self.enable_persistence:
                await self.event_store.save_snapshot(snapshot)
            
            self.snapshots["system"] = snapshot
            self.metrics["snapshots_created"] += 1
            
            self.logger.info(f"Created snapshot at sequence {self.sequence_counter}")
            
        except Exception as e:
            self.logger.error(f"Error creating snapshot: {e}")
    
    async def stop(self) -> None:
        """Stop the enhanced event bus"""
        try:
            self.logger.info("🛑 Stopping Enhanced Event Bus...")
            self.is_running = False
            
            # Create final snapshot
            if self.enable_snapshots:
                await self._create_snapshot()
            
            self.status = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "stopped"}
            self.logger.info("✅ Enhanced Event Bus stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping enhanced event bus: {e}")
    
    def subscribe(self, event_type: Union[EventType, str], callback: Callable) -> None:
        """Subscribe to an event type"""
        try:
            event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
            self.subscribers[event_type_str].append(callback)
            self.logger.info(f"Subscriber added for event type: {event_type_str}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to event: {e}")
    
    def unsubscribe(self, event_type: Union[EventType, str], callback: Callable) -> None:
        """Unsubscribe from an event type"""
        try:
            event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
            if event_type_str in self.subscribers:
                self.subscribers[event_type_str] = [
                    sub for sub in self.subscribers[event_type_str] if sub != callback
                ]
                self.logger.info(f"Subscriber removed for event type: {event_type_str}")
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from event: {e}")
    
    async def publish(
        self,
        event_type: Union[EventType, str],
        data: Any,
        source: str = "",
        correlation_id: Optional[str] = None,
        aggregate_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Publish an event to the bus"""
        try:
            # Convert string to EventType if needed
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    self.logger.error(f"Unknown event type: {event_type}")
                    return ""
            
            # Create event metadata
            metadata = EventMetadata(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                source=source,
                correlation_id=correlation_id,
                aggregate_id=aggregate_id,
                sequence_number=self.sequence_counter,
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
            self.sequence_counter += 1
            
            self.logger.debug(f"Event '{event_type.value}' published with ID {event.metadata.event_id}")
            return event.metadata.event_id
            
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
            return ""
    
    async def replay_events(
        self,
        aggregate_id: Optional[str] = None,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[Event]:
        """Replay events from the event store"""
        try:
            if not self.enable_persistence:
                self.logger.warning("Event persistence is disabled, cannot replay events")
                return []
            
            events = await self.event_store.get_events(
                aggregate_id=aggregate_id,
                from_sequence=from_sequence,
                to_sequence=to_sequence,
                event_types=event_types
            )
            
            self.metrics["replays_performed"] += 1
            self.logger.info(f"Replayed {len(events)} events")
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error replaying events: {e}")
            return []
    
    async def rebuild_from_events(
        self,
        aggregate_id: str,
        target_sequence: Optional[int] = None
    ) -> Dict[str, Any]:
        """Rebuild aggregate state from events"""
        try:
            # Get latest snapshot
            snapshot = await self.event_store.get_latest_snapshot(aggregate_id)
            
            start_sequence = 0
            state = {}
            
            if snapshot:
                start_sequence = snapshot.sequence_number + 1
                state = snapshot.state_data.copy()
                self.logger.info(f"Starting rebuild from snapshot at sequence {snapshot.sequence_number}")
            
            # Get events from snapshot point
            events = await self.event_store.get_events(
                aggregate_id=aggregate_id,
                from_sequence=start_sequence,
                to_sequence=target_sequence
            )
            
            # Apply events to rebuild state
            for event in events:
                # This is a simplified example - real state reconstruction would be more complex
                if event.event_type == EventType.TRADE_EXECUTED:
                    if "trades" not in state:
                        state["trades"] = []
                    state["trades"].append(event.data)
                elif event.event_type == EventType.PERFORMANCE_UPDATE:
                    state["performance"] = event.data
            
            self.logger.info(f"Rebuilt state for aggregate {aggregate_id} using {len(events)} events")
            return state
            
        except Exception as e:
            self.logger.error(f"Error rebuilding from events: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current event bus status"""
        return {
            **self.status,
            "metrics": self.metrics.copy(),
            "queue_size": self.event_queue.qsize(),
            "subscribers_count": {k: len(v) for k, v in self.subscribers.items()},
            "persistence_enabled": self.enable_persistence,
            "snapshots_enabled": self.enable_snapshots
        }
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event bus history"""
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Event]:
        """Get event history"""
        history = self.event_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return self.metrics.copy()


# Global instance
enhanced_event_bus: Optional[EnhancedEventBus] = None


async def setup_enhanced_event_bus(config: Optional[Dict[str, Any]] = None) -> Optional[EnhancedEventBus]:
    """Setup the enhanced event bus"""
    try:
        global enhanced_event_bus
        
        if config is None:
            config = {
                "event_bus": {
                    "processing_interval": 1,
                    "max_history": 1000,
                    "enable_persistence": True,
                    "enable_snapshots": True,
                    "snapshot_frequency": 100,
                    "storage_path": "event_store"
                }
            }
        
        enhanced_event_bus = EnhancedEventBus(config)
        success = await enhanced_event_bus.initialize()
        
        if success:
            return enhanced_event_bus
        
        return None
        
    except Exception as e:
        print(f"Error setting up enhanced event bus: {e}")
        return None