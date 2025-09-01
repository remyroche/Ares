# src/interfaces/enhanced_event_bus.py

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import asyncio
import json
import uuid

from src.utils.logger import system_logger
from src.utils.advanced_decorators import performance_monitor, PerformanceLevel
from src.utils.error_handler import handle_errors, handle_specific_errors
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
    version: str = "1.0.0"
    schema_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    correlation_id: str | None = None
    causation_id: str | None = None
    aggregate_id: str | None = None
    sequence_number: int = 0
    retry_count: int = 0
    status: EventStatus = EventStatus.PENDING
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    """Enhanced event structure with versioning and metadata"""

    event_type: EventType
    data: Any
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization"""
        md = asdict(self.metadata)
        # Ensure timestamp is serialized to ISO format
        if isinstance(self.metadata.timestamp, datetime):
            md["timestamp"] = self.metadata.timestamp.isoformat()
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "metadata": md,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary"""
        metadata_dict = data.get("metadata", {})
        ts_raw = metadata_dict.get("timestamp")
        timestamp = (
            datetime.fromisoformat(ts_raw)
            if isinstance(ts_raw, str)
            else datetime.now(timezone.utc)
        )
        status_value = metadata_dict.get("status", EventStatus.PENDING.value)
        if isinstance(status_value, str):
            try:
                status_enum = EventStatus(status_value)
            except ValueError:
                status_enum = EventStatus.PENDING
        else:
            status_enum = EventStatus.PENDING

        metadata = EventMetadata(
            event_id=metadata_dict.get("event_id", str(uuid.uuid4())),
            version=metadata_dict.get("version", "1.0.0"),
            schema_version=metadata_dict.get("schema_version", "1.0.0"),
            timestamp=timestamp,
            source=metadata_dict.get("source", ""),
            correlation_id=metadata_dict.get("correlation_id"),
            causation_id=metadata_dict.get("causation_id"),
            aggregate_id=metadata_dict.get("aggregate_id"),
            sequence_number=int(metadata_dict.get("sequence_number", 0)),
            retry_count=int(metadata_dict.get("retry_count", 0)),
            status=status_enum,
            tags=metadata_dict.get("tags", {}),
        )

        et = data.get("event_type")
        if isinstance(et, str):
            event_type = EventType(et)
        else:
            event_type = et

        return cls(event_type=event_type, data=data.get("data"), metadata=metadata)


@dataclass
class EventSnapshot:
    """Snapshot of system state at a point in time"""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: str = ""
    sequence_number: int = 0
    state_data: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


class IEventStore(ABC):
    """Interface for event storage implementations"""

    @abstractmethod
    async def save_event(self, event: Event) -> bool:
        """Save an event to the store"""

    @abstractmethod
    @abstractmethod
    async def save_snapshot(self, snapshot: EventSnapshot) -> bool:
        """Save a snapshot to the store"""

    @abstractmethod
    async def get_latest_snapshot(self, aggregate_id: str) -> EventSnapshot | None:
        """Get the latest snapshot for an aggregate"""


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
            self.logger.error(failed(f"Failed to save event: {e}"))
            return False

    async def save_snapshot(self, snapshot: EventSnapshot) -> bool:
        """Save a snapshot to file storage"""
        try:
            snapshot_file = (
                self.snapshots_path
                / f"snapshot_{snapshot.aggregate_id}_{snapshot.sequence_number}.json"
            )

            snapshot_data = asdict(snapshot)
            # Convert datetime to string for JSON serialization
            if isinstance(snapshot.timestamp, datetime):
                snapshot_data["timestamp"] = snapshot.timestamp.isoformat()

            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_data, f, indent=2, default=str)

            self.logger.debug(
                f"Saved snapshot {snapshot.snapshot_id} to {snapshot_file}",
            )
            return True

        except Exception as e:
            self.logger.error(failed(f"Failed to save snapshot: {e}"))
            return False

    async def get_latest_snapshot(self, aggregate_id: str) -> EventSnapshot | None:
        """Get the latest snapshot for an aggregate"""
        try:
            latest_snapshot: EventSnapshot | None = None
            latest_sequence = -1

            # Find the latest snapshot file for the aggregate
            for snapshot_file in self.snapshots_path.glob(
                f"snapshot_{aggregate_id}_*.json",
            ):
                with open(snapshot_file, encoding="utf-8") as f:
                    snapshot_data = json.load(f)

                    sequence_number = int(snapshot_data.get("sequence_number", 0))
                    if sequence_number > latest_sequence:
                        latest_sequence = sequence_number
                        ts = snapshot_data.get("timestamp")
                        if isinstance(ts, str):
                            snapshot_data["timestamp"] = datetime.fromisoformat(ts)
                        latest_snapshot = EventSnapshot(**snapshot_data)

            return latest_snapshot

        except Exception as e:
            self.logger.error(failed(f"Failed to retrieve latest snapshot: {e}"))
            return None


class EventVersionManager:
    """Manages event schema versioning and migration"""

    def __init__(self):
        self.logger = system_logger.getChild("EventVersionManager")
        self.version_mappings: dict[str, dict[str, Any]] = {}
        self._register_default_versions()

    def _register_default_versions(self):
        """Register default version mappings"""
        # Example version mappings for backward compatibility
        self.version_mappings = {
            "1.0.0": {
                "market_data_received": {
                    "required_fields": ["symbol", "price", "volume"],
                    "optional_fields": ["timestamp", "bid", "ask"],
                },
                "trade_executed": {
                    "required_fields": ["symbol", "side", "quantity", "price"],
                    "optional_fields": ["order_id", "commission"],
                },
            },
            "1.1.0": {
                "market_data_received": {
                    "required_fields": ["symbol", "price", "volume", "timestamp"],
                    "optional_fields": ["bid", "ask", "spread"],
                },
                "trade_executed": {
                    "required_fields": [
                        "symbol",
                        "side",
                        "quantity",
                        "price",
                        "order_id",
                    ],
                    "optional_fields": ["commission", "fees"],
                },
            },
        }

    def validate_event_schema(self, event: Event) -> bool:
        """Validate event against its schema version"""
        try:
            version = event.metadata.schema_version
            event_type = event.event_type.value

            if version not in self.version_mappings:
                self.logger.warning(warning(f"Unknown schema version: {version}"))
                return True  # Allow unknown versions for forward compatibility

            schema = self.version_mappings[version].get(event_type)
            if not schema:
                self.logger.warning(warning(f"No schema defined for event type: {event_type}"))
                return True

            # Validate required fields
            required_fields = schema.get("required_fields", [])
            if isinstance(event.data, dict):
                for field_name in required_fields:
                    if field_name not in event.data:
                        self.logger.error(
                            f"Missing required field '{field_name}' in event {event.metadata.event_id}",
                        )
                        return False

            return True

        except Exception as e:
            self.logger.error(validation_error(f"Schema validation error: {e}"))
            return False


class EnhancedEventBus:
    """
    Enhanced Event Bus with event sourcing, versioning, and persistence capabilities
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedEventBus")
        self.is_running = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

        # Configuration
        self.event_bus_config = self.config.get("event_bus", {})
        self.processing_interval = self.event_bus_config.get("processing_interval", 1)
        self.max_history = self.event_bus_config.get("max_history", 1000)
        self.enable_persistence = self.event_bus_config.get("enable_persistence", True)
        self.enable_snapshots = self.event_bus_config.get("enable_snapshots", True)
        self.snapshot_frequency = self.event_bus_config.get("snapshot_frequency", 100)
        self.storage_path = self.event_bus_config.get("storage_path", "event_store")

        # Core components
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: list[Event] = []
        self.sequence_counter = 0

        # Event sourcing components
        self.event_store: IEventStore | None = (
            FileEventStore(self.storage_path) if self.enable_persistence else None
        )
        self.version_manager = EventVersionManager()
        self.snapshots: dict[str, EventSnapshot] = {}

        # Metrics
        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "snapshots_created": 0,
            "replays_performed": 0,
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
    @performance_monitor(level=PerformanceLevel.DETAILED)
    @performance_monitor(level=PerformanceLevel.BASIC)
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
            self.logger.error(
                error(f"Error loading enhanced event bus configuration: {e}"),
            )

    def _validate_configuration(self) -> bool:
        """Validate event bus configuration"""
        try:
            if self.processing_interval <= 0:
                self.logger.error(invalid("Invalid processing interval"))
                return False

            if self.max_history <= 0:
                self.logger.error(invalid("Invalid max history"))
                return False

            if self.snapshot_frequency <= 0:
                self.logger.error(invalid("Invalid snapshot frequency"))
                return False

            self.logger.info("Enhanced event bus configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(error(f"Error validating configuration: {e}"))
            return False

    @performance_monitor(level=PerformanceLevel.BASIC)
    async def _initialize_event_processing(self) -> None:
        """Initialize event processing components"""
        try:
            self.event_queue = asyncio.Queue()
            self.event_history = []
            self.sequence_counter = 0

            # Initialize event store if persistence is enabled
            if self.enable_persistence and self.event_store is not None:
                # Load the latest sequence number from storage
                events = await self.event_store.get_events()
                if events:
                    self.sequence_counter = (
                        max(event.metadata.sequence_number for event in events) + 1
                    )

            self.logger.info("Enhanced event processing initialized successfully")

        except Exception as e:
            self.logger.error(
                initialization_error(
                    f"Error initializing enhanced event processing: {e}",
                ),
            )

    @performance_monitor(level=PerformanceLevel.BASIC)
    async def _load_event_history(self) -> None:
        """Load recent event history from storage"""
        try:
            if self.enable_persistence and self.event_store is not None:
                # Load recent events into memory
                events = await self.event_store.get_events()
                self.event_history = events[-self.max_history :] if events else []

                self.logger.info(
                    f"Loaded {len(self.event_history)} events from storage",
                )

        except Exception as e:
            self.logger.error(error(f"Error loading event history: {e}"))

    @performance_monitor(level=PerformanceLevel.DETAILED)
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
            self.logger.error(error(f"Error in enhanced event bus run: {e}"))
            self.is_running = False
            return False

    @performance_monitor(level=PerformanceLevel.DETAILED)
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
            while not self.event_queue.empty():
                event = await self.event_queue.get()
                success = await self._dispatch_event(event)

                if success:
                    self.metrics["events_processed"] += 1
                else:
                    self.metrics["events_failed"] += 1

                events_processed += 1

            # Create snapshot if needed
            if (
                self.enable_snapshots
                and self.metrics["events_processed"] % self.snapshot_frequency == 0
                and self.metrics["events_processed"] > 0
            ):
                await self._create_snapshot()

        except Exception as e:
            self.logger.error(error(f"Error in enhanced event processing: {e}"))

    @performance_monitor(level=PerformanceLevel.DETAILED)
    async def _dispatch_event(self, event: Event) -> bool:
        """Dispatch event to subscribers"""
        try:
            # Update event status
            event.metadata.status = EventStatus.PROCESSING

            # Validate event schema
            if not self.version_manager.validate_event_schema(event):
                self.logger.error(
                    f"Event {event.metadata.event_id} failed schema validation",
                )
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
                    self.logger.exception(
                        f"Error in event subscriber {getattr(subscriber, '__name__', str(subscriber))}: {e}",
                    )
                    event.metadata.retry_count += 1

            # Update event status
            event.metadata.status = EventStatus.PROCESSED

            # Persist event if enabled
            if self.enable_persistence and self.event_store is not None:
                await self.event_store.save_event(event)

            # Add to event history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)

            self.logger.debug(
                f"Event '{event_type_str}' dispatched to {len(subscribers)} subscribers",
            )
            return True

        except Exception as e:
            self.logger.error(error(f"Error dispatching event: {e}"))
            event.metadata.status = EventStatus.FAILED
            return False

    @performance_monitor(level=PerformanceLevel.BASIC)
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
                    "last_processed": datetime.now(timezone.utc).isoformat(),
                },
            )

            if self.enable_persistence and self.event_store is not None:
                await self.event_store.save_snapshot(snapshot)

            self.snapshots["system"] = snapshot
            self.metrics["snapshots_created"] += 1

            self.logger.info(f"Created snapshot at sequence {self.sequence_counter}")

        except Exception as e:
            self.logger.error(error(f"Error creating snapshot: {e}"))

    @performance_monitor(level=PerformanceLevel.BASIC)
    @performance_monitor(level=PerformanceLevel.DETAILED)
    @performance_monitor(level=PerformanceLevel.BASIC)
    @performance_monitor(level=PerformanceLevel.BASIC)

# Global instance
enhanced_event_bus: EnhancedEventBus | None = None

