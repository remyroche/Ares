# tests/test_enhanced_event_bus.py

import asyncio
import json
import pytest
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.interfaces.enhanced_event_bus import (
    EnhancedEventBus,
    Event,
    EventMetadata,
    EventSnapshot,
    EventStatus,
    EventType,
    EventVersionManager,
    FileEventStore,
    setup_enhanced_event_bus,
)


class TestEvent:
    """Test Event class"""
    
    def test_event_creation(self):
        """Test basic event creation"""
        metadata = EventMetadata(source="test", aggregate_id="test_aggregate")
        event = Event(
            event_type=EventType.MARKET_DATA_RECEIVED,
            data={"symbol": "BTCUSDT", "price": 50000},
            metadata=metadata
        )
        
        assert event.event_type == EventType.MARKET_DATA_RECEIVED
        assert event.data["symbol"] == "BTCUSDT"
        assert event.metadata.source == "test"
        assert event.metadata.aggregate_id == "test_aggregate"
    
    def test_event_to_dict(self):
        """Test event serialization to dictionary"""
        metadata = EventMetadata(source="test", aggregate_id="test_aggregate")
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            data={"symbol": "ETHUSDT", "quantity": 1.0},
            metadata=metadata
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "trade_executed"
        assert event_dict["data"]["symbol"] == "ETHUSDT"
        assert "metadata" in event_dict
        assert event_dict["metadata"]["source"] == "test"
    
    def test_event_from_dict(self):
        """Test event deserialization from dictionary"""
        event_dict = {
            "event_type": "analysis_completed",
            "data": {"result": "bullish"},
            "metadata": {
                "event_id": "test-id",
                "version": "1.0.0",
                "schema_version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "test",
                "correlation_id": None,
                "causation_id": None,
                "aggregate_id": "test_aggregate",
                "sequence_number": 1,
                "retry_count": 0,
                "status": "pending",
                "tags": {}
            }
        }
        
        event = Event.from_dict(event_dict)
        
        assert event.event_type == EventType.ANALYSIS_COMPLETED
        assert event.data["result"] == "bullish"
        assert event.metadata.source == "test"
        assert event.metadata.aggregate_id == "test_aggregate"


class TestEventVersionManager:
    """Test EventVersionManager class"""
    
    def test_validate_event_schema_valid(self):
        """Test schema validation with valid event"""
        version_manager = EventVersionManager()
        
        metadata = EventMetadata(schema_version="1.0.0")
        event = Event(
            event_type=EventType.MARKET_DATA_RECEIVED,
            data={"symbol": "BTCUSDT", "price": 50000, "volume": 100},
            metadata=metadata
        )
        
        assert version_manager.validate_event_schema(event) is True
    
    def test_validate_event_schema_missing_field(self):
        """Test schema validation with missing required field"""
        version_manager = EventVersionManager()
        
        metadata = EventMetadata(schema_version="1.0.0")
        event = Event(
            event_type=EventType.MARKET_DATA_RECEIVED,
            data={"symbol": "BTCUSDT"},  # Missing price and volume
            metadata=metadata
        )
        
        assert version_manager.validate_event_schema(event) is False
    
    def test_migrate_event(self):
        """Test event migration between versions"""
        version_manager = EventVersionManager()
        
        metadata = EventMetadata(schema_version="1.0.0")
        event = Event(
            event_type=EventType.MARKET_DATA_RECEIVED,
            data={"symbol": "BTCUSDT", "price": 50000, "volume": 100},
            metadata=metadata
        )
        
        migrated_event = version_manager.migrate_event(event, "1.1.0")
        
        assert migrated_event.metadata.schema_version == "1.1.0"
        assert "timestamp" in migrated_event.data


class TestFileEventStore:
    """Test FileEventStore class"""
    
    @pytest.mark.asyncio
    async def test_save_and_retrieve_event(self):
        """Test saving and retrieving events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileEventStore(temp_dir)
            
            metadata = EventMetadata(
                event_id="test-id",
                aggregate_id="test_aggregate",
                sequence_number=1
            )
            event = Event(
                event_type=EventType.TRADE_EXECUTED,
                data={"symbol": "BTCUSDT", "quantity": 1.0},
                metadata=metadata
            )
            
            # Save event
            success = await store.save_event(event)
            assert success is True
            
            # Retrieve events
            events = await store.get_events()
            assert len(events) == 1
            assert events[0].metadata.event_id == "test-id"
            assert events[0].data["symbol"] == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_retrieve_events_with_filters(self):
        """Test retrieving events with filters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileEventStore(temp_dir)
            
            # Create multiple events
            events_to_save = []
            for i in range(5):
                metadata = EventMetadata(
                    event_id=f"test-id-{i}",
                    aggregate_id="test_aggregate",
                    sequence_number=i
                )
                event = Event(
                    event_type=EventType.TRADE_EXECUTED,
                    data={"symbol": "BTCUSDT", "quantity": float(i)},
                    metadata=metadata
                )
                events_to_save.append(event)
                await store.save_event(event)
            
            # Test sequence filtering
            events = await store.get_events(from_sequence=2, to_sequence=3)
            assert len(events) == 2
            assert events[0].metadata.sequence_number == 2
            assert events[1].metadata.sequence_number == 3
            
            # Test aggregate filtering
            events = await store.get_events(aggregate_id="test_aggregate")
            assert len(events) == 5
            
            # Test event type filtering
            events = await store.get_events(event_types=[EventType.TRADE_EXECUTED])
            assert len(events) == 5
    
    @pytest.mark.asyncio
    async def test_save_and_retrieve_snapshot(self):
        """Test saving and retrieving snapshots"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FileEventStore(temp_dir)
            
            snapshot = EventSnapshot(
                snapshot_id="test-snapshot",
                aggregate_id="test_aggregate",
                sequence_number=10,
                state_data={"balance": 1000, "trades": []}
            )
            
            # Save snapshot
            success = await store.save_snapshot(snapshot)
            assert success is True
            
            # Retrieve snapshot
            retrieved_snapshot = await store.get_latest_snapshot("test_aggregate")
            assert retrieved_snapshot is not None
            assert retrieved_snapshot.snapshot_id == "test-snapshot"
            assert retrieved_snapshot.state_data["balance"] == 1000


class TestEnhancedEventBus:
    """Test EnhancedEventBus class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test event bus initialization"""
        config = {
            "event_bus": {
                "processing_interval": 1,
                "max_history": 100,
                "enable_persistence": False,  # Disable for testing
                "enable_snapshots": False
            }
        }
        
        event_bus = EnhancedEventBus(config)
        success = await event_bus.initialize()
        
        assert success is True
        assert event_bus.is_running is False
        assert event_bus.processing_interval == 1
        assert event_bus.max_history == 100
    
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        """Test publishing and subscribing to events"""
        config = {
            "event_bus": {
                "processing_interval": 0.1,
                "max_history": 100,
                "enable_persistence": False,
                "enable_snapshots": False
            }
        }
        
        event_bus = EnhancedEventBus(config)
        await event_bus.initialize()
        
        # Create a test subscriber
        received_events = []
        
        def test_subscriber(event: Event):
            received_events.append(event)
        
        # Subscribe to events
        event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, test_subscriber)
        
        # Publish an event
        event_id = await event_bus.publish(
            EventType.MARKET_DATA_RECEIVED,
            {"symbol": "BTCUSDT", "price": 50000},
            source="test"
        )
        
        assert event_id != ""
        
        # Process events
        await event_bus._process_events()
        
        # Check that subscriber received the event
        assert len(received_events) == 1
        assert received_events[0].data["symbol"] == "BTCUSDT"
        assert received_events[0].metadata.source == "test"
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from events"""
        config = {
            "event_bus": {
                "processing_interval": 0.1,
                "max_history": 100,
                "enable_persistence": False,
                "enable_snapshots": False
            }
        }
        
        event_bus = EnhancedEventBus(config)
        await event_bus.initialize()
        
        received_events = []
        
        def test_subscriber(event: Event):
            received_events.append(event)
        
        # Subscribe and then unsubscribe
        event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, test_subscriber)
        event_bus.unsubscribe(EventType.MARKET_DATA_RECEIVED, test_subscriber)
        
        # Publish an event
        await event_bus.publish(
            EventType.MARKET_DATA_RECEIVED,
            {"symbol": "BTCUSDT", "price": 50000}
        )
        
        # Process events
        await event_bus._process_events()
        
        # Check that subscriber did not receive the event
        assert len(received_events) == 0
    
    @pytest.mark.asyncio
    async def test_async_subscriber(self):
        """Test async event subscriber"""
        config = {
            "event_bus": {
                "processing_interval": 0.1,
                "max_history": 100,
                "enable_persistence": False,
                "enable_snapshots": False
            }
        }
        
        event_bus = EnhancedEventBus(config)
        await event_bus.initialize()
        
        received_events = []
        
        async def async_subscriber(event: Event):
            await asyncio.sleep(0.01)  # Simulate async work
            received_events.append(event)
        
        # Subscribe to events
        event_bus.subscribe(EventType.ANALYSIS_COMPLETED, async_subscriber)
        
        # Publish an event
        await event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {"result": "bullish"}
        )
        
        # Process events
        await event_bus._process_events()
        
        # Check that async subscriber received the event
        assert len(received_events) == 1
        assert received_events[0].data["result"] == "bullish"
    
    @pytest.mark.asyncio
    async def test_event_persistence(self):
        """Test event persistence functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 100,
                    "enable_persistence": True,
                    "enable_snapshots": False,
                    "storage_path": temp_dir
                }
            }
            
            event_bus = EnhancedEventBus(config)
            await event_bus.initialize()
            
            # Publish an event
            event_id = await event_bus.publish(
                EventType.TRADE_EXECUTED,
                {"symbol": "ETHUSDT", "quantity": 1.0},
                source="test"
            )
            
            # Process events
            await event_bus._process_events()
            
            # Check that event was persisted
            events = await event_bus.event_store.get_events()
            assert len(events) == 1
            assert events[0].data["symbol"] == "ETHUSDT"
    
    @pytest.mark.asyncio
    async def test_event_replay(self):
        """Test event replay functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 100,
                    "enable_persistence": True,
                    "enable_snapshots": False,
                    "storage_path": temp_dir
                }
            }
            
            event_bus = EnhancedEventBus(config)
            await event_bus.initialize()
            
            # Publish multiple events
            for i in range(5):
                await event_bus.publish(
                    EventType.TRADE_EXECUTED,
                    {"symbol": "BTCUSDT", "quantity": float(i)},
                    source="test",
                    aggregate_id="trader_1"
                )
            
            # Process events
            await event_bus._process_events()
            
            # Replay events
            replayed_events = await event_bus.replay_events(
                aggregate_id="trader_1",
                from_sequence=1,
                to_sequence=3
            )
            
            assert len(replayed_events) == 3
            assert replayed_events[0].metadata.sequence_number == 1
            assert replayed_events[2].metadata.sequence_number == 3
    
    @pytest.mark.asyncio
    async def test_snapshot_creation(self):
        """Test automatic snapshot creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 100,
                    "enable_persistence": True,
                    "enable_snapshots": True,
                    "snapshot_frequency": 3,  # Create snapshot every 3 events
                    "storage_path": temp_dir
                }
            }
            
            event_bus = EnhancedEventBus(config)
            await event_bus.initialize()
            
            # Publish events to trigger snapshot
            for i in range(5):
                await event_bus.publish(
                    EventType.TRADE_EXECUTED,
                    {"symbol": "BTCUSDT", "quantity": float(i)},
                    source="test"
                )
                await event_bus._process_events()
            
            # Check that snapshot was created
            assert event_bus.metrics["snapshots_created"] > 0
            assert "system" in event_bus.snapshots
    
    @pytest.mark.asyncio
    async def test_rebuild_from_events(self):
        """Test rebuilding state from events"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 100,
                    "enable_persistence": True,
                    "enable_snapshots": True,
                    "storage_path": temp_dir
                }
            }
            
            event_bus = EnhancedEventBus(config)
            await event_bus.initialize()
            
            # Publish trade events
            for i in range(3):
                await event_bus.publish(
                    EventType.TRADE_EXECUTED,
                    {"symbol": "BTCUSDT", "quantity": float(i + 1), "price": 50000 + i},
                    source="test",
                    aggregate_id="trader_1"
                )
            
            # Process events
            await event_bus._process_events()
            
            # Rebuild state
            state = await event_bus.rebuild_from_events("trader_1")
            
            assert "trades" in state
            assert len(state["trades"]) == 3
            assert state["trades"][0]["quantity"] == 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_subscriber(self):
        """Test error handling when subscriber throws exception"""
        config = {
            "event_bus": {
                "processing_interval": 0.1,
                "max_history": 100,
                "enable_persistence": False,
                "enable_snapshots": False
            }
        }
        
        event_bus = EnhancedEventBus(config)
        await event_bus.initialize()
        
        def failing_subscriber(event: Event):
            raise ValueError("Test error")
        
        def working_subscriber(event: Event):
            event.metadata.tags["processed"] = "true"
        
        # Subscribe both failing and working subscribers
        event_bus.subscribe(EventType.SYSTEM_ERROR, failing_subscriber)
        event_bus.subscribe(EventType.SYSTEM_ERROR, working_subscriber)
        
        # Publish an event
        await event_bus.publish(EventType.SYSTEM_ERROR, {"error": "test"})
        
        # Process events - should not fail despite exception in one subscriber
        await event_bus._process_events()
        
        # Check that working subscriber still processed the event
        assert len(event_bus.event_history) == 1
        assert event_bus.event_history[0].metadata.status == EventStatus.PROCESSED
    
    def test_get_status(self):
        """Test getting event bus status"""
        config = {
            "event_bus": {
                "processing_interval": 1,
                "max_history": 100,
                "enable_persistence": True,
                "enable_snapshots": True
            }
        }
        
        event_bus = EnhancedEventBus(config)
        status = event_bus.get_status()
        
        assert "metrics" in status
        assert "queue_size" in status
        assert "subscribers_count" in status
        assert "persistence_enabled" in status
        assert "snapshots_enabled" in status
        assert status["persistence_enabled"] is True
        assert status["snapshots_enabled"] is True
    
    def test_get_metrics(self):
        """Test getting event bus metrics"""
        config = {
            "event_bus": {
                "processing_interval": 1,
                "max_history": 100,
                "enable_persistence": False,
                "enable_snapshots": False
            }
        }
        
        event_bus = EnhancedEventBus(config)
        metrics = event_bus.get_metrics()
        
        assert "events_processed" in metrics
        assert "events_failed" in metrics
        assert "snapshots_created" in metrics
        assert "replays_performed" in metrics
        assert metrics["events_processed"] == 0
        assert metrics["events_failed"] == 0


class TestSetupFunction:
    """Test setup_enhanced_event_bus function"""
    
    @pytest.mark.asyncio
    async def test_setup_with_config(self):
        """Test setup with custom config"""
        config = {
            "event_bus": {
                "processing_interval": 2,
                "max_history": 200,
                "enable_persistence": False,
                "enable_snapshots": False
            }
        }
        
        event_bus = await setup_enhanced_event_bus(config)
        
        assert event_bus is not None
        assert event_bus.processing_interval == 2
        assert event_bus.max_history == 200
        assert event_bus.enable_persistence is False
    
    @pytest.mark.asyncio
    async def test_setup_with_default_config(self):
        """Test setup with default config"""
        event_bus = await setup_enhanced_event_bus()
        
        assert event_bus is not None
        assert event_bus.processing_interval == 1
        assert event_bus.max_history == 1000
        assert event_bus.enable_persistence is True


if __name__ == "__main__":
    pytest.main([__file__])