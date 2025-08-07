# tests/test_event_bus_integration.py

"""
Integration tests for the Enhanced Event Bus system.
Tests the complete event flow from publishing to persistence and replay.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import List

from src.interfaces.enhanced_event_bus import (
    EnhancedEventBus,
    Event,
    EventType,
    setup_enhanced_event_bus,
)


class MockTradingComponent:
    """Mock trading component for integration testing"""
    
    def __init__(self, name: str, event_bus: EnhancedEventBus):
        self.name = name
        self.event_bus = event_bus
        self.received_events: List[Event] = []
        self.correlation_map = {}
        
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, self.handle_market_data)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.handle_trade_executed)
        self.event_bus.subscribe(EventType.PERFORMANCE_UPDATE, self.handle_performance_update)
    
    async def handle_market_data(self, event: Event):
        """Handle market data events"""
        self.received_events.append(event)
        
        # Simulate trading decision based on market data
        if event.data.get("price", 0) < 50000:
            await self.event_bus.publish(
                EventType.TRADE_DECISION_MADE,
                {
                    "symbol": event.data.get("symbol"),
                    "action": "buy",
                    "quantity": 0.1,
                    "reason": "price_below_threshold"
                },
                source=self.name,
                correlation_id=event.metadata.correlation_id,
                aggregate_id=f"{self.name}_decisions"
            )
    
    async def handle_trade_executed(self, event: Event):
        """Handle trade execution events"""
        self.received_events.append(event)
        
        # Track correlation for testing
        if event.metadata.correlation_id:
            self.correlation_map[event.metadata.correlation_id] = event
        
        # Simulate performance update
        await self.event_bus.publish(
            EventType.PERFORMANCE_UPDATE,
            {
                "portfolio_value": 10000 + len(self.received_events) * 100,
                "trade_count": len([e for e in self.received_events if e.event_type == EventType.TRADE_EXECUTED]),
                "component": self.name
            },
            source=self.name,
            correlation_id=event.metadata.correlation_id
        )
    
    async def handle_performance_update(self, event: Event):
        """Handle performance update events"""
        self.received_events.append(event)


@pytest.mark.asyncio
class TestEventBusIntegration:
    """Integration tests for the enhanced event bus"""
    
    async def test_complete_event_flow(self):
        """Test complete event flow from publishing to processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 100,
                    "enable_persistence": True,
                    "enable_snapshots": False,  # Disable for faster testing
                    "storage_path": temp_dir
                }
            }
            
            # Setup event bus
            event_bus = await setup_enhanced_event_bus(config)
            assert event_bus is not None
            
            # Create mock components
            trader1 = MockTradingComponent("Trader1", event_bus)
            trader2 = MockTradingComponent("Trader2", event_bus)
            
            # Start event bus
            event_bus_task = asyncio.create_task(event_bus.run())
            
            try:
                # Publish market data events
                correlation_id = "test_correlation_1"
                
                await event_bus.publish(
                    EventType.MARKET_DATA_RECEIVED,
                    {
                        "symbol": "BTCUSDT",
                        "price": 49500,  # Below threshold to trigger buy
                        "volume": 100,
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    source="MarketDataProvider",
                    correlation_id=correlation_id,
                    aggregate_id="market_data_btc"
                )
                
                # Allow processing time
                await asyncio.sleep(0.5)
                
                # Simulate trade execution
                await event_bus.publish(
                    EventType.TRADE_EXECUTED,
                    {
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "quantity": 0.1,
                        "price": 49500,
                        "order_id": "test_order_1"
                    },
                    source="ExchangeAPI",
                    correlation_id=correlation_id,
                    aggregate_id="trader1_trades"
                )
                
                # Allow more processing time
                await asyncio.sleep(0.5)
                
                # Verify events were received and processed
                assert len(trader1.received_events) > 0
                assert len(trader2.received_events) > 0
                
                # Verify correlation tracking
                assert correlation_id in trader1.correlation_map or correlation_id in trader2.correlation_map
                
                # Verify different event types were processed
                event_types = {event.event_type for event in trader1.received_events + trader2.received_events}
                assert EventType.MARKET_DATA_RECEIVED in event_types
                assert EventType.TRADE_EXECUTED in event_types
                assert EventType.PERFORMANCE_UPDATE in event_types
                
                # Verify events were persisted
                stored_events = await event_bus.event_store.get_events()
                assert len(stored_events) > 0
                
                # Verify correlation chain
                correlated_events = [e for e in stored_events if e.metadata.correlation_id == correlation_id]
                assert len(correlated_events) >= 2  # At least market data + trade executed
                
            finally:
                await event_bus.stop()
                event_bus_task.cancel()
                
                try:
                    await event_bus_task
                except asyncio.CancelledError:
                    pass
    
    async def test_event_replay_and_reconstruction(self):
        """Test event replay and state reconstruction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 100,
                    "enable_persistence": True,
                    "enable_snapshots": True,
                    "snapshot_frequency": 5,  # Snapshot every 5 events
                    "storage_path": temp_dir
                }
            }
            
            # Setup and run event bus
            event_bus = await setup_enhanced_event_bus(config)
            assert event_bus is not None
            
            event_bus_task = asyncio.create_task(event_bus.run())
            
            try:
                aggregate_id = "integration_test_trader"
                
                # Publish a series of events
                for i in range(10):
                    await event_bus.publish(
                        EventType.TRADE_EXECUTED,
                        {
                            "symbol": "BTCUSDT",
                            "side": "buy" if i % 2 == 0 else "sell",
                            "quantity": 0.1 * (i + 1),
                            "price": 50000 + i * 100,
                            "order_id": f"integration_order_{i}"
                        },
                        source="IntegrationTest",
                        aggregate_id=aggregate_id
                    )
                
                # Allow processing time
                await asyncio.sleep(1.0)
                
                # Test event replay
                replayed_events = await event_bus.replay_events(
                    aggregate_id=aggregate_id,
                    from_sequence=2,
                    to_sequence=7
                )
                
                assert len(replayed_events) == 6  # Sequences 2-7 inclusive
                assert all(e.metadata.aggregate_id == aggregate_id for e in replayed_events)
                assert replayed_events[0].metadata.sequence_number == 2
                assert replayed_events[-1].metadata.sequence_number == 7
                
                # Test state reconstruction
                state = await event_bus.rebuild_from_events(aggregate_id)
                assert "trades" in state
                assert len(state["trades"]) == 10
                
                # Verify trade data
                buy_trades = [t for t in state["trades"] if t.get("side") == "buy"]
                sell_trades = [t for t in state["trades"] if t.get("side") == "sell"]
                assert len(buy_trades) == 5
                assert len(sell_trades) == 5
                
                # Test replay with event type filter
                trade_events = await event_bus.replay_events(
                    aggregate_id=aggregate_id,
                    event_types=[EventType.TRADE_EXECUTED]
                )
                assert len(trade_events) == 10
                assert all(e.event_type == EventType.TRADE_EXECUTED for e in trade_events)
                
                # Verify snapshot was created (due to snapshot_frequency=5)
                assert event_bus.metrics["snapshots_created"] > 0
                
            finally:
                await event_bus.stop()
                event_bus_task.cancel()
                
                try:
                    await event_bus_task
                except asyncio.CancelledError:
                    pass
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and system recovery"""
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
            
            event_bus = await setup_enhanced_event_bus(config)
            assert event_bus is not None
            
            # Create a failing subscriber
            failure_count = 0
            processed_events = []
            
            def failing_subscriber(event: Event):
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 2:
                    raise ValueError(f"Simulated failure {failure_count}")
                processed_events.append(event)
            
            def working_subscriber(event: Event):
                # This should always work
                event.metadata.tags["processed_by_working"] = "true"
            
            # Subscribe both failing and working subscribers
            event_bus.subscribe(EventType.SYSTEM_ERROR, failing_subscriber)
            event_bus.subscribe(EventType.SYSTEM_ERROR, working_subscriber)
            
            event_bus_task = asyncio.create_task(event_bus.run())
            
            try:
                # Publish events that will trigger failures
                for i in range(5):
                    await event_bus.publish(
                        EventType.SYSTEM_ERROR,
                        {"error_code": f"TEST_ERROR_{i}", "message": f"Test error {i}"},
                        source="IntegrationTest"
                    )
                
                # Allow processing time
                await asyncio.sleep(1.0)
                
                # Verify that working subscriber processed all events
                events_in_history = [e for e in event_bus.event_history if e.event_type == EventType.SYSTEM_ERROR]
                working_processed = [e for e in events_in_history if e.metadata.tags.get("processed_by_working") == "true"]
                assert len(working_processed) == 5
                
                # Verify that failing subscriber eventually succeeded (after 2 failures)
                assert len(processed_events) >= 3  # Should process at least 3 events after 2 failures
                
                # Verify events were still persisted despite subscriber failures
                stored_events = await event_bus.event_store.get_events(
                    event_types=[EventType.SYSTEM_ERROR]
                )
                assert len(stored_events) == 5
                
            finally:
                await event_bus.stop()
                event_bus_task.cancel()
                
                try:
                    await event_bus_task
                except asyncio.CancelledError:
                    pass
    
    async def test_metrics_and_monitoring(self):
        """Test metrics collection and monitoring functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "event_bus": {
                    "processing_interval": 0.1,
                    "max_history": 50,
                    "enable_persistence": True,
                    "enable_snapshots": True,
                    "snapshot_frequency": 3,
                    "storage_path": temp_dir
                }
            }
            
            event_bus = await setup_enhanced_event_bus(config)
            assert event_bus is not None
            
            # Track initial metrics
            initial_metrics = event_bus.get_metrics()
            assert initial_metrics["events_processed"] == 0
            assert initial_metrics["events_failed"] == 0
            assert initial_metrics["snapshots_created"] == 0
            assert initial_metrics["replays_performed"] == 0
            
            event_bus_task = asyncio.create_task(event_bus.run())
            
            try:
                # Publish events to generate metrics
                for i in range(10):
                    await event_bus.publish(
                        EventType.PERFORMANCE_UPDATE,
                        {"metric": f"test_metric_{i}", "value": i * 10},
                        source="MetricsTest",
                        aggregate_id="metrics_aggregate"
                    )
                
                # Allow processing time
                await asyncio.sleep(1.0)
                
                # Check updated metrics
                metrics = event_bus.get_metrics()
                assert metrics["events_processed"] == 10
                assert metrics["events_failed"] == 0
                
                # Snapshots should have been created (every 3 events)
                assert metrics["snapshots_created"] > 0
                
                # Test replay metrics
                await event_bus.replay_events(aggregate_id="metrics_aggregate")
                updated_metrics = event_bus.get_metrics()
                assert updated_metrics["replays_performed"] == 1
                
                # Test status information
                status = event_bus.get_status()
                assert "metrics" in status
                assert "queue_size" in status
                assert "subscribers_count" in status
                assert "persistence_enabled" in status
                assert "snapshots_enabled" in status
                
                assert status["persistence_enabled"] is True
                assert status["snapshots_enabled"] is True
                assert status["metrics"]["events_processed"] == 10
                
                # Test history
                history = event_bus.get_history(limit=5)
                assert len(history) <= 5
                assert all("timestamp" in entry for entry in history)
                
                event_history = event_bus.get_event_history(limit=5)
                assert len(event_history) <= 5
                assert all(isinstance(event, Event) for event in event_history)
                
            finally:
                await event_bus.stop()
                event_bus_task.cancel()
                
                try:
                    await event_bus_task
                except asyncio.CancelledError:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])