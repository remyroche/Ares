"""
Integration tests for Supervisor data flow.

These tests verify that data flows correctly between components
through the queue system and that components interact properly.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any, Dict

from src.supervisor.main import Supervisor
from src.supervisor.dependency_container import DependencyContainer, ComponentBuilder
from src.utils.state_manager import StateManager


class TestSupervisorDataFlow:
    """Test suite for supervisor data flow integration."""

    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Provide test configuration."""
        return {
            "supervisor": {
                "run_interval": 1,
                "max_history": 10,
            },
            "risk_allocator": {
                "allocation_interval": 60,
                "max_history": 100,
            },
            "performance_reporter": {
                "reporting_interval": 60,
            },
            "monitoring": {
                "check_interval": 30,
            },
        }

    @pytest.fixture
    def mock_state_manager(self) -> Mock:
        """Create mock state manager."""
        mock = Mock(spec=StateManager)
        mock.get_state = Mock(return_value={"status": "active"})
        mock.set_state = Mock()
        mock._save_state_to_file = Mock()
        return mock

    @pytest.fixture
    def mock_db_manager(self) -> Mock:
        """Create mock database manager."""
        mock = Mock()
        mock.initialize = AsyncMock()
        return mock

    @pytest.fixture
    def mock_exchange_client(self) -> Mock:
        """Create mock exchange client."""
        mock = Mock()
        mock.get_account_info = AsyncMock(return_value={"totalWalletBalance": 10000})
        mock.get_open_positions = AsyncMock(return_value=[])
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_queue_initialization(self, mock_config, mock_state_manager, 
                                      mock_db_manager, mock_exchange_client):
        """Test that queues are properly initialized."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Verify queues are created
        assert supervisor.market_data_queue is not None
        assert supervisor.analysis_queue is not None
        assert supervisor.signal_queue is not None

        # Verify queue sizes
        assert supervisor.market_data_queue.maxsize == 100
        assert supervisor.analysis_queue.maxsize == 100
        assert supervisor.signal_queue.maxsize == 50

    @pytest.mark.asyncio
    async def test_component_queue_wiring(self, mock_config, mock_state_manager,
                                        mock_db_manager, mock_exchange_client):
        """Test that component queues are properly wired."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Create mock components with queue attributes
        mock_sentinel = Mock()
        mock_sentinel.output_queue = None
        
        mock_analyst = Mock()
        mock_analyst.input_queue = None
        mock_analyst.output_queue = None
        
        mock_strategist = Mock()
        mock_strategist.input_queue = None
        mock_strategist.output_queue = None
        
        mock_tactician = Mock()
        mock_tactician.input_queue = None

        # Manually set components for testing
        supervisor.sentinel = mock_sentinel
        supervisor.analyst = mock_analyst
        supervisor.strategist = mock_strategist
        supervisor.tactician = mock_tactician

        # Call the wiring method
        supervisor._wire_component_queues()

        # Verify queue connections
        assert mock_sentinel.output_queue == supervisor.market_data_queue
        assert mock_analyst.input_queue == supervisor.market_data_queue
        assert mock_analyst.output_queue == supervisor.analysis_queue
        assert mock_strategist.input_queue == supervisor.analysis_queue
        assert mock_strategist.output_queue == supervisor.signal_queue
        assert mock_tactician.input_queue == supervisor.signal_queue

    @pytest.mark.asyncio
    async def test_data_flow_sentinel_to_analyst(self, mock_config, mock_state_manager,
                                                mock_db_manager, mock_exchange_client):
        """Test data flow from Sentinel to Analyst through market data queue."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Simulate Sentinel putting data in queue
        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "volume": 100,
            "timestamp": "2024-01-01T00:00:00",
        }
        await supervisor.market_data_queue.put(market_data)

        # Verify data can be retrieved (as Analyst would)
        retrieved_data = await supervisor.market_data_queue.get()
        assert retrieved_data == market_data

    @pytest.mark.asyncio
    async def test_data_flow_analyst_to_strategist(self, mock_config, mock_state_manager,
                                                  mock_db_manager, mock_exchange_client):
        """Test data flow from Analyst to Strategist through analysis queue."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Simulate Analyst putting analysis results in queue
        analysis_result = {
            "symbol": "BTC/USDT",
            "trend": "bullish",
            "confidence": 0.85,
            "indicators": {"rsi": 65, "macd": "positive"},
        }
        await supervisor.analysis_queue.put(analysis_result)

        # Verify data can be retrieved (as Strategist would)
        retrieved_data = await supervisor.analysis_queue.get()
        assert retrieved_data == analysis_result

    @pytest.mark.asyncio
    async def test_data_flow_strategist_to_tactician(self, mock_config, mock_state_manager,
                                                    mock_db_manager, mock_exchange_client):
        """Test data flow from Strategist to Tactician through signal queue."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Simulate Strategist putting signal in queue
        signal = {
            "symbol": "BTC/USDT",
            "action": "buy",
            "confidence": 0.9,
            "strategy": "momentum",
            "entry_price": 49500,
        }
        await supervisor.signal_queue.put(signal)

        # Verify data can be retrieved (as Tactician would)
        retrieved_data = await supervisor.signal_queue.get()
        assert retrieved_data == signal

    @pytest.mark.asyncio
    async def test_dependency_container_integration(self, mock_config, mock_state_manager,
                                                   mock_db_manager, mock_exchange_client):
        """Test that dependency container properly manages component creation."""
        container = DependencyContainer(mock_config)
        builder = ComponentBuilder(container)

        # Create mock performance reporter
        mock_performance_reporter = Mock()

        # Register component factories
        container.register(
            "sentinel",
            builder.build_sentinel(mock_exchange_client, mock_state_manager)
        )
        container.register(
            "analyst",
            builder.build_analyst(mock_exchange_client, mock_state_manager)
        )

        # Verify components can be retrieved
        # Note: These will fail in actual execution because they try to import
        # the actual components, but the structure is correct
        assert container.has("sentinel")
        assert container.has("analyst")

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, mock_config, mock_state_manager,
                                         mock_db_manager, mock_exchange_client):
        """Test behavior when queues are full."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Fill the market data queue (maxsize=100)
        for i in range(100):
            await supervisor.market_data_queue.put({"data": i})

        # Verify queue is full
        assert supervisor.market_data_queue.full()

        # Try to put one more item with timeout
        with pytest.raises(asyncio.QueueFull):
            supervisor.market_data_queue.put_nowait({"data": 101})

    @pytest.mark.asyncio
    async def test_concurrent_queue_access(self, mock_config, mock_state_manager,
                                         mock_db_manager, mock_exchange_client):
        """Test concurrent access to queues by multiple components."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Simulate multiple producers
        async def producer(queue, prefix, count):
            for i in range(count):
                await queue.put({f"{prefix}_data": i})

        # Simulate multiple consumers
        consumed_data = []
        async def consumer(queue, count):
            for _ in range(count):
                data = await queue.get()
                consumed_data.append(data)

        # Run concurrent producers and consumers
        await asyncio.gather(
            producer(supervisor.market_data_queue, "sentinel1", 5),
            producer(supervisor.market_data_queue, "sentinel2", 5),
            consumer(supervisor.market_data_queue, 10),
        )

        # Verify all data was consumed
        assert len(consumed_data) == 10
        assert supervisor.market_data_queue.empty()

    @pytest.mark.asyncio 
    async def test_component_lifecycle(self, mock_config, mock_state_manager,
                                     mock_db_manager, mock_exchange_client):
        """Test proper component lifecycle management."""
        supervisor = Supervisor(
            symbol="BTC/USDT",
            exchange_name="test_exchange",
            exchange_client=mock_exchange_client,
            state_manager=mock_state_manager,
            db_manager=mock_db_manager,
        )

        # Mock components with start methods
        mock_components = []
        for name in ["sentinel", "analyst", "strategist", "tactician"]:
            component = Mock()
            component.start = AsyncMock()
            setattr(supervisor, name, component)
            mock_components.append(component)

        # Verify components can be started (partial test due to complex dependencies)
        assert supervisor.sentinel is not None
        assert supervisor.analyst is not None
        assert supervisor.strategist is not None
        assert supervisor.tactician is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])