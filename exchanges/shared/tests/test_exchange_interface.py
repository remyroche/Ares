"""
Unit tests for ExchangeInterface.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from exchanges.base_exchange.exchange_interface import (
    IExchange, IExchangeAdapter, IMessageRouter, IResponseHandler,
    IEventPublisher, IExchangeManager, ExchangeType, ExchangeStatus,
    OrderSide, OrderType, OrderStatus, ExchangeConfig, OrderRequest,
    OrderResponse, MarketDataPoint, ExchangeMetrics, ExchangeEvent,
    ExchangeEventData
)


class TestExchangeInterface:
    """Test cases for ExchangeInterface."""

    def test_exchange_type_enum(self):
        """Test ExchangeType enum values."""
        assert ExchangeType.SPOT.value == "spot"
        assert ExchangeType.FUTURES.value == "futures"
        assert ExchangeType.MARGIN.value == "margin"
        assert ExchangeType.OPTIONS.value == "options"
        assert ExchangeType.DERIVATIVES.value == "derivatives"

    def test_exchange_status_enum(self):
        """Test ExchangeStatus enum values."""
        assert ExchangeStatus.DISCONNECTED.value == "disconnected"
        assert ExchangeStatus.CONNECTING.value == "connecting"
        assert ExchangeStatus.CONNECTED.value == "connected"
        assert ExchangeStatus.ERROR.value == "error"
        assert ExchangeStatus.MAINTENANCE.value == "maintenance"

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
        assert OrderStatus.FAILED.value == "failed"

    def test_exchange_config(self):
        """Test ExchangeConfig dataclass."""
        config = ExchangeConfig(
            name="test_exchange",
            exchange_type=ExchangeType.SPOT,
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://api.test.com",
            sandbox=True,
            rate_limits={"trading": 10},
            supported_symbols=["BTCUSDT", "ETHUSDT"],
            features={"futures": True}
        )
        
        assert config.name == "test_exchange"
        assert config.exchange_type == ExchangeType.SPOT
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.base_url == "https://api.test.com"
        assert config.sandbox is True
        assert config.rate_limits == {"trading": 10}
        assert config.supported_symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.features == {"futures": True}

    def test_order_request(self):
        """Test OrderRequest dataclass."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            stop_price=49000.0,
            time_in_force="GTC",
            client_order_id="test_123",
            metadata={"strategy": "test"}
        )
        
        assert request.symbol == "BTCUSDT"
        assert request.side == OrderSide.BUY
        assert request.order_type == OrderType.LIMIT
        assert request.quantity == 0.001
        assert request.price == 50000.0
        assert request.stop_price == 49000.0
        assert request.time_in_force == "GTC"
        assert request.client_order_id == "test_123"
        assert request.metadata == {"strategy": "test"}

    def test_order_response(self):
        """Test OrderResponse dataclass."""
        response = OrderResponse(
            order_id="ex_123",
            exchange_order_id="exchange_123",
            status=OrderStatus.FILLED,
            filled_quantity=0.001,
            remaining_quantity=0.0,
            average_price=50000.0,
            commission=0.001,
            commission_asset="BTC",
            executed_at="2024-01-01T12:00:00Z",
            error_message=None,
            metadata={"exchange": "test"}
        )
        
        assert response.order_id == "ex_123"
        assert response.exchange_order_id == "exchange_123"
        assert response.status == OrderStatus.FILLED
        assert response.filled_quantity == 0.001
        assert response.remaining_quantity == 0.0
        assert response.average_price == 50000.0
        assert response.commission == 0.001
        assert response.commission_asset == "BTC"
        assert response.executed_at == "2024-01-01T12:00:00Z"
        assert response.error_message is None
        assert response.metadata == {"exchange": "test"}

    def test_market_data_point(self):
        """Test MarketDataPoint dataclass."""
        data_point = MarketDataPoint(
            symbol="BTCUSDT",
            timestamp="2024-01-01T12:00:00Z",
            open=49000.0,
            high=51000.0,
            low=48500.0,
            close=50000.0,
            volume=100.0,
            interval="1h"
        )
        
        assert data_point.symbol == "BTCUSDT"
        assert data_point.timestamp == "2024-01-01T12:00:00Z"
        assert data_point.open == 49000.0
        assert data_point.high == 51000.0
        assert data_point.low == 48500.0
        assert data_point.close == 50000.0
        assert data_point.volume == 100.0
        assert data_point.interval == "1h"

    def test_exchange_metrics(self):
        """Test ExchangeMetrics dataclass."""
        metrics = ExchangeMetrics(
            exchange_name="test_exchange",
            connection_status=ExchangeStatus.CONNECTED,
            last_heartbeat="2024-01-01T12:00:00Z",
            request_count=1000,
            error_count=5,
            average_response_time=0.1,
            success_rate=99.5,
            active_orders=10,
            total_volume_24h=1000000.0
        )
        
        assert metrics.exchange_name == "test_exchange"
        assert metrics.connection_status == ExchangeStatus.CONNECTED
        assert metrics.last_heartbeat == "2024-01-01T12:00:00Z"
        assert metrics.request_count == 1000
        assert metrics.error_count == 5
        assert metrics.average_response_time == 0.1
        assert metrics.success_rate == 99.5
        assert metrics.active_orders == 10
        assert metrics.total_volume_24h == 1000000.0

    def test_exchange_event_enum(self):
        """Test ExchangeEvent enum values."""
        assert ExchangeEvent.CONNECTED.value == "connected"
        assert ExchangeEvent.DISCONNECTED.value == "disconnected"
        assert ExchangeEvent.ERROR.value == "error"
        assert ExchangeEvent.ORDER_EXECUTED.value == "order_executed"
        assert ExchangeEvent.ORDER_FAILED.value == "order_failed"
        assert ExchangeEvent.DATA_RECEIVED.value == "data_received"
        assert ExchangeEvent.RATE_LIMIT_EXCEEDED.value == "rate_limit_exceeded"
        assert ExchangeEvent.MAINTENANCE_MODE.value == "maintenance_mode"

    def test_exchange_event_data(self):
        """Test ExchangeEventData dataclass."""
        event_data = ExchangeEventData(
            event_type=ExchangeEvent.ORDER_EXECUTED,
            exchange_name="test_exchange",
            timestamp="2024-01-01T12:00:00Z",
            data={"order_id": "123", "symbol": "BTCUSDT"},
            metadata={"source": "api"}
        )
        
        assert event_data.event_type == ExchangeEvent.ORDER_EXECUTED
        assert event_data.exchange_name == "test_exchange"
        assert event_data.timestamp == "2024-01-01T12:00:00Z"
        assert event_data.data == {"order_id": "123", "symbol": "BTCUSDT"}
        assert event_data.metadata == {"source": "api"}

    def test_iexchange_abstract_methods(self):
        """Test that IExchange has required abstract methods."""
        # This test ensures the interface has the expected methods
        # We can't instantiate it directly, but we can check the methods exist
        methods = [method for method in dir(IExchange) if not method.startswith('_')]
        
        expected_methods = [
            'initialize', 'close', 'get_status', 'get_account_info',
            'get_balance', 'create_order', 'cancel_order', 'get_order_status',
            'get_open_orders', 'get_ticker', 'get_klines', '__aenter__', '__aexit__'
        ]
        
        for method in expected_methods:
            assert method in methods

    def test_iexchange_adapter_abstract_methods(self):
        """Test that IExchangeAdapter has required abstract methods."""
        methods = [method for method in dir(IExchangeAdapter) if not method.startswith('_')]
        
        expected_methods = [
            'connect', 'disconnect', 'is_connected', 'test_connection',
            'execute_order', 'query_order', 'cancel_order', 'get_account_info',
            'get_market_data', '__aenter__', '__aexit__'
        ]
        
        for method in expected_methods:
            assert method in methods

    def test_imessage_router_abstract_methods(self):
        """Test that IMessageRouter has required abstract methods."""
        methods = [method for method in dir(IMessageRouter) if not method.startswith('_')]
        
        expected_methods = [
            'route_order', 'route_data_request', 'route_response', 'broadcast_message'
        ]
        
        for method in expected_methods:
            assert method in methods

    def test_iresponse_handler_abstract_methods(self):
        """Test that IResponseHandler has required abstract methods."""
        methods = [method for method in dir(IResponseHandler) if not method.startswith('_')]
        
        expected_methods = [
            'handle_order_response', 'handle_data_response', 'handle_error_response',
            'handle_status_update'
        ]
        
        for method in expected_methods:
            assert method in methods

    def test_ievent_publisher_abstract_methods(self):
        """Test that IEventPublisher has required abstract methods."""
        methods = [method for method in dir(IEventPublisher) if not method.startswith('_')]
        
        expected_methods = [
            'publish_event', 'subscribe_to_events', 'unsubscribe_from_events'
        ]
        
        for method in expected_methods:
            assert method in methods

    def test_iexchange_manager_abstract_methods(self):
        """Test that IExchangeManager has required abstract methods."""
        methods = [method for method in dir(IExchangeManager) if not method.startswith('_')]
        
        expected_methods = [
            'add_exchange', 'remove_exchange', 'get_exchange', 'get_all_exchanges',
            'get_exchange_metrics', 'get_all_metrics'
        ]
        
        for method in expected_methods:
            assert method in methods

    @pytest.mark.asyncio
    async def test_iexchange_context_manager(self):
        """Test that IExchange can be used as async context manager."""
        # Create a mock implementation
        class MockExchange(IExchange):
            async def initialize(self):
                pass
            
            async def close(self):
                pass
            
            async def get_status(self):
                return ExchangeStatus.CONNECTED
            
            async def get_account_info(self):
                return {}
            
            async def get_balance(self, currency: str):
                return {}
            
            async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, quantity: float, price=None, **kwargs):
                return {}
            
            async def cancel_order(self, order_id: str):
                return {}
            
            async def get_order_status(self, order_id: str):
                return {}
            
            async def get_open_orders(self, symbol=None):
                return []
            
            async def get_ticker(self, symbol: str):
                return {}
            
            async def get_klines(self, symbol: str, interval: str, limit: int = 100):
                return []
        
        exchange = MockExchange()
        
        # Test async context manager
        async with exchange as e:
            assert e == exchange
            status = await e.get_status()
            assert status == ExchangeStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_iexchange_adapter_context_manager(self):
        """Test that IExchangeAdapter can be used as async context manager."""
        # Create a mock implementation
        class MockAdapter(IExchangeAdapter):
            async def connect(self):
                return True
            
            async def disconnect(self):
                pass
            
            async def is_connected(self):
                return True
            
            async def test_connection(self):
                return {}
            
            async def execute_order(self, order_request: OrderRequest):
                return OrderResponse(order_id="test")
            
            async def query_order(self, order_id: str):
                return OrderResponse(order_id=order_id)
            
            async def cancel_order(self, order_id: str):
                return OrderResponse(order_id=order_id)
            
            async def get_account_info(self):
                return {}
            
            async def get_market_data(self, symbol: str, data_type: str, **kwargs):
                return {}
        
        adapter = MockAdapter()
        
        # Test async context manager
        async with adapter as a:
            assert a == adapter
            assert await a.is_connected() is True

    @pytest.mark.asyncio
    async def test_iexchange_adapter_connection_failure(self):
        """Test that IExchangeAdapter raises error on connection failure."""
        # Create a mock implementation that fails to connect
        class MockAdapter(IExchangeAdapter):
            async def connect(self):
                return False
            
            async def disconnect(self):
                pass
            
            async def is_connected(self):
                return False
            
            async def test_connection(self):
                return {}
            
            async def execute_order(self, order_request: OrderRequest):
                return OrderResponse(order_id="test")
            
            async def query_order(self, order_id: str):
                return OrderResponse(order_id=order_id)
            
            async def cancel_order(self, order_id: str):
                return OrderResponse(order_id=order_id)
            
            async def get_account_info(self):
                return {}
            
            async def get_market_data(self, symbol: str, data_type: str, **kwargs):
                return {}
        
        adapter = MockAdapter()
        
        # Test that connection failure raises error
        with pytest.raises(ConnectionError):
            async with adapter:
                pass