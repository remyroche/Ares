"""
Unit tests for OrderManager.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from exchanges.shared.orders.order_manager import (
    OrderManager, Order, OrderSide, OrderType, OrderStatus
)


class TestOrderManager:
    """Test cases for OrderManager."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager instance for testing."""
        return OrderManager("test_exchange")

    @pytest.fixture
    def mock_execution_functions(self):
        """Create mock execution functions."""
        return {
            "create_order": AsyncMock(return_value={"order_id": "ex_123", "status": "open"}),
            "cancel_order": AsyncMock(return_value=True),
            "get_order_status": AsyncMock(return_value={
                "order_id": "ex_123",
                "status": "filled",
                "filled_quantity": 1.0,
                "average_price": 50000.0
            }),
            "get_open_orders": AsyncMock(return_value=[
                {
                    "order_id": "ex_123",
                    "client_order_id": "client_123",
                    "status": "open",
                    "filled_quantity": 0.0
                }
            ])
        }

    def test_initialization(self, order_manager):
        """Test OrderManager initialization."""
        assert order_manager.exchange_name == "test_exchange"
        assert len(order_manager.orders) == 0
        assert len(order_manager.orders_by_symbol) == 0
        assert len(order_manager.orders_by_status) == 0
        assert order_manager.max_orders_per_symbol == 1000
        assert order_manager.max_total_orders == 10000

    def test_register_execution_functions(self, order_manager, mock_execution_functions):
        """Test registering execution functions."""
        order_manager.register_execution_functions(**mock_execution_functions)
        
        assert "create_order" in order_manager.execution_functions
        assert "cancel_order" in order_manager.execution_functions
        assert "get_order_status" in order_manager.execution_functions
        assert "get_open_orders" in order_manager.execution_functions

    def test_create_order(self, order_manager):
        """Test creating an order."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            client_order_id="test_order_123"
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 0.001
        assert order.price == 50000.0
        assert order.client_order_id == "test_order_123"
        assert order.status == OrderStatus.PENDING
        assert order.order_id in order_manager.orders

    def test_create_order_without_client_id(self, order_manager):
        """Test creating an order without client order ID."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )
        
        assert order.client_order_id is not None
        assert order.client_order_id.startswith("test_exchange_")

    def test_create_order_with_metadata(self, order_manager):
        """Test creating an order with metadata."""
        metadata = {"strategy": "test", "risk_level": "low"}
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            metadata=metadata
        )
        
        assert order.metadata == metadata

    def test_store_order(self, order_manager):
        """Test storing an order."""
        order = Order(
            order_id="test_123",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        order_manager._store_order(order)
        
        assert "test_123" in order_manager.orders
        assert "BTCUSDT" in order_manager.orders_by_symbol
        assert "test_123" in order_manager.orders_by_symbol["BTCUSDT"]
        assert OrderStatus.PENDING in order_manager.orders_by_status
        assert "test_123" in order_manager.orders_by_status[OrderStatus.PENDING]

    def test_enforce_order_limits_symbol(self, order_manager):
        """Test enforcing order limits per symbol."""
        # Set low limit for testing
        order_manager.max_orders_per_symbol = 2
        
        # Create 3 orders for same symbol
        for i in range(3):
            order = order_manager.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.0 + i
            )
        
        # Should only have 2 orders (oldest removed)
        assert len(order_manager.orders_by_symbol["BTCUSDT"]) == 2

    def test_enforce_order_limits_total(self, order_manager):
        """Test enforcing total order limits."""
        # Set low limit for testing
        order_manager.max_total_orders = 2
        
        # Create 3 orders
        for i in range(3):
            order = order_manager.create_order(
                symbol=f"SYMBOL{i}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.0
            )
        
        # Should only have 2 orders total
        assert len(order_manager.orders) == 2

    @pytest.mark.asyncio
    async def test_submit_order_success(self, order_manager, mock_execution_functions):
        """Test successful order submission."""
        order_manager.register_execution_functions(**mock_execution_functions)
        
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        result = await order_manager.submit_order(order)
        
        assert result is True
        assert order.status == OrderStatus.OPEN
        assert order.exchange_order_id == "ex_123"
        assert order.exchange_response is not None

    @pytest.mark.asyncio
    async def test_submit_order_no_function(self, order_manager):
        """Test order submission without registered function."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        result = await order_manager.submit_order(order)
        
        assert result is False
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_submit_order_failure(self, order_manager):
        """Test order submission failure."""
        mock_functions = {
            "create_order": AsyncMock(return_value=None),
            "cancel_order": AsyncMock(),
            "get_order_status": AsyncMock(),
            "get_open_orders": AsyncMock()
        }
        order_manager.register_execution_functions(**mock_functions)
        
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        result = await order_manager.submit_order(order)
        
        assert result is False
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_submit_order_exception(self, order_manager):
        """Test order submission with exception."""
        mock_functions = {
            "create_order": AsyncMock(side_effect=Exception("Network error")),
            "cancel_order": AsyncMock(),
            "get_order_status": AsyncMock(),
            "get_open_orders": AsyncMock()
        }
        order_manager.register_execution_functions(**mock_functions)
        
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        result = await order_manager.submit_order(order)
        
        assert result is False
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_manager, mock_execution_functions):
        """Test successful order cancellation."""
        order_manager.register_execution_functions(**mock_execution_functions)
        
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order.exchange_order_id = "ex_123"
        order.status = OrderStatus.OPEN
        
        result = await order_manager.cancel_order(order.order_id)
        
        assert result is True
        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, order_manager):
        """Test cancelling non-existent order."""
        result = await order_manager.cancel_order("nonexistent")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_wrong_status(self, order_manager):
        """Test cancelling order with wrong status."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order.status = OrderStatus.FILLED
        
        result = await order_manager.cancel_order(order.order_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_no_function(self, order_manager):
        """Test cancelling order without registered function."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order.exchange_order_id = "ex_123"
        order.status = OrderStatus.OPEN
        
        result = await order_manager.cancel_order(order.order_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_update_order_status_success(self, order_manager, mock_execution_functions):
        """Test successful order status update."""
        order_manager.register_execution_functions(**mock_execution_functions)
        
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order.exchange_order_id = "ex_123"
        order.status = OrderStatus.OPEN
        
        result = await order_manager.update_order_status(order.order_id)
        
        assert result is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order.average_price == 50000.0
        assert order.filled_at is not None

    @pytest.mark.asyncio
    async def test_update_order_status_not_found(self, order_manager):
        """Test updating status of non-existent order."""
        result = await order_manager.update_order_status("nonexistent")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_update_order_status_no_exchange_id(self, order_manager):
        """Test updating status of order without exchange ID."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        result = await order_manager.update_order_status(order.order_id)
        
        assert result is False

    def test_map_exchange_status(self, order_manager):
        """Test mapping exchange status to OrderStatus."""
        assert order_manager._map_exchange_status("pending") == OrderStatus.PENDING
        assert order_manager._map_exchange_status("open") == OrderStatus.OPEN
        assert order_manager._map_exchange_status("filled") == OrderStatus.FILLED
        assert order_manager._map_exchange_status("partially_filled") == OrderStatus.PARTIALLY_FILLED
        assert order_manager._map_exchange_status("cancelled") == OrderStatus.CANCELLED
        assert order_manager._map_exchange_status("rejected") == OrderStatus.REJECTED
        assert order_manager._map_exchange_status("expired") == OrderStatus.EXPIRED
        assert order_manager._map_exchange_status("unknown") == OrderStatus.PENDING

    def test_update_order_indexes(self, order_manager):
        """Test updating order indexes after status change."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        # Change status
        order.status = OrderStatus.FILLED
        order_manager._update_order_indexes(order)
        
        # Check that order is in new status index
        assert order.order_id in order_manager.orders_by_status[OrderStatus.FILLED]
        # Check that order is not in old status index
        assert order.order_id not in order_manager.orders_by_status[OrderStatus.PENDING]

    def test_get_order(self, order_manager):
        """Test getting order by ID."""
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        
        retrieved = order_manager.get_order(order.order_id)
        
        assert retrieved == order

    def test_get_order_not_found(self, order_manager):
        """Test getting non-existent order."""
        result = order_manager.get_order("nonexistent")
        
        assert result is None

    def test_get_orders_by_symbol(self, order_manager):
        """Test getting orders by symbol."""
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=51000.0
        )
        order3 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        btc_orders = order_manager.get_orders_by_symbol("BTCUSDT")
        eth_orders = order_manager.get_orders_by_symbol("ETHUSDT")
        
        assert len(btc_orders) == 2
        assert len(eth_orders) == 1
        assert order1 in btc_orders
        assert order2 in btc_orders
        assert order3 in eth_orders

    def test_get_orders_by_status(self, order_manager):
        """Test getting orders by status."""
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        # Change one order status
        order2.status = OrderStatus.FILLED
        order_manager._update_order_indexes(order2)
        
        pending_orders = order_manager.get_orders_by_status(OrderStatus.PENDING)
        filled_orders = order_manager.get_orders_by_status(OrderStatus.FILLED)
        
        assert len(pending_orders) == 1
        assert len(filled_orders) == 1
        assert order1 in pending_orders
        assert order2 in filled_orders

    def test_get_open_orders(self, order_manager):
        """Test getting open orders."""
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        # Change one order status
        order2.status = OrderStatus.FILLED
        order_manager._update_order_indexes(order2)
        
        open_orders = order_manager.get_open_orders()
        
        assert len(open_orders) == 1
        assert order1 in open_orders

    def test_get_filled_orders(self, order_manager):
        """Test getting filled orders."""
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        # Change one order status
        order2.status = OrderStatus.FILLED
        order_manager._update_order_indexes(order2)
        
        filled_orders = order_manager.get_filled_orders()
        
        assert len(filled_orders) == 1
        assert order2 in filled_orders

    def test_get_cancelled_orders(self, order_manager):
        """Test getting cancelled orders."""
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        # Change one order status
        order2.status = OrderStatus.CANCELLED
        order_manager._update_order_indexes(order2)
        
        cancelled_orders = order_manager.get_cancelled_orders()
        
        assert len(cancelled_orders) == 1
        assert order2 in cancelled_orders

    def test_get_rejected_orders(self, order_manager):
        """Test getting rejected orders."""
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        # Change one order status
        order2.status = OrderStatus.REJECTED
        order_manager._update_order_indexes(order2)
        
        rejected_orders = order_manager.get_rejected_orders()
        
        assert len(rejected_orders) == 1
        assert order2 in rejected_orders

    @pytest.mark.asyncio
    async def test_sync_orders_from_exchange(self, order_manager, mock_execution_functions):
        """Test syncing orders from exchange."""
        order_manager.register_execution_functions(**mock_execution_functions)
        
        # Create local order
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order.exchange_order_id = "ex_123"
        order.client_order_id = "client_123"
        
        synced_count = await order_manager.sync_orders_from_exchange()
        
        assert synced_count == 1
        assert order.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_sync_orders_from_exchange_no_function(self, order_manager):
        """Test syncing orders without registered function."""
        synced_count = await order_manager.sync_orders_from_exchange()
        
        assert synced_count == 0

    def test_get_order_statistics(self, order_manager):
        """Test getting order statistics."""
        # Create orders with different statuses
        order1 = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order2 = order_manager.create_order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3000.0
        )
        
        # Change statuses
        order2.status = OrderStatus.FILLED
        order_manager._update_order_indexes(order2)
        
        stats = order_manager.get_order_statistics()
        
        assert stats["total_orders"] == 2
        assert "status_distribution" in stats
        assert "symbol_distribution" in stats
        assert stats["open_orders"] == 1
        assert stats["filled_orders"] == 1

    def test_cleanup_old_orders(self, order_manager):
        """Test cleaning up old orders."""
        # Create old order
        order = order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )
        order.status = OrderStatus.FILLED
        order.created_at = datetime.now() - timedelta(days=31)
        order_manager._update_order_indexes(order)
        
        cleaned = order_manager.cleanup_old_orders(max_age_days=30)
        
        assert cleaned == 1
        assert order.order_id not in order_manager.orders