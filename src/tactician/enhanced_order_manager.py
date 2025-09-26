"""Enhanced order manager utilities."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
import logging

from ...utils.logger import system_logger
from ...core.decorators import handles_errors

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class OrderRequest:
    """Order request data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OrderResponse:
    """Order response data structure."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float = 0.0
    price: Optional[float] = None
    average_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class EnhancedOrderManager:
    """Enhanced order manager with comprehensive order handling capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced order manager."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedOrderManager')
        self.order_config = config.get('order_manager', {})
        
        # Order management state
        self.active_orders: Dict[str, OrderResponse] = {}
        self.order_history: List[OrderResponse] = []
        self.pending_orders: List[OrderRequest] = []
        
        # Configuration
        self.max_retries = self.order_config.get('max_retries', 3)
        self.retry_delay = self.order_config.get('retry_delay', 1.0)
        self.order_timeout = self.order_config.get('order_timeout', 30.0)
        
        # Exchange client (injected via config)
        self.exchange_client = self.order_config.get('exchange_client')
        
        self.logger.info('Enhanced Order Manager initialized')

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='order manager initialization')
    async def initialize(self) -> bool:
        """Initialize the order manager."""
        try:
            self.logger.info('Initializing Enhanced Order Manager...')
            
            # Validate exchange client
            if not self.exchange_client:
                self.logger.warning('No exchange client configured - orders will be simulated')
            
            # Initialize order tracking
            await self._initialize_order_tracking()
            
            self.logger.info('✅ Enhanced Order Manager initialized successfully')
            return True
            
        except Exception as e:
            self.logger.error(f'❌ Order Manager initialization failed: {e}')
            return False

    async def _initialize_order_tracking(self):
        """Initialize order tracking systems."""
        try:
            # Load existing orders if available
            await self._load_existing_orders()
            
            # Start order monitoring task
            asyncio.create_task(self._monitor_orders())
            
        except Exception as e:
            self.logger.error(f'Error initializing order tracking: {e}')

    async def _load_existing_orders(self):
        """Load existing orders from persistent storage."""
        try:
            # This would load from database or file storage
            # For now, just initialize empty state
            pass
        except Exception as e:
            self.logger.error(f'Error loading existing orders: {e}')

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='order placement')
    async def place_order(self, order_request: OrderRequest) -> Optional[OrderResponse]:
        """Place an order with the exchange."""
        try:
            self.logger.info(f'Placing order: {order_request.side.value} {order_request.quantity} {order_request.symbol}')
            
            # Validate order request
            if not self._validate_order_request(order_request):
                return None
            
            # Generate order ID
            order_id = self._generate_order_id(order_request)
            
            # Create order response
            order_response = OrderResponse(
                order_id=order_id,
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                status=OrderStatus.PENDING
            )
            
            # Place order with exchange
            if self.exchange_client:
                success = await self._place_order_with_exchange(order_request, order_response)
                if not success:
                    order_response.status = OrderStatus.REJECTED
                    order_response.error_message = "Failed to place order with exchange"
            else:
                # Simulate order placement
                order_response.status = OrderStatus.FILLED
                order_response.filled_quantity = order_request.quantity
                order_response.average_price = order_request.price or 0.0
            
            # Track order
            self.active_orders[order_id] = order_response
            self.order_history.append(order_response)
            
            self.logger.info(f'✅ Order placed successfully: {order_id}')
            return order_response
            
        except Exception as e:
            self.logger.error(f'❌ Order placement failed: {e}')
            return None

    def _validate_order_request(self, order_request: OrderRequest) -> bool:
        """Validate order request parameters."""
        try:
            if not order_request.symbol:
                self.logger.error('Order symbol is required')
                return False
            
            if order_request.quantity <= 0:
                self.logger.error('Order quantity must be positive')
                return False
            
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not order_request.price:
                self.logger.error('Price is required for limit orders')
                return False
            
            if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not order_request.stop_price:
                self.logger.error('Stop price is required for stop orders')
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f'Order validation failed: {e}')
            return False

    def _generate_order_id(self, order_request: OrderRequest) -> str:
        """Generate unique order ID."""
        import uuid
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"{order_request.symbol}_{timestamp}_{unique_id}"

    async def _place_order_with_exchange(self, order_request: OrderRequest, order_response: OrderResponse) -> bool:
        """Place order with exchange client."""
        try:
            if not self.exchange_client:
                return False
            
            # Convert to exchange format
            exchange_order = self._convert_to_exchange_format(order_request)
            
            # Place order
            result = await self.exchange_client.place_order(exchange_order)
            
            if result and result.get('success'):
                order_response.order_id = result.get('order_id', order_response.order_id)
                order_response.status = OrderStatus.PENDING
                return True
            else:
                order_response.error_message = result.get('error', 'Unknown exchange error')
                return False
                
        except Exception as e:
            self.logger.error(f'Exchange order placement failed: {e}')
            return False

    def _convert_to_exchange_format(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Convert order request to exchange-specific format."""
        return {
            'symbol': order_request.symbol,
            'side': order_request.side.value,
            'type': order_request.order_type.value,
            'quantity': order_request.quantity,
            'price': order_request.price,
            'stopPrice': order_request.stop_price,
            'timeInForce': order_request.time_in_force,
            'clientOrderId': order_request.client_order_id
        }

    async def _monitor_orders(self):
        """Monitor active orders for status updates."""
        try:
            while True:
                await self._update_order_statuses()
                await asyncio.sleep(5.0)  # Check every 5 seconds
        except asyncio.CancelledError:
            self.logger.info('Order monitoring cancelled')
        except Exception as e:
            self.logger.error(f'Error in order monitoring: {e}')

    async def _update_order_statuses(self):
        """Update status of active orders."""
        try:
            if not self.exchange_client:
                return
            
            for order_id, order in list(self.active_orders.items()):
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                    # Check order status with exchange
                    status = await self.exchange_client.get_order_status(order_id)
                    if status:
                        order.status = OrderStatus(status.get('status', order.status.value))
                        order.filled_quantity = status.get('filled_quantity', order.filled_quantity)
                        order.average_price = status.get('average_price', order.average_price)
                        
                        # Remove completed orders from active tracking
                        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                            del self.active_orders[order_id]
                            
        except Exception as e:
            self.logger.error(f'Error updating order statuses: {e}')

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='order cancellation')
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f'Order not found: {order_id}')
                return False
            
            order = self.active_orders[order_id]
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                self.logger.warning(f'Cannot cancel order in status: {order.status.value}')
                return False
            
            # Cancel with exchange
            if self.exchange_client:
                success = await self.exchange_client.cancel_order(order_id)
                if success:
                    order.status = OrderStatus.CANCELLED
                    del self.active_orders[order_id]
                    self.logger.info(f'✅ Order cancelled: {order_id}')
                    return True
                else:
                    self.logger.error(f'Failed to cancel order: {order_id}')
                    return False
            else:
                # Simulate cancellation
                order.status = OrderStatus.CANCELLED
                del self.active_orders[order_id]
                self.logger.info(f'✅ Order cancelled (simulated): {order_id}')
                return True
                
        except Exception as e:
            self.logger.error(f'❌ Order cancellation failed: {e}')
            return False

    def get_active_orders(self) -> Dict[str, OrderResponse]:
        """Get all active orders."""
        return self.active_orders.copy()

    def get_order_history(self, limit: Optional[int] = None) -> List[OrderResponse]:
        """Get order history."""
        if limit:
            return self.order_history[-limit:]
        return self.order_history.copy()

    def get_order_by_id(self, order_id: str) -> Optional[OrderResponse]:
        """Get order by ID."""
        return self.active_orders.get(order_id) or next(
            (order for order in self.order_history if order.order_id == order_id), None
        )

    async def cleanup(self):
        """Cleanup order manager resources."""
        try:
            self.logger.info('Cleaning up Enhanced Order Manager...')
            
            # Cancel all pending orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)
            
            self.logger.info('✅ Enhanced Order Manager cleanup completed')
            
        except Exception as e:
            self.logger.error(f'❌ Order Manager cleanup failed: {e}')