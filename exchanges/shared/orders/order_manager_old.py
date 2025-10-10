"""
Order Management Utilities

Handles order creation, modification, cancellation, and status tracking.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    exchange_order_id: Optional[str] = None
    exchange_response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
        if self.metadata is None:
            self.metadata = {}


class OrderManager:
    """
    Manages order lifecycle, tracking, and execution.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"OrderManager.{exchange_name}")
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = {}
        self.orders_by_status: Dict[OrderStatus, List[str]] = {}
        
        # Order execution functions
        self.execution_functions: Dict[str, callable] = {}
        
        # Order limits
        self.max_orders_per_symbol = 1000
        self.max_total_orders = 10000
        
    def register_execution_functions(
        self,
        create_order: callable,
        cancel_order: callable,
        get_order_status: callable,
        get_open_orders: Optional[callable] = None
    ) -> None:
        """
        Register exchange-specific order execution functions.
        
        Args:
            create_order: Function to create order on exchange
            cancel_order: Function to cancel order on exchange
            get_order_status: Function to get order status from exchange
            get_open_orders: Optional function to get open orders
        """
        self.execution_functions = {
            "create_order": create_order,
            "cancel_order": cancel_order,
            "get_order_status": get_order_status,
            "get_open_orders": get_open_orders
        }
        
        self.logger.info("Registered order execution functions")
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Client-defined order ID
            metadata: Additional metadata
            
        Returns:
            Created Order object
        """
        # Generate order IDs
        order_id = str(uuid.uuid4())
        if client_order_id is None:
            client_order_id = f"{self.exchange_name}_{int(datetime.now().timestamp())}"
        
        # Create order
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        # Store order
        self._store_order(order)
        
        self.logger.info(f"Created order {order_id} for {symbol} {side.value} {quantity} @ {price}")
        return order
    
    def _store_order(self, order: Order) -> None:
        """Store order in internal data structures."""
        self.orders[order.order_id] = order
        
        # Index by symbol
        if order.symbol not in self.orders_by_symbol:
            self.orders_by_symbol[order.symbol] = []
        self.orders_by_symbol[order.symbol].append(order.order_id)
        
        # Index by status
        if order.status not in self.orders_by_status:
            self.orders_by_status[order.status] = []
        self.orders_by_status[order.status].append(order.order_id)
        
        # Enforce limits
        self._enforce_order_limits()
    
    def _enforce_order_limits(self) -> None:
        """Enforce order limits by removing oldest orders."""
        # Check symbol limits
        for symbol, order_ids in self.orders_by_symbol.items():
            if len(order_ids) > self.max_orders_per_symbol:
                # Remove oldest orders
                orders_to_remove = order_ids[:-self.max_orders_per_symbol]
                for order_id in orders_to_remove:
                    self._remove_order(order_id)
        
        # Check total limits
        total_orders = len(self.orders)
        if total_orders > self.max_total_orders:
            # Remove oldest orders
            sorted_orders = sorted(
                self.orders.values(),
                key=lambda x: x.created_at
            )
            orders_to_remove = sorted_orders[:-self.max_total_orders]
            for order in orders_to_remove:
                self._remove_order(order.order_id)
    
    def _remove_order(self, order_id: str) -> None:
        """Remove order from all data structures."""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # Remove from symbol index
        if order.symbol in self.orders_by_symbol:
            self.orders_by_symbol[order.symbol].remove(order_id)
            if not self.orders_by_symbol[order.symbol]:
                del self.orders_by_symbol[order.symbol]
        
        # Remove from status index
        if order.status in self.orders_by_status:
            self.orders_by_status[order.status].remove(order_id)
            if not self.orders_by_status[order.status]:
                del self.orders_by_status[order.status]
        
        # Remove from main storage
        del self.orders[order_id]
    
    async def submit_order(self, order: Order) -> bool:
        """
        Submit order to exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            True if submission successful
        """
        try:
            if "create_order" not in self.execution_functions:
                self.logger.error("No create_order function registered")
                return False
            
            # Prepare order parameters
            order_params = {
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "stop_price": order.stop_price,
                "client_order_id": order.client_order_id
            }
            
            # Submit to exchange
            exchange_response = await self.execution_functions["create_order"](**order_params)
            
            if exchange_response:
                # Update order with exchange response
                order.exchange_order_id = exchange_response.get("order_id")
                order.exchange_response = exchange_response
                order.status = OrderStatus.OPEN
                order.updated_at = datetime.now()
                
                self.logger.info(f"Order {order.order_id} submitted successfully")
                return True
            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                self.logger.error(f"Order {order.order_id} submission failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            order = self.get_order(order_id)
            if not order:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                self.logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status.value})")
                return False
            
            if "cancel_order" not in self.execution_functions:
                self.logger.error("No cancel_order function registered")
                return False
            
            # Cancel on exchange
            if order.exchange_order_id:
                success = await self.execution_functions["cancel_order"](
                    order.exchange_order_id, order.symbol
                )
                if not success:
                    self.logger.error(f"Failed to cancel order {order_id} on exchange")
                    return False
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now()
            order.updated_at = datetime.now()
            
            # Update indexes
            self._update_order_indexes(order)
            
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def update_order_status(self, order_id: str) -> bool:
        """
        Update order status from exchange.
        
        Args:
            order_id: Order ID to update
            
        Returns:
            True if update successful
        """
        try:
            order = self.get_order(order_id)
            if not order:
                return False
            
            if not order.exchange_order_id:
                return False
            
            if "get_order_status" not in self.execution_functions:
                return False
            
            # Get status from exchange
            exchange_status = await self.execution_functions["get_order_status"](
                order.exchange_order_id, order.symbol
            )
            
            if not exchange_status:
                return False
            
            # Update order with exchange data
            old_status = order.status
            
            # Map exchange status to our status
            order.status = self._map_exchange_status(exchange_status.get("status", ""))
            order.filled_quantity = float(exchange_status.get("filled_quantity", 0))
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.average_price = exchange_status.get("average_price")
            order.updated_at = datetime.now()
            
            # Set filled_at if order was just filled
            if old_status != OrderStatus.FILLED and order.status == OrderStatus.FILLED:
                order.filled_at = datetime.now()
            
            # Update indexes if status changed
            if old_status != order.status:
                self._update_order_indexes(order)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating order status {order_id}: {e}")
            return False
    
    def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange status to our OrderStatus enum."""
        status_mapping = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED
        }
        
        return status_mapping.get(exchange_status.lower(), OrderStatus.PENDING)
    
    def _update_order_indexes(self, order: Order) -> None:
        """Update order indexes after status change."""
        # Remove from old status index
        for status, order_ids in self.orders_by_status.items():
            if order.order_id in order_ids:
                order_ids.remove(order.order_id)
                if not order_ids:
                    del self.orders_by_status[status]
                break
        
        # Add to new status index
        if order.status not in self.orders_by_status:
            self.orders_by_status[order.status] = []
        self.orders_by_status[order.status].append(order.order_id)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders for a symbol."""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status."""
        order_ids = self.orders_by_status.get(status, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return (
            self.get_orders_by_status(OrderStatus.PENDING) +
            self.get_orders_by_status(OrderStatus.OPEN) +
            self.get_orders_by_status(OrderStatus.PARTIALLY_FILLED)
        )
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        return self.get_orders_by_status(OrderStatus.FILLED)
    
    def get_cancelled_orders(self) -> List[Order]:
        """Get all cancelled orders."""
        return self.get_orders_by_status(OrderStatus.CANCELLED)
    
    def get_rejected_orders(self) -> List[Order]:
        """Get all rejected orders."""
        return self.get_orders_by_status(OrderStatus.REJECTED)
    
    async def sync_orders_from_exchange(self) -> int:
        """Sync all open orders from exchange."""
        if "get_open_orders" not in self.execution_functions:
            return 0
        
        try:
            # Get open orders from exchange
            exchange_orders = await self.execution_functions["get_open_orders"]()
            if not exchange_orders:
                return 0
            
            synced_count = 0
            for exchange_order in exchange_orders:
                # Find matching local order
                local_order = None
                for order in self.orders.values():
                    if (order.exchange_order_id == exchange_order.get("order_id") or
                        order.client_order_id == exchange_order.get("client_order_id")):
                        local_order = order
                        break
                
                if local_order:
                    # Update local order with exchange data
                    local_order.status = self._map_exchange_status(exchange_order.get("status", ""))
                    local_order.filled_quantity = float(exchange_order.get("filled_quantity", 0))
                    local_order.remaining_quantity = local_order.quantity - local_order.filled_quantity
                    local_order.average_price = exchange_order.get("average_price")
                    local_order.updated_at = datetime.now()
                    synced_count += 1
            
            self.logger.info(f"Synced {synced_count} orders from exchange")
            return synced_count
            
        except Exception as e:
            self.logger.error(f"Error syncing orders from exchange: {e}")
            return 0
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        total_orders = len(self.orders)
        
        status_counts = {}
        for status, order_ids in self.orders_by_status.items():
            status_counts[status.value] = len(order_ids)
        
        symbol_counts = {}
        for symbol, order_ids in self.orders_by_symbol.items():
            symbol_counts[symbol] = len(order_ids)
        
        return {
            "total_orders": total_orders,
            "status_distribution": status_counts,
            "symbol_distribution": symbol_counts,
            "open_orders": len(self.get_open_orders()),
            "filled_orders": len(self.get_filled_orders()),
            "cancelled_orders": len(self.get_cancelled_orders()),
            "rejected_orders": len(self.get_rejected_orders())
        }
    
    def cleanup_old_orders(self, max_age_days: int = 30) -> int:
        """Clean up old orders."""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            if order.created_at < cutoff_time and order.status in [
                OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED
            ]:
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            self._remove_order(order_id)
        
        if orders_to_remove:
            self.logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
        
        return len(orders_to_remove)