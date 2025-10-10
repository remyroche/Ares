"""
Order Management

Handles order creation, modification, and tracking across exchanges.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation."""
    id: str
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
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity


class OrderManager:
    """Manages orders across exchanges."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"OrderManager.{exchange_name}")
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = {}
        self.orders_by_status: Dict[OrderStatus, List[str]] = {}
        
        # Exchange-specific functions
        self.exchange_functions: Dict[str, Callable] = {}
        
        # Order tracking
        self.pending_orders: List[str] = []
        self.filled_orders: List[str] = []
        self.canceled_orders: List[str] = []
        
        # Statistics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        
        # Initialize status tracking
        for status in OrderStatus:
            self.orders_by_status[status] = []
    
    def register_execution_functions(
        self,
        create_order: Optional[Callable] = None,
        cancel_order: Optional[Callable] = None,
        get_order_status: Optional[Callable] = None,
        get_open_orders: Optional[Callable] = None
    ) -> None:
        """Register exchange-specific execution functions."""
        if create_order:
            self.exchange_functions["create_order"] = create_order
        if cancel_order:
            self.exchange_functions["cancel_order"] = cancel_order
        if get_order_status:
            self.exchange_functions["get_order_status"] = get_order_status
        if get_open_orders:
            self.exchange_functions["get_open_orders"] = get_open_orders
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[str]:
        """Create a new order."""
        try:
            # Generate order ID
            order_id = client_order_id or str(uuid.uuid4())
            
            # Create order object
            order = Order(
                id=order_id,
                symbol=symbol.upper(),
                side=OrderSide(side.lower()),
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                status=OrderStatus.PENDING
            )
            
            # Store order
            self._store_order(order)
            
            # Execute on exchange if function is registered
            if "create_order" in self.exchange_functions:
                try:
                    result = await self.exchange_functions["create_order"](
                        symbol, side, order_type, quantity, price, stop_price, client_order_id
                    )
                    
                    if result:
                        # Update order with exchange response
                        order.exchange_order_id = result.get("orderId") or result.get("id")
                        order.status = OrderStatus.NEW
                        order.updated_at = datetime.now()
                        
                        # Update tracking
                        self._update_order_tracking(order)
                        
                        self.logger.info(f"Order {order_id} created successfully")
                        return order_id
                    else:
                        order.status = OrderStatus.REJECTED
                        self._update_order_tracking(order)
                        self.logger.error(f"Order {order_id} creation failed")
                        return None
                        
                except Exception as e:
                    order.status = OrderStatus.REJECTED
                    order.metadata["error"] = str(e)
                    self._update_order_tracking(order)
                    self.logger.error(f"Order {order_id} creation failed: {e}")
                    return None
            else:
                # No exchange function, just store as pending
                self.logger.warning("No create_order function registered, storing as pending")
                return order_id
                
        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            order = self.orders.get(order_id)
            if not order:
                self.logger.error(f"Order {order_id} not found")
                return False
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                self.logger.warning(f"Order {order_id} cannot be canceled, status: {order.status}")
                return False
            
            # Execute on exchange if function is registered
            if "cancel_order" in self.exchange_functions:
                try:
                    success = await self.exchange_functions["cancel_order"](order_id)
                    
                    if success:
                        order.status = OrderStatus.CANCELED
                        order.updated_at = datetime.now()
                        self._update_order_tracking(order)
                        self.logger.info(f"Order {order_id} canceled successfully")
                        return True
                    else:
                        self.logger.error(f"Failed to cancel order {order_id}")
                        return False
                        
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order_id}: {e}")
                    return False
            else:
                # No exchange function, just mark as canceled
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now()
                self._update_order_tracking(order)
                self.logger.warning("No cancel_order function registered, marking as canceled")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status from exchange."""
        try:
            order = self.orders.get(order_id)
            if not order:
                return None
            
            # Get from exchange if function is registered
            if "get_order_status" in self.exchange_functions:
                try:
                    result = await self.exchange_functions["get_order_status"](order_id)
                    
                    if result:
                        # Update order with exchange data
                        self._update_order_from_exchange(order, result)
                        return self._order_to_dict(order)
                    else:
                        return self._order_to_dict(order)
                        
                except Exception as e:
                    self.logger.error(f"Failed to get order status for {order_id}: {e}")
                    return self._order_to_dict(order)
            else:
                return self._order_to_dict(order)
                
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders from exchange."""
        try:
            # Get from exchange if function is registered
            if "get_open_orders" in self.exchange_functions:
                try:
                    result = await self.exchange_functions["get_open_orders"](symbol)
                    
                    if result:
                        # Update local orders with exchange data
                        for order_data in result:
                            self._update_local_orders_from_exchange(order_data)
                        
                        # Return open orders
                        return self._get_local_open_orders(symbol)
                    else:
                        return self._get_local_open_orders(symbol)
                        
                except Exception as e:
                    self.logger.error(f"Failed to get open orders: {e}")
                    return self._get_local_open_orders(symbol)
            else:
                return self._get_local_open_orders(symbol)
                
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def sync_orders_from_exchange(self) -> None:
        """Sync orders from exchange."""
        try:
            if "get_open_orders" in self.exchange_functions:
                result = await self.exchange_functions["get_open_orders"]()
                
                if result:
                    for order_data in result:
                        self._update_local_orders_from_exchange(order_data)
                        
        except Exception as e:
            self.logger.error(f"Failed to sync orders from exchange: {e}")
    
    def _store_order(self, order: Order) -> None:
        """Store order in internal data structures."""
        self.orders[order.id] = order
        
        # Add to symbol tracking
        if order.symbol not in self.orders_by_symbol:
            self.orders_by_symbol[order.symbol] = []
        self.orders_by_symbol[order.symbol].append(order.id)
        
        # Add to status tracking
        self.orders_by_status[order.status].append(order.id)
        
        # Update statistics
        self.total_orders += 1
    
    def _update_order_tracking(self, order: Order) -> None:
        """Update order tracking lists."""
        # Remove from old status
        if order.id in self.orders_by_status.get(order.status, []):
            self.orders_by_status[order.status].remove(order.id)
        
        # Add to new status
        self.orders_by_status[order.status].append(order.id)
        
        # Update specific tracking lists
        if order.status == OrderStatus.FILLED:
            if order.id not in self.filled_orders:
                self.filled_orders.append(order.id)
            if order.id in self.pending_orders:
                self.pending_orders.remove(order.id)
        elif order.status == OrderStatus.CANCELED:
            if order.id not in self.canceled_orders:
                self.canceled_orders.append(order.id)
            if order.id in self.pending_orders:
                self.pending_orders.remove(order.id)
        elif order.status in [OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            if order.id not in self.pending_orders:
                self.pending_orders.append(order.id)
    
    def _update_order_from_exchange(self, order: Order, exchange_data: Dict[str, Any]) -> None:
        """Update order with data from exchange."""
        # Map exchange status to our status
        status_mapping = {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        
        if "status" in exchange_data:
            exchange_status = exchange_data["status"].upper()
            if exchange_status in status_mapping:
                order.status = status_mapping[exchange_status]
        
        # Update quantities
        if "executedQty" in exchange_data:
            order.filled_quantity = float(exchange_data["executedQty"])
            order.remaining_quantity = order.quantity - order.filled_quantity
        
        if "avgPrice" in exchange_data:
            order.average_price = float(exchange_data["avgPrice"])
        
        # Update timestamps
        order.updated_at = datetime.now()
        
        # Update tracking
        self._update_order_tracking(order)
    
    def _update_local_orders_from_exchange(self, exchange_data: Dict[str, Any]) -> None:
        """Update local orders with exchange data."""
        exchange_order_id = exchange_data.get("orderId") or exchange_data.get("id")
        if not exchange_order_id:
            return
        
        # Find order by exchange ID
        for order in self.orders.values():
            if order.exchange_order_id == str(exchange_order_id):
                self._update_order_from_exchange(order, exchange_data)
                break
    
    def _get_local_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders from local storage."""
        open_statuses = [OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        open_orders = []
        
        for status in open_statuses:
            for order_id in self.orders_by_status[status]:
                order = self.orders.get(order_id)
                if order and (not symbol or order.symbol == symbol.upper()):
                    open_orders.append(self._order_to_dict(order))
        
        return open_orders
    
    def _order_to_dict(self, order: Order) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "stop_price": order.stop_price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "average_price": order.average_price,
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "client_order_id": order.client_order_id,
            "exchange_order_id": order.exchange_order_id,
            "metadata": order.metadata
        }
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders for a symbol."""
        order_ids = self.orders_by_symbol.get(symbol.upper(), [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status."""
        order_ids = self.orders_by_status.get(status, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "pending_orders": len(self.pending_orders),
            "filled_orders": len(self.filled_orders),
            "canceled_orders": len(self.canceled_orders),
            "orders_by_status": {
                status.value: len(order_ids) 
                for status, order_ids in self.orders_by_status.items()
            }
        }