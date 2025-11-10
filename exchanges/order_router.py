"""
Order Router

Routes orders to the appropriate exchange and handles order management.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.utils.tprint import tprint
from .exchange_registry import ExchangeRegistry


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class RoutedOrder:
    """Routed order data structure"""
    id: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: OrderStatus
    exchange_order_id: Optional[str]
    timestamp: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    error_message: Optional[str] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderRouter:
    """Routes orders to appropriate exchanges and manages order lifecycle"""
    
    def __init__(self, exchange_registry: ExchangeRegistry):
        self.exchange_registry = exchange_registry
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.routed_orders: Dict[str, RoutedOrder] = {}
        self.active_orders: Dict[str, RoutedOrder] = {}
        
        # Order monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.order_stats = {
            "total_routed": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "by_exchange": {},
            "by_symbol": {},
            "by_status": {}
        }
        
    async def start(self) -> None:
        """Start order router monitoring"""
        tprint(f"Starting order router monitoring", "INFO")
        if self._running:
            tprint(f"Order router already running, skipping start", "WARNING")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_orders())
        self.logger.info("Order router started")
        tprint(f"Order router started successfully", "SUCCESS")
    
    async def stop(self) -> None:
        """Stop order router monitoring"""
        tprint(f"Stopping order router monitoring", "INFO")
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Order router stopped")
        tprint(f"Order router stopped successfully", "SUCCESS")
    
    async def route_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Route order to specified exchange"""
        tprint(f"Routing order: exchange={exchange}, symbol={symbol}, side={side}, type={order_type}, quantity={quantity}, price={price}", "INFO")
        try:
            # Generate unique order ID
            order_id = f"{exchange}_{symbol}_{side}_{int(time.time() * 1000)}"
            tprint(f"Generated order ID: {order_id}", "INFO")
            
            # Create routed order
            routed_order = RoutedOrder(
                id=order_id,
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                exchange_order_id=None,
                timestamp=datetime.now(),
                metadata=kwargs
            )
            
            # Store order
            self.routed_orders[order_id] = routed_order
            self.active_orders[order_id] = routed_order
            
            # Get exchange instance
            tprint(f"Fetching exchange instance for {exchange}", "INFO")
            exchange_instance = await self.exchange_registry.get_exchange(exchange)
            if not exchange_instance:
                tprint(f"Exchange {exchange} not found or not available", "ERROR")
                raise ValueError(f"Exchange {exchange} not found or not available")
            
            # Submit order to exchange
            tprint(f"Submitting order {order_id} to exchange {exchange}", "INFO")
            await self._submit_order_to_exchange(routed_order, exchange_instance)

            # Update statistics
            self._update_order_stats(routed_order)

            self.logger.info(f"Order routed: {order_id} -> {exchange}")
            tprint(f"Order routed successfully: {order_id} -> {exchange}, status={routed_order.status.value}", "SUCCESS")

            return {
                "success": True,
                "order_id": order_id,
                "exchange": exchange,
                "status": routed_order.status.value,
                "exchange_order_id": routed_order.exchange_order_id
            }
            
        except Exception as e:
            tprint(f"Error routing order: {e}", "ERROR")
            self.logger.error(f"Error routing order: {e}")

            # Mark order as failed if it was created
            if 'order_id' in locals():
                routed_order.status = OrderStatus.FAILED
                routed_order.error_message = str(e)
                self._update_order_stats(routed_order)
                tprint(f"Order {order_id} marked as failed", "ERROR")

            return {
                "success": False,
                "error": str(e),
                "order_id": order_id if 'order_id' in locals() else None
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel routed order"""
        tprint(f"Cancelling order: order_id={order_id}", "INFO")
        try:
            if order_id not in self.routed_orders:
                tprint(f"Order {order_id} not found", "ERROR")
                raise ValueError(f"Order {order_id} not found")
            
            routed_order = self.routed_orders[order_id]

            if routed_order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                tprint(f"Cannot cancel order {order_id} in status: {routed_order.status.value}", "WARNING")
                raise ValueError(f"Cannot cancel order in status: {routed_order.status.value}")
            
            # Get exchange instance
            exchange_instance = await self.exchange_registry.get_exchange(routed_order.exchange)
            if not exchange_instance:
                raise ValueError(f"Exchange {routed_order.exchange} not found")
            
            # Cancel order on exchange
            if routed_order.exchange_order_id:
                tprint(f"Cancelling order {order_id} on exchange {routed_order.exchange}", "INFO")
                await exchange_instance.cancel_order(routed_order.symbol, routed_order.exchange_order_id)

            # Update order status
            routed_order.status = OrderStatus.CANCELLED
            routed_order.cancelled_at = datetime.now()

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            self.logger.info(f"Order cancelled: {order_id}")
            tprint(f"Order cancelled successfully: {order_id}", "SUCCESS")

            return {
                "success": True,
                "order_id": order_id,
                "status": routed_order.status.value
            }
            
        except Exception as e:
            tprint(f"Error cancelling order {order_id}: {e}", "ERROR")
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id
            }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        tprint(f"Getting order status: order_id={order_id}", "INFO")
        try:
            if order_id not in self.routed_orders:
                tprint(f"Order {order_id} not found", "ERROR")
                raise ValueError(f"Order {order_id} not found")
            
            routed_order = self.routed_orders[order_id]

            # Get updated status from exchange if order is active
            if routed_order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                tprint(f"Updating order status from exchange for {order_id}", "INFO")
                await self._update_order_status_from_exchange(routed_order)

            tprint(f"Order status retrieved: {order_id}, status={routed_order.status.value}", "SUCCESS")
            return {
                "success": True,
                "order_id": order_id,
                "status": routed_order.status.value,
                "filled_quantity": routed_order.filled_quantity,
                "average_price": routed_order.average_price,
                "exchange_order_id": routed_order.exchange_order_id,
                "error_message": routed_order.error_message,
                "timestamp": routed_order.timestamp.isoformat(),
                "submitted_at": routed_order.submitted_at.isoformat() if routed_order.submitted_at else None,
                "filled_at": routed_order.filled_at.isoformat() if routed_order.filled_at else None,
                "cancelled_at": routed_order.cancelled_at.isoformat() if routed_order.cancelled_at else None
            }
            
        except Exception as e:
            tprint(f"Error getting order status {order_id}: {e}", "ERROR")
            self.logger.error(f"Error getting order status {order_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id
            }
    
    async def get_active_orders(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active orders, optionally filtered by exchange or symbol"""
        tprint(f"Getting active orders: exchange={exchange}, symbol={symbol}", "INFO")
        active_orders = []
        
        for order in self.active_orders.values():
            if exchange and order.exchange != exchange:
                continue
            if symbol and order.symbol != symbol:
                continue
            
            active_orders.append({
                "order_id": order.id,
                "exchange": order.exchange,
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value,
                "exchange_order_id": order.exchange_order_id,
                "timestamp": order.timestamp.isoformat(),
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price
            })

        tprint(f"Retrieved {len(active_orders)} active orders", "SUCCESS")
        return active_orders
    
    async def get_order_history(self, exchange: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history"""
        orders = []
        
        for order in self.routed_orders.values():
            if exchange and order.exchange != exchange:
                continue
            if symbol and order.symbol != symbol:
                continue
            
            orders.append({
                "order_id": order.id,
                "exchange": order.exchange,
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value,
                "exchange_order_id": order.exchange_order_id,
                "timestamp": order.timestamp.isoformat(),
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "cancelled_at": order.cancelled_at.isoformat() if order.cancelled_at else None,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "commission": order.commission,
                "error_message": order.error_message
            })
        
        # Sort by timestamp descending and limit
        orders.sort(key=lambda x: x["timestamp"], reverse=True)
        return orders[:limit]
    
    async def _submit_order_to_exchange(self, routed_order: RoutedOrder, exchange_instance: Any) -> None:
        """Submit order to exchange"""
        tprint(f"Submitting order to exchange: order_id={routed_order.id}, exchange={routed_order.exchange}, symbol={routed_order.symbol}", "INFO")
        try:
            # Submit order
            result = await exchange_instance.create_order(
                symbol=routed_order.symbol,
                side=routed_order.side,
                quantity=routed_order.quantity,
                price=routed_order.price,
                order_type=routed_order.order_type
            )
            
            # Update order with exchange response
            if result:
                routed_order.exchange_order_id = result.get("orderId", result.get("id"))
                routed_order.status = OrderStatus.SUBMITTED
                routed_order.submitted_at = datetime.now()

                self.logger.info(f"Order submitted to {routed_order.exchange}: {routed_order.exchange_order_id}")
                tprint(f"Order submitted successfully: order_id={routed_order.id}, exchange_order_id={routed_order.exchange_order_id}", "SUCCESS")
            else:
                routed_order.status = OrderStatus.FAILED
                routed_order.error_message = "No response from exchange"
                tprint(f"Order submission failed: No response from exchange for order_id={routed_order.id}", "ERROR")
                
        except Exception as e:
            routed_order.status = OrderStatus.FAILED
            routed_order.error_message = str(e)
            tprint(f"Failed to submit order to {routed_order.exchange}: {e}", "ERROR")
            self.logger.error(f"Failed to submit order to {routed_order.exchange}: {e}")
    
    async def _update_order_status_from_exchange(self, routed_order: RoutedOrder) -> None:
        """Update order status from exchange"""
        try:
            if not routed_order.exchange_order_id:
                tprint(f"No exchange order ID for order {routed_order.id}, skipping status update", "WARNING")
                return

            exchange_instance = await self.exchange_registry.get_exchange(routed_order.exchange)
            if not exchange_instance:
                tprint(f"Exchange instance not found for {routed_order.exchange}", "WARNING")
                return
            
            # Get order status from exchange
            status_result = await exchange_instance.get_order_status(
                routed_order.symbol,
                routed_order.exchange_order_id
            )
            
            if status_result:
                tprint(f"Received status update for order {routed_order.id}", "INFO")
                await self._update_order_from_exchange_response(routed_order, status_result)

        except Exception as e:
            tprint(f"Error updating order status from exchange: {e}", "ERROR")
            self.logger.error(f"Error updating order status from exchange: {e}")
    
    async def _update_order_from_exchange_response(self, routed_order: RoutedOrder, status_result: Dict[str, Any]) -> None:
        """Update order from exchange response"""
        try:
            # Map exchange status to our status enum
            exchange_status = status_result.get("status", "").upper()
            tprint(f"Updating order {routed_order.id} from exchange response: status={exchange_status}", "INFO")

            if exchange_status in ["FILLED", "COMPLETED"]:
                routed_order.status = OrderStatus.FILLED
                routed_order.filled_quantity = float(status_result.get("executedQty", routed_order.quantity))
                routed_order.average_price = float(status_result.get("avgPrice", routed_order.price or 0))
                routed_order.filled_at = datetime.now()
                tprint(f"Order {routed_order.id} filled: quantity={routed_order.filled_quantity}, avg_price={routed_order.average_price}", "SUCCESS")

                # Remove from active orders
                if routed_order.id in self.active_orders:
                    del self.active_orders[routed_order.id]
                
            elif exchange_status in ["PARTIALLY_FILLED"]:
                routed_order.status = OrderStatus.PARTIALLY_FILLED
                routed_order.filled_quantity = float(status_result.get("executedQty", 0))
                routed_order.average_price = float(status_result.get("avgPrice", routed_order.price or 0))
                tprint(f"Order {routed_order.id} partially filled: quantity={routed_order.filled_quantity}", "INFO")

            elif exchange_status in ["CANCELLED", "CANCELED"]:
                routed_order.status = OrderStatus.CANCELLED
                routed_order.cancelled_at = datetime.now()
                tprint(f"Order {routed_order.id} cancelled", "WARNING")
                if routed_order.id in self.active_orders:
                    del self.active_orders[routed_order.id]

            elif exchange_status in ["REJECTED", "EXPIRED", "FAILED"]:
                routed_order.status = OrderStatus.REJECTED
                routed_order.error_message = status_result.get("errorMessage", "Order rejected by exchange")
                tprint(f"Order {routed_order.id} rejected: {routed_order.error_message}", "ERROR")
                if routed_order.id in self.active_orders:
                    del self.active_orders[routed_order.id]
            
            # Update commission if available
            if "commission" in status_result:
                routed_order.commission = float(status_result["commission"])
            
            # Update statistics
            self._update_order_stats(routed_order)

        except Exception as e:
            tprint(f"Error updating order from exchange response: {e}", "ERROR")
            self.logger.error(f"Error updating order from exchange response: {e}")
    
    async def _monitor_orders(self) -> None:
        """Monitor active orders and update their status"""
        tprint(f"Order monitoring loop started", "INFO")
        while self._running:
            try:
                # Check each active order
                if len(self.active_orders) > 0:
                    tprint(f"Monitoring {len(self.active_orders)} active orders", "INFO")
                for order in list(self.active_orders.values()):
                    await self._update_order_status_from_exchange(order)

                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                tprint(f"Order monitoring loop cancelled", "WARNING")
                break
            except Exception as e:
                tprint(f"Error in order monitoring: {e}", "ERROR")
                self.logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5)
    
    def _update_order_stats(self, routed_order: RoutedOrder) -> None:
        """Update order statistics"""
        self.order_stats["total_routed"] += 1
        
        # Update by exchange
        if routed_order.exchange not in self.order_stats["by_exchange"]:
            self.order_stats["by_exchange"][routed_order.exchange] = 0
        self.order_stats["by_exchange"][routed_order.exchange] += 1
        
        # Update by symbol
        if routed_order.symbol not in self.order_stats["by_symbol"]:
            self.order_stats["by_symbol"][routed_order.symbol] = 0
        self.order_stats["by_symbol"][routed_order.symbol] += 1
        
        # Update by status
        status_str = routed_order.status.value
        if status_str not in self.order_stats["by_status"]:
            self.order_stats["by_status"][status_str] = 0
        self.order_stats["by_status"][status_str] += 1
        
        # Update success/failure counts
        if routed_order.status == OrderStatus.FILLED:
            self.order_stats["successful_orders"] += 1
        elif routed_order.status in [OrderStatus.FAILED, OrderStatus.REJECTED]:
            self.order_stats["failed_orders"] += 1
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get order router statistics"""
        return {
            "running": self._running,
            "statistics": self.order_stats,
            "total_orders": len(self.routed_orders),
            "active_orders": len(self.active_orders),
            "timestamp": datetime.now().isoformat()
        }