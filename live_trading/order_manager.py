"""
Order Management System

Handles order creation, tracking, and execution for live trading.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import OrderType, OrderSide, TradingConfig
from ..src.interfaces.base_interfaces import TradeDecision


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: str = "USDT"
    timestamp: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to be filled"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.filled_quantity >= self.quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]


class OrderManager:
    """Manages order lifecycle and execution"""
    
    def __init__(self, config: TradingConfig, exchange_client: Any, simulator: Any = None):
        self.config = config
        self.exchange_client = exchange_client
        self.simulator = simulator  # PaperTradingSimulator instance for paper mode
        self.logger = logging.getLogger(__name__)
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        
        # Event handlers
        self.order_handlers: Dict[str, List[Callable[[Order], Awaitable[None]]]] = {
            "on_order_created": [],
            "on_order_filled": [],
            "on_order_cancelled": [],
            "on_order_failed": [],
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self) -> None:
        """Start the order manager"""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_orders())
        self.logger.info("Order manager started")
    
    async def stop(self) -> None:
        """Stop the order manager"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Order manager stopped")
    
    def register_handler(self, event_type: str, handler: Callable[[Order], Awaitable[None]]) -> None:
        """Register event handler"""
        if event_type in self.order_handlers:
            self.order_handlers[event_type].append(handler)
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Create a new order"""
        
        # Generate unique order ID
        order_id = f"{symbol}_{side.value}_{int(time.time() * 1000)}"
        
        # Create order object
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        # Store order
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        
        # Execute order based on mode
        if self.config.mode.value == "paper":
            await self._execute_paper_order(order)
        else:
            await self._execute_live_order(order)
        
        # Notify handlers
        await self._notify_handlers("on_order_created", order)
        
        self.logger.info(f"Order created: {order_id} - {symbol} {side.value} {quantity} @ {price or 'MARKET'}")
        
        return order
    
    async def create_order_from_decision(self, decision: TradeDecision) -> Order:
        """Create order from trade decision"""
        side = OrderSide.BUY if decision.action.lower() in ["buy", "long"] else OrderSide.SELL
        order_type = OrderType.MARKET if decision.price == 0 else OrderType.LIMIT
        
        return await self.create_order(
            symbol=decision.symbol,
            side=side,
            order_type=order_type,
            quantity=decision.quantity,
            price=decision.price if decision.price > 0 else None,
            metadata={
                "confidence": decision.confidence,
                "risk_score": decision.risk_score,
                "leverage": decision.leverage,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit
            }
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            self.logger.error(f"❌ Order not found: {order_id}")
            self.logger.warning("⚠️ Order cancellation failed - order does not exist")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            self.logger.error(f"❌ Order is not active: {order_id} (status: {order.status.value})")
            self.logger.warning("⚠️ Cannot cancel order - it is already completed, cancelled, or failed")
            return False
        
        try:
            if self.config.mode.value == "paper":
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
            else:
                # Cancel on exchange
                if order.exchange_order_id and self.exchange_client:
                    await self.exchange_client.cancel_order(order.symbol, order.exchange_order_id)
                
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            # Notify handlers
            await self._notify_handlers("on_order_cancelled", order)
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            self.logger.warning("⚠️ Order cancellation failed - order may still be active on exchange")
            order.error_message = str(e)
            order.status = OrderStatus.FAILED
            await self._notify_handlers("on_order_failed", order)
            return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders, optionally filtered by symbol"""
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history, optionally filtered by symbol"""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        # Sort by timestamp descending
        orders.sort(key=lambda x: x.timestamp, reverse=True)
        
        return orders[:limit]
    
    async def _execute_paper_order(self, order: Order) -> None:
        """Execute order in paper trading mode using the simulator"""
        try:
            # If simulator is available, use it for realistic paper trading
            if self.simulator:
                # Fetch order book for accurate simulation
                order_book = await self.exchange_client.get_order_book(order.symbol, limit=20)
                if not order_book:
                    raise Exception(f"Failed to fetch order book for {order.symbol}")
                
                # Extract trading signal metadata from order metadata if present
                trading_signal_metadata = order.metadata.get("trading_signal", {})
                
                # Simulate order using the simulator
                result = await self.simulator.simulate_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    quantity=order.quantity,
                    price=order.price,
                    order_book=order_book,
                    trading_signal_metadata=trading_signal_metadata
                )
                
                # Update order from simulator response
                if result.get("status") == "FILLED":
                    order.average_price = result.get("avgPrice", result.get("fillPrice", order.price or 0))
                    order.filled_quantity = result.get("filledQuantity", order.quantity)
                    order.status = OrderStatus.FILLED
                    order.commission = result.get("fee", 0.0)
                    order.exchange_order_id = result.get("orderId")
                    
                    # Remove from active orders
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                    
                    # Notify handlers
                    await self._notify_handlers("on_order_filled", order)
                    
                    self.logger.info(f"Paper order filled via simulator: {order.id} @ {order.average_price}")
                elif result.get("status") == "REJECTED":
                    order.status = OrderStatus.REJECTED
                    order.error_message = result.get("rejectedReason", "Order rejected by simulator")
                    await self._notify_handlers("on_order_failed", order)
                    self.logger.warning(f"Paper order rejected: {order.error_message}")
                else:
                    order.status = OrderStatus.SUBMITTED
                    order.updated_at = datetime.now()
            
            else:
                # Fallback to simple simulation if no simulator
                await asyncio.sleep(0.1)  # Simulate network delay
                
                if order.order_type == OrderType.MARKET:
                    ticker = await self.exchange_client.get_ticker(order.symbol)
                    if ticker and "last" in ticker:
                        order.average_price = float(ticker["last"])
                    else:
                        order.average_price = order.price or 50000.0
                    
                    order.filled_quantity = order.quantity
                    order.status = OrderStatus.FILLED
                    order.updated_at = datetime.now()
                    
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                    
                    await self._notify_handlers("on_order_filled", order)
                    self.logger.info(f"Paper order filled (simple): {order.id} @ {order.average_price}")
                else:
                    order.status = OrderStatus.SUBMITTED
                    order.updated_at = datetime.now()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute paper order {order.id}: {e}")
            self.logger.warning("⚠️ Paper order execution failed - order will not be filled")
            order.error_message = str(e)
            order.status = OrderStatus.FAILED
            await self._notify_handlers("on_order_failed", order)
    
    async def _execute_live_order(self, order: Order) -> None:
        """Execute order on live exchange"""
        try:
            # Prepare order parameters
            order_params = {
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value.upper()
            }
            
            if order.price:
                order_params["price"] = order.price
            
            # Submit order to exchange
            response = await self.exchange_client.create_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type.value.upper()
            )
            
            if response and "orderId" in response:
                order.exchange_order_id = str(response["orderId"])
                order.status = OrderStatus.SUBMITTED
                order.updated_at = datetime.now()
                
                self.logger.info(f"Live order submitted: {order.id} -> {order.exchange_order_id}")
            else:
                raise Exception("Invalid response from exchange")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute live order {order.id}: {e}")
            self.logger.warning("⚠️ Live order execution failed - order may not have been submitted to exchange")
            order.error_message = str(e)
            order.status = OrderStatus.FAILED
            await self._notify_handlers("on_order_failed", order)
    
    async def _monitor_orders(self) -> None:
        """Monitor active orders and update their status"""
        while self._running:
            try:
                # Check each active order
                for order_id, order in list(self.active_orders.items()):
                    if self.config.mode.value == "paper":
                        # For paper trading, we don't need to monitor
                        continue
                    
                    try:
                        # Get order status from exchange
                        if order.exchange_order_id and self.exchange_client:
                            status_response = await self.exchange_client.get_order_status(
                                order.symbol, order.exchange_order_id
                            )
                            
                            if status_response:
                                await self._update_order_from_exchange(order, status_response)
                    
                    except Exception as e:
                        self.logger.error(f"❌ Failed to check order status {order_id}: {e}")
                        self.logger.warning(f"⚠️ Order monitoring failed for {order_id} - status may be outdated")
                
                # Wait before next check
                await asyncio.sleep(self.config.data_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in order monitoring: {e}")
                self.logger.warning("⚠️ Order monitoring loop failed - continuing after retry delay")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _update_order_from_exchange(self, order: Order, status_response: Dict[str, Any]) -> None:
        """Update order status from exchange response"""
        try:
            # Map exchange status to our status enum
            exchange_status = status_response.get("status", "").upper()
            
            if exchange_status in ["FILLED", "COMPLETED"]:
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(status_response.get("executedQty", order.quantity))
                order.average_price = float(status_response.get("avgPrice", order.price or 0))
                
                # Remove from active orders
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                
                await self._notify_handlers("on_order_filled", order)
                
            elif exchange_status in ["PARTIALLY_FILLED", "PARTIALLY_FILLED"]:
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = float(status_response.get("executedQty", 0))
                order.average_price = float(status_response.get("avgPrice", order.price or 0))
                
            elif exchange_status in ["CANCELLED", "CANCELED"]:
                order.status = OrderStatus.CANCELLED
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                await self._notify_handlers("on_order_cancelled", order)
                
            elif exchange_status in ["REJECTED", "EXPIRED", "FAILED"]:
                order.status = OrderStatus.REJECTED if exchange_status == "REJECTED" else OrderStatus.EXPIRED
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                await self._notify_handlers("on_order_failed", order)
            
            order.updated_at = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update order {order.id}: {e}")
            self.logger.warning(f"⚠️ Order update failed for {order.id} - order status may be incorrect")
    
    async def _notify_handlers(self, event_type: str, order: Order) -> None:
        """Notify registered handlers of order events"""
        if event_type in self.order_handlers:
            for handler in self.order_handlers[event_type]:
                try:
                    await handler(order)
                except Exception as e:
                    self.logger.error(f"❌ Error in order handler: {e}")
                    self.logger.warning("⚠️ Order handler failed - continuing with other handlers")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get order performance metrics"""
        total_orders = len(self.orders)
        active_orders = len(self.active_orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        failed_orders = len([o for o in self.orders.values() if o.status in [OrderStatus.FAILED, OrderStatus.REJECTED]])
        
        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "failed_orders": failed_orders,
            "success_rate": filled_orders / total_orders if total_orders > 0 else 0.0,
            "failure_rate": failed_orders / total_orders if total_orders > 0 else 0.0
        }