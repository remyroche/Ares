"""
Order Simulator

Handles order simulation for paper trading mode.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

from .simulator_interface import ISimulator, SimulatedOrder, SimulatedPosition


class OrderSimulator(ISimulator):
    """Simulates order execution for paper trading"""

    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize the order simulator
        
        Args:
            initial_balance: Starting balance for simulation
        """
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.orders: Dict[str, SimulatedOrder] = {}
        self.positions: Dict[str, SimulatedPosition] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self._monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the simulator"""
        self.logger.info("Initializing Order Simulator")
        self.running = True
        self._monitoring_task = asyncio.create_task(self._monitor_orders())
        self.logger.info(f"Order Simulator initialized with balance: ${self.balance:,.2f}")

    async def close(self) -> None:
        """Close the simulator"""
        self.running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Order Simulator closed")

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a simulated order"""
        try:
            # Generate unique order ID
            order_id = f"SIM_{uuid.uuid4().hex[:8]}"
            
            # Create simulated order
            order = SimulatedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status="PENDING",
                created_at=datetime.now(timezone.utc),
                metadata=kwargs
            )
            
            # Store order
            self.orders[order_id] = order
            
            # Process order based on type
            if order_type.upper() == "MARKET":
                await self._process_market_order(order)
            elif order_type.upper() == "LIMIT":
                await self._process_limit_order(order)
            else:
                order.status = "REJECTED"
                order.metadata["error"] = f"Unsupported order type: {order_type}"
            
            self.logger.info(f"Created simulated order: {order_id} - {side} {quantity} {symbol}")
            
            return {
                "success": True,
                "order_id": order_id,
                "status": order.status,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "created_at": order.created_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating simulated order: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a simulated order"""
        try:
            if order_id not in self.orders:
                return {
                    "success": False,
                    "error": f"Order {order_id} not found"
                }
            
            order = self.orders[order_id]
            
            if order.status not in ["PENDING", "SUBMITTED", "PARTIALLY_FILLED"]:
                return {
                    "success": False,
                    "error": f"Cannot cancel order in status: {order.status}"
                }
            
            order.status = "CANCELLED"
            self.logger.info(f"Cancelled simulated order: {order_id}")
            
            return {
                "success": True,
                "order_id": order_id,
                "status": order.status
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling simulated order: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get simulated order status"""
        try:
            if order_id not in self.orders:
                return {
                    "success": False,
                    "error": f"Order {order_id} not found"
                }
            
            order = self.orders[order_id]
            
            return {
                "success": True,
                "order_id": order_id,
                "status": order.status,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "commission": order.commission,
                "created_at": order.created_at.isoformat(),
                "filled_at": order.filled_at.isoformat() if order.filled_at else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open simulated orders"""
        try:
            open_orders = []
            
            for order in self.orders.values():
                if order.status in ["PENDING", "SUBMITTED", "PARTIALLY_FILLED"]:
                    if symbol is None or order.symbol == symbol:
                        open_orders.append({
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side,
                            "order_type": order.order_type,
                            "quantity": order.quantity,
                            "price": order.price,
                            "status": order.status,
                            "filled_quantity": order.filled_quantity,
                            "created_at": order.created_at.isoformat()
                        })
            
            return open_orders
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get simulated positions"""
        try:
            positions = []
            
            for position in self.positions.values():
                if symbol is None or position.symbol == symbol:
                    positions.append({
                        "symbol": position.symbol,
                        "side": position.side,
                        "quantity": position.quantity,
                        "average_price": position.average_price,
                        "unrealized_pnl": position.unrealized_pnl,
                        "realized_pnl": position.realized_pnl,
                        "created_at": position.created_at.isoformat(),
                        "updated_at": position.updated_at.isoformat()
                    })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    async def get_portfolio_value(self) -> Dict[str, Any]:
        """Get simulated portfolio value"""
        try:
            total_value = self.balance
            total_unrealized_pnl = 0.0
            total_realized_pnl = 0.0
            
            for position in self.positions.values():
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
                
                # Calculate position value based on current market price
                if position.symbol in self.market_data:
                    current_price = self.market_data[position.symbol].get("price", position.average_price)
                    position_value = position.quantity * current_price
                    total_value += position_value
            
            return {
                "total_value": total_value,
                "cash_balance": self.balance,
                "unrealized_pnl": total_unrealized_pnl,
                "realized_pnl": total_realized_pnl,
                "total_pnl": total_unrealized_pnl + total_realized_pnl,
                "position_count": len(self.positions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return {
                "total_value": self.balance,
                "cash_balance": self.balance,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_pnl": 0.0,
                "position_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def update_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Update market data for simulation"""
        try:
            self.market_data[symbol] = market_data
            
            # Update unrealized PnL for positions
            if symbol in self.positions:
                position = self.positions[symbol]
                current_price = market_data.get("price", position.average_price)
                
                if position.side.upper() == "BUY":
                    position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                else:  # SELL
                    position.unrealized_pnl = (position.average_price - current_price) * position.quantity
                
                position.updated_at = datetime.now(timezone.utc)
            
            self.logger.debug(f"Updated market data for {symbol}: {market_data}")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")

    async def process_order_book(self, symbol: str, order_book: Dict[str, Any]) -> None:
        """Process order book data for accurate simulation"""
        try:
            self.order_books[symbol] = order_book
            
            # Use order book for more accurate fill simulation
            if symbol in self.orders:
                await self._process_pending_orders(symbol)
            
            self.logger.debug(f"Processed order book for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing order book: {e}")

    async def _process_market_order(self, order: SimulatedOrder) -> None:
        """Process a market order"""
        try:
            # Get current market price
            current_price = await self._get_current_price(order.symbol)
            
            if current_price is None:
                order.status = "REJECTED"
                order.metadata["error"] = "No market data available"
                return
            
            # Check if we have enough balance
            required_amount = order.quantity * current_price
            
            if order.side.upper() == "BUY" and required_amount > self.balance:
                order.status = "REJECTED"
                order.metadata["error"] = "Insufficient balance"
                return
            
            # Fill the order
            await self._fill_order(order, current_price)
            
        except Exception as e:
            order.status = "REJECTED"
            order.metadata["error"] = str(e)

    async def _process_limit_order(self, order: SimulatedOrder) -> None:
        """Process a limit order"""
        try:
            if order.price is None:
                order.status = "REJECTED"
                order.metadata["error"] = "Limit order requires price"
                return
            
            # Check if we have enough balance
            required_amount = order.quantity * order.price
            
            if order.side.upper() == "BUY" and required_amount > self.balance:
                order.status = "REJECTED"
                order.metadata["error"] = "Insufficient balance"
                return
            
            # Set status to submitted (will be filled when price is reached)
            order.status = "SUBMITTED"
            
        except Exception as e:
            order.status = "REJECTED"
            order.metadata["error"] = str(e)

    async def _fill_order(self, order: SimulatedOrder, fill_price: float) -> None:
        """Fill an order at the given price"""
        try:
            order.status = "FILLED"
            order.filled_quantity = order.quantity
            order.average_price = fill_price
            order.filled_at = datetime.now(timezone.utc)
            
            # Calculate commission (0.1% for simulation)
            commission_rate = 0.001
            order.commission = order.quantity * fill_price * commission_rate
            
            # Update balance
            if order.side.upper() == "BUY":
                self.balance -= (order.quantity * fill_price + order.commission)
            else:  # SELL
                self.balance += (order.quantity * fill_price - order.commission)
            
            # Update positions
            await self._update_position(order)
            
            self.logger.info(f"Filled order {order.order_id} at {fill_price}")
            
        except Exception as e:
            self.logger.error(f"Error filling order: {e}")

    async def _update_position(self, order: SimulatedOrder) -> None:
        """Update position after order fill"""
        try:
            symbol = order.symbol
            side = order.side.upper()
            quantity = order.filled_quantity
            price = order.average_price
            
            if symbol not in self.positions:
                self.positions[symbol] = SimulatedPosition(
                    symbol=symbol,
                    side=side,
                    quantity=0.0,
                    average_price=0.0,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
            
            position = self.positions[symbol]
            
            if position.side == side:
                # Same side - add to position
                total_quantity = position.quantity + quantity
                total_value = (position.quantity * position.average_price) + (quantity * price)
                position.average_price = total_value / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
            else:
                # Opposite side - reduce position
                if quantity >= position.quantity:
                    # Close position and potentially reverse
                    remaining_quantity = quantity - position.quantity
                    position.quantity = remaining_quantity
                    position.side = side
                    position.average_price = price
                else:
                    # Reduce position
                    position.quantity -= quantity
            
            position.updated_at = datetime.now(timezone.utc)
            
            # Remove position if quantity is zero
            if position.quantity == 0:
                del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            if symbol in self.market_data:
                return self.market_data[symbol].get("price")
            
            # Try to get from order book
            if symbol in self.order_books:
                bids = self.order_books[symbol].get("bids", [])
                asks = self.order_books[symbol].get("asks", [])
                
                if bids and asks:
                    best_bid = float(bids[0][0]) if bids[0] else 0
                    best_ask = float(asks[0][0]) if asks[0] else 0
                    return (best_bid + best_ask) / 2
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    async def _process_pending_orders(self, symbol: str) -> None:
        """Process pending orders for a symbol"""
        try:
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                return
            
            for order in self.orders.values():
                if (order.symbol == symbol and 
                    order.status == "SUBMITTED" and 
                    order.order_type.upper() == "LIMIT"):
                    
                    # Check if limit order should be filled
                    should_fill = False
                    
                    if order.side.upper() == "BUY" and current_price <= order.price:
                        should_fill = True
                    elif order.side.upper() == "SELL" and current_price >= order.price:
                        should_fill = True
                    
                    if should_fill:
                        await self._fill_order(order, order.price)
            
        except Exception as e:
            self.logger.error(f"Error processing pending orders: {e}")

    async def _monitor_orders(self) -> None:
        """Monitor orders for status updates"""
        while self.running:
            try:
                # Process pending orders for all symbols
                for symbol in set(order.symbol for order in self.orders.values()):
                    await self._process_pending_orders(symbol)
                
                # Wait before next check
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(1)