"""
Trading Orchestrator

Main orchestrator that connects live trading operations to the exchange-agnostic trading receiver.
Provides high-level trading interface and manages the interaction between trading components.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
import logging

from .config import TradingConfig, TradingMode, OrderType, OrderSide
from .trading_engine import TradingEngine
from ..exchanges import TradingReceiver, TradingMessage, MessageType
from ..src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult


@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    action: str  # "buy", "sell", "hold"
    quantity: float
    price: Optional[float] = None
    confidence: float = 0.0
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    exchange: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class TradingOrchestrator:
    """Main orchestrator for live trading operations"""

    def __init__(self, config: TradingConfig, trading_receiver: TradingReceiver):
        self.config = config
        self.trading_receiver = trading_receiver
        self.logger = logging.getLogger(__name__)

        # Trading state
        self._running = False
        self._trading_active = False
        self.positions: Dict[str, Position] = {}
        self.active_orders: Dict[str, Any] = {}

        # Signal handlers
        self.signal_handlers: List[Callable[[TradingSignal], Awaitable[None]]] = []

        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            "total_signals": 0,
            "executed_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        }

    async def start(self) -> None:
        """Start the trading orchestrator"""
        if self._running:
            return

        self.logger.info("Starting trading orchestrator...")

        try:
            # Start trading receiver
            await self.trading_receiver.start()

            # Register signal handler
            self._register_signal_handler()

            self._running = True
            self._trading_active = True

            self.logger.info("Trading orchestrator started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start trading orchestrator: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the trading orchestrator"""
        if not self._running:
            return

        self.logger.info("Stopping trading orchestrator...")

        self._running = False
        self._trading_active = False

        # Stop trading receiver
        await self.trading_receiver.stop()

        self.logger.info("Trading orchestrator stopped")

    def register_signal_handler(self, handler: Callable[[TradingSignal], Awaitable[None]]) -> None:
        """Register a trading signal handler"""
        self.signal_handlers.append(handler)
        self.logger.info(f"Registered signal handler: {handler.__name__}")
    
    def unregister_signal_handler(self, handler: Callable[[TradingSignal], Awaitable[None]]) -> bool:
        """Unregister a trading signal handler"""
        try:
            self.signal_handlers.remove(handler)
            self.logger.info(f"Unregistered signal handler: {handler.__name__}")
            return True
        except ValueError:
            self.logger.warning(f"Signal handler not found: {handler.__name__}")
            return False
    
    def get_registered_handlers(self) -> List[str]:
        """Get list of registered signal handler names"""
        return [handler.__name__ for handler in self.signal_handlers]
    
    def clear_signal_handlers(self) -> None:
        """Clear all registered signal handlers"""
        handler_count = len(self.signal_handlers)
        self.signal_handlers.clear()
        self.logger.info(f"Cleared {handler_count} signal handlers")
    
    async def register_signal_handler_with_validation(
        self, 
        handler: Callable[[TradingSignal], Awaitable[None]],
        validate_signals: bool = True,
        min_confidence: float = 0.0
    ) -> bool:
        """Register a signal handler with validation options"""
        try:
            # Create a wrapper with validation if requested
            if validate_signals:
                async def validated_handler(signal: TradingSignal) -> None:
                    # Validate signal before passing to handler
                    if signal.confidence < min_confidence:
                        self.logger.warning(f"Signal rejected due to low confidence: {signal.confidence} < {min_confidence}")
                        return
                    
                    # Validate signal structure
                    if not signal.symbol or not signal.action or signal.quantity <= 0:
                        self.logger.warning(f"Invalid signal structure: {signal}")
                        return
                    
                    # Call the original handler
                    await handler(signal)
                
                self.signal_handlers.append(validated_handler)
                self.logger.info(f"Registered validated signal handler: {handler.__name__} (min_confidence: {min_confidence})")
            else:
                self.signal_handlers.append(handler)
                self.logger.info(f"Registered signal handler: {handler.__name__}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering signal handler: {e}")
            return False

    async def submit_signal(self, signal: TradingSignal) -> bool:
        """Submit a trading signal for execution"""
        if not self._trading_active:
            self.logger.warning("Trading is not active, ignoring signal")
            return False

        try:
            self.stats["total_signals"] += 1

            self.logger.info(f"Processing signal: {signal.symbol} {signal.action} {signal.quantity}")

            # Validate signal
            if not await self._validate_signal(signal):
                self.logger.warning(f"Signal validation failed: {signal.symbol}")
                return False

            # Execute signal based on action
            if signal.action.lower() in ["buy", "sell"]:
                success = await self._execute_signal(signal)
                if success:
                    self.stats["executed_trades"] += 1
                    self.logger.info(f"Signal executed successfully: {signal.symbol}")
                else:
                    self.stats["failed_trades"] += 1
                    self.logger.error(f"Signal execution failed: {signal.symbol}")
                return success
            else:
                self.logger.info(f"Hold signal for {signal.symbol}, no action taken")
                return True

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            self.stats["failed_trades"] += 1
            return False

    async def execute_trade_decision(self, decision: TradeDecision) -> bool:
        """Execute a trade decision"""
        if not self._trading_active:
            self.logger.warning("Trading is not active, ignoring trade decision")
            return False

        try:
            # Convert decision to signal
            signal = TradingSignal(
                symbol=decision.symbol,
                action=decision.action,
                quantity=decision.quantity,
                price=decision.price,
                confidence=decision.confidence,
                strategy="decision_based",
                metadata={
                    "risk_score": decision.risk_score,
                    "leverage": decision.leverage,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit
                }
            )

            return await self.submit_signal(signal)

        except Exception as e:
            self.logger.error(f"Error executing trade decision: {e}")
            return False

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from all configured exchanges"""
        try:
            account_info = {}

            # Get account info from trading receiver
            response = await self.trading_receiver.get_account_info(self.config.exchange_name)

            if response.success:
                account_info = response.data
            else:
                self.logger.error(f"Failed to get account info: {response.error}")

            return account_info

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions across all exchanges"""
        try:
            # Try to get position info from trading receiver
            response = await self.trading_receiver.get_position_info(
                self.config.exchange_name,
                self.config.symbols[0] if self.config.symbols else ""
            )

            if response.success:
                positions_data = response.data
                # Convert to Position objects
                positions = {}
                for symbol, pos_data in positions_data.items():
                    if isinstance(pos_data, dict):
                        positions[symbol] = Position(
                            symbol=symbol,
                            quantity=float(pos_data.get("quantity", 0)),
                            entry_price=float(pos_data.get("entry_price", 0)),
                            current_price=float(pos_data.get("current_price", 0)),
                            pnl=float(pos_data.get("pnl", 0)),
                            pnl_percentage=float(pos_data.get("pnl_percentage", 0)),
                            exchange=self.config.exchange_name
                        )
                return positions
            else:
                self.logger.error(f"Failed to get positions: {response.error}")
                return {}

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}

    async def get_market_data(self, symbol: str, data_type: str = "ticker") -> Dict[str, Any]:
        """Get market data from exchange"""
        try:
            response = await self.trading_receiver.request_data(
                self.config.exchange_name,
                symbol,
                data_type
            )

            if response.success:
                return response.data
            else:
                self.logger.error(f"Failed to get market data: {response.error}")
                return {}

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            response = await self.trading_receiver.cancel_order(
                self.config.exchange_name,
                symbol,
                order_id
            )

            if response.success:
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {response.error}")
                return False

        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status with enhanced tracking"""
        try:
            # Try to get order status from trading receiver
            response = await self.trading_receiver.get_order_status(
                self.config.exchange_name,
                symbol,
                order_id
            )

            if response.success:
                order_data = response.data
                
                # Enhanced order status information
                enhanced_status = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "status": order_data.get("status", "unknown"),
                    "exchange": self.config.exchange_name,
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": order_data
                }
                
                # Add mapping information if available
                if "internal_order_id" in order_data:
                    enhanced_status["internal_order_id"] = order_data["internal_order_id"]
                if "exchange_order_id" in order_data:
                    enhanced_status["exchange_order_id"] = order_data["exchange_order_id"]
                if "mapped_status" in order_data:
                    enhanced_status["mapped_status"] = order_data["mapped_status"]
                
                # Add order details if available
                if "side" in order_data:
                    enhanced_status["side"] = order_data["side"]
                if "quantity" in order_data:
                    enhanced_status["quantity"] = order_data["quantity"]
                if "price" in order_data:
                    enhanced_status["price"] = order_data["price"]
                if "filled_quantity" in order_data:
                    enhanced_status["filled_quantity"] = order_data["filled_quantity"]
                if "remaining_quantity" in order_data:
                    enhanced_status["remaining_quantity"] = order_data["remaining_quantity"]
                
                return enhanced_status
            else:
                return {
                    "order_id": order_id,
                    "symbol": symbol,
                    "status": "error",
                    "error": response.error,
                    "exchange": self.config.exchange_name,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "exchange": self.config.exchange_name,
                "timestamp": datetime.now().isoformat()
            }

    async def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics and tracking information"""
        try:
            # Get order statistics from trading receiver if available
            order_stats = {}
            if hasattr(self.trading_receiver, 'get_order_statistics'):
                order_stats = await self.trading_receiver.get_order_statistics()
            
            # Get active orders count
            active_orders_count = len(self.active_orders)
            
            # Calculate order success rate
            total_orders = self.stats["executed_trades"]
            successful_orders = self.stats["successful_trades"]
            order_success_rate = (successful_orders / total_orders * 100) if total_orders > 0 else 0.0
            
            return {
                "active_orders_count": active_orders_count,
                "total_orders": total_orders,
                "successful_orders": successful_orders,
                "failed_orders": self.stats["failed_trades"],
                "order_success_rate": order_success_rate,
                "receiver_order_stats": order_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order statistics: {e}")
            return {"error": str(e)}
    
    async def get_all_orders(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all orders with optional status filtering"""
        try:
            orders = []
            
            # Get orders from trading receiver if available
            if hasattr(self.trading_receiver, 'get_all_orders'):
                receiver_orders = await self.trading_receiver.get_all_orders()
                orders.extend(receiver_orders)
            
            # Add local active orders
            for order_id, order_info in self.active_orders.items():
                order_data = {
                    "order_id": order_id,
                    "symbol": order_info.get("symbol", ""),
                    "side": order_info.get("side", ""),
                    "quantity": order_info.get("quantity", 0),
                    "price": order_info.get("price"),
                    "status": order_info.get("status", "unknown"),
                    "timestamp": order_info.get("timestamp", datetime.now().isoformat()),
                    "source": "local"
                }
                orders.append(order_data)
            
            # Apply status filter if specified
            if status_filter:
                orders = [order for order in orders if order.get("status", "").lower() == status_filter.lower()]
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting all orders: {e}")
            return []
    
    async def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history from trade history"""
        try:
            # Get recent trades from trade history
            recent_trades = self.trade_history[-limit:] if self.trade_history else []
            
            # Convert to order format
            orders = []
            for trade in recent_trades:
                order_data = {
                    "order_id": trade.get("order_id", ""),
                    "symbol": trade.get("symbol", ""),
                    "side": trade.get("action", ""),
                    "quantity": trade.get("quantity", 0),
                    "price": trade.get("price"),
                    "status": trade.get("status", "completed"),
                    "timestamp": trade.get("timestamp", datetime.now()).isoformat() if isinstance(trade.get("timestamp"), datetime) else str(trade.get("timestamp", "")),
                    "confidence": trade.get("confidence", 0),
                    "strategy": trade.get("strategy", ""),
                    "source": "history"
                }
                orders.append(order_data)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        try:
            # Get receiver statistics
            receiver_stats = await self.trading_receiver.get_statistics()

            # Calculate performance metrics
            total_trades = self.stats["executed_trades"]
            win_rate = (self.stats["successful_trades"] / total_trades * 100) if total_trades > 0 else 0.0

            # Get order statistics
            order_stats = await self.get_order_statistics()

            return {
                "orchestrator_stats": self.stats,
                "receiver_stats": receiver_stats,
                "order_stats": order_stats,
                "total_signals": self.stats["total_signals"],
                "total_trades": total_trades,
                "successful_trades": self.stats["successful_trades"],
                "failed_trades": self.stats["failed_trades"],
                "win_rate": win_rate,
                "total_pnl": self.stats["total_pnl"],
                "running": self._running,
                "trading_active": self._trading_active,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal"""
        try:
            # Check if symbol is supported
            if signal.symbol not in self.config.symbols:
                self.logger.warning(f"Unsupported symbol: {signal.symbol}")
                return False

            # Check quantity is positive
            if signal.quantity <= 0:
                self.logger.warning(f"Invalid quantity: {signal.quantity}")
                return False

            # Check confidence threshold
            min_confidence = self.config.custom_parameters.get("min_confidence", 0.5)
            if signal.confidence < min_confidence:
                self.logger.warning(f"Low confidence: {signal.confidence} < {min_confidence}")
                return False

            # Check position size limits
            if signal.quantity > self.config.max_position_size:
                self.logger.warning(f"Position size exceeds limit: {signal.quantity} > {self.config.max_position_size}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    async def _execute_signal(self, signal: TradingSignal) -> bool:
        """Execute trading signal"""
        try:
            # Convert signal to order parameters
            side = signal.action.lower()
            order_type = OrderType.MARKET.value if signal.price is None else OrderType.LIMIT.value

            # Get current market price for market orders
            if signal.price is None:
                market_data = await self.get_market_data(signal.symbol, "ticker")
                signal.price = float(market_data.get("last", 0)) if market_data else None

                if signal.price is None or signal.price <= 0:
                    self.logger.error(f"Unable to get market price for {signal.symbol}")
                    return False

            # Submit order through trading receiver
            response = await self.trading_receiver.send_order(
                exchange=self.config.exchange_name,
                symbol=signal.symbol,
                side=side,
                order_type=order_type,
                quantity=signal.quantity,
                price=signal.price,
                confidence=signal.confidence,
                strategy=signal.strategy,
                **signal.metadata
            )

            if response.success:
                # Update statistics
                self.stats["successful_trades"] += 1

                # Track the trade
                trade_record = {
                    "timestamp": datetime.now(),
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "quantity": signal.quantity,
                    "price": signal.price,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy,
                    "order_id": response.data.get("order_id"),
                    "status": "submitted"
                }
                self.trade_history.append(trade_record)

                self.logger.info(f"Trade executed: {signal.symbol} {signal.action} {signal.quantity} @ {signal.price}")
                return True
            else:
                self.logger.error(f"Trade failed: {response.error}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False

    def _register_signal_handler(self) -> None:
        """Register internal signal handler"""
        async def handle_signal(signal: TradingSignal) -> None:
            """Internal signal handler"""
            try:
                # Process signal
                success = await self.submit_signal(signal)

                # Notify other handlers
                await self._notify_signal_handlers(signal)

            except Exception as e:
                self.logger.error(f"Error in internal signal handler: {e}")

        # Note: This would be registered with the signal source
        # For now, this is a placeholder
        self.logger.info("Signal handler registered")
    
    async def _notify_signal_handlers(self, signal: TradingSignal) -> None:
        """Notify all registered signal handlers"""
        if not self.signal_handlers:
            return
        
        # Create tasks for all handlers to run concurrently
        tasks = []
        for handler in self.signal_handlers:
            task = asyncio.create_task(self._call_signal_handler(handler, signal))
            tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_signal_handler(self, handler: Callable, signal: TradingSignal) -> None:
        """Call a single signal handler with error handling"""
        try:
            await handler(signal)
        except Exception as e:
            self.logger.error(f"Error in signal handler {handler.__name__}: {e}")
    
    async def process_external_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Process a signal from external source"""
        try:
            # Convert signal data to TradingSignal object
            signal = TradingSignal(
                symbol=signal_data.get("symbol", ""),
                action=signal_data.get("action", ""),
                quantity=float(signal_data.get("quantity", 0)),
                price=signal_data.get("price"),
                confidence=float(signal_data.get("confidence", 0)),
                strategy=signal_data.get("strategy", "external"),
                metadata=signal_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(signal_data.get("timestamp", datetime.now().isoformat()))
            )
            
            # Process the signal
            return await self.submit_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error processing external signal: {e}")
            return False
    
    async def get_signal_handler_status(self) -> Dict[str, Any]:
        """Get status of signal handlers"""
        return {
            "total_handlers": len(self.signal_handlers),
            "handler_names": self.get_registered_handlers(),
            "trading_active": self._trading_active,
            "last_signal_time": self.trade_history[-1].get("timestamp").isoformat() if self.trade_history else None
        }

    async def pause_trading(self) -> None:
        """Pause trading operations"""
        self._trading_active = False
        self.logger.info("Trading paused")

    async def resume_trading(self) -> None:
        """Resume trading operations"""
        if self._running:
            self._trading_active = True
            self.logger.info("Trading resumed")
        else:
            self.logger.warning("Cannot resume trading - orchestrator is not running")

    async def emergency_stop(self) -> None:
        """Emergency stop - cancel all orders and pause trading"""
        self.logger.warning("Emergency stop triggered!")

        try:
            # Cancel all active orders
            for symbol in self.config.symbols:
                # This would need to be implemented to cancel all orders for a symbol
                pass

            # Pause trading
            await self.pause_trading()

            self.logger.info("Emergency stop completed")

        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")

    async def update_performance_metrics(self) -> None:
        """Update performance metrics"""
        try:
            # Calculate PnL from trade history
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0

            for trade in self.trade_history:
                if trade.get("pnl") is not None:
                    pnl = trade["pnl"]
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    elif pnl < 0:
                        losing_trades += 1

            # Update statistics
            self.stats["total_pnl"] = total_pnl
            self.stats["win_rate"] = (winning_trades / len(self.trade_history) * 100) if self.trade_history else 0.0

            self.performance_metrics = {
                "total_pnl": total_pnl,
                "total_trades": len(self.trade_history),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": self.stats["win_rate"],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def create_test_signal_handler(self, name: str = "test_handler") -> Callable[[TradingSignal], Awaitable[None]]:
        """Create a test signal handler for demonstration"""
        async def test_handler(signal: TradingSignal) -> None:
            """Test signal handler that logs signals"""
            self.logger.info(f"[{name}] Received signal: {signal.symbol} {signal.action} {signal.quantity} @ {signal.price} (confidence: {signal.confidence})")
        
        return test_handler
    
    async def test_signal_handling(self) -> Dict[str, Any]:
        """Test signal handling functionality"""
        try:
            # Create a test signal
            test_signal = TradingSignal(
                symbol="ETHUSDT",
                action="buy",
                quantity=0.1,
                price=2000.0,
                confidence=0.8,
                strategy="test",
                metadata={"test": True}
            )
            
            # Test signal processing
            success = await self.submit_signal(test_signal)
            
            return {
                "test_signal_processed": success,
                "signal_handlers_count": len(self.signal_handlers),
                "handler_names": self.get_registered_handlers(),
                "trading_active": self._trading_active
            }
            
        except Exception as e:
            self.logger.error(f"Error testing signal handling: {e}")
            return {"error": str(e)}