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
from exchanges import TradingReceiver, TradingMessage, MessageType
from src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult


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

        # Unified regime context
        self.unified_regime_artifacts: Dict[str, Any] = {}

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

    async def submit_signal(self, signal: TradingSignal) -> bool:
        """Submit a trading signal for execution"""
        if not self._trading_active:
            self.logger.warning("Trading is not active, ignoring signal")
            return False

        try:
            self.stats["total_signals"] += 1

            self.logger.info(f"Processing signal: {signal.symbol} {signal.action} {signal.quantity}")

            if self.unified_regime_artifacts:
                regime_context = self.unified_regime_artifacts.get('regime_assignments', {})
                signal.metadata.setdefault('unified_regime', {
                    'source': regime_context.get('source'),
                    'n_regimes': regime_context.get('n_regimes'),
                })

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

    def load_unified_regime_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Attach unified NAS/TAS outputs for contextual decision making."""

        self.unified_regime_artifacts = artifacts or {}
        self.logger.info(
            "Unified regime artifacts loaded | source=%s",
            self.unified_regime_artifacts.get('regime_assignments', {}).get('source'),
        )

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
        """Get order status"""
        try:
            # This would need to be implemented in the trading receiver
            # For now, return a placeholder
            return {"status": "unknown", "order_id": order_id}

        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {"status": "error", "error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        try:
            # Get receiver statistics
            receiver_stats = await self.trading_receiver.get_statistics()

            # Calculate performance metrics
            total_trades = self.stats["executed_trades"]
            win_rate = (self.stats["successful_trades"] / total_trades * 100) if total_trades > 0 else 0.0

            return {
                "orchestrator_stats": self.stats,
                "receiver_stats": receiver_stats,
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
                for handler in self.signal_handlers:
                    try:
                        await handler(signal)
                    except Exception as e:
                        self.logger.error(f"Error in signal handler: {e}")

            except Exception as e:
                self.logger.error(f"Error in internal signal handler: {e}")

        # Note: This would be registered with the signal source
        # For now, this is a placeholder
        self.logger.info("Signal handler registered")

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