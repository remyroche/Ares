"""
Trade Executor

Handles the execution of trading decisions and position management.
Provides risk management and execution optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult
from ..exchange.base_exchange import BaseExchange
from .trading_manager import TradingConfig


@dataclass
class ExecutionResult:
    """Result of trade execution"""
    success: bool
    order_id: Optional[str]
    executed_quantity: float
    execution_price: float
    fees: float
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


class TradeExecutor:
    """
    Handles trade execution and position management.

    This class is responsible for:
    - Executing trade decisions
    - Managing positions
    - Risk management during execution
    - Order optimization
    """

    def __init__(self, exchange: BaseExchange, config: TradingConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Position tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.execution_history: List[ExecutionResult] = []

        # Risk management
        self.max_slippage = 0.02  # 2% maximum slippage
        self.max_execution_time = 30  # seconds
        self.min_order_size = 10  # Minimum order size in USD

        # Event callbacks
        self.on_position_update: Optional[Callable] = None
        self.on_execution_complete: Optional[Callable] = None

        # Background tasks
        self.position_monitor_task = None
        self.is_running = False

    async def start(self) -> None:
        """Start the trade executor."""
        try:
            if self.is_running:
                self.logger.warning("TradeExecutor is already running")
                return

            self.logger.info("Starting TradeExecutor...")

            # Load existing positions
            await self._load_positions()

            # Start position monitoring
            self.position_monitor_task = asyncio.create_task(self._monitor_positions())

            self.is_running = True
            self.logger.info("✅ TradeExecutor started successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to start TradeExecutor: {e}")
            raise

    async def stop(self) -> None:
        """Stop the trade executor."""
        try:
            self.logger.info("Stopping TradeExecutor...")

            self.is_running = False

            if self.position_monitor_task:
                self.position_monitor_task.cancel()
                try:
                    await self.position_monitor_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("✅ TradeExecutor stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping TradeExecutor: {e}")

    async def execute_trade(self, trade_decision: TradeDecision) -> Optional[ExecutionResult]:
        """
        Execute a trade decision.

        Args:
            trade_decision: The trade decision to execute

        Returns:
            ExecutionResult object if successful, None otherwise
        """
        try:
            self.logger.info(f"Executing trade: {trade_decision.symbol} {trade_decision.action} "
                           f"{trade_decision.quantity} @ {trade_decision.price}")

            # Pre-execution validation
            validation_result = await self._validate_execution(trade_decision)
            if not validation_result['valid']:
                self.logger.error(f"Execution validation failed: {validation_result['message']}")
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    executed_quantity=0,
                    execution_price=0,
                    fees=0,
                    timestamp=datetime.now(),
                    error_message=validation_result['message']
                )

            # Optimize execution
            optimized_decision = await self._optimize_execution(trade_decision)

            # Execute the order
            execution_result = await self._execute_order(optimized_decision)

            if execution_result.success:
                # Update position tracking
                await self._update_position_tracking(optimized_decision, execution_result)

                # Record execution
                self.execution_history.append(execution_result)

                # Notify callback
                if self.on_execution_complete:
                    await self.on_execution_complete(execution_result)

                self.logger.info(f"✅ Trade executed successfully: {execution_result.order_id}")
            else:
                self.logger.error(f"❌ Trade execution failed: {execution_result.error_message}")

            return execution_result

        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                executed_quantity=0,
                execution_price=0,
                fees=0,
                timestamp=datetime.now(),
                error_message=str(e)
            )

    async def close_position(self, symbol: str, quantity: Optional[float] = None) -> Optional[ExecutionResult]:
        """
        Close a position for a symbol.

        Args:
            symbol: Trading symbol
            quantity: Quantity to close (None for all)

        Returns:
            ExecutionResult object if successful, None otherwise
        """
        try:
            # Get current position
            position = self.positions.get(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return None

            # Determine quantity to close
            close_quantity = quantity or abs(position.quantity)

            # Determine side (opposite of current position)
            side = "SELL" if position.quantity > 0 else "BUY"

            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                self.logger.error(f"Could not get current price for {symbol}")
                return None

            # Create trade decision
            trade_decision = TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action=side,
                quantity=close_quantity,
                price=current_price,
                leverage=1.0,
                stop_loss=0,
                take_profit=0,
                confidence=1.0,
                risk_score=0.1
            )

            return await self.execute_trade(trade_decision)

        except Exception as e:
            self.logger.error(f"❌ Error closing position for {symbol}: {e}")
            return None

    async def get_position_info(self, symbol: str) -> Optional[PositionInfo]:
        """Get position information for a symbol."""
        return self.positions.get(symbol)

    async def get_all_positions(self) -> Dict[str, PositionInfo]:
        """Get all current positions."""
        return self.positions.copy()

    async def get_position_value(self, symbol: str) -> float:
        """Get the current value of a position."""
        position = self.positions.get(symbol)
        if not position:
            return 0.0

        return abs(position.quantity) * position.current_price

    async def get_total_portfolio_value(self) -> float:
        """Get total portfolio value."""
        total_value = 0.0
        for position in self.positions.values():
            total_value += abs(position.quantity) * position.current_price
        return total_value

    async def _validate_execution(self, trade_decision: TradeDecision) -> Dict[str, Any]:
        """
        Validate trade execution.

        Args:
            trade_decision: The trade decision to validate

        Returns:
            Validation result dictionary
        """
        try:
            # Check minimum order size
            order_value = trade_decision.quantity * trade_decision.price
            if order_value < self.min_order_size:
                return {
                    'valid': False,
                    'message': f'Order value {order_value} below minimum {self.min_order_size}'
                }

            # Check position limits
            current_position_value = await self.get_total_portfolio_value()
            if current_position_value + order_value > self.config.max_position_size:
                return {
                    'valid': False,
                    'message': f'Position would exceed maximum size limit'
                }

            # Check if symbol is allowed
            if trade_decision.symbol not in self.config.symbols:
                return {
                    'valid': False,
                    'message': f'Symbol {trade_decision.symbol} not in allowed symbols'
                }

            return {'valid': True, 'message': 'Execution is valid'}

        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {str(e)}'}

    async def _optimize_execution(self, trade_decision: TradeDecision) -> TradeDecision:
        """
        Optimize trade execution for better results.

        Args:
            trade_decision: The trade decision to optimize

        Returns:
            Optimized trade decision
        """
        try:
            # Get current market price
            current_price = await self._get_current_price(trade_decision.symbol)
            if not current_price:
                return trade_decision

            # Check for slippage
            if trade_decision.price > 0:
                slippage = abs(current_price - trade_decision.price) / trade_decision.price
                if slippage > self.max_slippage:
                    self.logger.warning(f"High slippage detected: {slippage".2%"}")

            # Adjust price for market orders
            if trade_decision.price <= 0:  # Market order
                optimized_decision = TradeDecision(
                    timestamp=trade_decision.timestamp,
                    symbol=trade_decision.symbol,
                    action=trade_decision.action,
                    quantity=trade_decision.quantity,
                    price=current_price,
                    leverage=trade_decision.leverage,
                    stop_loss=trade_decision.stop_loss,
                    take_profit=trade_decision.take_profit,
                    confidence=trade_decision.confidence,
                    risk_score=trade_decision.risk_score
                )
            else:
                # Use provided price for limit orders
                optimized_decision = trade_decision

            return optimized_decision

        except Exception as e:
            self.logger.error(f"❌ Error optimizing execution: {e}")
            return trade_decision

    async def _execute_order(self, trade_decision: TradeDecision) -> ExecutionResult:
        """
        Execute the order on the exchange.

        Args:
            trade_decision: The trade decision to execute

        Returns:
            ExecutionResult object
        """
        try:
            start_time = datetime.now()

            # Create order on exchange
            order_result = await self.exchange.create_order(
                symbol=trade_decision.symbol,
                side=trade_decision.action.lower(),
                quantity=trade_decision.quantity,
                price=trade_decision.price,
                order_type="MARKET" if trade_decision.price <= 0 else "LIMIT"
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            if execution_time > self.max_execution_time:
                self.logger.warning(f"Order execution took {execution_time".2f"}s")

            if order_result:
                return ExecutionResult(
                    success=True,
                    order_id=order_result.get('orderId'),
                    executed_quantity=float(order_result.get('executedQty', trade_decision.quantity)),
                    execution_price=float(order_result.get('avgPrice', trade_decision.price)),
                    fees=float(order_result.get('fee', 0)),
                    timestamp=datetime.now()
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    executed_quantity=0,
                    execution_price=0,
                    fees=0,
                    timestamp=datetime.now(),
                    error_message="Order creation failed"
                )

        except Exception as e:
            self.logger.error(f"❌ Error executing order: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                executed_quantity=0,
                execution_price=0,
                fees=0,
                timestamp=datetime.now(),
                error_message=str(e)
            )

    async def _update_position_tracking(self, trade_decision: TradeDecision, execution_result: ExecutionResult):
        """
        Update position tracking after execution.

        Args:
            trade_decision: The trade decision
            execution_result: The execution result
        """
        try:
            symbol = trade_decision.symbol
            quantity = execution_result.executed_quantity
            price = execution_result.execution_price

            # Get current position
            current_position = self.positions.get(symbol)

            if current_position:
                # Update existing position
                if trade_decision.action == "BUY":
                    # Adding to long position
                    new_quantity = current_position.quantity + quantity
                    new_entry_price = (current_position.entry_price * current_position.quantity + price * quantity) / new_quantity

                    current_position.quantity = new_quantity
                    current_position.entry_price = new_entry_price
                else:
                    # Reducing long position or increasing short
                    new_quantity = current_position.quantity - quantity
                    current_position.quantity = new_quantity

                    # Close position if quantity reaches zero
                    if abs(new_quantity) < 1e-8:
                        del self.positions[symbol]
                        return

                current_position.current_price = price
                current_position.timestamp = datetime.now()

                # Calculate unrealized PnL
                if current_position.quantity > 0:
                    current_position.unrealized_pnl = (price - current_position.entry_price) * current_position.quantity
                else:
                    current_position.unrealized_pnl = (current_position.entry_price - price) * abs(current_position.quantity)

            else:
                # Create new position
                new_position = PositionInfo(
                    symbol=symbol,
                    quantity=quantity if trade_decision.action == "BUY" else -quantity,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    timestamp=datetime.now()
                )
                self.positions[symbol] = new_position

            # Notify callback
            if self.on_position_update:
                await self.on_position_update({
                    'symbol': symbol,
                    'quantity': self.positions[symbol].quantity,
                    'entry_price': self.positions[symbol].entry_price,
                    'current_price': self.positions[symbol].current_price,
                    'unrealized_pnl': self.positions[symbol].unrealized_pnl
                })

        except Exception as e:
            self.logger.error(f"❌ Error updating position tracking: {e}")

    async def _load_positions(self) -> None:
        """Load existing positions from exchange."""
        try:
            for symbol in self.config.symbols:
                try:
                    position_risk = await self.exchange.get_position_risk(symbol)
                    if position_risk and float(position_risk.get('positionAmt', 0)) != 0:
                        current_price = await self._get_current_price(symbol)
                        if current_price:
                            position_info = PositionInfo(
                                symbol=symbol,
                                quantity=float(position_risk.get('positionAmt', 0)),
                                entry_price=float(position_risk.get('entryPrice', current_price)),
                                current_price=current_price,
                                unrealized_pnl=float(position_risk.get('unRealizedProfit', 0)),
                                realized_pnl=0,
                                timestamp=datetime.now()
                            )
                            self.positions[symbol] = position_info
                except Exception as e:
                    self.logger.debug(f"Could not load position for {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"❌ Error loading positions: {e}")

    async def _monitor_positions(self) -> None:
        """Background task to monitor positions."""
        while self.is_running:
            try:
                # Update position prices and PnL
                for symbol, position in list(self.positions.items()):
                    try:
                        current_price = await self._get_current_price(symbol)
                        if current_price:
                            position.current_price = current_price

                            # Update unrealized PnL
                            if position.quantity > 0:
                                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                            else:
                                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)

                            position.timestamp = datetime.now()
                    except Exception as e:
                        self.logger.debug(f"Error updating position for {symbol}: {e}")

                await asyncio.sleep(5)  # Update every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in position monitoring: {e}")
                await asyncio.sleep(10)

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if not available
        """
        try:
            # Try to get price from exchange
            price = await self.exchange.fetch_price(symbol)
            return price

        except Exception as e:
            self.logger.debug(f"Could not get price for {symbol}: {e}")
            return None

    # Configuration methods
    def set_position_update_callback(self, callback: Callable):
        """Set callback for position updates."""
        self.on_position_update = callback

    def set_execution_complete_callback(self, callback: Callable):
        """Set callback for execution completion."""
        self.on_execution_complete = callback

    def set_max_slippage(self, slippage: float):
        """Set maximum allowed slippage."""
        self.max_slippage = slippage

    def set_max_execution_time(self, time_seconds: int):
        """Set maximum execution time."""
        self.max_execution_time = time_seconds

    def set_min_order_size(self, min_size: float):
        """Set minimum order size."""
        self.min_order_size = min_size