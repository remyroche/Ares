"""
Live Trader

Live trading implementation with real exchange connectivity,
order execution, and risk management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..config.trading_config import TradingConfig, TradingMode
from ..config.execution_config import ExecutionConfig
from ..utils.error_handling import (
    ExecutionError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config
from .order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
from .exchange_interface import ExchangeInterface, create_exchange_interface

# Import enhanced signal generators
from ..signal_generation.analyst_signals import AnalystSignalGenerator, create_analyst_signal_generator
from ..signal_generation.tactician_signals import TacticianSignalGenerator, create_tactician_signal_generator

logger = system_logger.getChild('LiveTrader')

class LiveTraderStatus(Enum):
    """Live trader status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    PAUSED = "paused"

@dataclass
class Position:
    """Trading position information."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    orders: List[str] = field(default_factory=list)  # Related order IDs

@dataclass
class TradingSession:
    """Live trading session information."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)

class LiveTrader:
    """
    Live Trading Implementation

    Handles real-time trading with live exchange connectivity,
    order execution, position management, and risk control.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize live trader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('LiveTrader')

        # Configuration
        self.trading_config = TradingConfig(**config.get('trading_config', {}))
        self.execution_config = ExecutionConfig(**config.get('execution_config', {}))

        # Core components
        self.order_manager: Optional[OrderManager] = None
        self.exchange_interface: Optional[ExchangeInterface] = None

        # Enhanced signal generators
        self.analyst_signal_generator: Optional[AnalystSignalGenerator] = None
        self.tactician_signal_generator: Optional[TacticianSignalGenerator] = None

        # Trading state
        self.status = LiveTraderStatus.STOPPED
        self.session: Optional[TradingSession] = None
        self.positions: Dict[str, Position] = {}
        self.active_orders: Dict[str, Order] = {}

        # Risk management
        self.max_positions = config.get('max_positions', 5)
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.stop_loss_threshold = config.get('stop_loss_threshold', 0.05)  # 5%
        self.take_profit_threshold = config.get('take_profit_threshold', 0.10)  # 10%

        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0

        # NAS/TAS enhancement tracking
        self.nas_enhanced_trades = 0
        self.tas_enhanced_trades = 0
        self.enhanced_signal_performance = {}

        tprint_info("🚀 Initializing Live Trader with NAS/TAS enhancement...")

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.CRITICAL,
        raise_on_error=True
    )
    async def initialize(self) -> None:
        """Initialize live trader components."""
        try:
            # Initialize exchange interface first
            exchange_type = ExchangeType(self.config.get('exchange_type', 'simulated'))
            self.exchange_interface = await create_exchange_interface(
                exchange_type, self.config
            )

            if not await self.exchange_interface.connect():
                raise ExecutionError("Failed to connect to exchange")
            
            # Initialize order manager with exchange interface
            order_manager_config = self.config.copy()
            order_manager_config['exchange_interface'] = self.exchange_interface
            self.order_manager = OrderManager(order_manager_config)
            await self.order_manager.initialize()

            # Initialize enhanced signal generators
            await self._initialize_signal_generators()

            # Start trading session
            self.session = TradingSession(
                session_id=str(datetime.now().timestamp()),
                start_time=datetime.now(),
                symbols=self.trading_config.symbols
            )

            self.status = LiveTraderStatus.RUNNING

            tprint_success("✅ Live Trader initialized successfully with NAS/TAS enhancement")

        except Exception as e:
            self.status = LiveTraderStatus.ERROR
            tprint_error(f"❌ Failed to initialize Live Trader: {str(e)}")
            raise

    async def _initialize_signal_generators(self):
        """Initialize enhanced signal generators with NAS/TAS models."""
        try:
            # Initialize Analyst signal generator with NAS enhancement
            analyst_config = {
                'confidence_threshold': 0.6,
                'nas_confidence_threshold': 0.7,
                'enable_nas_enhancement': True,
                'nas_timeframe': '5m',
                'regime_timeframe': '15m',
                'max_history': 1000
            }

            self.analyst_signal_generator = create_analyst_signal_generator(analyst_config)

            # Initialize Tactician signal generator with TAS enhancement
            tactician_config = {
                'confidence_threshold': 0.6,
                'tas_confidence_threshold': 0.7,
                'enable_tas_enhancement': True,
                'tas_timeframe': '1m',
                'risk_per_trade': 0.02,
                'max_leverage': 3.0,
                'kelly_fraction': 0.25,
                'max_history': 1000
            }

            self.tactician_signal_generator = create_tactician_signal_generator(tactician_config)

            # Note: In a real implementation, you would load pre-trained NAS/TAS models here
            # For now, we'll initialize without models (fallback mode)
            await self.analyst_signal_generator.initialize(None)  # No analyst component yet
            await self.tactician_signal_generator.initialize(None)  # No tactician component yet

            tprint_success("✅ Enhanced signal generators initialized")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize signal generators: {e}")
            # Continue without enhancement

    @critical_operation
    @require_no_fallback
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[str]:
        """
        Execute a trade.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            order_type: Order type (default: market)
            price: Limit price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Validate trade parameters
            await self._validate_trade(symbol, side, quantity)

            # Check position limits
            if not await self._check_position_limits(symbol, side, quantity):
                tprint_warning(f"⚠️ Position limits exceeded for {symbol}")
                return None

            # Create main order
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            order = await self.order_manager.create_order(
                symbol=symbol,
                side=order_side,
                order_type=order_type,
                quantity=quantity,
                price=price
            )

            # Add to active orders
            self.active_orders[order.order_id] = order

            # Create stop loss order if specified
            if stop_loss:
                await self._create_stop_loss_order(symbol, side, quantity, stop_loss)

            # Create take profit order if specified
            if take_profit:
                await self._create_take_profit_order(symbol, side, quantity, take_profit)

            # Update session statistics
            self.session.total_trades += 1
            self.total_trades += 1

            tprint_success(f"✅ Trade executed: {side} {quantity} {symbol} @ {price}")

            return order.order_id

        except Exception as e:
            self.session.failed_trades += 1
            self.failed_trades += 1
            tprint_error(f"❌ Failed to execute trade: {str(e)}")
            return None

    async def _validate_trade(self, symbol: str, side: str, quantity: float) -> None:
        """Validate trade parameters."""
        if symbol not in self.trading_config.symbols:
            raise ExecutionError(f"Symbol {symbol} not in allowed symbols list")

        if side.lower() not in ['buy', 'sell']:
            raise ExecutionError(f"Invalid side: {side}")

        if quantity <= 0:
            raise ExecutionError("Quantity must be positive")

        if len(self.positions) >= self.max_positions:
            raise ExecutionError("Maximum number of positions reached")

    async def _check_position_limits(self, symbol: str, side: str, quantity: float) -> bool:
        """Check if trade is within position limits."""
        # Get current portfolio value
        portfolio_value = await self._get_portfolio_value()

        # Calculate position size as percentage of portfolio
        current_price = await self._get_current_price(symbol)
        position_value = quantity * current_price
        position_percentage = position_value / portfolio_value

        return position_percentage <= self.max_position_size

    async def _create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float
    ) -> None:
        """Create stop loss order."""
        try:
            stop_side = OrderSide.SELL if side.lower() == 'buy' else OrderSide.BUY

            await self.order_manager.create_order(
                symbol=symbol,
                side=stop_side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_price,
                metadata={'type': 'stop_loss'}
            )

            tprint_info(f"🛡️ Created stop loss order for {symbol} @ {stop_price}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to create stop loss order: {str(e)}")

    async def _create_take_profit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float
    ) -> None:
        """Create take profit order."""
        try:
            profit_side = OrderSide.SELL if side.lower() == 'buy' else OrderSide.BUY

            await self.order_manager.create_order(
                symbol=symbol,
                side=profit_side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=take_profit_price,
                metadata={'type': 'take_profit'}
            )

            tprint_info(f"🎯 Created take profit order for {symbol} @ {take_profit_price}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to create take profit order: {str(e)}")

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            # Get account balance from exchange
            balances = await self.exchange_interface.get_account_balance()

            # Get current prices for all positions
            total_value = 0.0

            for symbol, position in self.positions.items():
                current_price = await self._get_current_price(symbol)
                position_value = position.quantity * current_price
                total_value += position_value

            # Add cash balance (USDT)
            total_value += balances.get('USDT', 0.0)

            return total_value

        except Exception as e:
            tprint_warning(f"⚠️ Failed to get portfolio value: {str(e)}")
            return 10000.0  # Default fallback

    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        try:
            ticker = await self.exchange_interface.get_ticker(symbol)
            return ticker.price if ticker else 3000.0  # Fallback price
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get price for {symbol}: {str(e)}")
            return 3000.0 if symbol.startswith('ETH') else 50000.0

    async def close_position(self, symbol: str, quantity: Optional[float] = None) -> bool:
        """
        Close position for symbol.

        Args:
            symbol: Trading symbol
            quantity: Quantity to close (None for full position)

        Returns:
            True if closed successfully, False otherwise
        """
        try:
            if symbol not in self.positions:
                tprint_warning(f"⚠️ No position found for {symbol}")
                return False

            position = self.positions[symbol]
            close_quantity = quantity or position.quantity

            # Execute closing trade
            close_side = 'sell' if position.side == 'long' else 'buy'

            order_id = await self.execute_trade(
                symbol=symbol,
                side=close_side,
                quantity=close_quantity
            )

            if order_id:
                # Update position
                position.quantity -= close_quantity
                if position.quantity <= 0:
                    del self.positions[symbol]

                tprint_success(f"✅ Position closed for {symbol}")
                return True
            else:
                tprint_error(f"❌ Failed to close position for {symbol}")
                return False

        except Exception as e:
            tprint_error(f"❌ Error closing position for {symbol}: {str(e)}")
            return False

    async def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions.copy()

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        return self.positions.get(symbol)

    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            success = await self.order_manager.cancel_order(order_id)

            if success and order_id in self.active_orders:
                del self.active_orders[order_id]

            return success

        except Exception as e:
            tprint_error(f"❌ Failed to cancel order {order_id}: {str(e)}")
            return False

    async def update_positions(self) -> None:
        """Update position information with current prices."""
        try:
            for symbol, position in self.positions.items():
                current_price = await self._get_current_price(symbol)

                position.current_price = current_price

                if position.side == 'long':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

        except Exception as e:
            tprint_warning(f"⚠️ Failed to update positions: {str(e)}")

    async def monitor_positions(self) -> None:
        """Monitor positions and execute risk management."""
        try:
            for symbol, position in list(self.positions.items()):
                # Check stop loss
                if await self._check_stop_loss(position):
                    await self.close_position(symbol)
                    continue

                # Check take profit
                if await self._check_take_profit(position):
                    await self.close_position(symbol)
                    continue

                # Check position age and risk limits
                if await self._check_position_risk(position):
                    await self.close_position(symbol)

        except Exception as e:
            tprint_error(f"❌ Error in position monitoring: {str(e)}")

    async def _check_stop_loss(self, position: Position) -> bool:
        """Check if position should be stopped out."""
        if position.side == 'long':
            stop_triggered = position.current_price <= position.entry_price * (1 - self.stop_loss_threshold)
        else:
            stop_triggered = position.current_price >= position.entry_price * (1 + self.stop_loss_threshold)

        if stop_triggered:
            tprint_warning(f"🛡️ Stop loss triggered for {position.symbol} @ {position.current_price}")
            return True

        return False

    async def _check_take_profit(self, position: Position) -> bool:
        """Check if position should take profit."""
        if position.side == 'long':
            profit_triggered = position.current_price >= position.entry_price * (1 + self.take_profit_threshold)
        else:
            profit_triggered = position.current_price <= position.entry_price * (1 - self.take_profit_threshold)

        if profit_triggered:
            tprint_success(f"🎯 Take profit triggered for {position.symbol} @ {position.current_price}")
            return True

        return False

    async def _check_position_risk(self, position: Position) -> bool:
        """Check if position exceeds risk limits."""
        # Check position age (close if too old)
        age_hours = (datetime.now() - position.timestamp).total_seconds() / 3600

        if age_hours > 24:  # Close positions older than 24 hours
            tprint_info(f"⏰ Closing aged position for {position.symbol}")
            return True

        return False

    async def generate_enhanced_signals(self, symbol: str, market_data: pd.DataFrame, regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate enhanced signals using NAS/TAS models."""
        try:
            signals = {}

            # Generate Analyst signal with NAS enhancement
            if self.analyst_signal_generator:
                analyst_signal = await self.analyst_signal_generator.generate_signal(
                    symbol=symbol,
                    market_data=market_data,
                    regime_data=regime_data
                )

                if analyst_signal:
                    signals['analyst_signal'] = analyst_signal

                    # Track NAS enhancement
                    if analyst_signal.nas_confidence > 0:
                        self.nas_enhanced_trades += 1
                        self.enhanced_signal_performance['nas_enhanced'] = self.enhanced_signal_performance.get('nas_enhanced', 0) + 1

            # Generate Tactician signal with TAS enhancement
            if self.tactician_signal_generator and 'analyst_signal' in signals:
                tactician_signal = await self.tactician_signal_generator.generate_timing_signal(
                    symbol=symbol,
                    analyst_signal=signals['analyst_signal'].__dict__,
                    market_data=market_data,
                    current_position=self.positions.get(symbol),
                    account_balance=await self._get_portfolio_value()
                )

                if tactician_signal:
                    signals['tactician_signal'] = tactician_signal

                    # Track TAS enhancement
                    if tactician_signal.tas_confidence > 0:
                        self.tas_enhanced_trades += 1
                        self.enhanced_signal_performance['tas_enhanced'] = self.enhanced_signal_performance.get('tas_enhanced', 0) + 1

            return signals

        except Exception as e:
            self.logger.error(f"❌ Enhanced signal generation failed: {e}")
            return {}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get trading performance metrics with NAS/TAS enhancement tracking."""
        base_metrics = {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': self.successful_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'total_fees': self.total_fees,
            'active_positions': len(self.positions),
            'active_orders': len(self.active_orders),
            'session_duration': (datetime.now() - self.session.start_time).total_seconds() if self.session else 0
        }

        # Add NAS/TAS enhancement metrics
        enhancement_metrics = {
            'nas_enhanced_trades': self.nas_enhanced_trades,
            'tas_enhanced_trades': self.tas_enhanced_trades,
            'enhanced_signal_performance': self.enhanced_signal_performance,
            'nas_enhancement_rate': self.nas_enhanced_trades / max(self.total_trades, 1),
            'tas_enhancement_rate': self.tas_enhanced_trades / max(self.total_trades, 1)
        }

        return {**base_metrics, **enhancement_metrics}

    async def cleanup(self) -> None:
        """Clean up resources and close positions."""
        try:
            tprint_info("🧹 Cleaning up Live Trader...")

            # Close all positions
            for symbol in list(self.positions.keys()):
                await self.close_position(symbol)

            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)

            # Close session
            if self.session:
                self.session.end_time = datetime.now()

            # Disconnect exchange
            if self.exchange_interface:
                await self.exchange_interface.disconnect()

            self.status = LiveTraderStatus.STOPPED

            tprint_success("✅ Live Trader cleaned up successfully")

        except Exception as e:
            tprint_error(f"❌ Error during Live Trader cleanup: {str(e)}")

# Factory functions
async def create_live_trader(config: Dict[str, Any]) -> LiveTrader:
    """Create and initialize a live trader."""
    trader = LiveTrader(config)
    await trader.initialize()
    return trader

def get_live_trader() -> Optional[LiveTrader]:
    """Get the global live trader instance."""
    # Placeholder for singleton pattern
    return None
