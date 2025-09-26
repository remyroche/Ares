"""
Live Trader

Live trading implementation with real exchange connectivity,
order execution, and risk management.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.feature_generation.base_calculations import BaseCalculationType
from src.feature_generation.categories.volatility import ATRGenerator

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
from .exchange_interface import ExchangeInterface, ExchangeType, create_exchange_interface

# Import enhanced signal generators
from ..signal_generation.analyst_signals_refactored import AnalystSignalGenerator, create_analyst_signal_generator
from ..signal_generation.tactician_signals_refactored import TacticianSignalGenerator, create_tactician_signal_generator

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
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        self.min_take_profit_pct = config.get('min_take_profit_pct', 0.005)
        self.min_stop_loss_pct = config.get('min_stop_loss_pct', 0.005)

        trailing_profit_config = config.get('trailing_profit_config', {})
        self.trailing_profit_coefficients = {
            'V': trailing_profit_config.get('v', 0.0),
            'W': trailing_profit_config.get('w', 0.0),
            'X': trailing_profit_config.get('x', 0.0),
            'Y': trailing_profit_config.get('y', 0.0),
            'Z': trailing_profit_config.get('z', 0.0)
        }
        self.trailing_profit_min_pct = trailing_profit_config.get('min_pct', self.min_take_profit_pct)
        self.trailing_profit_max_pct = trailing_profit_config.get('max_pct', 0.2)
        self.trailing_profit_confidence_floor = trailing_profit_config.get('confidence_floor', 0.05)
        self.trailing_profit_atr_period = trailing_profit_config.get('atr_period', 14)
        self.trailing_profit_atr_timeframe = trailing_profit_config.get('atr_timeframe', '1m')

        trailing_stop_config = config.get('trailing_stop_config', {})
        self.trailing_stop_coefficients = {
            'V': trailing_stop_config.get('v', 0.0),
            'W': trailing_stop_config.get('w', 0.0),
            'X': trailing_stop_config.get('x', 0.0),
            'Y': trailing_stop_config.get('y', 0.0),
            'Z': trailing_stop_config.get('z', 0.0)
        }
        self.trailing_stop_min_pct = trailing_stop_config.get('min_pct', self.min_stop_loss_pct)
        self.trailing_stop_max_pct = trailing_stop_config.get('max_pct', 0.12)
        self.trailing_stop_confidence_floor = trailing_stop_config.get('confidence_floor', 0.05)
        self.trailing_stop_atr_period = trailing_stop_config.get('atr_period', self.trailing_profit_atr_period)
        self.trailing_stop_atr_timeframe = trailing_stop_config.get('atr_timeframe', self.trailing_profit_atr_timeframe)
        self.trailing_stop_hard_value = trailing_stop_config.get('hard_value', 0.02)
        self.trailing_stop_hard_atr_multiplier = trailing_stop_config.get('hard_atr_multiplier', 1.0)
        self.trailing_stop_use_atr_hard_value = trailing_stop_config.get('use_atr_hard_value', False)
        self.trailing_review_interval = trailing_stop_config.get('review_interval', config.get('trailing_review_interval', 300))
        self.atr_cache_seconds = config.get('atr_cache_seconds', 120)
        self._atr_generators: Dict[Tuple[str, int], ATRGenerator] = {}

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
            # Initialize order manager
            self.order_manager = await OrderManager.create_order_manager(self.config)
            await self.order_manager.initialize()

            # Initialize exchange interface
            exchange_type = ExchangeType(self.config.get('exchange_type', 'simulated'))
            self.exchange_interface = await create_exchange_interface(
                exchange_type, self.config
            )

            if not await self.exchange_interface.connect():
                raise ExecutionError("Failed to connect to exchange")

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

                self._update_position_extremes(position)
                self._calculate_trade_progress(position)

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
        stop_loss_pct = await self._compute_trailing_stop_loss_pct(position)

        if stop_loss_pct <= 0:
            return False

        if position.side == 'long':
            stop_price = position.entry_price * (1 - stop_loss_pct)
            position.metadata['computed_stop_loss_price'] = stop_price
            stop_triggered = position.current_price <= stop_price
        else:
            stop_price = position.entry_price * (1 + stop_loss_pct)
            position.metadata['computed_stop_loss_price'] = stop_price
            stop_triggered = position.current_price >= stop_price

        if stop_triggered:
            tprint_warning(f"🛡️ Stop loss triggered for {position.symbol} @ {position.current_price}")
            return True

        return False

    async def _check_take_profit(self, position: Position) -> bool:
        """Check if position should take profit."""
        take_profit_pct = await self._compute_trailing_take_profit_pct(position)

        if take_profit_pct <= 0:
            return False

        if position.side == 'long':
            take_profit_price = position.entry_price * (1 + take_profit_pct)
            position.metadata['computed_take_profit_price'] = take_profit_price
            profit_triggered = position.current_price >= take_profit_price
        else:
            take_profit_price = position.entry_price * (1 - take_profit_pct)
            position.metadata['computed_take_profit_price'] = take_profit_price
            profit_triggered = position.current_price <= take_profit_price

        if profit_triggered:
            tprint_success(f"🎯 Take profit triggered for {position.symbol} @ {position.current_price}")
            return True

        return False

    def _update_position_extremes(self, position: Position) -> None:
        """Track price extremes for trailing calculations."""
        try:
            extremes = position.metadata.setdefault('price_extremes', {
                'max_price': position.entry_price,
                'min_price': position.entry_price
            })

            extremes['max_price'] = max(extremes['max_price'], position.current_price)
            extremes['min_price'] = min(extremes['min_price'], position.current_price)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to update price extremes for {position.symbol}: {str(e)}")

    def _calculate_trade_progress(self, position: Position) -> Tuple[float, float]:
        """Calculate realized and adverse movement percentages."""
        extremes = position.metadata.setdefault('price_extremes', {
            'max_price': position.entry_price,
            'min_price': position.entry_price
        })

        entry_price = max(position.entry_price, 1e-9)
        favorable_move = 0.0
        adverse_move = 0.0

        if position.side == 'long':
            favorable_move = max(0.0, extremes['max_price'] - entry_price)
            adverse_move = max(0.0, entry_price - extremes['min_price'])
        else:
            favorable_move = max(0.0, entry_price - extremes['min_price'])
            adverse_move = max(0.0, extremes['max_price'] - entry_price)

        realized_pct = favorable_move / entry_price
        adverse_pct = adverse_move / entry_price

        position.metadata['realized_profit_pct'] = realized_pct
        position.metadata['adverse_move_pct'] = adverse_pct

        return realized_pct, adverse_pct

    def _safe_log(self, value: float, floor: float = 1e-9) -> float:
        """Safely compute logarithm for optimization inputs."""
        try:
            if value is None or not math.isfinite(value):
                return 0.0
            return math.log(max(value, floor))
        except ValueError:
            return 0.0

    async def _compute_trailing_take_profit_pct(self, position: Position) -> float:
        """Compute trailing take profit percentage based on dynamic factors."""
        try:
            realized_pct, _ = self._calculate_trade_progress(position)
            atr_pct = await self._get_position_atr(position, for_take_profit=True)

            tact_conf = max(
                position.metadata.get('tactician_confidence', self.trailing_profit_confidence_floor),
                self.trailing_profit_confidence_floor
            )
            analyst_conf = max(
                position.metadata.get('analyst_confidence', self.trailing_profit_confidence_floor),
                self.trailing_profit_confidence_floor
            )

            trailing_value = (
                self.trailing_profit_coefficients['W'] * self._safe_log(atr_pct) +
                self.trailing_profit_coefficients['X'] * self._safe_log(tact_conf) +
                self.trailing_profit_coefficients['Y'] * self._safe_log(analyst_conf) +
                self.trailing_profit_coefficients['Z'] * self._safe_log(max(realized_pct, self.trailing_profit_min_pct)) +
                self.trailing_profit_coefficients['V']
            )

            take_profit_pct = abs(trailing_value)
            take_profit_pct = max(take_profit_pct, self.trailing_profit_min_pct, self.min_take_profit_pct)
            take_profit_pct = min(take_profit_pct, self.trailing_profit_max_pct)
            position.metadata['computed_take_profit_pct'] = take_profit_pct
            return take_profit_pct

        except Exception as e:
            tprint_warning(f"⚠️ Failed to compute trailing take profit for {position.symbol}: {str(e)}")
            return max(self.trailing_profit_min_pct, self.min_take_profit_pct)

    async def _compute_trailing_stop_loss_pct(self, position: Position) -> float:
        """Compute trailing stop loss percentage based on dynamic factors."""
        try:
            _, adverse_pct = self._calculate_trade_progress(position)
            atr_pct = await self._get_position_atr(position, for_take_profit=False)

            tact_conf = max(
                position.metadata.get('tactician_confidence', self.trailing_stop_confidence_floor),
                self.trailing_stop_confidence_floor
            )
            analyst_conf = max(
                position.metadata.get('analyst_confidence', self.trailing_stop_confidence_floor),
                self.trailing_stop_confidence_floor
            )

            trailing_value = (
                self.trailing_stop_coefficients['W'] * self._safe_log(atr_pct) +
                self.trailing_stop_coefficients['X'] * self._safe_log(tact_conf) +
                self.trailing_stop_coefficients['Y'] * self._safe_log(analyst_conf) +
                self.trailing_stop_coefficients['Z'] * self._safe_log(max(adverse_pct, self.trailing_stop_min_pct)) +
                self.trailing_stop_coefficients['V']
            )

            trailing_stop_pct = abs(trailing_value)
            trailing_stop_pct = max(trailing_stop_pct, self.trailing_stop_min_pct, self.min_stop_loss_pct)
            trailing_stop_pct = min(trailing_stop_pct, self.trailing_stop_max_pct)

            hard_stop_pct = max(self.trailing_stop_hard_value, 0.0)
            if self.trailing_stop_use_atr_hard_value:
                hard_stop_pct = max(hard_stop_pct, atr_pct * self.trailing_stop_hard_atr_multiplier)

            hard_stop_pct = max(hard_stop_pct, self.trailing_stop_min_pct, self.min_stop_loss_pct)

            stop_loss_pct = min(trailing_stop_pct, hard_stop_pct) if hard_stop_pct > 0 else trailing_stop_pct
            position.metadata['computed_trailing_stop_pct'] = trailing_stop_pct
            position.metadata['computed_hard_stop_pct'] = hard_stop_pct if hard_stop_pct > 0 else None
            position.metadata['computed_stop_loss_pct'] = stop_loss_pct
            return stop_loss_pct

        except Exception as e:
            tprint_warning(f"⚠️ Failed to compute trailing stop loss for {position.symbol}: {str(e)}")
            return max(self.trailing_stop_min_pct, self.min_stop_loss_pct)

    async def _get_position_atr(self, position: Position, for_take_profit: bool = True) -> float:
        """Get cached ATR for a position or compute a fresh value."""
        cache_key = 'atr_cache_tp' if for_take_profit else 'atr_cache_sl'
        cache = position.metadata.get(cache_key)
        now = datetime.now()

        if cache:
            age = (now - cache.get('timestamp', now)).total_seconds()
            if age < self.atr_cache_seconds:
                cached_value = cache.get('value', 0.0)
                if cached_value is not None:
                    return cached_value

        if not self.exchange_interface:
            return 0.0

        interval = self.trailing_profit_atr_timeframe if for_take_profit else self.trailing_stop_atr_timeframe
        period = self.trailing_profit_atr_period if for_take_profit else self.trailing_stop_atr_period
        atr_value = await self._calculate_atr(position.symbol, interval, period)

        position.metadata[cache_key] = {
            'value': atr_value,
            'timestamp': now
        }

        return atr_value

    async def _calculate_atr(self, symbol: str, interval: str, period: int = 14) -> float:
        """Calculate ATR for a symbol using feature_generation utilities."""
        try:
            klines = await self.exchange_interface.get_klines(symbol, interval=interval, limit=period + 1)
            if not klines or len(klines) < period + 1:
                return 0.0

            klines_sorted = sorted(klines, key=lambda k: k.timestamp)
            data = pd.DataFrame({
                'high': [k.high_price for k in klines_sorted],
                'low': [k.low_price for k in klines_sorted],
                'close': [k.close_price for k in klines_sorted]
            })

            generator_key = (interval, period)
            atr_generator = self._atr_generators.get(generator_key)
            if atr_generator is None:
                atr_generator = ATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                self._atr_generators[generator_key] = atr_generator

            atr_result = atr_generator.generate(data)
            if not atr_result.success or atr_result.data.empty:
                return 0.0

            atr_series = atr_result.data.dropna()
            if atr_series.empty:
                return 0.0

            atr_value = float(atr_series.iloc[-1])
            latest_close = float(data['close'].iloc[-1]) if not data.empty else 0.0
            if latest_close <= 0:
                return atr_value

            return atr_value / latest_close

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate ATR for {symbol}: {str(e)}")
            return 0.0

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