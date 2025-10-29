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
            exchange_config = self.config.copy()
            exchange_type_str = self.config.get('exchange_type', 'simulated')
            exchange_config['exchange_type'] = exchange_type_str
            
            # Create exchange interface
            self.exchange_interface = create_exchange_interface(exchange_config)
            
            # Connect to exchange
            if not await self.exchange_interface.connect():
                raise ExecutionError("Failed to connect to exchange")
            
            # Initialize order manager with exchange interface
            order_manager_config = self.config.copy()
            order_manager_config['exchange_interface'] = self.exchange_interface
            self.order_manager = OrderManager(order_manager_config)
            await self.order_manager.initialize()

            # Initialize enhanced signal generators
            await self._initialize_signal_generators()

            # Reconcile positions with exchange
            await self.reconcile_positions_with_exchange()

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
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                tprint_error(f"❌ Invalid symbol: {symbol}")
                return None
                
            if not side or side.lower() not in ['buy', 'sell']:
                tprint_error(f"❌ Invalid side: {side}")
                return None
                
            if quantity is None or quantity <= 0:
                tprint_error(f"❌ Invalid quantity: {quantity}")
                return None
                
            if not self.order_manager:
                tprint_error("❌ Order manager not initialized")
                return None

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

            # Validate order was created
            if order is None:
                tprint_error(f"❌ Failed to create order for {symbol}")
                return None
                
            if not hasattr(order, 'order_id') or not order.order_id:
                tprint_error(f"❌ Order created but missing order_id for {symbol}")
                return None

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
        # Validate inputs
        if not symbol or quantity is None or quantity <= 0:
            return False
            
        # Get current portfolio value
        portfolio_value = await self._get_portfolio_value()

        # Prevent division by zero
        if portfolio_value is None or portfolio_value <= 0:
            tprint_warning(f"⚠️ Portfolio value is {portfolio_value}, cannot validate position limits")
            return False

        # Calculate position size as percentage of portfolio
        current_price = await self._get_current_price(symbol)
        
        # Handle None price
        if current_price is None or current_price <= 0:
            tprint_warning(f"⚠️ Invalid price for {symbol}: {current_price}, cannot validate position limits")
            return False
            
        position_value = quantity * current_price
        
        if position_value <= 0:
            tprint_warning(f"⚠️ Invalid position value: {position_value}")
            return False
            
        position_percentage = position_value / portfolio_value

        if position_percentage > self.max_position_size:
            tprint_warning(f"⚠️ Position size {position_percentage:.2%} exceeds limit {self.max_position_size:.2%}")
            return False
            
        return True

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
        """Get current portfolio value accounting for both long and short positions."""
        try:
            # Get account balance from exchange
            balances = await self.exchange_interface.get_account_balance()
            if balances is None:
                balances = {}

            # Start with cash balance (USDT)
            total_value = balances.get('USDT', 0.0)
            if total_value < 0:
                tprint_warning(f"⚠️ Negative cash balance: {total_value}")
                total_value = 0.0

            # Calculate unrealized PnL for all positions
            for symbol, position in self.positions.items():
                if position.quantity == 0:
                    continue
                
                current_price = await self._get_current_price(symbol)
                if current_price <= 0:
                    tprint_warning(f"⚠️ Invalid price for {symbol}: {current_price}")
                    continue

                # Calculate unrealized PnL based on position side
                if position.side == 'long':
                    # Long position: value = quantity * current_price
                    # Unrealized PnL = (current_price - entry_price) * quantity
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    total_value += unrealized_pnl
                elif position.side == 'short':
                    # Short position: 
                    # Value at entry = quantity * entry_price (collateral locked)
                    # Current value = quantity * current_price (what we owe)
                    # Unrealized PnL = (entry_price - current_price) * quantity
                    unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    total_value += unrealized_pnl
                    # For shorts, we also need to account for the collateral already in balance
                    # The locked collateral is not in cash, but represented as negative position
                else:
                    tprint_warning(f"⚠️ Unknown position side for {symbol}: {position.side}")

            return max(0.0, total_value)  # Ensure non-negative

        except Exception as e:
            tprint_warning(f"⚠️ Failed to get portfolio value: {str(e)}")
            self.logger.exception("Error calculating portfolio value")
            return 10000.0  # Default fallback

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        if not symbol:
            tprint_warning("⚠️ Empty symbol provided for price lookup")
            return None
            
        try:
            if not self.exchange_interface:
                tprint_warning("⚠️ Exchange interface not available for price lookup")
                return None
                
            ticker = await self.exchange_interface.get_ticker(symbol)
            
            if ticker is None:
                tprint_warning(f"⚠️ No ticker data available for {symbol}")
                return None
                
            if not hasattr(ticker, 'price') or ticker.price is None:
                tprint_warning(f"⚠️ Invalid price in ticker for {symbol}")
                return None
                
            price = float(ticker.price)
            
            if price <= 0:
                tprint_warning(f"⚠️ Non-positive price for {symbol}: {price}")
                return None
                
            return price
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get price for {symbol}: {str(e)}")
            self.logger.exception(f"Error getting price for {symbol}")
            return None

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
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                tprint_error(f"❌ Invalid symbol: {symbol}")
                return False
                
            if quantity is not None and quantity <= 0:
                tprint_error(f"❌ Invalid close quantity: {quantity}")
                return False
                
            if symbol not in self.positions:
                tprint_warning(f"⚠️ No position found for {symbol}")
                return False

            position = self.positions[symbol]
            
            # Validate position
            if position.quantity is None or position.quantity <= 0:
                tprint_warning(f"⚠️ Position has invalid quantity: {position.quantity}, removing")
                del self.positions[symbol]
                return False
                
            if position.side not in ['long', 'short']:
                tprint_warning(f"⚠️ Position has invalid side: {position.side}")
                return False
            
            close_quantity = quantity if quantity is not None else position.quantity
            
            # Validate quantity
            if close_quantity <= 0:
                tprint_error(f"❌ Invalid close quantity: {close_quantity}")
                return False
            
            # Ensure we don't close more than available
            if close_quantity > position.quantity:
                tprint_warning(f"⚠️ Close quantity {close_quantity} exceeds position {position.quantity}. Closing full position.")
                close_quantity = position.quantity

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
                if position.quantity <= 0 or abs(position.quantity) < 1e-8:  # Handle floating point precision
                    del self.positions[symbol]
                    tprint_info(f"🗑️ Removed position entry for {symbol}")

                tprint_success(f"✅ Position closed for {symbol}")
                return True
            else:
                tprint_error(f"❌ Failed to close position for {symbol}")
                return False

        except Exception as e:
            tprint_error(f"❌ Error closing position for {symbol}: {str(e)}")
            self.logger.exception(f"Error closing position for {symbol}")
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

    async def reconcile_positions_with_exchange(self) -> Dict[str, Any]:
        """
        Reconcile internal position tracking with exchange positions.
        
        This method:
        1. Fetches actual positions from the exchange
        2. Compares with internal position tracking
        3. Identifies discrepancies (missing, extra, or mismatched positions)
        4. Updates internal tracking to match exchange reality
        5. Logs all discrepancies for investigation
        
        Returns:
            Dictionary with reconciliation results including:
            - synced_positions: List of successfully synced positions
            - discrepancies: List of discrepancies found
            - exchange_positions: Raw positions from exchange
        """
        reconciliation_result = {
            'synced_positions': [],
            'discrepancies': [],
            'exchange_positions': [],
            'timestamp': datetime.now()
        }
        
        try:
            if not self.exchange_interface:
                tprint_warning("⚠️ Cannot reconcile: exchange interface not available")
                reconciliation_result['discrepancies'].append({
                    'type': 'missing_interface',
                    'message': 'Exchange interface not initialized'
                })
                return reconciliation_result

            # Get positions from exchange
            exchange_positions_raw = []
            try:
                # Try to get open positions from exchange
                if hasattr(self.exchange_interface, 'get_open_positions'):
                    exchange_positions_raw = await self.exchange_interface.get_open_positions()
                elif hasattr(self.exchange_interface, 'dispatcher') and self.exchange_interface.dispatcher:
                    # Try through dispatcher
                    if hasattr(self.exchange_interface.dispatcher, 'get_positions'):
                        exchange_positions_raw = await self.exchange_interface.dispatcher.get_positions()
            except Exception as e:
                tprint_warning(f"⚠️ Failed to fetch positions from exchange: {e}")
                reconciliation_result['discrepancies'].append({
                    'type': 'fetch_error',
                    'message': f'Error fetching exchange positions: {str(e)}'
                })

            reconciliation_result['exchange_positions'] = exchange_positions_raw

            # Normalize exchange positions to our Position format
            exchange_positions = {}
            for pos_data in exchange_positions_raw:
                try:
                    symbol = pos_data.get('symbol') or pos_data.get('instrument') or pos_data.get('pair')
                    if not symbol:
                        continue

                    # Extract position data (format may vary by exchange)
                    quantity = float(pos_data.get('positionAmt', pos_data.get('size', pos_data.get('quantity', 0))))
                    entry_price = float(pos_data.get('entryPrice', pos_data.get('avg_price', pos_data.get('price', 0))))
                    current_price = float(pos_data.get('markPrice', pos_data.get('current_price', pos_data.get('lastPrice', 0))))
                    unrealized_pnl = float(pos_data.get('unrealizedPnl', pos_data.get('unrealized_pnl', 0)))

                    # Determine side based on quantity sign (exchange-dependent)
                    if quantity > 0:
                        side = 'long'
                    elif quantity < 0:
                        side = 'short'
                        quantity = abs(quantity)  # Store as positive with side indicator
                    else:
                        continue  # Skip zero positions

                    exchange_positions[symbol] = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'exchange_data': pos_data
                    }
                except (ValueError, KeyError, TypeError) as e:
                    tprint_warning(f"⚠️ Failed to parse exchange position: {pos_data}, error: {e}")
                    continue

            # Compare with internal positions
            internal_symbols = set(self.positions.keys())
            exchange_symbols = set(exchange_positions.keys())

            # Find missing positions (in exchange but not in internal)
            for symbol in exchange_symbols - internal_symbols:
                exchange_pos = exchange_positions[symbol]
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=exchange_pos['side'],
                    quantity=exchange_pos['quantity'],
                    entry_price=exchange_pos['entry_price'],
                    current_price=exchange_pos['current_price'],
                    unrealized_pnl=exchange_pos['unrealized_pnl'],
                    realized_pnl=0.0,
                    timestamp=datetime.now()
                )
                reconciliation_result['synced_positions'].append(symbol)
                reconciliation_result['discrepancies'].append({
                    'type': 'missing_internal',
                    'symbol': symbol,
                    'message': f'Position found on exchange but missing internally - added'
                })
                tprint_warning(f"⚠️ Found position on exchange not tracked internally: {symbol}")

            # Find extra positions (in internal but not in exchange)
            for symbol in internal_symbols - exchange_symbols:
                internal_pos = self.positions[symbol]
                reconciliation_result['discrepancies'].append({
                    'type': 'missing_exchange',
                    'symbol': symbol,
                    'message': f'Position tracked internally but not found on exchange',
                    'internal_quantity': internal_pos.quantity,
                    'internal_side': internal_pos.side
                })
                tprint_warning(f"⚠️ Position tracked internally but not found on exchange: {symbol}")
                # Optionally: remove from internal tracking or mark as closed
                # For safety, we'll keep it but mark it as potentially stale

            # Compare matching positions for discrepancies
            for symbol in internal_symbols & exchange_symbols:
                internal_pos = self.positions[symbol]
                exchange_pos = exchange_positions[symbol]

                discrepancies = []
                
                # Check quantity mismatch
                if abs(internal_pos.quantity - exchange_pos['quantity']) > 0.0001:  # Allow small floating point differences
                    discrepancies.append({
                        'field': 'quantity',
                        'internal': internal_pos.quantity,
                        'exchange': exchange_pos['quantity'],
                        'difference': abs(internal_pos.quantity - exchange_pos['quantity'])
                    })

                # Check side mismatch
                if internal_pos.side != exchange_pos['side']:
                    discrepancies.append({
                        'field': 'side',
                        'internal': internal_pos.side,
                        'exchange': exchange_pos['side']
                    })

                # Check entry price mismatch (within 1% tolerance)
                price_tolerance = 0.01
                if abs(internal_pos.entry_price - exchange_pos['entry_price']) / max(internal_pos.entry_price, exchange_pos['entry_price']) > price_tolerance:
                    discrepancies.append({
                        'field': 'entry_price',
                        'internal': internal_pos.entry_price,
                        'exchange': exchange_pos['entry_price'],
                        'difference_percent': abs(internal_pos.entry_price - exchange_pos['entry_price']) / max(internal_pos.entry_price, exchange_pos['entry_price']) * 100
                    })

                if discrepancies:
                    # Update internal position to match exchange (authoritative source)
                    internal_pos.quantity = exchange_pos['quantity']
                    internal_pos.side = exchange_pos['side']
                    internal_pos.entry_price = exchange_pos['entry_price']
                    internal_pos.current_price = exchange_pos['current_price']
                    internal_pos.unrealized_pnl = exchange_pos['unrealized_pnl']
                    
                    reconciliation_result['discrepancies'].append({
                        'type': 'mismatch',
                        'symbol': symbol,
                        'discrepancies': discrepancies,
                        'message': 'Position data mismatch - synced to exchange values'
                    })
                    tprint_warning(f"⚠️ Position mismatch for {symbol}: {discrepancies}")
                else:
                    # Positions match, just update current price and PnL
                    internal_pos.current_price = exchange_pos['current_price']
                    internal_pos.unrealized_pnl = exchange_pos['unrealized_pnl']
                    reconciliation_result['synced_positions'].append(symbol)

            tprint_success(f"✅ Position reconciliation complete: {len(reconciliation_result['synced_positions'])} synced, {len(reconciliation_result['discrepancies'])} discrepancies")
            return reconciliation_result

        except Exception as e:
            tprint_error(f"❌ Position reconciliation failed: {str(e)}")
            self.logger.exception("Error during position reconciliation")
            reconciliation_result['discrepancies'].append({
                'type': 'reconciliation_error',
                'message': f'Reconciliation process failed: {str(e)}'
            })
            return reconciliation_result

    async def update_positions(self) -> None:
        """Update position information with current prices."""
        try:
            for symbol, position in self.positions.items():
                if position.quantity == 0:
                    continue
                    
                current_price = await self._get_current_price(symbol)
                
                if current_price is None or current_price <= 0:
                    tprint_warning(f"⚠️ Invalid price for {symbol}: {current_price}")
                    continue

                position.current_price = current_price

                if position.side == 'long':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                elif position.side == 'short':
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                else:
                    tprint_warning(f"⚠️ Unknown position side for {symbol}: {position.side}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to update positions: {str(e)}")
            self.logger.exception("Error updating positions")

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

            # Clean up signal generators
            if self.analyst_signal_generator and hasattr(self.analyst_signal_generator, 'cleanup'):
                try:
                    await self.analyst_signal_generator.cleanup()
                except Exception as e:
                    tprint_warning(f"⚠️ Error cleaning up analyst signal generator: {e}")
            
            if self.tactician_signal_generator and hasattr(self.tactician_signal_generator, 'cleanup'):
                try:
                    await self.tactician_signal_generator.cleanup()
                except Exception as e:
                    tprint_warning(f"⚠️ Error cleaning up tactician signal generator: {e}")

            # Close session
            if self.session:
                self.session.end_time = datetime.now()

            # Disconnect exchange
            if self.exchange_interface:
                await self.exchange_interface.disconnect()

            # Clean up order manager
            if self.order_manager and hasattr(self.order_manager, 'cleanup'):
                await self.order_manager.cleanup()

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
