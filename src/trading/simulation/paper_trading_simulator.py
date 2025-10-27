#!/usr/bin/env python3
"""
Paper Trading Simulator

Simulates realistic trading execution with:
- Real-time price fetching from exchange
- Slippage calculation based on order size and market conditions
- Fee calculation (maker/taker fees)
- Position tracking (long/short)
- P&L calculation
- Risk management

This simulator provides a realistic trading experience without real money risk.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fees: float = 0.0
    slippage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

@dataclass
class Trade:
    """Trade execution data structure."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fees: float
    slippage: float
    timestamp: datetime = field(default_factory=datetime.now)

class PaperTradingSimulator:
    """
    Paper trading simulator that provides realistic trading simulation.
    
    Features:
    - Real-time price fetching from exchange interface
    - Slippage calculation based on order size and market volatility
    - Fee calculation (maker/taker fees)
    - Position tracking with P&L calculation
    - Risk management and position sizing
    - Trade history and performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize paper trading simulator.
        
        Args:
            config: Configuration dictionary containing:
                - exchange_interface: ExchangeInterface instance
                - initial_balance: Starting balance (default: 10000 USDT)
                - maker_fee: Maker fee rate (default: 0.001)
                - taker_fee: Taker fee rate (default: 0.001)
                - max_slippage: Maximum slippage percentage (default: 0.005)
                - slippage_model: Slippage calculation model
                - risk_limits: Risk management limits
        """
        self.config = config
        self.logger = system_logger.getChild('PaperTradingSimulator')
        
        # Exchange interface for real-time data
        self.exchange_interface = config.get('exchange_interface')
        if not self.exchange_interface:
            raise ValueError("Exchange interface is required")
        
        # Configuration
        self.initial_balance = config.get('initial_balance', 10000.0)
        self.maker_fee = config.get('maker_fee', 0.001)  # 0.1%
        self.taker_fee = config.get('taker_fee', 0.001)  # 0.1%
        self.max_slippage = config.get('max_slippage', 0.005)  # 0.5%
        self.slippage_model = config.get('slippage_model', 'linear')
        self.risk_limits = config.get('risk_limits', {})
        
        # Simulator state
        self.is_initialized = False
        self.is_running = False
        
        # Account state
        self.balances: Dict[str, float] = {'USDT': self.initial_balance}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.total_volume = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        # Price cache for performance
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 1.0  # 1 second cache TTL
        
        self.logger.info("PaperTradingSimulator initialized")
    
    @handles_errors(default_return=False)
    @traced
    async def initialize(self) -> bool:
        """
        Initialize paper trading simulator.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing PaperTradingSimulator...")
            
            # Validate exchange interface connection
            if not await self.exchange_interface.is_connected():
                raise Exception("Exchange interface not connected")
            
            # Initialize price cache
            await self._initialize_price_cache()
            
            # Set up risk limits
            self._setup_risk_limits()
            
            self.is_initialized = True
            self.logger.info("✅ PaperTradingSimulator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize PaperTradingSimulator: {e}")
            return False
    
    @handles_errors(default_return=None)
    async def _initialize_price_cache(self) -> None:
        """Initialize price cache with current market data."""
        try:
            # Get initial prices for common symbols
            common_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
            
            for symbol in common_symbols:
                try:
                    ticker = await self.exchange_interface.get_ticker(symbol)
                    if ticker:
                        self.price_cache[symbol] = {
                            'price': ticker.price,
                            'bid': ticker.bid_price,
                            'ask': ticker.ask_price,
                            'timestamp': datetime.now()
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to get initial price for {symbol}: {e}")
            
            self.logger.info(f"Price cache initialized with {len(self.price_cache)} symbols")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize price cache: {e}")
            raise
    
    @handles_errors(default_return=None)
    def _setup_risk_limits(self) -> None:
        """Set up risk management limits."""
        try:
            # Default risk limits
            default_limits = {
                'max_position_size': 0.1,  # 10% of portfolio per position
                'max_daily_loss': 0.05,    # 5% max daily loss
                'max_leverage': 1.0,       # No leverage for paper trading
                'stop_loss_pct': 0.02,     # 2% stop loss
                'take_profit_pct': 0.05    # 5% take profit
            }
            
            # Update with user-provided limits
            self.risk_limits.update(default_limits)
            
            self.logger.info("Risk limits configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup risk limits: {e}")
            raise
    
    @handles_errors(default_return=False)
    @traced
    async def start(self) -> bool:
        """
        Start the paper trading simulator.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            if not self.is_initialized:
                raise Exception("Simulator not initialized")
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start background tasks
            asyncio.create_task(self._price_update_loop())
            asyncio.create_task(self._position_update_loop())
            
            tprint_success("📄 Paper trading simulator started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start simulator: {e}")
            return False
    
    @handles_errors(default_return=None)
    async def stop(self) -> None:
        """Stop the paper trading simulator."""
        try:
            self.is_running = False
            
            # Generate final report
            await self._generate_final_report()
            
            tprint_success("✅ Paper trading simulator stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping simulator: {e}")
    
    @handles_errors(default_return={})
    @traced
    async def execute_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an order in the paper trading simulator.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            quantity: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Dict containing order execution result
        """
        try:
            if not self.is_running:
                raise Exception("Simulator not running")
            
            # Validate order parameters
            if not self._validate_order(symbol, side, order_type, quantity, price):
                return {'error': 'Invalid order parameters'}
            
            # Create order
            order = self._create_order(symbol, side, order_type, quantity, price)
            
            # Execute order
            execution_result = await self._execute_order_logic(order)
            
            if execution_result['status'] == 'filled':
                # Update account state
                await self._update_account_state(order, execution_result)
                
                # Record trade
                self._record_trade(order, execution_result)
                
                tprint_success(f"📄 Order executed: {side} {quantity} {symbol} @ {execution_result['price']:.4f}")
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"❌ Order execution failed: {e}")
            return {'error': str(e)}
    
    @handles_errors(default_return=False)
    def _validate_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float]) -> bool:
        """Validate order parameters."""
        try:
            # Basic validation
            if quantity <= 0:
                return False
            
            if order_type == 'limit' and (price is None or price <= 0):
                return False
            
            # Check balance for buy orders
            if side.lower() == 'buy':
                required_balance = quantity * (price or await self._get_current_price(symbol))
                if required_balance > self.balances.get('USDT', 0):
                    return False
            
            # Check position for sell orders
            if side.lower() == 'sell':
                current_position = self.positions.get(symbol, Position(symbol, OrderSide.BUY, 0, 0, 0))
                if quantity > abs(current_position.quantity):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Order validation failed: {e}")
            return False
    
    @handles_errors(default_return=None)
    def _create_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float]) -> Order:
        """Create order object."""
        try:
            order_id = str(uuid.uuid4())
            
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide(side.lower()),
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create order: {e}")
            raise
    
    @handles_errors(default_return={})
    @traced
    async def _execute_order_logic(self, order: Order) -> Dict[str, Any]:
        """Execute order logic with realistic pricing and slippage."""
        try:
            # Get current market price
            current_price = await self._get_current_price(order.symbol)
            if not current_price:
                return {'error': 'Unable to get current price', 'status': 'rejected'}
            
            # Calculate execution price based on order type
            if order.order_type == OrderType.MARKET:
                execution_price = current_price
            elif order.order_type == OrderType.LIMIT:
                execution_price = order.price
            else:
                return {'error': 'Unsupported order type', 'status': 'rejected'}
            
            # Calculate slippage
            slippage = await self._calculate_slippage(order, execution_price)
            final_price = execution_price * (1 + slippage)
            
            # Calculate fees
            fees = await self._calculate_fees(order, final_price)
            
            # Update order
            order.filled_quantity = order.quantity
            order.filled_price = final_price
            order.status = OrderStatus.FILLED
            order.fees = fees
            order.slippage = slippage
            order.filled_at = datetime.now()
            
            # Store order
            self.orders[order.order_id] = order
            
            return {
                'order_id': order.order_id,
                'status': 'filled',
                'price': final_price,
                'quantity': order.quantity,
                'fees': fees,
                'slippage': slippage,
                'timestamp': order.filled_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Order execution logic failed: {e}")
            return {'error': str(e), 'status': 'rejected'}
    
    @handles_errors(default_return=0.0)
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol with caching."""
        try:
            # Check cache first
            if symbol in self.price_cache:
                cache_data = self.price_cache[symbol]
                if (datetime.now() - cache_data['timestamp']).total_seconds() < self.cache_ttl:
                    return cache_data['price']
            
            # Fetch fresh price from exchange
            ticker = await self.exchange_interface.get_ticker(symbol)
            if ticker:
                price = ticker.price
                
                # Update cache
                self.price_cache[symbol] = {
                    'price': price,
                    'bid': ticker.bid_price,
                    'ask': ticker.ask_price,
                    'timestamp': datetime.now()
                }
                
                return price
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get current price for {symbol}: {e}")
            return 0.0
    
    @handles_errors(default_return=0.0)
    async def _calculate_slippage(self, order: Order, base_price: float) -> float:
        """Calculate slippage based on order size and market conditions."""
        try:
            # Base slippage calculation
            if self.slippage_model == 'linear':
                # Linear model: slippage increases with order size
                size_factor = min(order.quantity / 1000.0, 1.0)  # Normalize to 1000 units
                base_slippage = self.max_slippage * size_factor
            else:
                # Default: fixed small slippage
                base_slippage = self.max_slippage * 0.1
            
            # Add random component for realism
            random_factor = np.random.normal(0, 0.001)  # ±0.1% random variation
            slippage = base_slippage + random_factor
            
            # Ensure slippage is within bounds
            slippage = max(-self.max_slippage, min(slippage, self.max_slippage))
            
            return slippage
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate slippage: {e}")
            return 0.0
    
    @handles_errors(default_return=0.0)
    async def _calculate_fees(self, order: Order, execution_price: float) -> float:
        """Calculate trading fees."""
        try:
            # Determine if maker or taker
            # For simplicity, assume all orders are taker orders
            fee_rate = self.taker_fee
            
            # Calculate fee
            notional_value = order.quantity * execution_price
            fee = notional_value * fee_rate
            
            return fee
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate fees: {e}")
            return 0.0
    
    @handles_errors(default_return=None)
    async def _update_account_state(self, order: Order, execution_result: Dict[str, Any]) -> None:
        """Update account balances and positions after order execution."""
        try:
            symbol = order.symbol
            side = order.side
            quantity = order.quantity
            price = execution_result['price']
            fees = execution_result['fees']
            
            # Determine base and quote assets
            if symbol.endswith('USDT'):
                base_asset = symbol[:-4]
                quote_asset = 'USDT'
            else:
                base_asset = symbol[:3]
                quote_asset = symbol[3:]
            
            # Update balances
            if side == OrderSide.BUY:
                # Buying: reduce USDT, increase base asset
                cost = quantity * price + fees
                self.balances[quote_asset] -= cost
                self.balances[base_asset] = self.balances.get(base_asset, 0) + quantity
            else:
                # Selling: reduce base asset, increase USDT
                proceeds = quantity * price - fees
                self.balances[base_asset] -= quantity
                self.balances[quote_asset] = self.balances.get(quote_asset, 0) + proceeds
            
            # Update positions
            await self._update_position(symbol, side, quantity, price, fees)
            
            # Update performance metrics
            self.total_volume += quantity * price
            self.total_fees += fees
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update account state: {e}")
            raise
    
    @handles_errors(default_return=None)
    async def _update_position(self, symbol: str, side: OrderSide, quantity: float, price: float, fees: float) -> None:
        """Update position after trade execution."""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=0,
                    entry_price=0,
                    current_price=price
                )
            
            position = self.positions[symbol]
            current_quantity = position.quantity
            current_entry_price = position.entry_price
            
            if side == OrderSide.BUY:
                if current_quantity >= 0:  # Adding to long position
                    new_quantity = current_quantity + quantity
                    if new_quantity > 0:
                        new_entry_price = ((current_quantity * current_entry_price) + (quantity * price)) / new_quantity
                    else:
                        new_entry_price = price
                else:  # Reducing short position
                    new_quantity = current_quantity + quantity
                    new_entry_price = current_entry_price if new_quantity < 0 else price
            else:  # SELL
                if current_quantity <= 0:  # Adding to short position
                    new_quantity = current_quantity - quantity
                    if new_quantity < 0:
                        new_entry_price = ((abs(current_quantity) * current_entry_price) + (quantity * price)) / abs(new_quantity)
                    else:
                        new_entry_price = price
                else:  # Reducing long position
                    new_quantity = current_quantity - quantity
                    new_entry_price = current_entry_price if new_quantity > 0 else price
            
            # Update position
            position.quantity = new_quantity
            position.entry_price = new_entry_price
            position.current_price = price
            position.fees_paid += fees
            position.updated_at = datetime.now()
            
            # Calculate unrealized P&L
            await self._calculate_unrealized_pnl(position)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update position: {e}")
            raise
    
    @handles_errors(default_return=None)
    async def _calculate_unrealized_pnl(self, position: Position) -> None:
        """Calculate unrealized P&L for position."""
        try:
            if position.quantity == 0:
                position.unrealized_pnl = 0
                return
            
            current_price = await self._get_current_price(position.symbol)
            if current_price == 0:
                return
            
            position.current_price = current_price
            
            if position.quantity > 0:  # Long position
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # Short position
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate unrealized P&L: {e}")
    
    @handles_errors(default_return=None)
    def _record_trade(self, order: Order, execution_result: Dict[str, Any]) -> None:
        """Record trade execution."""
        try:
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_result['price'],
                fees=execution_result['fees'],
                slippage=execution_result['slippage']
            )
            
            self.trades.append(trade)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to record trade: {e}")
    
    @handles_errors(default_return=None)
    async def _price_update_loop(self) -> None:
        """Background task to update prices periodically."""
        try:
            while self.is_running:
                # Update prices for all active positions
                for symbol in self.positions.keys():
                    try:
                        current_price = await self._get_current_price(symbol)
                        if current_price > 0:
                            self.positions[symbol].current_price = current_price
                    except Exception as e:
                        self.logger.warning(f"Failed to update price for {symbol}: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            self.logger.error(f"❌ Price update loop failed: {e}")
    
    @handles_errors(default_return=None)
    async def _position_update_loop(self) -> None:
        """Background task to update position P&L."""
        try:
            while self.is_running:
                # Update unrealized P&L for all positions
                for position in self.positions.values():
                    if position.quantity != 0:
                        await self._calculate_unrealized_pnl(position)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except Exception as e:
            self.logger.error(f"❌ Position update loop failed: {e}")
    
    @handles_errors(default_return={})
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """Get account balance."""
        try:
            if asset:
                return {asset: self.balances.get(asset, 0.0)}
            return self.balances.copy()
        except Exception as e:
            self.logger.error(f"❌ Failed to get balance: {e}")
            return {}
    
    @handles_errors(default_return=[])
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions."""
        try:
            positions = []
            for pos in self.positions.values():
                if pos.quantity != 0 and (symbol is None or pos.symbol == symbol):
                    positions.append({
                        'symbol': pos.symbol,
                        'side': pos.side.value,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl,
                        'fees_paid': pos.fees_paid,
                        'created_at': pos.created_at.isoformat(),
                        'updated_at': pos.updated_at.isoformat()
                    })
            return positions
        except Exception as e:
            self.logger.error(f"❌ Failed to get positions: {e}")
            return []
    
    @handles_errors(default_return={})
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Calculate total portfolio value
            total_value = 0.0
            for asset, balance in self.balances.items():
                if asset == 'USDT':
                    total_value += balance
                else:
                    # Get current price for non-USDT assets
                    symbol = f"{asset}USDT"
                    price = await self._get_current_price(symbol)
                    total_value += balance * price
            
            # Calculate total P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Calculate returns
            initial_value = self.initial_balance
            total_return = (total_value - initial_value) / initial_value * 100 if initial_value > 0 else 0
            
            # Calculate trade statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.side == OrderSide.BUY])  # Simplified
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_value': total_value,
                'initial_balance': initial_value,
                'total_return_pct': total_return,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_pnl': total_pnl,
                'total_volume': self.total_volume,
                'total_fees': self.total_fees,
                'total_trades': total_trades,
                'win_rate_pct': win_rate,
                'positions_count': len([p for p in self.positions.values() if p.quantity != 0]),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance metrics: {e}")
            return {}
    
    @handles_errors(default_return=None)
    async def _generate_final_report(self) -> None:
        """Generate final performance report."""
        try:
            metrics = await self.get_performance_metrics()
            positions = await self.get_positions()
            
            tprint_structured("📊 Paper Trading Final Report", {
                'Total Value': f"${metrics.get('total_value', 0):.2f}",
                'Initial Balance': f"${metrics.get('initial_balance', 0):.2f}",
                'Total Return': f"{metrics.get('total_return_pct', 0):.2f}%",
                'Total P&L': f"${metrics.get('total_pnl', 0):.2f}",
                'Total Volume': f"${metrics.get('total_volume', 0):.2f}",
                'Total Fees': f"${metrics.get('total_fees', 0):.2f}",
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate': f"{metrics.get('win_rate_pct', 0):.2f}%",
                'Active Positions': metrics.get('positions_count', 0)
            })
            
            if positions:
                tprint_info("Active Positions:")
                for pos in positions:
                    tprint_info(f"  {pos['symbol']}: {pos['quantity']:.4f} @ ${pos['entry_price']:.4f} (P&L: ${pos['unrealized_pnl']:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate final report: {e}")