"""
Position Manager

This module manages trading positions, portfolio tracking, and PnL calculations.
Integrates with shared utilities for data processing, validation, and optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Import shared utilities
from ..utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from ..utils.common_utilities import calculate_data_quality_metrics, safe_convert_dtypes
from ..utils.math_validation import safe_divide, validate_finite, safe_log, safe_sqrt
from ..utils.serialization_utils import JSONSerializer, PickleSerializer
from ..utils.tprint import tprint
from ..utils.data.klines_parquet import KlinesParquetManager
from ..utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from ..utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
from ..utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from ..utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from ..utils.matrix_operations import UnifiedMatrixOperations

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    side: str  # 'long' or 'short'
    leverage: float = 1.0
    margin_used: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class PortfolioMetrics:
    """Portfolio metrics data structure"""
    total_value: float
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_margin_used: float
    available_margin: float
    leverage: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    timestamp: datetime


class PositionManager:
    """Manages trading positions and portfolio tracking"""
    
    def __init__(self, config: Any, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Portfolio metrics
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        self.metrics_history: List[PortfolioMetrics] = []
        
        # Initialize utilities
        self._init_utilities()
        
        # Performance tracking
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
    def _init_utilities(self):
        """Initialize shared utilities"""
        try:
            # Initialize data utilities
            self.klines_manager = KlinesParquetManager()
            self.matrix_ops = UnifiedMatrixOperations()
            
            # Initialize hardware optimizations if available
            if is_m1_available():
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                tprint("✅ M1 hardware optimizations enabled")
            else:
                self.m1_gpu_manager = None
                self.m1_memory_optimizer = None
                self.m1_cpu_optimizer = None
                tprint("⚠️ M1 hardware optimizations not available")
                
            # Initialize optimization utilities
            self.bayesian_optimizer = BayesianTPEOptimizer()
            
            tprint("✅ Position manager utilities initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing utilities: {e}")
            tprint(f"⚠️ Some utilities may not be available: {e}")
    
    async def start(self) -> None:
        """Start position management monitoring"""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_positions())
        self.logger.info("Position manager started")
        tprint("✅ Position manager started")
    
    async def stop(self) -> None:
        """Stop position management monitoring"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Position manager stopped")
        tprint("✅ Position manager stopped")
    
    async def update_position(self, symbol: str, quantity: float, price: float, 
                             side: str = "long", leverage: float = 1.0) -> None:
        """Update position for a symbol"""
        try:
            # Validate inputs
            quantity = validate_finite(quantity, "quantity")
            price = validate_finite(price, "price")
            leverage = validate_finite(leverage, "leverage")
            
            # Get current position
            current_position = self.positions.get(symbol)
            
            if current_position:
                # Update existing position
                new_quantity = current_position.quantity + quantity
                
                if abs(new_quantity) < 1e-8:  # Position closed
                    # Calculate realized PnL
                    realized_pnl = self._calculate_realized_pnl(
                        current_position.quantity, current_position.average_price, 
                        -current_position.quantity, price
                    )
                    
                    # Update trade history
                    self._add_trade_record(symbol, current_position.quantity, 
                                        current_position.average_price, price, 
                                        realized_pnl, "close")
                    
                    # Remove position
                    del self.positions[symbol]
                    tprint(f"✅ Position closed: {symbol} PnL: {realized_pnl:.2f}")
                    
                else:
                    # Update position
                    new_avg_price = self._calculate_average_price(
                        current_position.quantity, current_position.average_price,
                        quantity, price
                    )
                    
                    current_position.quantity = new_quantity
                    current_position.average_price = new_avg_price
                    current_position.timestamp = datetime.now()
                    
                    # Add to trade history
                    self._add_trade_record(symbol, quantity, price, price, 0.0, "update")
                    
                    tprint(f"✅ Position updated: {symbol} Qty: {new_quantity:.4f} Avg: {new_avg_price:.2f}")
            else:
                # Create new position
                if abs(quantity) > 1e-8:  # Only create if significant quantity
                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        average_price=price,
                        current_price=price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        timestamp=datetime.now(),
                        side=side,
                        leverage=leverage,
                        margin_used=abs(quantity) * price / leverage
                    )
                    
                    self.positions[symbol] = position
                    self._add_trade_record(symbol, quantity, price, price, 0.0, "open")
                    
                    tprint(f"✅ New position: {symbol} Qty: {quantity:.4f} Price: {price:.2f}")
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating position for {symbol}: {e}")
            tprint(f"❌ Position update failed: {e}")
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol"""
        return self.positions.get(symbol)
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
    
    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics"""
        if not self.portfolio_metrics:
            await self._update_portfolio_metrics()
        
        return self.portfolio_metrics
    
    async def calculate_position_pnl(self, symbol: str) -> Tuple[float, float]:
        """Calculate unrealized PnL for a position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return 0.0, 0.0
            
            # Get current market price
            ticker = await self.exchange_client.get_ticker(symbol)
            current_price = float(ticker.get("last", 0)) if ticker else position.current_price
            
            # Calculate unrealized PnL
            if position.side == "long":
                unrealized_pnl = (current_price - position.average_price) * position.quantity
            else:  # short
                unrealized_pnl = (position.average_price - current_price) * abs(position.quantity)
            
            # Calculate percentage return
            percentage_return = safe_divide(unrealized_pnl, 
                                         position.average_price * abs(position.quantity), 0.0)
            
            return unrealized_pnl, percentage_return
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating PnL for {symbol}: {e}")
            return 0.0, 0.0
    
    def _calculate_realized_pnl(self, quantity: float, avg_price: float, 
                              close_quantity: float, close_price: float) -> float:
        """Calculate realized PnL for a trade"""
        if quantity == 0 or close_quantity == 0:
            return 0.0
        
        # Determine if long or short
        if quantity > 0:  # Long position
            return (close_price - avg_price) * abs(close_quantity)
        else:  # Short position
            return (avg_price - close_price) * abs(close_quantity)
    
    def _calculate_average_price(self, current_qty: float, current_avg: float,
                               new_qty: float, new_price: float) -> float:
        """Calculate new average price after adding to position"""
        if current_qty == 0:
            return new_price
        
        total_qty = current_qty + new_qty
        if abs(total_qty) < 1e-8:
            return 0.0
        
        total_value = (current_qty * current_avg) + (new_qty * new_price)
        return safe_divide(total_value, total_qty, 0.0)
    
    def _add_trade_record(self, symbol: str, quantity: float, price: float, 
                         current_price: float, realized_pnl: float, action: str) -> None:
        """Add trade record to history"""
        trade_record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "current_price": current_price,
            "realized_pnl": realized_pnl,
            "action": action
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only recent history (last 10000 trades)
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-10000:]
    
    async def _monitor_positions(self) -> None:
        """Monitor positions continuously"""
        while self._running:
            try:
                # Update current prices and PnL
                for symbol, position in self.positions.items():
                    ticker = await self.exchange_client.get_ticker(symbol)
                    if ticker:
                        position.current_price = float(ticker.get("last", position.current_price))
                        
                        # Calculate unrealized PnL
                        unrealized_pnl, _ = await self.calculate_position_pnl(symbol)
                        position.unrealized_pnl = unrealized_pnl
                
                # Update portfolio metrics
                await self._update_portfolio_metrics()
                
                # Wait before next update
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in position monitoring: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _update_portfolio_metrics(self) -> None:
        """Update comprehensive portfolio metrics"""
        try:
            # Calculate totals
            total_value = 0.0
            total_unrealized_pnl = 0.0
            total_realized_pnl = 0.0
            total_margin_used = 0.0
            
            for position in self.positions.values():
                position_value = abs(position.quantity) * position.current_price
                total_value += position_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
                total_margin_used += position.margin_used
            
            # Calculate daily PnL from trade history
            today = datetime.now().date()
            daily_pnl = sum(
                trade.get("realized_pnl", 0.0) for trade in self.trade_history
                if trade.get("timestamp", datetime.now()).date() == today
            )
            
            # Calculate leverage
            account_info = await self.exchange_client.get_account_info()
            total_balance = float(account_info.get("totalBalance", 1.0))
            leverage = safe_divide(total_margin_used, total_balance, 0.0)
            
            # Calculate performance metrics
            sharpe_ratio = await self._calculate_sharpe_ratio()
            max_drawdown = await self._calculate_max_drawdown()
            win_rate = await self._calculate_win_rate()
            profit_factor = await self._calculate_profit_factor()
            
            # Create metrics
            self.portfolio_metrics = PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_unrealized_pnl + total_realized_pnl,
                daily_pnl=daily_pnl,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=total_realized_pnl,
                total_margin_used=total_margin_used,
                available_margin=total_balance - total_margin_used,
                leverage=leverage,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.metrics_history.append(self.portfolio_metrics)
            
            # Keep only recent history (last 1000 entries)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"❌ Error updating portfolio metrics: {e}")
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade history"""
        try:
            if len(self.trade_history) < 2:
                return 0.0
            
            # Get recent PnL values
            recent_pnl = [trade.get("realized_pnl", 0.0) for trade in self.trade_history[-100:]]
            
            if not recent_pnl:
                return 0.0
            
            mean_return = np.mean(recent_pnl)
            std_return = np.std(recent_pnl)
            
            return safe_divide(mean_return, std_return, 0.0)
            
        except Exception:
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.trade_history:
                return 0.0
            
            # Calculate cumulative PnL
            cumulative_pnl = []
            running_total = 0.0
            
            for trade in self.trade_history:
                running_total += trade.get("realized_pnl", 0.0)
                cumulative_pnl.append(running_total)
            
            if not cumulative_pnl:
                return 0.0
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0.0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                drawdown = peak - pnl
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception:
            return 0.0
    
    async def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        try:
            if not self.trade_history:
                return 0.0
            
            winning_trades = sum(1 for trade in self.trade_history 
                               if trade.get("realized_pnl", 0.0) > 0)
            total_trades = len(self.trade_history)
            
            return safe_divide(winning_trades, total_trades, 0.0)
            
        except Exception:
            return 0.0
    
    async def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            if not self.trade_history:
                return 0.0
            
            gross_profit = sum(trade.get("realized_pnl", 0.0) for trade in self.trade_history
                             if trade.get("realized_pnl", 0.0) > 0)
            gross_loss = abs(sum(trade.get("realized_pnl", 0.0) for trade in self.trade_history
                               if trade.get("realized_pnl", 0.0) < 0))
            
            return safe_divide(gross_profit, gross_loss, 0.0)
            
        except Exception:
            return 0.0