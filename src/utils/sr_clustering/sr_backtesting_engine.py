"""
SR Backtesting Engine

This module provides a backtesting engine specifically designed for Support/Resistance
level trading strategies with enhanced performance optimizations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod

from ..logger import system_logger

# Import optimization utilities
try:
    from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from ..matrix_operations import get_unified_matrix_operations, M1EnhancedMatrixOperations
    from ..hardware.memory_optimization import get_memory_manager, MemoryMonitor
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False
    get_m1_memory_optimizer = None
    get_unified_matrix_operations = None
    get_memory_manager = None

# Import PyTorch for hardware acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import VectorBT for vectorized backtesting
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

logger = logging.getLogger(__name__)

@dataclass
class SRLevel:
    """Support/Resistance level data structure."""
    price: float
    strength: float
    touches: int
    level_type: str  # 'support' or 'resistance'
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    confidence: float = 1.0
    breakout_probability: float = 0.5

@dataclass
class BacktestConfig:
    """Configuration for SR backtesting."""

    # Basic backtesting parameters
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Position sizing
    position_size: float = 0.1  # 10% of capital per position
    max_positions: int = 5
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%

    # SR level parameters
    min_touches: int = 3
    touch_tolerance: float = 0.002  # 0.2%
    strength_threshold: float = 0.5
    min_level_distance: float = 0.01  # 1%

    # Hardware optimization settings
    enable_m1_optimizations: bool = True
    enable_gpu_acceleration: bool = False
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_vectorized_operations: bool = True
    enable_caching: bool = True

    # Memory settings
    memory_limit_gb: float = 4.0
    chunk_size: int = 1000
    cache_size_mb: int = 100

    # Performance settings
    enable_numba_acceleration: bool = True
    max_workers: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.position_size <= 0 or self.position_size > 1:
            raise ValueError("position_size must be between 0 and 1")
        if self.commission < 0:
            raise ValueError("commission cannot be negative")
        if self.stop_loss_pct <= 0 or self.take_profit_pct <= 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be positive")

@dataclass
class BacktestResult:
    """Results from SR backtesting."""

    # Basic information
    strategy_name: str
    symbol: str
    timeframe: str
    success: bool

    # Financial metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int

    # SR-specific metrics
    levels_traded: int
    avg_level_strength: float
    breakout_success_rate: float
    support_resistance_ratio: float

    # Performance metrics
    execution_time: float
    memory_usage_mb: float

    # Detailed results
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    monthly_returns: Optional[pd.Series] = None

    # Error information
    error_message: Optional[str] = None

class SRBacktestingEngine:
    """
    Support/Resistance Backtesting Engine.

    This engine backtests trading strategies based on Support/Resistance levels
    with hardware-accelerated performance optimizations.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize the SR backtesting engine."""
        self.config = config
        self.logger = system_logger.getChild('SRBacktestingEngine')

        # Backtesting state
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = config.initial_capital
        self.cash: float = config.initial_capital
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

        # Hardware optimization setup
        self._setup_hardware_optimizations()

        # Performance tracking
        self.start_time = None
        self.memory_monitor = None

        self.logger.info("✅ SR Backtesting Engine initialized successfully")

    def _setup_hardware_optimizations(self):
        """Setup hardware optimizations based on configuration."""
        if not M1_OPTIMIZATIONS_AVAILABLE:
            self.logger.warning("M1 optimizations not available")
            return

        try:
            # Setup memory optimization
            if self.config.enable_memory_optimization:
                self.memory_monitor = get_memory_manager() if get_memory_manager else None
                if self.memory_monitor:
                    self.memory_monitor.set_memory_limit(self.config.memory_limit_gb * 1024**3)

            # Setup matrix operations
            if self.config.enable_vectorized_operations:
                self.matrix_ops = get_unified_matrix_operations()
            else:
                self.matrix_ops = None

        except Exception as e:
            self.logger.warning(f"Failed to setup hardware optimizations: {e}")

    async def backtest_sr_strategy(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[SRLevel],
        strategy_name: str = "SR_Strategy"
    ) -> BacktestResult:
        """
        Backtest SR trading strategy.

        Args:
            market_data: OHLCV market data
            sr_levels: List of SR levels to trade
            strategy_name: Name of the strategy

        Returns:
            BacktestResult with comprehensive results
        """
        self.logger.info(f"🔄 Starting backtest for strategy: {strategy_name}")
        self.start_time = datetime.now()

        try:
            # Reset backtesting state
            self._reset_backtest_state()

            # Validate inputs
            if not self._validate_inputs(market_data, sr_levels):
                return BacktestResult(
                    strategy_name=strategy_name,
                    symbol=getattr(market_data, 'name', 'UNKNOWN'),
                    timeframe='UNKNOWN',
                    success=False,
                    total_return=0.0,
                    annualized_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    levels_traded=0,
                    avg_level_strength=0.0,
                    breakout_success_rate=0.0,
                    support_resistance_ratio=0.0,
                    execution_time=0.0,
                    memory_usage_mb=0.0,
                    error_message="Invalid inputs"
                )

            # Filter and sort SR levels
            valid_levels = self._filter_sr_levels(sr_levels)

            # Run backtest
            await self._run_backtest(market_data, valid_levels)

            # Calculate performance metrics
            performance = self._calculate_performance_metrics(market_data, strategy_name)

            execution_time = (datetime.now() - self.start_time).total_seconds()
            memory_usage = self._get_memory_usage()

            self.logger.info(f"✅ Backtest completed successfully in {execution_time:.2f}s")

            return BacktestResult(
                strategy_name=strategy_name,
                symbol=getattr(market_data, 'name', 'UNKNOWN'),
                timeframe='UNKNOWN',
                success=True,
                total_return=performance['total_return'],
                annualized_return=performance['annualized_return'],
                sharpe_ratio=performance['sharpe_ratio'],
                max_drawdown=performance['max_drawdown'],
                win_rate=performance['win_rate'],
                profit_factor=performance['profit_factor'],
                total_trades=performance['total_trades'],
                levels_traded=len(valid_levels),
                avg_level_strength=np.mean([level.strength for level in valid_levels]) if valid_levels else 0.0,
                breakout_success_rate=performance.get('breakout_success_rate', 0.0),
                support_resistance_ratio=performance.get('support_resistance_ratio', 0.0),
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                trades=self.trades.copy(),
                equity_curve=pd.Series(self.equity_curve, index=market_data.index[:len(self.equity_curve)]),
                monthly_returns=performance.get('monthly_returns')
            )

        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=getattr(market_data, 'name', 'UNKNOWN'),
                timeframe='UNKNOWN',
                success=False,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                levels_traded=0,
                avg_level_strength=0.0,
                breakout_success_rate=0.0,
                support_resistance_ratio=0.0,
                execution_time=(datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0,
                memory_usage_mb=self._get_memory_usage(),
                error_message=str(e)
            )

    def _reset_backtest_state(self):
        """Reset backtesting state for new backtest."""
        self.current_positions.clear()
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.trades.clear()
        self.equity_curve = [self.config.initial_capital]

    def _validate_inputs(self, market_data: pd.DataFrame, sr_levels: List[SRLevel]) -> bool:
        """Validate backtest inputs."""
        if market_data is None or len(market_data) == 0:
            self.logger.error("Market data is None or empty")
            return False

        if not sr_levels:
            self.logger.error("No SR levels provided")
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        return True

    def _filter_sr_levels(self, sr_levels: List[SRLevel]) -> List[SRLevel]:
        """Filter SR levels based on configuration criteria."""
        filtered_levels = []

        for level in sr_levels:
            # Filter by minimum touches
            if level.touches < self.config.min_touches:
                continue

            # Filter by strength threshold
            if level.strength < self.config.strength_threshold:
                continue

            # Filter by minimum distance from current price
            current_price = self.config.initial_capital  # This should be market data based
            price_distance = abs(level.price - current_price) / current_price
            if price_distance < self.config.min_level_distance:
                continue

            filtered_levels.append(level)

        self.logger.info(f"Filtered {len(sr_levels)} levels to {len(filtered_levels)} valid levels")
        return filtered_levels

    async def _run_backtest(self, market_data: pd.DataFrame, sr_levels: List[SRLevel]):
        """Run the main backtesting loop."""
        self.logger.info(f"Running backtest with {len(sr_levels)} SR levels")

        # Sort levels by price for efficient processing
        sr_levels.sort(key=lambda x: x.price)

        for idx, row in market_data.iterrows():
            current_price = row['close']
            current_time = idx

            # Update equity curve
            self.equity_curve.append(self.portfolio_value)

            # Check for level interactions
            await self._process_level_interactions(current_price, current_time, sr_levels, row)

            # Update portfolio value (simplified)
            self.portfolio_value = self.cash + sum(self.current_positions.values())

    async def _process_level_interactions(
        self,
        current_price: float,
        current_time: pd.Timestamp,
        sr_levels: List[SRLevel],
        market_row: pd.Series
    ):
        """Process interactions with SR levels."""
        for level in sr_levels:
            # Check if price is near level (within tolerance)
            price_distance = abs(current_price - level.price) / level.price

            if price_distance <= self.config.touch_tolerance:
                # Generate trading signal based on level type
                signal = self._generate_trading_signal(level, current_price, market_row)

                if signal:
                    await self._execute_trade(signal, current_price, current_time, level, market_row)

    def _generate_trading_signal(
        self,
        level: SRLevel,
        current_price: float,
        market_row: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on SR level interaction."""
        # Simple breakout strategy
        if level.level_type == 'resistance' and current_price > level.price:
            return {
                'action': 'sell',
                'level_type': 'resistance',
                'breakout': True,
                'confidence': level.confidence
            }
        elif level.level_type == 'support' and current_price < level.price:
            return {
                'action': 'buy',
                'level_type': 'support',
                'breakout': True,
                'confidence': level.confidence
            }

        return None

    async def _execute_trade(
        self,
        signal: Dict[str, Any],
        current_price: float,
        current_time: pd.Timestamp,
        level: SRLevel,
        market_row: pd.Series
    ):
        """Execute trade based on signal."""
        action = signal['action']

        # Calculate position size
        position_value = self.portfolio_value * self.config.position_size

        if action == 'buy' and self.cash >= position_value:
            # Buy signal
            quantity = position_value / current_price
            cost = quantity * current_price * (1 + self.config.commission)

            if self.cash >= cost:
                self.cash -= cost
                self.current_positions[current_time] = quantity

                # Record trade
                self.trades.append({
                    'timestamp': current_time,
                    'action': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'level_price': level.price,
                    'level_type': level.level_type,
                    'pnl': 0.0
                })

        elif action == 'sell' and len(self.current_positions) > 0:
            # Sell signal - close all positions
            for pos_time, quantity in list(self.current_positions.items()):
                pnl = (current_price - market_row['open']) * quantity  # Simplified PnL
                revenue = quantity * current_price * (1 - self.config.commission)

                self.cash += revenue
                del self.current_positions[pos_time]

                # Record trade
                self.trades.append({
                    'timestamp': current_time,
                    'action': 'sell',
                    'price': current_price,
                    'quantity': quantity,
                    'level_price': level.price,
                    'level_type': level.level_type,
                    'pnl': pnl
                })

    def _calculate_performance_metrics(self, market_data: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }

        # Extract trade data
        trade_df = pd.DataFrame(self.trades)
        if len(trade_df) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }

        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            if self.equity_curve[i-1] > 0:
                ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
                returns.append(ret)

        if not returns:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': len(self.trades)
            }

        returns = np.array(returns)

        # Calculate metrics
        total_return = (self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital

        # Annualized return (assuming daily data)
        num_days = len(market_data)
        if num_days > 0:
            years = num_days / 252  # Assuming trading days per year
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        else:
            annualized_return = 0.0

        # Sharpe ratio
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Win rate and profit factor
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        losing_trades = len(self.trades) - winning_trades

        win_rate = winning_trades / len(self.trades) if self.trades else 0.0

        total_profits = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        total_losses = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))

        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades)
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

def get_backtesting_engine(config: Optional[BacktestConfig] = None) -> SRBacktestingEngine:
    """
    Factory function to create SR backtesting engine.

    Args:
        config: Backtest configuration (creates default if None)

    Returns:
        Configured SRBacktestingEngine instance
    """
    if config is None:
        config = BacktestConfig()

    return SRBacktestingEngine(config)

# Export main classes and functions
__all__ = [
    'SRBacktestingEngine',
    'BacktestConfig',
    'SRLevel',
    'BacktestResult',
    'get_backtesting_engine'
]
