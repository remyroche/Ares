"""
Vectorized Backtesting Engine

This module provides a vectorized backtesting engine for high-performance
portfolio backtesting with hardware acceleration support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..logger import system_logger

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    """Backtesting modes for different optimization levels."""
    VECTORIZED = "vectorized"
    GPU_ACCELERATED = "gpu_accelerated"

@dataclass
class VectorizedBacktestConfig:
    """Configuration for vectorized backtesting."""
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    enable_memory_optimization: bool = True
    memory_limit_gb: float = 8.0
    enable_parallel_processing: bool = True
    chunk_size: int = 1000

@dataclass
class BacktestResult:
    """Results from vectorized backtesting."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    equity_curve: Optional[pd.Series] = None
    trades: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

class VectorizedBacktestingEngine:
    """
    Vectorized Backtesting Engine for high-performance portfolio backtesting.

    This engine provides hardware-accelerated backtesting capabilities with
    support for large datasets and complex trading strategies.
    """

    def __init__(self, config: Optional[VectorizedBacktestConfig] = None):
        """Initialize the vectorized backtesting engine."""
        self.config = config or VectorizedBacktestConfig()
        self.logger = system_logger.getChild('VectorizedBacktestingEngine')

        # Backtesting state
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = self.config.initial_capital
        self.cash: float = self.config.initial_capital
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

        # Performance tracking
        self.start_time = None

        self.logger.info("✅ Vectorized Backtesting Engine initialized successfully")

    def run_vectorized_backtest(self, signals: pd.Series, prices: pd.Series,
                               **kwargs) -> BacktestResult:
        """
        Run vectorized backtesting on trading signals.

        Args:
            signals: Trading signals (1 for long, -1 for short, 0 for neutral)
            prices: Price series for the asset
            **kwargs: Additional parameters

        Returns:
            BacktestResult with performance metrics
        """
        try:
            self.logger.info(f"🚀 Starting vectorized backtest with {len(signals)} signals")

            # Initialize backtesting
            self._initialize_backtest()

            # Generate positions from signals
            positions = self._generate_positions(signals)

            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions, prices)

            # Generate equity curve
            equity_curve = self._generate_equity_curve(portfolio_returns)

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio_returns, equity_curve)

            # Generate trade records
            trades = self._generate_trade_records(positions, prices)

            result = BacktestResult(
                total_return=metrics['total_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                total_trades=len(trades),
                avg_trade_return=metrics['avg_trade_return'],
                equity_curve=equity_curve,
                trades=trades
            )

            self.logger.info(f"✅ Vectorized backtest completed: {result.total_return:.2%} total return")
            return result

        except Exception as e:
            self.logger.error(f"❌ Vectorized backtest failed: {e}")
            return BacktestResult()

    def _initialize_backtest(self):
        """Initialize backtesting state."""
        self.current_positions.clear()
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.trades.clear()
        self.equity_curve.clear()

    def _generate_positions(self, signals: pd.Series) -> pd.Series:
        """Generate position sizes from trading signals."""
        # Simple position sizing: 100% long/short based on signals
        positions = signals.copy()
        return positions

    def _calculate_portfolio_returns(self, positions: pd.Series, prices: pd.Series) -> pd.Series:
        """Calculate portfolio returns from positions and prices."""
        # Calculate price returns
        price_returns = prices.pct_change().fillna(0)

        # Calculate strategy returns (position * price return)
        strategy_returns = positions.shift(1) * price_returns

        # Apply transaction costs
        transaction_costs = self._calculate_transaction_costs(positions)
        strategy_returns -= transaction_costs

        return strategy_returns

    def _calculate_transaction_costs(self, positions: pd.Series) -> pd.Series:
        """Calculate transaction costs for position changes."""
        position_changes = positions.diff().abs()
        costs = position_changes * (self.config.commission + self.config.slippage)
        return costs

    def _generate_equity_curve(self, returns: pd.Series) -> pd.Series:
        """Generate equity curve from returns."""
        equity_curve = (1 + returns).cumprod() * self.config.initial_capital
        return equity_curve

    def _calculate_performance_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0
            }

        # Total return
        total_return = (equity_curve.iloc[-1] / self.config.initial_capital) - 1

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            annual_return = returns.mean() * 252  # Assuming daily returns
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        # Win rate and average trade return (simplified)
        positive_returns = (returns > 0).sum()
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0.0
        avg_trade_return = returns.mean() if len(returns) > 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return
        }

    def _generate_trade_records(self, positions: pd.Series, prices: pd.Series) -> List[Dict[str, Any]]:
        """Generate detailed trade records."""
        trades = []

        # Simple trade generation (can be enhanced)
        position_changes = positions.diff()
        trade_indices = position_changes[position_changes != 0].index

        for idx in trade_indices:
            trade = {
                'entry_date': idx,
                'entry_price': prices.loc[idx],
                'position_size': positions.loc[idx],
                'type': 'long' if positions.loc[idx] > 0 else 'short'
            }
            trades.append(trade)

        return trades

def get_vectorized_backtesting_engine(config: Optional[VectorizedBacktestConfig] = None) -> VectorizedBacktestingEngine:
    """Get a vectorized backtesting engine instance."""
    return VectorizedBacktestingEngine(config)
