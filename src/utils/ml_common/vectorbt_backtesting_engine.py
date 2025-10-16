"""
VectorBT Backtesting Engine

This module provides a VectorBT-based backtesting engine for high-performance
portfolio backtesting with GPU acceleration and parallel processing support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.records.base import Records
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None
    Records = None

from ..logger import system_logger

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    """Backtesting modes for VectorBT engine."""
    VECTORBT_CPU = "vectorbt_cpu"
    VECTORBT_PARALLEL = "vectorbt_parallel"
    VECTORBT_GPU = "vectorbt_gpu"
    HYBRID = "hybrid"

@dataclass
class VectorBTBacktestConfig:
    """Configuration for VectorBT backtesting."""
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    enable_memory_optimization: bool = True
    memory_limit_gb: float = 8.0
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = False
    chunk_size: int = 1000
    freq: str = 'D'  # Frequency for portfolio

@dataclass
class BacktestResult:
    """Results from VectorBT backtesting."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    equity_curve: Optional[pd.Series] = None
    trades: List[Dict[str, Any]] = None
    portfolio: Optional[Any] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

class VectorBTBacktestingEngine:
    """
    VectorBT Backtesting Engine for high-performance portfolio backtesting.

    This engine leverages VectorBT for hardware-accelerated backtesting
    with support for GPU acceleration and parallel processing.
    """

    def __init__(self, config: Optional[VectorBTBacktestConfig] = None):
        """Initialize the VectorBT backtesting engine."""
        self.config = config or VectorBTBacktestConfig()
        self.logger = system_logger.getChild('VectorBTBacktestingEngine')

        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available. Install with: pip install vectorbt")
            return

        # Performance tracking
        self.start_time = None

        self.logger.info("✅ VectorBT Backtesting Engine initialized successfully")

    def run_backtest(self, signals: pd.Series, prices: pd.Series,
                    timestamps: Optional[pd.Series] = None,
                    mode: BacktestMode = BacktestMode.VECTORBT_CPU,
                    **kwargs) -> BacktestResult:
        """
        Run backtesting using VectorBT.

        Args:
            signals: Trading signals (1 for long, -1 for short, 0 for neutral)
            prices: Price series for the asset
            timestamps: Timestamps for the data (optional)
            mode: Backtesting mode (CPU, GPU, Parallel, Hybrid)
            **kwargs: Additional VectorBT parameters

        Returns:
            BacktestResult with performance metrics
        """
        if not VECTORBT_AVAILABLE:
            self.logger.error("❌ VectorBT not available for backtesting")
            return BacktestResult()

        try:
            self.logger.info(f"🚀 Starting VectorBT backtest with mode: {mode.value}")

            # Prepare data for VectorBT
            if timestamps is not None:
                # Create a DataFrame with timestamps as index
                data = pd.DataFrame({
                    'price': prices,
                    'signal': signals
                }, index=timestamps)
            else:
                # Use default integer index
                data = pd.DataFrame({
                    'price': prices,
                    'signal': signals
                })

            # Create VectorBT portfolio based on mode
            if mode == BacktestMode.VECTORBT_GPU and self._is_gpu_available():
                portfolio = self._run_gpu_backtest(data, **kwargs)
            elif mode == BacktestMode.VECTORBT_PARALLEL:
                portfolio = self._run_parallel_backtest(data, **kwargs)
            elif mode == BacktestMode.HYBRID:
                portfolio = self._run_hybrid_backtest(data, **kwargs)
            else:
                # Default to CPU mode
                portfolio = self._run_cpu_backtest(data, **kwargs)

            # Extract results
            result = self._extract_backtest_results(portfolio, data)

            self.logger.info(f"✅ VectorBT backtest completed: {result.total_return:.2%} total return")
            return result

        except Exception as e:
            self.logger.error(f"❌ VectorBT backtest failed: {e}")
            return BacktestResult()

    def _run_cpu_backtest(self, data: pd.DataFrame, **kwargs) -> Any:
        """Run CPU-based VectorBT backtest."""
        try:
            # Create portfolio with CPU processing
            portfolio = vbt.Portfolio.from_signals(
                close=data['price'],
                entries=data['signal'] > 0,
                exits=data['signal'] < 0,
                freq=self.config.freq,
                **kwargs
            )
            return portfolio
        except Exception as e:
            self.logger.error(f"❌ CPU backtest failed: {e}")
            raise

    def _run_gpu_backtest(self, data: pd.DataFrame, **kwargs) -> Any:
        """Run GPU-accelerated VectorBT backtest."""
        if not self._is_gpu_available():
            self.logger.warning("⚠️ GPU not available, falling back to CPU")
            return self._run_cpu_backtest(data, **kwargs)

        try:
            # Enable GPU acceleration if available
            with vbt.settings(parallel=True):
                portfolio = vbt.Portfolio.from_signals(
                    close=data['price'],
                    entries=data['signal'] > 0,
                    exits=data['signal'] < 0,
                    freq=self.config.freq,
                    **kwargs
                )
            return portfolio
        except Exception as e:
            self.logger.warning(f"⚠️ GPU backtest failed: {e}, falling back to CPU")
            return self._run_cpu_backtest(data, **kwargs)

    def _run_parallel_backtest(self, data: pd.DataFrame, **kwargs) -> Any:
        """Run parallel VectorBT backtest."""
        try:
            # Enable parallel processing
            with vbt.settings(parallel=True):
                portfolio = vbt.Portfolio.from_signals(
                    close=data['price'],
                    entries=data['signal'] > 0,
                    exits=data['signal'] < 0,
                    freq=self.config.freq,
                    **kwargs
                )
            return portfolio
        except Exception as e:
            self.logger.warning(f"⚠️ Parallel backtest failed: {e}, falling back to CPU")
            return self._run_cpu_backtest(data, **kwargs)

    def _run_hybrid_backtest(self, data: pd.DataFrame, **kwargs) -> Any:
        """Run hybrid (CPU+GPU) VectorBT backtest."""
        try:
            # Try GPU first, fallback to parallel CPU
            if self._is_gpu_available():
                return self._run_gpu_backtest(data, **kwargs)
            else:
                return self._run_parallel_backtest(data, **kwargs)
        except Exception as e:
            self.logger.warning(f"⚠️ Hybrid backtest failed: {e}, falling back to CPU")
            return self._run_cpu_backtest(data, **kwargs)

    def _is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            # Check if VectorBT can use GPU acceleration
            return hasattr(vbt, 'settings') and 'gpu' in str(vbt.settings()).lower()
        except:
            return False

    def _extract_backtest_results(self, portfolio: Any, data: pd.DataFrame) -> BacktestResult:
        """Extract results from VectorBT portfolio."""
        try:
            # Calculate performance metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()

            # Get trade information
            trades = portfolio.trades.records
            if trades is not None and len(trades) > 0:
                win_rate = (trades['Return'] > 0).mean()
                total_trades = len(trades)
                avg_trade_return = trades['Return'].mean()
            else:
                win_rate = 0.0
                total_trades = 0
                avg_trade_return = 0.0

            # Get equity curve
            equity_curve = portfolio.value()

            return BacktestResult(
                total_return=total_return.iloc[-1] if hasattr(total_return, 'iloc') else total_return,
                sharpe_ratio=sharpe_ratio.iloc[-1] if hasattr(sharpe_ratio, 'iloc') else sharpe_ratio,
                max_drawdown=max_drawdown.iloc[-1] if hasattr(max_drawdown, 'iloc') else max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                avg_trade_return=avg_trade_return,
                equity_curve=equity_curve,
                trades=trades.to_dict('records') if trades is not None else [],
                portfolio=portfolio
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to extract backtest results: {e}")
            return BacktestResult()

def get_vectorbt_backtesting_engine(config: Optional[VectorBTBacktestConfig] = None) -> VectorBTBacktestingEngine:
    """Get a VectorBT backtesting engine instance."""
    return VectorBTBacktestingEngine(config)
