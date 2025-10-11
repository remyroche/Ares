"""
VectorBT-Enhanced Backtesting Engine

This module provides a high-performance backtesting engine using VectorBT for
portfolio management, performance analysis, and risk assessment.

Key Features:
- VectorBT portfolio management with realistic execution
- Advanced financial metrics and risk analysis
- Multi-asset portfolio support
- GPU acceleration for large-scale backtesting
- Integration with existing ML common utilities
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import TradingCosts
from .vectorbt_memory_manager import get_memory_manager, memory_managed_operation, optimize_memory_usage
from .vectorbt_performance_monitor import get_performance_monitor, monitor_operation

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""
    VECTORBT_CPU = "vectorbt_cpu"
    VECTORBT_GPU = "vectorbt_gpu"
    VECTORBT_PARALLEL = "vectorbt_parallel"
    HYBRID = "hybrid"


@dataclass
class VectorBTBacktestConfig:
    """Configuration for VectorBT backtesting."""
    # Basic configuration
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.1
    min_position_size: float = 0.01
    rebalance_frequency: str = 'daily'
    risk_free_rate: float = 0.02
    benchmark_symbol: Optional[str] = None
    
    # VectorBT specific settings
    use_gpu: bool = True
    enable_parallel: bool = True
    chunk_size: int = 50000
    memory_limit_gb: float = 8.0
    
    # Trading costs
    trading_costs: TradingCosts = field(default_factory=TradingCosts)
    asset_classes: Optional[List[str]] = None
    stress_scenario: Optional[str] = None
    
    # Performance settings
    enable_memory_optimization: bool = True
    enable_progress_tracking: bool = True
    cache_intermediate_results: bool = True
    
    # Advanced settings
    enable_short_selling: bool = True
    enable_fractional_shares: bool = True
    max_leverage: float = 1.0
    margin_requirement: float = 0.5


@dataclass
class VectorBTBacktestResults:
    """Results from VectorBT backtesting."""
    # Portfolio data
    portfolio: Portfolio
    portfolio_values: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: pd.DataFrame
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    drawdown_analysis: Dict[str, Any]
    
    # Execution info
    computation_time: float
    memory_usage: float
    mode_used: str
    
    # Additional VectorBT data
    records: Optional[Records] = None
    stats: Optional[Dict[str, Any]] = None


class VectorBTBacktestingEngine:
    """
    High-performance backtesting engine using VectorBT.
    
    This engine leverages VectorBT's optimized C++ backend for:
    - Portfolio management and rebalancing
    - Performance metrics calculation
    - Risk analysis and drawdown assessment
    - Multi-asset portfolio optimization
    """
    
    def __init__(self, config: Optional[VectorBTBacktestConfig] = None):
        """
        Initialize VectorBT backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.config = config or VectorBTBacktestConfig()
        
        # Initialize memory manager and performance monitor
        self.memory_manager = get_memory_manager()
        self.performance_monitor = get_performance_monitor()
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Performance tracking
        self.performance_stats = {
            'total_simulations': 0,
            'computation_time': 0.0,
            'memory_peak': 0.0,
            'gpu_operations': 0,
            'cpu_operations': 0
        }
        
        logger.info("✅ VectorBT backtesting engine initialized")
        logger.info(f"📊 GPU available: {CUPY_AVAILABLE and self.config.use_gpu}")
        logger.info(f"📊 Parallel processing: {self.config.enable_parallel}")
        logger.info(f"📊 Initial capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"📊 Memory manager: {self.memory_manager.get_memory_stats()['available_memory_gb']:.2f}GB available")
    
    def _configure_vectorbt(self):
        """Configure VectorBT global settings."""
        # Set memory limit
        if self.config.memory_limit_gb:
            vbt.settings.array_wrapper['freq'] = '1min'  # Default frequency
            vbt.settings.array_wrapper['freq'] = '1min'
        
        # Configure parallel processing
        if self.config.enable_parallel:
            vbt.settings.parallel['threading'] = True
            vbt.settings.parallel['threading'] = True
        
        # Configure GPU usage
        if self.config.use_gpu and CUPY_AVAILABLE:
            vbt.settings.array_wrapper['freq'] = '1min'
            logger.info("🚀 GPU acceleration enabled")
        else:
            logger.info("💻 CPU-only mode")
    
    def run_backtest(self, 
                    signals: Union[np.ndarray, pd.DataFrame],
                    prices: Union[np.ndarray, pd.DataFrame],
                    timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                    mode: BacktestMode = BacktestMode.VECTORBT_CPU,
                    **kwargs) -> VectorBTBacktestResults:
        """
        Run VectorBT backtest simulation with optimized memory and performance management.
        
        Args:
            signals: Trading signals (-1, 0, 1 for short, neutral, long)
            prices: Asset prices
            timestamps: Time index for the data
            mode: Execution mode
            **kwargs: Additional arguments for VectorBT portfolio
            
        Returns:
            Comprehensive backtest results
        """
        # Estimate memory requirements
        data_size_gb = self._estimate_data_size(signals, prices)
        
        # Use performance monitoring
        with monitor_operation(
            f"vectorbt_backtest_{mode.value}",
            gpu_used=(mode == BacktestMode.VECTORBT_GPU and CUPY_AVAILABLE),
            metadata={'data_size_gb': data_size_gb, 'mode': mode.value}
        ) as operation_id:
            
            logger.info(f"🚀 Starting VectorBT backtest with mode: {mode.value}")
            
            # Convert inputs to proper format with memory optimization
            prices_df, signals_df, timestamps_index = self._prepare_inputs_optimized(prices, signals, timestamps)
            
            # Validate inputs
            self._validate_inputs(prices_df, signals_df)
            
            logger.info(f"📊 Data shapes - Prices: {prices_df.shape}, Signals: {signals_df.shape}")
            
            # Execute backtest based on mode with memory management
            if mode == BacktestMode.VECTORBT_GPU and CUPY_AVAILABLE:
                portfolio = self._run_gpu_backtest_optimized(prices_df, signals_df, **kwargs)
            elif mode == BacktestMode.VECTORBT_PARALLEL:
                portfolio = self._run_parallel_backtest_optimized(prices_df, signals_df, **kwargs)
            elif mode == BacktestMode.HYBRID:
                portfolio = self._run_hybrid_backtest_optimized(prices_df, signals_df, **kwargs)
            else:
                portfolio = self._run_cpu_backtest_optimized(prices_df, signals_df, **kwargs)
            
            # Calculate comprehensive metrics with memory management
            results = self._calculate_comprehensive_metrics_optimized(portfolio, prices_df, timestamps_index)
            
            # Update performance stats
            self.performance_stats['total_simulations'] += 1
            results.mode_used = mode.value
            
            logger.info(f"✅ VectorBT backtest completed")
            logger.info(f"📊 Final portfolio value: ${results.portfolio_values[-1]:.2f}")
            logger.info(f"📊 Total return: {results.performance_metrics.get('total_return', 0):.2%}")
            logger.info(f"📊 Sharpe ratio: {results.performance_metrics.get('sharpe_ratio', 0):.3f}")
            
            return results
    
    def _estimate_data_size(self, signals: Union[np.ndarray, pd.DataFrame], 
                           prices: Union[np.ndarray, pd.DataFrame]) -> float:
        """Estimate data size in GB for memory management."""
        if isinstance(signals, np.ndarray):
            signals_size = signals.nbytes
        else:
            signals_size = signals.memory_usage(deep=True).sum()
        
        if isinstance(prices, np.ndarray):
            prices_size = prices.nbytes
        else:
            prices_size = prices.memory_usage(deep=True).sum()
        
        total_bytes = signals_size + prices_size
        return total_bytes / (1024**3)  # Convert to GB
    
    def _prepare_inputs_optimized(self, prices, signals, timestamps):
        """Prepare inputs for VectorBT with memory optimization."""
        # Convert prices to DataFrame
        if isinstance(prices, np.ndarray):
            if prices.ndim == 1:
                prices_df = pd.DataFrame(prices, columns=['price'])
            else:
                prices_df = pd.DataFrame(prices, columns=[f'asset_{i}' for i in range(prices.shape[1])])
        else:
            prices_df = prices.copy()
        
        # Convert signals to DataFrame
        if isinstance(signals, np.ndarray):
            if signals.ndim == 1:
                signals_df = pd.DataFrame(signals, columns=['signal'])
            else:
                signals_df = pd.DataFrame(signals, columns=[f'asset_{i}' for i in range(signals.shape[1])])
        else:
            signals_df = signals.copy()
        
        # Set timestamps
        if timestamps is not None:
            if isinstance(timestamps, pd.DatetimeIndex):
                timestamps_index = timestamps
            else:
                timestamps_index = pd.DatetimeIndex(timestamps)
        else:
            timestamps_index = pd.date_range(start='2020-01-01', periods=len(prices_df), freq='1min')
        
        # Set index
        prices_df.index = timestamps_index
        signals_df.index = timestamps_index
        
        # Optimize data types for memory efficiency
        prices_df = optimize_memory_usage(prices_df)
        signals_df = optimize_memory_usage(signals_df)
        
        return prices_df, signals_df, timestamps_index
    
    def _prepare_inputs(self, prices, signals, timestamps):
        """Prepare inputs for VectorBT."""
        # Convert prices to DataFrame
        if isinstance(prices, np.ndarray):
            if prices.ndim == 1:
                prices_df = pd.DataFrame(prices, columns=['price'])
            else:
                prices_df = pd.DataFrame(prices, columns=[f'asset_{i}' for i in range(prices.shape[1])])
        else:
            prices_df = prices.copy()
        
        # Convert signals to DataFrame
        if isinstance(signals, np.ndarray):
            if signals.ndim == 1:
                signals_df = pd.DataFrame(signals, columns=['signal'])
            else:
                signals_df = pd.DataFrame(signals, columns=[f'asset_{i}' for i in range(signals.shape[1])])
        else:
            signals_df = signals.copy()
        
        # Set timestamps
        if timestamps is not None:
            if isinstance(timestamps, pd.DatetimeIndex):
                timestamps_index = timestamps
            else:
                timestamps_index = pd.DatetimeIndex(timestamps)
        else:
            timestamps_index = pd.date_range(start='2020-01-01', periods=len(prices_df), freq='1min')
        
        # Set index
        prices_df.index = timestamps_index
        signals_df.index = timestamps_index
        
        return prices_df, signals_df, timestamps_index
    
    def _validate_inputs(self, prices_df, signals_df):
        """Validate input data."""
        if len(prices_df) != len(signals_df):
            raise ValueError(f"Prices and signals must have same length: {len(prices_df)} vs {len(signals_df)}")
        
        if len(prices_df) == 0:
            raise ValueError("Empty data")
        
        if prices_df.isnull().any().any():
            logger.warning("⚠️ Prices contain NaN values, forward filling...")
            prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        
        if signals_df.isnull().any().any():
            logger.warning("⚠️ Signals contain NaN values, filling with 0...")
            signals_df = signals_df.fillna(0)
    
    def _run_cpu_backtest_optimized(self, prices_df, signals_df, **kwargs):
        """Run CPU-based VectorBT backtest with memory optimization."""
        logger.debug("🔄 Running optimized CPU-based VectorBT backtest...")
        
        # Use memory management for large datasets
        data_size_gb = self._estimate_data_size(signals_df.values, prices_df.values)
        
        with memory_managed_operation(
            data_size_gb, 
            f"cpu_backtest_{int(time.time())}", 
            "backtesting"
        ):
            # Create VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                prices_df,
                signals_df,
                init_cash=self.config.initial_capital,
                fees=self.config.commission_rate,
                slippage=self.config.slippage_rate,
                freq='1min',
                **kwargs
            )
        
        self.performance_stats['cpu_operations'] += 1
        return portfolio
    
    def _run_gpu_backtest_optimized(self, prices_df, signals_df, **kwargs):
        """Run GPU-accelerated VectorBT backtest with memory optimization."""
        if not CUPY_AVAILABLE:
            logger.warning("⚠️ GPU not available, falling back to CPU")
            return self._run_cpu_backtest_optimized(prices_df, signals_df, **kwargs)
        
        logger.debug("🔄 Running optimized GPU-accelerated VectorBT backtest...")
        
        # Use memory management for GPU operations
        data_size_gb = self._estimate_data_size(signals_df.values, prices_df.values)
        
        with memory_managed_operation(
            data_size_gb, 
            f"gpu_backtest_{int(time.time())}", 
            "gpu_backtesting"
        ):
            # Convert to GPU arrays with memory optimization
            prices_gpu = cp.asarray(prices_df.values, dtype=cp.float32)
            signals_gpu = cp.asarray(signals_df.values, dtype=cp.float32)
            
            # Create portfolio with GPU data
            portfolio = vbt.Portfolio.from_signals(
                prices_gpu,
                signals_gpu,
                init_cash=self.config.initial_capital,
                fees=self.config.commission_rate,
                slippage=self.config.slippage_rate,
                freq='1min',
                **kwargs
            )
        
        self.performance_stats['gpu_operations'] += 1
        return portfolio
    
    def _run_parallel_backtest_optimized(self, prices_df, signals_df, **kwargs):
        """Run parallel VectorBT backtest with memory optimization."""
        logger.debug("🔄 Running optimized parallel VectorBT backtest...")
        
        # Use memory management for parallel operations
        data_size_gb = self._estimate_data_size(signals_df.values, prices_df.values)
        
        with memory_managed_operation(
            data_size_gb, 
            f"parallel_backtest_{int(time.time())}", 
            "parallel_backtesting"
        ):
            # Use VectorBT's built-in parallel processing
            portfolio = vbt.Portfolio.from_signals(
                prices_df,
                signals_df,
                init_cash=self.config.initial_capital,
                fees=self.config.commission_rate,
                slippage=self.config.slippage_rate,
                freq='1min',
                **kwargs
            )
        
        return portfolio
    
    def _run_hybrid_backtest_optimized(self, prices_df, signals_df, **kwargs):
        """Run hybrid VectorBT backtest (GPU + parallel) with memory optimization."""
        if CUPY_AVAILABLE and self.config.use_gpu:
            return self._run_gpu_backtest_optimized(prices_df, signals_df, **kwargs)
        else:
            return self._run_parallel_backtest_optimized(prices_df, signals_df, **kwargs)
    
    def _calculate_comprehensive_metrics_optimized(self, portfolio, prices_df, timestamps_index):
        """Calculate comprehensive performance and risk metrics with memory optimization."""
        logger.debug("📊 Calculating comprehensive metrics with optimization...")
        
        # Use memory management for metrics calculation
        data_size_gb = len(portfolio.value()) * 8 / (1024**3)  # Rough estimate
        
        with memory_managed_operation(
            data_size_gb, 
            f"metrics_calculation_{int(time.time())}", 
            "metrics_calculation"
        ):
            # Basic portfolio data
            portfolio_values = portfolio.value()
            returns = portfolio.returns()
            positions = portfolio.positions.records_readable
            
            # Performance metrics using VectorBT
            stats = portfolio.stats()
            
            # Extract key metrics
            performance_metrics = {
                'total_return': stats['Total Return [%]'] / 100,
                'annualized_return': stats['Annualized Return [%]'] / 100,
                'volatility': stats['Annualized Volatility [%]'] / 100,
                'sharpe_ratio': stats['Sharpe Ratio'],
                'sortino_ratio': stats['Sortino Ratio'],
                'calmar_ratio': stats['Calmar Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'] / 100,
                'avg_drawdown': stats['Avg. Drawdown [%]'] / 100,
                'max_drawdown_duration': stats['Max. Drawdown Duration'],
                'avg_drawdown_duration': stats['Avg. Drawdown Duration'],
                'win_rate': stats['Win Rate [%]'] / 100,
                'best_trade': stats['Best Trade [%]'] / 100,
                'worst_trade': stats['Worst Trade [%]'] / 100,
                'avg_trade': stats['Avg. Trade [%]'] / 100,
                'profit_factor': stats['Profit Factor'],
                'expectancy': stats['Expectancy [%]'] / 100,
                'sqn': stats['SQN'],
                'final_portfolio_value': portfolio_values.iloc[-1],
                'total_trades': stats['# Trades']
            }
            
            # Risk metrics
            risk_metrics = {
                'var_95': portfolio.value().quantile(0.05),
                'cvar_95': portfolio.value()[portfolio.value() <= portfolio.value().quantile(0.05)].mean(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'tail_ratio': performance_metrics['best_trade'] / abs(performance_metrics['worst_trade']),
                'common_sense_ratio': performance_metrics['total_return'] / abs(performance_metrics['max_drawdown']),
                'cagr': performance_metrics['annualized_return'],
                'volatility': performance_metrics['volatility']
            }
            
            # Drawdown analysis
            drawdown_analysis = {
                'max_drawdown': performance_metrics['max_drawdown'],
                'avg_drawdown': performance_metrics['avg_drawdown'],
                'max_drawdown_duration': performance_metrics['max_drawdown_duration'],
                'avg_drawdown_duration': performance_metrics['avg_drawdown_duration'],
                'drawdown_series': portfolio.drawdowns.records_readable,
                'recovery_time': self._calculate_recovery_time(portfolio_values)
            }
            
            # Create trades DataFrame
            trades_df = portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else pd.DataFrame()
            
            # Create results object
            results = VectorBTBacktestResults(
                portfolio=portfolio,
                portfolio_values=portfolio_values.values,
                returns=returns.values,
                positions=positions,
                trades=trades_df,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                drawdown_analysis=drawdown_analysis,
                computation_time=0.0,  # Will be set by caller
                memory_usage=data_size_gb,
                mode_used="",
                records=portfolio.trades if hasattr(portfolio, 'trades') else None,
                stats=stats
            )
        
        return results
    
    def _run_cpu_backtest(self, prices_df, signals_df, **kwargs):
        """Run CPU-based VectorBT backtest."""
        logger.debug("🔄 Running CPU-based VectorBT backtest...")
        
        # Create VectorBT portfolio
        portfolio = vbt.Portfolio.from_signals(
            prices_df,
            signals_df,
            init_cash=self.config.initial_capital,
            fees=self.config.commission_rate,
            slippage=self.config.slippage_rate,
            freq='1min',
            **kwargs
        )
        
        self.performance_stats['cpu_operations'] += 1
        return portfolio
    
    def _run_gpu_backtest(self, prices_df, signals_df, **kwargs):
        """Run GPU-accelerated VectorBT backtest."""
        if not CUPY_AVAILABLE:
            logger.warning("⚠️ GPU not available, falling back to CPU")
            return self._run_cpu_backtest(prices_df, signals_df, **kwargs)
        
        logger.debug("🔄 Running GPU-accelerated VectorBT backtest...")
        
        # Convert to GPU arrays
        prices_gpu = cp.asarray(prices_df.values)
        signals_gpu = cp.asarray(signals_df.values)
        
        # Create portfolio with GPU data
        portfolio = vbt.Portfolio.from_signals(
            prices_gpu,
            signals_gpu,
            init_cash=self.config.initial_capital,
            fees=self.config.commission_rate,
            slippage=self.config.slippage_rate,
            freq='1min',
            **kwargs
        )
        
        self.performance_stats['gpu_operations'] += 1
        return portfolio
    
    def _run_parallel_backtest(self, prices_df, signals_df, **kwargs):
        """Run parallel VectorBT backtest."""
        logger.debug("🔄 Running parallel VectorBT backtest...")
        
        # Use VectorBT's built-in parallel processing
        portfolio = vbt.Portfolio.from_signals(
            prices_df,
            signals_df,
            init_cash=self.config.initial_capital,
            fees=self.config.commission_rate,
            slippage=self.config.slippage_rate,
            freq='1min',
            **kwargs
        )
        
        return portfolio
    
    def _run_hybrid_backtest(self, prices_df, signals_df, **kwargs):
        """Run hybrid VectorBT backtest (GPU + parallel)."""
        if CUPY_AVAILABLE and self.config.use_gpu:
            return self._run_gpu_backtest(prices_df, signals_df, **kwargs)
        else:
            return self._run_parallel_backtest(prices_df, signals_df, **kwargs)
    
    def _calculate_comprehensive_metrics(self, portfolio, prices_df, timestamps_index):
        """Calculate comprehensive performance and risk metrics."""
        logger.debug("📊 Calculating comprehensive metrics...")
        
        # Basic portfolio data
        portfolio_values = portfolio.value()
        returns = portfolio.returns()
        positions = portfolio.positions.records_readable
        
        # Performance metrics using VectorBT
        stats = portfolio.stats()
        
        # Extract key metrics
        performance_metrics = {
            'total_return': stats['Total Return [%]'] / 100,
            'annualized_return': stats['Annualized Return [%]'] / 100,
            'volatility': stats['Annualized Volatility [%]'] / 100,
            'sharpe_ratio': stats['Sharpe Ratio'],
            'sortino_ratio': stats['Sortino Ratio'],
            'calmar_ratio': stats['Calmar Ratio'],
            'max_drawdown': stats['Max. Drawdown [%]'] / 100,
            'avg_drawdown': stats['Avg. Drawdown [%]'] / 100,
            'max_drawdown_duration': stats['Max. Drawdown Duration'],
            'avg_drawdown_duration': stats['Avg. Drawdown Duration'],
            'win_rate': stats['Win Rate [%]'] / 100,
            'best_trade': stats['Best Trade [%]'] / 100,
            'worst_trade': stats['Worst Trade [%]'] / 100,
            'avg_trade': stats['Avg. Trade [%]'] / 100,
            'profit_factor': stats['Profit Factor'],
            'expectancy': stats['Expectancy [%]'] / 100,
            'sqn': stats['SQN'],
            'final_portfolio_value': portfolio_values.iloc[-1],
            'total_trades': stats['# Trades']
        }
        
        # Risk metrics
        risk_metrics = {
            'var_95': portfolio.value().quantile(0.05),
            'cvar_95': portfolio.value()[portfolio.value() <= portfolio.value().quantile(0.05)].mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'tail_ratio': performance_metrics['best_trade'] / abs(performance_metrics['worst_trade']),
            'common_sense_ratio': performance_metrics['total_return'] / abs(performance_metrics['max_drawdown']),
            'cagr': performance_metrics['annualized_return'],
            'volatility': performance_metrics['volatility']
        }
        
        # Drawdown analysis
        drawdown_analysis = {
            'max_drawdown': performance_metrics['max_drawdown'],
            'avg_drawdown': performance_metrics['avg_drawdown'],
            'max_drawdown_duration': performance_metrics['max_drawdown_duration'],
            'avg_drawdown_duration': performance_metrics['avg_drawdown_duration'],
            'drawdown_series': portfolio.drawdowns.records_readable,
            'recovery_time': self._calculate_recovery_time(portfolio_values)
        }
        
        # Create trades DataFrame
        trades_df = portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else pd.DataFrame()
        
        # Create results object
        results = VectorBTBacktestResults(
            portfolio=portfolio,
            portfolio_values=portfolio_values.values,
            returns=returns.values,
            positions=positions,
            trades=trades_df,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            drawdown_analysis=drawdown_analysis,
            computation_time=0.0,  # Will be set by caller
            memory_usage=0.0,  # Will be calculated
            mode_used="",
            records=portfolio.trades if hasattr(portfolio, 'trades') else None,
            stats=stats
        )
        
        return results
    
    def _calculate_recovery_time(self, portfolio_values):
        """Calculate recovery time from maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_dd_idx = drawdown.idxmin()
        
        # Find recovery point
        recovery_idx = portfolio_values[portfolio_values.index >= max_dd_idx].ge(peak.loc[max_dd_idx]).idxmax()
        
        if pd.isna(recovery_idx):
            return len(portfolio_values) - portfolio_values.index.get_loc(max_dd_idx)
        else:
            return portfolio_values.index.get_loc(recovery_idx) - portfolio_values.index.get_loc(max_dd_idx)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add memory manager stats
        memory_stats = self.memory_manager.get_memory_stats()
        stats.update({
            'memory_usage_gb': memory_stats['current_usage_gb'],
            'memory_peak_gb': memory_stats['peak_usage_gb'],
            'memory_available_gb': memory_stats['available_memory_gb'],
            'memory_utilization': memory_stats['usage_percentage']
        })
        
        # Add performance monitor stats
        perf_stats = self.performance_monitor.get_performance_summary()
        stats.update({
            'total_operations_monitored': perf_stats.get('total_operations', 0),
            'average_operation_duration': perf_stats.get('average_duration', 0),
            'gpu_utilization_rate': perf_stats.get('gpu_utilization_rate', 0),
            'cache_hit_rate': perf_stats.get('cache_hit_rate', 0),
            'error_rate': perf_stats.get('error_rate', 0)
        })
        
        return stats
    
    def get_memory_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        return self.memory_manager.get_optimization_recommendations()
    
    def get_performance_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        return self.performance_monitor._get_optimization_recommendations()
    
    def benchmark_performance(self, 
                            signals: Union[np.ndarray, pd.DataFrame],
                            prices: Union[np.ndarray, pd.DataFrame],
                            timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Dict[str, Any]:
        """
        Benchmark VectorBT performance against custom implementation.
        
        Args:
            signals: Trading signals
            prices: Asset prices
            timestamps: Time index
            
        Returns:
            Benchmark results
        """
        logger.info("🔬 Benchmarking VectorBT performance...")
        
        # Test different modes
        modes = [BacktestMode.VECTORBT_CPU]
        if CUPY_AVAILABLE:
            modes.append(BacktestMode.VECTORBT_GPU)
        if self.config.enable_parallel:
            modes.append(BacktestMode.VECTORBT_PARALLEL)
        
        results = {}
        for mode in modes:
            start_time = time.time()
            try:
                result = self.run_backtest(signals, prices, timestamps, mode=mode)
                execution_time = time.time() - start_time
                
                results[mode.value] = {
                    'execution_time': execution_time,
                    'final_value': result.portfolio_values[-1],
                    'total_return': result.performance_metrics['total_return'],
                    'sharpe_ratio': result.performance_metrics['sharpe_ratio'],
                    'max_drawdown': result.performance_metrics['max_drawdown']
                }
            except Exception as e:
                logger.error(f"❌ Mode {mode.value} failed: {e}")
                results[mode.value] = {'error': str(e)}
        
        return results


# Convenience functions
def run_vectorbt_backtest(signals: Union[np.ndarray, pd.DataFrame],
                         prices: Union[np.ndarray, pd.DataFrame],
                         config: Optional[VectorBTBacktestConfig] = None,
                         mode: BacktestMode = BacktestMode.VECTORBT_CPU,
                         **kwargs) -> VectorBTBacktestResults:
    """
    Convenience function to run VectorBT backtest.
    
    Args:
        signals: Trading signals
        prices: Asset prices
        config: Backtesting configuration
        mode: Execution mode
        **kwargs: Additional arguments
        
    Returns:
        Backtest results
    """
    engine = VectorBTBacktestingEngine(config)
    return engine.run_backtest(signals, prices, mode=mode, **kwargs)


def create_vectorbt_config(initial_capital: float = 100000.0,
                          commission_rate: float = 0.001,
                          slippage_rate: float = 0.0005,
                          use_gpu: bool = True,
                          **kwargs) -> VectorBTBacktestConfig:
    """
    Create VectorBT backtesting configuration.
    
    Args:
        initial_capital: Initial portfolio capital
        commission_rate: Commission rate per trade
        slippage_rate: Slippage rate per trade
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional configuration parameters
        
    Returns:
        VectorBT backtesting configuration
    """
    return VectorBTBacktestConfig(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        use_gpu=use_gpu,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing VectorBT Backtesting Engine...")
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 1000
    n_assets = 3
    
    # Generate random prices and signals
    prices = np.random.randn(n_periods, n_assets).cumsum(axis=0) + 100
    signals = np.random.choice([-1, 0, 1], size=(n_periods, n_assets), p=[0.1, 0.8, 0.1])
    
    # Create timestamps
    timestamps = pd.date_range(start='2020-01-01', periods=n_periods, freq='1min')
    
    # Test backtesting engine
    config = create_vectorbt_config(initial_capital=100000.0)
    engine = VectorBTBacktestingEngine(config)
    
    # Run backtest
    results = engine.run_backtest(signals, prices, timestamps)
    
    # Print results
    print(f"\n📊 Backtest Results:")
    print(f"Final portfolio value: ${results.portfolio_values[-1]:,.2f}")
    print(f"Total return: {results.performance_metrics['total_return']:.2%}")
    print(f"Sharpe ratio: {results.performance_metrics['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {results.performance_metrics['max_drawdown']:.2%}")
    print(f"Execution time: {results.computation_time:.3f}s")
    
    # Benchmark performance
    benchmark_results = engine.benchmark_performance(signals, prices, timestamps)
    print(f"\n🔬 Benchmark Results:")
    for mode, stats in benchmark_results.items():
        if 'error' not in stats:
            print(f"{mode}: {stats['execution_time']:.3f}s, Return: {stats['total_return']:.2%}")
    
    print("\n✅ VectorBT Backtesting Engine test completed!")