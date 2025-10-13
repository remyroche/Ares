"""
Unified Vectorization Manager for Backtesting

This module provides a centralized manager for all VectorBT operations across
the backtesting system, ensuring consistent optimization and performance.

Key Features:
- Centralized VectorBT configuration
- Unified rolling operations
- Performance monitoring and optimization
- Memory management
- GPU acceleration support
- Cross-component analytics
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
from pathlib import Path
import json

# VectorBT imports
from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
from src.feature_selection.vectorbt.vectorbt_unified_framework import VectorBTUnifiedFramework

# Common utilities
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import validate_finite, validate_positive
from src.core.decorators import handles_errors, traced, log_execution_time

logger = logging.getLogger(__name__)


class VectorBTOperationType(Enum):
    """Types of VectorBT operations."""
    ROLLING_STATISTICS = "rolling_statistics"
    CORRELATION_ANALYSIS = "correlation_analysis"
    RISK_METRICS = "risk_metrics"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    REGIME_ANALYSIS = "regime_analysis"
    BOOTSTRAP_SAMPLING = "bootstrap_sampling"
    FEATURE_SELECTION = "feature_selection"
    PARAMETER_OPTIMIZATION = "parameter_optimization"


@dataclass
class VectorBTConfig:
    """Configuration for VectorBT unified manager."""
    # Basic settings
    enable_parallel: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False
    chunk_size: int = 1000
    fast_fail: bool = False
    enable_logging: bool = True
    
    # Performance settings
    max_workers: int = 4
    memory_limit_mb: int = 1024
    cache_size: int = 100
    
    # Analytics settings
    enable_rolling_analytics: bool = True
    enable_cross_metric_analysis: bool = True
    enable_predictive_analytics: bool = True
    analytics_window_size: int = 20
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorBTOperationResult:
    """Result of a VectorBT operation."""
    operation_type: VectorBTOperationType
    success: bool
    result: Any
    execution_time: float
    memory_used_mb: float
    vectorbt_optimized: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorBTUnifiedManager:
    """
    Unified Vectorization Manager for centralized VectorBT operations.
    
    This manager provides:
    - Centralized VectorBT configuration
    - Unified rolling operations
    - Performance monitoring
    - Memory management
    - Cross-component analytics
    """
    
    def __init__(self, config: VectorBTConfig):
        """Initialize VectorBT unified manager."""
        self.config = config
        self.logger = logger.getChild('VectorBTUnifiedManager')
        
        # Initialize VectorBT components
        self.rolling_optimizer = VectorBTRollingOptimizer(
            enable_gpu=config.enable_gpu_acceleration,
            enable_parallel=config.enable_parallel,
            memory_efficient=config.enable_memory_optimization,
            chunk_size=config.chunk_size,
            fast_fail=config.fast_fail,
            enable_logging=config.enable_logging
        )
        
        self.unified_framework = VectorBTUnifiedFramework()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_time': 0.0,
            'total_memory_used': 0.0,
            'operations_by_type': {},
            'average_execution_time': 0.0,
            'memory_efficiency': 0.0
        }
        
        # Operation cache
        self.operation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Analytics storage
        self.analytics_data = {}
        self.cross_metric_correlations = {}
        
        self.logger.info("🚀 VectorBTUnifiedManager initialized")
        self.logger.info(f"📊 Parallel processing: {config.enable_parallel}")
        self.logger.info(f"📊 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"📊 GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"📊 Analytics enabled: {config.enable_rolling_analytics}")
    
    @traced(span_name='vectorbt_operation')
    async def execute_operation(self, operation_type: VectorBTOperationType, 
                              operation_func: Callable, *args, **kwargs) -> VectorBTOperationResult:
        """Execute a VectorBT operation with monitoring and optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            self.logger.debug(f"🔄 Executing {operation_type.value} operation")
            
            # Check cache first
            cache_key = self._generate_cache_key(operation_type, operation_func, args, kwargs)
            if cache_key in self.operation_cache:
                self.cache_hits += 1
                cached_result = self.operation_cache[cache_key]
                self.logger.debug(f"📋 Cache hit for {operation_type.value}")
                return cached_result
            
            self.cache_misses += 1
            
            # Execute operation
            result = await operation_func(*args, **kwargs)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_used = end_memory - start_memory
            
            # Create result
            operation_result = VectorBTOperationResult(
                operation_type=operation_type,
                success=True,
                result=result,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                vectorbt_optimized=True,
                metadata={
                    'cache_key': cache_key,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Cache result if appropriate
            if self._should_cache_result(operation_type, execution_time):
                self.operation_cache[cache_key] = operation_result
                if len(self.operation_cache) > self.config.cache_size:
                    self._evict_oldest_cache_entry()
            
            # Update performance stats
            self._update_performance_stats(operation_result)
            
            self.logger.debug(f"✅ {operation_type.value} completed in {execution_time:.3f}s")
            return operation_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_used = end_memory - start_memory
            
            self.logger.error(f"❌ {operation_type.value} operation failed: {e}")
            
            operation_result = VectorBTOperationResult(
                operation_type=operation_type,
                success=False,
                result=None,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                vectorbt_optimized=False,
                error=str(e),
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'error_type': type(e).__name__
                }
            )
            
            self._update_performance_stats(operation_result)
            return operation_result
    
    async def rolling_statistics(self, data: Union[pd.Series, pd.DataFrame], 
                               window: int, operations: List[str] = None) -> Dict[str, Any]:
        """Calculate rolling statistics using VectorBT with enhanced backtesting operations."""
        if operations is None:
            operations = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
        
        async def _calculate_rolling_stats():
            results = {}
            
            for op in operations:
                if hasattr(self.rolling_optimizer, f'rolling_{op}'):
                    func = getattr(self.rolling_optimizer, f'rolling_{op}')
                    results[op] = func(data, window=window)
                else:
                    self.logger.warning(f"⚠️ Rolling operation {op} not available")
            
            return results
        
        result = await self.execute_operation(
            VectorBTOperationType.ROLLING_STATISTICS,
            _calculate_rolling_stats
        )
        
        return result.result if result.success else {}
    
    async def backtesting_metrics(self, equity_curve: Union[pd.Series, np.ndarray], 
                                returns: Union[pd.Series, np.ndarray],
                                window: int = 20) -> Dict[str, Any]:
        """Calculate comprehensive backtesting metrics using VectorBT."""
        async def _calculate_backtesting_metrics():
            try:
                # Convert to pandas Series if needed
                if isinstance(equity_curve, np.ndarray):
                    equity_series = pd.Series(equity_curve)
                else:
                    equity_series = equity_curve
                
                if isinstance(returns, np.ndarray):
                    returns_series = pd.Series(returns)
                else:
                    returns_series = returns
                
                metrics = {}
                
                # Basic performance metrics
                total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
                metrics['total_return'] = float(total_return)
                metrics['annualized_return'] = float((1 + total_return) ** (252 / len(equity_series)) - 1)
                
                # Rolling volatility using VectorBT
                rolling_vol = self.rolling_optimizer.rolling_std(returns_series, window=window)
                metrics['volatility'] = float(rolling_vol.mean() * np.sqrt(252))
                metrics['rolling_volatility'] = float(rolling_vol.iloc[-1] * np.sqrt(252) if not rolling_vol.empty else 0)
                
                # Sharpe ratio
                if metrics['volatility'] > 0:
                    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
                else:
                    metrics['sharpe_ratio'] = 0.0
                
                # Rolling Sharpe ratio
                rolling_mean = self.rolling_optimizer.rolling_mean(returns_series, window=window)
                rolling_std = self.rolling_optimizer.rolling_std(returns_series, window=window)
                rolling_sharpe = rolling_mean / rolling_std
                metrics['rolling_sharpe_ratio'] = float(rolling_sharpe.mean())
                
                # Drawdown analysis using VectorBT
                rolling_max = self.rolling_optimizer.rolling_max(equity_series, window=window)
                rolling_drawdown = (equity_series - rolling_max) / rolling_max
                metrics['max_drawdown'] = float(rolling_drawdown.min())
                metrics['avg_drawdown'] = float(rolling_drawdown[rolling_drawdown < 0].mean())
                
                # Calmar ratio
                if metrics['max_drawdown'] != 0:
                    metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
                else:
                    metrics['calmar_ratio'] = 0.0
                
                # Sortino ratio
                downside_returns = returns_series[returns_series < 0]
                if len(downside_returns) > 0:
                    downside_std = self.rolling_optimizer.rolling_std(downside_returns, window=window)
                    metrics['sortino_ratio'] = float(rolling_mean.mean() / downside_std.mean() if downside_std.mean() > 0 else 0)
                else:
                    metrics['sortino_ratio'] = 0.0
                
                # VaR calculations
                var_95 = self.rolling_optimizer.rolling_quantile(returns_series, window=window, q=0.05)
                var_99 = self.rolling_optimizer.rolling_quantile(returns_series, window=window, q=0.01)
                metrics['var_95'] = float(var_95.mean())
                metrics['var_99'] = float(var_99.mean())
                
                # Tail risk metrics
                rolling_skew = self.rolling_optimizer.rolling_skew(returns_series, window=window)
                rolling_kurt = self.rolling_optimizer.rolling_kurt(returns_series, window=window)
                metrics['skewness'] = float(rolling_skew.mean())
                metrics['kurtosis'] = float(rolling_kurt.mean())
                
                # Trend analysis
                time_index = pd.Series(range(len(returns_series)), index=returns_series.index)
                trend_correlation = self.rolling_optimizer.rolling_corr(returns_series, time_index, window=window)
                metrics['trend_correlation'] = float(trend_correlation.mean())
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"❌ Backtesting metrics calculation failed: {e}")
                return {}
        
        result = await self.execute_operation(
            VectorBTOperationType.PERFORMANCE_ANALYSIS,
            _calculate_backtesting_metrics
        )
        
        return result.result if result.success else {}
    
    async def regime_analysis(self, returns: Union[pd.Series, np.ndarray], 
                            window: int = 20) -> Dict[str, Any]:
        """Perform regime analysis using VectorBT rolling operations."""
        async def _calculate_regime_analysis():
            try:
                if isinstance(returns, np.ndarray):
                    returns_series = pd.Series(returns)
                else:
                    returns_series = returns
                
                analysis = {}
                
                # Volatility regime analysis
                rolling_vol = self.rolling_optimizer.rolling_std(returns_series, window=window)
                vol_mean = rolling_vol.mean()
                vol_std = rolling_vol.std()
                
                # Classify volatility regimes
                high_vol_threshold = vol_mean + vol_std
                low_vol_threshold = vol_mean - vol_std
                
                high_vol_periods = rolling_vol > high_vol_threshold
                low_vol_periods = rolling_vol < low_vol_threshold
                normal_vol_periods = ~(high_vol_periods | low_vol_periods)
                
                analysis['volatility_regimes'] = {
                    'high_volatility_periods': int(high_vol_periods.sum()),
                    'low_volatility_periods': int(low_vol_periods.sum()),
                    'normal_volatility_periods': int(normal_vol_periods.sum()),
                    'high_vol_threshold': float(high_vol_threshold),
                    'low_vol_threshold': float(low_vol_threshold)
                }
                
                # Momentum regime analysis
                rolling_momentum = self.rolling_optimizer.rolling_mean(returns_series, window=window)
                momentum_mean = rolling_momentum.mean()
                momentum_std = rolling_momentum.std()
                
                high_momentum_threshold = momentum_mean + momentum_std
                low_momentum_threshold = momentum_mean - momentum_std
                
                high_momentum_periods = rolling_momentum > high_momentum_threshold
                low_momentum_periods = rolling_momentum < low_momentum_threshold
                normal_momentum_periods = ~(high_momentum_periods | low_momentum_periods)
                
                analysis['momentum_regimes'] = {
                    'high_momentum_periods': int(high_momentum_periods.sum()),
                    'low_momentum_periods': int(low_momentum_periods.sum()),
                    'normal_momentum_periods': int(normal_momentum_periods.sum()),
                    'high_momentum_threshold': float(high_momentum_threshold),
                    'low_momentum_threshold': float(low_momentum_threshold)
                }
                
                # Regime stability analysis
                vol_of_vol = self.rolling_optimizer.rolling_std(rolling_vol, window=min(10, len(rolling_vol)))
                momentum_of_momentum = self.rolling_optimizer.rolling_std(rolling_momentum, window=min(10, len(rolling_momentum)))
                
                analysis['regime_stability'] = {
                    'volatility_stability': float(1.0 / (1.0 + vol_of_vol.mean()) if not vol_of_vol.empty else 0),
                    'momentum_stability': float(1.0 / (1.0 + momentum_of_momentum.mean()) if not momentum_of_momentum.empty else 0)
                }
                
                return analysis
                
            except Exception as e:
                self.logger.error(f"❌ Regime analysis failed: {e}")
                return {}
        
        result = await self.execute_operation(
            VectorBTOperationType.REGIME_ANALYSIS,
            _calculate_regime_analysis
        )
        
        return result.result if result.success else {}
    
    async def portfolio_optimization_metrics(self, returns: Union[pd.Series, np.ndarray],
                                           benchmark_returns: Union[pd.Series, np.ndarray] = None,
                                           window: int = 20) -> Dict[str, Any]:
        """Calculate portfolio optimization metrics using VectorBT."""
        async def _calculate_portfolio_metrics():
            try:
                if isinstance(returns, np.ndarray):
                    returns_series = pd.Series(returns)
                else:
                    returns_series = returns
                
                metrics = {}
                
                # Basic portfolio metrics
                metrics['mean_return'] = float(returns_series.mean())
                metrics['volatility'] = float(returns_series.std())
                metrics['sharpe_ratio'] = float(metrics['mean_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0)
                
                # Rolling metrics
                rolling_mean = self.rolling_optimizer.rolling_mean(returns_series, window=window)
                rolling_std = self.rolling_optimizer.rolling_std(returns_series, window=window)
                rolling_sharpe = rolling_mean / rolling_std
                
                metrics['rolling_sharpe_mean'] = float(rolling_sharpe.mean())
                metrics['rolling_sharpe_std'] = float(rolling_sharpe.std())
                
                # Risk metrics
                var_95 = self.rolling_optimizer.rolling_quantile(returns_series, window=window, q=0.05)
                var_99 = self.rolling_optimizer.rolling_quantile(returns_series, window=window, q=0.01)
                
                metrics['var_95'] = float(var_95.mean())
                metrics['var_99'] = float(var_99.mean())
                
                # Expected Shortfall (Conditional VaR)
                es_95 = returns_series[returns_series <= var_95.mean()].mean()
                es_99 = returns_series[returns_series <= var_99.mean()].mean()
                
                metrics['expected_shortfall_95'] = float(es_95)
                metrics['expected_shortfall_99'] = float(es_99)
                
                # Benchmark comparison if provided
                if benchmark_returns is not None:
                    if isinstance(benchmark_returns, np.ndarray):
                        benchmark_series = pd.Series(benchmark_returns)
                    else:
                        benchmark_series = benchmark_returns
                    
                    # Ensure same length
                    min_len = min(len(returns_series), len(benchmark_series))
                    returns_aligned = returns_series.iloc[:min_len]
                    benchmark_aligned = benchmark_series.iloc[:min_len]
                    
                    # Calculate excess returns
                    excess_returns = returns_aligned - benchmark_aligned
                    
                    # Information ratio
                    metrics['information_ratio'] = float(excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0)
                    
                    # Rolling correlation with benchmark
                    rolling_corr = self.rolling_optimizer.rolling_corr(returns_aligned, benchmark_aligned, window=window)
                    metrics['rolling_correlation'] = float(rolling_corr.mean())
                    
                    # Beta calculation
                    covariance = self.rolling_optimizer.rolling_cov(returns_aligned, benchmark_aligned, window=window)
                    benchmark_variance = self.rolling_optimizer.rolling_var(benchmark_aligned, window=window)
                    rolling_beta = covariance / benchmark_variance
                    metrics['rolling_beta'] = float(rolling_beta.mean())
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"❌ Portfolio optimization metrics calculation failed: {e}")
                return {}
        
        result = await self.execute_operation(
            VectorBTOperationType.PORTFOLIO_OPTIMIZATION,
            _calculate_portfolio_metrics
        )
        
        return result.result if result.success else {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_operations']
            stats['memory_efficiency'] = stats['total_memory_used'] / stats['total_operations']
        
        stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        stats['cache_size'] = len(self.operation_cache)
        
        return stats
    
    def _generate_cache_key(self, operation_type: VectorBTOperationType, 
                          operation_func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for the operation."""
        # Simple cache key based on operation type and function name
        key_parts = [
            operation_type.value,
            operation_func.__name__,
            str(hash(str(args))),
            str(hash(str(sorted(kwargs.items()))))
        ]
        return "_".join(key_parts)
    
    def _should_cache_result(self, operation_type: VectorBTOperationType, 
                           execution_time: float) -> bool:
        """Determine if a result should be cached."""
        # Cache expensive operations
        expensive_operations = {
            VectorBTOperationType.BOOTSTRAP_SAMPLING,
            VectorBTOperationType.REGIME_ANALYSIS,
            VectorBTOperationType.CORRELATION_ANALYSIS
        }
        
        return (operation_type in expensive_operations or 
                execution_time > 1.0)  # Cache operations taking more than 1 second
    
    def _evict_oldest_cache_entry(self):
        """Evict the oldest cache entry."""
        if self.operation_cache:
            oldest_key = next(iter(self.operation_cache))
            del self.operation_cache[oldest_key]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _update_performance_stats(self, result: VectorBTOperationResult):
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_execution_time'] += result.execution_time
        self.performance_stats['total_memory_used'] += result.memory_used_mb
        
        if result.success:
            self.performance_stats['successful_operations'] += 1
        else:
            self.performance_stats['failed_operations'] += 1
        
        # Update operation type stats
        op_type = result.operation_type.value
        if op_type not in self.performance_stats['operations_by_type']:
            self.performance_stats['operations_by_type'][op_type] = 0
        self.performance_stats['operations_by_type'][op_type] += 1


# Global manager instance
_global_manager = None

def get_vectorbt_unified_manager(config: VectorBTConfig = None) -> VectorBTUnifiedManager:
    """Get the global VectorBT unified manager instance."""
    global _global_manager
    if _global_manager is None:
        if config is None:
            config = VectorBTConfig()
        _global_manager = VectorBTUnifiedManager(config)
    return _global_manager

def reset_vectorbt_unified_manager():
    """Reset the global VectorBT unified manager."""
    global _global_manager
    _global_manager = None
