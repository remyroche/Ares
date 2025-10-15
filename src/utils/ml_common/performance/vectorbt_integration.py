"""
VectorBT Integration for Performance Optimization

This module provides VectorBT-specific performance optimizations,
integrating with the VectorBTRollingOptimizer and other performance modules.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import weakref

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None

# Import VectorBT rolling optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

# Import performance modules
try:
    from .caching_strategies import get_ml_common_cache, CacheConfig, CacheStrategy
    from .memory_profiler import get_memory_profiler, MemoryOptimizationConfig
    from .async_patterns import get_async_operation_manager, AsyncOperationType
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorBTOperationType(Enum):
    """Types of VectorBT operations."""
    ROLLING = "rolling"
    FINANCIAL_METRICS = "financial_metrics"
    BACKTESTING = "backtesting"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    CROSS_VALIDATION = "cross_validation"
    FEATURE_ENGINEERING = "feature_engineering"

@dataclass
class VectorBTPerformanceConfig:
    """Configuration for VectorBT performance optimization."""
    
    # Basic settings
    enable_vectorbt: bool = True
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    enable_async_processing: bool = True
    
    # VectorBT settings
    use_rolling_optimizer: bool = True
    optimize_dataframes: bool = True
    enable_gpu_acceleration: bool = False
    
    # Caching settings
    cache_rolling_operations: bool = True
    cache_financial_metrics: bool = True
    cache_backtesting_results: bool = True
    cache_ttl_seconds: int = 3600
    
    # Memory settings
    enable_memory_profiling: bool = True
    memory_threshold_mb: float = 1000.0
    chunk_size: int = 10000
    
    # Async settings
    enable_async_operations: bool = True
    max_concurrent_operations: int = 4
    operation_timeout: float = 30.0

class VectorBTPerformanceOptimizer:
    """VectorBT performance optimizer with caching and memory management."""
    
    def __init__(self, config: Optional[VectorBTPerformanceConfig] = None):
        self.config = config or VectorBTPerformanceConfig()
        self.logger = logger.getChild('VectorBTPerformanceOptimizer')
        
        # Initialize VectorBT rolling optimizer
        self._rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.config.use_rolling_optimizer:
            self._rolling_optimizer = get_vectorbt_rolling_optimizer()
        
        # Initialize performance modules
        self._cache = None
        self._memory_profiler = None
        self._async_manager = None
        
        if PERFORMANCE_MODULES_AVAILABLE:
            if self.config.enable_caching:
                cache_config = CacheConfig(
                    strategy=CacheStrategy.VECTORBT,
                    enable_vectorbt_optimizations=True,
                    ttl_seconds=self.config.cache_ttl_seconds
                )
                self._cache = get_ml_common_cache(cache_config)
            
            if self.config.enable_memory_optimization:
                memory_config = MemoryOptimizationConfig(
                    enable_vectorbt_optimizations=True,
                    enable_memory_profiling=self.config.enable_memory_profiling
                )
                self._memory_profiler = get_memory_profiler(memory_config)
            
            if self.config.enable_async_processing:
                from .async_patterns import AsyncConfig
                async_config = AsyncConfig(
                    enable_vectorbt_optimizations=True,
                    max_concurrent_operations=self.config.max_concurrent_operations
                )
                self._async_manager = get_async_operation_manager(async_config)
    
    async def initialize(self):
        """Initialize the VectorBT performance optimizer."""
        if self._cache:
            await self._cache.initialize()
        
        if self._async_manager:
            await self._async_manager.initialize()
        
        self.logger.info("VectorBT performance optimizer initialized")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for VectorBT operations."""
        if not VECTORBT_AVAILABLE or not PANDAS_AVAILABLE:
            return df
        
        try:
            # Use VectorBT rolling optimizer if available
            if self._rolling_optimizer:
                return self._rolling_optimizer.optimize_dataframe(df)
            
            # Basic optimizations
            # Ensure numeric types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
            
            # Ensure proper indexing
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'date' in df.columns:
                    df = df.set_index('date')
            
            return df
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return df
    
    async def rolling_operation(
        self,
        data: pd.DataFrame,
        operation: str,
        window: int,
        **kwargs
    ) -> pd.DataFrame:
        """Perform optimized rolling operation with caching."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for rolling operations")
        
        # Generate cache key
        cache_key = f"rolling_{operation}_{window}_{hash(str(data.shape))}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for rolling {operation}")
                return cached_result
        
        # Optimize data
        optimized_data = self.optimize_dataframe(data)
        
        # Perform rolling operation
        if operation == 'mean':
            result = rolling_mean(optimized_data, window, **kwargs)
        elif operation == 'std':
            result = rolling_std(optimized_data, window, **kwargs)
        elif operation == 'var':
            result = rolling_var(optimized_data, window, **kwargs)
        elif operation == 'min':
            result = rolling_min(optimized_data, window, **kwargs)
        elif operation == 'max':
            result = rolling_max(optimized_data, window, **kwargs)
        elif operation == 'sum':
            result = rolling_sum(optimized_data, window, **kwargs)
        elif operation == 'corr':
            result = rolling_corr(optimized_data, window, **kwargs)
        elif operation == 'cov':
            result = rolling_cov(optimized_data, window, **kwargs)
        elif operation == 'quantile':
            result = rolling_quantile(optimized_data, window, **kwargs)
        elif operation == 'skew':
            result = rolling_skew(optimized_data, window, **kwargs)
        elif operation == 'kurt':
            result = rolling_kurt(optimized_data, window, **kwargs)
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}")
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def financial_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate financial metrics with caching."""
        # Generate cache key
        cache_key = f"financial_metrics_{hash(str(portfolio_values.shape))}_{hash(str(returns.shape))}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug("Cache hit for financial metrics")
                return cached_result
        
        # Calculate metrics using VectorBT
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for financial metrics")
        
        try:
            # Basic return metrics
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown metrics
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Additional metrics
            win_rate = (returns > 0).mean()
            profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else float('inf')
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(returns),
                'positive_trades': (returns > 0).sum(),
                'negative_trades': (returns < 0).sum()
            }
            
            # Benchmark comparison if provided
            if benchmark_values is not None:
                benchmark_returns = benchmark_values.pct_change().dropna()
                benchmark_total_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                benchmark_sharpe = (benchmark_total_return * 252) / benchmark_volatility if benchmark_volatility > 0 else 0
                
                metrics.update({
                    'benchmark_total_return': benchmark_total_return,
                    'benchmark_volatility': benchmark_volatility,
                    'benchmark_sharpe_ratio': benchmark_sharpe,
                    'excess_return': annualized_return - (benchmark_total_return * 252),
                    'information_ratio': (annualized_return - (benchmark_total_return * 252)) / volatility if volatility > 0 else 0
                })
            
            # Cache result
            if self._cache:
                await self._cache.set(cache_key, metrics, ttl=self.config.cache_ttl_seconds)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Financial metrics calculation failed: {e}")
            raise
    
    async def backtesting_operation(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform backtesting with VectorBT optimization."""
        # Generate cache key
        cache_key = f"backtesting_{hash(str(data.shape))}_{strategy_func.__name__}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug("Cache hit for backtesting")
                return cached_result
        
        # Optimize data
        optimized_data = self.optimize_dataframe(data)
        
        # Execute strategy
        if self._async_manager:
            result = await self._async_manager.execute_async(
                strategy_func,
                optimized_data,
                operation_type=AsyncOperationType.VECTORBT_OPERATION,
                **kwargs
            )
        else:
            result = strategy_func(optimized_data, **kwargs)
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def portfolio_optimization(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform portfolio optimization with VectorBT."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for portfolio optimization")
        
        # Generate cache key
        cache_key = f"portfolio_opt_{hash(str(returns.shape))}_{risk_free_rate}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug("Cache hit for portfolio optimization")
                return cached_result
        
        try:
            # Basic portfolio optimization
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Calculate optimal weights (simplified Markowitz optimization)
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(expected_returns))
            
            # Optimal weights for maximum Sharpe ratio
            numerator = inv_cov @ (expected_returns - risk_free_rate)
            denominator = ones.T @ inv_cov @ (expected_returns - risk_free_rate)
            optimal_weights = numerator / denominator
            
            # Calculate portfolio metrics
            portfolio_return = optimal_weights.T @ expected_returns
            portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            result = {
                'optimal_weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_free_rate': risk_free_rate
            }
            
            # Cache result
            if self._cache:
                await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            raise
    
    async def cross_validation_operation(
        self,
        data: pd.DataFrame,
        model_func: Callable,
        cv_folds: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform cross-validation with VectorBT optimization."""
        # Generate cache key
        cache_key = f"cv_{hash(str(data.shape))}_{model_func.__name__}_{cv_folds}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug("Cache hit for cross-validation")
                return cached_result
        
        # Optimize data
        optimized_data = self.optimize_dataframe(data)
        
        # Perform cross-validation
        fold_size = len(optimized_data) // cv_folds
        cv_results = []
        
        for i in range(cv_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < cv_folds - 1 else len(optimized_data)
            
            # Split data
            train_data = optimized_data.iloc[:start_idx]
            test_data = optimized_data.iloc[start_idx:end_idx]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Train and evaluate model
            if self._async_manager:
                fold_result = await self._async_manager.execute_async(
                    model_func,
                    train_data,
                    test_data,
                    operation_type=AsyncOperationType.VECTORBT_OPERATION,
                    **kwargs
                )
            else:
                fold_result = model_func(train_data, test_data, **kwargs)
            
            cv_results.append(fold_result)
        
        # Aggregate results
        if cv_results:
            result = {
                'cv_folds': cv_folds,
                'fold_results': cv_results,
                'mean_score': np.mean([r.get('score', 0) for r in cv_results if isinstance(r, dict)]),
                'std_score': np.std([r.get('score', 0) for r in cv_results if isinstance(r, dict)])
            }
        else:
            result = {
                'cv_folds': cv_folds,
                'fold_results': [],
                'mean_score': 0.0,
                'std_score': 0.0
            }
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def feature_engineering(
        self,
        data: pd.DataFrame,
        feature_funcs: List[Callable],
        **kwargs
    ) -> pd.DataFrame:
        """Perform feature engineering with VectorBT optimization."""
        # Generate cache key
        cache_key = f"features_{hash(str(data.shape))}_{len(feature_funcs)}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug("Cache hit for feature engineering")
                return cached_result
        
        # Optimize data
        optimized_data = self.optimize_dataframe(data)
        
        # Apply feature functions
        result_data = optimized_data.copy()
        
        for feature_func in feature_funcs:
            try:
                if self._async_manager:
                    features = await self._async_manager.execute_async(
                        feature_func,
                        result_data,
                        operation_type=AsyncOperationType.VECTORBT_OPERATION,
                        **kwargs
                    )
                else:
                    features = feature_func(result_data, **kwargs)
                
                # Add features to result
                if isinstance(features, pd.DataFrame):
                    result_data = pd.concat([result_data, features], axis=1)
                elif isinstance(features, pd.Series):
                    result_data[features.name or f"feature_{len(result_data.columns)}"] = features
                
            except Exception as e:
                self.logger.warning(f"Feature function {feature_func.__name__} failed: {e}")
                continue
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result_data, ttl=self.config.cache_ttl_seconds)
        
        return result_data

# Global VectorBT performance optimizer
_global_vectorbt_optimizer: Optional[VectorBTPerformanceOptimizer] = None

def get_vectorbt_performance_optimizer(config: Optional[VectorBTPerformanceConfig] = None) -> VectorBTPerformanceOptimizer:
    """Get the global VectorBT performance optimizer."""
    global _global_vectorbt_optimizer
    
    if _global_vectorbt_optimizer is None:
        _global_vectorbt_optimizer = VectorBTPerformanceOptimizer(config)
    
    return _global_vectorbt_optimizer

def vectorbt_optimize(data: pd.DataFrame) -> pd.DataFrame:
    """Optimize data for VectorBT operations."""
    optimizer = get_vectorbt_performance_optimizer()
    return optimizer.optimize_dataframe(data)

def vectorbt_cached(operation_type: VectorBTOperationType = VectorBTOperationType.ROLLING):
    """Decorator for VectorBT operations with caching."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_vectorbt_performance_optimizer()
            await optimizer.initialize()
            
            # Generate cache key
            cache_key = f"vectorbt_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check cache first
            if optimizer._cache:
                cached_result = await optimizer._cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            if optimizer._cache:
                await optimizer._cache.set(cache_key, result, ttl=optimizer.config.cache_ttl_seconds)
            
            return result
        
        return async_wrapper
    return decorator

def vectorbt_async_execute(
    operation_type: VectorBTOperationType = VectorBTOperationType.ROLLING,
    timeout: float = 30.0
):
    """Decorator for async VectorBT operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_vectorbt_performance_optimizer()
            await optimizer.initialize()
            
            if optimizer._async_manager:
                return await optimizer._async_manager.execute_async(
                    func,
                    *args,
                    operation_type=AsyncOperationType.VECTORBT_OPERATION,
                    timeout=timeout,
                    **kwargs
                )
            else:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return async_wrapper
    return decorator

# Convenience functions
async def vectorbt_rolling_mean(data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Optimized rolling mean with VectorBT."""
    optimizer = get_vectorbt_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.rolling_operation(data, 'mean', window, **kwargs)

async def vectorbt_rolling_std(data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Optimized rolling standard deviation with VectorBT."""
    optimizer = get_vectorbt_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.rolling_operation(data, 'std', window, **kwargs)

async def vectorbt_financial_metrics(
    portfolio_values: pd.Series,
    returns: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    **kwargs
) -> Dict[str, Any]:
    """Calculate financial metrics with VectorBT optimization."""
    optimizer = get_vectorbt_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.financial_metrics(portfolio_values, returns, benchmark_values, **kwargs)

async def vectorbt_backtesting(
    data: pd.DataFrame,
    strategy_func: Callable,
    **kwargs
) -> Dict[str, Any]:
    """Perform backtesting with VectorBT optimization."""
    optimizer = get_vectorbt_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.backtesting_operation(data, strategy_func, **kwargs)