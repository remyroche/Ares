"""
VectorBT Optimization Module for Profit Labeling

This module provides unified VectorBT optimization utilities for the profit labeling system,
offering significant performance improvements over pandas operations while maintaining
backward compatibility and robust error handling.

Key Features:
- Optimized rolling operations with automatic fallback
- Memory-efficient operations for large datasets
- Performance monitoring and statistics
- Unified configuration management
- Robust error handling and fallback mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import psutil
import warnings
from functools import wraps
from contextlib import contextmanager

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


@dataclass
class VectorBTConfig:
    """Configuration for VectorBT optimization."""
    
    # Core settings
    enable_vectorbt: bool = True
    vectorbt_threshold: int = 1000  # Minimum data size to use VectorBT
    enable_gpu_acceleration: bool = False
    memory_limit_gb: float = 8.0
    fallback_to_pandas: bool = True
    performance_monitoring: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_duration_seconds: int = 300
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    # Memory management
    memory_efficiency_mode: bool = True
    chunk_size: int = 10000
    enable_memory_monitoring: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 0.1
    silent_failures: bool = False


@dataclass
class PerformanceStats:
    """Performance statistics for VectorBT operations."""
    
    operation_name: str
    execution_time: float
    data_size: int
    memory_used_mb: float
    success: bool
    fallback_used: bool
    timestamp: float = field(default_factory=time.time)


class VectorBTOptimizer:
    """
    Unified VectorBT optimization manager for profit labeling operations.
    
    This class provides optimized implementations of common operations used in
    profit labeling, with automatic fallback to pandas when VectorBT is not
    available or fails.
    """
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        """Initialize VectorBT optimizer."""
        self.config = config or VectorBTConfig()
        self.logger = logging.getLogger('VectorBTOptimizer')
        self.performance_stats: List[PerformanceStats] = []
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # Initialize performance monitoring
        if self.config.performance_monitoring:
            self._setup_performance_monitoring()
        
        tprint_info("🚀 VectorBT Optimizer initialized")
        tprint_info(f"   → VectorBT available: {VECTORBT_AVAILABLE}")
        tprint_info(f"   → GPU acceleration: {CUPY_AVAILABLE and self.config.enable_gpu_acceleration}")
        tprint_info(f"   → Memory limit: {self.config.memory_limit_gb}GB")
        tprint_info(f"   → Threshold: {self.config.vectorbt_threshold} samples")
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring infrastructure."""
        try:
            # Monitor memory usage
            if self.config.enable_memory_monitoring:
                self._initial_memory = psutil.virtual_memory().used
            else:
                self._initial_memory = 0
        except Exception as e:
            tprint_warning(f"⚠️ Performance monitoring setup failed: {e}")
    
    def _should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame, np.ndarray]) -> bool:
        """Determine if VectorBT should be used based on data characteristics."""
        if not self.config.enable_vectorbt or not VECTORBT_AVAILABLE:
            tprint_info("🔧 VectorBT disabled or not available")
            return False
        
        # Check data size
        if hasattr(data, '__len__'):
            data_size = len(data)
        else:
            tprint_info("🔧 Data has no length attribute")
            return False
        
        if data_size < self.config.vectorbt_threshold:
            tprint_info(f"🔧 Data size {data_size} below VectorBT threshold {self.config.vectorbt_threshold}")
            return False
        
        # Check memory availability
        if self.config.memory_efficiency_mode:
            try:
                estimated_memory = data_size * 8 * 4  # Rough estimate in bytes
                available_memory = psutil.virtual_memory().available
                memory_limit = self.config.memory_limit_gb * 1024**3
                
                if estimated_memory > min(available_memory * 0.5, memory_limit):
                    tprint_info(f"🔧 Insufficient memory: estimated {estimated_memory/1024**2:.1f}MB > available {min(available_memory * 0.5, memory_limit)/1024**2:.1f}MB")
                    return False
            except Exception as e:
                tprint_warning(f"⚠️ Memory check failed: {e}")
                pass  # Continue if memory check fails
        
        tprint_info(f"✅ VectorBT recommended for data size {data_size}")
        return True
    
    def _check_memory_availability(self, data_size: int) -> bool:
        """Check if there's enough memory for VectorBT operations."""
        try:
            estimated_memory = data_size * 8 * 4  # Rough estimate
            available_memory = psutil.virtual_memory().available
            memory_limit = self.config.memory_limit_gb * 1024**3
            
            is_available = estimated_memory < min(available_memory * 0.5, memory_limit)
            tprint_info(f"🧠 Memory check: estimated {estimated_memory/1024**2:.1f}MB, available {min(available_memory * 0.5, memory_limit)/1024**2:.1f}MB, sufficient: {is_available}")
            return is_available
        except Exception as e:
            tprint_warning(f"⚠️ Memory availability check failed: {e}")
            return True  # Default to True if we can't check
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_usage = psutil.virtual_memory().used / 1024**2
            tprint_info(f"🧠 Current memory usage: {memory_usage:.1f}MB")
            return memory_usage
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get memory usage: {e}")
            return 0.0
    
    def _log_performance(self, operation_name: str, execution_time: float, 
                        data_size: int, success: bool, fallback_used: bool):
        """Log performance statistics."""
        if not self.config.performance_monitoring:
            return
        
        memory_used = self._get_memory_usage()
        
        stats = PerformanceStats(
            operation_name=operation_name,
            execution_time=execution_time,
            data_size=data_size,
            memory_used_mb=memory_used,
            success=success,
            fallback_used=fallback_used
        )
        
        self.performance_stats.append(stats)
        
        # Log performance info
        method = "VectorBT" if not fallback_used else "Pandas"
        tprint_info(f"⚡ {operation_name}: {execution_time:.3f}s ({method}, {data_size} samples, {memory_used:.1f}MB)")
        
        # Keep only recent stats (last 1000 entries)
        if len(self.performance_stats) > 1000:
            self.performance_stats = self.performance_stats[-1000:]
    
    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key for operation."""
        try:
            # Create hash from operation and arguments
            key_data = f"{operation}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return f"{operation}_{time.time()}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if not self.config.enable_caching:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps[cache_key]
        return cache_age < self.config.cache_duration_seconds
    
    def _safe_vectorbt_operation(self, operation_func: Callable, operation_name: str,
                                *args, **kwargs) -> Any:
        """Safely execute VectorBT operation with fallback and performance monitoring."""
        start_time = time.time()
        data_size = len(args[0]) if args and hasattr(args[0], '__len__') else 0
        success = False
        fallback_used = False
        
        try:
            # Check if we should use VectorBT
            if self._should_use_vectorbt(args[0] if args else None):
                # Check cache first
                if self.config.enable_caching:
                    cache_key = self._get_cache_key(operation_name, *args, **kwargs)
                    if self._is_cache_valid(cache_key):
                        return self.cache[cache_key]
                
                # Execute VectorBT operation
                result = operation_func(*args, **kwargs)
                success = True
                
                # Cache result
                if self.config.enable_caching:
                    self.cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()
                
            else:
                # Use pandas fallback
                result = self._pandas_fallback_operation(operation_name, *args, **kwargs)
                fallback_used = True
                success = True
                
        except Exception as e:
            if not self.config.silent_failures:
                tprint_warning(f"⚠️ VectorBT operation {operation_name} failed: {e}")
            
            # Try pandas fallback
            try:
                result = self._pandas_fallback_operation(operation_name, *args, **kwargs)
                fallback_used = True
                success = True
            except Exception as fallback_error:
                if not self.config.silent_failures:
                    tprint_error(f"❌ Both VectorBT and pandas fallback failed for {operation_name}: {fallback_error}")
                raise fallback_error
        
        # Log performance
        execution_time = time.time() - start_time
        self._log_performance(operation_name, execution_time, data_size, success, fallback_used)
        
        return result
    
    def _pandas_fallback_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Fallback to pandas operations when VectorBT fails."""
        data = args[0] if args else None
        window = kwargs.get('window', 20)
        
        if operation_name == 'rolling_mean':
            return data.rolling(window=window).mean()
        elif operation_name == 'rolling_std':
            return data.rolling(window=window).std()
        elif operation_name == 'rolling_var':
            return data.rolling(window=window).var()
        elif operation_name == 'rolling_min':
            return data.rolling(window=window).min()
        elif operation_name == 'rolling_max':
            return data.rolling(window=window).max()
        elif operation_name == 'rolling_sum':
            return data.rolling(window=window).sum()
        elif operation_name == 'rolling_corr':
            x, y = args[0], args[1]
            return x.rolling(window=window).corr(y)
        elif operation_name == 'rolling_apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func)
        else:
            raise ValueError(f"Unknown operation: {operation_name}")
    
    # Optimized rolling operations
    def rolling_mean(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling mean with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_mean, 'rolling_mean', data, window=window, **kwargs
        )
    
    def rolling_std(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling standard deviation with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_std, 'rolling_std', data, window=window, **kwargs
        )
    
    def rolling_var(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling variance with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_var, 'rolling_var', data, window=window, **kwargs
        )
    
    def rolling_min(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling minimum with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_min, 'rolling_min', data, window=window, **kwargs
        )
    
    def rolling_max(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling maximum with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_max, 'rolling_max', data, window=window, **kwargs
        )
    
    def rolling_sum(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling sum with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_sum, 'rolling_sum', data, window=window, **kwargs
        )
    
    def rolling_corr(self, x: pd.Series, y: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling correlation with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_corr, 'rolling_corr', x, y, window=window, **kwargs
        )
    
    def rolling_apply(self, data: pd.Series, func: Callable, window: int, **kwargs) -> pd.Series:
        """Optimized rolling apply with VectorBT."""
        return self._safe_vectorbt_operation(
            rolling_apply, 'rolling_apply', data, func, window=window, **kwargs
        )
    
    # Statistical operations
    def calculate_volatility(self, returns: pd.Series, window: int = 20, 
                           annualize: bool = True) -> pd.Series:
        """Calculate volatility using VectorBT for optimal performance."""
        def _volatility_calc(data, win):
            if VECTORBT_AVAILABLE and self._should_use_vectorbt(data):
                rolling_std = rolling_std(data, window=win)
                if annualize:
                    return rolling_std * np.sqrt(252)
                return rolling_std
            else:
                rolling_std = data.rolling(window=win).std()
                if annualize:
                    return rolling_std * np.sqrt(252)
                return rolling_std
        
        return self._safe_vectorbt_operation(
            _volatility_calc, 'calculate_volatility', returns, window
        )
    
    def calculate_returns(self, prices: pd.Series, method: str = 'pct_change') -> pd.Series:
        """Calculate returns using VectorBT for optimal performance."""
        def _returns_calc(data, meth):
            if meth == 'pct_change':
                return data.pct_change()
            elif meth == 'log':
                return np.log(data / data.shift(1))
            else:
                return data.pct_change()
        
        return self._safe_vectorbt_operation(
            _returns_calc, 'calculate_returns', prices, method
        )
    
    def calculate_moving_average(self, data: pd.Series, window: int, 
                               method: str = 'simple') -> pd.Series:
        """Calculate moving average using VectorBT."""
        if method == 'simple':
            return self.rolling_mean(data, window)
        elif method == 'exponential':
            # Use pandas for EMA as VectorBT doesn't have direct EMA
            return data.ewm(span=window).mean()
        else:
            return self.rolling_mean(data, window)
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using VectorBT for optimal performance."""
        def _rsi_calc(data, win):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=win).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=win).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return self._safe_vectorbt_operation(
            _rsi_calc, 'calculate_rsi', prices, window
        )
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, 
                                num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands using VectorBT."""
        def _bb_calc(series, win, std):
            if VECTORBT_AVAILABLE and self._should_use_vectorbt(series):
                rolling_mean = rolling_mean(series, window=win)
                rolling_std = rolling_std(series, window=win)
            else:
                rolling_mean = series.rolling(window=win).mean()
                rolling_std = series.rolling(window=win).std()
            
            upper = rolling_mean + (rolling_std * std)
            lower = rolling_mean - (rolling_std * std)
            return upper, rolling_mean, lower
        
        return self._safe_vectorbt_operation(
            _bb_calc, 'calculate_bollinger_bands', data, window, num_std
        )
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate rolling correlation matrix using VectorBT."""
        def _corr_matrix_calc(df, win):
            if VECTORBT_AVAILABLE and self._should_use_vectorbt(df):
                # Use VectorBT for correlation calculation
                corr_data = {}
                for col1 in df.columns:
                    corr_data[col1] = {}
                    for col2 in df.columns:
                        if col1 == col2:
                            corr_data[col1][col2] = pd.Series(1.0, index=df.index)
                        else:
                            corr_data[col1][col2] = rolling_corr(df[col1], df[col2], window=win)
                return pd.DataFrame(corr_data)
            else:
                return df.rolling(window=win).corr()
        
        return self._safe_vectorbt_operation(
            _corr_matrix_calc, 'calculate_correlation_matrix', data, window
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_stats:
            return {'message': 'No performance data available'}
        
        # Calculate statistics
        total_operations = len(self.performance_stats)
        successful_operations = sum(1 for stat in self.performance_stats if stat.success)
        vectorbt_operations = sum(1 for stat in self.performance_stats if not stat.fallback_used)
        pandas_fallbacks = sum(1 for stat in self.performance_stats if stat.fallback_used)
        
        avg_execution_time = np.mean([stat.execution_time for stat in self.performance_stats])
        avg_memory_usage = np.mean([stat.memory_used_mb for stat in self.performance_stats])
        
        # Performance by operation type
        operation_stats = {}
        for stat in self.performance_stats:
            if stat.operation_name not in operation_stats:
                operation_stats[stat.operation_name] = {
                    'count': 0,
                    'avg_time': 0.0,
                    'success_rate': 0.0
                }
            
            op_stat = operation_stats[stat.operation_name]
            op_stat['count'] += 1
            op_stat['avg_time'] += stat.execution_time
            op_stat['success_rate'] += 1 if stat.success else 0
        
        # Calculate averages
        for op_name, op_stat in operation_stats.items():
            op_stat['avg_time'] /= op_stat['count']
            op_stat['success_rate'] /= op_stat['count']
        
        return {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
            'vectorbt_operations': vectorbt_operations,
            'pandas_fallbacks': pandas_fallbacks,
            'vectorbt_usage_rate': vectorbt_operations / total_operations if total_operations > 0 else 0,
            'avg_execution_time': avg_execution_time,
            'avg_memory_usage_mb': avg_memory_usage,
            'operation_breakdown': operation_stats
        }
    
    def clear_cache(self):
        """Clear operation cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        tprint_info("🗑️ VectorBT optimizer cache cleared")
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats.clear()
        tprint_info("📊 Performance statistics reset")


# Global optimizer instance
_global_optimizer: Optional[VectorBTOptimizer] = None


def get_vectorbt_optimizer(config: Optional[VectorBTConfig] = None) -> VectorBTOptimizer:
    """Get global VectorBT optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        tprint_info("🏭 Creating new global VectorBT optimizer instance")
        _global_optimizer = VectorBTOptimizer(config)
    else:
        tprint_info("♻️ Reusing existing global VectorBT optimizer instance")
    return _global_optimizer


def reset_global_optimizer():
    """Reset global optimizer instance."""
    global _global_optimizer
    tprint_info("🔄 Resetting global VectorBT optimizer instance")
    _global_optimizer = None


# Convenience functions for easy usage
def optimized_rolling_mean(data: pd.Series, window: int, **kwargs) -> pd.Series:
    """Convenience function for optimized rolling mean."""
    tprint_info(f"📊 Computing optimized rolling mean (window={window})")
    optimizer = get_vectorbt_optimizer()
    return optimizer.rolling_mean(data, window, **kwargs)


def optimized_rolling_std(data: pd.Series, window: int, **kwargs) -> pd.Series:
    """Convenience function for optimized rolling standard deviation."""
    tprint_info(f"📊 Computing optimized rolling std (window={window})")
    optimizer = get_vectorbt_optimizer()
    return optimizer.rolling_std(data, window, **kwargs)


def optimized_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Convenience function for optimized volatility calculation."""
    tprint_info(f"📊 Computing optimized volatility (window={window}, annualize={annualize})")
    optimizer = get_vectorbt_optimizer()
    return optimizer.calculate_volatility(returns, window, annualize)


def optimized_returns(prices: pd.Series, method: str = 'pct_change') -> pd.Series:
    """Convenience function for optimized returns calculation."""
    tprint_info(f"📊 Computing optimized returns (method={method})")
    optimizer = get_vectorbt_optimizer()
    return optimizer.calculate_returns(prices, method)


# Performance monitoring decorator
def monitor_vectorbt_performance(operation_name: str = None):
    """Decorator to monitor VectorBT operation performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            op_name = operation_name or func.__name__
            tprint_info(f"⚡ VectorBT operation {op_name} completed in {execution_time:.3f}s")
            
            return result
        return wrapper
    return decorator