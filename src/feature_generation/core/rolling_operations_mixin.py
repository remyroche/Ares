"""
Rolling Operations Mixin

This mixin provides common rolling operations functionality for feature generators,
including optimized rolling calculations, window management, and performance tracking.

Usage:
    class MyFeatureGenerator(VectorizedFeatureGenerator, RollingOperationsMixin):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Use self.rolling_mean(), self.rolling_std(), etc.
            return self.rolling_mean(data['close'], window=20)
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import time

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None

class RollingOperationsMixin:
    """Mixin class that provides rolling operations capabilities for feature generators."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize rolling operations settings
        self.use_vectorbt_rolling = getattr(self, 'use_vectorbt_rolling', True)
        self.rolling_threshold = getattr(self, 'rolling_threshold', 1000)
        self.enable_rolling_cache = getattr(self, 'enable_rolling_cache', True)
        self.rolling_cache_size = getattr(self, 'rolling_cache_size', 100)

        # Performance tracking
        self.rolling_stats = {
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'cached_operations': 0,
            'total_rolling_time': 0.0,
            'operations_count': 0
        }

        # Rolling cache
        if self.enable_rolling_cache:
            self._rolling_cache = {}

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def _should_use_vectorbt_rolling(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """Determine if VectorBT should be used for rolling operations."""
        if not VECTORBT_AVAILABLE or not self.use_vectorbt_rolling:
            return False

        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size >= self.rolling_threshold

    def rolling_mean(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling mean with optimization."""
        return self._rolling_operation(data, 'mean', window, **kwargs)

    def rolling_std(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling standard deviation with optimization."""
        return self._rolling_operation(data, 'std', window, **kwargs)

    def rolling_var(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling variance with optimization."""
        return self._rolling_operation(data, 'var', window, **kwargs)

    def rolling_min(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling minimum with optimization."""
        return self._rolling_operation(data, 'min', window, **kwargs)

    def rolling_max(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling maximum with optimization."""
        return self._rolling_operation(data, 'max', window, **kwargs)

    def rolling_sum(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling sum with optimization."""
        return self._rolling_operation(data, 'sum', window, **kwargs)

    def rolling_corr(self, data: pd.Series, other: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling correlation with optimization."""
        return self._rolling_operation(data, 'corr', window, other=other, **kwargs)

    def rolling_cov(self, data: pd.Series, other: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling covariance with optimization."""
        return self._rolling_operation(data, 'cov', window, other=other, **kwargs)

    def rolling_apply(self, data: pd.Series, func: Callable, window: int, **kwargs) -> pd.Series:
        """Calculate rolling apply with optimization."""
        return self._rolling_operation(data, 'apply', window, func=func, **kwargs)

    def _rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform rolling operation with caching and optimization."""
        start_time = time.time()

        # Check cache first
        if self.enable_rolling_cache:
            cache_key = self._generate_rolling_cache_key(data, operation, window, **kwargs)
            if cache_key in self._rolling_cache:
                self.rolling_stats['cached_operations'] += 1
                return self._rolling_cache[cache_key]

        # Perform operation
        if self._should_use_vectorbt_rolling(data):
            result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
            self.rolling_stats['vectorbt_operations'] += 1
        else:
            result = self._pandas_rolling_operation(data, operation, window, **kwargs)
            self.rolling_stats['pandas_operations'] += 1

        # Cache result
        if self.enable_rolling_cache:
            self._cache_rolling_result(cache_key, result)

        # Update stats
        self.rolling_stats['operations_count'] += 1
        self.rolling_stats['total_rolling_time'] += time.time() - start_time

        return result

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation."""
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                if other is not None:
                    return rolling_corr(data, other, window=window, **kwargs)
                else:
                    raise ValueError("Correlation operation requires 'other' parameter")
            elif operation == 'cov':
                other = kwargs.get('other')
                if other is not None:
                    return rolling_cov(data, other, window=window, **kwargs)
                else:
                    raise ValueError("Covariance operation requires 'other' parameter")
            elif operation == 'apply':
                func = kwargs.get('func')
                if func is not None:
                    # VectorBT rolling_apply expects (data, window, func, **kwargs)
                    return rolling_apply(data, window, func, **kwargs)
                else:
                    raise ValueError("Apply operation requires 'func' parameter")
            else:
                raise ValueError(f"Unsupported VectorBT operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform pandas rolling operation."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            if other is not None:
                return data.rolling(window=window).corr(other)
            else:
                raise ValueError("Correlation operation requires 'other' parameter")
        elif operation == 'cov':
            other = kwargs.get('other')
            if other is not None:
                return data.rolling(window=window).cov(other)
            else:
                raise ValueError("Covariance operation requires 'other' parameter")
        elif operation == 'apply':
            func = kwargs.get('func')
            if func is not None:
                return data.rolling(window=window).apply(func, **kwargs)
            else:
                raise ValueError("Apply operation requires 'func' parameter")
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")

    def _generate_rolling_cache_key(self, data: pd.Series, operation: str, window: int, **kwargs) -> str:
        """Generate cache key for rolling operation."""
        import hashlib

        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]

        return f"{operation}_{window}_{data_hash}_{params_hash}"

    def _cache_rolling_result(self, cache_key: str, result: pd.Series) -> None:
        """Cache rolling operation result."""
        if not self.enable_rolling_cache:
            return

        try:
            # Limit cache size
            if len(self._rolling_cache) >= self.rolling_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._rolling_cache))
                del self._rolling_cache[oldest_key]

            self._rolling_cache[cache_key] = result

        except Exception as e:
            self.logger.warning(f"Rolling cache storage failed: {e}")

    def batch_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform multiple rolling operations in batch for efficiency.

        Args:
            data: Input DataFrame
            operations: List of operation dictionaries with keys:
                - column: Column name
                - operation: Operation type
                - window: Window size
                - name: Output column name
                - **kwargs: Additional parameters

        Returns:
            DataFrame with rolling features
        """
        results = data.copy()

        for op in operations:
            column = op.get('column')
            operation = op.get('operation')
            window = op.get('window')
            name = op.get('name', f"{column}_{operation}_{window}")
            kwargs = {k: v for k, v in op.items() if k not in ['column', 'operation', 'window', 'name']}

            if column in data.columns:
                try:
                    results[name] = self._rolling_operation(
                        data[column], operation, window, **kwargs
                    )
                except Exception as e:
                    self.logger.warning(f"Batch rolling operation failed for {name}: {e}")

        return results

    def get_rolling_stats(self) -> Dict[str, Any]:
        """Get rolling operations statistics."""
        stats = self.rolling_stats.copy()

        if stats['operations_count'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['operations_count'] * 100
            )
            stats['pandas_usage_percentage'] = (
                stats['pandas_operations'] / stats['operations_count'] * 100
            )
            stats['cache_hit_percentage'] = (
                stats['cached_operations'] / stats['operations_count'] * 100
            )
            stats['average_operation_time'] = (
                stats['total_rolling_time'] / stats['operations_count']
            )
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['pandas_usage_percentage'] = 0
            stats['cache_hit_percentage'] = 0
            stats['average_operation_time'] = 0

        return stats

    def reset_rolling_stats(self) -> None:
        """Reset rolling operations statistics."""
        self.rolling_stats = {
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'cached_operations': 0,
            'total_rolling_time': 0.0,
            'operations_count': 0
        }

    def clear_rolling_cache(self) -> None:
        """Clear rolling operations cache."""
        if hasattr(self, '_rolling_cache'):
            self._rolling_cache.clear()

def rolling_optimized(operation: str, window: int):
    """Decorator to automatically apply rolling optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, data: pd.Series, *args, **kwargs):
            if hasattr(self, '_rolling_operation'):
                return self._rolling_operation(data, operation, window, **kwargs)
            return func(self, data, *args, **kwargs)
        return wrapper
    return decorator
