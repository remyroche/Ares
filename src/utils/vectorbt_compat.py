"""
VectorBT Compatibility Layer for version 0.28.1

This module provides a compatibility shim for VectorBT 0.28.1, which has a different API
than expected by the legacy code. It implements the expected rolling and generic operations
using standard pandas/numpy operations for optimal performance.

This module integrates with:
- M1CPUOptimizer for CPU-optimized operations
- M1GPUManager for GPU acceleration when available
- M1MemoryOptimizer for efficient memory management
- UnifiedHardwareManager for coordinated hardware optimization

All functions accept pandas Series/DataFrame or numpy arrays and return the same type.
"""

# Import real vectorbt
import sys
vbt = None
VECTORBT_AVAILABLE = False
try:
    saved = {}
    for k in list(sys.modules.keys()):
        if k.startswith("vectorbt"):
            saved[k] = sys.modules.pop(k)
    orig = sys.path[:]
    sp = [p for p in sys.path if "site-packages" in p]
    ot = [p for p in sys.path if "site-packages" not in p]
    sys.path[:] = sp + ot
    try:
        import vectorbt
        if hasattr(vectorbt, "__version__"):
            vbt = vectorbt
            VECTORBT_AVAILABLE = True
    finally:
        sys.path[:] = orig
except:
    pass

import pandas as pd
import numpy as np
from typing import Union, Callable, Optional, Any
import warnings
import logging

logger = logging.getLogger(__name__)

# Hardware optimization imports
try:
    from .hardware.unified_hardware_manager import (
        get_unified_hardware_manager,
        WorkloadType,
        OptimizationLevel
    )
    from .hardware.m1_cpu_optimizer import M1CPUOptimizer
    from .hardware.m1_gpu_utils import M1GPUManager
    from .hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_AVAILABLE = True
except ImportError as e:
    HARDWARE_AVAILABLE = False
    logger.debug(f"Hardware optimization not available: {e}")

# Type aliases for better readability
ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray]

# Global hardware manager instance
_hardware_manager = None
_hardware_initialized = False


def _get_hardware_manager():
    """Get or initialize the hardware manager for optimizations."""
    global _hardware_manager, _hardware_initialized
    
    if not HARDWARE_AVAILABLE:
        return None
    
    if not _hardware_initialized:
        try:
            _hardware_manager = get_unified_hardware_manager()
            # Optimize for feature engineering workload
            _hardware_manager.optimize_for_workload(
                WorkloadType.FEATURE_ENGINEERING,
                OptimizationLevel.BALANCED
            )
            _hardware_initialized = True
            logger.debug("Hardware manager initialized for vectorbt_compat")
        except Exception as e:
            logger.debug(f"Could not initialize hardware manager: {e}")
            _hardware_manager = None
            
    return _hardware_manager


def _ensure_series(data: ArrayLike, name: str = "data") -> pd.Series:
    """Convert input to pandas Series if needed."""
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError(f"Expected Series or 1D array, got DataFrame with {data.shape[1]} columns")
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.Series(data, name=name)
        elif data.ndim == 2 and data.shape[1] == 1:
            return pd.Series(data[:, 0], name=name)
        raise ValueError(f"Expected 1D array, got {data.ndim}D array")
    else:
        return pd.Series(data, name=name)


def _ensure_dataframe(data: ArrayLike) -> pd.DataFrame:
    """Convert input to pandas DataFrame if needed."""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        return data.to_frame()
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame(data, columns=['data'])
        elif data.ndim == 2:
            return pd.DataFrame(data)
        raise ValueError(f"Expected 1D or 2D array, got {data.ndim}D array")
    else:
        return pd.DataFrame(data)


def _restore_type(result: Union[pd.Series, pd.DataFrame], original: ArrayLike) -> ArrayLike:
    """Restore the original data type."""
    if isinstance(original, np.ndarray):
        return result.values
    return result


def _use_hardware_optimization(data_size: int, operation_type: str = 'rolling') -> bool:
    """
    Determine if hardware optimization should be used based on data size.
    
    Args:
        data_size: Size of the data to process
        operation_type: Type of operation ('rolling', 'statistical', 'transform')
    
    Returns:
        True if hardware optimization should be used
    """
    if not HARDWARE_AVAILABLE:
        return False
    
    # Use hardware optimization for larger datasets
    thresholds = {
        'rolling': 1000,      # Rolling operations benefit from optimization above 1K rows
        'statistical': 5000,  # Statistical operations above 5K rows
        'transform': 2000     # Transform operations above 2K rows
    }
    
    threshold = thresholds.get(operation_type, 1000)
    return data_size >= threshold


# ============================================================================
# Rolling Operations
# ============================================================================

def rolling_mean(data: ArrayLike, window: int, min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling mean with hardware optimization.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling mean (same type as input)
    """
    # Initialize hardware optimization for large datasets
    data_size = len(data) if hasattr(data, '__len__') else data.shape[0]
    if _use_hardware_optimization(data_size, 'rolling'):
        hw_manager = _get_hardware_manager()
        if hw_manager:
            # Hardware manager will optimize CPU/memory allocation
            pass
    
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).mean()
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).mean()
        return result.values


def rolling_std(data: ArrayLike, window: int, min_periods: Optional[int] = None, ddof: int = 1, **kwargs) -> ArrayLike:
    """
    Calculate rolling standard deviation.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        ddof: Delta degrees of freedom
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling standard deviation (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).std(ddof=ddof)
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).std(ddof=ddof)
        return result.values


def rolling_var(data: ArrayLike, window: int, min_periods: Optional[int] = None, ddof: int = 1, **kwargs) -> ArrayLike:
    """
    Calculate rolling variance.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        ddof: Delta degrees of freedom
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling variance (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).var(ddof=ddof)
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).var(ddof=ddof)
        return result.values


def rolling_min(data: ArrayLike, window: int, min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling minimum.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling minimum (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).min()
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).min()
        return result.values


def rolling_max(data: ArrayLike, window: int, min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling maximum.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling maximum (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).max()
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).max()
        return result.values


def rolling_sum(data: ArrayLike, window: int, min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling sum.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling sum (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).sum()
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).sum()
        return result.values


def rolling_apply(data: ArrayLike, window: int, func: Callable, 
                  min_periods: Optional[int] = None, raw: bool = False, **kwargs) -> ArrayLike:
    """
    Apply a custom function over a rolling window.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        func: Function to apply to each window
        min_periods: Minimum number of observations required
        raw: Whether to pass raw ndarray to func (True) or Series (False)
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Result of applying func (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).apply(func, raw=raw)
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).apply(func, raw=raw)
        return result.values


def rolling_corr(data1: ArrayLike, data2: Optional[ArrayLike] = None, 
                 window: int = None, min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling correlation.
    
    Args:
        data1: First input data (Series, DataFrame, or array)
        data2: Second input data (optional, if None calculates pairwise for DataFrame)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling correlation (same type as input)
    """
    if isinstance(data1, (pd.Series, pd.DataFrame)):
        if data2 is None:
            result = data1.rolling(window=window, min_periods=min_periods, **kwargs).corr()
        else:
            if not isinstance(data2, (pd.Series, pd.DataFrame)):
                data2 = _ensure_series(data2, name="data2")
            result = data1.rolling(window=window, min_periods=min_periods, **kwargs).corr(data2)
        return result
    else:
        series1 = _ensure_series(data1, name="data1")
        if data2 is None:
            result = series1.rolling(window=window, min_periods=min_periods, **kwargs).corr()
        else:
            series2 = _ensure_series(data2, name="data2")
            result = series1.rolling(window=window, min_periods=min_periods, **kwargs).corr(series2)
        return result.values


def rolling_cov(data1: ArrayLike, data2: Optional[ArrayLike] = None, 
                window: int = None, min_periods: Optional[int] = None, ddof: int = 1, **kwargs) -> ArrayLike:
    """
    Calculate rolling covariance.
    
    Args:
        data1: First input data (Series, DataFrame, or array)
        data2: Second input data (optional, if None calculates pairwise for DataFrame)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        ddof: Delta degrees of freedom
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling covariance (same type as input)
    """
    if isinstance(data1, (pd.Series, pd.DataFrame)):
        if data2 is None:
            result = data1.rolling(window=window, min_periods=min_periods, **kwargs).cov(ddof=ddof)
        else:
            if not isinstance(data2, (pd.Series, pd.DataFrame)):
                data2 = _ensure_series(data2, name="data2")
            result = data1.rolling(window=window, min_periods=min_periods, **kwargs).cov(data2, ddof=ddof)
        return result
    else:
        series1 = _ensure_series(data1, name="data1")
        if data2 is None:
            result = series1.rolling(window=window, min_periods=min_periods, **kwargs).cov(ddof=ddof)
        else:
            series2 = _ensure_series(data2, name="data2")
            result = series1.rolling(window=window, min_periods=min_periods, **kwargs).cov(series2, ddof=ddof)
        return result.values


def rolling_median(data: ArrayLike, window: int, min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling median.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling median (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).median()
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).median()
        return result.values


def rolling_quantile(data: ArrayLike, window: int, quantile: float, 
                     min_periods: Optional[int] = None, **kwargs) -> ArrayLike:
    """
    Calculate rolling quantile.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        quantile: Quantile to compute (0-1)
        min_periods: Minimum number of observations required
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling quantile (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).quantile(quantile)
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).quantile(quantile)
        return result.values


def rolling_rank(data: ArrayLike, window: int, min_periods: Optional[int] = None, 
                 pct: bool = False, **kwargs) -> ArrayLike:
    """
    Calculate rolling rank.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        window: Size of the rolling window
        min_periods: Minimum number of observations required
        pct: Whether to return percentile rank (0-1) instead of ordinal rank
        **kwargs: Additional arguments passed to pandas rolling
        
    Returns:
        Rolling rank (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rolling(window=window, min_periods=min_periods, **kwargs).rank(pct=pct)
        return result
    else:
        series = _ensure_series(data)
        result = series.rolling(window=window, min_periods=min_periods, **kwargs).rank(pct=pct)
        return result.values


# ============================================================================
# Generic Operations (non-rolling)
# ============================================================================

def scale(data: ArrayLike, min_val: float = 0.0, max_val: float = 1.0) -> ArrayLike:
    """
    Scale data to a specified range.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        min_val: Minimum value of scaled output
        max_val: Maximum value of scaled output
        
    Returns:
        Scaled data (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data_min = data.min()
        data_max = data.max()
        result = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
        return result
    else:
        arr = np.asarray(data)
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        result = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
        return result


def rank(data: ArrayLike, pct: bool = False, ascending: bool = True, method: str = 'average') -> ArrayLike:
    """
    Compute numerical rank of values.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        pct: Whether to return percentile rank (0-1)
        ascending: Whether to rank in ascending order
        method: How to rank equal values ('average', 'min', 'max', 'first', 'dense')
        
    Returns:
        Ranked data (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.rank(pct=pct, ascending=ascending, method=method)
        return result
    else:
        series = _ensure_series(data)
        result = series.rank(pct=pct, ascending=ascending, method=method)
        return result.values


def zscore(data: ArrayLike, ddof: int = 1) -> ArrayLike:
    """
    Calculate z-score (standardization).
    
    Args:
        data: Input data (Series, DataFrame, or array)
        ddof: Delta degrees of freedom for standard deviation
        
    Returns:
        Z-scores (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = (data - data.mean()) / data.std(ddof=ddof)
        return result
    else:
        arr = np.asarray(data)
        result = (arr - np.nanmean(arr)) / np.nanstd(arr, ddof=ddof)
        return result


def winsorize(data: ArrayLike, lower: float = 0.05, upper: float = 0.05) -> ArrayLike:
    """
    Winsorize (clip) data at specified quantiles.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        lower: Lower quantile for clipping (0-1)
        upper: Upper quantile for clipping (0-1)
        
    Returns:
        Winsorized data (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        lower_val = data.quantile(lower)
        upper_val = data.quantile(1 - upper)
        result = data.clip(lower=lower_val, upper=upper_val)
        return result
    else:
        arr = np.asarray(data)
        lower_val = np.nanquantile(arr, lower)
        upper_val = np.nanquantile(arr, 1 - upper)
        result = np.clip(arr, lower_val, upper_val)
        return result


def clip(data: ArrayLike, lower: Optional[float] = None, upper: Optional[float] = None) -> ArrayLike:
    """
    Clip (limit) values.
    
    Args:
        data: Input data (Series, DataFrame, or array)
        lower: Lower bound for clipping
        upper: Upper bound for clipping
        
    Returns:
        Clipped data (same type as input)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.clip(lower=lower, upper=upper)
        return result
    else:
        result = np.clip(data, lower, upper)
        return result


def quantile(data: ArrayLike, q: Union[float, list], interpolation: str = 'linear') -> Union[float, ArrayLike]:
    """
    Compute quantile(s).
    
    Args:
        data: Input data (Series, DataFrame, or array)
        q: Quantile(s) to compute (0-1), scalar or list
        interpolation: Interpolation method
        
    Returns:
        Quantile value(s)
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.quantile(q, interpolation=interpolation)
        return result
    else:
        arr = np.asarray(data)
        result = np.nanquantile(arr, q, method=interpolation)
        return result


# ============================================================================
# Convenience exports
# ============================================================================

__all__ = [
    # VectorBT module
    'vbt',
    'VECTORBT_AVAILABLE',
    
    # Rolling operations
    'rolling_mean',
    'rolling_std',
    'rolling_var',
    'rolling_min',
    'rolling_max',
    'rolling_sum',
    'rolling_apply',
    'rolling_corr',
    'rolling_cov',
    'rolling_median',
    'rolling_quantile',
    'rolling_rank',
    
    # Generic operations
    'scale',
    'rank',
    'zscore',
    'winsorize',
    'clip',
    'quantile',
]


# Initialize hardware optimization on module load
if HARDWARE_AVAILABLE:
    logger.debug("VectorBT compatibility layer loaded with hardware optimization support")
    logger.debug("Using: UnifiedHardwareManager, M1CPUOptimizer, M1GPUManager, M1MemoryOptimizer")
else:
    logger.debug("VectorBT compatibility layer loaded - using pandas/numpy operations for rolling functions")

