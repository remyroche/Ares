"""
Vectorized Operations for Financial Time Series Analysis

This module provides highly optimized vectorized operations to replace
slow pandas.apply() calls with efficient numpy-based implementations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Callable
from numba import jit, njit, prange
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")


@njit(parallel=True, fastmath=True)
def vectorized_zscore_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling z-score using Numba."""
    n = len(values)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        
        if std_val > 0:
            result[i] = (values[i] - mean_val) / std_val
        else:
            result[i] = 0.0
    
    return result


def vectorized_zscore(series: pd.Series, window: int, use_numba: bool = True) -> pd.Series:
    """Vectorized z-score calculation."""
    if use_numba:
        result = vectorized_zscore_numba(series.values, window)
        return pd.Series(result, index=series.index)
    else:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std


@njit(parallel=True, fastmath=True)
def vectorized_percentile_rank_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling percentile rank using Numba."""
    n = len(values)
    result = np.full(n, np.nan)
    
    if window >= n:
        return result
    
    for i in prange(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        current_val = values[i]
        
        rank = np.sum(window_data <= current_val)
        percentile = rank / len(window_data)
        result[i] = percentile
    
    return result


def vectorized_percentile_rank(series: pd.Series, window: int, use_numba: bool = True) -> pd.Series:
    """Vectorized percentile rank calculation."""
    if use_numba:
        result = vectorized_percentile_rank_numba(series.values, window)
        return pd.Series(result, index=series.index)
    else:
        return series.rolling(window).apply(
            lambda x: (x <= x.iloc[-1]).sum() / len(x),
            raw=False
        )


def optimize_dataframe_operations(df: pd.DataFrame, use_numba: bool = True) -> pd.DataFrame:
    """Automatically optimize common DataFrame operations."""
    tprint_info(f"🚀 Optimizing DataFrame operations (Numba: {use_numba})")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = df.copy()
    
    for col in numeric_cols:
        series = df[col]
        # Add optimized features
        result[f'{col}_zscore_20'] = vectorized_zscore(series, 20, use_numba)
        result[f'{col}_pct_20'] = vectorized_percentile_rank(series, 20, use_numba)
    
    tprint_info(f"✅ Optimized {len(numeric_cols)} numeric columns")
    return result


# Export main functions
__all__ = [
    'vectorized_zscore',
    'vectorized_percentile_rank', 
    'optimize_dataframe_operations',
    'vectorized_zscore_numba',
    'vectorized_percentile_rank_numba'
]
