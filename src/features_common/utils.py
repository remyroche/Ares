"""
Common utilities for features_common module.

This module provides shared imports and utilities to reduce duplication
across the features_common package.
"""

import numpy as np
import pandas as pd

# Math validation functions are now imported from constants.py to avoid circular imports

# Import constants to avoid circular imports
from .constants import (
    VECTORBT_AVAILABLE, TPRINT_AVAILABLE, MATH_VALIDATION_AVAILABLE,
    safe_divide, check_for_inf_nan, validate_numeric_array, is_valid_number
)

# Import utility functions
if TPRINT_AVAILABLE:
    from src.utils.tprint import tprint
    tprint("🔧 [features_common.utils] Initializing common utilities module", color="cyan")
else:
    print("⚠️  [features_common.utils] tprint not available")

# VectorBT optimization modules removed to avoid circular imports
# These should be imported directly where needed

# VectorBT imports (lazy loaded to avoid circular imports)
_vbt = None

def get_vbt():
    """Lazy load VectorBT to avoid circular imports."""
    global _vbt
    if _vbt is None:
        if VECTORBT_AVAILABLE:
            import vectorbt as vbt
            _vbt = vbt
            if TPRINT_AVAILABLE:
                tprint("✅ [features_common.utils] VectorBT loaded successfully", color="green")
        else:
            _vbt = None
            if TPRINT_AVAILABLE:
                tprint("⚠️  [features_common.utils] VectorBT not available", color="yellow")
    return _vbt

# For backward compatibility, create a lazy vbt object
class LazyVBT:
    def __getattr__(self, name):
        vbt_module = get_vbt()
        if vbt_module is None:
            raise AttributeError(f"VectorBT not available, cannot access {name}")
        return getattr(vbt_module, name)

    def __call__(self, *args, **kwargs):
        vbt_module = get_vbt()
        if vbt_module is None:
            raise RuntimeError("VectorBT not available")
        return vbt_module(*args, **kwargs)

# Set vbt to lazy loader to avoid circular imports
vbt = LazyVBT()

# CuPy support removed

# Rolling and transformation functions using pandas/numpy (VectorBT generic functions not available in current version)
def rolling_mean(data, window, **kwargs):
    """Rolling mean using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).mean()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).mean()
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rolling_std(data, window, **kwargs):
    """Rolling standard deviation using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).std()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).std()
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rolling_var(data, window, **kwargs):
    """Rolling variance using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).var()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).var()
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rolling_min(data, window, **kwargs):
    """Rolling minimum using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).min()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).min()
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rolling_max(data, window, **kwargs):
    """Rolling maximum using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).max()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).max()
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rolling_sum(data, window, **kwargs):
    """Rolling sum using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).sum()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).sum()
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rolling_apply(data, window, func, **kwargs):
    """Rolling apply using pandas."""
    if isinstance(data, pd.DataFrame):
        return data.rolling(window=window, **kwargs).apply(func)
    elif isinstance(data, pd.Series):
        return data.rolling(window=window, **kwargs).apply(func)
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def scale(data, method='standardize'):
    """Scale data using standardization or normalization."""
    if isinstance(data, pd.DataFrame):
        if method == 'standardize':
            return (data - data.mean()) / data.std()
        elif method == 'normalize':
            return (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    elif isinstance(data, pd.Series):
        if method == 'standardize':
            return (data - data.mean()) / data.std()
        elif method == 'normalize':
            return (data - data.min()) / (data.max() - data.min())
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def rank(data, method='average'):
    """Rank data."""
    if isinstance(data, pd.DataFrame):
        return data.rank(method=method)
    elif isinstance(data, pd.Series):
        return data.rank(method=method)
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def zscore(data):
    """Calculate z-score (standard score)."""
    return scale(data, method='standardize')

def winsorize(data, limits=0.05):
    """Winsorize data by clipping extreme values."""
    if isinstance(data, pd.DataFrame):
        lower_bounds = data.quantile(limits)
        upper_bounds = data.quantile(1 - limits)
        return data.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    elif isinstance(data, pd.Series):
        lower_bound = data.quantile(limits)
        upper_bound = data.quantile(1 - limits)
        return data.clip(lower=lower_bound, upper=upper_bound)
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def clip(data, lower=None, upper=None):
    """Clip data to specified bounds."""
    if isinstance(data, pd.DataFrame):
        return data.clip(lower=lower, upper=upper)
    elif isinstance(data, pd.Series):
        return data.clip(lower=lower, upper=upper)
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

def quantile(data, q=0.5):
    """Calculate quantiles."""
    if isinstance(data, pd.DataFrame):
        return data.quantile(q)
    elif isinstance(data, pd.Series):
        return data.quantile(q)
    else:
        raise ValueError("Data must be pandas DataFrame or Series")

VECTORBT_ROLLING_AVAILABLE = True
VECTORBT_OPTIMIZER_AVAILABLE = VECTORBT_AVAILABLE

# VectorBT optimization classes (stub implementations)
class VectorBTRollingOptimizer:
    """Stub implementation of VectorBT rolling optimizer."""
    def __init__(self, *args, **kwargs):
        pass

def get_vectorbt_rolling_optimizer(*args, **kwargs):
    """Get VectorBT rolling optimizer."""
    return VectorBTRollingOptimizer(*args, **kwargs)

class UnifiedVectorizationManager:
    """Stub implementation of unified vectorization manager."""
    def __init__(self, *args, **kwargs):
        pass

def get_unified_vectorization_manager(*args, **kwargs):
    """Get unified vectorization manager."""
    return UnifiedVectorizationManager(*args, **kwargs)

# Math validation utilities (imported from constants to avoid circular imports)
if TPRINT_AVAILABLE and MATH_VALIDATION_AVAILABLE:
    tprint("✅ [features_common.utils] Math validation utilities loaded", color="green")
elif TPRINT_AVAILABLE:
    tprint("⚠️  [features_common.utils] Math validation utilities not available", color="yellow")

if TPRINT_AVAILABLE:
    tprint("🔧 [features_common.utils] Common utilities module initialized", color="cyan")
