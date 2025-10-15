"""
Common utilities for features_common module.

This module provides shared imports and utilities to reduce duplication
across the features_common package.
"""

import numpy as np
import pandas as pd

def check_for_inf_nan(data, name="data"):
    """Check for infinite or NaN values in data."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        has_inf = np.isinf(data).any().any() if isinstance(data, pd.DataFrame) else np.isinf(data).any()
        has_nan = data.isna().any().any() if isinstance(data, pd.DataFrame) else data.isna().any()
    else:
        has_inf = np.isinf(data).any()
        has_nan = np.isnan(data).any()
    
    if has_inf:
        print(f"⚠️ Warning: {name} contains infinite values")
    if has_nan:
        print(f"⚠️ Warning: {name} contains NaN values")
    
    return not (has_inf or has_nan)

def validate_numeric_array(data, name="data"):
    """Validate that data is numeric and contains no infinite or NaN values."""
    if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        try:
            data = np.array(data)
        except:
            raise ValueError(f"{name} must be convertible to a numeric array")
    
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"{name} must contain numeric data")
    
    return check_for_inf_nan(data, name)

def is_valid_number(value):
    """Check if a value is a valid number (not NaN or infinite)."""
    try:
        return np.isfinite(float(value))
    except (ValueError, TypeError):
        return False

# Import utility functions
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
    tprint("🔧 [features_common.utils] Initializing common utilities module", color="cyan")
except ImportError:
    TPRINT_AVAILABLE = False
    print("⚠️  [features_common.utils] tprint not available")

# Import VectorBT optimization modules
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_OPTIMIZER_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] VectorBT optimization modules loaded successfully", color="green")
except ImportError as e:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    if TPRINT_AVAILABLE:
        tprint(f"⚠️  [features_common.utils] VectorBT optimization modules not available: {e}", color="yellow")

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] VectorBT loaded successfully", color="green")
except ImportError as e:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    if TPRINT_AVAILABLE:
        tprint(f"⚠️  [features_common.utils] VectorBT not available: {e}", color="yellow")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] CuPy available for GPU acceleration", color="green")
except ImportError as e:
    CUPY_AVAILABLE = False
    cp = None
    if TPRINT_AVAILABLE:
        tprint(f"⚠️  [features_common.utils] CuPy not available for GPU acceleration: {e}", color="yellow")

# Math validation imports
try:
    from src.utils.math_validation import (
        safe_divide,
        check_for_inf_nan,
        validate_numeric_array,
        is_valid_number
    )
    MATH_VALIDATION_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] Math validation utilities loaded", color="green")
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    if TPRINT_AVAILABLE:
        tprint(f"⚠️  [features_common.utils] Math validation utilities not available: {e}", color="yellow")

if TPRINT_AVAILABLE:
    tprint("🔧 [features_common.utils] Common utilities module initialized", color="cyan")