"""
Common utilities for features_common module.

This module provides shared imports and utilities to reduce duplication
across the features_common package.
"""

# Import utility functions
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

# Import VectorBT optimization modules
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_OPTIMIZER_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] VectorBT optimization modules loaded successfully", color="green")
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    if TPRINT_AVAILABLE:
        tprint("⚠️  [features_common.utils] VectorBT optimization modules not available", color="yellow")

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] VectorBT loaded successfully", color="green")
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    if TPRINT_AVAILABLE:
        tprint("⚠️  [features_common.utils] VectorBT not available", color="yellow")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    if TPRINT_AVAILABLE:
        tprint("✅ [features_common.utils] CuPy available for GPU acceleration", color="green")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    if TPRINT_AVAILABLE:
        tprint("⚠️  [features_common.utils] CuPy not available for GPU acceleration", color="yellow")

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
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    if TPRINT_AVAILABLE:
        tprint("⚠️  [features_common.utils] Math validation utilities not available", color="yellow")

if TPRINT_AVAILABLE:
    tprint("🔧 [features_common.utils] Common utilities module initialized", color="cyan")