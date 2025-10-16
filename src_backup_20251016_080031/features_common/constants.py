"""
Common constants for the features_common module.

This module provides shared constants to avoid circular import issues.
"""

# VectorBT availability check
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# TPrint availability check
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

# Math validation availability check
try:
    from src.utils.math_validation import (
        safe_divide,
        check_for_inf_nan,
        validate_numeric_array,
        is_valid_number
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    # Define fallback functions
    def safe_divide(a, b, default=0.0): return default
    def check_for_inf_nan(data, name="data"): return True
    def validate_numeric_array(data, name="data"): return True
    def is_valid_number(value): return False
