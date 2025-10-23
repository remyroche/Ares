"""
Common constants for the features_common module.

Notes:
- Do NOT import heavy optional deps (like vectorbt/numba) at module import time.
  Importing `vectorbt` pulls in `numba`/`llvmlite`, which can be slow or fail on
  some systems. We expose a lightweight availability probe and defer the actual
  import to call sites that need it (see utils.get_vbt / LazyVBT).
"""

import os
import importlib.util

# VectorBT availability check (non-loading)
# - Honours an env kill switch to force-disable VectorBT without importing it
# - Uses importlib.util.find_spec to avoid triggering import-time side effects
_env_disable_vbt = os.getenv("ARES_DISABLE_VECTORBT", "0").strip() in {"1", "true", "yes", "on"}
if _env_disable_vbt:
    VECTORBT_AVAILABLE = False
    vbt = None
else:
    VECTORBT_AVAILABLE = importlib.util.find_spec("vectorbt") is not None
    vbt = None  # actual module is loaded lazily in utils.get_vbt()

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
