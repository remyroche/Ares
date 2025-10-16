"""
Feature Generators Compatibility Module

This module provides compatibility for code that imports FeatureGenerators from
src.feature_generation.utils.feature_generators, redirecting to the new unified
feature generation system.
"""

import logging
import warnings
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "Importing FeatureGenerators from src.feature_generation.utils.feature_generators is deprecated. "
    "Please use: from src.feature_generation import FeatureGenerators",
    DeprecationWarning,
    stacklevel=2
)

try:
    # Try to import from the new unified feature generation system
    from ..core.feature_bank import FeatureBank as NewFeatureGenerators
    logger.info("✅ Successfully imported FeatureBank from new unified system")

    # Export the new class as the old name for compatibility
    FeatureGenerators = NewFeatureGenerators

except ImportError as e:
    logger.warning(f"⚠️ Failed to import from new system: {e}")

    # Fallback to original implementation if available
    try:
        from .feature_generators import FeatureGenerators as OriginalFeatureGenerators
        FeatureGenerators = OriginalFeatureGenerators
    except ImportError as e2:
        logger.error(f"❌ All FeatureGenerators compatibility layers failed: {e2}")
        raise ImportError("No compatible FeatureGenerators implementation found")

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    # VectorBT 0.28.1 has different API - use pandas rolling operations as fallback
    VECTORBT_AVAILABLE = True
    
    # Create wrapper functions for compatibility
    def rolling_mean(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).mean()
        return None
    
    def rolling_std(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).std()
        return None
    
    def rolling_var(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).var()
        return None
    
    def rolling_min(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).min()
        return None
    
    def rolling_max(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).max()
        return None
    
    def rolling_sum(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).sum()
        return None
    
    def rolling_apply(data, window, func, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).apply(func)
        return None
    
    def rolling_corr(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).corr()
        return None
    
    def rolling_cov(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).cov()
        return None
    
    # Statistical functions - use numpy/pandas equivalents
    def scale(data, **kwargs):
        import numpy as np
        return (data - np.mean(data)) / np.std(data)
    
    def rank(data, **kwargs):
        if hasattr(data, 'rank'):
            return data.rank(**kwargs)
        return None
    
    def zscore(data, **kwargs):
        import numpy as np
        return (data - np.mean(data)) / np.std(data)
    
    def winsorize(data, limits=None, **kwargs):
        import numpy as np
        if limits is None:
            limits = (0.05, 0.05)
        return np.clip(data, np.quantile(data, limits[0]), np.quantile(data, 1-limits[1]))
    
    def clip(data, min_val=None, max_val=None, **kwargs):
        import numpy as np
        return np.clip(data, min_val, max_val)
    
    def quantile(data, q, **kwargs):
        import numpy as np
        return np.quantile(data, q)
        
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

# CuPy imports for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Remove this problematic fallback since we already have NewFeatureGenerators working
# The FeatureGenerators is already set to NewFeatureGenerators above

# Export the class
__all__ = ['FeatureGenerators']

class VectorBTHelper:
    """Helper class for VectorBT operations."""
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

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
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
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
        else:
            raise ValueError(f"Unsupported operation: {operation}")
