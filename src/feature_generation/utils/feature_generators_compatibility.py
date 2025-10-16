"""
Feature Generators Compatibility Module

This module provides compatibility for code that imports FeatureGenerators from
src.feature_generation.utils.feature_generators, redirecting to the new unified
feature generation system.
"""

import logging
import warnings

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
    from ..feature_generation import FeatureGenerators as NewFeatureGenerators
    logger.info("✅ Successfully imported FeatureGenerators from new unified system")

    # Export the new class as the old name for compatibility
    FeatureGenerators = NewFeatureGenerators

except ImportError as e:
    logger.warning(f"⚠️ Failed to import from new system: {e}")

    # Try simple compatibility layer
    try:
        from ..feature_generation.compatibility.simple_hmm_compatibility import FeatureGenerators as SimpleFeatureGenerators
        logger.info("✅ Using simple HMM compatibility layer")
        FeatureGenerators = SimpleFeatureGenerators

    except ImportError as e2:
        logger.warning(f"⚠️ Simple compatibility layer also failed: {e2}")

        # Fallback to original implementation if available
        try:
            from .feature_generators import FeatureGenerators as OriginalFeatureGenerators
            FeatureGenerators = OriginalFeatureGenerators
        except ImportError as e3:
            logger.error(f"❌ All FeatureGenerators compatibility layers failed: {e3}")
            raise ImportError("No compatible FeatureGenerators implementation found")

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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

# CuPy imports for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    logger.warning("⚠️ Using original FeatureGenerators as fallback")
    FeatureGenerators = OriginalFeatureGenerators

except ImportError as e3:
    logger.error(f"❌ Original FeatureGenerators also not available: {e3}")

    # Create a minimal fallback class
    class FeatureGenerators:
        """Minimal fallback FeatureGenerators class."""

        def __init__(self):
            self.logger = logger.getChild('FeatureGenerators')
            self.logger.warning("⚠️ Using minimal fallback FeatureGenerators")

        def generate_features_for_hmm(self, data):
            """Minimal fallback implementation."""
            self.logger.info("📊 Minimal fallback: returning data as-is")
            return data

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
