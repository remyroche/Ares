"""
VectorBT compatibility layer for VectorBT 0.28.1+ API.

This module provides a compatibility layer that exposes the API expected by the
Ares codebase. It integrates with VectorBTRollingOptimizer and UnifiedVectorizationManager
for efficient, vectorized computations.

The module automatically detects if VectorBT is available and provides appropriate
fallbacks when it's not.

Usage:
    from src.utils.vectorbt_compat import vbt, rolling_mean, rolling_std, VECTORBT_AVAILABLE
    
For optimized computations, the underlying functions use:
- VectorBTRollingOptimizer for rolling operations
- UnifiedVectorizationManager for vectorized batch processing
- Standard pandas/numpy for reliable fallbacks
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
from typing import Any, Optional, Union
import logging
import importlib.util

logger = logging.getLogger(__name__)


class MockVbtSettings:
    """Mock settings object for VectorBT compatibility."""
    def __init__(self):
        self.array_wrapper = MockSettingsDict()
        self.parallel = MockSettingsDict()
        self.threading = MockSettingsDict()


class MockSettingsDict(dict):
    """Mock settings dictionary that accepts any key/value assignment."""
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # Return None for missing keys instead of raising KeyError
            return None


# Check if VectorBT package is actually available
# We need to import the real vectorbt from site-packages, not ourselves (src.vectorbt)
VECTORBT_AVAILABLE = False
_vectorbt_module = None
vbt = None

try:
    # Import the real vectorbt by manipulating sys.path AND sys.modules
    _original_path = sys.path[:]
    _saved_modules = {}
    
    try:
        # Save and remove any vectorbt modules from sys.modules (including src.vectorbt)
        for key in list(sys.modules.keys()):
            if key == 'vectorbt' or key.startswith('vectorbt.'):
                _saved_modules[key] = sys.modules.pop(key)
        
        # Move site-packages paths to the front to prioritize the real vectorbt
        site_packages_paths = [p for p in sys.path if 'site-packages' in p or 'dist-packages' in p]
        other_paths = [p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p]
        sys.path = site_packages_paths + other_paths
        
        # Now import vectorbt fresh (should get the one from site-packages)
        import vectorbt as _vbt_pkg
        
        # Verify it's the real one
        if hasattr(_vbt_pkg, '__version__') and 'src/vectorbt' not in _vbt_pkg.__file__:
            # Successfully imported real vectorbt
            VECTORBT_AVAILABLE = True
            _vectorbt_module = _vbt_pkg
            vbt = _vbt_pkg
            logger.debug(f"VectorBT {_vbt_pkg.__version__} loaded from {_vbt_pkg.__file__}")
        else:
            raise ImportError(f"Got wrong vectorbt from {_vbt_pkg.__file__}")
    finally:
        # Restore original path and modules (but keep vectorbt if we successfully imported the real one)
        sys.path = _original_path
        if not VECTORBT_AVAILABLE:
            # Restore the saved modules since we failed
            sys.modules.update(_saved_modules)
    
    if not VECTORBT_AVAILABLE:
        raise ImportError("Real VectorBT package not found in site-packages")
        
except (ImportError, AttributeError, OSError) as e:
    VECTORBT_AVAILABLE = False
    logger.warning(f"VectorBT not available: {e}, using fallback implementations")
    import traceback
    logger.debug(f"Full traceback: {traceback.format_exc()}")
    # Create a minimal mock as fallback
    class MockVbtObject:
        """Mock vbt object for compatibility."""
        def __init__(self):
            self.__version__ = "not_available"
            self._settings = None

        @property
        def settings(self):
            """Mock settings object for compatibility."""
            if self._settings is None:
                self._settings = MockVbtSettings()
            return self._settings

    vbt = MockVbtObject()


# Lazy import cache to avoid circular imports
_compat_cache = {}
_compat_loading = False


def _lazy_load_compat():
    """Lazily load the compatibility layer to avoid circular imports during initialization."""
    global _compat_cache, _compat_loading
    
    if _compat_cache:
        return _compat_cache
    
    if _compat_loading:
        # Already in the process of loading, return empty dict to break recursion
        logger.debug("Recursive import of vectorbt_compat detected, breaking cycle")
        return {}
    
    _compat_loading = True
    try:
        # Try relative import first (normal package usage)
        try:
            from ..utils.vectorbt_compat import (
                rolling_mean,
                rolling_std,
                rolling_var,
                rolling_min,
                rolling_max,
                rolling_sum,
                rolling_apply,
                rolling_corr,
                rolling_cov,
                rolling_median,
                rolling_quantile,
                rolling_rank,
                scale,
                rank,
                zscore,
                winsorize,
                clip,
                quantile
            )
        except (ImportError, ValueError) as e:
            # Fallback to absolute import (script usage)
            from src.utils.vectorbt_compat import (
                rolling_mean,
                rolling_std,
                rolling_var,
                rolling_min,
                rolling_max,
                rolling_sum,
                rolling_apply,
                rolling_corr,
                rolling_cov,
                rolling_median,
                rolling_quantile,
                rolling_rank,
                scale,
                rank,
                zscore,
                winsorize,
                clip,
                quantile
            )
        
        _compat_cache.update({
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_var': rolling_var,
            'rolling_min': rolling_min,
            'rolling_max': rolling_max,
            'rolling_sum': rolling_sum,
            'rolling_apply': rolling_apply,
            'rolling_corr': rolling_corr,
            'rolling_cov': rolling_cov,
            'rolling_median': rolling_median,
            'rolling_quantile': rolling_quantile,
            'rolling_rank': rolling_rank,
            'scale': scale,
            'rank': rank,
            'zscore': zscore,
            'winsorize': winsorize,
            'clip': clip,
            'quantile': quantile,
        })
        logger.debug("VectorBT compatibility functions loaded successfully")
    except ImportError as e:
        logger.warning(f"Failed to load vectorbt_compat: {e}")
    finally:
        _compat_loading = False
    
    return _compat_cache


def __getattr__(name):
    """
    Lazy loading of compatibility functions.
    This prevents circular imports by only loading functions when they're actually accessed.
    """
    # Load compatibility layer on first access
    compat = _lazy_load_compat()
    
    if name in compat:
        return compat[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all public API
__all__ = [
    # Core objects
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
    
    # Statistical operations  
    'scale',
    'rank',
    'zscore',
    'winsorize',
    'clip',
    'quantile',
]
