"""
Backwards Compatibility Module for Feature Selection Framework

This module provides backwards compatibility for the original FeatureSelectionFramework
by importing and exposing the new modular components.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# Import the new modular framework
try:
    from ...training.utils.feature_selection import FeatureSelectionFramework
    
    # Issue deprecation warning
    warnings.warn(
        "The feature_selection.py module has been refactored into modular components. "
        "Please consider migrating to src/training/utils/feature_selection/ for better maintainability.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("⚠️ DEPRECATION WARNING: feature_selection.py has been refactored.")
    print("📁 New location: src/training/utils/feature_selection/")
    print("✅ Backwards compatibility maintained.")
    
except ImportError as e:
    print(f"⚠️ Could not import new modular components: {e}")
    print("🔄 Please ensure the new modules are available.")
    raise

# Re-export for backwards compatibility
__all__ = ['FeatureSelectionFramework']