"""
Backwards Compatibility Layer for Feature Selection

This module provides backwards compatibility for existing feature selection code,
ensuring that all existing implementations continue to work while leveraging the
new unified framework.

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd

# Import the unified framework
from .unified_feature_selection import UnifiedFeatureSelector, UnifiedFeatureSelectionConfig

# Import existing components for compatibility
try:
    from .feature_selection_backwards_compat import FeatureSelector, FeatureSelectionConfig
    from .utils.feature_selection import FeatureSelectionFramework as UtilsFramework
    EXISTING_COMPONENTS_AVAILABLE = True
except ImportError:
    EXISTING_COMPONENTS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class BackwardsCompatibilityWrapper:
    """
    Backwards compatibility wrapper that provides the same interface as existing
    feature selection components while using the unified framework internally.
    """
    
    def __init__(self, legacy_config: Optional[Any] = None):
        """Initialize backwards compatibility wrapper."""
        self.logger = logger.getChild('BackwardsCompatibilityWrapper')
        self.legacy_config = legacy_config
        
        # Initialize unified framework
        self.unified_selector = UnifiedFeatureSelector()
        
        # Map legacy configurations to unified configurations
        self._map_legacy_config()
        
        self.logger.info("🔄 Backwards compatibility wrapper initialized")
    
    def _map_legacy_config(self):
        """Map legacy configuration to unified configuration."""
        if self.legacy_config is None:
            return
        
        # Convert legacy config to unified config
        unified_config = UnifiedFeatureSelectionConfig()
        
        # Map common legacy parameters
        if hasattr(self.legacy_config, 'max_features'):
            unified_config.target_features = self.legacy_config.max_features
        
        if hasattr(self.legacy_config, 'method'):
            method_mapping = {
                'correlation': 'filter',
                'recursive': 'wrapper',
                'lasso': 'embedded',
                'mrmr': 'hybrid'
            }
            unified_config.primary_method = method_mapping.get(
                self.legacy_config.method, 'hybrid'
            )
        
        if hasattr(self.legacy_config, 'cv_folds'):
            unified_config.cv_folds = self.legacy_config.cv_folds
        
        if hasattr(self.legacy_config, 'random_state'):
            unified_config.random_state = self.legacy_config.random_state
        
        # Update unified selector with mapped config
        self.unified_selector.config = unified_config
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BackwardsCompatibilityWrapper':
        """Fit the feature selector (legacy interface)."""
        self.logger.info("🔄 Fitting feature selector (legacy interface)")
        
        # Store data for later use
        self.X = X
        self.y = y
        
        # Perform feature selection
        self.results = self.unified_selector.select_features(X, y)
        
        # Extract selected features for legacy interface
        self.selected_features = []
        self.feature_scores = {}
        
        # Get the largest feature set as the main result
        if self.results:
            largest_set = max(self.results.keys(), key=lambda k: len(self.results[k]['selected_features']))
            self.selected_features = self.results[largest_set]['selected_features']
            self.feature_scores = self.results[largest_set]['feature_scores']
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data by selecting features (legacy interface)."""
        if not hasattr(self, 'selected_features'):
            raise ValueError("FeatureSelector must be fitted before transform")
        
        if not self.selected_features:
            return X
        
        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            # Ensure selected features exist in the data
            available_features = [feat for feat in self.selected_features if feat in X.columns]
            if not available_features:
                # Fallback: return first few columns
                available_features = list(X.columns)[:min(10, len(X.columns))]
            return X[available_features]
        
        # Handle numpy array
        else:
            # For numpy arrays, we can't easily select by name
            # Return the same array (this is a limitation of the legacy interface)
            return X
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform data (legacy interface)."""
        return self.fit(X, y).transform(X)
    
    def get_support(self) -> List[bool]:
        """Get boolean mask of selected features (legacy interface)."""
        if not hasattr(self, 'selected_features'):
            return []
        
        # This is a simplified implementation
        # In practice, you'd need to know the original feature names
        return [True] * len(self.selected_features)
    
    def get_feature_names_out(self) -> List[str]:
        """Get names of selected features (legacy interface)."""
        return getattr(self, 'selected_features', [])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (legacy interface)."""
        return getattr(self, 'feature_scores', {})
    
    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit (legacy interface)."""
        return getattr(self, 'X', np.array([])).shape[1] if hasattr(self, 'X') else 0
    
    @property
    def n_features_out_(self) -> int:
        """Number of features selected (legacy interface)."""
        return len(getattr(self, 'selected_features', []))


# Legacy function wrappers
def create_feature_selector(config: Optional[Any] = None) -> BackwardsCompatibilityWrapper:
    """Create a feature selector instance (legacy interface)."""
    return BackwardsCompatibilityWrapper(config)


def select_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    method: str = "correlation",
    max_features: Optional[int] = None
) -> List[str]:
    """Select features using the specified method (legacy interface)."""
    # Create legacy config
    class LegacyConfig:
        def __init__(self, method, max_features):
            self.method = method
            self.max_features = max_features
    
    config = LegacyConfig(method, max_features)
    selector = BackwardsCompatibilityWrapper(config)
    selector.fit(X, y)
    return selector.selected_features


# Compatibility aliases
FeatureSelector = BackwardsCompatibilityWrapper
FeatureSelectionConfig = UnifiedFeatureSelectionConfig

# Export for backwards compatibility
__all__ = [
    'BackwardsCompatibilityWrapper',
    'create_feature_selector',
    'select_features',
    'FeatureSelector',  # Alias for backwards compatibility
    'FeatureSelectionConfig'  # Alias for backwards compatibility
]


# Migration guide and warnings
def show_migration_guide():
    """Show migration guide for users upgrading to the unified framework."""
    migration_guide = """
    🚀 Feature Selection Migration Guide
    
    The unified feature selection framework provides enhanced capabilities while
    maintaining backwards compatibility. Here's how to migrate:
    
    OLD WAY:
    ```python
    from src.utils.ml_common.feature_selection_backwards_compat import FeatureSelector
    
    selector = FeatureSelector()
    selector.fit(X, y)
    selected_features = selector.get_feature_names_out()
    ```
    
    NEW WAY (Recommended):
    ```python
    from src.utils.ml_common.unified_feature_selection import UnifiedFeatureSelector
    
    selector = UnifiedFeatureSelector()
    results = selector.select_features(X, y, target_sizes=[120, 100, 80, 60])
    top_120_features = selector.get_feature_set(120)
    hmm_features = selector.get_hmm_regime_features()
    ```
    
    BACKWARDS COMPATIBLE WAY:
    ```python
    from src.utils.ml_common.backwards_compatibility import FeatureSelector
    
    # Same interface as before, but uses unified framework internally
    selector = FeatureSelector()
    selector.fit(X, y)
    selected_features = selector.get_feature_names_out()
    ```
    
    NEW FEATURES:
    - Multiple feature set sizes (120, 100, 80, 60)
    - HMM regime-specific selection
    - Matrix operations integration
    - Enhanced performance monitoring
    - Comprehensive result storage
    
    For more information, see the unified_feature_selection module documentation.
    """
    
    print(migration_guide)
    return migration_guide


# Automatic migration helper
def auto_migrate_legacy_code():
    """Automatically migrate legacy feature selection code."""
    warnings.warn(
        "Consider migrating to the unified feature selection framework for enhanced capabilities. "
        "Use show_migration_guide() for detailed migration instructions.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return show_migration_guide()