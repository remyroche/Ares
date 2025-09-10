"""
Compatibility module for step08_advanced_feature_selection imports.

This module provides backwards compatibility for imports that were previously
available in the deleted step08_advanced_feature_selection.py file.

All functionality has been moved to ml_commons.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

# Import the actual functionality from the new locations
try:
    from src.utils.ml_common.feature_selection import (
        UnifiedFeatureSelectionManager,
        AdvancedFeatureSelector
    )
    FUNCTIONALITY_AVAILABLE = True
except ImportError as e:
    FUNCTIONALITY_AVAILABLE = False
    warnings.warn(f"Step08 functionality not available: {e}")


class Step08AdvancedFeatureSelection:
    """
    Compatibility class for Step08AdvancedFeatureSelection.
    
    This class provides backwards compatibility for the Step08AdvancedFeatureSelection
    that was previously available in step08_advanced_feature_selection.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced feature selection."""
        if not FUNCTIONALITY_AVAILABLE:
            raise ImportError("Step08 functionality not available. Please ensure ml_commons is properly installed.")
        
        self.config = config or {}
        self.feature_selector = UnifiedFeatureSelectionManager(self.config)
        
        warnings.warn(
            "Step08AdvancedFeatureSelection is deprecated. Use UnifiedFeatureSelectionManager from src.utils.ml_common.feature_selection instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def select_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Select features using the unified feature selection manager."""
        return self.feature_selector.select_features(X, y, **kwargs)
    
    def rank_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Rank features using the unified feature selection manager."""
        return self.feature_selector.rank_features(X, y, **kwargs)
    
    def filter_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Filter features using the unified feature selection manager."""
        return self.feature_selector.filter_features(X, y, **kwargs)


class AdvancedFeatureSelectionStep:
    """
    Compatibility class for AdvancedFeatureSelectionStep.
    
    This class provides backwards compatibility for the AdvancedFeatureSelectionStep
    that was previously available in step08_advanced_feature_selection.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced feature selection step."""
        if not FUNCTIONALITY_AVAILABLE:
            raise ImportError("Step08 functionality not available. Please ensure ml_commons is properly installed.")
        
        self.config = config or {}
        self.feature_selector = UnifiedFeatureSelectionManager(self.config)
        
        warnings.warn(
            "AdvancedFeatureSelectionStep is deprecated. Use UnifiedFeatureSelectionManager from src.utils.ml_common.feature_selection instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute feature selection using the unified feature selection manager."""
        return self.feature_selector.execute(data, **kwargs)
    
    def select_best_features(self, X: Any, y: Any, **kwargs) -> Any:
        """Select best features using the unified feature selection manager."""
        return self.feature_selector.select_best_features(X, y, **kwargs)


# Backwards compatibility exports
__all__ = [
    'Step08AdvancedFeatureSelection',
    'AdvancedFeatureSelectionStep',
    'FUNCTIONALITY_AVAILABLE'
]