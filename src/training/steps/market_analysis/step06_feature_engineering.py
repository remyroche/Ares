"""
Compatibility module for step06_feature_engineering imports.

This module provides backwards compatibility for imports that were previously
available in the deleted step06_feature_engineering.py file.

All functionality has been moved to ml_commons and step06_utilities.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

# Import the actual functionality from the new locations
try:
    from src.utils.step06_utilities import (
        EnhancedFeatureEngineeringStep,
        EnhancedFeatureEngineering
    )
    from src.utils.ml_common.feature_selection import (
        UnifiedFeatureSelectionManager
    )
    from src.utils.ml_common.matrix_operations import (
        get_enhanced_matrix_operations
    )
    FUNCTIONALITY_AVAILABLE = True
except ImportError as e:
    FUNCTIONALITY_AVAILABLE = False
    warnings.warn(f"Step06 functionality not available: {e}")


class FeatureInteractionEngine:
    """
    Compatibility class for FeatureInteractionEngine.
    
    This class provides backwards compatibility for the FeatureInteractionEngine
    that was previously available in step06_feature_engineering.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature interaction engine."""
        if not FUNCTIONALITY_AVAILABLE:
            raise ImportError("Step06 functionality not available. Please ensure ml_commons and step06_utilities are properly installed.")
        
        self.config = config or {}
        self.enhanced_engine = EnhancedFeatureEngineeringStep(self.config)
        self.feature_selector = UnifiedFeatureSelectionManager(self.config)
        self.matrix_ops = get_enhanced_matrix_operations()
        
        warnings.warn(
            "FeatureInteractionEngine is deprecated. Use EnhancedFeatureEngineeringStep from src.utils.step06_utilities instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def create_feature_interactions(self, data: Any, **kwargs) -> Any:
        """Create feature interactions using the enhanced feature engineering."""
        return self.enhanced_engine.create_feature_interactions(data, **kwargs)
    
    def extract_technical_features(self, data: Any, **kwargs) -> Any:
        """Extract technical features using the enhanced feature engineering."""
        return self.enhanced_engine.extract_technical_features(data, **kwargs)
    
    def create_regime_features(self, data: Any, **kwargs) -> Any:
        """Create regime-aware features using the enhanced feature engineering."""
        return self.enhanced_engine.create_regime_features(data, **kwargs)


class FeatureEngineeringStep:
    """
    Compatibility class for FeatureEngineeringStep.
    
    This class provides backwards compatibility for the FeatureEngineeringStep
    that was previously available in step06_feature_engineering.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature engineering step."""
        if not FUNCTIONALITY_AVAILABLE:
            raise ImportError("Step06 functionality not available. Please ensure ml_commons and step06_utilities are properly installed.")
        
        self.config = config or {}
        self.enhanced_engine = EnhancedFeatureEngineeringStep(self.config)
        
        warnings.warn(
            "FeatureEngineeringStep is deprecated. Use EnhancedFeatureEngineeringStep from src.utils.step06_utilities instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute feature engineering using the enhanced feature engineering."""
        return self.enhanced_engine.execute(data, **kwargs)
    
    def create_features(self, data: Any, **kwargs) -> Any:
        """Create features using the enhanced feature engineering."""
        return self.enhanced_engine.create_features(data, **kwargs)


# Backwards compatibility exports
__all__ = [
    'FeatureInteractionEngine',
    'FeatureEngineeringStep',
    'FUNCTIONALITY_AVAILABLE'
]