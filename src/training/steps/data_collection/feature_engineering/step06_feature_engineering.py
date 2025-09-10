"""
Compatibility module for step06_feature_engineering imports in data collection.

This module provides backwards compatibility for imports that were previously
available in the deleted step06_feature_engineering.py file in data collection.

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
    FUNCTIONALITY_AVAILABLE = True
except ImportError as e:
    FUNCTIONALITY_AVAILABLE = False
    warnings.warn(f"Step06 functionality not available: {e}")


class FeatureEngineeringStep:
    """
    Compatibility class for FeatureEngineeringStep in data collection.
    
    This class provides backwards compatibility for the FeatureEngineeringStep
    that was previously available in data collection step06_feature_engineering.py.
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
    
    def extract_basic_features(self, data: Any, **kwargs) -> Any:
        """Extract basic features using the enhanced feature engineering."""
        return self.enhanced_engine.extract_basic_features(data, **kwargs)


# Backwards compatibility exports
__all__ = [
    'FeatureEngineeringStep',
    'FUNCTIONALITY_AVAILABLE'
]