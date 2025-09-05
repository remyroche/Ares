"""
from .step06_feature_engineering import FeatureEngineeringStep
Feature Engineering Step06 Components

This package contains feature engineering components for step06 including:
- Feature engineering step implementation
- Technical indicator extraction
- Feature interaction creation
- Feature selection and validation
"""

try:
    FEATURE_ENGINEERING_STEP_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_STEP_AVAILABLE = False

__all__ = [
    'FeatureEngineeringStep',
    'FEATURE_ENGINEERING_STEP_AVAILABLE'
]