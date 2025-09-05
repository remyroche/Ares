"""
Data Collection Step06 Components

This package contains data collection components for step06 including:
- Feature engineering step
- Data preprocessing
- Feature selection
- Data validation
"""

try:
    from .feature_engineering.step06_feature_engineering import FeatureEngineeringStep
    FEATURE_ENGINEERING_STEP_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_STEP_AVAILABLE = False

__all__ = [
    'FeatureEngineeringStep',
    'FEATURE_ENGINEERING_STEP_AVAILABLE'
]