"""
Market Analysis Step06 Components

This package contains market analysis components for step06 including:
- Feature interaction engineering
- Technical indicator extraction
- Correlation analysis
- Regime-aware feature engineering
"""

try:
    from .step06_feature_engineering import FeatureInteractionEngine
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

__all__ = [
    'FeatureInteractionEngine',
    'FEATURE_ENGINEERING_AVAILABLE'
]