"""
Shared Feature Engineering Module

This module provides shared feature engineering utilities that are used
in both training and inference (signal generation) to ensure consistency.
"""

from .feature_engineer import (
    FeatureEngineer,
    AnalystFeatureEngineer,
    TacticianFeatureEngineer,
    engineer_analyst_features,
    engineer_tactician_features
)
from .feature_validator import (
    FeatureValidator,
    validate_feature_set,
    compare_feature_sets
)

__all__ = [
    'FeatureEngineer',
    'AnalystFeatureEngineer',
    'TacticianFeatureEngineer',
    'engineer_analyst_features',
    'engineer_tactician_features',
    'FeatureValidator',
    'validate_feature_set',
    'compare_feature_sets',
]
