"""
Feature Interaction Constructor Module

This module provides infrastructure for mapping selected interaction features
back to their base features and calculating them automatically for live trading.
"""

from src.interaction_features_constructor.feature_decomposer import FeatureDecomposer
from src.interaction_features_constructor.feature_calculator import FeatureCalculator
from src.interaction_features_constructor.feature_metadata_store import FeatureMetadataStore

__all__ = [
    'FeatureDecomposer',
    'FeatureCalculator',
    'FeatureMetadataStore',
]
