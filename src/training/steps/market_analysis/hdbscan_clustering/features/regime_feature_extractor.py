"""
Regime Feature Extractor for HDBSCAN clustering.

This module provides a RegimeFeatureExtractor class that wraps the centralized
RegimeFeatureGenerator from the feature generation system.
"""

from src.feature_generation.categories.regime_features import RegimeFeatureGenerator

# Alias for backward compatibility with the HDBSCAN clustering system
RegimeFeatureExtractor = RegimeFeatureGenerator

__all__ = ['RegimeFeatureExtractor']
