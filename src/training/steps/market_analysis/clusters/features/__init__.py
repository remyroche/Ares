"""
Features Module for Market Analysis Clustering.

This module provides comprehensive feature management capabilities including:
- Feature selection and filtering
- Feature preprocessing and scaling
- Feature analysis and diagnostics

Classes:
    FeatureSelector: Select and filter features based on various criteria
    FeaturePreprocessor: Preprocess features (scaling, NaN handling, dimensionality reduction)
    FeatureAnalyzer: Analyze feature importance, correlations, and stability
"""

from .selector import FeatureSelector, FeatureSelectorConfig
from .preprocessor import FeaturePreprocessor, FeaturePreprocessorConfig
from .analyzer import FeatureAnalyzer, FeatureAnalyzerConfig

__all__ = [
    'FeatureSelector',
    'FeatureSelectorConfig',
    'FeaturePreprocessor',
    'FeaturePreprocessorConfig',
    'FeatureAnalyzer',
    'FeatureAnalyzerConfig',
]
