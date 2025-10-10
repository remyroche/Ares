"""
Feature Comparison Framework

This module provides utilities to compare different feature engineering approaches
including VWAP-based features, volatility normalization, and their combinations.

Enhanced with standardized feature definitions, consolidation, and validation.
"""

from .feature_comparison_utils import FeatureComparisonUtils
from .relevance_analyzer import RelevanceAnalyzer
from .comparison_report import ComparisonReport
from .feature_versions import FeatureVersions
from .optimized_feature_versions import OptimizedFeatureVersions
from .standardized_features import StandardizedFeatureGenerator
from .feature_consolidation import FeatureConsolidator, FeatureValidator
from .enhanced_comparison_runner import EnhancedFeatureComparisonRunner
from .robust_scaling import RobustFeatureScaler, MultiMethodScaler

__all__ = [
    'FeatureComparisonUtils',
    'RelevanceAnalyzer', 
    'ComparisonReport',
    'FeatureVersions',
    'OptimizedFeatureVersions',
    'StandardizedFeatureGenerator',
    'FeatureConsolidator',
    'FeatureValidator',
    'EnhancedFeatureComparisonRunner',
    'RobustFeatureScaler',
    'MultiMethodScaler'
]