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
from .time_series_validation import TimeSeriesValidator, PurgedGroupKFold, WalkForwardValidator, OutOfSampleValidator
from .stability_metrics import FeatureStabilityAnalyzer
from .diagnostics import FeatureDiagnostics
from .method_settings import MethodSettings
from .enhanced_relevance_analyzer import EnhancedRelevanceAnalyzer
from .analyst_labeler_integration import AnalystLabelerIntegration
from .pre_screening_pipeline import PreScreeningPipeline
from .feature_scorecard import FeatureScorecard
from .compute_aware_optimizer import ComputeAwareOptimizer
from .family_diverse_features import FamilyDiverseFeatureGenerator
from .feature_acceleration_dilation import FeatureAccelerationDilation

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
    'MultiMethodScaler',
    'TimeSeriesValidator',
    'PurgedGroupKFold',
    'WalkForwardValidator',
    'OutOfSampleValidator',
    'FeatureStabilityAnalyzer',
    'FeatureDiagnostics',
    'MethodSettings',
    'EnhancedRelevanceAnalyzer',
    'AnalystLabelerIntegration',
    'PreScreeningPipeline',
    'FeatureScorecard',
    'ComputeAwareOptimizer',
    'FamilyDiverseFeatureGenerator',
    'FeatureAccelerationDilation'
]