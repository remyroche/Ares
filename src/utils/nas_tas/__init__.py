"""
NAS TAS (Neural Architecture Search - Tree Architecture Search) Utilities

This module provides common utilities for NAS and TAS operations including
backtesting engines, regime detection, and optimization tools.
"""

from .unified_evaluator import (
    UnifiedEvaluator,
    EvaluationConfig,
    EvaluationResult,
    ModelType,
    EvaluationMode,
    MetricType
)

from .unified_multi_objective import (
    UnifiedMultiObjectiveOptimizer,
    PerformanceEstimator,
    ArchitectureFeatures,
    PerformancePrediction,
    PerformanceMetric,
    EstimatorType,
    OptimizationConfig,
    MultiObjectiveResult
)

from .nas_feature_extractor import (
    NASFeatureExtractor,
    NASFeatureConfig,
    FeatureExtractionResult,
    create_nas_feature_extractor,
    extract_features_for_clustering
)

from .bayesian_search import (
    BayesianTreeSearch,
    TreeBayesianOptimizer,
    TreeGaussianProcess,
    BayesianConfig
)

from .search_space import (
    SearchSpace,
    SearchSpaceConfig,
    ParameterRange,
    SearchSpaceType,
    OptimizationStrategy,
    create_default_nas_search_space,
    create_tree_search_space
)

from .risk_analysis import (
    RiskAnalyzer,
    RiskConfig,
    RiskResult,
    RiskMetric
)

from .backtesting_engine import (
    BacktestingEngine,
    BacktestingConfig,
    BacktestingResult,
    BacktestingMode
)


__all__ = [
    'UnifiedEvaluator',
    'EvaluationConfig', 
    'EvaluationResult',
    'ModelType',
    'EvaluationMode',
    'MetricType',
    'UnifiedMultiObjectiveOptimizer',
    'PerformanceEstimator',
    'ArchitectureFeatures',
    'PerformancePrediction',
    'PerformanceMetric',
    'EstimatorType',
    'OptimizationConfig',
    'MultiObjectiveResult',
    'NASFeatureExtractor',
    'NASFeatureConfig',
    'FeatureExtractionResult',
    'create_nas_feature_extractor',
    'extract_features_for_clustering',
    'BayesianTreeSearch',
    'TreeBayesianOptimizer',
    'TreeGaussianProcess',
    'BayesianConfig',
    'SearchSpace',
    'SearchSpaceConfig',
    'ParameterRange',
    'SearchSpaceType',
    'OptimizationStrategy',
    'create_default_nas_search_space',
    'create_tree_search_space',
    'RiskAnalyzer',
    'RiskConfig',
    'RiskResult',
    'RiskMetric'
    'create_tree_search_space'
    'BacktestingEngine',
    'BacktestingConfig', 
    'BacktestingResult',
    'BacktestingMode'
]