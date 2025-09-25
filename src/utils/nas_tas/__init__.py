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

from .evolutionary_search import (
    EvolutionaryTreeSearch,
    TreeGeneticAlgorithm,
    TreeNSGA2,
    EvolutionaryConfig
)

from .uncertainty_estimation import (
    TreeUncertaintyEstimator,
    TreeEnsembleUncertainty,
    TreeBayesianUncertainty,
    UncertaintyConfig
)

from .confidence_scoring import (
    TreeConfidenceScorer,
    TreeReliabilityEstimator,
    TreeCalibrationScorer,
    ConfidenceConfig
)

# Import NAS modules
from .nas import (
    NeuralArchitectureSearch,
    ArchitectureConfig,
    ArchitectureCandidate,
    ArchitectureSearchSpace,
    search_neural_architecture,
    AdaptiveRegimeNAS,
    AdaptiveRegimeNASConfig,
    RegimeDetector
)

# Import TAS modules
from .tas import (
    TreeBasedArchitectureSearch,
    TreeArchitectureConfig,
    TreeArchitectureCandidate,
    TreeArchitectureSearchSpace,
    search_tree_architecture,
    PureTreeNAS,
    PureTreeNASConfig,
    UnsupervisedTreeNAS,
    UnsupervisedTreeNASConfig,
    RegimeTradingTreeNAS,
    RegimeTradingTreeNASConfig,
    TradingTreeArchitectureSearch,
    TradingTASConfig,
    TradingRegime,
    TradingTASResult,
    TradingObjective,
    MarketRegime
)

# Import Hybrid NAS System
from .hybrid_nas_system import (
    HybridNASSystem,
    HybridNASConfig,
    HybridArchitectureCandidate,
    optimize_hybrid_architecture,
    analyze_data_characteristics
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
    'RiskMetric',
    'BacktestingEngine',
    'BacktestingConfig', 
    'BacktestingResult',
    'BacktestingMode',
    'EvolutionaryTreeSearch',
    'TreeGeneticAlgorithm',
    'TreeNSGA2',
    'EvolutionaryConfig',
    'TreeUncertaintyEstimator',
    'TreeEnsembleUncertainty',
    'TreeBayesianUncertainty',
    'UncertaintyConfig',
    'TreeConfidenceScorer',
    'TreeReliabilityEstimator',
    'TreeCalibrationScorer',
    'ConfidenceConfig',
    
    # NAS modules
    'NeuralArchitectureSearch',
    'ArchitectureConfig',
    'ArchitectureCandidate',
    'ArchitectureSearchSpace',
    'search_neural_architecture',
    'AdaptiveRegimeNAS',
    'AdaptiveRegimeNASConfig',
    'RegimeDetector',
    
    # TAS modules
    'TreeBasedArchitectureSearch',
    'TreeArchitectureConfig',
    'TreeArchitectureCandidate',
    'TreeArchitectureSearchSpace',
    'search_tree_architecture',
    'PureTreeNAS',
    'PureTreeNASConfig',
    'UnsupervisedTreeNAS',
    'UnsupervisedTreeNASConfig',
    'RegimeTradingTreeNAS',
    'RegimeTradingTreeNASConfig',
    'TradingTreeArchitectureSearch',
    'TradingTASConfig',
    'TradingRegime',
    'TradingTASResult',
    'TradingObjective',
    'MarketRegime',
    
    # Hybrid NAS System
    'HybridNASSystem',
    'HybridNASConfig',
    'HybridArchitectureCandidate',
    'optimize_hybrid_architecture',
    'analyze_data_characteristics'
]