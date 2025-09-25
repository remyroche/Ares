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

# Training-related utilities
from .regime_aware_trainer import (
    RegimeAwareTrainer,
    RegimeAwareTrainingConfig,
    RegimeTrainingResult,
    ModelType,
    RegimeTrainingStrategy
)

from .training_orchestrator import (
    TrainingOrchestrator,
    OrchestratorConfig,
    OrchestrationResult,
    OrchestrationMode
)

from .model_selector import (
    ModelSelector,
    ModelSelectionConfig,
    ModelSelectionResult,
    SelectionStrategy,
    RoutingMethod
)

from .model_manager import (
    ModelManager,
    ModelManagerConfig,
    ModelMetadata,
    ModelDeploymentResult,
    ModelStatus,
    DeploymentStrategy
)

from .performance_tracker import (
    PerformanceTracker,
    PerformanceConfig,
    PerformanceRecord,
    PerformanceAlert,
    PerformanceReport,
    PerformanceMetric,
    AlertType
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
    # Training-related utilities
    'RegimeAwareTrainer',
    'RegimeAwareTrainingConfig',
    'RegimeTrainingResult',
    'ModelType',
    'RegimeTrainingStrategy',
    'TrainingOrchestrator',
    'OrchestratorConfig',
    'OrchestrationResult',
    'OrchestrationMode',
    'ModelSelector',
    'ModelSelectionConfig',
    'ModelSelectionResult',
    'SelectionStrategy',
    'RoutingMethod',
    'ModelManager',
    'ModelManagerConfig',
    'ModelMetadata',
    'ModelDeploymentResult',
    'ModelStatus',
    'DeploymentStrategy',
    'PerformanceTracker',
    'PerformanceConfig',
    'PerformanceRecord',
    'PerformanceAlert',
    'PerformanceReport',
    'PerformanceMetric',
    'AlertType'
]