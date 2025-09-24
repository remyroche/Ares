"""
Shared utilities for hybrid NAS-TAS regime detection.

This module provides comprehensive utilities used by both NAS and TAS regime detection systems,
including:
- Feature collection using pre-existing feature_generator/
- Economic significance evaluation
- Trading viability assessment
- Data pipeline utilities
- Advanced Search Strategies (RL, Bayesian, Evolutionary)
- Search space definitions for NAS and TAS
- Performance estimators and surrogate models
- Architecture encoding/decoding systems
- Constraint validation systems
- Bayesian optimization with grid optimization
- Advanced evolutionary algorithms (NSGA-II, SPEA2)
- Hardware optimization based on hardware/
- Advanced Analysis Components
- Metrics reporting and analysis
"""

# Original utilities
from .data_pipeline import DataPipelineManager, MarketDataProcessor
from .feature_collection import FeatureCollectionManager, StandardizedFeatureCalculator
from .economic_significance import EconomicSignificanceEvaluator, EconomicSignificanceResult
from .trading_viability import TradingViabilityEvaluator, TradingViabilityResult
from .search_strategies import AdvancedSearchStrategy, BayesianOptimizer, GridOptimizer
from .evolutionary_algorithms import NSGA2Optimizer, SPEA2Optimizer, EvolutionaryAlgorithm
from .hardware_optimization import HardwareOptimizer, PerformanceMonitor
from .analysis_components import AdvancedAnalysisComponent, RegimeAnalyzer, ClusterAnalyzer
from .metrics_reporting import MetricsReporter, ConsolidatedMetricsReport

# New comprehensive shared utilities
from .search_spaces import (
    NeuralSearchSpace, TreeSearchSpace, ArchitectureConstraints,
    NeuralArchitecture, TreeArchitecture, LayerSpecification, TreeSpecification,
    create_neural_search_space, create_tree_search_space
)
from .performance_estimators import (
    UnifiedPerformanceEstimator, NeuralPerformanceEstimator, TreePerformanceEstimator,
    ArchitectureFeatures, PerformancePrediction,
    create_unified_performance_estimator, create_neural_performance_estimator, create_tree_performance_estimator
)
from .architecture_encoders import (
    UnifiedArchitectureEncoder, NeuralArchitectureEncoder, TreeArchitectureEncoder,
    EncodingResult, DecodingResult,
    create_unified_architecture_encoder, create_neural_architecture_encoder, create_tree_architecture_encoder
)
from .constraint_systems import (
    UnifiedConstraintValidator, NeuralConstraintValidator, TreeConstraintValidator,
    ConstraintValidationResult, ConstraintViolation,
    create_unified_constraint_validator, create_neural_constraint_validator, create_tree_constraint_validator
)
from .advanced_search_strategies import (
    ReinforcementLearningSearch, EnhancedBayesianOptimization, AdaptiveEvolutionarySearch,
    RLState, RLAction, RLReward, SearchStrategyResult,
    create_rl_search_strategy, create_enhanced_bayesian_search, create_adaptive_evolutionary_search
)
from .unified_search_algorithms import (
    UnifiedSearchManager, BayesianOptimizationSearch, EvolutionaryAlgorithmSearch,
    create_unified_search_manager, create_search_algorithm
)
from .unified_clustering_algorithms import (
    UnifiedClusteringAlgorithm, create_unified_clustering_algorithm
)

__all__ = [
    # Original utilities
    'DataPipelineManager', 'MarketDataProcessor',
    'FeatureCollectionManager', 'StandardizedFeatureCalculator',
    'EconomicSignificanceEvaluator', 'EconomicSignificanceResult',
    'TradingViabilityEvaluator', 'TradingViabilityResult',
    'AdvancedSearchStrategy', 'BayesianOptimizer', 'GridOptimizer',
    'NSGA2Optimizer', 'SPEA2Optimizer', 'EvolutionaryAlgorithm',
    'HardwareOptimizer', 'PerformanceMonitor',
    'AdvancedAnalysisComponent', 'RegimeAnalyzer', 'ClusterAnalyzer',
    'MetricsReporter', 'ConsolidatedMetricsReport',

    # New comprehensive shared utilities - Search Spaces
    'NeuralSearchSpace', 'TreeSearchSpace', 'ArchitectureConstraints',
    'NeuralArchitecture', 'TreeArchitecture', 'LayerSpecification', 'TreeSpecification',
    'create_neural_search_space', 'create_tree_search_space',

    # Performance Estimators
    'UnifiedPerformanceEstimator', 'NeuralPerformanceEstimator', 'TreePerformanceEstimator',
    'ArchitectureFeatures', 'PerformancePrediction',
    'create_unified_performance_estimator', 'create_neural_performance_estimator', 'create_tree_performance_estimator',

    # Architecture Encoders
    'UnifiedArchitectureEncoder', 'NeuralArchitectureEncoder', 'TreeArchitectureEncoder',
    'EncodingResult', 'DecodingResult',
    'create_unified_architecture_encoder', 'create_neural_architecture_encoder', 'create_tree_architecture_encoder',

    # Constraint Systems
    'UnifiedConstraintValidator', 'NeuralConstraintValidator', 'TreeConstraintValidator',
    'ConstraintValidationResult', 'ConstraintViolation',
    'create_unified_constraint_validator', 'create_neural_constraint_validator', 'create_tree_constraint_validator',

    # Advanced Search Strategies
    'ReinforcementLearningSearch', 'EnhancedBayesianOptimization', 'AdaptiveEvolutionarySearch',
    'RLState', 'RLAction', 'RLReward', 'SearchStrategyResult',
    'create_rl_search_strategy', 'create_enhanced_bayesian_search', 'create_adaptive_evolutionary_search',

    # Unified Search and Clustering
    'UnifiedSearchManager', 'BayesianOptimizationSearch', 'EvolutionaryAlgorithmSearch',
    'create_unified_search_manager', 'create_search_algorithm',
    'UnifiedClusteringAlgorithm', 'create_unified_clustering_algorithm'
]