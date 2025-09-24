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
from .position_aware_trading import (
    PositionAwareTradingAnalyzer, PositionAwareConfig, PositionAwareResult,
    create_position_aware_analyzer, quick_position_aware_analysis
)
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

# New Unified Utilities
from .unified_economic_evaluator import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig, EconomicSignificanceResult,
    create_unified_economic_evaluator, quick_economic_evaluation
)
from .unified_trading_viability_evaluator import (
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig, TradingViabilityResult,
    create_unified_trading_viability_evaluator, quick_trading_viability_evaluation
)
from .unified_multi_objective_optimizer import (
    UnifiedMultiObjectiveOptimizer, OptimizationConfig, OptimizationResult,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization
)
from .unified_hardware_optimizer import (
    UnifiedHardwareOptimizer, HardwareConfig, PerformanceMetrics,
    create_unified_hardware_optimizer, quick_hardware_optimization
)
from .unified_regime_analyzer import (
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig, RegimeAnalysisResult,
    create_unified_regime_analyzer, quick_regime_analysis
)
from .unified_config_manager import (
    UnifiedConfigManager, UnifiedRegimeConfig,
    create_unified_config_manager, load_config_from_file, create_environment_config
)
from .unified_validation_system import (
    UnifiedValidationSystem, ValidationConfig, ValidationResult,
    create_unified_validation_system, quick_validation
)

# ML Common Integration - Shared utilities for TAS and NAS
from .ml_common_integration import (
    SharedMLUtilitiesManager, create_shared_ml_utilities_manager,
    TASSharedMLUtilities, NASSharedMLUtilities, HybridSharedMLUtilities,
    MLUtilityType, MLUtilityConfig
)

# New Unified Architecture Management
from .unified_architecture_config import (
    BaseArchitectureConfig, TASArchitectureConfig, NASArchitectureConfig, HybridArchitectureConfig,
    ArchitectureType, SearchStrategy, OptimizationObjective, MarketRegime,
    create_tas_config, create_nas_config, create_hybrid_config,
    create_quick_config, create_comprehensive_config
)

from .unified_performance_monitor import (
    UnifiedPerformanceMonitor, PerformanceMetric, MonitoringLevel,
    PerformanceSnapshot, PerformanceTrend, RegimePerformanceProfile,
    create_performance_monitor, create_basic_monitor, create_real_time_monitor
)

from .unified_meta_learning import (
    UnifiedMetaLearner, MetaLearningMethod, AdaptationType,
    MetaTask, AdaptationResult, MetaLearningConfig,
    TASMetaModel, NASMetaModel, HybridMetaModel,
    create_meta_learner, create_few_shot_learner, create_continual_learner
)

from .unified_hardware_manager import (
    UnifiedHardwareManager, HardwareType, OptimizationLevel, WorkloadType,
    HardwareMetrics, OptimizationResult, HardwareConfig,
    create_hardware_manager, create_basic_hardware_manager, create_aggressive_hardware_manager
)

from .unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationType, EvaluationMetric,
    EvaluationResult, EvaluationConfig,
    create_evaluation_framework, create_basic_evaluator, create_trading_evaluator
)

__all__ = [
    # Original utilities
    'DataPipelineManager', 'MarketDataProcessor',
    'FeatureCollectionManager', 'StandardizedFeatureCalculator',
    'EconomicSignificanceEvaluator', 'EconomicSignificanceResult',
    'TradingViabilityEvaluator', 'TradingViabilityResult',

    # Position-Aware Trading
    'PositionAwareTradingAnalyzer', 'PositionAwareConfig', 'PositionAwareResult',
    'create_position_aware_analyzer', 'quick_position_aware_analysis',

    # Search Strategies
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
    'UnifiedClusteringAlgorithm', 'create_unified_clustering_algorithm',

    # New Unified Utilities
    'UnifiedEconomicSignificanceEvaluator', 'EconomicEvaluationConfig', 'EconomicSignificanceResult',
    'UnifiedTradingViabilityEvaluator', 'TradingViabilityConfig', 'TradingViabilityResult',
    'UnifiedMultiObjectiveOptimizer', 'OptimizationConfig', 'OptimizationResult',
    'UnifiedHardwareOptimizer', 'HardwareConfig', 'PerformanceMetrics',
    'UnifiedRegimeAnalyzer', 'RegimeAnalysisConfig', 'RegimeAnalysisResult',
    'UnifiedConfigManager', 'UnifiedRegimeConfig',
    'UnifiedValidationSystem', 'ValidationConfig', 'ValidationResult',

    # Convenience functions for new utilities
    'create_unified_economic_evaluator', 'quick_economic_evaluation',
    'create_unified_trading_viability_evaluator', 'quick_trading_viability_evaluation',
    'create_unified_multi_objective_optimizer', 'quick_multi_objective_optimization',
    'create_unified_hardware_optimizer', 'quick_hardware_optimization',
    'create_unified_regime_analyzer', 'quick_regime_analysis',
    'create_unified_config_manager', 'load_config_from_file', 'create_environment_config',
    'create_unified_validation_system', 'quick_validation',

    # ML Common Integration - Shared utilities for TAS and NAS
    'SharedMLUtilitiesManager', 'create_shared_ml_utilities_manager',
    'TASSharedMLUtilities', 'NASSharedMLUtilities', 'HybridSharedMLUtilities',
    'MLUtilityType', 'MLUtilityConfig',

    # New Unified Architecture Management
    'BaseArchitectureConfig', 'TASArchitectureConfig', 'NASArchitectureConfig', 'HybridArchitectureConfig',
    'ArchitectureType', 'SearchStrategy', 'OptimizationObjective', 'MarketRegime',
    'create_tas_config', 'create_nas_config', 'create_hybrid_config',
    'create_quick_config', 'create_comprehensive_config',

    # Unified Performance Monitoring
    'UnifiedPerformanceMonitor', 'PerformanceMetric', 'MonitoringLevel',
    'PerformanceSnapshot', 'PerformanceTrend', 'RegimePerformanceProfile',
    'create_performance_monitor', 'create_basic_monitor', 'create_real_time_monitor',

    # Unified Meta-Learning
    'UnifiedMetaLearner', 'MetaLearningMethod', 'AdaptationType',
    'MetaTask', 'AdaptationResult', 'MetaLearningConfig',
    'TASMetaModel', 'NASMetaModel', 'HybridMetaModel',
    'create_meta_learner', 'create_few_shot_learner', 'create_continual_learner',

    # Unified Hardware Management
    'UnifiedHardwareManager', 'HardwareType', 'OptimizationLevel', 'WorkloadType',
    'HardwareMetrics', 'OptimizationResult', 'HardwareConfig',
    'create_hardware_manager', 'create_basic_hardware_manager', 'create_aggressive_hardware_manager',

    # Unified Evaluation Framework
    'UnifiedEvaluationFramework', 'EvaluationType', 'EvaluationMetric',
    'EvaluationResult', 'EvaluationConfig',
    'create_evaluation_framework', 'create_basic_evaluator', 'create_trading_evaluator'
]