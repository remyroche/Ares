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

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Missing config classes as placeholders
@dataclass
class EconomicSignificanceConfig:
    """Configuration for economic significance evaluation."""
    significance_threshold: float = 0.5
    min_regime_duration: int = 10
    enable_position_aware_analysis: bool = True
    enable_economic_indicators: bool = True
    enable_bootstrap_analysis: bool = True

# TradingViabilityConfig is now imported from unified_trading_viability_evaluator

@dataclass
class SearchStrategyConfig:
    """Configuration for search strategies."""
    max_iterations: int = 100
    use_bayesian_optimization: bool = True

@dataclass
class EvolutionaryAlgorithmConfig:
    """Configuration for evolutionary algorithms."""
    population_size: int = 50
    max_generations: int = 100
    use_nsga2: bool = True
    use_spea2: bool = True

@dataclass
class HardwareOptimizationConfig:
    """Configuration for hardware optimization."""
    use_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

@dataclass
class MetricsReportingConfig:
    """Configuration for metrics reporting."""
    include_detailed_metrics: bool = True
    include_visualization_data: bool = True
    include_performance_metrics: bool = True
    include_economic_metrics: bool = True
    include_trading_metrics: bool = True
    report_format: str = "json"  # "json", "csv", "html"
    save_to_file: bool = True
    output_directory: str = "reports"

# Placeholder managers
class EconomicSignificanceEvaluator:
    def __init__(self, config: EconomicSignificanceConfig):
        self.config = config

# TradingViabilityEvaluator is now imported from unified_trading_viability_evaluator

class SearchStrategyManager:
    def __init__(self, config: SearchStrategyConfig):
        self.config = config

class EvolutionaryAlgorithmManager:
    def __init__(self, config: EvolutionaryAlgorithmConfig):
        self.config = config

class HardwareOptimizer:
    def __init__(self, config: HardwareOptimizationConfig):
        self.config = config

# Original utilities
from .data_pipeline import DataPipelineManager, DataPipelineConfig, MarketDataProcessor
from .feature_collection import FeatureCollectionManager, FeatureCollectionConfig, StandardizedFeatureCalculator
# Removed redundant imports - now using unified versions
from .position_aware_trading import (
    PositionAwareTradingAnalyzer, PositionAwareConfig, PositionAwareResult,
    create_position_aware_analyzer, quick_position_aware_analysis
)
from .search_strategies import AdvancedSearchStrategy, BayesianOptimizer, GridOptimizer
from .evolutionary_algorithms import NSGA2Optimizer, SPEA2Optimizer, EvolutionaryAlgorithm
# from .hardware_optimization import HardwareOptimizer, PerformanceMonitor  # DEPRECATED
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

# Alias for compatibility
TradingViabilityEvaluator = UnifiedTradingViabilityEvaluator
from .unified_multi_objective_optimizer import (
    UnifiedMultiObjectiveOptimizer, OptimizationConfig, OptimizationResult,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization
)
from .unified_hardware_optimizer import (
    UnifiedHardwareOptimizer, HardwareConfig, PerformanceMetrics,
    create_unified_hardware_optimizer, quick_hardware_optimization
)
# Evaluation utilities are available in unified_evaluation_framework.py
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

# ML Common Integration - Integration with existing utilities
from .ml_common_integration import (
    MLCommonIntegration, MLCommonIntegrationType, MLCommonIntegrationConfig,
    create_ml_common_integration, create_tas_ml_common_integration,
    create_nas_ml_common_integration, create_hybrid_ml_common_integration
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

from .unified_ensemble_search_space import (
    UnifiedEnsembleSearchSpace, EnsembleArchitecture, EnsembleSearchResult,
    EnsembleMethod, EnsembleCombinationStrategy, EnsembleSearchSpaceConfig,
    create_unified_ensemble_search_space, quick_ensemble_search
)

from .unified_architecture_compression import (
    UnifiedArchitectureCompressor, CompressionResult, CompressionMethod, CompressionLevel, CompressionConfig,
    create_unified_architecture_compressor, quick_compress_architecture
)

from .unified_search_space_evolution import (
    UnifiedSearchSpaceEvolutionManager, EvolutionEvent, EvolutionResult, EvolutionTrigger, EvolutionAction,
    UnifiedEvolutionConfig, create_unified_evolution_manager, quick_evolution_setup
)

from .unified_hardware_manager import (
    UnifiedHardwareManager, HardwareType, CanonicalOptimizationLevel as OptimizationLevel, WorkloadType,
    HardwareMetrics, OptimizationResult, HardwareConfig,
    create_hardware_manager, create_basic_hardware_manager, create_aggressive_hardware_manager
)

from .unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationType, EvaluationMetric,
    EvaluationResult, EvaluationConfig,
    create_evaluation_framework, create_basic_evaluator, create_trading_evaluator
)

# Clustering Quality Metrics
from .clustering_quality_metrics import (
    ClusteringQualityMetrics, ClusteringQualityConfig, ClusteringQualityResult,
    create_clustering_quality_evaluator, quick_clustering_evaluation
)

__all__ = [
    # Original utilities
    'DataPipelineManager', 'MarketDataProcessor',
    'FeatureCollectionManager', 'StandardizedFeatureCalculator',
    # Removed redundant exports - now using unified versions

    # Position-Aware Trading
    'PositionAwareTradingAnalyzer', 'PositionAwareConfig', 'PositionAwareResult',
    'create_position_aware_analyzer', 'quick_position_aware_analysis',

    # Search Strategies
    'AdvancedSearchStrategy', 'BayesianOptimizer', 'GridOptimizer',
    'NSGA2Optimizer', 'SPEA2Optimizer', 'EvolutionaryAlgorithm',
    # 'HardwareOptimizer', 'PerformanceMonitor',  # DEPRECATED
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

    # ML Common Integration - Integration with existing utilities
    'MLCommonIntegration', 'MLCommonIntegrationType', 'MLCommonIntegrationConfig',
    'create_ml_common_integration', 'create_tas_ml_common_integration',
    'create_nas_ml_common_integration', 'create_hybrid_ml_common_integration',

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

    # Unified Ensemble Search Space
    'UnifiedEnsembleSearchSpace', 'EnsembleArchitecture', 'EnsembleSearchResult',
    'EnsembleMethod', 'EnsembleCombinationStrategy', 'EnsembleSearchSpaceConfig',
    'create_unified_ensemble_search_space', 'quick_ensemble_search',

    # Unified Architecture Compression
    'UnifiedArchitectureCompressor', 'CompressionResult', 'CompressionMethod', 'CompressionLevel', 'CompressionConfig',
    'create_unified_architecture_compressor', 'quick_compress_architecture',

    # Unified Search Space Evolution
    'UnifiedSearchSpaceEvolutionManager', 'EvolutionEvent', 'EvolutionResult', 'EvolutionTrigger', 'EvolutionAction',
    'UnifiedEvolutionConfig', 'create_unified_evolution_manager', 'quick_evolution_setup',

    # Unified Hardware Management
    'UnifiedHardwareManager', 'HardwareType', 'OptimizationLevel', 'WorkloadType',
    'HardwareMetrics', 'OptimizationResult', 'HardwareConfig',
    'create_hardware_manager', 'create_basic_hardware_manager', 'create_aggressive_hardware_manager',

    # Unified Evaluation Framework
    'UnifiedEvaluationFramework', 'EvaluationType', 'EvaluationMetric',
    'EvaluationResult', 'EvaluationConfig',
    'create_evaluation_framework', 'create_basic_evaluator', 'create_trading_evaluator',

    # Clustering Quality Metrics
    'ClusteringQualityMetrics', 'ClusteringQualityConfig', 'ClusteringQualityResult',
    'create_clustering_quality_evaluator', 'quick_clustering_evaluation'
]