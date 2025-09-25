"""
NAS-TAS Training Utilities

This module contains all training-related utilities for Neural Architecture Search (NAS)
and Tree Architecture Search (TAS) systems. These utilities are specifically designed
for training and optimization processes, not for live trading operations.

Main Components:
- Unified Search Engine: Core search algorithms for architecture optimization
- Multi-Objective Optimizer: Advanced optimization strategies
- Regime Detector: Market regime detection for training data analysis
- Architecture Config: Configuration management for training architectures
- Utilities: General training utilities and helper functions
- Constraint Systems: Architecture constraint validation
- Backward Compatibility: Legacy component adapters
- ML Common Integration: Integration with existing ML utilities
- Economic Evaluator: Economic significance evaluation during training
- Ensemble Management: Dynamic ensemble optimization for training
- Unified Regime Detection: Enhanced regime detection with economic significance
- Training Orchestration: Comprehensive training pipeline management
"""

# Core unified regime detection components
from .unified_regime_config import (
    UnifiedRegimeConfig,
    RegimeDetectionMethod,
    OptimizationStrategy as RegimeOptimizationStrategy,
    EconomicEvaluationMode
)

from .unified_regime_detector import (
    UnifiedRegimeDetector,
    UnifiedRegimeResult
)

from .unified_result import (
    UnifiedRegimeResult as UnifiedResult
)

# Core search and optimization components
from .unified_search_engine import (
    UnifiedSearchEngine,
    SearchConfig,
    SearchResult,
    SearchStrategy,
    ArchitectureType,
    OptimizationObjective,
    BayesianSearchStrategy,
    EvolutionarySearchStrategy,
    RandomSearchStrategy,
    create_unified_search_engine,
    quick_search
)

from .unified_multi_objective_optimizer import (
    UnifiedMultiObjectiveOptimizer,
    UnifiedMultiObjectiveConfig,
    UnifiedOptimizationResult,
    ParetoSolution,
    ObjectiveType,
    OptimizationAlgorithm,
    create_unified_multi_objective_optimizer,
    quick_multi_objective_optimization
)

from .unified_architecture_config import (
    BaseArchitectureConfig,
    TASArchitectureConfig,
    NASArchitectureConfig,
    HybridArchitectureConfig,
    ArchitectureType as ConfigArchitectureType,
    SearchStrategy as ConfigSearchStrategy,
    OptimizationObjective as ConfigOptimizationObjective,
    MarketRegime,
    create_tas_config,
    create_nas_config,
    create_hybrid_config,
    create_quick_config,
    create_comprehensive_config
)

from .unified_utilities import (
    UnifiedUtilities,
    UnifiedUtilityConfig,
    ArchitectureType as UtilArchitectureType,
    DataType,
    create_unified_utilities,
    quick_data_validation,
    quick_data_optimization
)

from .constraint_systems import (
    BaseConstraintValidator,
    NeuralConstraintValidator,
    TreeConstraintValidator,
    UnifiedConstraintValidator,
    ArchitectureConstraints,
    ConstraintViolation,
    ConstraintValidationResult,
    ConstraintType,
    ConstraintSeverity,
    create_neural_constraint_validator,
    create_tree_constraint_validator,
    create_unified_constraint_validator
)

from .backward_compatibility import (
    LegacyNASEngineAdapter,
    LegacyTASEngineAdapter,
    LegacyMultiObjectiveOptimizerAdapter,
    LegacyEconomicEvaluatorAdapter,
    LegacyRegimeDetectorAdapter,
    LegacyUtilitiesAdapter,
    migrate_config_to_unified,
    create_legacy_component,
    get_migration_guide,
    deprecated_warning
)

from .ml_common_integration import (
    MLCommonIntegration,
    MLCommonIntegrationConfig,
    MLCommonIntegrationType,
    MLUtilityType,
    MLUtilityConfig,
    MLCommonIntegrationManager,
    create_ml_common_integration,
    create_tas_ml_common_integration,
    create_nas_ml_common_integration,
    create_hybrid_ml_common_integration,
    create_shared_ml_utilities_manager
)

from .economic_evaluator import (
    EconomicRegimeEvaluator,
    create_economic_evaluator
)

from .dynamic_ensemble_manager import (
    DynamicEnsembleManager,
    EnsembleConfig,
    EnsembleModel,
    EnsembleResult
)

# Import search space utilities
from .search_space import (
    create_default_nas_search_space,
    create_tree_search_space,
    SearchSpace,
    SearchSpaceConfig,
    ParameterRange,
    SearchSpaceType,
    OptimizationStrategy as SearchSpaceOptimizationStrategy
)

# Maintain backwards compatibility by preferring the regime configuration enum
# while still exposing the search-space specific strategy under an explicit name.
OptimizationStrategy = RegimeOptimizationStrategy

# Import risk analysis
from .risk_analysis.risk_analysis import (
    RiskAnalyzer,
    RiskConfig,
    RiskResult,
    RiskMetric
)

# Import backtesting engine
from .backtesting_engine import (
    BacktestingEngine,
    BacktestingConfig,
    BacktestingResult,
    BacktestingMode
)

# Import evolutionary search
from .evolutionary_search import (
    EvolutionaryTreeSearch,
    TreeGeneticAlgorithm,
    TreeNSGA2,
    EvolutionaryConfig
)

# Import uncertainty estimation
from .uncertainty_estimation import (
    TreeUncertaintyEstimator,
    TreeEnsembleUncertainty,
    TreeBayesianUncertainty,
    UncertaintyConfig
)

# Import confidence scoring
from .confidence_scoring import (
    TreeConfidenceScorer,
    TreeReliabilityEstimator,
    TreeCalibrationScorer,
    ConfidenceConfig
)

from .shared_services import (
    DataValidationResult,
    FeatureEngineeringResult,
    ModelManagerService,
    ModelSelectorService,
    PerformanceTrackerService,
    RegimeTrainerService,
    SharedOrchestrationServices,
    engineer_core_features,
    run_shared_risk_analysis,
    validate_market_data,
)

# Import unified evaluator
from .unified_evaluator import (
    UnifiedEvaluator,
    EvaluationConfig,
    EvaluationResult,
    ModelType,
    EvaluationMode,
    MetricType
)

# Shared helper utilities exposed for consumers that previously relied on
# duplicated fallback implementations.
from .shared_logging import (
    TPRINT_AVAILABLE,
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_progress,
    tprint_performance,
    tprint_timer,
    TPrintConfig,
    LogLevel,
)

from .shared_serialization import (
    SERIALIZATION_AVAILABLE,
    JSONSerializer,
    PickleSerializer,
    ParquetSerializer,
    UniversalSerializer,
)

# Import unified hardware manager
from .unified_hardware import (
    UnifiedHardwareManager,
    HardwareAccelerationConfig,
    WorkloadType,
    OptimizationLevel,
    PerformanceMetrics,
    create_unified_hardware_manager,
    get_hardware_manager
)

# Import Hybrid NAS System
from .hybrid_nas_system import (
    HybridNASSystem,
    HybridNASConfig,
    HybridArchitectureCandidate,
    optimize_hybrid_architecture,
    analyze_data_characteristics
)

# Import ensemble optimizer
from .ensemble_optimizer import (
    EnsembleOptimizer,
    OptimizationConfig
)


# Export all main classes and functions
__all__ = [
    # Core unified regime detection
    'UnifiedRegimeConfig',
    'RegimeDetectionMethod',
    'OptimizationStrategy',
    'RegimeOptimizationStrategy',
    'SearchSpaceOptimizationStrategy',
    'create_default_nas_search_space',
    'create_tree_search_space',
    'SearchSpace',
    'SearchSpaceConfig',
    'ParameterRange',
    'SearchSpaceType',
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
    'UnifiedEvaluator',
    'EvaluationConfig',
    'EvaluationResult',
    'ModelType',
    'EvaluationMode',
    'MetricType',
    'UnifiedHardwareManager',
    'HardwareAccelerationConfig',
    'WorkloadType',
    'OptimizationLevel',
    'PerformanceMetrics',
    'create_unified_hardware_manager',
    'get_hardware_manager',
    'EconomicEvaluationMode',
    'UnifiedRegimeDetector',
    'UnifiedRegimeResult',
    'UnifiedResult',
  
    # Core search and optimization
    'UnifiedSearchEngine',
    'SearchConfig',
    'SearchResult',
    'SearchStrategy',
    'ArchitectureType',
    'OptimizationObjective',
    'BayesianSearchStrategy',
    'EvolutionarySearchStrategy',
    'RandomSearchStrategy',
    'create_unified_search_engine',
    'quick_search',
    
    # Multi-objective optimization
    'UnifiedMultiObjectiveOptimizer',
    'UnifiedMultiObjectiveConfig',
    'UnifiedOptimizationResult',
    'ParetoSolution',
    'ObjectiveType',
    'OptimizationAlgorithm',
    'create_unified_multi_objective_optimizer',
    'quick_multi_objective_optimization',
    
    # Architecture configuration
    'BaseArchitectureConfig',
    'TASArchitectureConfig',
    'NASArchitectureConfig',
    'HybridArchitectureConfig',
    'ConfigArchitectureType',
    'ConfigSearchStrategy',
    'ConfigOptimizationObjective',
    'MarketRegime',
    'create_tas_config',
    'create_nas_config',
    'create_hybrid_config',
    'create_quick_config',
    'create_comprehensive_config',
    
    # Utilities
    'UnifiedUtilities',
    'UnifiedUtilityConfig',
    'UtilArchitectureType',
    'DataType',
    'create_unified_utilities',
    'quick_data_validation',
    'quick_data_optimization',
    
    # Constraint systems
    'BaseConstraintValidator',
    'NeuralConstraintValidator',
    'TreeConstraintValidator',
    'UnifiedConstraintValidator',
    'ArchitectureConstraints',
    'ConstraintViolation',
    'ConstraintValidationResult',
    'ConstraintType',
    'ConstraintSeverity',
    'create_neural_constraint_validator',
    'create_tree_constraint_validator',
    'create_unified_constraint_validator',
    
    # Backward compatibility
    'LegacyNASEngineAdapter',
    'LegacyTASEngineAdapter',
    'LegacyMultiObjectiveOptimizerAdapter',
    'LegacyEconomicEvaluatorAdapter',
    'LegacyRegimeDetectorAdapter',
    'LegacyUtilitiesAdapter',
    'migrate_config_to_unified',
    'create_legacy_component',
    'get_migration_guide',
    'deprecated_warning',
    
    # ML Common integration
    'MLCommonIntegration',
    'MLCommonIntegrationConfig',
    'MLCommonIntegrationType',
    'MLUtilityType',
    'MLUtilityConfig',
    'MLCommonIntegrationManager',
    'create_ml_common_integration',
    'create_tas_ml_common_integration',
    'create_nas_ml_common_integration',
    'create_hybrid_ml_common_integration',
    'create_shared_ml_utilities_manager',
    
    # Economic evaluation
    'EconomicRegimeEvaluator',
    'create_economic_evaluator',
    
    # Ensemble management
    'DynamicEnsembleManager',
    'EnsembleConfig',
    'EnsembleModel',
    'EnsembleResult',
    'EnsembleOptimizer',
    'OptimizationConfig',
    
    # Evolutionary algorithms
    'EvolutionaryTreeSearch',
    'TreeGeneticAlgorithm',
    'TreeNSGA2',
    'EvolutionaryConfig',
    
    # Hybrid NAS System
    'HybridNASSystem',
    'HybridNASConfig',
    'HybridArchitectureCandidate',
    'optimize_hybrid_architecture',
    'analyze_data_characteristics',

    # Shared orchestration services
    'RegimeTrainerService',
    'ModelSelectorService',
    'ModelManagerService',
    'PerformanceTrackerService',
    'SharedOrchestrationServices',
    'DataValidationResult',
    'FeatureEngineeringResult',
    'validate_market_data',
    'engineer_core_features',
    'run_shared_risk_analysis',
]

__version__ = "1.0.0"
__author__ = "NAS-TAS Training Utilities"
