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
    OptimizationStrategy,
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

from .ensemble_optimizer import (
    EnsembleOptimizer,
    OptimizationConfig
)

from .evolutionary_algorithms import (
    EvolutionaryTreeSearch,
    TreeGeneticAlgorithm,
    TreeNSGA2,
    EvolutionaryConfig
)

# Export all main classes and functions
__all__ = [
    # Core unified regime detection
    'UnifiedRegimeConfig',
    'RegimeDetectionMethod',
    'OptimizationStrategy',
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
    'EvolutionaryConfig'
]

__version__ = "1.0.0"
__author__ = "NAS-TAS Training Utilities"
