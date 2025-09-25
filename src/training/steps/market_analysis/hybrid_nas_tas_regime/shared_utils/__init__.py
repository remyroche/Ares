"""
Unified NAS-TAS System

This package provides a unified interface for both Neural Architecture Search (NAS)
and Tree Architecture Search (TAS) systems. It consolidates all search strategies,
optimization algorithms, evaluation methods, and utilities into a single, cohesive framework.

Key Features:
- Unified search engine supporting both neural and tree architectures
- Comprehensive multi-objective optimization with multiple algorithms
- Advanced economic evaluation and trading viability assessment
- Flexible regime detection with multiple methods
- Hardware optimization and parallel processing support
- Backward compatibility with existing NAS/TAS components
- Unified configuration system for easy management

Main Components:
- UnifiedSearchEngine: Unified interface for all search strategies
- UnifiedMultiObjectiveOptimizer: Multi-objective optimization with NSGA-II, Bayesian, etc.
- UnifiedEconomicEvaluator: Economic significance and trading viability evaluation
- UnifiedRegimeDetector: Regime detection with clustering, HMM, neural networks, etc.
- UnifiedUtilities: Common utilities for data processing and validation
- UnifiedConfig: Comprehensive configuration management system

Usage Example:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
        UnifiedSearchEngine, SearchConfig, ArchitectureType, SearchStrategy
    )
    
    # Create unified search engine
    config = SearchConfig(
        architecture_type=ArchitectureType.HYBRID,
        search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
        max_iterations=100
    )
    
    engine = UnifiedSearchEngine(config)
    
    # Perform search
    result = engine.search(search_space, objective_function)
    
    print(f"Best architecture: {result.best_architecture}")
    print(f"Best score: {max(result.best_scores.values())}")
"""

# Import main unified components
from .unified_search_engine import (
    UnifiedSearchEngine,
    SearchConfig,
    SearchResult,
    SearchStrategy,
    ArchitectureType,
    OptimizationObjective,
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

from .unified_economic_evaluator import (
    UnifiedEconomicEvaluator,
    EconomicEvaluationConfig,
    EconomicEvaluationResult,
    EconomicMetrics,
    EvaluationType,
    create_unified_economic_evaluator,
    quick_economic_evaluation
)

from .unified_regime_detector import (
    UnifiedRegimeDetector,
    RegimeDetectionConfig,
    RegimeDetectionResult,
    RegimeInfo,
    RegimeDetectionMethod,
    create_unified_regime_detector
)

from .unified_utilities import (
    UnifiedUtilities,
    UnifiedUtilityConfig,
    DataType,
    create_unified_utilities,
    quick_data_validation,
    quick_data_optimization
)

from .unified_config import (
    UnifiedConfig,
    SearchConfig as UnifiedSearchConfig,
    OptimizationConfig,
    EvaluationConfig,
    RegimeDetectionConfig as UnifiedRegimeDetectionConfig,
    UtilityConfig,
    ConfigManager,
    ConfigFormat,
    config_manager,
    get_config,
    set_config,
    create_default_config,
    load_config_from_file
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

# Version information
__version__ = "1.0.0"
__author__ = "Unified NAS-TAS System Team"
__email__ = "unified-system@example.com"

# Main exports
__all__ = [
    # Core unified components
    'UnifiedSearchEngine',
    'UnifiedMultiObjectiveOptimizer', 
    'UnifiedEconomicEvaluator',
    'UnifiedRegimeDetector',
    'UnifiedUtilities',
    'UnifiedConfig',
    
    # Configuration classes
    'SearchConfig',
    'UnifiedMultiObjectiveConfig',
    'EconomicEvaluationConfig',
    'RegimeDetectionConfig',
    'UnifiedUtilityConfig',
    'OptimizationConfig',
    'EvaluationConfig',
    'UtilityConfig',
    
    # Result classes
    'SearchResult',
    'UnifiedOptimizationResult',
    'EconomicEvaluationResult',
    'RegimeDetectionResult',
    'EconomicMetrics',
    'ParetoSolution',
    'RegimeInfo',
    
    # Enums
    'SearchStrategy',
    'ArchitectureType',
    'OptimizationObjective',
    'ObjectiveType',
    'OptimizationAlgorithm',
    'EvaluationType',
    'DataType',
    'RegimeDetectionMethod',
    'ConfigFormat',
    
    # Utility functions
    'create_unified_search_engine',
    'create_unified_multi_objective_optimizer',
    'create_unified_economic_evaluator',
    'create_unified_regime_detector',
    'create_unified_utilities',
    'quick_search',
    'quick_multi_objective_optimization',
    'quick_economic_evaluation',
    'quick_data_validation',
    'quick_data_optimization',
    
    # Configuration management
    'ConfigManager',
    'config_manager',
    'get_config',
    'set_config',
    'create_default_config',
    'load_config_from_file',
    
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
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]

# Convenience functions for quick access
def create_unified_system(config: Optional[UnifiedConfig] = None) -> Dict[str, Any]:
    """
    Create a complete unified system with all components.
    
    Args:
        config: Unified configuration (optional)
        
    Returns:
        Dictionary containing all unified components
    """
    if config is None:
        config = create_default_config()
    
    return {
        'search_engine': UnifiedSearchEngine(config.search_config),
        'optimizer': UnifiedMultiObjectiveOptimizer(config.optimization_config),
        'evaluator': UnifiedEconomicEvaluator(config.evaluation_config),
        'regime_detector': UnifiedRegimeDetector(config.regime_detection_config),
        'utilities': UnifiedUtilities(config.utility_config),
        'config': config
    }

def get_system_info() -> Dict[str, Any]:
    """Get information about the unified system."""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'components': [
            'UnifiedSearchEngine',
            'UnifiedMultiObjectiveOptimizer',
            'UnifiedEconomicEvaluator', 
            'UnifiedRegimeDetector',
            'UnifiedUtilities',
            'UnifiedConfig'
        ],
        'supported_architectures': ['neural', 'tree', 'hybrid'],
        'supported_search_strategies': [strategy.value for strategy in SearchStrategy],
        'supported_optimization_algorithms': [algo.value for algo in OptimizationAlgorithm],
        'supported_evaluation_types': [eval_type.value for eval_type in EvaluationType],
        'supported_regime_detection_methods': [method.value for method in RegimeDetectionMethod]
    }

# Add convenience functions to __all__
__all__.extend([
    'create_unified_system',
    'get_system_info'
])