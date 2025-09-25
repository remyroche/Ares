"""Compatibility exports for legacy ``shared_utils`` imports.

This module intentionally re-exports the canonical NAS/TAS interfaces from the
parent :mod:`src.utils.nas_tas` package so downstream code that historically
imported from ``src.utils.nas_tas.shared_utils`` continues to work without
maintaining a separate, divergent implementation surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..unified_search_engine import (
    ArchitectureType,
    OptimizationObjective,
    SearchConfig,
    SearchResult,
    SearchStrategy,
    UnifiedSearchEngine,
    create_unified_search_engine,
    quick_search,
)
from ..unified_multi_objective_optimizer import (
    ObjectiveType,
    OptimizationAlgorithm,
    ParetoSolution,
    UnifiedMultiObjectiveConfig,
    UnifiedMultiObjectiveOptimizer,
    UnifiedOptimizationResult,
    create_unified_multi_objective_optimizer,
    quick_multi_objective_optimization,
)
from ..unified_economic_evaluator import (
    EconomicEvaluationConfig,
    EconomicEvaluationResult,
    EconomicMetrics,
    EvaluationType,
    UnifiedEconomicEvaluator,
    create_economic_evaluator as create_unified_economic_evaluator,
    quick_economic_evaluation,
)
from ..unified_regime_detector import (
    RegimeDetectionConfig,
    RegimeDetectionMethod,
    RegimeDetectionResult,
    RegimeInfo,
    UnifiedRegimeDetector,
    create_unified_regime_detector,
)
from ..unified_utilities import (
    DataType,
    UnifiedUtilities,
    UnifiedUtilityConfig,
    create_unified_utilities,
    quick_data_optimization,
    quick_data_validation,
)
from .unified_config import (
    ConfigFormat,
    ConfigManager,
    EvaluationConfig,
    OptimizationConfig,
    UnifiedConfig,
    UtilityConfig,
    config_manager,
    create_default_config,
    get_config,
    load_config_from_file,
    set_config,
)
from ..backward_compatibility import (
    LegacyEconomicEvaluatorAdapter,
    LegacyMultiObjectiveOptimizerAdapter,
    LegacyNASEngineAdapter,
    LegacyRegimeDetectorAdapter,
    LegacyTASEngineAdapter,
    LegacyUtilitiesAdapter,
    create_legacy_component,
    deprecated_warning,
    get_migration_guide,
    migrate_config_to_unified,
)

__version__ = "1.0.0"
__author__ = "Unified NAS-TAS System Team"
__email__ = "unified-system@example.com"

__all__ = [
    # Core unified components
    "UnifiedSearchEngine",
    "UnifiedMultiObjectiveOptimizer",
    "UnifiedEconomicEvaluator",
    "UnifiedRegimeDetector",
    "UnifiedUtilities",
    "UnifiedConfig",

    # Configuration classes
    "SearchConfig",
    "UnifiedMultiObjectiveConfig",
    "EconomicEvaluationConfig",
    "RegimeDetectionConfig",
    "UnifiedUtilityConfig",
    "OptimizationConfig",
    "EvaluationConfig",
    "UtilityConfig",

    # Result classes
    "SearchResult",
    "UnifiedOptimizationResult",
    "EconomicEvaluationResult",
    "RegimeDetectionResult",
    "EconomicMetrics",
    "ParetoSolution",
    "RegimeInfo",

    # Enums
    "SearchStrategy",
    "ArchitectureType",
    "OptimizationObjective",
    "ObjectiveType",
    "OptimizationAlgorithm",
    "EvaluationType",
    "DataType",
    "RegimeDetectionMethod",
    "ConfigFormat",

    # Utility functions
    "create_unified_search_engine",
    "create_unified_multi_objective_optimizer",
    "create_unified_economic_evaluator",
    "create_unified_regime_detector",
    "create_unified_utilities",
    "quick_search",
    "quick_multi_objective_optimization",
    "quick_economic_evaluation",
    "quick_data_validation",
    "quick_data_optimization",

    # Configuration management
    "ConfigManager",
    "config_manager",
    "get_config",
    "set_config",
    "create_default_config",
    "load_config_from_file",

    # Backward compatibility helpers
    "LegacyNASEngineAdapter",
    "LegacyTASEngineAdapter",
    "LegacyMultiObjectiveOptimizerAdapter",
    "LegacyEconomicEvaluatorAdapter",
    "LegacyRegimeDetectorAdapter",
    "LegacyUtilitiesAdapter",
    "migrate_config_to_unified",
    "create_legacy_component",
    "get_migration_guide",
    "deprecated_warning",

    # Metadata
    "__version__",
    "__author__",
    "__email__",

    # Convenience helpers
    "create_unified_system",
    "get_system_info",
]


def create_unified_system(config: Optional[UnifiedConfig] = None) -> Dict[str, Any]:
    """Construct a dictionary with the core unified NAS/TAS components."""

    if config is None:
        config = create_default_config()

    return {
        "search_engine": UnifiedSearchEngine(config.search_config),
        "optimizer": UnifiedMultiObjectiveOptimizer(config.optimization_config),
        "evaluator": UnifiedEconomicEvaluator(config.evaluation_config),
        "regime_detector": UnifiedRegimeDetector(config.regime_detection_config),
        "utilities": UnifiedUtilities(config.utility_config),
        "config": config,
    }


def get_system_info() -> Dict[str, Any]:
    """Return metadata about the shared NAS/TAS system interface."""

    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "components": [
            "UnifiedSearchEngine",
            "UnifiedMultiObjectiveOptimizer",
            "UnifiedEconomicEvaluator",
            "UnifiedRegimeDetector",
            "UnifiedUtilities",
            "UnifiedConfig",
        ],
        "supported_architectures": [arch.value for arch in ArchitectureType],
        "supported_search_strategies": [strategy.value for strategy in SearchStrategy],
        "supported_optimization_algorithms": [algo.value for algo in OptimizationAlgorithm],
        "supported_evaluation_types": [eval_type.value for eval_type in EvaluationType],
        "supported_regime_detection_methods": [method.value for method in RegimeDetectionMethod],
    }
