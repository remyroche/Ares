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

from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType
from typing import Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

_FAILED_IMPORTS: Dict[str, Exception] = {}
_SYMBOL_REGISTRY: Dict[str, Tuple[str, str]] = {}
__all__: list[str] = []


def _register_symbol(name: str, value: Optional[object]) -> None:
    """Register an exported symbol and track it in ``__all__`` when available."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)


def _export_symbols(module_path: str, symbol_map: Mapping[str, str]) -> None:
    """Safely import ``module_path`` and expose ``symbol_map`` entries.

    Args:
        module_path: Relative module path (e.g. ``".foo"``).
        symbol_map: Mapping from attribute name inside the module to the public name
            that should be exported from this package.
    """
    for attribute_name, public_name in symbol_map.items():
        _SYMBOL_REGISTRY[public_name] = (module_path, attribute_name)

    try:
        module: ModuleType = import_module(module_path, package=__name__)
    except Exception as exc:  # pragma: no cover - defensive
        _FAILED_IMPORTS[module_path] = exc
        logger.error("Failed to import module '%s': %s", module_path, exc, exc_info=exc)
        for alias in symbol_map.values():
            _register_symbol(alias, None)
        return

    for attribute_name, public_name in symbol_map.items():
        try:
            value = getattr(module, attribute_name)
        except AttributeError as exc:  # pragma: no cover - defensive
            key = f"{module_path}.{attribute_name}"
            _FAILED_IMPORTS[key] = exc
            logger.error(
                "Failed to import '%s' from '%s': %s", attribute_name, module_path, exc, exc_info=exc
            )
            value = None
        _register_symbol(public_name, value)


_IMPORT_STRUCTURE: Mapping[str, Mapping[str, str]] = {
    ".unified_regime_config": {
        "UnifiedRegimeConfig": "UnifiedRegimeConfig",
        "RegimeDetectionMethod": "RegimeDetectionMethod",
        "OptimizationStrategy": "RegimeOptimizationStrategy",
        "EconomicEvaluationMode": "EconomicEvaluationMode",
    },
    ".unified_regime_detector": {
        "UnifiedRegimeDetector": "UnifiedRegimeDetector",
        "UnifiedRegimeResult": "UnifiedRegimeResult",
    },
    ".unified_result": {
        "UnifiedRegimeResult": "UnifiedResult",
    },
    ".unified_search_engine": {
        "UnifiedSearchEngine": "UnifiedSearchEngine",
        "SearchConfig": "SearchConfig",
        "SearchResult": "SearchResult",
        "SearchStrategy": "SearchStrategy",
        "ArchitectureType": "ArchitectureType",
        "OptimizationObjective": "OptimizationObjective",
        "BayesianSearchStrategy": "BayesianSearchStrategy",
        "EvolutionarySearchStrategy": "EvolutionarySearchStrategy",
        "RandomSearchStrategy": "RandomSearchStrategy",
        "create_unified_search_engine": "create_unified_search_engine",
        "quick_search": "quick_search",
    },
    ".unified_multi_objective_optimizer": {
        "UnifiedMultiObjectiveOptimizer": "UnifiedMultiObjectiveOptimizer",
        "UnifiedMultiObjectiveConfig": "UnifiedMultiObjectiveConfig",
        "UnifiedOptimizationResult": "UnifiedOptimizationResult",
        "ParetoSolution": "ParetoSolution",
        "ObjectiveType": "ObjectiveType",
        "OptimizationAlgorithm": "OptimizationAlgorithm",
        "create_unified_multi_objective_optimizer": "create_unified_multi_objective_optimizer",
        "quick_multi_objective_optimization": "quick_multi_objective_optimization",
    },
    ".unified_architecture_config": {
        "BaseArchitectureConfig": "BaseArchitectureConfig",
        "TASArchitectureConfig": "TASArchitectureConfig",
        "NASArchitectureConfig": "NASArchitectureConfig",
        "HybridArchitectureConfig": "HybridArchitectureConfig",
        "ArchitectureType": "ConfigArchitectureType",
        "SearchStrategy": "ConfigSearchStrategy",
        "OptimizationObjective": "ConfigOptimizationObjective",
        "MarketRegime": "MarketRegime",
        "create_tas_config": "create_tas_config",
        "create_nas_config": "create_nas_config",
        "create_hybrid_config": "create_hybrid_config",
        "create_quick_config": "create_quick_config",
        "create_comprehensive_config": "create_comprehensive_config",
    },
    ".unified_utilities": {
        "UnifiedUtilities": "UnifiedUtilities",
        "UnifiedUtilityConfig": "UnifiedUtilityConfig",
        "ArchitectureType": "UtilArchitectureType",
        "DataType": "DataType",
        "create_unified_utilities": "create_unified_utilities",
        "quick_data_validation": "quick_data_validation",
        "quick_data_optimization": "quick_data_optimization",
    },
    ".constraint_systems": {
        "BaseConstraintValidator": "BaseConstraintValidator",
        "NeuralConstraintValidator": "NeuralConstraintValidator",
        "TreeConstraintValidator": "TreeConstraintValidator",
        "UnifiedConstraintValidator": "UnifiedConstraintValidator",
        "ArchitectureConstraints": "ArchitectureConstraints",
        "ConstraintViolation": "ConstraintViolation",
        "ConstraintValidationResult": "ConstraintValidationResult",
        "ConstraintType": "ConstraintType",
        "ConstraintSeverity": "ConstraintSeverity",
        "create_neural_constraint_validator": "create_neural_constraint_validator",
        "create_tree_constraint_validator": "create_tree_constraint_validator",
        "create_unified_constraint_validator": "create_unified_constraint_validator",
    },
    ".backward_compatibility": {
        "LegacyNASEngineAdapter": "LegacyNASEngineAdapter",
        "LegacyTASEngineAdapter": "LegacyTASEngineAdapter",
        "LegacyMultiObjectiveOptimizerAdapter": "LegacyMultiObjectiveOptimizerAdapter",
        "LegacyEconomicEvaluatorAdapter": "LegacyEconomicEvaluatorAdapter",
        "LegacyRegimeDetectorAdapter": "LegacyRegimeDetectorAdapter",
        "LegacyUtilitiesAdapter": "LegacyUtilitiesAdapter",
        "migrate_config_to_unified": "migrate_config_to_unified",
        "create_legacy_component": "create_legacy_component",
        "get_migration_guide": "get_migration_guide",
        "deprecated_warning": "deprecated_warning",
    },
    ".ml_common_integration": {
        "MLCommonIntegration": "MLCommonIntegration",
        "MLCommonIntegrationConfig": "MLCommonIntegrationConfig",
        "MLCommonIntegrationType": "MLCommonIntegrationType",
        "MLUtilityType": "MLUtilityType",
        "MLUtilityConfig": "MLUtilityConfig",
        "MLCommonIntegrationManager": "MLCommonIntegrationManager",
        "create_ml_common_integration": "create_ml_common_integration",
        "create_tas_ml_common_integration": "create_tas_ml_common_integration",
        "create_nas_ml_common_integration": "create_nas_ml_common_integration",
        "create_hybrid_ml_common_integration": "create_hybrid_ml_common_integration",
        "create_shared_ml_utilities_manager": "create_shared_ml_utilities_manager",
    },
    ".economic_evaluator": {
        "EconomicRegimeEvaluator": "EconomicRegimeEvaluator",
        "create_economic_evaluator": "create_economic_evaluator",
    },
    ".search_space": {
        "create_default_nas_search_space": "create_default_nas_search_space",
        "create_tree_search_space": "create_tree_search_space",
        "SearchSpace": "SearchSpace",
        "SearchSpaceConfig": "SearchSpaceConfig",
        "ParameterRange": "ParameterRange",
        "SearchSpaceType": "SearchSpaceType",
        "OptimizationStrategy": "SearchSpaceOptimizationStrategy",
    },
    ".risk_analysis.risk_analysis": {
        "RiskAnalyzer": "RiskAnalyzer",
        "RiskConfig": "RiskConfig",
        "RiskResult": "RiskResult",
        "RiskMetric": "RiskMetric",
    },
    ".evolutionary_search": {
        "EvolutionaryTreeSearch": "EvolutionaryTreeSearch",
        "TreeGeneticAlgorithm": "TreeGeneticAlgorithm",
        "TreeNSGA2": "TreeNSGA2",
        "EvolutionaryConfig": "EvolutionaryConfig",
    },
    ".shared_services": {
        "DataValidationResult": "DataValidationResult",
        "FeatureEngineeringResult": "FeatureEngineeringResult",
        "ModelManagerService": "ModelManagerService",
        "ModelSelectorService": "ModelSelectorService",
        "PerformanceTrackerService": "PerformanceTrackerService",
        "RegimeTrainerService": "RegimeTrainerService",
        "SharedOrchestrationServices": "SharedOrchestrationServices",
        "engineer_core_features": "engineer_core_features",
        "run_shared_risk_analysis": "run_shared_risk_analysis",
        "validate_market_data": "validate_market_data",
    },
    ".unified_evaluator": {
        "UnifiedEvaluator": "UnifiedEvaluator",
        "EvaluationConfig": "EvaluationConfig",
        "EvaluationResult": "EvaluationResult",
        "ModelType": "ModelType",
        "EvaluationMode": "EvaluationMode",
        "MetricType": "MetricType",
    },
    ".shared_logging": {
        "TPRINT_AVAILABLE": "TPRINT_AVAILABLE",
        "tprint": "tprint",
        "tprint_debug": "tprint_debug",
        "tprint_info": "tprint_info",
        "tprint_warning": "tprint_warning",
        "tprint_error": "tprint_error",
        "tprint_success": "tprint_success",
        "tprint_progress": "tprint_progress",
        "tprint_performance": "tprint_performance",
        "tprint_timer": "tprint_timer",
        "TPrintConfig": "TPrintConfig",
        "LogLevel": "LogLevel",
    },
    ".shared_serialization": {
        "SERIALIZATION_AVAILABLE": "SERIALIZATION_AVAILABLE",
        "JSONSerializer": "JSONSerializer",
        "PickleSerializer": "PickleSerializer",
        "ParquetSerializer": "ParquetSerializer",
        "UniversalSerializer": "UniversalSerializer",
    },
    ".unified_hardware": {
        "UnifiedHardwareManager": "UnifiedHardwareManager",
        "HardwareAccelerationConfig": "HardwareAccelerationConfig",
        "WorkloadType": "WorkloadType",
        "OptimizationLevel": "OptimizationLevel",
        "PerformanceMetrics": "PerformanceMetrics",
        "create_unified_hardware_manager": "create_unified_hardware_manager",
        "get_hardware_manager": "get_hardware_manager",
    },
    ".hybrid_nas_system": {
        "HybridNASSystem": "HybridNASSystem",
        "HybridNASConfig": "HybridNASConfig",
        "HybridArchitectureCandidate": "HybridArchitectureCandidate",
        "optimize_hybrid_architecture": "optimize_hybrid_architecture",
        "analyze_data_characteristics": "analyze_data_characteristics",
    },
    ".ensemble_optimizer": {
        "EnsembleOptimizer": "EnsembleOptimizer",
        "OptimizationConfig": "OptimizationConfig",
    },
    ".architectures.neural": {
        "NeuralArchitecture": "NeuralArchitecture",
        "LayerSpec": "LayerSpec",
    },
    ".search_strategies": {
        "StrategyRegistry": "SearchStrategyRegistry",
        "RandomSearchStrategy": "PluginRandomSearchStrategy",
        "GridSearchStrategy": "PluginGridSearchStrategy",
        "OptunaSearchStrategy": "OptunaSearchStrategy",
        "HyperbandSearchStrategy": "HyperbandSearchStrategy",
    },
}

for module_name, symbols in _IMPORT_STRUCTURE.items():
    _export_symbols(module_name, symbols)

# Maintain backwards compatibility for the optimization strategy alias used by
# historical clients of this package.
OptimizationStrategy = globals().get("RegimeOptimizationStrategy")
_register_symbol("OptimizationStrategy", OptimizationStrategy)


def get_failed_imports() -> Dict[str, Exception]:
    """Return a copy of the import failures captured during package initialisation."""
    return dict(_FAILED_IMPORTS)


_register_symbol("get_failed_imports", get_failed_imports)


def reload_failed_import(module_path: str) -> None:
    """Retry importing a module that previously failed.

    Args:
        module_path: The relative module path recorded in ``_IMPORT_STRUCTURE``.

    Raises:
        ImportError: If the module cannot be imported or if the module path is unknown.
    """

    symbol_map = _IMPORT_STRUCTURE.get(module_path)
    if symbol_map is None:
        raise ImportError(f"Unknown NAS-TAS module '{module_path}'.")

    try:
        module = import_module(module_path, package=__name__)
    except Exception as exc:  # pragma: no cover - defensive
        _FAILED_IMPORTS[module_path] = exc
        raise ImportError(f"Failed to import '{module_path}': {exc}") from exc

    for attribute_name, public_name in symbol_map.items():
        try:
            value = getattr(module, attribute_name)
        except AttributeError as exc:  # pragma: no cover - defensive
            key = f"{module_path}.{attribute_name}"
            _FAILED_IMPORTS[key] = exc
            _register_symbol(public_name, None)
            raise ImportError(
                f"Module '{module_path}' is missing expected attribute '{attribute_name}'"
            ) from exc
        else:
            _register_symbol(public_name, value)
            _FAILED_IMPORTS.pop(f"{module_path}.{attribute_name}", None)

    _FAILED_IMPORTS.pop(module_path, None)


_register_symbol("reload_failed_import", reload_failed_import)


def __getattr__(name: str) -> object:
    """Attempt to lazily import NAS-TAS symbols on first access."""

    module_info = _SYMBOL_REGISTRY.get(name)
    if not module_info:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attribute_name = module_info
    try:
        module = import_module(module_path, package=__name__)
        value = getattr(module, attribute_name)
    except Exception as exc:  # pragma: no cover - defensive
        key = module_path if isinstance(exc, ImportError) else f"{module_path}.{attribute_name}"
        if isinstance(exc, Exception):
            _FAILED_IMPORTS[key] = exc
        raise AttributeError(
            f"Failed to load '{name}' from '{module_path}': {exc}"
        ) from exc

    _register_symbol(name, value)
    _FAILED_IMPORTS.pop(module_path, None)
    _FAILED_IMPORTS.pop(f"{module_path}.{attribute_name}", None)
    return value


def __dir__() -> list[str]:
    """Expose dynamically registered NAS-TAS symbols for auto-complete."""

    return sorted(set(__all__) | set(_SYMBOL_REGISTRY))

__version__ = "1.0.0"
_register_symbol("__version__", __version__)
__author__ = "NAS-TAS Training Utilities"
_register_symbol("__author__", __author__)
