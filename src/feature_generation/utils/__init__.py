"""Convenience imports and dependency-aware fallbacks for feature generation utilities."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, Tuple, Type, TypeVar


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_missing_dependencies: Dict[str, ImportError] = {}
_T = TypeVar("_T")


def _register_missing(name: str, exc: ImportError) -> None:
    """Record a missing dependency and emit a warning once per symbol."""

    if name in _missing_dependencies:
        return

    _missing_dependencies[name] = exc
    logger.warning("Optional feature generation component '%s' unavailable: %s", name, exc)


def _missing_callable(name: str, exc: ImportError) -> Callable[..., None]:
    """Create a callable that raises a helpful runtime error when used."""

    def _missing(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            f"{name} is unavailable because optional feature-generation dependencies could not be imported"
        ) from exc

    return _missing


def _missing_class(name: str, exc: ImportError) -> Type[object]:
    """Create a placeholder class that raises on instantiation."""

    class _Missing:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                f"{name} is unavailable because optional feature-generation dependencies could not be imported"
            ) from exc

    _Missing.__name__ = name
    return _Missing


def _assign_missing(names: Iterable[str], exc: ImportError, *, kind: str = "callable") -> Tuple[object, ...]:
    """Utility helper that returns fallbacks for each missing symbol."""

    factories = {
        "callable": _missing_callable,
        "class": _missing_class,
    }
    factory = factories[kind]
    return tuple(factory(name, exc) for name in names)


# Import main utility classes for easy access while providing informative fallbacks.
try:
    from .step06_utility_container import (
        Step06UtilityContainer,
        UtilityConfig,
        get_utility_container,
        utility_container_context,
        inject_utilities,
    )
except ImportError as exc:  # pragma: no cover - optional import
    _register_missing("step06_utility_container", exc)
    (
        Step06UtilityContainer,
        UtilityConfig,
        get_utility_container,
        utility_container_context,
        inject_utilities,
    ) = (
        _assign_missing(("Step06UtilityContainer",), exc, kind="class")[0],
        _assign_missing(("UtilityConfig",), exc, kind="class")[0],
        _missing_callable("get_utility_container", exc),
        _missing_callable("utility_container_context", exc),
        _missing_callable("inject_utilities", exc),
    )

try:
    from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering
except ImportError as exc:  # pragma: no cover - optional import
    _register_missing("step06_enhanced_feature_engineering", exc)
    EnhancedFeatureEngineering = _assign_missing(("EnhancedFeatureEngineering",), exc, kind="class")[0]

try:
    from .optimization import (
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        get_feature_optimizer,
        optimize_feature_lookback,
        get_optimization_config,
        LookbackOptimizer,  # Backward compatibility
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - optional import
    OPTIMIZATION_AVAILABLE = False
    _register_missing("optimization", exc)
    (
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        get_feature_optimizer,
        optimize_feature_lookback,
        get_optimization_config,
        LookbackOptimizer,
    ) = (
        _assign_missing(("FeatureGenerationOptimizer",), exc, kind="class")[0],
        _assign_missing(("FeatureOptimizationConfig",), exc, kind="class")[0],
        _assign_missing(("FeatureOptimizationResult",), exc, kind="class")[0],
        _assign_missing(("OptimizationMethod",), exc, kind="class")[0],
        _missing_callable("get_feature_optimizer", exc),
        _missing_callable("optimize_feature_lookback", exc),
        _missing_callable("get_optimization_config", exc),
        _assign_missing(("LookbackOptimizer",), exc, kind="class")[0],
    )

try:
    from .cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline
    from .fractional_differentiation_pipeline import FractionalDifferentiationPipeline
    from .enhanced_matrix_operations import EnhancedMatrixOperations
except ImportError as exc:  # pragma: no cover - optional import
    _register_missing("advanced_utilities", exc)
    (
        CrossTimeframeAnalysisPipeline,
        FractionalDifferentiationPipeline,
        EnhancedMatrixOperations,
    ) = (
        _assign_missing(("CrossTimeframeAnalysisPipeline",), exc, kind="class")[0],
        _assign_missing(("FractionalDifferentiationPipeline",), exc, kind="class")[0],
        _assign_missing(("EnhancedMatrixOperations",), exc, kind="class")[0],
    )

try:
    from .math_validation import (
        validate_feature_quality,
        validate_features_dataframe,
        feature_validation_decorator,
    )
except ImportError as exc:  # pragma: no cover - optional import
    _register_missing("math_validation", exc)
    (
        validate_feature_quality,
        validate_features_dataframe,
        feature_validation_decorator,
    ) = (
        _missing_callable("validate_feature_quality", exc),
        _missing_callable("validate_features_dataframe", exc),
        _missing_callable("feature_validation_decorator", exc),
    )


__version__ = "2.0.0"
__description__ = "Feature Generation Utils - Advanced feature engineering and optimization utilities"

__all__ = [
    "Step06UtilityContainer",
    "UtilityConfig",
    "get_utility_container",
    "utility_container_context",
    "inject_utilities",
    "EnhancedFeatureEngineering",
    "FeatureGenerationOptimizer",
    "FeatureOptimizationConfig",
    "FeatureOptimizationResult",
    "OptimizationMethod",
    "get_feature_optimizer",
    "optimize_feature_lookback",
    "get_optimization_config",
    "LookbackOptimizer",
    "CrossTimeframeAnalysisPipeline",
    "FractionalDifferentiationPipeline",
    "EnhancedMatrixOperations",
    "validate_feature_quality",
    "validate_features_dataframe",
    "feature_validation_decorator",
    "OPTIMIZATION_AVAILABLE",
]


def list_missing_dependencies() -> Dict[str, ImportError]:
    """Expose the optional imports that failed so callers can introspect the module state."""

    return dict(_missing_dependencies)
