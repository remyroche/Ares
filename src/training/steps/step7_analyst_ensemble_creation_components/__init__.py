# src/training/steps/step7_analyst_ensemble_creation/__init__.py

"""
Step 7: Analyst Ensemble Creation Package

This package contains optimized ensemble training implementations for analyst models.
"""

from src.utils.warning_symbols import (
    connection_error,
    critical,
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    missing,
    problem,
    timeout,
    validation_error,
    warning,
)

# Optional export shim: keep package importable even if class is renamed/absent
try:
    from .optimized_ensemble_training import (
        HighPerformanceEnsembleManager as OptimizedEnsembleTraining,
    )  # type: ignore

    __all__ = ["OptimizedEnsembleTraining", "HighPerformanceEnsembleManager"]
except Exception:
    __all__ = []
