"""
Centralized Dependency Management for Model Training Pipeline

This module provides:
- Centralized import management with proper error handling
- Dependency validation and health checks
- Clean fallback mechanisms
- Circular dependency prevention
- Import organization and optimization
"""

import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.tprint import tprint_error, tprint_warning, tprint_success, tprint_info

logger = system_logger.getChild('DependencyManager')

@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    import_path: str
    is_critical: bool = False
    fallback_available: bool = False
    error_message: str = ""
    required_for: List[str] = None

    def __post_init__(self):
        if self.required_for is None:
            self.required_for = []

class DependencyManager:
    """Centralized dependency management for training pipeline."""

    def __init__(self):
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
        self._register_core_dependencies()
        self._validate_environment()

    def _register_core_dependencies(self):
        """Register all core dependencies for the training pipeline."""

        # Critical dependencies - fail fast if not available
        self._register_dependency(DependencyInfo(
            name="numpy",
            import_path="numpy",
            is_critical=True,
            error_message="NumPy is required for all numerical operations",
            required_for=["data_processing", "ml_training", "feature_engineering"]
        ))

        self._register_dependency(DependencyInfo(
            name="pandas",
            import_path="pandas",
            is_critical=True,
            error_message="Pandas is required for all data operations",
            required_for=["data_processing", "feature_engineering", "data_loading"]
        ))

        # ML framework dependencies
        self._register_dependency(DependencyInfo(
            name="sklearn",
            import_path="sklearn",
            is_critical=True,
            error_message="Scikit-learn is required for core ML operations",
            required_for=["ml_training", "feature_selection", "model_evaluation"]
        ))

        self._register_dependency(DependencyInfo(
            name="lightgbm",
            import_path="lightgbm",
            is_critical=False,
            fallback_available=True,
            error_message="LightGBM not available, will use fallback models",
            required_for=["ml_training", "tactician_models"]
        ))

        self._register_dependency(DependencyInfo(
            name="catboost",
            import_path="catboost",
            is_critical=False,
            fallback_available=True,
            error_message="CatBoost not available, will use fallback models",
            required_for=["ml_training", "analyst_models", "tactician_models"]
        ))

        # Hardware optimization dependencies
        self._register_dependency(DependencyInfo(
            name="hardware_optimizers",
            import_path="src.utils.hardware.m1_gpu_utils",
            is_critical=False,
            fallback_available=True,
            error_message="Hardware optimizers not available, using standard processing",
            required_for=["performance_optimization", "memory_management"]
        ))

        self._register_dependency(DependencyInfo(
            name="common_operations",
            import_path="src.utils.common_operations",
            is_critical=True,
            error_message="Common operations utilities are required for safe math operations",
            required_for=["math_validation", "data_safety", "error_handling"]
        ))

        self._register_dependency(DependencyInfo(
            name="tprint",
            import_path="src.utils.tprint",
            is_critical=True,
            error_message="tprint is required for enhanced logging",
            required_for=["logging", "monitoring", "user_interface"]
        ))

        # Advanced ML dependencies
        self._register_dependency(DependencyInfo(
            name="optuna",
            import_path="optuna",
            is_critical=False,
            fallback_available=True,
            error_message="Optuna not available, using grid search for hyperparameter optimization",
            required_for=["hyperparameter_optimization", "model_tuning"]
        ))

        self._register_dependency(DependencyInfo(
            name="tensorflow",
            import_path="tensorflow",
            is_critical=False,
            fallback_available=True,
            error_message="TensorFlow not available, TCN models will not be available",
            required_for=["deep_learning", "tcn_models"]
        ))

    def _register_dependency(self, dep_info: DependencyInfo):
        """Register a dependency in the system."""
        self.dependencies[dep_info.name] = dep_info

    def _validate_environment(self):
        """Validate the runtime environment and available dependencies."""
        tprint_info("🔍 Validating runtime environment and dependencies...")

        missing_critical = []
        available_fallbacks = []

        for name, dep_info in self.dependencies.items():
            try:
                # Try to import the dependency
                module = self._safe_import(dep_info.import_path)
                self.loaded_modules[name] = module

                if dep_info.is_critical:
                    tprint_success(f"✅ {name} (critical) - OK")
                else:
                    tprint_info(f"✅ {name} - OK")

            except ImportError as e:
                if dep_info.is_critical:
                    missing_critical.append((name, dep_info.error_message))
                    tprint_error(f"❌ {name} (critical) - MISSING: {dep_info.error_message}")
                else:
                    available_fallbacks.append(name)
                    tprint_warning(f"⚠️ {name} - MISSING: {dep_info.error_message}")

        # Handle critical missing dependencies
        if missing_critical:
            critical_errors = [f"- {name}: {msg}" for name, msg in missing_critical]
            error_message = "Critical dependencies missing:\n" + "\n".join(critical_errors)
            error_message += "\n\nPlease install missing dependencies before running training pipeline."
            raise ImportError(error_message)

        # Report fallback availability
        if available_fallbacks:
            tprint_info(f"ℹ️ Fallback mechanisms available for: {', '.join(available_fallbacks)}")

        tprint_success("✅ Environment validation completed")

    def _safe_import(self, import_path: str) -> Any:
        """Safely import a module."""
        try:
            # Handle different import patterns
            if import_path.startswith('src.'):
                # Local imports
                module = __import__(import_path, fromlist=[''])
                return module
            else:
                # Standard library or pip imports
                module = __import__(import_path)
                return module
        except ImportError as e:
            raise ImportError(f"Failed to import {import_path}: {e}")

    def get_module(self, name: str) -> Optional[Any]:
        """Get a loaded module by name."""
        return self.loaded_modules.get(name)

    def is_available(self, name: str) -> bool:
        """Check if a dependency is available."""
        return name in self.loaded_modules

    def require_critical(self, name: str) -> Any:
        """Require a critical dependency to be available."""
        if not self.is_available(name):
            dep_info = self.dependencies.get(name)
            if dep_info and dep_info.is_critical:
                raise ImportError(f"Critical dependency '{name}' is not available: {dep_info.error_message}")
            else:
                raise ImportError(f"Dependency '{name}' is not available")

        return self.get_module(name)

    def get_training_dependencies_status(self) -> Dict[str, Any]:
        """Get status of all training-related dependencies."""
        return {
            'critical_missing': [name for name, dep in self.dependencies.items()
                               if dep.is_critical and not self.is_available(name)],
            'optional_missing': [name for name, dep in self.dependencies.items()
                               if not dep.is_critical and not self.is_available(name)],
            'available': list(self.loaded_modules.keys()),
            'fallback_available': [name for name in self.dependencies.keys()
                                 if not self.is_available(name) and self.dependencies[name].fallback_available]
        }

# Global dependency manager instance
_dependency_manager: Optional[DependencyManager] = None

def get_dependency_manager() -> DependencyManager:
    """Get or create the global dependency manager."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager

def validate_training_environment() -> bool:
    """Validate that all required dependencies for training are available."""
    try:
        manager = get_dependency_manager()
        status = manager.get_training_dependencies_status()

        if status['critical_missing']:
            tprint_error("❌ Training cannot proceed - critical dependencies missing:")
            for dep in status['critical_missing']:
                tprint_error(f"   - {dep}")
            return False

        tprint_success("✅ Training environment validation passed")
        return True

    except Exception as e:
        tprint_error(f"❌ Environment validation failed: {e}")
        return False

# Convenience functions for common imports
def get_numpy():
    """Get numpy with proper validation."""
    return get_dependency_manager().require_critical("numpy")

def get_pandas():
    """Get pandas with proper validation."""
    return get_dependency_manager().require_critical("pandas")

def get_sklearn():
    """Get sklearn with proper validation."""
    return get_dependency_manager().require_critical("sklearn")

def get_tprint():
    """Get tprint with proper validation."""
    return get_dependency_manager().require_critical("tprint")

def get_common_operations():
    """Get common operations with proper validation."""
    return get_dependency_manager().require_critical("common_operations")

def get_lightgbm():
    """Get lightgbm if available, None otherwise."""
    manager = get_dependency_manager()
    if manager.is_available("lightgbm"):
        return manager.get_module("lightgbm")
    return None

def get_catboost():
    """Get catboost if available, None otherwise."""
    manager = get_dependency_manager()
    if manager.is_available("catboost"):
        return manager.get_module("catboost")
    return None