#!/usr/bin/env python3
"""Step03 Import Management.

Centralized import management to resolve circular imports and organize
optional dependencies with proper fallbacks.
"""

import sys
import importlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = logging.getLogger(__name__)

# Core dependencies (always required)
CORE_DEPENDENCIES = {
    'numpy': 'np',
    'pandas': 'pd',
    'scipy': 'scipy',
    'sklearn': 'sklearn',
    'asyncio': 'asyncio',
    'json': 'json',
    'time': 'time',
    'datetime': 'datetime',
    'pathlib': 'Path',
    'typing': 'typing',
    'logging': 'logging',
    'warnings': 'warnings',
    'gc': 'gc',
    'os': 'os',
    'sys': 'sys'
}

# Optional dependencies with fallbacks
OPTIONAL_DEPENDENCIES = {
    'optuna': {
        'imports': ['optuna', 'optuna.samplers', 'optuna.pruners'],
        'fallback': None,
        'required_for': ['bayesian_optimization']
    },
    'hmmlearn': {
        'imports': ['hmmlearn', 'hmmlearn.hmm'],
        'fallback': None,
        'required_for': ['hmm_clustering']
    },
    'lightgbm': {
        'imports': ['lightgbm'],
        'fallback': None,
        'required_for': ['ml_transition_detection']
    },
    'numba': {
        'imports': ['numba'],
        'fallback': None,
        'required_for': ['performance_optimization']
    },
    'psutil': {
        'imports': ['psutil'],
        'fallback': None,
        'required_for': ['memory_management']
    },
    'joblib': {
        'imports': ['joblib'],
        'fallback': None,
        'required_for': ['model_persistence']
    }
}

# Internal module imports
INTERNAL_MODULES = {
    'config': 'step03_config',
    'technical_indicators': 'step03_technical_indicators',
    'memory_manager': 'step03_memory_manager',
    'bayesian_optimization': 'step03_bayesian_parameter_optimization',
    'ensemble_clustering': 'step03_ensemble_clustering',
    'economic_validator': 'step03_economic_significance_validator',
    'ml_transition_detector': 'step03_enhanced_ml_transition_detector'
}


class ImportManager:
    """Manages imports and resolves circular dependencies."""
    @log_important_calls
    
    def __init__(self):
        self._imported_modules = {}
        self._failed_imports = set()
        self._module_path = Path(__file__).parent
    
    def import_core_dependencies(self) -> Dict[str, Any]:
        """Import all core dependencies."""
        core_modules = {}
        
        for module_name, alias in CORE_DEPENDENCIES.items():
            try:
                module = importlib.import_module(module_name)
                core_modules[alias] = module
                self._imported_modules[module_name] = module
                logger.debug(f"Imported core dependency: {module_name}")
            except ImportError as e:
                logger.error(f"Failed to import core dependency {module_name}: {e}")
                raise ImportError(f"Core dependency {module_name} is required but not available")
        
        return core_modules
    
    def import_optional_dependency(self, dependency_name: str) -> Optional[Any]:
        """Import an optional dependency with fallback."""
        if dependency_name in self._failed_imports:
            return None
        
        if dependency_name in self._imported_modules:
            return self._imported_modules[dependency_name]
        
        if dependency_name not in OPTIONAL_DEPENDENCIES:
            logger.warning(f"Unknown optional dependency: {dependency_name}")
            return None
        
        dependency_info = OPTIONAL_DEPENDENCIES[dependency_name]
        
        try:
            # Import the main module
            main_module = importlib.import_module(dependency_info['imports'][0])
            self._imported_modules[dependency_name] = main_module
            
            # Import submodules if specified
            for submodule in dependency_info['imports'][1:]:
                try:
                    importlib.import_module(submodule)
                except ImportError:
                    logger.warning(f"Could not import submodule {submodule} for {dependency_name}")
            
            logger.debug(f"Successfully imported optional dependency: {dependency_name}")
            return main_module
            
        except ImportError as e:
            logger.warning(f"Optional dependency {dependency_name} not available: {e}")
            self._failed_imports.add(dependency_name)
            
            if dependency_info['fallback']:
                logger.info(f"Using fallback for {dependency_name}")
                return dependency_info['fallback']
            
            return None
    
    def import_internal_module(self, module_name: str) -> Optional[Any]:
        """Import internal module with circular import protection."""
        if module_name in self._imported_modules:
            return self._imported_modules[module_name]
        
        if module_name not in INTERNAL_MODULES:
            logger.error(f"Unknown internal module: {module_name}")
            return None
        
        try:
            # Add current directory to path if not already there
            if str(self._module_path) not in sys.path:
                sys.path.insert(0, str(self._module_path))
            
            # Import the module
            module = importlib.import_module(INTERNAL_MODULES[module_name])
            self._imported_modules[module_name] = module
            logger.debug(f"Successfully imported internal module: {module_name}")
            return module
            
        except ImportError as e:
            logger.error(f"Failed to import internal module {module_name}: {e}")
            return None
    
    def check_dependencies_for_feature(self, feature_name: str) -> Dict[str, bool]:
        """Check if all required dependencies for a feature are available."""
        availability = {}
        
        for dep_name, dep_info in OPTIONAL_DEPENDENCIES.items():
            if feature_name in dep_info['required_for']:
                availability[dep_name] = dep_name not in self._failed_imports
        
        return availability
    
    def get_available_features(self) -> List[str]:
        """Get list of features that can be used with current dependencies."""
        available_features = []
        
        # Check each feature
        all_features = set()
        for dep_info in OPTIONAL_DEPENDENCIES.values():
            all_features.update(dep_info['required_for'])
        
        for feature in all_features:
            deps_available = self.check_dependencies_for_feature(feature)
            if all(deps_available.values()):
                available_features.append(feature)
        
        return available_features
    
    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing dependencies."""
        return list(self._failed_imports)
    
    def import_all_available(self) -> Dict[str, Any]:
        """Import all available dependencies."""
        all_modules = {}
        
        # Import core dependencies
        all_modules.update(self.import_core_dependencies())
        
        # Import optional dependencies
        for dep_name in OPTIONAL_DEPENDENCIES.keys():
            module = self.import_optional_dependency(dep_name)
            if module is not None:
                all_modules[dep_name] = module
        
        return all_modules


# Global import manager instance
_global_import_manager = ImportManager()


def get_import_manager() -> ImportManager:
    """Get global import manager instance."""
    return _global_import_manager


def safe_import(module_name: str, fallback: Any = None) -> Any:
    """Safely import a module with fallback."""
    manager = get_import_manager()
    
    if module_name in CORE_DEPENDENCIES:
        return manager.import_core_dependencies().get(CORE_DEPENDENCIES[module_name])
    elif module_name in OPTIONAL_DEPENDENCIES:
        return manager.import_optional_dependency(module_name) or fallback
    elif module_name in INTERNAL_MODULES:
        return manager.import_internal_module(module_name) or fallback
    else:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            return fallback


def check_feature_availability(feature_name: str) -> bool:
    """Check if a feature is available with current dependencies."""
    manager = get_import_manager()
    deps_available = manager.check_dependencies_for_feature(feature_name)
    return all(deps_available.values())


def get_available_features() -> List[str]:
    """Get list of available features."""
    manager = get_import_manager()
    return manager.get_available_features()


# Convenience imports for common use cases
def get_ml_imports() -> Dict[str, Any]:
    """Get imports needed for ML functionality."""
    return {
        'sklearn': safe_import('sklearn'),
        'lightgbm': safe_import('lightgbm'),
        'numpy': safe_import('numpy'),
        'pandas': safe_import('pandas')
    }


def get_optimization_imports() -> Dict[str, Any]:
    """Get imports needed for optimization functionality."""
    return {
        'optuna': safe_import('optuna'),
        'numpy': safe_import('numpy'),
        'scipy': safe_import('scipy')
    }


def get_hmm_imports() -> Dict[str, Any]:
    """Get imports needed for HMM functionality."""
    return {
        'hmmlearn': safe_import('hmmlearn'),
        'numpy': safe_import('numpy'),
        'sklearn': safe_import('sklearn')
    }


def get_memory_imports() -> Dict[str, Any]:
    """Get imports needed for memory management."""
    return {
        'psutil': safe_import('psutil'),
        'gc': safe_import('gc'),
        'numpy': safe_import('numpy'),
        'pandas': safe_import('pandas')
    }


# Initialize imports on module load
_global_import_manager.import_all_available()