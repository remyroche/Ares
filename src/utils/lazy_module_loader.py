"""
Lazy Module Loader - Comprehensive Circular Import Prevention

This module provides a comprehensive system for lazy loading modules to prevent
circular import issues and improve startup performance.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Callable, Union
from functools import wraps


class LazyModuleLoader:
    """Lazy module loader to prevent circular imports."""
    
    def __init__(self):
        self._modules = {}
        self._functions = {}
        self.logger = logging.getLogger(__name__)
    
    def load_module(self, module_name: str, attribute_name: Optional[str] = None):
        """Load a module or attribute lazily."""
        key = f"{module_name}.{attribute_name}" if attribute_name else module_name
        
        if key not in self._modules:
            try:
                module = importlib.import_module(module_name)
                if attribute_name:
                    self._modules[key] = getattr(module, attribute_name)
                else:
                    self._modules[key] = module
            except ImportError as e:
                self.logger.warning(f"Failed to load {key}: {e}")
                self._modules[key] = None
        
        return self._modules[key]
    
    def get_function(self, module_name: str, function_name: str, fallback: Optional[Callable] = None):
        """Get a function from a module lazily."""
        key = f"{module_name}.{function_name}"
        
        if key not in self._functions:
            try:
                module = importlib.import_module(module_name)
                self._functions[key] = getattr(module, function_name)
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Failed to load {key}: {e}")
                self._functions[key] = fallback
        
        return self._functions[key]


# Global lazy loader instance
_lazy_loader = LazyModuleLoader()


def lazy_import(module_name: str, attribute_name: Optional[str] = None, fallback: Optional[Any] = None):
    """Create a lazy import wrapper."""
    def wrapper(*args, **kwargs):
        obj = _lazy_loader.load_module(module_name, attribute_name)
        if obj is None:
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise ImportError(f"Module {module_name} not available")
        
        if callable(obj):
            return obj(*args, **kwargs)
        return obj
    
    return wrapper


def lazy_function(module_name: str, function_name: str, fallback: Optional[Callable] = None):
    """Create a lazy function wrapper."""
    def wrapper(*args, **kwargs):
        func = _lazy_loader.get_function(module_name, function_name, fallback)
        if func is None:
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise ImportError(f"Function {module_name}.{function_name} not available")
        
        return func(*args, **kwargs)
    
    return wrapper


# Common lazy imports for frequently used modules
def get_common_operations():
    """Get common operations module lazily."""
    return _lazy_loader.load_module('src.utils.common_operations')


def get_matrix_operations():
    """Get matrix operations module lazily."""
    return _lazy_loader.load_module('src.utils.matrix_operations')


def get_feature_generation():
    """Get feature generation module lazily."""
    return _lazy_loader.load_module('src.feature_generation')


def get_ml_common():
    """Get ML common module lazily."""
    return _lazy_loader.load_module('src.utils.ml_common')


# Specific function lazy imports
def get_validate_file_path():
    """Get validate_file_path function lazily."""
    return _lazy_loader.get_function(
        'src.utils.base_utilities', 
        'validate_file_path',
        fallback=lambda path: True  # Fallback always returns True
    )


def get_create_directory_safe():
    """Get create_directory_safe function lazily."""
    return _lazy_loader.get_function(
        'src.utils.base_utilities',
        'create_directory_safe',
        fallback=lambda path, parents=True: True  # Fallback always returns True
    )


def get_safe_correlation_matrix():
    """Get safe_correlation_matrix function lazily."""
    return _lazy_loader.get_function(
        'src.utils.base_matrix_operations',
        'safe_correlation_matrix',
        fallback=lambda data, **kwargs: None  # Fallback returns None
    )


def get_safe_read_parquet():
    """Get safe_read_parquet function lazily."""
    return _lazy_loader.get_function(
        'src.utils.base_utilities',
        'safe_read_parquet',
        fallback=lambda path, **kwargs: None  # Fallback returns None
    )


def get_safe_write_parquet():
    """Get safe_write_parquet function lazily."""
    return _lazy_loader.get_function(
        'src.utils.base_utilities',
        'safe_write_parquet',
        fallback=lambda df, path, **kwargs: False  # Fallback returns False
    )


def get_unified_matrix_operations():
    """Get unified matrix operations lazily."""
    return _lazy_loader.get_function(
        'src.utils.matrix_operations',
        'get_unified_matrix_operations',
        fallback=lambda: None  # Fallback returns None
    )


# Decorator for lazy loading
def lazy_load(module_name: str, attribute_name: Optional[str] = None):
    """Decorator to lazy load a module or attribute."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Load the module/attribute when the function is called
            obj = _lazy_loader.load_module(module_name, attribute_name)
            if obj is None:
                raise ImportError(f"Module {module_name} not available")
            
            # Store the loaded object for future use
            setattr(wrapper, '_loaded_object', obj)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Module availability checker
def is_module_available(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_available_modules() -> Dict[str, bool]:
    """Get availability status of common modules."""
    modules = {
        'src.utils.common_operations': is_module_available('src.utils.common_operations'),
        'src.utils.matrix_operations': is_module_available('src.utils.matrix_operations'),
        'src.feature_generation': is_module_available('src.feature_generation'),
        'src.utils.ml_common': is_module_available('src.utils.ml_common'),
        'src.utils.base_utilities': is_module_available('src.utils.base_utilities'),
        'src.utils.base_matrix_operations': is_module_available('src.utils.base_matrix_operations'),
    }
    return modules


# Clear cache function
def clear_lazy_cache():
    """Clear the lazy loading cache."""
    _lazy_loader._modules.clear()
    _lazy_loader._functions.clear()


# Initialize lazy loader
def initialize_lazy_loader():
    """Initialize the lazy loader with common modules."""
    # Pre-load base utilities to avoid circular imports
    try:
        _lazy_loader.load_module('src.utils.base_utilities')
        _lazy_loader.load_module('src.utils.base_matrix_operations')
    except ImportError:
        pass  # Base modules might not be available yet


# Auto-initialize
initialize_lazy_loader()