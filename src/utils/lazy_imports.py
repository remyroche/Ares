"""
Lazy Import Utilities

This module provides lazy import functionality to avoid circular import issues
and improve module loading performance.
"""

import importlib
from typing import Any, Dict, Optional
from functools import wraps


class LazyImport:
    """Lazy import wrapper to defer module loading until actually needed."""
    
    def __init__(self, module_name: str, attribute_name: Optional[str] = None):
        self.module_name = module_name
        self.attribute_name = attribute_name
        self._module = None
        self._attribute = None
    
    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {self.module_name}: {e}")
        
        if self.attribute_name:
            if self._attribute is None:
                self._attribute = getattr(self._module, self.attribute_name)
            return getattr(self._attribute, name)
        
        return getattr(self._module, name)
    
    def __call__(self, *args, **kwargs):
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {self.module_name}: {e}")
        
        if self.attribute_name:
            if self._attribute is None:
                self._attribute = getattr(self._module, self.attribute_name)
            return self._attribute(*args, **kwargs)
        
        return self._module(*args, **kwargs)


def lazy_import(module_name: str, attribute_name: Optional[str] = None):
    """Create a lazy import wrapper for a module or attribute."""
    return LazyImport(module_name, attribute_name)


# Common operations lazy imports
def get_validate_file_path():
    """Get validate_file_path function lazily."""
    try:
        from src.utils.base_utilities import validate_file_path
        return validate_file_path
    except ImportError:
        # Fallback implementation
        def validate_file_path(file_path):
            import os
            from pathlib import Path
            try:
                path = Path(file_path)
                return path.exists() and path.is_file()
            except (TypeError, ValueError):
                return False
        return validate_file_path


def get_safe_correlation_matrix():
    """Get safe_correlation_matrix function lazily."""
    try:
        from src.utils.base_matrix_operations import safe_correlation_matrix
        return safe_correlation_matrix
    except ImportError:
        # Fallback implementation
        def safe_correlation_matrix(data, **kwargs):
            import numpy as np
            import pandas as pd
            try:
                if isinstance(data, pd.DataFrame):
                    return data.corr(**kwargs)
                else:
                    return np.corrcoef(data, **kwargs)
            except Exception:
                return np.eye(min(data.shape))
        return safe_correlation_matrix


def get_create_directory_safe():
    """Get create_directory_safe function lazily."""
    try:
        from src.utils.base_utilities import create_directory_safe
        return create_directory_safe
    except ImportError:
        # Fallback implementation
        def create_directory_safe(directory_path):
            import os
            from pathlib import Path
            try:
                path = Path(directory_path)
                path.mkdir(parents=True, exist_ok=True)
                return True
            except Exception:
                return False
        return create_directory_safe


def get_unified_matrix_operations():
    """Get unified matrix operations lazily."""
    try:
        from src.utils.matrix_operations import get_unified_matrix_operations
        return get_unified_matrix_operations
    except ImportError:
        # Fallback implementation
        def get_unified_matrix_operations():
            return None
        return get_unified_matrix_operations


# Lazy import instances for common functions
validate_file_path = lazy_import('src.utils.common_operations', 'validate_file_path')
safe_correlation_matrix = lazy_import('src.utils.matrix_operations', 'safe_correlation_matrix')
create_directory_safe = lazy_import('src.utils.common_operations', 'create_directory_safe')
unified_matrix_operations = lazy_import('src.utils.matrix_operations', 'get_unified_matrix_operations')
