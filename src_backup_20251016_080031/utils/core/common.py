from src.utils.tprint import tprint

"""
Consolidated Common Operations Utility Module

This module consolidates functionality from common_operations.py, common_utilities.py, and common.py
into a single, well-organized utility module with comprehensive error handling.
"""

import argparse
import asyncio
import datetime
import glob
import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Create logger instance early to avoid undefined variable errors
logger = logging.getLogger(__name__)

# Import validation with proper error handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NumPy not available: {e}")
    np = None
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pandas not available: {e}")
    
    class _PDStub:
        class DataFrame:
            pass
        class Series:
            pass
    pd = _PDStub()
    PANDAS_AVAILABLE = False

# Import logger utilities with fallback
try:
    from ..logger import log_error_with_context
except ImportError:
    def log_error_with_context(logger: logging.Logger, error: Exception, context: Any=None, operation: Any='', recovery_attempted: Any=False) -> None:
        logger.error(f'Error in {operation}: {error}')

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def safe_json_load(path: Union[str, Path]) -> Dict[str, Any]:
    """Safely load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def safe_json_dump(data: Union[pd.DataFrame, Dict[str, Any]], path: Union[str, Path], indent: int = 2) -> bool:
    """Safely dump JSON data."""
    try:
        ensure_directory(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        tprint(f'Error saving JSON to {path}: {e}')
        return False

def safe_read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Safely read parquet file."""
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
        else:
            tprint(f'Parquet file not found: {path}')
            return pd.DataFrame()
    except Exception as e:
        tprint(f'Error reading parquet file {path}: {e}')
        return pd.DataFrame()

def ensure_directory(path: Union[str, Path]) -> None:
    """Ensure directory exists, create if it doesn't."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")

# =============================================================================
# DATA OPERATIONS
# =============================================================================

def safe_dataframe_operation(df: pd.DataFrame, operation: str, **kwargs) -> Any:
    """
    Safely perform operations on a DataFrame with error handling.
    
    Args:
        df: DataFrame to operate on
        operation: Name of the operation to perform
        **kwargs: Additional arguments for the operation
        
    Returns:
        Result of the operation or None if failed
    """
    try:
        if operation == 'dropna':
            return df.dropna(**kwargs)
        elif operation == 'fillna':
            return df.fillna(**kwargs)
        elif operation == 'sort_values':
            return df.sort_values(**kwargs)
        elif operation == 'reset_index':
            return df.reset_index(**kwargs)
        elif operation == 'copy':
            return df.copy(**kwargs)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return df
    except Exception as e:
        logger.error(f"Error performing {operation} on DataFrame: {e}")
        return df

# =============================================================================
# DICTIONARY AND LIST OPERATIONS
# =============================================================================

def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    return dictionary.get(key, default)

def safe_set(dictionary: Dict, key: str, value: Any) -> None:
    """Safely set value in dictionary."""
    dictionary[key] = value

def safe_list_get(lst: List, index: int, default: Any = None) -> Any:
    """Safely get item from list by index."""
    try:
        return lst[index]
    except (IndexError, TypeError):
        return default

def safe_list_append(lst: List, item: Any) -> None:
    """Safely append item to list."""
    if isinstance(lst, list):
        lst.append(item)

def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def flatten_list(nested_list: List) -> List:
    """Flatten nested list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# =============================================================================
# TYPE VALIDATION AND CONVERSION
# =============================================================================

def validate_type(value: Any, expected_type: type) -> bool:
    """Validate if value is of expected type."""
    return isinstance(value, expected_type)

def safe_convert(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely convert value to target type."""
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def create_fallback_logger(name: str = 'fallback'):
    """Create a fallback logger when the main logging system is unavailable.

    Args:
        name: Name for the logger (default: 'fallback')

    Returns:
        logging.Logger: Configured fallback logger
    """
    logger = logging.getLogger(name)
    # Avoid duplicate emissions
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Align default fallback level with light verbosity
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger

def create_fallback_decorator():
    """Create a fallback decorator that accepts keyword arguments like fallback."""
    def decorator(*args, **kwargs):
        def inner_decorator(func):
            return func
        return inner_decorator
    return decorator

# =============================================================================
# MAIN COMMON OPERATIONS CLASS
# =============================================================================

class CommonOperations:
    """Consolidated common operations class."""
    
    def __init__(self):
        self.logger = create_fallback_logger()
    
    def safe_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Safely execute an operation with error handling."""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in operation {operation.__name__}: {e}")
            return None
    
    def batch_operation(self, items: List[Any], operation: Callable, **kwargs) -> List[Any]:
        """Apply operation to a batch of items."""
        results = []
        for item in items:
            result = self.safe_operation(operation, item, **kwargs)
            results.append(result)
        return results

# Global instance
_common_operations = CommonOperations()

def get_common_operations() -> CommonOperations:
    """Get the global common operations instance."""
    return _common_operations
