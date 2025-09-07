"""
Data utilities with passthrough functions for common operations.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, Union
import logging

def safe_dataframe_operation(df: pd.DataFrame, operation: str, **kwargs) -> Any:
    """Safely perform dataframe operations."""
    try:
        if hasattr(df, operation):
            return getattr(df, operation)(**kwargs)
        return None
    except Exception:
        return None

def validate_dataframe(df: Any) -> bool:
    """Validate if object is a dataframe."""
    return isinstance(df, pd.DataFrame)

def safe_numpy_operation(arr: np.ndarray, operation: str, **kwargs) -> Any:
    """Safely perform numpy operations."""
    try:
        if hasattr(arr, operation):
            return getattr(arr, operation)(**kwargs)
        return None
    except Exception:
        return None

def validate_numpy_array(arr: Any) -> bool:
    """Validate if object is a numpy array."""
    return isinstance(arr, np.ndarray)

def get_data_info(data: Any) -> dict:
    """Get basic info about data object."""
    info = {
        'type': type(data).__name__,
        'shape': getattr(data, 'shape', None),
        'dtype': getattr(data, 'dtype', None),
        'size': getattr(data, 'size', len(data) if hasattr(data, '__len__') else None)
    }
    return info

# Export all functions
__all__ = [
    'safe_dataframe_operation',
    'validate_dataframe',
    'safe_numpy_operation',
    'validate_numpy_array',
    'get_data_info'
]
