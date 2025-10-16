"""
Core constants for the Ares project.
"""

import os
from typing import List, Dict, Any

# Database constants
DEFAULT_DATABASE_PATH = os.path.join(os.getcwd(), "data", "ares.db")
DEFAULT_MAX_RECOVERY_ATTEMPTS = 3

# Feature engineering constants
FEATURE_POOL_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'returns', 'volatility', 'momentum', 'rsi', 'macd'
]

# Validation constants
VALIDATION_FUNCTIONS = [
    'validate_dataframe',
    'validate_numpy_array',
    'validate_type',
    'validate_datetime'
]

# Time constants
UTC = 'UTC'

# Type checking constant
TYPE_CHECKING = False

# Framework availability flags
TORCH_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    pass

# Synchronous operations flag
SYNCHRONOUS = True

# Data operation errors (moved from error_classes for consistency)
DATA_OPERATION_ERRORS = {
    'read_error': 'Failed to read data',
    'write_error': 'Failed to write data',
    'validation_error': 'Data validation failed',
    'format_error': 'Invalid data format'
}

# Machine learning constants
RFE = 'Recursive Feature Elimination'
_RFC = 'Random Forest Classifier'

# Generic constants (commonly used single letters)
S = 'string'
U = 'unicode'

# Export all constants
__all__ = [
    'DEFAULT_DATABASE_PATH',
    'DEFAULT_MAX_RECOVERY_ATTEMPTS',
    'FEATURE_POOL_COLUMNS',
    'VALIDATION_FUNCTIONS',
    'UTC',
    'TYPE_CHECKING',
    'TORCH_AVAILABLE',
    'SYNCHRONOUS',
    'DATA_OPERATION_ERRORS',
    'RFE',
    '_RFC',
    'S',
    'U'
]
