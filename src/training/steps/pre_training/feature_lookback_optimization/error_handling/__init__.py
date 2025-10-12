"""
Error Handling Module.

This module provides standardized error handling for feature lookback optimization.
"""

# Import centralized tprint utilities
from ..utils.tprint_utils import tprint_debug, TPRINT_AVAILABLE

from .error_handler import (
    StandardizedErrorHandler, 
    ErrorSeverity, 
    ErrorCategory,
    ErrorDetails,
    OptimizationError,
    DataValidationError,
    ScoringError,
    CacheError,
    MemoryError,
    safe_operation,
    safe_mi_calculation,
    safe_correlation_calculation,
    safe_dataframe_operation,
    safe_numpy_operation,
    get_error_handler
)

# Log module import
if TPRINT_AVAILABLE:
    tprint_debug("🔧 Error Handling Module imported successfully")

__all__ = [
    'StandardizedErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorDetails',
    'OptimizationError',
    'DataValidationError',
    'ScoringError',
    'CacheError',
    'MemoryError',
    'safe_operation',
    'safe_mi_calculation',
    'safe_correlation_calculation',
    'safe_dataframe_operation',
    'safe_numpy_operation',
    'get_error_handler'
]
