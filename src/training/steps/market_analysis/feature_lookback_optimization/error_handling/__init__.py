"""
Error Handling Module.

This module provides standardized error handling for feature lookback optimization.
"""

from .error_handler import StandardizedErrorHandler, ErrorSeverity, ErrorCategory, ErrorDetails, ErrorRecoveryResult

__all__ = [
    'StandardizedErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorDetails',
    'ErrorRecoveryResult'
]
