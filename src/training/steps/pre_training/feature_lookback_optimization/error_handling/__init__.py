"""
Error Handling Module.

This module provides standardized error handling for feature lookback optimization.
"""

# Import centralized tprint utilities
from ..utils.tprint_utils import tprint_debug, TPRINT_AVAILABLE

from .error_handler import StandardizedErrorHandler, ErrorSeverity, ErrorCategory, ErrorDetails
# ErrorRecoveryResult doesn't exist - removed from import

# Log module import
if TPRINT_AVAILABLE:
    tprint_debug("🔧 Error Handling Module imported successfully")

__all__ = [
    'StandardizedErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorDetails'
]
