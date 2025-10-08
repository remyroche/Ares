"""
Error Handling Module.

This module provides standardized error handling for feature lookback optimization.
"""

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

from .error_handler import StandardizedErrorHandler, ErrorSeverity, ErrorCategory, ErrorDetails
# ErrorRecoveryResult doesn't exist - removed from import

# Log module import
if TPRINT_AVAILABLE:
    tprint_debug("🔧 Error Handling Module imported successfully")

__all__ = [
    'StandardizedErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorDetails',
    'ErrorRecoveryResult'
]
