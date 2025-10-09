"""
Feature Selection Error Handling Module

This module provides comprehensive error handling and recovery for feature selection
operations using tprint for enhanced logging and user feedback.
"""

from .enhanced_error_handler import (
    FeatureSelectionError,
    InsufficientDataError,
    SelectionConvergenceError,
    ConfigurationError,
    EnhancedErrorHandler,
    robust_feature_selection,
    create_error_handler
)

__all__ = [
    'FeatureSelectionError',
    'InsufficientDataError', 
    'SelectionConvergenceError',
    'ConfigurationError',
    'EnhancedErrorHandler',
    'robust_feature_selection',
    'create_error_handler'
]