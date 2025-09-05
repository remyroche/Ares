#!/usr/bin/env python3
"""Error handling package for data collection pipeline."""

from .enhanced_error_handler import (
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    ErrorReport,
    DataCollectionError,
    DataQualityError,
    NetworkError,
    StorageError,
    ValidationError,
    ProcessingError,
    ConfigurationError,
    PermissionError,
    TimeoutError,
    EnhancedErrorHandler
)

__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'RecoveryStrategy',
    'ErrorContext',
    'ErrorReport',
    'DataCollectionError',
    'DataQualityError',
    'NetworkError',
    'StorageError',
    'ValidationError',
    'ProcessingError',
    'ConfigurationError',
    'PermissionError',
    'TimeoutError',
    'EnhancedErrorHandler'
]