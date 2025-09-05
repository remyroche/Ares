#!/usr/bin/env python3
"""Utils package for data collection pipeline."""

try:
    from .data_operations_utils import (
        DataFormat,
        CompressionType,
        DataOperationResult,
        DataQualityMetrics,
        DataFormatter,
        DataAnalyzer,
        DataAccessManager,
        DataStorageManager,
        ErrorHandler
    )
except ImportError:
    # Fallback implementations
    class DataFormat:
        pass
    class CompressionType:
        pass
    class DataOperationResult:
        pass
    class DataQualityMetrics:
        pass
    class DataFormatter:
        pass
    class DataAnalyzer:
        pass
    class DataAccessManager:
        pass
    class DataStorageManager:
        pass
    class ErrorHandler:
        pass

__all__ = [
    'DataFormat',
    'CompressionType',
    'DataOperationResult',
    'DataQualityMetrics',
    'DataFormatter',
    'DataAnalyzer',
    'DataAccessManager',
    'DataStorageManager',
    'ErrorHandler'
]