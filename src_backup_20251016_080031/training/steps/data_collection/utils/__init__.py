#!/usr/bin/env python3
"""Utils package for data collection pipeline."""

# Import common operations functions
from .common_operations import (
    safe_json_load,
    safe_json_save,
    validate_data_directory,
    ensure_directory_exists,
    get_file_size_mb,
    validate_parquet_file,
    calculate_data_quality_metrics,
    format_file_size,
    get_directory_stats,
    safe_remove_file,
    create_backup_file
)

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
    # Common operations
    'safe_json_load',
    'safe_json_save',
    'validate_data_directory',
    'ensure_directory_exists',
    'get_file_size_mb',
    'validate_parquet_file',
    'calculate_data_quality_metrics',
    'format_file_size',
    'get_directory_stats',
    'safe_remove_file',
    'create_backup_file',
    # Data operations utils
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