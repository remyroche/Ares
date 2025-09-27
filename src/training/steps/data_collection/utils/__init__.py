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
except ImportError as import_error:
    # Fallback implementations that surface the missing dependency clearly while still
    # providing lightweight value objects for the simple data containers that callers
    # may expect during import time.
    from dataclasses import dataclass, field
    from enum import Enum
    from typing import Any, Dict, List, Optional

    class DataFormat(str, Enum):
        """Minimal enum describing supported data formats."""

        PARQUET = "parquet"
        CSV = "csv"
        JSON = "json"
        PICKLE = "pickle"

    class CompressionType(str, Enum):
        """Minimal enum describing supported compression algorithms."""

        NONE = "none"
        GZIP = "gzip"
        BZIP2 = "bzip2"
        LZ4 = "lz4"

    @dataclass(slots=True)
    class DataOperationResult:
        """Simple container that mirrors the public attributes of the full result type."""

        success: bool
        message: str
        data: Optional[Any] = None
        metadata: Optional[Dict[str, Any]] = None
        execution_time: float = 0.0
        warnings: List[str] = field(default_factory=list)
        errors: List[str] = field(default_factory=list)

    @dataclass(slots=True)
    class DataQualityMetrics:
        """Container representing the key quality metrics returned by analyzers."""

        total_rows: int = 0
        total_columns: int = 0
        null_counts: Dict[str, int] = field(default_factory=dict)
        duplicate_count: int = 0
        data_types: Dict[str, str] = field(default_factory=dict)
        memory_usage: int = 0
        file_size: int = 0
        quality_score: float = 0.0
        issues: List[str] = field(default_factory=list)
        timestamp: Optional[str] = None

    class _UnavailableComponent:
        """Helper that surfaces unavailability when instantiation is attempted."""

        __slots__ = ()

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
            raise RuntimeError(
                "The optional 'data_operations_utils' components are unavailable because "
                f"the module could not be imported ({import_error})."
            ) from import_error

    class DataFormatter(_UnavailableComponent):
        """Fallback that raises a descriptive error when used."""

    class DataAnalyzer(_UnavailableComponent):
        """Fallback that raises a descriptive error when used."""

    class DataAccessManager(_UnavailableComponent):
        """Fallback that raises a descriptive error when used."""

    class DataStorageManager(_UnavailableComponent):
        """Fallback that raises a descriptive error when used."""

    class ErrorHandler(_UnavailableComponent):
        """Fallback that raises a descriptive error when used."""

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
