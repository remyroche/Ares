"""
Pipeline Standards and Utilities

This module provides standardized utilities for the data pipeline including:
- Import management with consistent fallback patterns
- Directory structure standardization
- Timestamp format standardization
- Schema validation
- Data quality validation
- File naming conventions
- Metadata standards
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DataQualityLevel(Enum):
    """Data quality levels for validation."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: DataQualityLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    column: Optional[str] = None
    row_count: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStandards:
    """Centralized pipeline standards and utilities."""

    def __init__(self):
        """Initialize PipelineStandards."""
        self.logger = self._setup_logger()
        self.is_initialized = True

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the pipeline standards."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the given name."""
        return logging.getLogger(name)

    def build_path(self, path_type: str, exchange: str, symbol: str) -> Path:
        """Build standardized path for data storage."""
        base_path = Path("data_cache") / exchange / symbol / path_type
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def standardize_timestamp(self, timestamp: Union[str, datetime, Any]) -> str:
        """Standardize timestamp format."""
        if isinstance(timestamp, str):
            try:
                if PANDAS_AVAILABLE:
                    timestamp = pd.to_datetime(timestamp)
                else:
                    # Fallback to datetime parsing
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        elif PANDAS_AVAILABLE and hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()
        
        if isinstance(timestamp, datetime):
            return timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def validate_schema(self, data: Any, expected_schema: Dict[str, str]) -> ValidationResult:
        """Validate DataFrame against expected schema."""
        if not PANDAS_AVAILABLE:
            # Fallback validation without pandas
            return ValidationResult(
                is_valid=True,
                issues=[],
                score=1.0,
                metadata={"note": "Pandas not available, skipping schema validation"}
            )
        
        issues = []
        is_valid = True
        
        for expected_col, expected_type in expected_schema.items():
            if expected_col not in data.columns:
                issues.append(ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Missing required column: {expected_col}",
                    column=expected_col
                ))
                is_valid = False
            else:
                # Basic type validation
                actual_type = str(data[expected_col].dtype)
                if expected_type not in actual_type and actual_type not in expected_type:
                    issues.append(ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column {expected_col} has type {actual_type}, expected {expected_type}",
                        column=expected_col
                    ))
        
        score = 1.0 - (len(issues) / len(expected_schema)) if expected_schema else 1.0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            score=score,
            metadata={"schema_columns": list(expected_schema.keys())}
        )

    def standardize_filename(self, base_name: str, timestamp: Optional[Union[str, datetime]] = None, 
                           extension: str = ".parquet") -> str:
        """Generate standardized filename."""
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = self.standardize_timestamp(timestamp)
        return f"{base_name}_{timestamp_str}{extension}"

    def get_metadata_standards(self) -> Dict[str, Any]:
        """Get standard metadata fields."""
        return {
            "created_at": datetime.now().isoformat(),
            "pipeline_version": "1.2.3",
            "data_format": "parquet",
            "encoding": "utf-8",
            "compression": "snappy"
        }


# Global instance
pipeline_standards = PipelineStandards()


def get_pipeline_logger(name: str) -> logging.Logger:
    """Get a pipeline logger instance."""
    return pipeline_standards.get_logger(name)


def build_standard_path(path_type: str, exchange: str, symbol: str) -> Path:
    """Build standard path for data storage."""
    return pipeline_standards.build_path(path_type, exchange, symbol)


def standardize_timestamp(timestamp: Union[str, datetime, Any]) -> str:
    """Standardize timestamp format."""
    return pipeline_standards.standardize_timestamp(timestamp)


def validate_schema(data: Any, expected_schema: Dict[str, str]) -> ValidationResult:
    """Validate DataFrame against expected schema."""
    return pipeline_standards.validate_schema(data, expected_schema)