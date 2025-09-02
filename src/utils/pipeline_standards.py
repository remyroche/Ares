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

# Try to import optional dependencies
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

from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class DataQualityLevel(Enum):
    """Data quality levels."""
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
    issues: List[ValidationIssue]
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PipelineStandards:
    """Pipeline standards and utilities."""
    
    def __init__(self):
        """Initialize pipeline standards."""
        self.logger = logging.getLogger(__name__)
        self.data_quality_levels = DataQualityLevel
        self.timestamp_format = "%Y-%m-%d_%H-%M-%S"
        self.file_naming_convention = "snake_case"
    
    def get_timestamp(self) -> str:
        """Get current timestamp in standard format."""
        return datetime.now(timezone.utc).strftime(self.timestamp_format)
    
    def validate_file_path(self, file_path: Union[str, Path]) -> bool:
        """Validate file path format."""
        path = Path(file_path)
        return path.suffix in ['.csv', '.parquet', '.json', '.pkl', '.joblib']
    
    def get_standard_directory_structure(self) -> Dict[str, str]:
        """Get standard directory structure."""
        return {
            'data': 'data/',
            'models': 'data_cache/models/',
            'logs': 'logs/',
            'reports': 'reports/',
            'configs': 'config/',
            'scripts': 'scripts/'
        }

# Global instance
pipeline_standards = PipelineStandards()