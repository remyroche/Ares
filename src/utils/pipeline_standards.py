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
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

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
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PipelineStandards:
    """Pipeline standards and utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.standards_version = "1.0.0"
    
    def validate_file_structure(self, file_path: Path) -> ValidationResult:
        """Validate file structure according to standards."""
        issues = []
        
        if not file_path.exists():
            issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message=f"File does not exist: {file_path}"
            ))
            return ValidationResult(is_valid=False, issues=issues)
        
        # Basic validation - file exists and is readable
        try:
            with open(file_path, 'r') as f:
                f.read(1)
        except Exception as e:
            issues.append(ValidationIssue(
                severity=DataQualityLevel.CRITICAL,
                message=f"Cannot read file: {e}"
            ))
            return ValidationResult(is_valid=False, issues=issues)
        
        return ValidationResult(is_valid=True, issues=issues)
    
    def get_timestamp_format(self) -> str:
        """Get standardized timestamp format."""
        return "%Y-%m-%d %H:%M:%S UTC"
    
    def format_timestamp(self, dt: datetime) -> str:
        """Format timestamp according to standards."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(self.get_timestamp_format())

# Global instance
pipeline_standards = PipelineStandards()