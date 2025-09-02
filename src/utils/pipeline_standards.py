"""
Pipeline Standards and Data Quality Framework

This module provides standardized pipeline components and data quality validation.
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
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Data quality levels
class DataQualityLevel(str, Enum):
    """Data quality severity levels."""
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
    summary: Optional[Dict[str, Any]] = None

@dataclass
class PipelineStandards:
    """Pipeline standards configuration."""
    name: str = "default"
    version: str = "1.0.0"
    description: str = "Default pipeline standards"
    data_quality_threshold: DataQualityLevel = DataQualityLevel.WARNING
    validation_enabled: bool = True
    logging_enabled: bool = True
    metrics_enabled: bool = True

# Default instance
pipeline_standards = PipelineStandards()

# Export main classes
__all__ = [
    "DataQualityLevel",
    "ValidationIssue", 
    "ValidationResult",
    "PipelineStandards",
    "pipeline_standards"
]