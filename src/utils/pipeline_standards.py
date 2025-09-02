"""
Pipeline Standards and Validation Framework

This module provides pipeline standards, validation results, and data quality levels
for the Ares trading bot system.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from src.utils.logger import system_logger

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
    issues: List[ValidationIssue]
    metadata: Optional[Dict[str, Any]] = None

class PipelineStandards:
    """Pipeline standards and validation framework."""
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize PipelineStandards."""
        self.config = config or {}
        self.logger = system_logger.getChild("PipelineStandards")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize PipelineStandards."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False

# Create global instance
pipeline_standards = PipelineStandards()

# Export the main classes and functions
__all__ = [
    "DataQualityLevel",
    "ValidationIssue",
    "ValidationResult",
    "PipelineStandards",
    "pipeline_standards"
]