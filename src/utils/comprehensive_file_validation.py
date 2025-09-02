"""
Comprehensive File Validation Framework

This module provides comprehensive file validation capabilities for the Ares trading bot system.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.logger import system_logger

class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    issue_type: str
    severity: ValidationSeverity
    description: str
    details: Optional[Dict[str, Any]] = None
    affected_columns: Optional[List[str]] = None
    affected_rows: Optional[List[int]] = None

@dataclass
class FileValidationResult:
    """Result of file validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    metadata: Optional[Dict[str, Any]] = None

class ComprehensiveFileValidator:
    """Comprehensive file validator for various file types."""
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize ComprehensiveFileValidator."""
        self.config = config or {}
        self.logger = system_logger.getChild("ComprehensiveFileValidator")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize ComprehensiveFileValidator."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    
    def validate_file_format(self, file_path: str, expected_format: Optional[str] = None, context: str = "") -> FileValidationResult:
        """Validate file format and basic structure."""
        try:
            issues = []
            
            # Check if file exists
            if not os.path.exists(file_path):
                issues.append(ValidationIssue(
                    issue_type="file_not_found",
                    severity=ValidationSeverity.CRITICAL,
                    description=f"File not found: {file_path}",
                    details={"file_path": file_path}
                ))
                return FileValidationResult(is_valid=False, issues=issues)
            
            # Check if it's a file
            if not os.path.isfile(file_path):
                issues.append(ValidationIssue(
                    issue_type="not_a_file",
                    severity=ValidationSeverity.CRITICAL,
                    description=f"Path exists but is not a file: {file_path}",
                    details={"file_path": file_path}
                ))
                return FileValidationResult(is_valid=False, issues=issues)
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                issues.append(ValidationIssue(
                    issue_type="empty_file",
                    severity=ValidationSeverity.WARNING,
                    description=f"File is empty: {file_path}",
                    details={"file_path": file_path, "file_size": file_size}
                ))
            
            # Check file extension if expected format is provided
            if expected_format:
                file_ext = Path(file_path).suffix.lower()
                if file_ext != expected_format.lower():
                    issues.append(ValidationIssue(
                        issue_type="format_mismatch",
                        severity=ValidationSeverity.WARNING,
                        description=f"File format mismatch: expected {expected_format}, got {file_ext}",
                        details={"expected": expected_format, "actual": file_ext, "file_path": file_path}
                    ))
            
            # Determine if validation passed
            is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
            
            metadata = {
                "file_path": file_path,
                "file_size": file_size,
                "context": context,
                "validation_timestamp": self.logger.handlers[0].formatter.formatTime(logging.LogRecord("", 0, "", 0, "", (), None)) if self.logger.handlers else None
            }
            
            return FileValidationResult(
                is_valid=is_valid,
                issues=issues,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.exception(f"❌ Error in file format validation: {e}")
            return FileValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    issue_type="validation_error",
                    severity=ValidationSeverity.CRITICAL,
                    description=f"Validation error: {e}",
                    details={"error": str(e), "file_path": file_path}
                )]
            )

# Export the main classes and functions
__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "FileValidationResult",
    "ComprehensiveFileValidator"
]