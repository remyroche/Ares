"""
Comprehensive File Format Validation Module

This module provides comprehensive file format validation for steps 1, 1.5, 2, and 4.
It includes validation for:
- Type of file
- Type of strings, boolean values, etc.
- Number of columns
- Column names
- Column completeness (no empty values)
- Index validation
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards


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
    file_path: str
    file_type: str
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    validation_timestamp: datetime


class ComprehensiveFileValidator:
    """
    Comprehensive file format validator for training pipeline steps.
    
    Validates:
    - File type and format
    - Data types (strings, booleans, numerics, etc.)
    - Number of columns
    - Column names
    - Column completeness (no empty values)
    - Index validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the validator with configuration."""
        self.logger = system_logger.getChild("ComprehensiveFileValidator")
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            "file_types": {
                "parquet": {
                    "extensions": [".parquet"],
                    "description": "Parquet file format",
                    "required": True
                },
                "csv": {
                    "extensions": [".csv"],
                    "description": "CSV file format",
                    "required": False
                },
                "json": {
                    "extensions": [".json"],
                    "description": "JSON file format",
                    "required": False
                }
            },
            "data_quality": {
                "max_null_ratio": 0.5,  # Maximum allowed null ratio per column
                "min_rows": 1,  # Minimum number of rows required
                "max_duplicate_ratio": 0.1,  # Maximum allowed duplicate ratio
                "check_data_types": True,
                "check_index": True,
                "check_column_names": True,
                "check_completeness": True,
                "check_file_paths": True,
                "max_filename_length": 255,
                "prefer_relative_paths": True,
                "validate_filename_patterns": True
            },
            "expected_filename_patterns": {
                "step1": {
                    "klines": r"klines_{exchange}_{symbol}_{timeframe}_consolidated\.parquet",
                    "aggtrades": r"aggtrades_{exchange}_{symbol}_consolidated\.parquet"
                },
                "step01_5": {
                    "unified_data": r"unified_{exchange}_{symbol}_{timeframe}\.parquet",
                    "config": r"unified_{exchange}_{symbol}_{timeframe}_config\.json"
                },
                "step2": {
                    "features": r"features_{exchange}_{symbol}_{split}\.parquet"
                },
                "step4": {
                    "labeled": r"{exchange}_{symbol}_labeled_{split}\.parquet"
                }
            },
            "expected_schemas": {
                "klines": {
                    "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
                    "expected_types": {
                        "timestamp": ["int64", "datetime64[ns]"],
                        "open": ["float64"],
                        "high": ["float64"],
                        "low": ["float64"],
                        "close": ["float64"],
                        "volume": ["float64"]
                    },
                    "description": "OHLCV klines data"
                },
                "aggtrades": {
                    "required_columns": ["timestamp", "price", "quantity"],
                    "expected_types": {
                        "timestamp": ["int64", "datetime64[ns]"],
                        "price": ["float64"],
                        "quantity": ["float64"]
                    },
                    "description": "Aggregated trades data"
                },
                "futures": {
                    "required_columns": ["timestamp", "fundingRate"],
                    "expected_types": {
                        "timestamp": ["int64", "datetime64[ns]"],
                        "fundingRate": ["float64"]
                    },
                    "description": "Futures funding rate data"
                },
                "features": {
                    "required_columns": [],  # Features can have various columns
                    "expected_types": {
                        "timestamp": ["int64", "datetime64[ns]"],
                        # Other columns can be float64, int64, etc.
                    },
                    "description": "Feature engineering output"
                }
            }
        }
    
    def validate_file_format(
        self, 
        file_path: str, 
        expected_schema: Optional[str] = None,
        step_name: str = "unknown"
    ) -> FileValidationResult:
        """
        Comprehensive file format validation.
        
        Args:
            file_path: Path to the file to validate
            expected_schema: Expected schema name (klines, aggtrades, futures, features)
            step_name: Name of the step for logging context
            
        Returns:
            FileValidationResult with validation details
        """
        self.logger.info(f"🔍 Validating file format for {step_name}: {file_path}")
        
        issues = []
        summary = {
            "file_path": file_path,
            "step_name": step_name,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # 1. Validate file type
        file_type_result = self._validate_file_type(file_path)
        issues.extend(file_type_result["issues"])
        summary["file_type"] = file_type_result["file_type"]
        
        if not file_type_result["is_valid"]:
            return FileValidationResult(
                is_valid=False,
                file_path=file_path,
                file_type=file_type_result["file_type"],
                issues=issues,
                summary=summary,
                validation_timestamp=datetime.now()
            )
        
        # 2. Load and validate data
        try:
            df = self._load_file(file_path, file_type_result["file_type"])
            summary["shape"] = df.shape
            summary["memory_usage"] = df.memory_usage(deep=True).sum()
            
            # 3. Validate number of columns
            column_count_result = self._validate_column_count(df, expected_schema)
            issues.extend(column_count_result["issues"])
            summary["column_count"] = len(df.columns)
            
            # 4. Validate column names
            column_names_result = self._validate_column_names(df, expected_schema)
            issues.extend(column_names_result["issues"])
            summary["column_names"] = list(df.columns)
            
            # 5. Validate data types
            data_types_result = self._validate_data_types(df, expected_schema)
            issues.extend(data_types_result["issues"])
            summary["data_types"] = df.dtypes.to_dict()
            
            # 6. Validate column completeness (no empty values)
            completeness_result = self._validate_column_completeness(df)
            issues.extend(completeness_result["issues"])
            summary["null_counts"] = df.isnull().sum().to_dict()
            summary["null_ratios"] = (df.isnull().sum() / len(df)).to_dict()
            
            # 7. Validate index
            index_result = self._validate_index(df)
            issues.extend(index_result["issues"])
            summary["index_info"] = {
                "type": str(type(df.index)),
                "is_unique": df.index.is_unique,
                "is_monotonic": df.index.is_monotonic_increasing if hasattr(df.index, 'is_monotonic_increasing') else None
            }
            
            # 8. Additional quality checks
            quality_result = self._validate_data_quality(df)
            issues.extend(quality_result["issues"])
            summary["quality_metrics"] = quality_result["metrics"]
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="file_loading_error",
                severity=ValidationSeverity.CRITICAL,
                description=f"Failed to load file: {str(e)}",
                details={"error": str(e)}
            ))
            return FileValidationResult(
                is_valid=False,
                file_path=file_path,
                file_type=file_type_result["file_type"],
                issues=issues,
                summary=summary,
                validation_timestamp=datetime.now()
            )
        
        # Determine overall validity
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        error_issues = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        
        is_valid = len(critical_issues) == 0 and len(error_issues) == 0
        
        # Log validation results
        if is_valid:
            self.logger.info(f"✅ File validation passed for {step_name}: {file_path}")
        else:
            self.logger.warning(f"⚠️ File validation issues found for {step_name}: {file_path}")
            for issue in issues:
                self.logger.warning(f"   - {issue.severity.value.upper()}: {issue.description}")
        
        return FileValidationResult(
            is_valid=is_valid,
            file_path=file_path,
            file_type=file_type_result["file_type"],
            issues=issues,
            summary=summary,
            validation_timestamp=datetime.now()
        )
    
    def _validate_file_path_and_name(self, file_path: str) -> Dict[str, Any]:
        """Validate file path and name structure."""
        issues = []
        
        # Check if file exists
        if not os.path.exists(file_path):
            issues.append(ValidationIssue(
                issue_type="file_not_found",
                severity=ValidationSeverity.CRITICAL,
                description=f"File does not exist: {file_path}"
            ))
            return {
                "is_valid": False,
                "issues": issues
            }
        
        # Validate path structure
        path_obj = Path(file_path)
        
        # Check for invalid characters in path
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        path_str = str(path_obj)
        found_invalid_chars = [char for char in invalid_chars if char in path_str]
        if found_invalid_chars:
            issues.append(ValidationIssue(
                issue_type="invalid_path_characters",
                severity=ValidationSeverity.ERROR,
                description=f"Path contains invalid characters: {found_invalid_chars}",
                details={"invalid_chars": found_invalid_chars, "path": path_str}
            ))
        
        # Check for absolute path (optional validation)
        if self.config.get("data_quality", {}).get("prefer_relative_paths", False):
            if path_obj.is_absolute():
                issues.append(ValidationIssue(
                    issue_type="absolute_path_detected",
                    severity=ValidationSeverity.WARNING,
                    description="File uses absolute path (relative paths preferred)",
                    details={"path": path_str}
                ))
        
        # Validate filename structure
        filename = path_obj.name
        
        # Check filename length
        max_filename_length = self.config.get("data_quality", {}).get("max_filename_length", 255)
        if len(filename) > max_filename_length:
            issues.append(ValidationIssue(
                issue_type="filename_too_long",
                severity=ValidationSeverity.WARNING,
                description=f"Filename too long ({len(filename)} chars, max: {max_filename_length})",
                details={"filename": filename, "length": len(filename)}
            ))
        
        # Check for expected filename patterns based on step
        expected_patterns = self.config.get("expected_filename_patterns", {})
        if expected_patterns and self.config.get("data_quality", {}).get("validate_filename_patterns", True):
            # Validate filename against expected patterns
            import re
            pattern_matched = False
            
            for step_patterns in expected_patterns.values():
                for pattern_name, pattern in step_patterns.items():
                    try:
                        if re.match(pattern, filename):
                            pattern_matched = True
                            break
                    except re.error:
                        # Invalid regex pattern, skip
                        continue
                if pattern_matched:
                    break
            
            if not pattern_matched:
                issues.append(ValidationIssue(
                    issue_type="filename_pattern_mismatch",
                    severity=ValidationSeverity.WARNING,
                    description=f"Filename '{filename}' doesn't match expected patterns",
                    details={"filename": filename, "expected_patterns": list(expected_patterns.keys())}
                ))
        
        return {
            "is_valid": len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]) == 0,
            "issues": issues,
            "path_info": {
                "filename": filename,
                "directory": str(path_obj.parent),
                "is_absolute": path_obj.is_absolute(),
                "path_length": len(path_str)
            }
        }
    
    def _validate_file_type(self, file_path: str) -> Dict[str, Any]:
        """Validate file type and existence."""
        issues = []
        
        # First validate path and name
        path_validation = self._validate_file_path_and_name(file_path)
        issues.extend(path_validation["issues"])
        
        if not path_validation["is_valid"]:
            return {
                "is_valid": False,
                "file_type": "unknown",
                "issues": issues
            }
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            issues.append(ValidationIssue(
                issue_type="empty_file",
                severity=ValidationSeverity.CRITICAL,
                description=f"File is empty: {file_path}"
            ))
            return {
                "is_valid": False,
                "file_type": "unknown",
                "issues": issues
            }
        
        # Determine file type from extension
        file_extension = Path(file_path).suffix.lower()
        supported_extensions = []
        for file_type, config in self.config["file_types"].items():
            supported_extensions.extend(config["extensions"])
        
        if file_extension not in supported_extensions:
            issues.append(ValidationIssue(
                issue_type="unsupported_file_type",
                severity=ValidationSeverity.ERROR,
                description=f"Unsupported file type: {file_extension}. Supported: {supported_extensions}",
                details={"file_extension": file_extension, "supported_extensions": supported_extensions}
            ))
        
        # Determine file type
        file_type = "unknown"
        for file_type_name, config in self.config["file_types"].items():
            if file_extension in config["extensions"]:
                file_type = file_type_name
                break
        
        return {
            "is_valid": len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0,
            "file_type": file_type,
            "issues": issues,
            "file_size": file_size,
            "path_info": path_validation.get("path_info", {})
        }
    
    def _load_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load file based on its type."""
        if file_type == "parquet":
            return pd.read_parquet(file_path)
        elif file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _validate_column_count(self, df: pd.DataFrame, expected_schema: Optional[str]) -> Dict[str, Any]:
        """Validate number of columns."""
        issues = []
        
        if expected_schema and expected_schema in self.config["expected_schemas"]:
            schema = self.config["expected_schemas"][expected_schema]
            expected_count = len(schema["required_columns"])
            actual_count = len(df.columns)
            
            if actual_count < expected_count:
                issues.append(ValidationIssue(
                    issue_type="insufficient_columns",
                    severity=ValidationSeverity.ERROR,
                    description=f"Expected at least {expected_count} columns, got {actual_count}",
                    details={"expected": expected_count, "actual": actual_count}
                ))
        
        return {
            "issues": issues
        }
    
    def _validate_column_names(self, df: pd.DataFrame, expected_schema: Optional[str]) -> Dict[str, Any]:
        """Validate column names."""
        issues = []
        
        if expected_schema and expected_schema in self.config["expected_schemas"]:
            schema = self.config["expected_schemas"][expected_schema]
            required_columns = set(schema["required_columns"])
            actual_columns = set(df.columns)
            
            missing_columns = required_columns - actual_columns
            if missing_columns:
                issues.append(ValidationIssue(
                    issue_type="missing_required_columns",
                    severity=ValidationSeverity.ERROR,
                    description=f"Missing required columns: {missing_columns}",
                    affected_columns=list(missing_columns)
                ))
        
        # Check for duplicate column names
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            issues.append(ValidationIssue(
                issue_type="duplicate_column_names",
                severity=ValidationSeverity.ERROR,
                description=f"Duplicate column names found: {duplicate_columns}",
                affected_columns=duplicate_columns
            ))
        
        return {
            "issues": issues
        }
    
    def _validate_data_types(self, df: pd.DataFrame, expected_schema: Optional[str]) -> Dict[str, Any]:
        """Validate data types of columns."""
        issues = []
        
        if expected_schema and expected_schema in self.config["expected_schemas"]:
            schema = self.config["expected_schemas"][expected_schema]
            expected_types = schema["expected_types"]
            
            for column, expected_type_list in expected_types.items():
                if column in df.columns:
                    actual_type = str(df[column].dtype)
                    if actual_type not in expected_type_list:
                        issues.append(ValidationIssue(
                            issue_type="incorrect_data_type",
                            severity=ValidationSeverity.WARNING,
                            description=f"Column '{column}' has type {actual_type}, expected one of {expected_type_list}",
                            affected_columns=[column],
                            details={"expected": expected_type_list, "actual": actual_type}
                        ))
        
        # Check for mixed data types in columns
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if object column contains mixed types
                unique_types = set(type(x) for x in df[column].dropna())
                if len(unique_types) > 1:
                    issues.append(ValidationIssue(
                        issue_type="mixed_data_types",
                        severity=ValidationSeverity.WARNING,
                        description=f"Column '{column}' contains mixed data types: {unique_types}",
                        affected_columns=[column],
                        details={"types": list(unique_types)}
                    ))
        
        return {
            "issues": issues
        }
    
    def _validate_column_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate column completeness (no empty values)."""
        issues = []
        max_null_ratio = self.config["data_quality"]["max_null_ratio"]
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_ratio = null_count / len(df)
            
            if null_ratio > max_null_ratio:
                issues.append(ValidationIssue(
                    issue_type="high_null_ratio",
                    severity=ValidationSeverity.WARNING,
                    description=f"Column '{column}' has {null_ratio:.2%} null values (max: {max_null_ratio:.2%})",
                    affected_columns=[column],
                    details={"null_count": int(null_count), "null_ratio": float(null_ratio)}
                ))
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            issues.append(ValidationIssue(
                issue_type="empty_columns",
                severity=ValidationSeverity.ERROR,
                description=f"Completely empty columns found: {empty_columns}",
                affected_columns=empty_columns
            ))
        
        return {
            "issues": issues
        }
    
    def _validate_index(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame index."""
        issues = []
        
        # Check if index is unique
        if not df.index.is_unique:
            duplicate_indices = df.index[df.index.duplicated()].tolist()
            issues.append(ValidationIssue(
                issue_type="duplicate_index",
                severity=ValidationSeverity.ERROR,
                description=f"Duplicate index values found: {len(duplicate_indices)} duplicates",
                details={"duplicate_count": len(duplicate_indices)}
            ))
        
        # Check if index is monotonic (for time series data)
        if hasattr(df.index, 'is_monotonic_increasing'):
            if not df.index.is_monotonic_increasing:
                issues.append(ValidationIssue(
                    issue_type="non_monotonic_index",
                    severity=ValidationSeverity.WARNING,
                    description="Index is not monotonically increasing (may indicate out-of-order data)"
                ))
        
        # Check for null values in index
        if df.index.isnull().any():
            null_index_count = df.index.isnull().sum()
            issues.append(ValidationIssue(
                issue_type="null_index_values",
                severity=ValidationSeverity.ERROR,
                description=f"Index contains {null_index_count} null values",
                details={"null_count": int(null_index_count)}
            ))
        
        return {
            "issues": issues
        }
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Additional data quality checks."""
        issues = []
        metrics = {}
        
        # Check for infinite values
        infinite_counts = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            infinite_count = np.isinf(df[column]).sum()
            if infinite_count > 0:
                infinite_counts[column] = infinite_count
                issues.append(ValidationIssue(
                    issue_type="infinite_values",
                    severity=ValidationSeverity.WARNING,
                    description=f"Column '{column}' contains {infinite_count} infinite values",
                    affected_columns=[column],
                    details={"infinite_count": int(infinite_count)}
                ))
        
        metrics["infinite_counts"] = infinite_counts
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_ratio = duplicate_rows / len(df)
            if duplicate_ratio > self.config["data_quality"]["max_duplicate_ratio"]:
                issues.append(ValidationIssue(
                    issue_type="high_duplicate_ratio",
                    severity=ValidationSeverity.WARNING,
                    description=f"High duplicate ratio: {duplicate_ratio:.2%} ({duplicate_rows} rows)",
                    details={"duplicate_count": int(duplicate_rows), "duplicate_ratio": float(duplicate_ratio)}
                ))
        
        metrics["duplicate_rows"] = int(duplicate_rows)
        
        # Check for constant columns (zero variance)
        constant_columns = []
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].nunique() <= 1:
                constant_columns.append(column)
        
        if constant_columns:
            issues.append(ValidationIssue(
                issue_type="constant_columns",
                severity=ValidationSeverity.WARNING,
                description=f"Constant columns found: {constant_columns}",
                affected_columns=constant_columns
            ))
        
        metrics["constant_columns"] = constant_columns
        
        return {
            "issues": issues,
            "metrics": metrics
        }


# Convenience functions for specific step validation
def validate_step1_file(file_path: str) -> FileValidationResult:
    """Validate file for step 1 (data collection)."""
    validator = ComprehensiveFileValidator()
    return validator.validate_file_format(file_path, expected_schema="klines", step_name="step1")


def validate_step1_5_file(file_path: str) -> FileValidationResult:
    """Validate file for step 1.5 (data conversion)."""
    validator = ComprehensiveFileValidator()
    return validator.validate_file_format(file_path, expected_schema="klines", step_name="step01_5")


def validate_step2_file(file_path: str) -> FileValidationResult:
    """Validate file for step 2 (feature engineering)."""
    validator = ComprehensiveFileValidator()
    return validator.validate_file_format(file_path, expected_schema="features", step_name="step2")


def validate_step4_file(file_path: str) -> FileValidationResult:
    """Validate file for step 4 (processing and labeling)."""
    validator = ComprehensiveFileValidator()
    return validator.validate_file_format(file_path, expected_schema="features", step_name="step4")


# Decorator for automatic validation
def validate_file_format(step_name: str, expected_schema: Optional[str] = None):
    """Decorator to automatically validate file format in pipeline steps."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Execute the function
            result = func(*args, **kwargs)
            
            # If result is a file path, validate it
            if isinstance(result, str) and os.path.exists(result):
                validator = ComprehensiveFileValidator()
                validation_result = validator.validate_file_format(
                    result, expected_schema=expected_schema, step_name=step_name
                )
                
                if not validation_result.is_valid:
                    # Log validation issues but don't fail the step
                    logging.warning(f"File validation issues in {step_name}: {result}")
                    for issue in validation_result.issues:
                        logging.warning(f"  - {issue.severity.value}: {issue.description}")
            
            return result
        
        return wrapper
    return decorator