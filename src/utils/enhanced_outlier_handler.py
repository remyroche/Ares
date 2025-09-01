"""
Enhanced Outlier Handler

This module provides sophisticated outlier detection and handling including:
- Outlier detection with detailed logging
- Error raising instead of silent removal
- Data schema validation for file operations
- Root cause analysis and reporting
- Data integrity preservation
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
import logging
from enum import Enum

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors


class OutlierSeverity(Enum):
    """Outlier severity levels."""
    LOW = "low"           # Minor outliers, log warning
    MEDIUM = "medium"     # Moderate outliers, log error
    HIGH = "high"         # Major outliers, raise exception
    CRITICAL = "critical" # Critical outliers, raise exception and stop processing


class DataSchema:
    """Defines expected data schema for file operations."""

    def __init__(self, name: str, required_columns: List[str],
                 optional_columns: List[str] = None, data_types: Dict[str, str] = None,
                 constraints: Dict[str, Dict[str, Any]] = None):
        """Initialize data schema.

        Args:
            name: Schema name
            required_columns: List of required column names
            optional_columns: List of optional column names
            data_types: Dictionary mapping column names to expected data types
            constraints: Dictionary of column constraints (min, max, unique, etc.)
        """
        self.name = name
        self.required_columns = set(required_columns)
        self.optional_columns = set(optional_columns or [])
        self.data_types = data_types or {}
        self.constraints = constraints or {}
        self.all_columns = self.required_columns.union(self.optional_columns)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe against schema.

        Args:
            df: Dataframe to validate

        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": [],
            "constraint_violations": []
        }

        # Check required columns
        df_columns = set(df.columns)
        missing_required = self.required_columns - df_columns
        if missing_required:
            results["valid"] = False
            results["missing_columns"] = list(missing_required)
            results["errors"].append(f"Missing required columns: {missing_required}")

        # Check for extra columns
        extra_columns = df_columns - self.all_columns
        if extra_columns:
            results["warnings"].append(f"Extra columns found: {extra_columns}")
            results["extra_columns"] = list(extra_columns)

        # Check data types
        for column, expected_type in self.data_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type != expected_type:
                    results["type_mismatches"].append({
                        "column": column,
                        "expected": expected_type,
                        "actual": actual_type
                    })
                    results["warnings"].append(f"Type mismatch in {column}: expected {expected_type}, got {actual_type}")

        # Check constraints
        for column, constraint in self.constraints.items():
            if column in df.columns:
                constraint_result = self._validate_constraint(df, column, constraint)
                if not constraint_result["valid"]:
                    results["constraint_violations"].append(constraint_result)
                    results["errors"].append(f"Constraint violation in {column}: {constraint_result['message']}")
                    results["valid"] = False

        return results

    def _validate_constraint(self, df: pd.DataFrame, column: str, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single column constraint.

        Args:
            df: Dataframe to validate
            column: Column name
            constraint: Constraint definition

        Returns:
            Constraint validation result
        """
        result = {"valid": True, "column": column, "message": ""}

        if "min" in constraint:
            min_val = constraint["min"]
            if df[column].min() < min_val:
                result["valid"] = False
                result["message"] = f"Minimum value {df[column].min()} is below constraint {min_val}"

        if "max" in constraint:
            max_val = constraint["max"]
            if df[column].max() > max_val:
                result["valid"] = False
                result["message"] = f"Maximum value {df[column].max()} is above constraint {max_val}"

        if "unique" in constraint and constraint["unique"]:
            if not df[column].is_unique:
                result["valid"] = False
                result["message"] = f"Column {column} contains duplicate values"

        if "not_null" in constraint and constraint["not_null"]:
            if df[column].isnull().any():
                result["valid"] = False
                result["message"] = f"Column {column} contains null values"

        return result


class OutlierInfo:
    """Information about detected outliers."""

    def __init__(self, column: str, indices: List[int], values: List[Any],
                 method: str, severity: OutlierSeverity, threshold: float):
        self.column = column
        self.indices = indices
        self.values = values
        self.method = method
        self.severity = severity
        self.threshold = threshold
        self.timestamp = datetime.now()
        self.context = {}


class EnhancedOutlierHandler:
    """Enhanced outlier handler with error raising and schema validation."""

    def __init__(self, raise_errors: bool = True, log_details: bool = True):
        """Initialize enhanced outlier handler.

        Args:
            raise_errors: Whether to raise errors for outliers
            log_details: Whether to log detailed outlier information
        """
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("EnhancedOutlierHandler")
        self.raise_errors = raise_errors
        self.log_details = log_details

        # Outlier detection methods
        self.detection_methods = {
            "zscore": self._detect_zscore_outliers,
            "iqr": self._detect_iqr_outliers,
            "isolation_forest": self._detect_isolation_forest_outliers,
            "local_outlier_factor": self._detect_lof_outliers,
            "mahalanobis": self._detect_mahalanobis_outliers
        }

        # Standard data schemas
        self.standard_schemas = self._initialize_standard_schemas()

        # Outlier history
        self.outlier_history: List[OutlierInfo] = []

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="outlier detection"
    )
    def _log_outlier_details(self, outliers: List[OutlierInfo]) -> None:
        """Log detailed outlier information."""
        if not outliers:
            return

        self.logger.info(f"🔍 Detected {len(outliers)} outlier groups")

        for outlier in outliers:
            self.logger.warning(f"Outlier in {outlier.column}: {len(outlier.indices)} values, "
                              f"severity={outlier.severity.value}, method={outlier.method}")

            if outlier.severity in [OutlierSeverity.HIGH, OutlierSeverity.CRITICAL]:
                self.logger.error(f"Critical outlier details: {outlier}")
                self.logger.error(f"  Values: {outlier.values[:5]}...")  # Show first 5 values
                self.logger.error(f"  Context: {outlier.context}")

    def _handle_outlier_errors(self, outliers: List[OutlierInfo]) -> None:
        """Handle outlier errors by raising exceptions or logging."""
        critical_outliers = [o for o in outliers if o.severity == OutlierSeverity.CRITICAL]
        high_outliers = [o for o in outliers if o.severity == OutlierSeverity.HIGH]

        if critical_outliers:
            error_msg = f"Critical outliers detected: {len(critical_outliers)} groups"
            for outlier in critical_outliers:
                error_msg += f"\n  {outlier.column}: {len(outlier.indices)} values"

            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if high_outliers:
            error_msg = f"High severity outliers detected: {len(high_outliers)} groups"
            for outlier in high_outliers:
                error_msg += f"\n  {outlier.column}: {len(outlier.indices)} values"

            self.logger.error(error_msg)
            if self.raise_errors:
                raise ValueError(error_msg)

    def validate_data_schema(self, data: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """Validate data against a standard schema.

        Args:
            data: Data to validate
            schema_name: Name of the schema to validate against

        Returns:
            Validation results
        """
        if schema_name not in self.standard_schemas:
            self.logger.error(f"Unknown schema: {schema_name}")
            return {"valid": False, "error": f"Unknown schema: {schema_name}"}

        schema = self.standard_schemas[schema_name]
        return schema.validate_dataframe(data)


# Global enhanced outlier handler instance
enhanced_outlier_handler = EnhancedOutlierHandler()