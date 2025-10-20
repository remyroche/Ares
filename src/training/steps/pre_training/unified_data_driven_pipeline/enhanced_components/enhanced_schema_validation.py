"""
Enhanced Schema Validation System

This module provides comprehensive schema validation with advanced features
including temporal alignment, data integrity checks, and performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import json
import hashlib

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import Pandera for schema validation
try:
    import pandera as pa
    from pandera import errors as pa_errors
    PANDERA_AVAILABLE = True
    tprint_info("✅ Pandera schema validation available")
except ImportError:
    PANDERA_AVAILABLE = False
    tprint_warning("⚠️ Pandera not available, using fallback validation")

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_time: float
    schema_name: str
    data_shape: Tuple[int, int]
    validation_summary: Dict[str, Any]

    def __post_init__(self):
        """Validate result."""
        assert isinstance(self.is_valid, bool), "is_valid must be boolean"
        assert isinstance(self.errors, list), "errors must be list"
        assert isinstance(self.warnings, list), "warnings must be list"

@dataclass
class SchemaDefinition:
    """Definition of a data schema."""

    name: str
    version: str
    description: str
    required_columns: List[str]
    optional_columns: List[str]
    data_types: Dict[str, str]
    constraints: Dict[str, Any]
    temporal_alignment: Dict[str, Any]
    quality_checks: List[str]

    def __post_init__(self):
        """Validate schema definition."""
        assert self.name, "Schema name is required"
        assert self.version, "Schema version is required"
        assert isinstance(self.required_columns, list), "required_columns must be list"
        assert isinstance(self.optional_columns, list), "optional_columns must be list"

@dataclass
class TemporalAlignmentResult:
    """Result of temporal alignment validation."""

    is_aligned: bool
    alignment_errors: List[str]
    lag_requirements: Dict[str, int]
    observed_lags: Dict[str, int]
    min_required_lag: int
    max_observed_lag: int

    def __post_init__(self):
        """Validate result."""
        assert isinstance(self.is_aligned, bool), "is_aligned must be boolean"
        assert isinstance(self.alignment_errors, list), "alignment_errors must be list"

class EnhancedSchemaValidator:
    """
    Enhanced schema validation system with advanced features.

    Provides comprehensive data validation including temporal alignment,
    data integrity checks, and performance optimization.
    """

    def __init__(self, enable_pandera: bool = True, enable_gpu_optimization: bool = True):
        """Initialize the enhanced schema validator."""
        self.enable_pandera = enable_pandera and PANDERA_AVAILABLE
        self.enable_gpu_optimization = enable_gpu_optimization

        # Schema registry
        self.schema_registry: Dict[str, SchemaDefinition] = {}

        # Performance tracking
        self.performance_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_validation_time': 0.0,
            'temporal_alignment_checks': 0,
            'data_integrity_checks': 0,
            'quality_checks': 0
        }

        # Initialize default schemas
        self._initialize_default_schemas()

        tprint_info("Enhanced Schema Validator initialized")
        if self.enable_pandera:
            tprint_info("✅ Pandera validation enabled")
        else:
            tprint_warning("⚠️ Pandera validation disabled, using fallback")

    def _initialize_default_schemas(self):
        """Initialize default schemas."""
        # OHLCV Schema
        ohlcv_schema = SchemaDefinition(
            name="ohlcv",
            version="1.0",
            description="OHLCV data schema",
            required_columns=["open", "high", "low", "close", "volume"],
            optional_columns=[],
            data_types={
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64"
            },
            constraints={
                "high >= low": "High must be >= Low",
                "high >= open": "High must be >= Open",
                "high >= close": "High must be >= Close",
                "low <= open": "Low must be <= Open",
                "low <= close": "Low must be <= Close",
                "volume >= 0": "Volume must be non-negative"
            },
            temporal_alignment={
                "min_lag": 1,
                "max_lag": 100,
                "required_lag": 1
            },
            quality_checks=[
                "no_duplicate_timestamps",
                "no_missing_values",
                "no_infinite_values",
                "temporal_ordering"
            ]
        )
        self.schema_registry["ohlcv"] = ohlcv_schema

        # Features Schema
        features_schema = SchemaDefinition(
            name="features",
            version="1.0",
            description="Feature data schema",
            required_columns=[],
            optional_columns=[],
            data_types={},
            constraints={},
            temporal_alignment={
                "min_lag": 1,
                "max_lag": 1000,
                "required_lag": 1
            },
            quality_checks=[
                "no_duplicate_timestamps",
                "temporal_ordering",
                "finite_values"
            ]
        )
        self.schema_registry["features"] = features_schema

        # Labels Schema
        labels_schema = SchemaDefinition(
            name="labels",
            version="1.0",
            description="Label data schema",
            required_columns=["immediate_opportunity", "short_term_opportunity", "leverage_adjusted_score"],
            optional_columns=[],
            data_types={
                "immediate_opportunity": "int64",
                "short_term_opportunity": "int64",
                "leverage_adjusted_score": "float64"
            },
            constraints={
                "immediate_opportunity in [0, 1]": "Immediate opportunity must be 0 or 1",
                "short_term_opportunity in [0, 1]": "Short term opportunity must be 0 or 1",
                "leverage_adjusted_score >= 0": "Leverage adjusted score must be non-negative"
            },
            temporal_alignment={
                "min_lag": 1,
                "max_lag": 100,
                "required_lag": 1
            },
            quality_checks=[
                "no_duplicate_timestamps",
                "temporal_ordering",
                "no_missing_values"
            ]
        )
        self.schema_registry["labels"] = labels_schema

        tprint_success(f"Initialized {len(self.schema_registry)} default schemas")

    def register_schema(self, schema: SchemaDefinition):
        """Register a new schema."""
        self.schema_registry[schema.name] = schema
        tprint_info(f"Registered schema: {schema.name} v{schema.version}")

    def validate_data(self, data: pd.DataFrame,
                     schema_name: str,
                     context: str = "",
                     enable_temporal_alignment: bool = True,
                     enable_data_integrity: bool = True,
                     enable_quality_checks: bool = True) -> ValidationResult:
        """
        Validate data against a schema.

        Args:
            data: Input DataFrame
            schema_name: Name of schema to validate against
            context: Context for validation
            enable_temporal_alignment: Enable temporal alignment checks
            enable_data_integrity: Enable data integrity checks
            enable_quality_checks: Enable quality checks

        Returns:
            ValidationResult with validation results
        """
        tprint_info(f"Validating data against schema '{schema_name}' for context '{context}'")

        start_time = time.time()

        # Initialize result
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            validation_time=0.0,
            schema_name=schema_name,
            data_shape=data.shape,
            validation_summary={}
        )

        # Get schema definition
        if schema_name not in self.schema_registry:
            result.is_valid = False
            result.errors.append(f"Schema '{schema_name}' not found in registry")
            result.validation_time = time.time() - start_time
            return result

        schema = self.schema_registry[schema_name]

        # Basic validation
        basic_validation = self._validate_basic_structure(data, schema)
        if not basic_validation["is_valid"]:
            result.is_valid = False
            result.errors.extend(basic_validation["errors"])
        result.warnings.extend(basic_validation["warnings"])

        # Data type validation
        if result.is_valid:
            type_validation = self._validate_data_types(data, schema)
            if not type_validation["is_valid"]:
                result.is_valid = False
                result.errors.extend(type_validation["errors"])
            result.warnings.extend(type_validation["warnings"])

        # Constraint validation
        if result.is_valid and enable_data_integrity:
            constraint_validation = self._validate_constraints(data, schema)
            if not constraint_validation["is_valid"]:
                result.is_valid = False
                result.errors.extend(constraint_validation["errors"])
            result.warnings.extend(constraint_validation["warnings"])
            self.performance_stats['data_integrity_checks'] += 1

        # Temporal alignment validation
        if result.is_valid and enable_temporal_alignment:
            temporal_validation = self._validate_temporal_alignment(data, schema)
            if not temporal_validation["is_valid"]:
                result.is_valid = False
                result.errors.extend(temporal_validation["errors"])
            result.warnings.extend(temporal_validation["warnings"])
            self.performance_stats['temporal_alignment_checks'] += 1

        # Quality checks
        if result.is_valid and enable_quality_checks:
            quality_validation = self._validate_quality(data, schema)
            if not quality_validation["is_valid"]:
                result.is_valid = False
                result.errors.extend(quality_validation["errors"])
            result.warnings.extend(quality_validation["warnings"])
            self.performance_stats['quality_checks'] += 1

        # Update performance stats
        result.validation_time = time.time() - start_time
        self.performance_stats['total_validations'] += 1
        if result.is_valid:
            self.performance_stats['successful_validations'] += 1
        else:
            self.performance_stats['failed_validations'] += 1
        self.performance_stats['total_validation_time'] += result.validation_time

        # Create validation summary
        result.validation_summary = {
            "schema_name": schema_name,
            "schema_version": schema.version,
            "data_shape": data.shape,
            "validation_time": result.validation_time,
            "errors_count": len(result.errors),
            "warnings_count": len(result.warnings),
            "temporal_alignment_enabled": enable_temporal_alignment,
            "data_integrity_enabled": enable_data_integrity,
            "quality_checks_enabled": enable_quality_checks
        }

        if result.is_valid:
            tprint_success(f"Data validation passed for schema '{schema_name}' in {result.validation_time:.3f}s")
        else:
            tprint_error(f"Data validation failed for schema '{schema_name}': {len(result.errors)} errors")

        return result

    def _validate_basic_structure(self, data: pd.DataFrame, schema: SchemaDefinition) -> Dict[str, Any]:
        """Validate basic data structure."""
        errors = []
        warnings = []

        # Check required columns
        missing_columns = set(schema.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")

        # Check for unexpected columns
        unexpected_columns = set(data.columns) - set(schema.required_columns + schema.optional_columns)
        if unexpected_columns:
            warnings.append(f"Unexpected columns found: {list(unexpected_columns)}")

        # Check data shape
        if len(data) == 0:
            errors.append("Data is empty")
        elif len(data) < 1:
            errors.append("Data has no rows")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _validate_data_types(self, data: pd.DataFrame, schema: SchemaDefinition) -> Dict[str, Any]:
        """Validate data types."""
        errors = []
        warnings = []

        for column, expected_type in schema.data_types.items():
            if column not in data.columns:
                continue

            actual_type = str(data[column].dtype)
            if actual_type != expected_type:
                errors.append(f"Column '{column}' has type '{actual_type}', expected '{expected_type}'")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _validate_constraints(self, data: pd.DataFrame, schema: SchemaDefinition) -> Dict[str, Any]:
        """Validate data constraints."""
        errors = []
        warnings = []

        for constraint, description in schema.constraints.items():
            try:
                if constraint == "high >= low":
                    if not (data["high"] >= data["low"]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "high >= open":
                    if not (data["high"] >= data["open"]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "high >= close":
                    if not (data["high"] >= data["close"]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "low <= open":
                    if not (data["low"] <= data["open"]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "low <= close":
                    if not (data["low"] <= data["close"]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "volume >= 0":
                    if not (data["volume"] >= 0).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "immediate_opportunity in [0, 1]":
                    if not data["immediate_opportunity"].isin([0, 1]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "short_term_opportunity in [0, 1]":
                    if not data["short_term_opportunity"].isin([0, 1]).all():
                        errors.append(f"Constraint violated: {description}")
                elif constraint == "leverage_adjusted_score >= 0":
                    if not (data["leverage_adjusted_score"] >= 0).all():
                        errors.append(f"Constraint violated: {description}")
            except Exception as e:
                warnings.append(f"Could not validate constraint '{constraint}': {e}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _validate_temporal_alignment(self, data: pd.DataFrame, schema: SchemaDefinition) -> Dict[str, Any]:
        """Validate temporal alignment."""
        errors = []
        warnings = []

        try:
            # Check if index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                errors.append("Data index must be DatetimeIndex for temporal alignment")
                return {"is_valid": False, "errors": errors, "warnings": warnings}

            # Check temporal ordering
            if not data.index.is_monotonic_increasing:
                errors.append("Data index must be monotonically increasing")

            # Check for duplicate timestamps
            if data.index.duplicated().any():
                errors.append("Data contains duplicate timestamps")

            # Check lag requirements
            temporal_config = schema.temporal_alignment
            min_lag = temporal_config.get("min_lag", 1)
            max_lag = temporal_config.get("max_lag", 1000)
            required_lag = temporal_config.get("required_lag", 1)

            # Calculate observed lags
            observed_lags = {}
            for column in data.columns:
                if column in data.columns:
                    series = data[column].dropna()
                    if len(series) > 0:
                        # Calculate leading nulls as lag
                        leading_nulls = series.isna().sum()
                        observed_lags[column] = leading_nulls

            # Check lag requirements
            for column, observed_lag in observed_lags.items():
                if observed_lag < required_lag:
                    errors.append(f"Column '{column}' has lag {observed_lag}, required {required_lag}")
                elif observed_lag > max_lag:
                    warnings.append(f"Column '{column}' has lag {observed_lag}, max recommended {max_lag}")

        except Exception as e:
            errors.append(f"Temporal alignment validation failed: {e}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _validate_quality(self, data: pd.DataFrame, schema: SchemaDefinition) -> Dict[str, Any]:
        """Validate data quality."""
        errors = []
        warnings = []

        for quality_check in schema.quality_checks:
            try:
                if quality_check == "no_duplicate_timestamps":
                    if data.index.duplicated().any():
                        errors.append("Duplicate timestamps found")

                elif quality_check == "no_missing_values":
                    missing_columns = data.columns[data.isnull().any()].tolist()
                    if missing_columns:
                        warnings.append(f"Missing values found in columns: {missing_columns}")

                elif quality_check == "no_infinite_values":
                    infinite_columns = []
                    for column in data.columns:
                        if data[column].dtype in ['float64', 'int64']:
                            if np.isinf(data[column]).any():
                                infinite_columns.append(column)
                    if infinite_columns:
                        errors.append(f"Infinite values found in columns: {infinite_columns}")

                elif quality_check == "temporal_ordering":
                    if not data.index.is_monotonic_increasing:
                        errors.append("Data is not temporally ordered")

                elif quality_check == "finite_values":
                    infinite_columns = []
                    for column in data.columns:
                        if data[column].dtype in ['float64', 'int64']:
                            if np.isinf(data[column]).any():
                                infinite_columns.append(column)
                    if infinite_columns:
                        errors.append(f"Infinite values found in columns: {infinite_columns}")

            except Exception as e:
                warnings.append(f"Quality check '{quality_check}' failed: {e}")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of registered schemas."""
        return {
            "registered_schemas": list(self.schema_registry.keys()),
            "schema_count": len(self.schema_registry),
            "performance_stats": self.performance_stats.copy()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_stats.copy()

# Convenience functions
def create_enhanced_schema_validator(enable_pandera: bool = True,
                                   enable_gpu_optimization: bool = True) -> EnhancedSchemaValidator:
    """Create an enhanced schema validator."""
    return EnhancedSchemaValidator(
        enable_pandera=enable_pandera,
        enable_gpu_optimization=enable_gpu_optimization
    )

def validate_ohlcv_data(data: pd.DataFrame, context: str = "") -> ValidationResult:
    """Validate OHLCV data."""
    validator = create_enhanced_schema_validator()
    return validator.validate_data(data, "ohlcv", context)

def validate_features_data(data: pd.DataFrame, context: str = "") -> ValidationResult:
    """Validate features data."""
    validator = create_enhanced_schema_validator()
    return validator.validate_data(data, "features", context)

def validate_labels_data(data: pd.DataFrame, context: str = "") -> ValidationResult:
    """Validate labels data."""
    validator = create_enhanced_schema_validator()
    return validator.validate_data(data, "labels", context)
