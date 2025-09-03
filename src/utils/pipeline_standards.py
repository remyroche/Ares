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

import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    details: dict[str, Any] | None = None
    column: str | None = None
    row_count: int | None = None


@dataclass
class ValidationResult:
    """Result of data validation."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStandards:
    """Centralized pipeline standards and utilities."""

    # Standard directory structure
    DIRECTORY_STRUCTURE = {
        "raw_data": "data_cache/{exchange}/{asset}",
        "unified_data": "data_cache/{exchange}/{asset}/unified",
        "processed_data": "data_cache/{exchange}/{asset}/processed",
        "reports": "data_cache/{exchange}/{asset}/reports",
        "backup": "data_cache/{exchange}/{asset}/backup",
        "temp": "data_cache/{exchange}/{asset}/temp",
    }

    # Standard file naming conventions
    FILE_NAMING = {
        "klines": "klines_{exchange}_{asset}_{timeframe}_consolidated.parquet",
        "aggtrades": "aggtrades_{exchange}_{asset}_consolidated.parquet",
        "futures": "futures_{exchange}_{asset}_consolidated.parquet",
        "unified": "unified_{exchange}_{asset}_{timeframe}.parquet",
        "unified_partitioned": "unified/{exchange}/{asset}/{timeframe}/year={year}/month={month:02d}/day={day:02d}/part-0.parquet",
        "validation_report": "validation_report_{exchange}_{asset}_{timeframe}_{timestamp}.json",
        "quality_report": "quality_report_{exchange}_{asset}_{timeframe}_{timestamp}.json",
    }

    # Standard data schemas
    SCHEMAS = {
        "klines": {
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "optional_columns": [
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ],
            "data_types": {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            },
        },
        "aggtrades": {
            "required_columns": ["timestamp", "price", "quantity"],
            "optional_columns": ["first_trade_id", "last_trade_id", "trade_time", "is_buyer_maker"],
            "data_types": {"timestamp": "int64", "price": "float64", "quantity": "float64", "is_buyer_maker": "bool"},
        },
        "futures": {
            "required_columns": ["timestamp", "fundingRate"],
            "optional_columns": ["symbol", "mark_price", "index_price", "next_funding_time"],
            "data_types": {"timestamp": "int64", "fundingRate": "float64"},
        },
        "unified": {
            "required_columns": [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "exchange",
                "symbol",
                "timeframe",
            ],
            "optional_columns": [
                "year",
                "month",
                "day",
                "trade_volume",
                "trade_count",
                "avg_price",
                "min_price",
                "max_price",
                "volume_ratio",
                "funding_rate",
            ],
            "data_types": {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
                "exchange": "string",
                "symbol": "string",
                "timeframe": "string",
                "year": "int16",
                "month": "int8",
                "day": "int8",
            },
        },
    }

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "min_rows": 100,
        "max_null_percentage": 0.1,  # 10%
        "max_duplicate_percentage": 0.05,  # 5%
        "min_quality_score": 0.8,
        "max_correlation": 0.95,
        "timestamp_consistency_threshold": 0.99,  # 99% of timestamps should be consistent
    }

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def safe_import(module_name: str, fallback_value: Any = None, logger: logging.Logger | None = None) -> Any:
        """
        Safely import a module with consistent fallback pattern.

        Args:
            module_name: Name of the module to import
            fallback_value: Value to return if import fails
            logger: Logger instance for warnings

        Returns:
            Imported module or fallback value
        """
        try:
            return __import__(module_name, fromlist=["*"])
        except ImportError as e:
            if logger:
                logger.warning(f"⚠️ Failed to import {module_name}: {e}. Using fallback.")
            return fallback_value

    @staticmethod
    def validate_environment_dependencies(
        required_modules: list[str], logger: logging.Logger | None = None,
    ) -> dict[str, bool]:
        """
        Validate that required dependencies are available.

        Args:
            required_modules: List of required module names
            logger: Logger instance for warnings

        Returns:
            Dictionary mapping module names to availability status
        """
        availability = {}
        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
                availability[module] = True
            except ImportError:
                availability[module] = False
                missing_modules.append(module)

        if missing_modules and logger:
            logger.warning(f"⚠️ Missing required modules: {missing_modules}")

        return availability

    @staticmethod
    def build_path(path_type: str, exchange: str, asset: str, **kwargs) -> str:
        """
        Build standardized path using the directory structure.

        Args:
            path_type: Type of path (raw_data, unified_data, etc.)
            exchange: Exchange name
            asset: Asset symbol
            **kwargs: Additional path parameters

        Returns:
            Standardized path
        """
        if path_type not in PipelineStandards.DIRECTORY_STRUCTURE:
            msg = f"Unknown path type: {path_type}"
            raise ValueError(msg)

        path_template = PipelineStandards.DIRECTORY_STRUCTURE[path_type]
        return path_template.format(exchange=exchange.lower(), asset=asset.lower(), **kwargs)

    @staticmethod
    def standardize_timestamp(
        df: pd.DataFrame, column: str = "timestamp", target_format: str = "int64",
    ) -> pd.DataFrame:
        """
        Standardize timestamp column to consistent format.

        Args:
            df: DataFrame to process
            column: Timestamp column name
            target_format: Target format ("int64" for milliseconds, "datetime64[ns]" for datetime)

        Returns:
            DataFrame with standardized timestamp
        """
        if column not in df.columns:
            return df

        df = df.copy()

        try:
            if target_format == "int64":
                # Convert to int64 milliseconds
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = (pd.to_datetime(df[column], utc=True).astype("int64") // 10**6).astype("int64")
                else:
                    ts_numeric = pd.to_numeric(df[column], errors="coerce")
                    if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
                        # Already in nanoseconds, convert to milliseconds
                        df[column] = (ts_numeric // 10**6).astype("int64")
                    else:
                        # Assume already in milliseconds
                        df[column] = ts_numeric.astype("int64")

            elif target_format == "datetime64[ns]":
                # Convert to datetime64[ns]
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column], utc=True)
                else:
                    ts_numeric = pd.to_numeric(df[column], errors="coerce")
                    if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
                        # Nanoseconds
                        df[column] = pd.to_datetime(ts_numeric, unit="ns", utc=True)
                    else:
                        # Milliseconds
                        df[column] = pd.to_datetime(ts_numeric, unit="ms", utc=True)

        except Exception as e:
            msg = f"Failed to standardize timestamp column '{column}': {e}"
            raise ValueError(msg)

        return df

    @staticmethod
    def validate_timestamp_format(
        df: pd.DataFrame, column: str = "timestamp", expected_format: str = "int64",
    ) -> ValidationResult:
        """
        Validate timestamp format consistency.

        Args:
            df: DataFrame to validate
            column: Timestamp column name
            expected_format: Expected format

        Returns:
            Validation result
        """
        result = ValidationResult(passed=True)

        if column not in df.columns:
            result.passed = False
            result.issues.append(
                ValidationIssue(severity=DataQualityLevel.CRITICAL, message=f"Timestamp column '{column}' not found"),
            )
            return result

        try:
            # Check for null values
            null_count = df[column].isnull().sum()
            if null_count > 0:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Found {null_count} null timestamps",
                        details={"null_count": null_count, "total_count": len(df)},
                    ),
                )

            # Check format consistency
            if expected_format == "int64":
                if not pd.api.types.is_integer_dtype(df[column]):
                    result.passed = False
                    result.issues.append(
                        ValidationIssue(
                            severity=DataQualityLevel.CRITICAL,
                            message=f"Timestamp column '{column}' is not integer type",
                            details={"actual_type": str(df[column].dtype)},
                        ),
                    )

                # Check for reasonable timestamp range (2000-2030)
                min_ts = df[column].min()
                max_ts = df[column].max()
                expected_min = pd.Timestamp("2000-01-01", tz="UTC").value // 10**6
                expected_max = pd.Timestamp("2030-01-01", tz="UTC").value // 10**6

                if min_ts < expected_min or max_ts > expected_max:
                    result.warnings.append(
                        ValidationIssue(
                            severity=DataQualityLevel.WARNING,
                            message="Timestamp range outside expected bounds",
                            details={
                                "min_ts": min_ts,
                                "max_ts": max_ts,
                                "expected_min": expected_min,
                                "expected_max": expected_max,
                            },
                        ),
                    )

            elif expected_format == "datetime64[ns]":
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    result.passed = False
                    result.issues.append(
                        ValidationIssue(
                            severity=DataQualityLevel.CRITICAL,
                            message=f"Timestamp column '{column}' is not datetime type",
                            details={"actual_type": str(df[column].dtype)},
                        ),
                    )

            # Check for duplicates
            duplicate_count = df[column].duplicated().sum()
            if duplicate_count > 0:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Found {duplicate_count} duplicate timestamps",
                        details={"duplicate_count": duplicate_count},
                    ),
                )

            # Check for monotonicity
            if not df[column].is_monotonic_increasing:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING, message="Timestamps are not monotonically increasing",
                    ),
                )

        except Exception as e:
            result.passed = False
            result.issues.append(
                ValidationIssue(severity=DataQualityLevel.CRITICAL, message=f"Error validating timestamp format: {e}"),
            )

        return result

    @staticmethod
    def validate_schema(df: pd.DataFrame, schema_name: str) -> ValidationResult:
        """
        Validate DataFrame against standard schema.

        Args:
            df: DataFrame to validate
            schema_name: Name of the schema to validate against

        Returns:
            Validation result
        """
        if schema_name not in PipelineStandards.SCHEMAS:
            msg = f"Unknown schema: {schema_name}"
            raise ValueError(msg)

        schema = PipelineStandards.SCHEMAS[schema_name]
        result = ValidationResult(passed=True)

        # Check required columns
        missing_required = [col for col in schema["required_columns"] if col not in df.columns]
        if missing_required:
            result.passed = False
            result.issues.append(
                ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Missing required columns: {missing_required}",
                    details={"missing_columns": missing_required},
                ),
            )

        # Check data types for existing columns
        for column, expected_type in schema["data_types"].items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type != expected_type:
                    result.warnings.append(
                        ValidationIssue(
                            severity=DataQualityLevel.WARNING,
                            message=f"Column '{column}' has type {actual_type}, expected {expected_type}",
                            column=column,
                            details={"actual_type": actual_type, "expected_type": expected_type},
                        ),
                    )

        # Check for null values in required columns
        for column in schema["required_columns"]:
            if column in df.columns:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    null_percentage = null_count / len(df)
                    if null_percentage > PipelineStandards.QUALITY_THRESHOLDS["max_null_percentage"]:
                        result.issues.append(
                            ValidationIssue(
                                severity=DataQualityLevel.CRITICAL,
                                message=f"Column '{column}' has too many null values",
                                column=column,
                                details={"null_count": null_count, "null_percentage": null_percentage},
                            ),
                        )
                    else:
                        result.warnings.append(
                            ValidationIssue(
                                severity=DataQualityLevel.WARNING,
                                message=f"Column '{column}' has {null_count} null values",
                                column=column,
                                details={"null_count": null_count, "null_percentage": null_percentage},
                            ),
                        )

        return result

    @staticmethod
    def enforce_schema(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
        """
        Enforce schema by converting data types and adding missing columns.

        Args:
            df: DataFrame to enforce schema on
            schema_name: Name of the schema to enforce

        Returns:
            DataFrame with enforced schema
        """
        if schema_name not in PipelineStandards.SCHEMAS:
            msg = f"Unknown schema: {schema_name}"
            raise ValueError(msg)

        schema = PipelineStandards.SCHEMAS[schema_name]
        df = df.copy()

        # Add missing optional columns with default values
        for column in schema["optional_columns"]:
            if column not in df.columns:
                if schema["data_types"][column] == "float64":
                    df[column] = 0.0
                elif schema["data_types"][column] == "int64":
                    df[column] = 0
                elif schema["data_types"][column] == "string":
                    df[column] = ""
                elif schema["data_types"][column] == "bool":
                    df[column] = False

        # Convert data types
        for column, expected_type in schema["data_types"].items():
            if column in df.columns:
                try:
                    if expected_type == "int64":
                        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int64")
                    elif expected_type == "float64":
                        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype("float64")
                    elif expected_type == "string":
                        df[column] = df[column].astype("string")
                    elif expected_type == "bool":
                        df[column] = df[column].astype("boolean")
                except Exception as e:
                    msg = f"Failed to convert column '{column}' to {expected_type}: {e}"
                    raise ValueError(msg)

        return df

    @staticmethod
    def validate_data_quality(
        df: pd.DataFrame, schema_name: str, quality_thresholds: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Comprehensive data quality validation.

        Args:
            df: DataFrame to validate
            schema_name: Schema name for validation
            quality_thresholds: Custom quality thresholds

        Returns:
            Validation result with quality score
        """
        thresholds = quality_thresholds or PipelineStandards.QUALITY_THRESHOLDS
        result = ValidationResult(passed=True)

        # Basic checks
        if df is None or df.empty:
            result.passed = False
            result.issues.append(
                ValidationIssue(severity=DataQualityLevel.CRITICAL, message="DataFrame is None or empty"),
            )
            return result

        # Check minimum rows
        if len(df) < thresholds["min_rows"]:
            result.passed = False
            result.issues.append(
                ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Too few rows: {len(df)} < {thresholds['min_rows']}",
                    details={"row_count": len(df), "min_required": thresholds["min_rows"]},
                ),
            )

        # Schema validation
        schema_result = PipelineStandards.validate_schema(df, schema_name)
        result.issues.extend(schema_result.issues)
        result.warnings.extend(schema_result.warnings)

        # Timestamp validation if present
        if "timestamp" in df.columns:
            ts_result = PipelineStandards.validate_timestamp_format(df, "timestamp")
            result.issues.extend(ts_result.issues)
            result.warnings.extend(ts_result.warnings)

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df) if len(df) > 0 else 0
        if duplicate_percentage > thresholds["max_duplicate_percentage"]:
            result.issues.append(
                ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Too many duplicate rows: {duplicate_percentage:.2%}",
                    details={"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_percentage},
                ),
            )

        # Check for infinite values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            infinite_count = np.isinf(df[column]).sum()
            if infinite_count > 0:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Column '{column}' has {infinite_count} infinite values",
                        column=column,
                        details={"infinite_count": infinite_count},
                    ),
                )

        # Calculate quality score
        total_checks = 5  # Basic checks, schema, timestamp, duplicates, infinite values
        passed_checks = total_checks - len([i for i in result.issues if i.severity == DataQualityLevel.CRITICAL])
        result.quality_score = passed_checks / total_checks

        # Determine if validation passed
        result.passed = result.quality_score >= thresholds["min_quality_score"] and not any(
            i.severity == DataQualityLevel.CRITICAL for i in result.issues
        )

        return result

    @staticmethod
    def generate_file_name(file_type: str, exchange: str, asset: str, timeframe: str = None, **kwargs) -> str:
        """
        Generate standardized file name.

        Args:
            file_type: Type of file (klines, aggtrades, etc.)
            exchange: Exchange name
            asset: Asset symbol
            timeframe: Timeframe (optional)
            **kwargs: Additional parameters

        Returns:
            Standardized file name
        """
        if file_type not in PipelineStandards.FILE_NAMING:
            msg = f"Unknown file type: {file_type}"
            raise ValueError(msg)

        template = PipelineStandards.FILE_NAMING[file_type]
        params = {
            "exchange": exchange.upper(),
            "asset": asset.upper(),
            "timeframe": timeframe or "1m",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            **kwargs,
        }

        return template.format(**params)

    @staticmethod
    def create_metadata(schema_name: str, exchange: str, asset: str, timeframe: str, **kwargs) -> dict[str, Any]:
        """
        Create standardized metadata for files.

        Args:
            schema_name: Schema name
            exchange: Exchange name
            asset: Asset symbol
            timeframe: Timeframe
            **kwargs: Additional metadata

        Returns:
            Standardized metadata dictionary
        """
        return {
            "schema_name": schema_name,
            "exchange": exchange.upper(),
            "asset": asset.upper(),
            "timeframe": timeframe,
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_version": "1.0.0",
            "data_format": "parquet",
            "compression": "snappy",
            **kwargs,
        }


    @staticmethod
    def validate_cross_step_consistency(
        data_dict: dict[str, pd.DataFrame], step_sequence: list[str],
    ) -> ValidationResult:
        """
        Validate data consistency across pipeline steps.

        Args:
            data_dict: Dictionary of dataframes from different steps
            step_sequence: Sequence of steps to validate

        Returns:
            ValidationResult with consistency issues
        """
        result = ValidationResult(passed=True)

        if len(data_dict) < 2:
            return result

        # Get the first dataframe as reference
        reference_df = None
        for step in step_sequence:
            if step in data_dict and data_dict[step] is not None:
                reference_df = data_dict[step]
                break

        if reference_df is None:
            result.issues.append(
                ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message="No reference dataframe found for consistency validation",
                ),
            )
            result.passed = False
            return result

        reference_length = len(reference_df)
        reference_columns = set(reference_df.columns)

        # Check each step's data consistency
        for step in step_sequence:
            if step not in data_dict or data_dict[step] is None:
                continue

            df = data_dict[step]

            # Check row count consistency
            if len(df) != reference_length:
                result.issues.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Row count mismatch in {step}: {len(df)} vs {reference_length}",
                        details={"step": step, "actual_count": len(df), "expected_count": reference_length},
                    ),
                )

            # Check for common columns
            common_columns = reference_columns.intersection(set(df.columns))
            if len(common_columns) < len(reference_columns) * 0.8:  # At least 80% common columns
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Low column overlap in {step}: {len(common_columns)}/{len(reference_columns)}",
                        details={
                            "step": step,
                            "common_columns": len(common_columns),
                            "total_columns": len(reference_columns),
                        },
                    ),
                )

        result.passed = len(result.issues) == 0
        return result

    @staticmethod
    def track_data_lineage(data: pd.DataFrame, source_step: str, transformations: list[str]) -> dict[str, Any]:
        """
        Track data lineage and transformations.

        Args:
            data: The dataframe
            source_step: Source step name
            transformations: List of transformations applied

        Returns:
            Lineage tracking information
        """
        return {
            "source_step": source_step,
            "transformations": transformations,
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "columns": list(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "dtypes": data.dtypes.to_dict(),
        }


    @staticmethod
    def calculate_comprehensive_quality_score(data: pd.DataFrame, context: str = "general") -> float:
        """
        Calculate comprehensive data quality score.

        Args:
            data: Dataframe to score
            context: Context for scoring (e.g., "klines", "features", "labels")

        Returns:
            Quality score between 0 and 1
        """
        if data is None or len(data) == 0:
            return 0.0

        scores = []

        # Completeness score
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        scores.append(completeness)

        # Consistency score (no duplicates)
        consistency = 1 - (data.duplicated().sum() / len(data))
        scores.append(consistency)

        # Validity score (no infinite values)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            infinite_ratio = np.isinf(data[numeric_cols]).sum().sum() / (len(data) * len(numeric_cols))
            validity = 1 - infinite_ratio
        else:
            validity = 1.0
        scores.append(validity)

        # Timeliness score (if timestamp column exists)
        if "timestamp" in data.columns:
            try:
                # Check if timestamps are in reasonable range
                timestamps = pd.to_datetime(data["timestamp"], unit="s")
                now = pd.Timestamp.now()
                time_diff = abs((timestamps - now).dt.total_seconds())
                timeliness = 1 - min(time_diff.mean() / (365 * 24 * 3600), 1.0)  # Normalize to 1 year
                scores.append(timeliness)
            except:
                scores.append(0.5)  # Default score if timestamp parsing fails

        # Context-specific scoring
        if context == "klines":
            # Additional checks for klines data
            required_cols = ["open", "high", "low", "close", "volume"]
            if all(col in data.columns for col in required_cols):
                # Check OHLC consistency
                ohlc_valid = (
                    (data["high"] >= data["low"])
                    & (data["high"] >= data["open"])
                    & (data["high"] >= data["close"])
                    & (data["low"] <= data["open"])
                    & (data["low"] <= data["close"])
                ).mean()
                scores.append(ohlc_valid)

        return np.mean(scores)

    @staticmethod
    def validate_feature_engineering_output(features: pd.DataFrame, original_data: pd.DataFrame) -> ValidationResult:
        """
        Validate feature engineering output.

        Args:
            features: Engineered features dataframe
            original_data: Original input data

        Returns:
            ValidationResult with validation issues
        """
        result = ValidationResult(passed=True)

        if features is None or original_data is None:
            result.issues.append(
                ValidationIssue(severity=DataQualityLevel.CRITICAL, message="Features or original data is None"),
            )
            result.passed = False
            return result

        # Check that features have same number of rows as original data
        if len(features) != len(original_data):
            result.issues.append(
                ValidationIssue(
                    severity=DataQualityLevel.CRITICAL,
                    message=f"Feature count mismatch: {len(features)} vs {len(original_data)}",
                    details={"feature_count": len(features), "original_count": len(original_data)},
                ),
            )
            result.passed = False

        # Check for NaN values in features
        nan_counts = features.isnull().sum()
        high_nan_cols = nan_counts[nan_counts > len(features) * 0.1]  # More than 10% NaN
        if len(high_nan_cols) > 0:
            result.warnings.append(
                ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message=f"Features with high NaN values: {list(high_nan_cols.index)}",
                    details={"high_nan_features": list(high_nan_cols.index)},
                ),
            )

        # Check for infinite values in features
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            infinite_counts = np.isinf(numeric_features).sum()
            infinite_cols = infinite_counts[infinite_counts > 0]
            if len(infinite_cols) > 0:
                result.warnings.append(
                    ValidationIssue(
                        severity=DataQualityLevel.WARNING,
                        message=f"Features with infinite values: {list(infinite_cols.index)}",
                        details={"infinite_features": list(infinite_cols.index)},
                    ),
                )

        # Check for constant features
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            result.warnings.append(
                ValidationIssue(
                    severity=DataQualityLevel.WARNING,
                    message=f"Constant features detected: {constant_features}",
                    details={"constant_features": constant_features},
                ),
            )

        # Calculate quality score
        result.quality_score = PipelineStandards.calculate_comprehensive_quality_score(features, "features")

        return result


# Global instance for easy access
pipeline_standards = PipelineStandards()
