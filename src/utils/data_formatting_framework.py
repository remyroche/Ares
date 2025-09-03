"""
Data Formatting and Standardization Framework

This module provides standardized data formatting including:
- Data type standardization
- Column naming conventions
- Data structure normalization
- Format validation and enforcement
- Cross-step format consistency
- Format transformation utilities
"""
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from .logger import system_logger
from .pipeline_standards import PipelineStandards, pipeline_standards


class DataFormat(Enum):
    """Standard data formats."""

    KLINES = "klines"
    FEATURES = "features"
    LABELS = "labels"
    PREDICTIONS = "predictions"
    METADATA = "metadata"
    CONFIG = "config"


class ColumnNamingConvention(Enum):
    """Column naming conventions."""

    SNAKE_CASE = "snake_case"
    CAMEL_CASE = "camel_case"
    UPPER_CASE = "upper_case"
    LOWER_CASE = "lower_case"


class DataFormattingFramework:
    """Comprehensive data formatting and standardization framework."""

    def __init__(self):
        """Initialize data formatting framework."""
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("DataFormatting")
        self.format_history: List[Dict[str, Any]] = []

        # Formatting policies
        self.formatting_policies = {
            "column_naming_convention": ColumnNamingConvention.SNAKE_CASE,
            "timestamp_format": "unix_seconds",
            "numeric_precision": 8,
            "auto_rename_columns": True,
            "strict_formatting": True,
            "preserve_original": True,
        }

        # Standard data formats
        self.standard_formats = {
            DataFormat.KLINES: {
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
                "column_order": ["timestamp", "open", "high", "low", "close", "volume"],
            },
            DataFormat.FEATURES: {
                "required_columns": ["timestamp"],
                "optional_columns": [],
                "data_types": {"timestamp": "int64"},
                "column_order": ["timestamp"],
            },
            DataFormat.LABELS: {
                "required_columns": ["timestamp", "label"],
                "optional_columns": ["label_probability", "label_confidence"],
                "data_types": {
                    "timestamp": "int64",
                    "label": "int64",
                    "label_probability": "float64",
                    "label_confidence": "float64",
                },
                "column_order": ["timestamp", "label"],
            },
            DataFormat.PREDICTIONS: {
                "required_columns": ["timestamp", "prediction"],
                "optional_columns": ["prediction_probability", "prediction_confidence"],
                "data_types": {
                    "timestamp": "int64",
                    "prediction": "float64",
                    "prediction_probability": "float64",
                    "prediction_confidence": "float64",
                },
                "column_order": ["timestamp", "prediction"],
            },
        }

    @handles_errors(Exception, fallback=None)
    def standardize_format(
        self, data: pd.DataFrame, target_format: DataFormat, preserve_original: bool = None
    ) -> pd.DataFrame:
        """Standardize data to a specific format."

        Args:
            data: Data to standardize
            target_format: Target format to standardize to
            preserve_original: Whether to preserve original data

        Returns:
            Standardized data
        """
        if preserve_original is None:
            preserve_original = self.formatting_policies["preserve_original"]

        if preserve_original:
            standardized_data = data.copy()
        else:
            standardized_data = data

        # Get format specification
        if target_format not in self.standard_formats:
            raise ValueError(f"Unknown target format: {target_format}")

        format_spec = self.standard_formats[target_format]

        # Standardize column names
        if self.formatting_policies["auto_rename_columns"]:
            standardized_data = self._standardize_column_names(standardized_data)

        # Standardize data types
        standardized_data = self._standardize_data_types(standardized_data, format_spec["data_types"])

        # Ensure required columns exist
        standardized_data = self._ensure_required_columns(standardized_data, format_spec["required_columns"])

        # Reorder columns
        standardized_data = self._reorder_columns(standardized_data, format_spec["column_order"])

        # Validate format
        if self.formatting_policies["strict_formatting"]:
            self._validate_format(standardized_data, target_format)

        # Log formatting operation
        self._log_formatting_operation(data, standardized_data, target_format)

        return standardized_data

    def _standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names according to naming convention."""
        convention = self.formatting_policies["column_naming_convention"]

        new_columns = {}
        for col in data.columns:
            if convention == ColumnNamingConvention.SNAKE_CASE:
                new_name = self._to_snake_case(col)
            elif convention == ColumnNamingConvention.CAMEL_CASE:
                new_name = self._to_camel_case(col)
            elif convention == ColumnNamingConvention.UPPER_CASE:
                new_name = col.upper()
            elif convention == ColumnNamingConvention.LOWER_CASE:
                new_name = col.lower()
            else:
                new_name = col

            new_columns[col] = new_name

        # Rename columns
        data = data.rename(columns=new_columns)

        return data

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        import re

        # Convert camelCase to snake_case
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
        # Convert spaces and special characters to underscores
        text = re.sub(r"[^a-zA-Z0-9]", "_", text)
        # Convert to lowercase
        text = text.lower()
        # Remove multiple underscores
        text = re.sub(r"_+", "_", text)
        # Remove leading/trailing underscores
        text = text.strip("_")
        return text

    def _to_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        import re

        # Convert to snake_case first
        text = self._to_snake_case(text)
        # Convert to camelCase
        words = text.split("_")
        if len(words) > 1:
            return words[0] + "".join(word.capitalize() for word in words[1:])
        return text

    def _standardize_data_types(self, data: pd.DataFrame, data_types: Dict[str, str]) -> pd.DataFrame:
        """Standardize data types according to specification."""
        for column, target_type in data_types.items():
            if column in data.columns:
                try:
                    if target_type == "int64":
                        data[column] = pd.to_numeric(data[column], errors="coerce").astype("int64")
                    elif target_type == "float64":
                        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
                    elif target_type == "string":
                        data[column] = data[column].astype(str)
                    elif target_type == "datetime":
                        data[column] = pd.to_datetime(data[column], errors="coerce")

                    self.logger.debug(f"Standardized column '{column}' to type '{target_type}'")
                except Exception as e:
                    self.logger.warning(f"Failed to standardize column '{column}' to type '{target_type}': {e}")

        return data

    def _ensure_required_columns(self, data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Ensure all required columns exist."""
        missing_columns = set(required_columns) - set(data.columns)

        for column in missing_columns:
            if column == "timestamp":
                # Create timestamp column if missing
                data[column] = (
                    pd.date_range(start=datetime.now(), periods=len(data), freq="1min").astype(np.int64) // 10**9
                )
            else:
                # Create column with default value
                data[column] = 0.0
                self.logger.warning(f"Created missing required column '{column}' with default value")

        return data

    def _reorder_columns(self, data: pd.DataFrame, column_order: List[str]) -> pd.DataFrame:
        """Reorder columns according to specification."""
        # Get columns that exist in the data
        existing_ordered_columns = [col for col in column_order if col in data.columns]

        # Get remaining columns
        remaining_columns = [col for col in data.columns if col not in existing_ordered_columns]

        # Reorder columns
        final_column_order = existing_ordered_columns + remaining_columns

        return data[final_column_order]

    def _validate_format(self, data: pd.DataFrame, target_format: DataFormat) -> None:
        """Validate that data conforms to the target format."""
        format_spec = self.standard_formats[target_format]

        # Check required columns
        missing_columns = set(format_spec["required_columns"]) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns for format {target_format}: {missing_columns}")

        # Check data types
        for column, expected_type in format_spec["data_types"].items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    self.logger.warning(f"Column '{column}' has type {actual_type}, expected {expected_type}")

    def _log_formatting_operation(
        self, original_data: pd.DataFrame, formatted_data: pd.DataFrame, target_format: DataFormat
    ) -> None:
        """Log formatting operation."""
        operation = {
            "timestamp": datetime.now().isoformat(),
            "target_format": target_format.value,
            "original_shape": original_data.shape,
            "formatted_shape": formatted_data.shape,
            "columns_changed": list(
                set(original_data.columns) - set(formatted_data.columns)
                | set(formatted_data.columns) - set(original_data.columns)
            ),
        }

        self.format_history.append(operation)
        self.logger.info(f"Formatted data to {target_format.value}: {original_data.shape} -> {formatted_data.shape}")

    def normalize_timestamps(
        self, data: pd.DataFrame, timestamp_column: str = "timestamp", target_format: str = "unix_seconds"
    ) -> pd.DataFrame:
        """Normalize timestamps to a standard format."

        Args:
            data: Data containing timestamps
            timestamp_column: Name of timestamp column
            target_format: Target timestamp format

        Returns:
            Data with normalized timestamps
        """
        if timestamp_column not in data.columns:
            self.logger.warning(f"Timestamp column '{timestamp_column}' not found")
            return data

        normalized_data = data.copy()

        try:
            # Convert to datetime first
            timestamps = pd.to_datetime(normalized_data[timestamp_column], unit="s", errors="coerce")

            if target_format == "unix_seconds":
                normalized_data[timestamp_column] = timestamps.astype(np.int64) // 10**9
            elif target_format == "unix_milliseconds":
                normalized_data[timestamp_column] = timestamps.astype(np.int64) // 10**6
            elif target_format == "iso_string":
                normalized_data[timestamp_column] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S")
            elif target_format == "datetime":
                normalized_data[timestamp_column] = timestamps
            else:
                self.logger.warning(f"Unknown timestamp format: {target_format}")
                return data

            self.logger.info(f"Normalized timestamps to format: {target_format}")

        except Exception as e:
            self.logger.error(f"Failed to normalize timestamps: {e}")

        return normalized_data

    def round_numeric_columns(self, data: pd.DataFrame, precision: int = None) -> pd.DataFrame:
        """Round numeric columns to specified precision."

        Args:
            data: Data to round
            precision: Number of decimal places

        Returns:
            Data with rounded numeric columns
        """
        if precision is None:
            precision = self.formatting_policies["numeric_precision"]

        rounded_data = data.copy()

        # Round numeric columns
        numeric_columns = rounded_data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            rounded_data[column] = rounded_data[column].round(precision)

        self.logger.info(f"Rounded {len(numeric_columns)} numeric columns to {precision} decimal places")

        return rounded_data

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = "intelligent",
        limit: int = None,
        symbol: str = None,
        exchange: str = None,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Handle missing values according to specified strategy."

        Args:
            data: Data with missing values
            strategy: Strategy for handling missing values
            limit: Limit for forward/backward fill
            symbol: Trading symbol for data download (for intelligent strategy)
            exchange: Exchange name for data download (for intelligent strategy)
            timeframe: Timeframe for data download (for intelligent strategy)

        Returns:
            Data with handled missing values
        """
        if strategy == "intelligent":
            # Use enhanced missing value handler for intelligent gap filling
            from .enhanced_missing_value_handler import enhanced_missing_value_handler
            
            return enhanced_missing_value_handler.handle_missing_values_intelligently(
                data, "timestamp", symbol, exchange, timeframe
            )

        # Fallback to traditional strategies
        handled_data = data.copy()

        if strategy == "forward_fill":
            handled_data = handled_data.fillna(method="ffill", limit=limit)
        elif strategy == "backward_fill":
            handled_data = handled_data.fillna(method="bfill", limit=limit)
        elif strategy == "interpolate":
            handled_data = handled_data.interpolate(method="linear", limit=limit)
        elif strategy == "drop":
            handled_data = handled_data.dropna()
        elif strategy == "zero":
            handled_data = handled_data.fillna(0)
        elif strategy == "median":
            for column in handled_data.columns:
                if handled_data[column].dtype in ["float64", "int64"]:
                    median_value = handled_data[column].median()
                    handled_data[column].fillna(median_value, inplace=True)
        else:
            self.logger.warning(f"Unknown missing value strategy: {strategy}")
            return data

        missing_before = data.isnull().sum().sum()
        missing_after = handled_data.isnull().sum().sum()

        self.logger.info(f"Handled missing values using '{strategy}': {missing_before} -> {missing_after}")

        return handled_data

    def validate_data_format(self, data: pd.DataFrame, expected_format: DataFormat) -> Dict[str, Any]:
        """Validate that data conforms to expected format."

        Args:
            data: Data to validate
            expected_format: Expected format

        Returns:
            Validation results
        """
        if expected_format not in self.standard_formats:
            return {"valid": False, "error": f"Unknown format: {expected_format}"}

        format_spec = self.standard_formats[expected_format]
        validation_results = {"valid": True, "format": expected_format.value, "issues": [], "warnings": []}

        # Check required columns
        missing_columns = set(format_spec["required_columns"]) - set(data.columns)
        if missing_columns:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Missing required columns: {missing_columns}")

        # Check data types
        for column, expected_type in format_spec["data_types"].items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    validation_results["warnings"].append(
                        f"Column '{column}' has type {actual_type}, expected {expected_type}"
                    )

        # Check for missing values in required columns
        for column in format_spec["required_columns"]:
            if column in data.columns and data[column].isnull().any():
                missing_count = data[column].isnull().sum()
                validation_results["warnings"].append(f"Column '{column}' has {missing_count} missing values")

        return validation_results

    def get_format_specification(self, data_format: DataFormat) -> Dict[str, Any]:
        """Get format specification for a data format."

        Args:
            data_format: Data format to get specification for

        Returns:
            Format specification
        """
        if data_format not in self.standard_formats:
            raise ValueError(f"Unknown data format: {data_format}")

        return self.standard_formats[data_format].copy()

    def list_available_formats(self) -> List[str]:
        """List all available data formats."

        Returns:
            List of available formats
        """
        return [format.value for format in self.standard_formats.keys()]

    def add_custom_format(self, format_name: str, format_spec: Dict[str, Any]) -> None:
        """Add a custom data format."

        Args:
            format_name: Name of the custom format
            format_spec: Format specification
        """
        # Validate format specification
        required_keys = ["required_columns", "data_types", "column_order"]
        missing_keys = set(required_keys) - set(format_spec.keys())

        if missing_keys:
            raise ValueError(f"Missing required keys in format specification: {missing_keys}")

        # Add custom format
        self.standard_formats[DataFormat(format_name)] = format_spec
        self.logger.info(f"Added custom format: {format_name}")

    def get_formatting_report(self, data: pd.DataFrame, target_format: DataFormat) -> Dict[str, Any]:
        """Generate formatting report for data."

        Args:
            data: Data to analyze
            target_format: Target format

        Returns:
            Formatting report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "target_format": target_format.value,
            "current_validation": self.validate_data_format(data, target_format),
            "formatting_operations": self.format_history[-10:] if self.format_history else [],
        }

        # Add format comparison
        if target_format in self.standard_formats:
            format_spec = self.standard_formats[target_format]
            report["format_comparison"] = {
                "required_columns": format_spec["required_columns"],
                "current_columns": list(data.columns),
                "missing_columns": list(set(format_spec["required_columns"]) - set(data.columns)),
                "extra_columns": list(set(data.columns) - set(format_spec["required_columns"])),
            }

        return report


# Global data formatting framework instance
data_formatting_framework = DataFormattingFramework()
