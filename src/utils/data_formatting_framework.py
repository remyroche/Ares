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

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors


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
            "preserve_original": True
        }

        # Standard data formats
        self.standard_formats = {
            DataFormat.KLINES: {
                "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "optional_columns": ["quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"],
                "data_types": {
                    "timestamp": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64"
                },
                "column_order": ["timestamp", "open", "high", "low", "close", "volume"]
            },
            DataFormat.FEATURES: {
                "required_columns": ["timestamp"],
                "optional_columns": [],
                "data_types": {
                    "timestamp": "int64"
                },
                "column_order": ["timestamp"]
            },
            DataFormat.LABELS: {
                "required_columns": ["timestamp", "label"],
                "optional_columns": ["label_probability", "label_confidence"],
                "data_types": {
                    "timestamp": "int64",
                    "label": "int64",
                    "label_probability": "float64",
                    "label_confidence": "float64"
                },
                "column_order": ["timestamp", "label"]
            },
            DataFormat.PREDICTIONS: {
                "required_columns": ["timestamp", "prediction"],
                "optional_columns": ["prediction_probability", "prediction_confidence"],
                "data_types": {
                    "timestamp": "int64",
                    "prediction": "float64",
                    "prediction_probability": "float64",
                    "prediction_confidence": "float64"
                },
                "column_order": ["timestamp", "prediction"]
            }
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data formatting"
    )
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
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
        # Convert spaces and special characters to underscores
        text = re.sub(r'[^a-zA-Z0-9]', '_', text)
        # Convert to lowercase
        text = text.lower()
        # Remove multiple underscores
        text = re.sub(r'_+', '_', text)
        # Remove leading/trailing underscores
        text = text.strip('_')
        return text

    def _to_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        import re
        # Convert to snake_case first
        text = self._to_snake_case(text)
        # Convert to camelCase
        words = text.split('_')
        if len(words) > 1:
            return words[0] + ''.join(word.capitalize() for word in words[1:])
        return text

    def _standardize_data_types(self, data: pd.DataFrame, data_types: Dict[str, str]) -> pd.DataFrame:
        """Standardize data types according to specification."""
        for column, target_type in data_types.items():
            if column in data.columns:
                try:
                    if target_type == "int64":
                        data[column] = pd.to_numeric(data[column], errors='coerce').astype('int64')
                    elif target_type == "float64":
                        data[column] = pd.to_numeric(data[column], errors='coerce').astype('float64')
                    elif target_type == "string":
                        data[column] = data[column].astype(str)
                    elif target_type == "datetime":
                        data[column] = pd.to_datetime(data[column], errors='coerce')

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
                data[column] = pd.date_range(start=datetime.now(), periods=len(data), freq='1min').astype(np.int64) // 10**9
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

    def _log_formatting_operation(self, original_data: pd.DataFrame, formatted_data: pd.DataFrame,
                                target_format: DataFormat) -> None:
        """Log formatting operation."""
        operation = {
            "timestamp": datetime.now().isoformat(),
            "target_format": target_format.value,
            "original_shape": original_data.shape,
            "formatted_shape": formatted_data.shape,
            "columns_changed": list(set(original_data.columns) - set(formatted_data.columns) |
                                   set(formatted_data.columns) - set(original_data.columns))
        }

        self.format_history.append(operation)
        self.logger.info(f"Formatted data to {target_format.value}: {original_data.shape} -> {formatted_data.shape}")

    def validate_data_format(self, data: pd.DataFrame, expected_format: DataFormat) -> Dict[str, Any]:
        """Validate that data conforms to expected format.

        Args:
            data: Data to validate
            expected_format: Expected format

        Returns:
            Validation results
        """
        if expected_format not in self.standard_formats:
            return {"valid": False, "error": f"Unknown format: {expected_format}"}

        format_spec = self.standard_formats[expected_format]
        validation_results = {
            "valid": True,
            "format": expected_format.value,
            "issues": [],
            "warnings": []
        }

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
                    validation_results["warnings"].append(f"Column '{column}' has type {actual_type}, expected {expected_type}")

        # Check for missing values in required columns
        for column in format_spec["required_columns"]:
            if column in data.columns and data[column].isnull().any():
                missing_count = data[column].isnull().sum()
                validation_results["warnings"].append(f"Column '{column}' has {missing_count} missing values")

        return validation_results


# Global data formatting framework instance
data_formatting_framework = DataFormattingFramework()