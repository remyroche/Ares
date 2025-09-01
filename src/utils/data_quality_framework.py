"""
Comprehensive Data Quality Framework

This module provides standardized data quality management including:
- Data validation and schema enforcement
- Data formatting and standardization
- Quality scoring and metrics
- Data cleaning and preprocessing
- Quality gates and validation rules
- Data profiling and analysis
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from enum import Enum

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger
from .error_handler import handle_errors


class DataQualityLevel(Enum):
    """Data quality levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationRule:
    """Defines a validation rule for data quality."""

    def __init__(self, name: str, rule_type: str, parameters: Dict[str, Any],
                 severity: DataQualityLevel = DataQualityLevel.MEDIUM):
        self.name = name
        self.rule_type = rule_type
        self.parameters = parameters
        self.severity = severity
        self.created_at = datetime.now().isoformat()

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against this rule."""
        raise NotImplementedError("Subclasses must implement validate method")


class SchemaValidationRule(ValidationRule):
    """Validates data schema and structure."""

    def __init__(self, required_columns: List[str], optional_columns: List[str] = None,
                 data_types: Dict[str, str] = None, **kwargs):
        super().__init__("schema_validation", "schema", {
            "required_columns": required_columns,
            "optional_columns": optional_columns or [],
            "data_types": data_types or {}
        }, **kwargs)

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema."""
        issues = []
        warnings = []

        # Check required columns
        missing_columns = set(self.parameters["required_columns"]) - set(data.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        # Check data types
        for column, expected_type in self.parameters["data_types"].items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    warnings.append(f"Column '{column}' has type {actual_type}, expected {expected_type}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "severity": self.severity.value
        }


class RangeValidationRule(ValidationRule):
    """Validates data ranges and bounds."""

    def __init__(self, column: str, min_value: Optional[float] = None, max_value: Optional[float] = None,
                 allow_nan: bool = True, **kwargs):
        super().__init__("range_validation", "range", {
            "column": column,
            "min_value": min_value,
            "max_value": max_value,
            "allow_nan": allow_nan
        }, **kwargs)

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data ranges."""
        issues = []
        warnings = []

        if self.parameters["column"] not in data.columns:
            issues.append(f"Column '{self.parameters['column']}' not found")
            return {"passed": False, "issues": issues, "warnings": warnings, "severity": self.severity.value}

        column_data = data[self.parameters["column"]]

        # Check for NaN values
        if not self.parameters["allow_nan"] and column_data.isna().any():
            nan_count = column_data.isna().sum()
            issues.append(f"Column '{self.parameters['column']}' contains {nan_count} NaN values")

        # Check min value
        if self.parameters["min_value"] is not None:
            below_min = column_data < self.parameters["min_value"]
            if below_min.any():
                below_min_count = below_min.sum()
                issues.append(f"Column '{self.parameters['column']}' has {below_min_count} values below minimum {self.parameters['min_value']}")

        # Check max value
        if self.parameters["max_value"] is not None:
            above_max = column_data > self.parameters["max_value"]
            if above_max.any():
                above_max_count = above_max.sum()
                issues.append(f"Column '{self.parameters['column']}' has {above_max_count} values above maximum {self.parameters['max_value']}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "severity": self.severity.value
        }


class CompletenessValidationRule(ValidationRule):
    """Validates data completeness."""

    def __init__(self, columns: List[str], max_missing_ratio: float = 0.1, **kwargs):
        super().__init__("completeness_validation", "completeness", {
            "columns": columns,
            "max_missing_ratio": max_missing_ratio
        }, **kwargs)

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness."""
        issues = []
        warnings = []

        for column in self.parameters["columns"]:
            if column not in data.columns:
                issues.append(f"Column '{column}' not found")
                continue

            missing_ratio = data[column].isna().sum() / len(data)
            if missing_ratio > self.parameters["max_missing_ratio"]:
                issues.append(f"Column '{column}' has {missing_ratio:.2%} missing values (max: {self.parameters['max_missing_ratio']:.2%})")
            elif missing_ratio > 0:
                warnings.append(f"Column '{column}' has {missing_ratio:.2%} missing values")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "severity": self.severity.value
        }


class ConsistencyValidationRule(ValidationRule):
    """Validates data consistency."""

    def __init__(self, column: str, allowed_values: List[Any] = None,
                 pattern: str = None, case_sensitive: bool = True, **kwargs):
        super().__init__("consistency_validation", "consistency", {
            "column": column,
            "allowed_values": allowed_values,
            "pattern": pattern,
            "case_sensitive": case_sensitive
        }, **kwargs)

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data consistency."""
        issues = []
        warnings = []

        if self.parameters["column"] not in data.columns:
            issues.append(f"Column '{self.parameters['column']}' not found")
            return {"passed": False, "issues": issues, "warnings": warnings, "severity": self.severity.value}

        column_data = data[self.parameters["column"]]

        # Check allowed values
        if self.parameters["allowed_values"] is not None:
            invalid_values = column_data[~column_data.isin(self.parameters["allowed_values"])]
            if len(invalid_values) > 0:
                unique_invalid = invalid_values.unique()
                issues.append(f"Column '{self.parameters['column']}' contains invalid values: {unique_invalid}")

        # Check pattern
        if self.parameters["pattern"] is not None:
            import re
            flags = 0 if self.parameters["case_sensitive"] else re.IGNORECASE
            pattern = re.compile(self.parameters["pattern"], flags)

            non_matching = column_data[~column_data.astype(str).str.match(pattern, na=False)]
            if len(non_matching) > 0:
                issues.append(f"Column '{self.parameters['column']}' contains values not matching pattern: {self.parameters['pattern']}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "severity": self.severity.value
        }


class DataQualityFramework:
    """Comprehensive data quality management framework."""

    def __init__(self):
        """Initialize data quality framework."""
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("DataQuality")
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.quality_history: List[Dict[str, Any]] = []

        # Data quality policies
        self.quality_policies = {
            "strict_validation": True,
            "auto_clean": False,
            "quality_gates": True,
            "profiling_enabled": True,
            "max_issues_critical": 0,
            "max_issues_high": 5,
            "max_issues_medium": 20,
            "max_issues_low": 50
        }

        # Initialize standard validation rules
        self._initialize_standard_rules()

    def _initialize_standard_rules(self):
        """Initialize standard validation rules for common data types."""

        # Klines data validation
        klines_schema = SchemaValidationRule(
            required_columns=["timestamp", "open", "high", "low", "close", "volume"],
            optional_columns=["quote_asset_volume", "number_of_trades"],
            data_types={
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64"
            },
            severity=DataQualityLevel.CRITICAL
        )
        self.add_validation_rule("klines_schema", klines_schema)

        # OHLC consistency validation
        ohlc_consistency = ConsistencyValidationRule(
            column="high",
            allowed_values=None,
            pattern=None,
            severity=DataQualityLevel.HIGH
        )
        self.add_validation_rule("ohlc_consistency", ohlc_consistency)

        # Price range validation
        price_range = RangeValidationRule(
            column="close",
            min_value=0.0,
            max_value=None,
            allow_nan=False,
            severity=DataQualityLevel.HIGH
        )
        self.add_validation_rule("price_range", price_range)

        # Volume validation
        volume_validation = RangeValidationRule(
            column="volume",
            min_value=0.0,
            max_value=None,
            allow_nan=False,
            severity=DataQualityLevel.MEDIUM
        )
        self.add_validation_rule("volume_validation", volume_validation)

    def add_validation_rule(self, name: str, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            name: Rule name
            rule: Validation rule object
        """
        self.validation_rules[name] = rule
        self.logger.info(f"Added validation rule: {name}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data validation"
    )
    def validate_data(self, data: pd.DataFrame, rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate data against specified rules.

        Args:
            data: Data to validate
            rules: List of rule names to apply. If None, applies all rules.

        Returns:
            Validation results
        """
        if rules is None:
            rules = list(self.validation_rules.keys())

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "rules_applied": rules,
            "results": {},
            "summary": {
                "total_rules": len(rules),
                "passed_rules": 0,
                "failed_rules": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            }
        }

        for rule_name in rules:
            if rule_name not in self.validation_rules:
                self.logger.warning(f"Validation rule not found: {rule_name}")
                continue

            rule = self.validation_rules[rule_name]
            result = rule.validate(data)
            validation_results["results"][rule_name] = result

            # Update summary
            if result["passed"]:
                validation_results["summary"]["passed_rules"] += 1
            else:
                validation_results["summary"]["failed_rules"] += 1

            # Count issues by severity
            issue_count = len(result["issues"])
            if rule.severity == DataQualityLevel.CRITICAL:
                validation_results["summary"]["critical_issues"] += issue_count
            elif rule.severity == DataQualityLevel.HIGH:
                validation_results["summary"]["high_issues"] += issue_count
            elif rule.severity == DataQualityLevel.MEDIUM:
                validation_results["summary"]["medium_issues"] += issue_count
            elif rule.severity == DataQualityLevel.LOW:
                validation_results["summary"]["low_issues"] += issue_count

        # Determine overall validation status
        validation_results["overall_passed"] = self._evaluate_validation_status(validation_results["summary"])

        # Log validation results
        self._log_validation_results(validation_results)

        # Store in history
        self.quality_history.append(validation_results)

        return validation_results

    def _evaluate_validation_status(self, summary: Dict[str, Any]) -> bool:
        """Evaluate overall validation status based on quality policies."""
        if summary["critical_issues"] > self.quality_policies["max_issues_critical"]:
            return False
        if summary["high_issues"] > self.quality_policies["max_issues_high"]:
            return False
        if summary["medium_issues"] > self.quality_policies["max_issues_medium"]:
            return False
        if summary["low_issues"] > self.quality_policies["max_issues_low"]:
            return False
        return True

    def _log_validation_results(self, results: Dict[str, Any]) -> None:
        """Log validation results."""
        summary = results["summary"]

        if results["overall_passed"]:
            self.logger.info(f"Data validation passed: {summary['passed_rules']}/{summary['total_rules']} rules passed")
        else:
            self.logger.error(f"Data validation failed: {summary['failed_rules']}/{summary['total_rules']} rules failed")
            self.logger.error(f"Issues: Critical={summary['critical_issues']}, High={summary['high_issues']}, Medium={summary['medium_issues']}, Low={summary['low_issues']}")

    def format_data(self, data: pd.DataFrame, data_type: str = "klines") -> pd.DataFrame:
        """Format data according to standardized formats.

        Args:
            data: Data to format
            data_type: Type of data (klines, features, etc.)

        Returns:
            Formatted data
        """
        formatted_data = data.copy()

        if data_type == "klines":
            formatted_data = self._format_klines_data(formatted_data)
        elif data_type == "features":
            formatted_data = self._format_features_data(formatted_data)
        elif data_type == "labels":
            formatted_data = self._format_labels_data(formatted_data)
        else:
            self.logger.warning(f"Unknown data type for formatting: {data_type}")

        return formatted_data

    def _format_klines_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format klines data."""
        formatted = data.copy()

        # Ensure timestamp is int64
        if "timestamp" in formatted.columns:
            formatted["timestamp"] = pd.to_numeric(formatted["timestamp"], errors='coerce').astype('int64')

        # Ensure OHLCV columns are float64
        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_columns:
            if col in formatted.columns:
                formatted[col] = pd.to_numeric(formatted[col], errors='coerce').astype('float64')

        # Sort by timestamp
        if "timestamp" in formatted.columns:
            formatted = formatted.sort_values("timestamp").reset_index(drop=True)

        return formatted

    def _format_features_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format features data."""
        formatted = data.copy()

        # Ensure all numeric columns are float64
        numeric_columns = formatted.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            formatted[col] = pd.to_numeric(formatted[col], errors='coerce').astype('float64')

        # Handle infinite values
        formatted = formatted.replace([np.inf, -np.inf], np.nan)

        return formatted

    def _format_labels_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format labels data."""
        formatted = data.copy()

        # Ensure label columns are int64
        label_columns = [col for col in formatted.columns if "label" in col.lower()]
        for col in label_columns:
            formatted[col] = pd.to_numeric(formatted[col], errors='coerce').astype('int64')

        return formatted

    def clean_data(self, data: pd.DataFrame, cleaning_rules: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Clean data according to specified rules.

        Args:
            data: Data to clean
            cleaning_rules: Cleaning rules to apply

        Returns:
            Cleaned data
        """
        if not self.quality_policies["auto_clean"]:
            self.logger.info("Auto-cleaning disabled, returning original data")
            return data

        cleaned_data = data.copy()

        # Default cleaning rules
        default_rules = {
            "remove_duplicates": True,
            "handle_missing_values": True,
            "remove_outliers": False,
            "normalize_whitespace": True
        }

        if cleaning_rules:
            default_rules.update(cleaning_rules)

        # Remove duplicates
        if default_rules["remove_duplicates"]:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            if removed_duplicates > 0:
                self.logger.info(f"Removed {removed_duplicates} duplicate rows")

        # Handle missing values
        if default_rules["handle_missing_values"]:
            # For numeric columns, fill with median
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if cleaned_data[col].isna().any():
                    median_value = cleaned_data[col].median()
                    cleaned_data[col].fillna(median_value, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with median: {median_value}")

        # Enhanced outlier handling (if enabled)
        outlier_handling = default_rules.get("outlier_handling", "detect_only")
        if outlier_handling != "none":
            cleaned_data = self._handle_outliers_enhanced(cleaned_data, default_rules)

        # Normalize whitespace in string columns
        if default_rules["normalize_whitespace"]:
            string_columns = cleaned_data.select_dtypes(include=['object']).columns
            for col in string_columns:
                cleaned_data[col] = cleaned_data[col].astype(str).str.strip()

        return cleaned_data

    def _remove_outliers(self, data: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        cleaned_data = data.copy()

        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if method == "iqr":
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    cleaned_data = cleaned_data[~outliers]
                    self.logger.info(f"Removed {outlier_count} outliers from {col}")

        return cleaned_data

    def _handle_outliers_enhanced(self, data: pd.DataFrame, cleaning_rules: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers using enhanced outlier handler with error raising."""
        from .enhanced_outlier_handler import enhanced_outlier_handler, OutlierSeverity

        # Get outlier handling configuration
        outlier_config = cleaning_rules.get("outlier_config", {})
        method = outlier_config.get("method", "zscore")
        threshold = outlier_config.get("threshold", 3.0)
        severity_threshold = OutlierSeverity(outlier_config.get("severity_threshold", "medium"))
        raise_errors = outlier_config.get("raise_errors", True)

        # Detect outliers with enhanced handler
        outliers = enhanced_outlier_handler.detect_outliers(
            data=data,
            columns=outlier_config.get("columns"),
            method=method,
            threshold=threshold,
            severity_threshold=severity_threshold
        )

        # Log outlier detection results
        if outliers:
            self.logger.warning(f"Detected {len(outliers)} outlier groups")
            for outlier in outliers:
                self.logger.warning(f"  {outlier.column}: {len(outlier.indices)} values, severity={outlier.severity.value}")

        # Return original data (enhanced handler raises errors instead of removing)
        return data

    def profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile.

        Args:
            data: Data to profile

        Returns:
            Data profile
        """
        if not self.quality_policies["profiling_enabled"]:
            return {"profiling_disabled": True}

        profile = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum(),
            "columns": {},
            "summary": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "missing_values": data.isnull().sum().sum(),
                "duplicate_rows": data.duplicated().sum(),
                "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(data.select_dtypes(include=['object']).columns),
                "datetime_columns": len(data.select_dtypes(include=['datetime']).columns)
            }
        }

        # Profile each column
        for column in data.columns:
            col_data = data[column]
            col_profile = {
                "dtype": str(col_data.dtype),
                "missing_count": col_data.isnull().sum(),
                "missing_ratio": col_data.isnull().sum() / len(col_data),
                "unique_count": col_data.nunique(),
                "unique_ratio": col_data.nunique() / len(col_data)
            }

            # Numeric column statistics
            if pd.api.types.is_numeric_dtype(col_data):
                col_profile.update({
                    "min": float(col_data.min()) if not col_data.isna().all() else None,
                    "max": float(col_data.max()) if not col_data.isna().all() else None,
                    "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                    "median": float(col_data.median()) if not col_data.isna().all() else None,
                    "std": float(col_data.std()) if not col_data.isna().all() else None,
                    "zero_count": (col_data == 0).sum(),
                    "negative_count": (col_data < 0).sum(),
                    "infinite_count": np.isinf(col_data).sum()
                })

            # Categorical column statistics
            elif pd.api.types.is_object_dtype(col_data):
                value_counts = col_data.value_counts()
                col_profile.update({
                    "top_values": value_counts.head(5).to_dict(),
                    "empty_string_count": (col_data == "").sum(),
                    "whitespace_only_count": col_data.astype(str).str.strip().eq("").sum()
                })

            profile["columns"][column] = col_profile

        return profile

    def get_quality_report(self, data: pd.DataFrame, include_profile: bool = True) -> Dict[str, Any]:
        """Generate comprehensive data quality report.

        Args:
            data: Data to analyze
            include_profile: Whether to include data profiling

        Returns:
            Quality report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "validation_results": self.validate_data(data),
            "quality_score": self.calculate_quality_score(data)
        }

        if include_profile:
            report["data_profile"] = self.profile_data(data)

        # Add quality metrics
        report["quality_metrics"] = {
            "completeness": self._calculate_completeness_score(data),
            "consistency": self._calculate_consistency_score(data),
            "accuracy": self._calculate_accuracy_score(data),
            "timeliness": self._calculate_timeliness_score(data)
        }

        return report

    def calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score.

        Args:
            data: Data to score

        Returns:
            Quality score between 0 and 1
        """
        scores = []

        # Completeness score
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        scores.append(completeness)

        # Consistency score (no duplicates)
        consistency = 1 - (data.duplicated().sum() / len(data))
        scores.append(consistency)

        # Validity score (no infinite values in numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            infinite_ratio = np.isinf(data[numeric_cols]).sum().sum() / (len(data) * len(numeric_cols))
            validity = 1 - infinite_ratio
        else:
            validity = 1.0
        scores.append(validity)

        # Range validity score
        range_scores = []
        for col in numeric_cols:
            if col in ["open", "high", "low", "close", "volume"]:
                # Check for negative values in price/volume columns
                negative_ratio = (data[col] < 0).sum() / len(data)
                range_scores.append(1 - negative_ratio)

        if range_scores:
            scores.append(np.mean(range_scores))

        return np.mean(scores)

    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """Calculate completeness score."""
        return 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate consistency score."""
        return 1 - (data.duplicated().sum() / len(data))

    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate accuracy score."""
        # This is a simplified accuracy score
        # In practice, you would implement domain-specific accuracy checks
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 1.0

        # Check for reasonable ranges
        accuracy_scores = []
        for col in numeric_cols:
            if col in ["open", "high", "low", "close"]:
                # Check OHLC consistency
                if all(c in data.columns for c in ["open", "high", "low", "close"]):
                    ohlc_valid = ((data["high"] >= data["low"]) &
                                  (data["high"] >= data["open"]) &
                                  (data["high"] >= data["close"]) &
                                  (data["low"] <= data["open"]) &
                                  (data["low"] <= data["close"])).mean()
                    accuracy_scores.append(ohlc_valid)

        return np.mean(accuracy_scores) if accuracy_scores else 1.0

    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score."""
        if "timestamp" not in data.columns:
            return 1.0

        try:
            # Check if timestamps are in reasonable range
            timestamps = pd.to_datetime(data["timestamp"], unit='s')
            now = pd.Timestamp.now()
            time_diff = abs((timestamps - now).dt.total_seconds())
            timeliness = 1 - min(time_diff.mean() / (365 * 24 * 3600), 1.0)  # Normalize to 1 year
            return timeliness
        except:
            return 0.5  # Default score if timestamp parsing fails


# Global data quality framework instance
data_quality_framework = DataQualityFramework()