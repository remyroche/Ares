"""Enhanced Outlier Handler.

This module provides sophisticated outlier detection and handling including:
- Outlier detection with detailed logging
- Error raising instead of silent removal
- Data schema validation for file operations
- Root cause analysis and reporting
- Data integrity preservation
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from .logger import system_logger
from .pipeline_standards import PipelineStandards, pipeline_standards

class OutlierSeverity(Enum):
    """Outlier severity levels."""

    LOW = "low"  # Minor outliers, log warning
    MEDIUM = "medium"  # Moderate outliers, log error
    HIGH = "high"  # Major outliers, raise exception
    CRITICAL = "critical"  # Critical outliers, raise exception and stop processing

class DataSchema:
    """Defines expected data schema for file operations."""

    def __init__(
        self,
        name: str,
        required_columns: List[str],
        optional_columns: List[str] = None,
        data_types: Dict[str, str] = None,
        constraints: Dict[str, Dict[str, Any]] = None,
    ):
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
            "constraint_violations": [],
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
                    results["type_mismatches"].append(
                        {
                            "column": column,
                            "expected": expected_type,
                            "actual": actual_type,
                        }
                    )
                    results["warnings"].append(
                        f"Type mismatch in {column}: expected {expected_type}, got {actual_type}"
                    )

        # Check constraints
        for column, constraint in self.constraints.items():
            if column in df.columns:
                if "not_null" in constraint and constraint["not_null"]:
                    if df[column].isnull().any():
                        results["constraint_violations"].append(
                            f"Column {column} contains null values"
                        )
                        results["warnings"].append(f"Null values found in {column}")

                if "unique" in constraint and constraint["unique"]:
                    if df[column].duplicated().any():
                        results["constraint_violations"].append(
                            f"Column {column} contains duplicate values"
                        )
                        results["warnings"].append(
                            f"Duplicate values found in {column}"
                        )

                if "min" in constraint:
                    min_val = constraint["min"]
                    if (df[column] < min_val).any():
                        results["constraint_violations"].append(
                            f"Column {column} contains values below minimum {min_val}"
                        )
                        results["warnings"].append(
                            f"Values below minimum {min_val} found in {column}"
                        )

                if "max" in constraint:
                    max_val = constraint["max"]
                    if (df[column] > max_val).any():
                        results["constraint_violations"].append(
                            f"Column {column} contains values above maximum {max_val}"
                        )
                        results["warnings"].append(
                            f"Values above maximum {max_val} found in {column}"
                        )

        return results

class OutlierInfo:
    """Information about detected outliers."""

    def __init__(
        self,
        column: str,
        indices: List[int],
        values: List[Any],
        method: str,
        severity: OutlierSeverity,
        threshold: float,
    ):
        """Initialize outlier information.

        Args:
            column: Column name where outliers were detected
            indices: Row indices of outliers
            values: Actual outlier values
            method: Detection method used
            severity: Severity level of outliers
            threshold: Threshold used for detection
        """
        self.column = column
        self.indices = indices
        self.values = values
        self.method = method
        self.severity = severity
        self.threshold = threshold
        self.timestamp = datetime.now()
        self.context = {}

    def __str__(self):
        return f"OutlierInfo(column={self.column}, count={len(self.indices)}, severity={self.severity.value}, method={self.method})"

    def __repr__(self):
        return self.__str__()

class EnhancedOutlierHandler:
    """Enhanced outlier detection and handling with multiple methods and severity
    classification."""

    def __init__(self, raise_errors: bool = True, log_details: bool = True):
        """Initialize enhanced outlier handler.

        Args:
            raise_errors: Whether to raise exceptions for critical/high outliers
            log_details: Whether to log detailed outlier information
        """
        self.raise_errors = raise_errors
        self.log_details = log_details
        self.logger = system_logger.getChild("EnhancedOutlierHandler")
        self.outlier_history = []

        # Standard schemas for common data types
        self.standard_schemas = {
            "klines": DataSchema(
                name="klines",
                required_columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
                data_types={
                    "timestamp": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                },
                constraints={
                    "open": {"min": 0, "not_null": True},
                    "high": {"min": 0, "not_null": True},
                    "low": {"min": 0, "not_null": True},
                    "close": {"min": 0, "not_null": True},
                    "volume": {"min": 0, "not_null": True},
                },
            ),
            "features": DataSchema(
                name="features",
                required_columns=["timestamp"],
                optional_columns=[],  # Features can vary
                data_types={"timestamp": "int64"},
                constraints={"timestamp": {"not_null": True}},
            ),
            "labels": DataSchema(
                name="labels",
                required_columns=["timestamp", "label"],
                data_types={"timestamp": "int64", "label": "object"},
                constraints={
                    "timestamp": {"not_null": True},
                    "label": {"not_null": True},
                },
            ),
        }

        # Available detection methods
        self.detection_methods = {
            "zscore": self._detect_zscore_outliers,
            "iqr": self._detect_iqr_outliers,
            "isolation_forest": self._detect_isolation_forest_outliers,
            "local_outlier_factor": self._detect_lof_outliers,
            "mahalanobis": self._detect_mahalanobis_outliers,
        }

        self.logger.info(
            f"🔍 Enhanced Outlier Handler initialized with {len(self.detection_methods)} detection methods"
        )

    @handle_errors(
        exceptions=(Exception,), default_return=[], context="outlier_detection"
    )
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0,
        columns: List[str] = None,
        raise_errors: bool = None,
    ) -> List[OutlierInfo]:
        """Detect outliers in data using specified method.

        Args:
            data: Data to analyze
            method: Detection method to use
            threshold: Threshold for outlier detection
            columns: Specific columns to analyze (None for all numeric)
            raise_errors: Override default error raising behavior

        Returns:
            List of OutlierInfo objects
        """
        if raise_errors is None:
            raise_errors = self.raise_errors

        if method not in self.detection_methods:
            self.logger.error(f"Unknown detection method: {method}")
            return []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        all_outliers = []

        for column in columns:
            if column not in data.columns:
                self.logger.warning(f"Column {column} not found in data")
                continue

            if not np.issubdtype(data[column].dtype, np.number):
                self.logger.warning(f"Column {column} is not numeric, skipping")
                continue

            # Remove null values for analysis
            clean_data = data[column].dropna()
            if len(clean_data) == 0:
                self.logger.warning(f"Column {column} has no valid data")
                continue

            # Detect outliers using specified method
            outliers = self.detection_methods[method](data, column, threshold)
            all_outliers.extend(outliers)

        # Log and handle outliers
        if all_outliers:
            self._log_outlier_details(all_outliers)
            if raise_errors:
                self._handle_outlier_errors(all_outliers)

            # Add to history
            self.outlier_history.extend(all_outliers)

        return all_outliers

    def _detect_zscore_outliers(
        self, data: pd.DataFrame, column: str, threshold: float
    ) -> List[OutlierInfo]:
        """Detect outliers using Z-score method."""
        outliers = []

        try:
            # Calculate Z-scores
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outlier_indices = np.where(z_scores > threshold)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()

                # Determine severity based on Z-score
                max_z_score = z_scores.max()
                if max_z_score > threshold * 3:
                    severity = OutlierSeverity.CRITICAL
                elif max_z_score > threshold * 2:
                    severity = OutlierSeverity.HIGH
                elif max_z_score > threshold * 1.5:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method="zscore",
                    severity=severity,
                    threshold=threshold,
                )
                outlier_info.context = {
                    "z_scores": z_scores[outlier_indices].tolist(),
                    "max_z_score": max_z_score,
                    "mean": data[column].mean(),
                    "std": data[column].std(),
                }
                outliers.append(outlier_info)

        except Exception as e:
            self.logger.error(f"Error in Z-score outlier detection: {e}")

        return outliers

    def _detect_iqr_outliers(
        self, data: pd.DataFrame, column: str, threshold: float
    ) -> List[OutlierInfo]:
        """Detect outliers using IQR method."""
        outliers = []

        try:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_indices = np.where(
                (data[column] < lower_bound) | (data[column] > upper_bound)
            )[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()

                # Determine severity based on distance from bounds
                distances = []
                for idx in outlier_indices:
                    val = data[column].iloc[idx]
                    if val < lower_bound:
                        distances.append((lower_bound - val) / IQR)
                    else:
                        distances.append((val - upper_bound) / IQR)

                max_distance = max(distances)
                if max_distance > threshold * 2:
                    severity = OutlierSeverity.CRITICAL
                elif max_distance > threshold * 1.5:
                    severity = OutlierSeverity.HIGH
                elif max_distance > threshold * 1.2:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method="iqr",
                    severity=severity,
                    threshold=threshold,
                )
                outlier_info.context = {
                    "Q1": Q1,
                    "Q3": Q3,
                    "IQR": IQR,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "max_distance": max_distance,
                }
                outliers.append(outlier_info)

        except Exception as e:
            self.logger.error(f"Error in IQR outlier detection: {e}")

        return outliers

    def _detect_isolation_forest_outliers(
        self, data: pd.DataFrame, column: str, threshold: float
    ) -> List[OutlierInfo]:
        """Detect outliers using Isolation Forest method."""
        outliers = []

        try:
            from sklearn.ensemble import IsolationForest

            # Prepare data for isolation forest
            X = data[column].values.reshape(-1, 1)

            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X)

            # Find outliers (predictions == -1)
            outlier_indices = np.where(predictions == -1)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()

                # Determine severity based on anomaly scores
                anomaly_scores = iso_forest.decision_function(X)
                outlier_scores = anomaly_scores[outlier_indices]
                min_score = min(outlier_scores)

                if min_score < -0.5:
                    severity = OutlierSeverity.CRITICAL
                elif min_score < -0.3:
                    severity = OutlierSeverity.HIGH
                elif min_score < -0.1:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method="isolation_forest",
                    severity=severity,
                    threshold=threshold,
                )
                outlier_info.context = {
                    "anomaly_scores": outlier_scores.tolist(),
                    "min_score": min_score,
                    "contamination": 0.1,
                }
                outliers.append(outlier_info)

        except ImportError:
            self.logger.warning(
                "scikit-learn not available for isolation forest outlier detection"
            )

        except Exception as e:
            self.logger.error(f"Error in isolation forest outlier detection: {e}")

        return outliers

    def _detect_lof_outliers(
        self, data: pd.DataFrame, column: str, threshold: float
    ) -> List[OutlierInfo]:
        """Detect outliers using Local Outlier Factor method."""
        outliers = []

        try:
            from sklearn.neighbors import LocalOutlierFactor

            # Prepare data for LOF
            X = data[column].values.reshape(-1, 1)

            # Fit LOF
            lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
            predictions = lof.fit_predict(X)

            # Find outliers (predictions == -1)
            outlier_indices = np.where(predictions == -1)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()

                # Determine severity based on LOF scores
                lof_scores = lof.negative_outlier_factor_
                outlier_scores = lof_scores[outlier_indices]
                min_score = min(outlier_scores)

                if min_score < -1.5:
                    severity = OutlierSeverity.CRITICAL
                elif min_score < -1.2:
                    severity = OutlierSeverity.HIGH
                elif min_score < -1.0:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method="local_outlier_factor",
                    severity=severity,
                    threshold=threshold,
                )
                outlier_info.context = {
                    "lof_scores": outlier_scores.tolist(),
                    "min_score": min_score,
                    "contamination": 0.1,
                }
                outliers.append(outlier_info)

        except ImportError:
            self.logger.warning("scikit-learn not available for LOF outlier detection")

        except Exception as e:
            self.logger.error(f"Error in LOF outlier detection: {e}")

        return outliers

    def _detect_mahalanobis_outliers(
        self, data: pd.DataFrame, column: str, threshold: float
    ) -> List[OutlierInfo]:
        """Detect outliers using Mahalanobis distance method."""
        outliers = []

        try:
            from scipy.stats import chi2

            # For single column, use modified Z-score approach
            median = data[column].median()
            mad = np.median(np.abs(data[column] - median))

            if mad == 0:
                return outliers

            modified_z_scores = 0.6745 * (data[column] - median) / mad
            outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()

                # Determine severity based on modified Z-score
                max_score = np.abs(modified_z_scores).max()
                if max_score > threshold * 2:
                    severity = OutlierSeverity.CRITICAL
                elif max_score > threshold * 1.5:
                    severity = OutlierSeverity.HIGH
                elif max_score > threshold * 1.2:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method="mahalanobis",
                    severity=severity,
                    threshold=threshold,
                )
                outlier_info.context = {
                    "modified_z_scores": modified_z_scores[outlier_indices].tolist(),
                    "max_score": max_score,
                    "median": median,
                    "mad": mad,
                }
                outliers.append(outlier_info)

        except ImportError:
            self.logger.warning("scipy not available for Mahalanobis outlier detection")

        except Exception as e:
            self.logger.error(f"Error in Mahalanobis outlier detection: {e}")

        return outliers

    def _log_outlier_details(self, outliers: List[OutlierInfo]) -> None:
        """Log detailed outlier information."""
        if not outliers:
            return

        self.logger.info(f"🔍 Detected {len(outliers)} outlier groups")

        for outlier in outliers:
            self.logger.warning(
                f"Outlier in {outlier.column}: {len(outlier.indices)} values, "
                f"severity={outlier.severity.value}, method={outlier.method}"
            )

            if outlier.severity in [OutlierSeverity.HIGH, OutlierSeverity.CRITICAL]:
                self.logger.error(f"Critical outlier details: {outlier}")
                self.logger.error(
                    f"  Values: {outlier.values[:5]}..."
                )  # Show first 5 values
                self.logger.error(f"  Context: {outlier.context}")

    def _handle_outlier_errors(self, outliers: List[OutlierInfo]) -> None:
        """Handle outlier errors by raising exceptions or logging."""
        critical_outliers = [
            o for o in outliers if o.severity == OutlierSeverity.CRITICAL
        ]
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

    def validate_data_schema(
        self, data: pd.DataFrame, schema_name: str
    ) -> Dict[str, Any]:
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

    def create_custom_schema(
        self,
        name: str,
        required_columns: List[str],
        optional_columns: List[str] = None,
        data_types: Dict[str, str] = None,
        constraints: Dict[str, Dict[str, Any]] = None,
    ) -> DataSchema:
        """Create a custom data schema.

        Args:
            name: Schema name
            required_columns: List of required column names
            optional_columns: List of optional column names
            data_types: Dictionary mapping column names to expected data types
            constraints: Dictionary of column constraints

        Returns:
            Created data schema
        """
        schema = DataSchema(
            name, required_columns, optional_columns, data_types, constraints
        )

        self.standard_schemas[name] = schema
        self.logger.info(f"Created custom schema: {name}")
        return schema

    def get_schema_info(self, schema_name: str) -> Dict[str, Any]:
        """Get information about a schema.

        Args:
            schema_name: Name of the schema

        Returns:
            Schema information
        """
        if schema_name not in self.standard_schemas:
            return {"error": f"Schema {schema_name} not found"}

        schema = self.standard_schemas[schema_name]
        return {
            "name": schema.name,
            "required_columns": list(schema.required_columns),
            "optional_columns": list(schema.optional_columns),
            "all_columns": list(schema.all_columns),
            "data_types": schema.data_types,
            "constraints": schema.constraints,
        }

    def list_available_schemas(self) -> List[str]:
        """List all available schemas.

        Returns:
            List of schema names
        """
        return list(self.standard_schemas.keys())

    def get_outlier_report(self) -> Dict[str, Any]:
        """Generate comprehensive outlier report.

        Returns:
            Outlier analysis report
        """
        if not self.outlier_history:
            return {"message": "No outliers detected"}

        # Group outliers by severity
        severity_counts = {}
        column_counts = {}
        method_counts = {}

        for outlier in self.outlier_history:
            # Severity counts
            severity = outlier.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Column counts
            column = outlier.column
            if column not in column_counts:
                column_counts[column] = {"count": 0, "total_values": 0}
            column_counts[column]["count"] += 1
            column_counts[column]["total_values"] += len(outlier.indices)

            # Method counts
            method = outlier.method
            method_counts[method] = method_counts.get(method, 0) + 1

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_outlier_groups": len(self.outlier_history),
            "severity_distribution": severity_counts,
            "column_distribution": column_counts,
            "method_distribution": method_counts,
            "recent_outliers": [
                {
                    "column": o.column,
                    "count": len(o.indices),
                    "severity": o.severity.value,
                    "method": o.method,
                    "timestamp": o.timestamp.isoformat(),
                }
                for o in self.outlier_history[-10:]  # Last 10 outliers
            ],
        }

        return report

# Global enhanced outlier handler instance
enhanced_outlier_handler = EnhancedOutlierHandler()
