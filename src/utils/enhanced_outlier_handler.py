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
    """Enumeration for outlier severity levels."""
    LOW = "low"           # Minor outliers, log warning
    MEDIUM = "medium"     # Moderate outliers, log error
    HIGH = "high"         # Major outliers, raise exception
    CRITICAL = "critical" # Critical outliers, raise exception and stop processing


class DataSchema:
    """Defines expected data schema for file operations."""
    
    def __init__(
        self, 
        name: str, 
        required_columns: List[str], 
        optional_columns: Optional[List[str]] = None,
        data_types: Optional[Dict[str, type]] = None,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None
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
        """Validate DataFrame against this schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
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
        for col, expected_type in self.data_types.items():
            if col in df.columns:
                if not all(isinstance(val, expected_type) for val in df[col].dropna()):
                    results["type_mismatches"].append(f"Column {col} has unexpected type")
                    results["warnings"].append(f"Type mismatch in column {col}")

        # Check constraints
        for col, constraint in self.constraints.items():
            if col in df.columns:
                if "min" in constraint and df[col].min() < constraint["min"]:
                    results["constraint_violations"].append(f"Column {col} violates min constraint")
                    results["warnings"].append(f"Min constraint violation in {col}")
                
                if "max" in constraint and df[col].max() > constraint["max"]:
                    results["constraint_violations"].append(f"Column {col} violates max constraint")
                    results["warnings"].append(f"Max constraint violation in {col}")

        return results


class OutlierInfo:
    """Information about detected outliers."""
    
    def __init__(
        self, 
        column: str, 
        indices: List[int], 
        values: List[Any], 
        severity: OutlierSeverity,
        method: str,
        threshold: float,
        description: str
    ):
        self.column = column
        self.indices = indices
        self.values = values
        self.severity = severity
        self.method = method
        self.threshold = threshold
        self.description = description
        self.timestamp = datetime.now()
        self.handled = False
        self.handling_method = None

    def __str__(self) -> str:
        return f"Outlier({self.column}, {len(self.indices)} points, {self.severity.value}, {self.method})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "column": self.column,
            "indices": self.indices,
            "values": self.values,
            "severity": self.severity.value,
            "method": self.method,
            "threshold": self.threshold,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "handled": self.handled,
            "handling_method": self.handling_method
        }


class EnhancedOutlierHandler:
    """Enhanced outlier handler with intelligent detection and handling."""
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        raise_on_critical: bool = True,
        log_all_outliers: bool = True
    ):
        """Initialize the enhanced outlier handler.
        
        Args:
            config: Configuration dictionary
            raise_on_critical: Whether to raise exceptions on critical outliers
            log_all_outliers: Whether to log all detected outliers
        """
        self.config = config or {}
        self.raise_on_critical = raise_on_critical
        self.log_all_outliers = log_all_outliers
        self.logger = system_logger.getChild("EnhancedOutlierHandler")
        self.standards = pipeline_standards
        
        # Default outlier detection parameters
        self.default_params = {
            "iqr_multiplier": 1.5,
            "z_score_threshold": 3.0,
            "isolation_forest_contamination": 0.1,
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 5
        }
        
        # Update with config if provided
        if "outlier_detection" in self.config:
            self.default_params.update(self.config["outlier_detection"])
        
        self.logger.info("🚀 Enhanced Outlier Handler initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=[],
        context="outlier detection"
    )
    def detect_outliers(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> List[OutlierInfo]:
        """Detect outliers using multiple methods.
        
        Args:
            data: DataFrame to analyze
            columns: Columns to analyze (uses all numeric columns if None)
            methods: Detection methods to use (uses all if None)
            **kwargs: Additional parameters for detection methods
            
        Returns:
            List of detected outliers
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if methods is None:
            methods = ["iqr", "zscore", "isolation_forest"]
        
        all_outliers = []
        
        for column in columns:
            if column not in data.columns:
                self.logger.warning(f"Column {column} not found in data")
                continue
                
            if not np.issubdtype(data[column].dtype, np.number):
                self.logger.warning(f"Column {column} is not numeric, skipping")
                continue
            
            column_outliers = self._detect_column_outliers(data, column, methods, **kwargs)
            all_outliers.extend(column_outliers)
        
        if self.log_all_outliers:
            self.logger.info(f"Detected {len(all_outliers)} outliers across {len(columns)} columns")
        
        return all_outliers

    def _detect_column_outliers(
        self, 
        data: pd.DataFrame, 
        column: str, 
        methods: List[str], 
        **kwargs
    ) -> List[OutlierInfo]:
        """Detect outliers in a specific column using specified methods."""
        outliers = []
        column_data = data[column].dropna()
        
        if len(column_data) == 0:
            return outliers
        
        for method in methods:
            try:
                if method == "iqr":
                    method_outliers = self._detect_iqr_outliers(column_data, column, **kwargs)
                elif method == "zscore":
                    method_outliers = self._detect_zscore_outliers(column_data, column, **kwargs)
                elif method == "isolation_forest":
                    method_outliers = self._detect_isolation_forest_outliers(column_data, column, **kwargs)
                elif method == "dbscan":
                    method_outliers = self._detect_dbscan_outliers(column_data, column, **kwargs)
                else:
                    self.logger.warning(f"Unknown outlier detection method: {method}")
                    continue
                
                if method_outliers:
                    outliers.append(method_outliers)
                    
            except Exception as e:
                self.logger.error(f"Error in {method} outlier detection for column {column}: {e}")
        
        return outliers

    def _detect_iqr_outliers(
        self, 
        data: pd.Series, 
        column: str, 
        **kwargs
    ) -> Optional[OutlierInfo]:
        """Detect outliers using IQR method."""
        multiplier = kwargs.get("iqr_multiplier", self.default_params["iqr_multiplier"])
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        
        if outlier_indices:
            severity = self._classify_outlier_severity(len(outlier_indices), len(data))
            return OutlierInfo(
                column=column,
                indices=outlier_indices,
                values=outlier_values,
                severity=severity,
                method="iqr",
                threshold=multiplier,
                description=f"IQR outliers (Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f})"
            )
        
        return None

    def _detect_zscore_outliers(
        self, 
        data: pd.Series, 
        column: str, 
        **kwargs
    ) -> Optional[OutlierInfo]:
        """Detect outliers using Z-score method."""
        threshold = kwargs.get("z_score_threshold", self.default_params["z_score_threshold"])
        
        z_scores = np.abs((data - data.mean()) / data.std())
        outlier_mask = z_scores > threshold
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        
        if outlier_indices:
            severity = self._classify_outlier_severity(len(outlier_indices), len(data))
            return OutlierInfo(
                column=column,
                indices=outlier_indices,
                values=outlier_values,
                severity=severity,
                method="zscore",
                threshold=threshold,
                description=f"Z-score outliers (threshold={threshold})"
            )
        
        return None

    def _detect_isolation_forest_outliers(
        self, 
        data: pd.Series, 
        column: str, 
        **kwargs
    ) -> Optional[OutlierInfo]:
        """Detect outliers using Isolation Forest method."""
        try:
            from sklearn.ensemble import IsolationForest
            
            contamination = kwargs.get(
                "isolation_forest_contamination", 
                self.default_params["isolation_forest_contamination"]
            )
            
            # Reshape data for sklearn
            X = data.values.reshape(-1, 1)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            outlier_mask = outlier_labels == -1
            outlier_indices = data[outlier_mask].index.tolist()
            outlier_values = data[outlier_mask].tolist()
            
            if outlier_indices:
                severity = self._classify_outlier_severity(len(outlier_indices), len(data))
                return OutlierInfo(
                    column=column,
                    indices=outlier_indices,
                    values=outlier_values,
                    severity=severity,
                    method="isolation_forest",
                    threshold=contamination,
                    description=f"Isolation Forest outliers (contamination={contamination})"
                )
                
        except ImportError:
            self.logger.warning("scikit-learn not available, skipping Isolation Forest")
        except Exception as e:
            self.logger.error(f"Error in Isolation Forest detection: {e}")
        
        return None

    def _detect_dbscan_outliers(
        self, 
        data: pd.Series, 
        column: str, 
        **kwargs
    ) -> Optional[OutlierInfo]:
        """Detect outliers using DBSCAN method."""
        try:
            from sklearn.cluster import DBSCAN
            
            eps = kwargs.get("dbscan_eps", self.default_params["dbscan_eps"])
            min_samples = kwargs.get("dbscan_min_samples", self.default_params["dbscan_min_samples"])
            
            # Reshape data for sklearn
            X = data.values.reshape(-1, 1)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X)
            
            # Points with label -1 are outliers
            outlier_mask = cluster_labels == -1
            outlier_indices = data[outlier_mask].index.tolist()
            outlier_values = data[outlier_mask].tolist()
            
            if outlier_indices:
                severity = self._classify_outlier_severity(len(outlier_indices), len(data))
                return OutlierInfo(
                    column=column,
                    indices=outlier_indices,
                    values=outlier_values,
                    severity=severity,
                    method="dbscan",
                    threshold=eps,
                    description=f"DBSCAN outliers (eps={eps}, min_samples={min_samples})"
                )
                
        except ImportError:
            self.logger.warning("scikit-learn not available, skipping DBSCAN")
        except Exception as e:
            self.logger.error(f"Error in DBSCAN detection: {e}")
        
        return None

    def _classify_outlier_severity(self, outlier_count: int, total_count: int) -> OutlierSeverity:
        """Classify outlier severity based on count and proportion."""
        proportion = outlier_count / total_count
        
        if proportion > 0.1:  # More than 10% outliers
            return OutlierSeverity.CRITICAL
        elif proportion > 0.05:  # More than 5% outliers
            return OutlierSeverity.HIGH
        elif proportion > 0.02:  # More than 2% outliers
            return OutlierSeverity.MEDIUM
        else:
            return OutlierSeverity.LOW

    @handle_errors(
        exceptions=(Exception,),
        default_return=data,
        context="outlier handling"
    )
    def handle_outliers(
        self, 
        data: pd.DataFrame, 
        outliers: List[OutlierInfo],
        strategy: str = "remove",
        **kwargs
    ) -> pd.DataFrame:
        """Handle detected outliers according to specified strategy.
        
        Args:
            data: DataFrame containing outliers
            outliers: List of detected outliers
            strategy: Handling strategy ("remove", "cap", "interpolate", "log_only")
            **kwargs: Additional parameters for handling strategy
            
        Returns:
            DataFrame with outliers handled
        """
        if not outliers:
            return data
        
        # Sort outliers by severity (most severe first)
        severity_order = {
            OutlierSeverity.CRITICAL: 0,
            OutlierSeverity.HIGH: 1,
            OutlierSeverity.MEDIUM: 2,
            OutlierSeverity.LOW: 3
        }
        sorted_outliers = sorted(outliers, key=lambda x: severity_order[x.severity])
        
        handled_data = data.copy()
        
        for outlier in sorted_outliers:
            try:
                if outlier.severity == OutlierSeverity.CRITICAL and self.raise_on_critical:
                    raise ValueError(f"Critical outliers detected in column {outlier.column}: {outlier}")
                
                if strategy == "remove":
                    handled_data = self._remove_outliers(handled_data, outlier)
                elif strategy == "cap":
                    handled_data = self._cap_outliers(handled_data, outlier)
                elif strategy == "interpolate":
                    handled_data = self._interpolate_outliers(handled_data, outlier)
                elif strategy == "log_only":
                    self._log_outlier(outlier)
                    continue
                else:
                    self.logger.warning(f"Unknown strategy: {strategy}, using log_only")
                    self._log_outlier(outlier)
                    continue
                
                outlier.handled = True
                outlier.handling_method = strategy
                
            except Exception as e:
                self.logger.error(f"Error handling outlier in column {outlier.column}: {e}")
                if outlier.severity in [OutlierSeverity.HIGH, OutlierSeverity.CRITICAL]:
                    raise
        
        return handled_data

    def _remove_outliers(self, data: pd.DataFrame, outlier: OutlierInfo) -> pd.DataFrame:
        """Remove outliers from data."""
        self.logger.info(f"Removing {len(outlier.indices)} outliers from column {outlier.column}")
        return data.drop(outlier.indices)

    def _cap_outliers(self, data: pd.DataFrame, outlier: OutlierInfo) -> pd.DataFrame:
        """Cap outliers at threshold values."""
        self.logger.info(f"Capping {len(outlier.indices)} outliers in column {outlier.column}")
        
        # Calculate bounds based on method
        if outlier.method == "iqr":
            Q1 = data[outlier.column].quantile(0.25)
            Q3 = data[outlier.column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier.threshold * IQR
            upper_bound = Q3 + outlier.threshold * IQR
        elif outlier.method == "zscore":
            mean_val = data[outlier.column].mean()
            std_val = data[outlier.column].std()
            lower_bound = mean_val - outlier.threshold * std_val
            upper_bound = mean_val + outlier.threshold * std_val
        else:
            # Use percentiles for other methods
            lower_bound = data[outlier.column].quantile(0.01)
            upper_bound = data[outlier.column].quantile(0.99)
        
        # Cap outliers
        data_copy = data.copy()
        data_copy.loc[outlier.indices, outlier.column] = data_copy.loc[outlier.indices, outlier.column].clip(
            lower=lower_bound, upper=upper_bound
        )
        
        return data_copy

    def _interpolate_outliers(self, data: pd.DataFrame, outlier: OutlierInfo) -> pd.DataFrame:
        """Interpolate outliers using surrounding values."""
        self.logger.info(f"Interpolating {len(outlier.indices)} outliers in column {outlier.column}")
        
        data_copy = data.copy()
        data_copy.loc[outlier.indices, outlier.column] = np.nan
        
        # Interpolate missing values
        data_copy[outlier.column] = data_copy[outlier.column].interpolate(method='linear')
        
        # Fill any remaining NaNs with forward/backward fill
        data_copy[outlier.column] = data_copy[outlier.column].fillna(method='ffill').fillna(method='bfill')
        
        return data_copy

    def _log_outlier(self, outlier: OutlierInfo) -> None:
        """Log outlier information."""
        log_msg = f"Outlier detected in {outlier.column}: {len(outlier.indices)} points, {outlier.severity.value}"
        if outlier.severity == OutlierSeverity.LOW:
            self.logger.warning(log_msg)
        elif outlier.severity == OutlierSeverity.MEDIUM:
            self.logger.error(log_msg)
        else:
            self.logger.critical(log_msg)

    def generate_outlier_report(self, outliers: List[OutlierInfo]) -> Dict[str, Any]:
        """Generate comprehensive outlier report."""
        if not outliers:
            return {"message": "No outliers detected"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_outliers": len(outliers),
            "columns_affected": list(set(o.column for o in outliers)),
            "severity_summary": {},
            "method_summary": {},
            "column_summary": {},
            "outlier_details": []
        }
        
        # Severity summary
        for severity in OutlierSeverity:
            severity_outliers = [o for o in outliers if o.severity == severity]
            report["severity_summary"][severity.value] = len(severity_outliers)
        
        # Method summary
        for outlier in outliers:
            method = outlier.method
            if method not in report["method_summary"]:
                report["method_summary"][method] = 0
            report["method_summary"][method] += 1
        
        # Column summary
        for outlier in outliers:
            column = outlier.column
            if column not in report["column_summary"]:
                report["column_summary"][column] = {
                    "total_outliers": 0,
                    "severity_counts": {s.value: 0 for s in OutlierSeverity},
                    "methods_used": set()
                }
            
            report["column_summary"][column]["total_outliers"] += 1
            report["column_summary"][column]["severity_counts"][outlier.severity.value] += 1
            report["column_summary"][column]["methods_used"].add(outlier.method)
        
        # Convert sets to lists for JSON serialization
        for column_info in report["column_summary"].values():
            column_info["methods_used"] = list(column_info["methods_used"])
        
        # Outlier details
        for outlier in outliers:
            report["outlier_details"].append(outlier.to_dict())
        
        return report

    def validate_data_integrity(self, data: pd.DataFrame, schema: DataSchema) -> Dict[str, Any]:
        """Validate data integrity using schema."""
        validation_results = schema.validate_dataframe(data)
        
        if not validation_results["valid"]:
            self.logger.error(f"Data validation failed for schema {schema.name}")
            for error in validation_results["errors"]:
                self.logger.error(f"  - {error}")
        
        return validation_results


# Global enhanced outlier handler instance
enhanced_outlier_handler = EnhancedOutlierHandler()