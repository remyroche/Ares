#!/usr/bin/env python3
from src.utils.tprint import tprint

import numpy as np
import pandas as pd

"""
Unified Data Quality Orchestrator

This module unites all quality checking scripts and modules from the data_quality/ directory
and related modules throughout the project. It provides a single entry point for comprehensive
data quality assessment, validation, and monitoring.

Features:
- Data validation and schema enforcement
- Data quality scoring and metrics
- Data cleaning and preprocessing
- Data profiling and analysis
- Quality policy management
- Cross-step quality consistency
- Multicollinearity detection
- Label imbalance analysis
- Feature redundancy identification
- Outlier detection and handling
- Temporal data validation
- Format compatibility checking
- Dependency graph analysis
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    tprint("Warning: pandas not available, some functionality will be limited")

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    import time
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    tprint("Warning: sklearn not available, some functionality will be limited")

try:
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    tprint("Warning: networkx not available, dependency graph functionality will be limited")

# Import project-specific modules
try:
    from src.utils.enhanced_logger import (
        create_detailed_step_report,
        log_step_metrics,
        log_step_report,
    )
    from src.utils.enhanced_outlier_handler import (
        OutlierSeverity,
        enhanced_outlier_handler,
    )
    from src.utils.logger import system_logger
except ImportError:
    # Fallback logger if project modules aren't available
    system_logger = logging.getLogger("UnifiedQualityOrchestrator")
    tprint("Warning: Some project modules not available, using fallback implementations")


class QualityLevel(Enum):
    """Quality level enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class DataQualityLevel(Enum):
    """Data quality issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataFormat(Enum):
    """Standard data formats."""
    KLINES = "klines"
    FEATURES = "features"
    LABELS = "labels"
    PREDICTIONS = "predictions"
    METADATA = "metadata"
    CONFIG = "config"


@dataclass
class QualityThresholds:
    """Quality validation thresholds."""
    max_nan_ratio: float = 0.0  # Zero tolerance for NaN
    max_infinite_count: int = 0  # Zero tolerance for infinite values
    min_unique_values: int = 2
    max_constant_ratio: float = 0.95
    max_gap_hours: int = 48
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001
    max_correlation_threshold: float = 0.95
    min_feature_count: int = 40
    vif_threshold: float = 5.0  # For multicollinearity detection


@dataclass
class QualityResult:
    """Result of data quality validation."""
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_issue(self, issue_type: str, description: str):
        """Add a quality issue."""
        self.issues.append(f"{issue_type}: {description}")
        self.passed = False

    def add_warning(self, warning_type: str, description: str):
        """Add a quality warning."""
        self.warnings.append(f"{warning_type}: {description}")

    def add_metric(self, name: str, value: Any):
        """Add a quality metric."""
        self.metrics[name] = value

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the validation result."""
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "warning_count": len(self.warnings),
            "metrics": self.metrics,
            "issues": self.issues[:5],  # First 5 issues
            "warnings": self.warnings[:5],  # First 5 warnings
        }


@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""
    completeness: float
    consistency: float
    validity: float
    timeliness: float
    uniqueness: float
    accuracy: float
    overall_score: float
    quality_level: QualityLevel
    issues: list[str]
    warnings: list[str]
    recommendations: list[str]
    timestamp: datetime


@dataclass
class CompatibilityMetrics:
    """Data compatibility metrics container."""
    format_compatible: bool
    schema_compatible: bool
    type_compatible: bool
    index_compatible: bool
    temporal_aligned: bool
    overall_compatible: bool
    issues: list[str]
    warnings: list[str]
    conversions_applied: list[str]
    timestamp: datetime


class UnifiedQualityOrchestrator:
    """
    Unified orchestrator for all data quality checking functionality.

    This class integrates:
    - Data validation and schema enforcement
    - Quality scoring and metrics
    - Data cleaning and preprocessing
    - Data profiling and analysis
    - Multicollinearity detection
    - Label imbalance analysis
    - Feature redundancy identification
    - Outlier detection and handling
    - Temporal data validation
    - Format compatibility checking
    - Dependency graph analysis
    """

    def __init__(self, thresholds: QualityThresholds | None = None):
        """Initialize the unified quality orchestrator."""
        self.thresholds = thresholds or QualityThresholds()
        self.logger = system_logger.getChild("UnifiedQualityOrchestrator")

        # Quality policies
        self.quality_policies = {
            "strict_validation": True,
            "auto_clean": True,
            "profiling_enabled": True,
            "max_issues_critical": 0,
            "max_issues_high": 5,
            "max_issues_medium": 20,
            "max_issues_low": 100,
        }

        # Validation rules
        self.validation_rules = {
            "klines_schema": {
                "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "data_types": {
                    "timestamp": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                },
                "constraints": {
                    "timestamp": {"min": 0, "max": None},
                    "open": {"min": 0, "max": None},
                    "high": {"min": 0, "max": None},
                    "low": {"min": 0, "max": None},
                    "close": {"min": 0, "max": None},
                    "volume": {"min": 0, "max": None},
                },
            },
            "features_schema": {
                "required_columns": ["timestamp"],
                "data_types": {
                    "timestamp": "int64",
                },
                "constraints": {
                    "timestamp": {"min": 0, "max": None},
                },
            },
            "labels_schema": {
                "required_columns": ["timestamp", "label"],
                "data_types": {
                    "timestamp": "int64",
                    "label": "int64",
                },
                "constraints": {
                    "timestamp": {"min": 0, "max": None},
                    "label": {"min": 0, "max": None},
                },
            },
        }

    def validate_dataframe_quality(self, df, context: str = "") -> QualityResult:
        """Validate DataFrame quality with comprehensive checks."""
        if not PANDAS_AVAILABLE:
            msg = "pandas is required for data quality validation"
            raise ImportError(msg)

        result = QualityResult()

        if df is None or (hasattr(df, "empty") and df.empty):
            result.add_issue("empty_data", "DataFrame is None or empty")
            return result

        # Basic metrics
        result.add_metric("rows", len(df))
        result.add_metric("columns", len(df.columns))
        if hasattr(df, "memory_usage"):
            result.add_metric("memory_mb", df.memory_usage(deep=True).sum() / 1024 / 1024)
        else:
            result.add_metric("memory_mb", "unknown")

        # Check for NaN values
        self._validate_nan_values(df, result)

        # Check for infinite values
        self._validate_infinite_values(df, result)

        # Check for constant columns
        self._validate_constant_columns(df, result)

        # Check for duplicate rows
        self._validate_duplicates(df, result)

        # Check data types
        self._validate_data_types(df, result)

        # Check for outliers (if enhanced_outlier_handler is available)
        if "enhanced_outlier_handler" in globals():
            self._validate_outliers(df, result)

        return result

    def _validate_nan_values(self, df: pd.DataFrame, result: QualityResult):
        """Validate NaN values in DataFrame."""
        try:
            nan_counts = df.isnull().sum()
            total_cells = len(df) * len(df.columns)
            nan_ratio = nan_counts.sum() / total_cells

            result.add_metric("nan_ratio", nan_ratio)
            result.add_metric("nan_counts", nan_counts.to_dict())

            if nan_ratio > self.thresholds.max_nan_ratio:
                problematic_columns = nan_counts[nan_counts > 0].to_dict()
                result.add_issue("high_nan_ratio", 
                    f"NaN ratio {nan_ratio:.2%} exceeds threshold {self.thresholds.max_nan_ratio:.2%}. "
                    f"Problematic columns: {problematic_columns}")

            # Check for columns with high NaN ratios
            high_nan_cols = nan_counts[nan_counts > len(df) * 0.5]
            if not high_nan_cols.empty:
                result.add_warning("high_nan_columns", 
                    f"Columns with >50% NaN values: {list(high_nan_cols.index)}. "
                    f"These columns may need data cleaning or removal.")
        except Exception as e:
            result.add_issue("nan_validation_error", f"Failed to validate NaN values: {str(e)}")

    def _validate_infinite_values(self, df: pd.DataFrame, result: QualityResult):
        """Validate infinite values in DataFrame."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number])
            if numeric_cols.empty:
                result.add_metric("infinite_count", 0)
                result.add_metric("infinite_counts", {})
                return
                
            inf_counts = np.isinf(numeric_cols).sum()
            total_inf = inf_counts.sum()

            result.add_metric("infinite_count", total_inf)
            result.add_metric("infinite_counts", inf_counts.to_dict())

            if total_inf > self.thresholds.max_infinite_count:
                problematic_columns = inf_counts[inf_counts > 0].to_dict()
                result.add_issue("infinite_values", 
                    f"Found {total_inf} infinite values exceeding threshold {self.thresholds.max_infinite_count}. "
                    f"Problematic columns: {problematic_columns}. "
                    f"Check for division by zero or mathematical operations that produce infinity.")
        except Exception as e:
            result.add_issue("infinite_validation_error", f"Failed to validate infinite values: {str(e)}")

    def _validate_constant_columns(self, df: pd.DataFrame, result: QualityResult):
        """Validate constant columns in DataFrame."""
        try:
            constant_cols = []
            near_constant_cols = []
            
            for col in df.columns:
                unique_count = df[col].nunique()
                if unique_count == 1:
                    constant_cols.append(col)
                elif unique_count == 2 and len(df) > 10:
                    # Check if one value dominates (>95% of data)
                    value_counts = df[col].value_counts()
                    max_ratio = value_counts.iloc[0] / len(df)
                    if max_ratio > 0.95:
                        near_constant_cols.append((col, max_ratio))

            result.add_metric("constant_columns", constant_cols)
            result.add_metric("near_constant_columns", near_constant_cols)

            if constant_cols:
                result.add_warning("constant_columns", 
                    f"🚨 Constant columns found: {constant_cols}. "
                    f"These columns provide no information and should be removed.")
            
            if near_constant_cols:
                result.add_warning("near_constant_columns", 
                    f"⚠️ Near-constant columns found: {near_constant_cols}. "
                    f"These columns have very little variation and may not be useful for analysis.")
        except Exception as e:
            result.add_issue("constant_validation_error", f"Failed to validate constant columns: {str(e)}")

    def _validate_duplicates(self, df: pd.DataFrame, result: QualityResult):
        """Validate duplicate rows in DataFrame."""
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)

        result.add_metric("duplicate_count", duplicate_count)
        result.add_metric("duplicate_ratio", duplicate_ratio)

        if duplicate_ratio > 0.1:  # More than 10% duplicates
            result.add_issue("high_duplicates", f"Duplicate ratio {duplicate_ratio:.2%} is too high")

    def _validate_data_types(self, df: pd.DataFrame, result: QualityResult):
        """Validate data types in DataFrame."""
        dtypes = df.dtypes.to_dict()
        result.add_metric("data_types", dtypes)

        # Check for mixed types in columns
        mixed_type_cols = []
        for col in df.columns:
            if df[col].apply(type).nunique() > 1:
                mixed_type_cols.append(col)

        if mixed_type_cols:
            result.add_warning("mixed_types", f"Mixed type columns: {mixed_type_cols}")

    def _validate_outliers(self, df: pd.DataFrame, result: QualityResult):
        """Validate outliers using enhanced outlier handler."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_results = {}

            for col in numeric_cols:
                outliers = enhanced_outlier_handler(df[col], severity=OutlierSeverity.HIGH)
                outlier_results[col] = len(outliers)

            result.add_metric("outlier_counts", outlier_results)

            total_outliers = sum(outlier_results.values())
            if total_outliers > 0:
                result.add_warning("outliers_detected", f"Found {total_outliers} outliers across numeric columns")

        except Exception as e:
            result.add_warning("outlier_validation_failed", f"Outlier validation failed: {str(e)}")

    def analyze_multicollinearity(self, data: pd.DataFrame, vif_threshold: float = None) -> dict[str, Any]:
        """
        Analyze multicollinearity using VIF and correlation analysis.

        Args:
            data: Input DataFrame
            vif_threshold: VIF threshold for flagging multicollinearity

        Returns:
            Dictionary with multicollinearity analysis results
        """
        if not SKLEARN_AVAILABLE:
            msg = "sklearn is required for multicollinearity analysis"
            raise ImportError(msg)

        if vif_threshold is None:
            vif_threshold = self.thresholds.vif_threshold

        self.logger.info("🔍 Analyzing multicollinearity...")

        try:
            # Remove non-numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                self.logger.warning("⚠️ No numeric columns found for multicollinearity analysis")
                return {
                    "error": "No numeric columns found",
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            if len(numeric_data.columns) < 2:
                self.logger.warning("⚠️ Need at least 2 numeric columns for multicollinearity analysis")
                return {
                    "error": "Need at least 2 numeric columns",
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            # Remove potential label columns
            potential_label_columns = [
                "label", "target", "y", "class", "Label", "Target", "Y", "Class",
            ]
            actual_label_columns = [
                col for col in numeric_data.columns if col in potential_label_columns
            ]

            if actual_label_columns:
                self.logger.warning(
                    f"⚠️ Removing label columns from multicollinearity analysis: {actual_label_columns}",
                )
                numeric_data = numeric_data.drop(columns=actual_label_columns)

            # Check for constant columns that would cause VIF calculation issues
            constant_cols = [col for col in numeric_data.columns if numeric_data[col].nunique() <= 1]
            if constant_cols:
                self.logger.warning(f"⚠️ Removing constant columns for VIF analysis: {constant_cols}")
                numeric_data = numeric_data.drop(columns=constant_cols)

            if len(numeric_data.columns) < 2:
                return {
                    "error": "Not enough non-constant numeric columns for VIF analysis",
                    "constant_columns_removed": constant_cols,
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            # Handle NaN values
            nan_counts = numeric_data.isnull().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"⚠️ Found NaN values in numeric data, imputing with median. NaN counts: {nan_counts[nan_counts > 0].to_dict()}")
            
            imputer = SimpleImputer(strategy="median")
            data_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index,
            )

            # Calculate VIF scores
            vif_scores = {}
            high_vif_features = []
            vif_errors = []

            for col in data_imputed.columns:
                other_cols = [c for c in data_imputed.columns if c != col]
                if len(other_cols) > 0:
                    try:
                        X = data_imputed[other_cols]
                        y = data_imputed[col]

                        # Check for perfect correlation (would cause division by zero)
                        if X.corrwith(y).abs().max() >= 0.999:
                            vif_scores[col] = float('inf')
                            high_vif_features.append(col)
                            vif_errors.append(f"Perfect correlation detected for {col}")
                            continue

                        reg = LinearRegression()
                        reg.fit(X, y)
                        r_squared = reg.score(X, y)

                        # Calculate VIF
                        if r_squared >= 0.999:
                            vif_scores[col] = float('inf')
                            high_vif_features.append(col)
                        else:
                            vif = 1 / (1 - r_squared)
                            vif_scores[col] = vif

                            if vif > vif_threshold:
                                high_vif_features.append(col)
                    except Exception as e:
                        vif_errors.append(f"Error calculating VIF for {col}: {str(e)}")
                        vif_scores[col] = None

            # Calculate correlation matrix
            try:
                correlation_matrix = data_imputed.corr()
                high_corr_pairs = []

                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr_val = abs(correlation_matrix.iloc[i, j])
                        if corr_val > self.thresholds.max_correlation_threshold:
                            col1 = correlation_matrix.columns[i]
                            col2 = correlation_matrix.columns[j]
                            high_corr_pairs.append((col1, col2, corr_val))
            except Exception as e:
                self.logger.error(f"❌ Error calculating correlation matrix: {str(e)}")
                correlation_matrix = None
                high_corr_pairs = []

            result = {
                "vif_scores": vif_scores,
                "high_vif_features": high_vif_features,
                "correlation_matrix": correlation_matrix.to_dict() if correlation_matrix is not None else None,
                "high_correlation_pairs": high_corr_pairs,
                "vif_threshold": vif_threshold,
                "correlation_threshold": self.thresholds.max_correlation_threshold,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            if vif_errors:
                result["vif_errors"] = vif_errors
                self.logger.warning(f"⚠️ VIF calculation errors: {vif_errors}")

            if high_vif_features:
                self.logger.warning(f"🚨 High VIF features detected: {high_vif_features}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Multicollinearity analysis failed: {str(e)}")
            return {
                "error": f"Multicollinearity analysis failed: {str(e)}",
                "analysis_timestamp": datetime.now().isoformat(),
            }

    def analyze_label_imbalance(self, labels: pd.Series) -> dict[str, Any]:
        """
        Analyze label imbalance in classification datasets.

        Args:
            labels: Series containing labels

        Returns:
            Dictionary with label imbalance analysis results
        """
        if not PANDAS_AVAILABLE:
            msg = "pandas is required for label imbalance analysis"
            raise ImportError(msg)

        self.logger.info("🔍 Analyzing label imbalance...")

        # Count occurrences of each label
        label_counts = labels.value_counts()
        total_samples = len(labels)

        # Calculate imbalance metrics
        imbalance_ratio = label_counts.max() / label_counts.min()
        entropy = -(label_counts / total_samples * np.log2(label_counts / total_samples)).sum()
        gini_coefficient = 1 - ((label_counts / total_samples) ** 2).sum()

        # Determine imbalance level
        if imbalance_ratio > 10:
            imbalance_level = "severe"
        elif imbalance_ratio > 5:
            imbalance_level = "moderate"
        elif imbalance_ratio > 2:
            imbalance_level = "mild"
        else:
            imbalance_level = "balanced"

        # Generate recommendations
        recommendations = []
        if imbalance_level == "severe":
            recommendations.extend([
                "Consider using SMOTE or other oversampling techniques",
                "Use class weights in model training",
                "Evaluate using F1-score or other balanced metrics",
            ])
        elif imbalance_level == "moderate":
            recommendations.extend([
                "Consider using class weights",
                "Monitor model performance on minority classes",
            ])

        return {
            "label_counts": label_counts.to_dict(),
            "total_samples": total_samples,
            "imbalance_ratio": imbalance_ratio,
            "entropy": entropy,
            "gini_coefficient": gini_coefficient,
            "imbalance_level": imbalance_level,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def analyze_feature_redundancy(self, data: pd.DataFrame, correlation_threshold: float = None) -> dict[str, Any]:
        """
        Analyze feature redundancy based on correlation analysis.

        Args:
            data: Input DataFrame
            correlation_threshold: Correlation threshold for identifying redundant features

        Returns:
            Dictionary with feature redundancy analysis results
        """
        if not PANDAS_AVAILABLE:
            msg = "pandas is required for feature redundancy analysis"
            raise ImportError(msg)

        if correlation_threshold is None:
            correlation_threshold = self.thresholds.max_correlation_threshold

        self.logger.info("🔍 Analyzing feature redundancy...")

        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        # Find highly correlated feature pairs
        redundant_pairs = []
        redundant_features = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > correlation_threshold:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    redundant_pairs.append((col1, col2, corr_val))
                    redundant_features.add(col1)
                    redundant_features.add(col2)

        # Calculate redundancy metrics
        total_features = len(numeric_data.columns)
        redundant_feature_count = len(redundant_features)
        redundancy_ratio = redundant_feature_count / total_features

        # Generate recommendations
        recommendations = []
        if redundancy_ratio > 0.3:
            recommendations.append("High feature redundancy detected - consider feature selection")
        elif redundancy_ratio > 0.1:
            recommendations.append("Moderate feature redundancy - review highly correlated features")

        if redundant_pairs:
            recommendations.append("Remove one feature from each highly correlated pair")

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "redundant_pairs": redundant_pairs,
            "redundant_features": list(redundant_features),
            "total_features": total_features,
            "redundant_feature_count": redundant_feature_count,
            "redundancy_ratio": redundancy_ratio,
            "correlation_threshold": correlation_threshold,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def validate_temporal_data(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> QualityResult:
        """
        Validate temporal aspects of data.

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column

        Returns:
            QualityResult with temporal validation results
        """
        result = QualityResult()

        if timestamp_col not in df.columns:
            result.add_issue("missing_timestamp", f"Timestamp column '{timestamp_col}' not found")
            return result

        # Check timestamp format
        try:
            timestamps = pd.to_datetime(df[timestamp_col], unit="s")
            result.add_metric("timestamp_format", "unix_timestamp")
        except:
            try:
                timestamps = pd.to_datetime(df[timestamp_col])
                result.add_metric("timestamp_format", "datetime")
            except:
                result.add_issue("invalid_timestamp_format", f"Could not parse timestamps in column '{timestamp_col}'")
                return result

        # Check for gaps
        timestamps_sorted = timestamps.sort_values()
        time_diffs = timestamps_sorted.diff().dropna()

        if len(time_diffs) > 0:
            max_gap = time_diffs.max()
            min_gap = time_diffs.min()
            mean_gap = time_diffs.mean()

            result.add_metric("max_gap", str(max_gap))
            result.add_metric("min_gap", str(min_gap))
            result.add_metric("mean_gap", str(mean_gap))

            # Check for gaps exceeding threshold
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=self.thresholds.max_gap_hours)]
            if not large_gaps.empty:
                result.add_issue("large_gaps", f"Found {len(large_gaps)} gaps larger than {self.thresholds.max_gap_hours} hours")

        # Check for duplicates
        duplicate_timestamps = timestamps.duplicated().sum()
        if duplicate_timestamps > 0:
            result.add_issue("duplicate_timestamps", f"Found {duplicate_timestamps} duplicate timestamps")

        # Check for future timestamps
        now = pd.Timestamp.now()
        future_timestamps = (timestamps > now).sum()
        if future_timestamps > 0:
            result.add_warning("future_timestamps", f"Found {future_timestamps} timestamps in the future")

        return result

    def generate_comprehensive_report(self, data: pd.DataFrame, context: str = "") -> dict[str, Any]:
        """
        Generate a comprehensive data quality report.

        Args:
            data: Input DataFrame
            context: Context description for the data

        Returns:
            Comprehensive quality report
        """
        self.logger.info(f"🔍 Generating comprehensive quality report for {context}")

        report = {
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape if data is not None else None,
            "quality_validation": None,
            "multicollinearity_analysis": None,
            "feature_redundancy_analysis": None,
            "temporal_validation": None,
            "summary": {},
        }

        # Basic quality validation
        if data is not None:
            quality_result = self.validate_dataframe_quality(data, context)
            report["quality_validation"] = quality_result.get_summary()

            # Multicollinearity analysis (for numeric data)
            try:
                numeric_data = data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    multicollinearity = self.analyze_multicollinearity(numeric_data)
                    report["multicollinearity_analysis"] = multicollinearity
            except Exception as e:
                report["multicollinearity_analysis"] = {"error": str(e)}

            # Feature redundancy analysis
            try:
                feature_redundancy = self.analyze_feature_redundancy(data)
                report["feature_redundancy_analysis"] = feature_redundancy
            except Exception as e:
                report["feature_redundancy_analysis"] = {"error": str(e)}

            # Temporal validation (if timestamp column exists)
            timestamp_cols = [col for col in data.columns if "time" in col.lower() or col == "timestamp"]
            if timestamp_cols:
                try:
                    temporal_validation = self.validate_temporal_data(data, timestamp_cols[0])
                    report["temporal_validation"] = temporal_validation.get_summary()
                except Exception as e:
                    report["temporal_validation"] = {"error": str(e)}

        # Generate summary
        report["summary"] = self._generate_summary(report)

        return report

    def _generate_summary(self, report: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of the comprehensive report."""
        summary = {
            "overall_quality": "unknown",
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "warnings": 0,
            "recommendations": [],
        }

        # Count issues from quality validation
        if report.get("quality_validation"):
            quality_val = report["quality_validation"]
            summary["critical_issues"] += quality_val.get("issue_count", 0)

        # Count issues from temporal validation
        if report.get("temporal_validation"):
            temp_val = report["temporal_validation"]
            summary["critical_issues"] += temp_val.get("issue_count", 0)

        # Determine overall quality
        if summary["critical_issues"] == 0:
            summary["overall_quality"] = "excellent"
        elif summary["critical_issues"] <= 2:
            summary["overall_quality"] = "good"
        elif summary["critical_issues"] <= 5:
            summary["overall_quality"] = "acceptable"
        elif summary["critical_issues"] <= 10:
            summary["overall_quality"] = "poor"
        else:
            summary["overall_quality"] = "critical"

        # Add recommendations
        if report.get("multicollinearity_analysis"):
            multicoll = report["multicollinearity_analysis"]
            if multicoll.get("high_vif_features"):
                summary["recommendations"].append("Consider removing high VIF features")

        if report.get("feature_redundancy_analysis"):
            redundancy = report["feature_redundancy_analysis"]
            if redundancy.get("redundancy_ratio", 0) > 0.3:
                summary["recommendations"].append("High feature redundancy - consider feature selection")

        return summary

    def save_report(self, report: dict[str, Any], filename: str = None) -> str:
        """
        Save the quality report to a file.

        Args:
            report: Quality report to save
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"✅ Quality report saved to: {output_path}")
        return str(output_path)

    def analyze_directory(self, directory_path: str, file_pattern: str = "*", recursive: bool = True) -> dict[str, Any]:
        """
        Analyze all data files in a directory.

        Args:
            directory_path: Path to directory to analyze
            file_pattern: Glob pattern for file matching (default: "*")
            recursive: Whether to search subdirectories recursively

        Returns:
            Dictionary with directory analysis results
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            msg = f"Path is not a directory: {directory_path}"
            raise ValueError(msg)

        self.logger.info(f"🔍 Analyzing directory: {directory_path}")

        # Find all data files
        if recursive:
            data_files = list(directory.rglob(file_pattern))
        else:
            data_files = list(directory.glob(file_pattern))

        # Filter for supported data formats
        supported_extensions = {".csv", ".parquet", ".json"}
        data_files = [f for f in data_files if f.is_file() and f.suffix.lower() in supported_extensions]

        if not data_files:
            self.logger.warning(f"No supported data files found in {directory_path}")
            return {"error": "No supported data files found"}

        self.logger.info(f"Found {len(data_files)} data files to analyze")

        # Analyze each file
        results = {}
        summary_stats = {
            "total_files": len(data_files),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "overall_quality_scores": [],
            "critical_issues_total": 0,
            "high_issues_total": 0,
            "medium_issues_total": 0,
            "low_issues_total": 0,
            "warnings_total": 0,
            "recommendations": set(),
        }

        for file_path in data_files:
            try:
                self.logger.info(f"Analyzing file: {file_path.name}")

                # Load data
                data = self._load_data_file(file_path)
                if data is None:
                    continue

                # Generate report
                context = f"File: {file_path.name} (Path: {file_path.relative_to(directory)})"
                report = self.generate_comprehensive_report(data, context)

                # Store results
                results[str(file_path)] = report
                summary_stats["successful_analyses"] += 1

                # Aggregate statistics
                if report.get("summary"):
                    summary = report["summary"]
                    summary_stats["overall_quality_scores"].append(summary.get("overall_quality", "unknown"))

                    # Count issues by severity
                    if report.get("quality_validation"):
                        quality_val = report["quality_validation"]
                        summary_stats["critical_issues_total"] += quality_val.get("issue_count", 0)

                    if report.get("temporal_validation"):
                        temp_val = report["temporal_validation"]
                        summary_stats["critical_issues_total"] += temp_val.get("issue_count", 0)

                    # Collect recommendations
                    if summary.get("recommendations"):
                        summary_stats["recommendations"].update(summary["recommendations"])

            except Exception as e:
                self.logger.exception(f"Failed to analyze {file_path.name}: {e}")
                results[str(file_path)] = {"error": str(e)}
                summary_stats["failed_analyses"] += 1

        # Generate directory summary
        directory_summary = self._generate_directory_summary(summary_stats, results)

        return {
            "directory_path": str(directory),
            "analysis_timestamp": datetime.now().isoformat(),
            "file_pattern": file_pattern,
            "recursive": recursive,
            "summary": directory_summary,
            "file_results": results,
        }

    def _load_data_file(self, file_path: Path) -> pd.DataFrame | None:
        """Load a single data file with detailed error reporting."""
        try:
            if file_path.suffix.lower() == ".csv":
                try:
                    return pd.read_csv(file_path)
                except pd.errors.EmptyDataError:
                    self.logger.error(f"❌ CSV file is empty: {file_path.name}")
                    return None
                except pd.errors.ParserError as e:
                    self.logger.error(f"❌ CSV parsing error in {file_path.name}: {str(e)}. Check for malformed CSV data, incorrect delimiters, or encoding issues.")
                    return None
                except UnicodeDecodeError as e:
                    self.logger.error(f"❌ Encoding error in CSV file {file_path.name}: {str(e)}. Try specifying encoding (e.g., encoding='utf-8' or encoding='latin-1').")
                    return None
            elif file_path.suffix.lower() == ".parquet":
                try:
                    return pd.read_parquet(file_path)
                except Exception as e:
                    if "not a parquet file" in str(e).lower():
                        self.logger.error(f"❌ File {file_path.name} is not a valid Parquet file. Check if the file is corrupted or in a different format.")
                    else:
                        self.logger.error(f"❌ Parquet reading error in {file_path.name}: {str(e)}. Check if the file is corrupted or requires specific engine.")
                    return None
            elif file_path.suffix.lower() == ".json":
                try:
                    return pd.read_json(file_path)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON parsing error in {file_path.name}: {str(e)}. Check for malformed JSON syntax, missing brackets, or invalid characters.")
                    return None
                except ValueError as e:
                    if "arrays must all be same length" in str(e).lower():
                        self.logger.error(f"❌ JSON structure error in {file_path.name}: Arrays have different lengths. Check JSON data consistency.")
                    else:
                        self.logger.error(f"❌ JSON value error in {file_path.name}: {str(e)}. Check JSON data types and structure.")
                    return None
            else:
                self.logger.warning(f"⚠️ Unsupported file format: {file_path.suffix} for file {file_path.name}. Supported formats: .csv, .parquet, .json")
                return None
        except FileNotFoundError:
            self.logger.error(f"❌ File not found: {file_path.name}. Check if the file exists and the path is correct.")
            return None
        except PermissionError:
            self.logger.error(f"❌ Permission denied accessing {file_path.name}. Check file permissions.")
            return None
        except MemoryError:
            self.logger.error(f"❌ Memory error loading {file_path.name}. File may be too large for available memory.")
            return None
        except Exception as e:
            self.logger.exception(f"❌ Unexpected error loading {file_path.name}: {str(e)}. Check file integrity and format.")
            return None

    def _generate_directory_summary(self, summary_stats: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary for directory analysis."""
        # Calculate overall quality score
        quality_scores = summary_stats["overall_quality_scores"]
        if quality_scores:
            quality_counts = {}
            for score in quality_scores:
                quality_counts[score] = quality_counts.get(score, 0) + 1

            # Determine overall directory quality
            if "critical" in quality_counts:
                overall_quality = "critical"
            elif "poor" in quality_counts:
                overall_quality = "poor"
            elif "acceptable" in quality_counts:
                overall_quality = "acceptable"
            elif "good" in quality_counts:
                overall_quality = "good"
            else:
                overall_quality = "excellent"
        else:
            overall_quality = "unknown"
            quality_counts = {}

        # Generate directory-level recommendations
        directory_recommendations = list(summary_stats["recommendations"])

        # Add file-specific recommendations
        if summary_stats["failed_analyses"] > 0:
            directory_recommendations.append(f"Review {summary_stats['failed_analyses']} failed file analyses")

        if summary_stats["critical_issues_total"] > 0:
            directory_recommendations.append(f"Address {summary_stats['critical_issues_total']} critical issues across files")

        if summary_stats["overall_quality_scores"].count("poor") > len(summary_stats["overall_quality_scores"]) * 0.3:
            directory_recommendations.append("Many files have poor quality - consider systematic data cleaning")

        return {
            "overall_quality": overall_quality,
            "total_files": summary_stats["total_files"],
            "successful_analyses": summary_stats["successful_analyses"],
            "failed_analyses": summary_stats["failed_analyses"],
            "success_rate": summary_stats["successful_analyses"] / summary_stats["total_files"] if summary_stats["total_files"] > 0 else 0,
            "quality_distribution": quality_counts,
            "critical_issues_total": summary_stats["critical_issues_total"],
            "high_issues_total": summary_stats["high_issues_total"],
            "medium_issues_total": summary_stats["medium_issues_total"],
            "low_issues_total": summary_stats["low_issues_total"],
            "warnings_total": summary_stats["warnings_total"],
            "recommendations": directory_recommendations,
        }

    def analyze_file_batch(self, file_paths: list[str], parallel: bool = False) -> dict[str, Any]:
        """
        Analyze multiple files in batch.

        Args:
            file_paths: List of file paths to analyze
            parallel: Whether to process files in parallel (not yet implemented)

        Returns:
            Dictionary with batch analysis results
        """
        self.logger.info(f"🔍 Analyzing batch of {len(file_paths)} files")

        results = {}
        summary_stats = {
            "total_files": len(file_paths),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "overall_quality_scores": [],
            "critical_issues_total": 0,
            "high_issues_total": 0,
            "medium_issues_total": 0,
            "low_issues_total": 0,
            "warnings_total": 0,
            "recommendations": set(),
        }

        for file_path in file_paths:
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    self.logger.warning(f"File not found: {file_path}")
                    results[file_path] = {"error": "File not found"}
                    summary_stats["failed_analyses"] += 1
                    continue

                # Load and analyze data
                data = self._load_data_file(file_path_obj)
                if data is None:
                    summary_stats["failed_analyses"] += 1
                    continue

                # Generate report
                context = f"File: {file_path_obj.name}"
                report = self.generate_comprehensive_report(data, context)

                # Store results
                results[file_path] = report
                summary_stats["successful_analyses"] += 1

                # Aggregate statistics
                if report.get("summary"):
                    summary = report["summary"]
                    summary_stats["overall_quality_scores"].append(summary.get("overall_quality", "unknown"))

                    # Count issues
                    if report.get("quality_validation"):
                        quality_val = report["quality_validation"]
                        summary_stats["critical_issues_total"] += quality_val.get("issue_count", 0)

                    if report.get("temporal_validation"):
                        temp_val = report["temporal_validation"]
                        summary_stats["critical_issues_total"] += temp_val.get("issue_count", 0)

                    # Collect recommendations
                    if summary.get("recommendations"):
                        summary_stats["recommendations"].update(summary["recommendations"])

            except Exception as e:
                self.logger.exception(f"Failed to analyze {file_path}: {e}")
                results[file_path] = {"error": str(e)}
                summary_stats["failed_analyses"] += 1

        # Generate batch summary
        batch_summary = self._generate_directory_summary(summary_stats, results)

        return {
            "batch_analysis_timestamp": datetime.now().isoformat(),
            "file_paths": file_paths,
            "summary": batch_summary,
            "file_results": results,
        }

    def scan_directory_for_data_files(self, directory_path: str, recursive: bool = True) -> list[str]:
        """
        Scan a directory for data files.

        Args:
            directory_path: Path to directory to scan
            recursive: Whether to search subdirectories recursively

        Returns:
            List of data file paths found
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            msg = f"Path is not a directory: {directory_path}"
            raise ValueError(msg)

        # Supported file extensions
        supported_extensions = {".csv", ".parquet", ".json"}

        # Find files
        all_files = directory.rglob("*") if recursive else directory.glob("*")

        # Filter for data files
        data_files = [
            str(f) for f in all_files
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        return sorted(data_files)

    def get_directory_summary(self, directory_path: str, file_pattern: str = "*", recursive: bool = True) -> dict[str, Any]:
        """
        Get a quick summary of data files in a directory without full analysis.

        Args:
            directory_path: Path to directory
            file_pattern: Glob pattern for file matching
            recursive: Whether to search subdirectories recursively

        Returns:
            Quick directory summary
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            msg = f"Path is not a directory: {directory_path}"
            raise ValueError(msg)

        # Find data files
        data_files = self.scan_directory_for_data_files(directory_path, recursive)

        # Group by file type
        file_types = {}
        total_size = 0

        for file_path in data_files:
            file_path_obj = Path(file_path)
            file_type = file_path_obj.suffix.lower()
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0

            if file_type not in file_types:
                file_types[file_type] = {"count": 0, "total_size": 0}

            file_types[file_type]["count"] += 1
            file_types[file_type]["total_size"] += file_size
            total_size += file_size

        return {
            "directory_path": str(directory),
            "scan_timestamp": datetime.now().isoformat(),
            "file_pattern": file_pattern,
            "recursive": recursive,
            "total_files": len(data_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": file_types,
            "supported_formats": [".csv", ".parquet", ".json"],
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Unified Data Quality Orchestrator")
    parser.add_argument("--data_path", required=True, help="Path to data file or directory")
    parser.add_argument("--context", default="", help="Context description for the data")
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument("--thresholds", help="JSON file with custom thresholds")
    parser.add_argument("--mode", choices=["file", "directory", "auto"], default="auto",
                       help="Analysis mode: file, directory, or auto-detect")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="Search subdirectories recursively (for directory mode)")
    parser.add_argument("--file_pattern", default="*",
                       help="File pattern for directory analysis (e.g., '*.csv')")
    parser.add_argument("--quick_scan", action="store_true",
                       help="Quick directory scan without full analysis")

    args = parser.parse_args()

    # Load custom thresholds if provided
    thresholds = None
    if args.thresholds:
        try:
            with open(args.thresholds) as f:
                threshold_data = json.load(f)
                thresholds = QualityThresholds(**threshold_data)
        except Exception as e:
            tprint(f"Warning: Could not load custom thresholds: {e}")

    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator(thresholds)

    # Determine analysis mode
    data_path = Path(args.data_path)
    if args.mode == "auto":
        if data_path.is_file():
            mode = "file"
        elif data_path.is_dir():
            mode = "directory"
        else:
            tprint(f"Path not found: {data_path}")
            return
    else:
        mode = args.mode

    # Perform analysis based on mode
    if mode == "file":
        # Single file analysis
        if not data_path.is_file():
            tprint(f"Path is not a file: {data_path}")
            return

        if data_path.suffix.lower() not in [".csv", ".parquet", ".json"]:
            tprint(f"Unsupported file format: {data_path.suffix}")
            return

        tprint(f"📁 Analyzing single file: {data_path.name}")

        # Load data
        try:
            if data_path.suffix.lower() == ".csv":
                data = pd.read_csv(data_path)
            elif data_path.suffix.lower() == ".parquet":
                data = pd.read_parquet(data_path)
            elif data_path.suffix.lower() == ".json":
                data = pd.read_json(data_path)
        except Exception as e:
            tprint(f"Error loading file: {e}")
            return

        # Generate comprehensive report
        report = orchestrator.generate_comprehensive_report(data, args.context or f"File: {data_path.name}")

        # Save report
        if args.output:
            output_file = orchestrator.save_report(report, args.output)
        else:
            output_file = orchestrator.save_report(report)

        # Print summary
        summary = report.get("summary", {})
        tprint("\n📊 QUALITY REPORT SUMMARY")
        tprint(f"Overall Quality: {summary.get('overall_quality', 'unknown').upper()}")
        tprint(f"Critical Issues: {summary.get('critical_issues', 0)}")
        tprint(f"Recommendations: {len(summary.get('recommendations', []))}")
        tprint(f"Report saved to: {output_file}")

    elif mode == "directory":
        # Directory analysis
        if not data_path.is_dir():
            tprint(f"Path is not a directory: {data_path}")
            return

        tprint(f"📁 Analyzing directory: {data_path}")

        if args.quick_scan:
            # Quick directory scan
            tprint("🔍 Performing quick directory scan...")
            scan_summary = orchestrator.get_directory_summary(
                str(data_path),
                args.file_pattern,
                args.recursive,
            )

            tprint("\n📊 DIRECTORY SCAN SUMMARY")
            tprint(f"Directory: {scan_summary['directory_path']}")
            tprint(f"Total data files: {scan_summary['total_files']}")
            tprint(f"Total size: {scan_summary['total_size_mb']:.2f} MB")
            tprint("File types:")
            for file_type, info in scan_summary["file_types"].items():
                tprint(f"  - {file_type}: {info['count']} files ({info['total_size'] / (1024*1024):.2f} MB)")

            # Save scan summary
            if args.output:
                output_file = args.output
            else:
                output_file = f"directory_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(output_file, "w") as f:
                json.dump(scan_summary, f, indent=2, default=str)

            tprint(f"Scan summary saved to: {output_file}")

        else:
            # Full directory analysis
            tprint("🔍 Performing full directory analysis...")
            directory_report = orchestrator.analyze_directory(
                str(data_path),
                args.file_pattern,
                args.recursive,
            )

            if "error" in directory_report:
                tprint(f"❌ Directory analysis failed: {directory_report['error']}")
                return

            # Print directory summary
            summary = directory_report.get("summary", {})
            tprint("\n📊 DIRECTORY QUALITY SUMMARY")
            tprint(f"Directory: {directory_report['directory_path']}")
            tprint(f"Total files: {summary['total_files']}")
            tprint(f"Successful analyses: {summary['successful_analyses']}")
            tprint(f"Failed analyses: {summary['failed_analyses']}")
            tprint(f"Success rate: {summary['success_rate']:.1%}")
            tprint(f"Overall Quality: {summary['overall_quality'].upper()}")
            tprint(f"Critical Issues Total: {summary['critical_issues_total']}")

            if summary.get("quality_distribution"):
                tprint("Quality Distribution:")
                for quality, count in summary["quality_distribution"].items():
                    tprint(f"  - {quality.capitalize()}: {count} files")

            if summary.get("recommendations"):
                tprint("Recommendations:")
                for rec in summary["recommendations"][:5]:  # Show first 5
                    tprint(f"  - {rec}")

            # Save directory report
            if args.output:
                output_file = args.output
            else:
                output_file = f"directory_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            orchestrator.save_report(directory_report, output_file)
            tprint(f"Directory report saved to: {output_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
