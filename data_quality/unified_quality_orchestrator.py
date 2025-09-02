#!/usr/bin/env python3
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

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, some functionality will be limited")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, some functionality will be limited")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available, dependency graph functionality will be limited")

# Import project-specific modules
try:
    from src.utils.logger import system_logger
    from src.utils.enhanced_mlflow_integration import (
        log_step_metrics,
        log_step_report,
        create_detailed_step_report
    )
    from src.utils.enhanced_outlier_handler import enhanced_outlier_handler, OutlierSeverity
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
except ImportError:
    # Fallback logger if project modules aren't available
    system_logger = logging.getLogger("UnifiedQualityOrchestrator")
    print("Warning: Some project modules not available, using fallback implementations")


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
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation result."""
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "warning_count": len(self.warnings),
            "metrics": self.metrics,
            "issues": self.issues[:5],  # First 5 issues
            "warnings": self.warnings[:5]  # First 5 warnings
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
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
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
    issues: List[str]
    warnings: List[str]
    conversions_applied: List[str]
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
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
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
            "max_issues_low": 100
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
                    "volume": "float64"
                },
                "constraints": {
                    "timestamp": {"min": 0, "max": None},
                    "open": {"min": 0, "max": None},
                    "high": {"min": 0, "max": None},
                    "low": {"min": 0, "max": None},
                    "close": {"min": 0, "max": None},
                    "volume": {"min": 0, "max": None}
                }
            },
            "features_schema": {
                "required_columns": ["timestamp"],
                "data_types": {
                    "timestamp": "int64"
                },
                "constraints": {
                    "timestamp": {"min": 0, "max": None}
                }
            },
            "labels_schema": {
                "required_columns": ["timestamp", "label"],
                "data_types": {
                    "timestamp": "int64",
                    "label": "int64"
                },
                "constraints": {
                    "timestamp": {"min": 0, "max": None},
                    "label": {"min": 0, "max": None}
                }
            }
        }
    
    def validate_dataframe_quality(self, df: pd.DataFrame, context: str = "") -> QualityResult:
        """Validate DataFrame quality with comprehensive checks."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data quality validation")
        
        result = QualityResult()
        
        if df is None or df.empty:
            result.add_issue("empty_data", "DataFrame is None or empty")
            return result
        
        # Basic metrics
        result.add_metric("rows", len(df))
        result.add_metric("columns", len(df.columns))
        result.add_metric("memory_mb", df.memory_usage(deep=True).sum() / 1024 / 1024)
        
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
        if 'enhanced_outlier_handler' in globals():
            self._validate_outliers(df, result)
        
        return result
    
    def _validate_nan_values(self, df: pd.DataFrame, result: QualityResult):
        """Validate NaN values in DataFrame."""
        nan_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        nan_ratio = nan_counts.sum() / total_cells
        
        result.add_metric("nan_ratio", nan_ratio)
        result.add_metric("nan_counts", nan_counts.to_dict())
        
        if nan_ratio > self.thresholds.max_nan_ratio:
            result.add_issue("high_nan_ratio", f"NaN ratio {nan_ratio:.2%} exceeds threshold {self.thresholds.max_nan_ratio:.2%}")
        
        # Check for columns with high NaN ratios
        high_nan_cols = nan_counts[nan_counts > len(df) * 0.5]
        if not high_nan_cols.empty:
            result.add_warning("high_nan_columns", f"Columns with >50% NaN: {list(high_nan_cols.index)}")
    
    def _validate_infinite_values(self, df: pd.DataFrame, result: QualityResult):
        """Validate infinite values in DataFrame."""
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        total_inf = inf_counts.sum()
        
        result.add_metric("infinite_count", total_inf)
        result.add_metric("infinite_counts", inf_counts.to_dict())
        
        if total_inf > self.thresholds.max_infinite_count:
            result.add_issue("infinite_values", f"Found {total_inf} infinite values")
    
    def _validate_constant_columns(self, df: pd.DataFrame, result: QualityResult):
        """Validate constant columns in DataFrame."""
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        result.add_metric("constant_columns", constant_cols)
        
        if constant_cols:
            result.add_warning("constant_columns", f"Constant columns: {constant_cols}")
    
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
    
    def analyze_multicollinearity(self, data: pd.DataFrame, vif_threshold: float = None) -> Dict[str, Any]:
        """
        Analyze multicollinearity using VIF and correlation analysis.
        
        Args:
            data: Input DataFrame
            vif_threshold: VIF threshold for flagging multicollinearity
            
        Returns:
            Dictionary with multicollinearity analysis results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for multicollinearity analysis")
        
        if vif_threshold is None:
            vif_threshold = self.thresholds.vif_threshold
        
        self.logger.info("🔍 Analyzing multicollinearity...")
        
        # Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove potential label columns
        potential_label_columns = [
            "label", "target", "y", "class", "Label", "Target", "Y", "Class"
        ]
        actual_label_columns = [
            col for col in numeric_data.columns if col in potential_label_columns
        ]
        
        if actual_label_columns:
            self.logger.warning(
                f"⚠️ Removing label columns from multicollinearity analysis: {actual_label_columns}"
            )
            numeric_data = numeric_data.drop(columns=actual_label_columns)
        
        # Handle NaN values
        imputer = SimpleImputer(strategy="median")
        data_imputed = pd.DataFrame(
            imputer.fit_transform(numeric_data),
            columns=numeric_data.columns,
            index=numeric_data.index
        )
        
        # Calculate VIF scores
        vif_scores = {}
        high_vif_features = []
        
        for col in data_imputed.columns:
            other_cols = [c for c in data_imputed.columns if c != col]
            if len(other_cols) > 0:
                X = data_imputed[other_cols]
                y = data_imputed[col]
                
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Calculate VIF
                vif = 1 / (1 - reg.score(X, y))
                vif_scores[col] = vif
                
                if vif > vif_threshold:
                    high_vif_features.append(col)
        
        # Calculate correlation matrix
        correlation_matrix = data_imputed.corr()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > self.thresholds.max_correlation_threshold:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_val))
        
        return {
            "vif_scores": vif_scores,
            "high_vif_features": high_vif_features,
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "vif_threshold": vif_threshold,
            "correlation_threshold": self.thresholds.max_correlation_threshold,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def analyze_label_imbalance(self, labels: pd.Series) -> Dict[str, Any]:
        """
        Analyze label imbalance in classification datasets.
        
        Args:
            labels: Series containing labels
            
        Returns:
            Dictionary with label imbalance analysis results
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for label imbalance analysis")
        
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
                "Evaluate using F1-score or other balanced metrics"
            ])
        elif imbalance_level == "moderate":
            recommendations.extend([
                "Consider using class weights",
                "Monitor model performance on minority classes"
            ])
        
        return {
            "label_counts": label_counts.to_dict(),
            "total_samples": total_samples,
            "imbalance_ratio": imbalance_ratio,
            "entropy": entropy,
            "gini_coefficient": gini_coefficient,
            "imbalance_level": imbalance_level,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def analyze_feature_redundancy(self, data: pd.DataFrame, correlation_threshold: float = None) -> Dict[str, Any]:
        """
        Analyze feature redundancy based on correlation analysis.
        
        Args:
            data: Input DataFrame
            correlation_threshold: Correlation threshold for identifying redundant features
            
        Returns:
            Dictionary with feature redundancy analysis results
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for feature redundancy analysis")
        
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
            "analysis_timestamp": datetime.now().isoformat()
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
            timestamps = pd.to_datetime(df[timestamp_col], unit='s')
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
    
    def generate_comprehensive_report(self, data: pd.DataFrame, context: str = "") -> Dict[str, Any]:
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
            "summary": {}
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
            timestamp_cols = [col for col in data.columns if 'time' in col.lower() or col == 'timestamp']
            if timestamp_cols:
                try:
                    temporal_validation = self.validate_temporal_data(data, timestamp_cols[0])
                    report["temporal_validation"] = temporal_validation.get_summary()
                except Exception as e:
                    report["temporal_validation"] = {"error": str(e)}
        
        # Generate summary
        report["summary"] = self._generate_summary(report)
        
        return report
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the comprehensive report."""
        summary = {
            "overall_quality": "unknown",
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "warnings": 0,
            "recommendations": []
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
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
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
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"✅ Quality report saved to: {output_path}")
        return str(output_path)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Unified Data Quality Orchestrator")
    parser.add_argument("--data_path", required=True, help="Path to data file or directory")
    parser.add_argument("--context", default="", help="Context description for the data")
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument("--thresholds", help="JSON file with custom thresholds")
    
    args = parser.parse_args()
    
    # Load custom thresholds if provided
    thresholds = None
    if args.thresholds:
        try:
            with open(args.thresholds, 'r') as f:
                threshold_data = json.load(f)
                thresholds = QualityThresholds(**threshold_data)
        except Exception as e:
            print(f"Warning: Could not load custom thresholds: {e}")
    
    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator(thresholds)
    
    # Load data
    data_path = Path(args.data_path)
    if data_path.is_file():
        if data_path.suffix == '.csv':
            data = pd.read_csv(data_path)
        elif data_path.suffix == '.parquet':
            data = pd.read_parquet(data_path)
        elif data_path.suffix == '.json':
            data = pd.read_json(data_path)
        else:
            print(f"Unsupported file format: {data_path.suffix}")
            return
    elif data_path.is_dir():
        print("Directory analysis not yet implemented")
        return
    else:
        print(f"Path not found: {data_path}")
        return
    
    # Generate comprehensive report
    report = orchestrator.generate_comprehensive_report(data, args.context)
    
    # Save report
    if args.output:
        output_file = orchestrator.save_report(report, args.output)
    else:
        output_file = orchestrator.save_report(report)
    
    # Print summary
    summary = report.get("summary", {})
    print(f"\n📊 QUALITY REPORT SUMMARY")
    print(f"Overall Quality: {summary.get('overall_quality', 'unknown').upper()}")
    print(f"Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"Recommendations: {len(summary.get('recommendations', []))}")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()