"""
Enhanced Data Quality Validation Utilities

This module provides comprehensive data quality validation capabilities for the training pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from src.utils.logger import system_logger
except ImportError:
    system_logger = logging.getLogger("EnhancedDataQualityValidator")


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


class EnhancedDataQualityValidator:
    """Enhanced data quality validator with comprehensive checks."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.logger = system_logger.getChild("DataQualityValidator")
    
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
        
        # Check for constant features
        self._validate_constant_features(df, result)
        
        # Check for price anomalies (if OHLC columns exist)
        self._validate_price_anomalies(df, result)
        
        # Check for timestamp consistency
        self._validate_timestamp_consistency(df, result)
        
        # Check for data type issues
        self._validate_data_types(df, result)
        
        # Check for correlation issues
        self._validate_correlations(df, result)
        
        # Log results
        self._log_validation_results(result, context)
        
        return result
    
    def _validate_nan_values(self, df: pd.DataFrame, result: QualityResult):
        """Validate NaN values in DataFrame."""
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        nan_ratio = total_nans / (len(df) * len(df.columns)) if len(df) > 0 and len(df.columns) > 0 else 0
        
        result.add_metric("nan_count", total_nans)
        result.add_metric("nan_ratio", nan_ratio)
        result.add_metric("nan_by_column", nan_counts.to_dict())
        
        if nan_ratio > self.thresholds.max_nan_ratio:
            result.add_issue("nan_values", f"NaN ratio {nan_ratio:.4f} exceeds threshold {self.thresholds.max_nan_ratio}")
        
        # Check for columns with high NaN ratios
        high_nan_columns = nan_counts[nan_counts > len(df) * 0.1]  # More than 10% NaN
        if not high_nan_columns.empty:
            result.add_warning("high_nan_columns", f"Columns with >10% NaN: {list(high_nan_columns.index)}")
    
    def _validate_infinite_values(self, df: pd.DataFrame, result: QualityResult):
        """Validate infinite values in DataFrame."""
        infinite_counts = {}
        total_infinites = 0
        
        for col in df.select_dtypes(include=[np.number]).columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                infinite_counts[col] = infinite_count
                total_infinites += infinite_count
        
        result.add_metric("infinite_count", total_infinites)
        result.add_metric("infinite_columns", infinite_counts)
        
        if total_infinites > self.thresholds.max_infinite_count:
            result.add_issue("infinite_values", f"Found {total_infinites} infinite values in columns: {list(infinite_counts.keys())}")
    
    def _validate_constant_features(self, df: pd.DataFrame, result: QualityResult):
        """Validate constant features in DataFrame."""
        constant_features = []
        low_variance_features = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count < self.thresholds.min_unique_values:
                constant_features.append(col)
            elif unique_count < 5:  # Low variance warning
                low_variance_features.append(col)
        
        result.add_metric("constant_features", constant_features)
        result.add_metric("low_variance_features", low_variance_features)
        
        if constant_features:
            result.add_issue("constant_features", f"Found {len(constant_features)} constant features: {constant_features}")
        
        if low_variance_features:
            result.add_warning("low_variance_features", f"Found {len(low_variance_features)} low variance features: {low_variance_features}")
    
    def _validate_price_anomalies(self, df: pd.DataFrame, result: QualityResult):
        """Validate price anomalies in OHLC data."""
        price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        
        if not price_columns:
            return
        
        anomalies = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check for negative prices
            for col in price_columns:
                if row[col] < -self.thresholds.price_tolerance:
                    anomalies.append({
                        "row": i,
                        "column": col,
                        "value": row[col],
                        "type": "negative_price"
                    })
            
            # Check for OHLC consistency
            if all(col in price_columns for col in ['open', 'high', 'low', 'close']):
                if row['high'] < row['low']:
                    anomalies.append({
                        "row": i,
                        "type": "high_low_inversion",
                        "high": row['high'],
                        "low": row['low']
                    })
                
                if row['close'] > row['high'] or row['close'] < row['low']:
                    anomalies.append({
                        "row": i,
                        "type": "close_outside_range",
                        "close": row['close'],
                        "high": row['high'],
                        "low": row['low']
                    })
        
        result.add_metric("price_anomalies", anomalies)
        
        if anomalies:
            result.add_issue("price_anomalies", f"Found {len(anomalies)} price anomalies")
    
    def _validate_timestamp_consistency(self, df: pd.DataFrame, result: QualityResult):
        """Validate timestamp consistency."""
        if 'timestamp' not in df.columns:
            return
        
        issues = []
        
        try:
            # Convert timestamp to datetime if needed
            timestamps = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            
            # Check for invalid timestamps
            invalid_timestamps = timestamps.isna().sum()
            if invalid_timestamps > 0:
                issues.append({
                    "type": "invalid_timestamps",
                    "count": invalid_timestamps
                })
            
            # Check for gaps
            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps) > 1:
                expected_interval = pd.Timedelta(minutes=1)  # Assuming 1-minute data
                time_diffs = valid_timestamps.diff().dropna()
                
                large_gaps = time_diffs[time_diffs > expected_interval * 2]
                if not large_gaps.empty:
                    issues.append({
                        "type": "large_gaps",
                        "count": len(large_gaps),
                        "max_gap_minutes": large_gaps.max().total_seconds() / 60
                    })
            
            # Check for duplicates
            duplicates = valid_timestamps.duplicated()
            if duplicates.any():
                issues.append({
                    "type": "duplicate_timestamps",
                    "count": duplicates.sum()
                })
            
            # Check for future timestamps
            future_timestamps = valid_timestamps[valid_timestamps > pd.Timestamp.now(tz='UTC')]
            if not future_timestamps.empty:
                issues.append({
                    "type": "future_timestamps",
                    "count": len(future_timestamps)
                })
            
        except Exception as e:
            issues.append({
                "type": "timestamp_parsing_error",
                "error": str(e)
            })
        
        result.add_metric("timestamp_issues", issues)
        
        if issues:
            result.add_issue("timestamp_issues", f"Found {len(issues)} timestamp issues")
    
    def _validate_data_types(self, df: pd.DataFrame, result: QualityResult):
        """Validate data types in DataFrame."""
        issues = []
        
        # Check for mixed data types in columns
        for col in df.columns:
            try:
                # Try to infer the intended type
                if col in ['timestamp']:
                    if not pd.api.types.is_integer_dtype(df[col]):
                        issues.append({
                            "column": col,
                            "expected": "int64",
                            "actual": str(df[col].dtype)
                        })
                elif col in ['open', 'high', 'low', 'close', 'volume']:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        issues.append({
                            "column": col,
                            "expected": "numeric",
                            "actual": str(df[col].dtype)
                        })
            except Exception as e:
                issues.append({
                    "column": col,
                    "error": f"Type validation error: {e}"
                })
        
        result.add_metric("data_type_issues", issues)
        
        if issues:
            result.add_issue("data_type_issues", f"Found {len(issues)} data type issues")
    
    def _validate_correlations(self, df: pd.DataFrame, result: QualityResult):
        """Validate correlations between numeric columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return
        
        try:
            # Calculate correlations
            corr_matrix = df[numeric_columns].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds.max_correlation_threshold:
                        high_corr_pairs.append({
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": corr_value
                        })
            
            result.add_metric("high_correlations", high_corr_pairs)
            
            if high_corr_pairs:
                result.add_warning("high_correlations", f"Found {len(high_corr_pairs)} highly correlated column pairs")
        
        except Exception as e:
            result.add_warning("correlation_calculation_error", f"Could not calculate correlations: {e}")
    
    def _log_validation_results(self, result: QualityResult, context: str):
        """Log validation results."""
        status = "PASSED" if result.passed else "FAILED"
        self.logger.info(f"Quality validation for {context}: {status} ({len(result.issues)} issues, {len(result.warnings)} warnings)")
        
        if result.issues:
            for issue in result.issues[:3]:  # Log first 3 issues
                self.logger.warning(f"  - {issue}")
            if len(result.issues) > 3:
                self.logger.warning(f"  ... and {len(result.issues) - 3} more issues")
        
        if result.warnings:
            for warning in result.warnings[:3]:  # Log first 3 warnings
                self.logger.info(f"  - {warning}")
            if len(result.warnings) > 3:
                self.logger.info(f"  ... and {len(result.warnings) - 3} more warnings")


class UnifiedDataQualityValidator(EnhancedDataQualityValidator):
    """Specialized validator for unified data format."""
    
    def validate_unified_data_quality(self, df: pd.DataFrame, context: str = "") -> QualityResult:
        """Validate unified DataFrame quality with additional checks."""
        # Run base validation
        result = super().validate_dataframe_quality(df, context)
        
        # Add unified-specific validations
        self._validate_unified_structure(df, result)
        self._validate_data_consistency(df, result)
        
        return result
    
    def _validate_unified_structure(self, df: pd.DataFrame, result: QualityResult):
        """Validate unified data structure."""
        issues = []
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'exchange', 'symbol', 'timeframe']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append({
                "type": "missing_columns",
                "columns": missing_columns
            })
        
        # Check data types for required columns
        if 'timestamp' in df.columns and not pd.api.types.is_integer_dtype(df['timestamp']):
            issues.append({
                "type": "timestamp_dtype",
                "expected": "int64",
                "actual": str(df['timestamp'].dtype)
            })
        
        # Check for date columns
        date_columns = ['year', 'month', 'day']
        missing_date_columns = [col for col in date_columns if col not in df.columns]
        if missing_date_columns:
            issues.append({
                "type": "missing_date_columns",
                "columns": missing_date_columns
            })
        
        result.add_metric("unified_structure_issues", issues)
        
        if issues:
            result.add_issue("unified_structure", f"Found {len(issues)} unified structure issues")
    
    def _validate_data_consistency(self, df: pd.DataFrame, result: QualityResult):
        """Validate data consistency across exchanges/symbols."""
        issues = []
        
        # Check for consistent data across exchanges
        if 'exchange' in df.columns:
            exchange_counts = df['exchange'].value_counts()
            if len(exchange_counts) > 1:
                # Check if all exchanges have similar data volumes
                mean_count = exchange_counts.mean()
                std_count = exchange_counts.std()
                cv = std_count / mean_count if mean_count > 0 else 0
                
                if cv > 0.5:  # Coefficient of variation > 50%
                    issues.append({
                        "type": "uneven_exchange_distribution",
                        "exchange_counts": exchange_counts.to_dict(),
                        "coefficient_of_variation": cv
                    })
        
        # Check for consistent data across symbols
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            if len(symbol_counts) > 1:
                mean_count = symbol_counts.mean()
                std_count = symbol_counts.std()
                cv = std_count / mean_count if mean_count > 0 else 0
                
                if cv > 0.5:
                    issues.append({
                        "type": "uneven_symbol_distribution",
                        "symbol_counts": symbol_counts.to_dict(),
                        "coefficient_of_variation": cv
                    })
        
        result.add_metric("consistency_issues", issues)
        
        if issues:
            result.add_issue("data_consistency", f"Found {len(issues)} consistency issues")


# Convenience functions
def quick_validate_dataframe(df: pd.DataFrame, context: str = "") -> QualityResult:
    """Quick validation of DataFrame quality."""
    validator = EnhancedDataQualityValidator()
    return validator.validate_dataframe_quality(df, context)


def validate_unified_dataframe(df: pd.DataFrame, context: str = "") -> QualityResult:
    """Validate unified DataFrame quality."""
    validator = UnifiedDataQualityValidator()
    return validator.validate_unified_data_quality(df, context)


def check_dataframe_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick health check of DataFrame."""
    if df is None or df.empty:
        return {"healthy": False, "reason": "DataFrame is None or empty"}
    
    # Basic health checks
    nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 and len(df.columns) > 0 else 0
    infinite_count = sum(np.isinf(df[col]).sum() for col in df.select_dtypes(include=[np.number]).columns)
    
    health_status = {
        "healthy": True,
        "shape": df.shape,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "nan_ratio": nan_ratio,
        "infinite_count": infinite_count,
        "issues": []
    }
    
    if nan_ratio > 0.1:  # More than 10% NaN
        health_status["healthy"] = False
        health_status["issues"].append("High NaN ratio")
    
    if infinite_count > 0:
        health_status["healthy"] = False
        health_status["issues"].append("Infinite values present")
    
    return health_status
