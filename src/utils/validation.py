"""
Unified Validation System - Enhanced Data Validation Framework

This module provides comprehensive data validation functionality consolidated
from multiple validation utilities, including HMM statistical validation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import warnings

from .error_handler import UnifiedErrorHandler, ValidationError, DataQualityError

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

class ValidationConfig:
    """Configuration for validation system."""

    def __init__(self,
                 strict_mode: bool = True,
                 log_warnings: bool = True,
                 auto_fix: bool = False,
                 max_missing_percentage: float = 50.0,
                 max_duplicate_percentage: float = 10.0):
        self.strict_mode = strict_mode
        self.log_warnings = log_warnings
        self.auto_fix = auto_fix
        self.max_missing_percentage = max_missing_percentage
        self.max_duplicate_percentage = max_duplicate_percentage

# =============================================================================
# UNIFIED VALIDATOR
# =============================================================================

class UnifiedValidator:
    """Unified validator that consolidates all validation functionality."""

    def __init__(self, config: ValidationConfig = None, error_handler: UnifiedErrorHandler = None):
        self.config = config or ValidationConfig()
        self.error_handler = error_handler or UnifiedErrorHandler()
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}

    def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Any] = None,
                          required_columns: List[str] = None,
                          min_rows: int = 0) -> Dict[str, Any]:
        """Comprehensive DataFrame validation."""
        results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'metrics': {},
            'suggestions': []
        }

        try:
            # Basic structure validation
            if df is None:
                results['valid'] = False
                results['issues'].append("DataFrame is None")
                return results

            if df.empty:
                results['valid'] = False
                results['issues'].append("DataFrame is empty")
                return results

            if min_rows > 0 and len(df) < min_rows:
                results['valid'] = False
                results['issues'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

            # Column validation
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    results['valid'] = False
                    results['issues'].append(f"Missing required columns: {missing_columns}")

            # Schema validation
            if schema:
                schema_results = self._validate_schema(df, schema)
                results['issues'].extend(schema_results['issues'])
                results['warnings'].extend(schema_results['warnings'])
                if not schema_results['valid']:
                    results['valid'] = False

            # Data quality metrics
            quality_metrics = self._calculate_quality_metrics(df)
            results['metrics'] = quality_metrics

            # Check quality thresholds
            if quality_metrics['missing_percentage'] > self.config.max_missing_percentage:
                results['warnings'].append(f"High missing data percentage: {quality_metrics['missing_percentage']:.1f}%")
                if self.config.strict_mode:
                    results['valid'] = False

            if quality_metrics['duplicate_percentage'] > self.config.max_duplicate_percentage:
                results['warnings'].append(f"High duplicate percentage: {quality_metrics['duplicate_percentage']:.1f}%")
                if self.config.strict_mode:
                    results['valid'] = False

            # Generate suggestions
            results['suggestions'] = self._generate_suggestions(df, quality_metrics)

        except Exception as e:
            self.error_handler.handle_error(e, "DataFrame validation")
            results['valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")

        return results

    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DataFrame against schema."""
        results = {'valid': True, 'issues': [], 'warnings': []}

        for column, rules in schema.items():
            if column not in df.columns:
                results['issues'].append(f"Schema column '{column}' not found in DataFrame")
                results['valid'] = False
                continue

            column_data = df[column]

            # Data type validation
            if 'dtype' in rules:
                expected_dtype = rules['dtype']
                if not pd.api.types.is_dtype_equal(column_data.dtype, expected_dtype):
                    results['warnings'].append(f"Column '{column}' has dtype {column_data.dtype}, expected {expected_dtype}")

            # Null validation
            if 'nullable' in rules and not rules['nullable']:
                null_count = column_data.isnull().sum()
                if null_count > 0:
                    results['issues'].append(f"Column '{column}' has {null_count} null values but nullable=False")
                    results['valid'] = False

            # Range validation
            if 'min_value' in rules or 'max_value' in rules:
                if pd.api.types.is_numeric_dtype(column_data):
                    min_val = rules.get('min_value')
                    max_val = rules.get('max_value')

                    if min_val is not None:
                        below_min = (column_data < min_val).sum()
                        if below_min > 0:
                            results['issues'].append(f"Column '{column}' has {below_min} values below minimum {min_val}")
                            results['valid'] = False

                    if max_val is not None:
                        above_max = (column_data > max_val).sum()
                        if above_max > 0:
                            results['issues'].append(f"Column '{column}' has {above_max} values above maximum {max_val}")
                            results['valid'] = False

            # Unique validation
            if 'unique' in rules and rules['unique']:
                duplicate_count = column_data.duplicated().sum()
                if duplicate_count > 0:
                    results['issues'].append(f"Column '{column}' has {duplicate_count} duplicate values but unique=True")
                    results['valid'] = False

        return results

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        try:
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()

            metrics = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'missing_percentage': (missing_cells / total_cells * 100) if total_cells > 0 else 0,
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / len(df) * 100) if len(df) > 0 else 0,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }

            # Column-specific metrics
            column_metrics = {}
            for col in df.columns:
                col_data = df[col]
                column_metrics[col] = {
                    'dtype': str(col_data.dtype),
                    'null_count': col_data.isnull().sum(),
                    'null_percentage': (col_data.isnull().sum() / len(col_data) * 100) if len(col_data) > 0 else 0,
                    'unique_count': col_data.nunique(),
                    'unique_percentage': (col_data.nunique() / len(col_data) * 100) if len(col_data) > 0 else 0
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    column_metrics[col].update({
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'zero_count': (col_data == 0).sum(),
                        'negative_count': (col_data < 0).sum(),
                        'infinite_count': np.isinf(col_data).sum()
                    })

            metrics['column_metrics'] = column_metrics
            return metrics

        except Exception as e:
            self.error_handler.handle_error(e, "Quality metrics calculation")
            return {}

    def _generate_suggestions(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for data improvement."""
        suggestions = []

        # Missing data suggestions
        if metrics.get('missing_percentage', 0) > 10:
            suggestions.append("Consider imputation strategies for missing data")

        # Duplicate data suggestions
        if metrics.get('duplicate_percentage', 0) > 5:
            suggestions.append("Review and remove duplicate rows if appropriate")

        # Memory optimization suggestions
        if metrics.get('memory_usage', 0) > 100 * 1024 * 1024:  # 100MB
            suggestions.append("Consider optimizing data types to reduce memory usage")

        # Column-specific suggestions
        for col, col_metrics in metrics.get('column_metrics', {}).items():
            if col_metrics.get('null_percentage', 0) > 50:
                suggestions.append(f"Column '{col}' has high missing data - consider dropping or imputing")

            if col_metrics.get('unique_percentage', 0) < 5:
                suggestions.append(f"Column '{col}' has low uniqueness - consider if it's needed")

        return suggestions

    def validate_numeric_data(self, data: Union[pd.Series, np.ndarray, List],
                            name: str = "numeric_data") -> Dict[str, Any]:
        """Validate numeric data."""
        results = {'valid': True, 'issues': [], 'warnings': []}

        try:
            if isinstance(data, (list, tuple)):
                data = np.array(data)
            elif isinstance(data, pd.Series):
                data = data.values

            # Check for non-finite values
            non_finite_count = np.sum(~np.isfinite(data))
            if non_finite_count > 0:
                results['issues'].append(f"{name} contains {non_finite_count} non-finite values")
                results['valid'] = False

            # Check for extreme values
            if len(data) > 0:
                q99 = np.percentile(data[np.isfinite(data)], 99)
                q01 = np.percentile(data[np.isfinite(data)], 1)
                extreme_count = np.sum((data > q99 * 10) | (data < q01 * 10))
                if extreme_count > 0:
                    results['warnings'].append(f"{name} contains {extreme_count} extreme values")

        except Exception as e:
            self.error_handler.handle_error(e, f"Numeric data validation: {name}")
            results['valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")

        return results

    def validate_timestamp_data(self, timestamps: Union[pd.Series, List],
                              name: str = "timestamps") -> Dict[str, Any]:
        """Validate timestamp data."""
        results = {'valid': True, 'issues': [], 'warnings': []}

        try:
            if isinstance(timestamps, list):
                timestamps = pd.Series(timestamps)

            # Convert to datetime
            try:
                dt_series = pd.to_datetime(timestamps)
            except Exception as e:
                results['valid'] = False
                results['issues'].append(f"Invalid timestamp format: {str(e)}")
                return results

            # Check for null timestamps
            null_count = dt_series.isnull().sum()
            if null_count > 0:
                results['issues'].append(f"{name} contains {null_count} null timestamps")
                results['valid'] = False

            # Check for duplicate timestamps
            duplicate_count = dt_series.duplicated().sum()
            if duplicate_count > 0:
                results['warnings'].append(f"{name} contains {duplicate_count} duplicate timestamps")

            # Check for reasonable date range
            if len(dt_series) > 0:
                min_date = dt_series.min()
                max_date = dt_series.max()

                if min_date < datetime(1900, 1, 1):
                    results['warnings'].append(f"{name} contains dates before 1900")

                if max_date > datetime.now() + timedelta(days=365):
                    results['warnings'].append(f"{name} contains future dates beyond 1 year")

        except Exception as e:
            self.error_handler.handle_error(e, f"Timestamp validation: {name}")
            results['valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")

        return results

    def validate_correlation_matrix(self, corr_matrix: np.ndarray,
                                  name: str = "correlation_matrix") -> Dict[str, Any]:
        """Validate correlation matrix."""
        results = {'valid': True, 'issues': [], 'warnings': []}

        try:
            # Check shape
            if corr_matrix.ndim != 2:
                results['valid'] = False
                results['issues'].append(f"{name} must be 2-dimensional")
                return results

            if corr_matrix.shape[0] != corr_matrix.shape[1]:
                results['valid'] = False
                results['issues'].append(f"{name} must be square")
                return results

            # Check diagonal values
            diagonal_values = np.diag(corr_matrix)
            if not np.allclose(diagonal_values, 1.0):
                results['issues'].append(f"{name} diagonal values are not all 1.0")
                results['valid'] = False

            # Check value range
            if not np.all((corr_matrix >= -1) & (corr_matrix <= 1)):
                results['valid'] = False
                results['issues'].append(f"{name} contains values outside [-1, 1] range")

            # Check symmetry
            if not np.allclose(corr_matrix, corr_matrix.T):
                results['warnings'].append(f"{name} is not symmetric")

        except Exception as e:
            self.error_handler.handle_error(e, f"Correlation matrix validation: {name}")
            results['valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        return {
            'total_validations': len(self.validation_results),
            'successful_validations': sum(1 for r in self.validation_results.values() if r.get('valid', False)),
            'failed_validations': sum(1 for r in self.validation_results.values() if not r.get('valid', False)),
            'recent_results': list(self.validation_results.items())[-10:]
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_dataframe(df: pd.DataFrame, schema: Dict[str, Any] = None,
                      required_columns: List[str] = None,
                      min_rows: int = 0) -> Dict[str, Any]:
    """Validate DataFrame using unified validator."""
    validator = UnifiedValidator()
    return validator.validate_dataframe(df, schema, required_columns, min_rows)

def validate_numeric_data(data: Union[pd.Series, np.ndarray, List],
                        name: str = "numeric_data") -> Dict[str, Any]:
    """Validate numeric data using unified validator."""
    validator = UnifiedValidator()
    return validator.validate_numeric_data(data, name)

def validate_timestamp_data(timestamps: Union[pd.Series, List],
                          name: str = "timestamps") -> Dict[str, Any]:
    """Validate timestamp data using unified validator."""
    validator = UnifiedValidator()
    return validator.validate_timestamp_data(timestamps, name)

def validate_correlation_matrix(corr_matrix: np.ndarray,
                              name: str = "correlation_matrix") -> Dict[str, Any]:
    """Validate correlation matrix using unified validator."""
    validator = UnifiedValidator()
    return validator.validate_correlation_matrix(corr_matrix, name)

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_unified_validator: Optional[UnifiedValidator] = None

def get_unified_validator() -> UnifiedValidator:
    """Get the global unified validator."""
    global _unified_validator
    if _unified_validator is None:
        _unified_validator = UnifiedValidator()
    return _unified_validator

def validate_data_quality(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Validate data quality using unified validator."""
    validator = UnifiedValidator()
    return validator.validate_dataframe(df, **kwargs)

def setup_unified_validation(config: ValidationConfig = None) -> UnifiedValidator:
    """Setup unified validation system."""
    global _unified_validator
    _unified_validator = UnifiedValidator(config)
    return _unified_validator

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize the unified validator by default
if _unified_validator is None:
    setup_unified_validation()
