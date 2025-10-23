"""
Data Validation Module

This module provides comprehensive data validation for clustering operations including
schema validation, NaN/inf checks, and Population Stability Index (PSI) drift detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import warnings
from datetime import datetime, timedelta

# Import utility modules
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    format_bytes, chunked_iterable, parallel_map, timed_operation,
    get_current_datetime, format_datetime, parse_datetime,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    get_logger, integrate_with_m1_optimizers, cleanup_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    is_m1_available, is_mps_available, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, validate_file_path, get_file_size,
    check_disk_space, safe_rolling, safe_groupby_operation, safe_apply_function,
    safe_filter_dataframe, create_summary_statistics, safe_resample,
    align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
    sanitize_string, math_safe, validate_correlation_matrix, safe_matrix_inverse,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    calculate_data_quality_metrics, get_dataframe_info, create_data_quality_report
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

from src.utils.math_validation import (
    MathValidationError, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array as math_validate_numeric_array
)

# Import hardware utilities
try:
    from src.utils.hardware.m1_gpu_utils import is_m1_available as hw_is_m1_available, is_mps_available as hw_is_mps_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer as hw_get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as hw_get_m1_cpu_optimizer
except ImportError:
    hw_is_m1_available = lambda: False
    hw_is_mps_available = lambda: False
    hw_get_m1_memory_optimizer = lambda: None
    hw_get_m1_cpu_optimizer = lambda: None

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.validation.validation_utils import (
        ConfigurationValidator, DataValidator as MLDataValidator, ResourceValidator,
        ExecutionValidator, ResultValidator
    )
    # CVLSA PerformanceAnalytics removed - no longer available
    PerformanceAnalytics = None
    from src.utils.ml_common.optimization.shared_utils.integration_verification import SharedUtilsIntegrationVerifier
except ImportError:
    BayesianTPEOptimizer = None
    ConfigurationValidator = None
    MLDataValidator = None
    ResourceValidator = None
    ExecutionValidator = None
    ResultValidator = None
    PerformanceAnalytics = None
    SharedUtilsIntegrationVerifier = None

logger = logging.getLogger(__name__)

def calculate_data_quality_score(df: pd.DataFrame, data_quality: Dict[str, Any]) -> float:
    """Calculate an overall data quality score from 0-100."""
    try:
        score = 100.0

        # Penalize missing values
        missing_ratio = data_quality.get('missing_percentage', 0)
        if missing_ratio > 0:
            score -= min(50, missing_ratio * 2)  # Up to 50 points for missing data

        # Penalize duplicate rows
        duplicate_ratio = data_quality.get('duplicate_percentage', 0)
        if duplicate_ratio > 0:
            score -= min(20, duplicate_ratio * 0.5)  # Up to 20 points for duplicates

        # Penalize lack of numeric columns
        numeric_ratio = safe_divide(data_quality.get('numeric_columns', 0), data_quality.get('total_columns', 1), 0.0)
        if numeric_ratio < 0.5:
            score -= (0.5 - numeric_ratio) * 20  # Up to 20 points for insufficient numeric data

        return max(0.0, score)

    except Exception as e:
        logger.warning(f"Failed to calculate data quality score: {e}")
        return 50.0  # Default moderate score

class DataValidator:
    """Comprehensive data validation system for clustering operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize data validator."""
        self.logger = logger or logging.getLogger(__name__)
        self.validation_history: List[Dict[str, Any]] = []
        self.psi_baselines: Dict[str, Dict[str, Any]] = {}
        self.schema_baselines: Dict[str, Dict[str, Any]] = {}

        # PSI thresholds
        self.psi_warning_threshold = 0.1
        self.psi_critical_threshold = 0.25

        # Data quality thresholds
        self.max_nan_ratio = 0.1  # 10% max NaN ratio per column
        self.max_inf_ratio = 0.01  # 1% max inf ratio per column
        self.min_feature_count = 2
        self.max_feature_count = 10000

    def validate_schema(self, data: Union[pd.DataFrame, np.ndarray],
                       expected_schema: Optional[Dict[str, Any]] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """Validate data schema against expected structure."""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'schema_info': {},
                'timestamp': datetime.now().isoformat()
            }

            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                if data.ndim != 2:
                    validation_result['errors'].append(f"Array must be 2-dimensional, got {data.ndim}D")
                    validation_result['is_valid'] = False
                    return validation_result

                # Create DataFrame with generic column names
                df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
            else:
                df = data.copy()

            # Basic schema information
            validation_result['schema_info'] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'index_type': str(df.index.dtype) if hasattr(df.index, 'dtype') else 'unknown',
                'memory_usage': df.memory_usage(deep=True).sum()
            }

            # Check feature count
            n_features = df.shape[1]
            if n_features < self.min_feature_count:
                error_msg = f"Too few features: {n_features} < {self.min_feature_count}"
                validation_result['errors'].append(error_msg)
                validation_result['is_valid'] = False

            if n_features > self.max_feature_count:
                warning_msg = f"High number of features: {n_features} > {self.max_feature_count}"
                validation_result['warnings'].append(warning_msg)

            # Validate expected schema if provided
            if expected_schema:
                schema_errors = self._validate_against_expected_schema(df, expected_schema, strict)
                validation_result['errors'].extend(schema_errors['errors'])
                validation_result['warnings'].extend(schema_errors['warnings'])

                if schema_errors['errors']:
                    validation_result['is_valid'] = False

            # Check for problematic column names
            problematic_columns = []
            for col in df.columns:
                if not isinstance(col, str):
                    problematic_columns.append(str(col))
                elif not col.replace('_', '').replace('-', '').isalnum():
                    problematic_columns.append(col)

            if problematic_columns:
                warning_msg = f"Problematic column names found: {problematic_columns}"
                validation_result['warnings'].append(warning_msg)

            # Record validation
            self.validation_history.append({
                'validation_type': 'schema',
                'result': validation_result,
                'timestamp': datetime.now()
            })

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate schema: {e}")
            return {
                'is_valid': False,
                'errors': [f"Schema validation failed: {str(e)}"],
                'warnings': [],
                'schema_info': {},
                'timestamp': datetime.now().isoformat()
            }

    def _validate_against_expected_schema(self, df: pd.DataFrame,
                                        expected_schema: Dict[str, Any],
                                        strict: bool) -> Dict[str, List[str]]:
        """Validate DataFrame against expected schema."""
        errors = []
        warnings = []

        # Check required columns
        expected_columns = expected_schema.get('required_columns', [])
        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            if strict:
                errors.append(f"Missing required columns: {list(missing_columns)}")
            else:
                warnings.append(f"Missing optional columns: {list(missing_columns)}")

        # Check column types if specified
        expected_types = expected_schema.get('column_types', {})
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)

                # Type compatibility check
                if not self._types_compatible(actual_type, expected_type):
                    if strict:
                        errors.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")
                    else:
                        warnings.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")

        # Check expected shape if specified
        expected_shape = expected_schema.get('expected_shape')
        if expected_shape:
            if len(expected_shape) == 2:
                expected_rows, expected_cols = expected_shape
                if df.shape[0] != expected_rows:
                    warnings.append(f"Row count mismatch: expected {expected_rows}, got {df.shape[0]}")
                if df.shape[1] != expected_cols:
                    if strict:
                        errors.append(f"Column count mismatch: expected {expected_cols}, got {df.shape[1]}")
                    else:
                        warnings.append(f"Column count mismatch: expected {expected_cols}, got {df.shape[1]}")

        return {'errors': errors, 'warnings': warnings}

    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual type is compatible with expected type."""
        # Simple type compatibility mapping
        compatibility_map = {
            'int64': ['int', 'numeric', 'number'],
            'float64': ['float', 'numeric', 'number'],
            'object': ['string', 'category', 'object'],
            'datetime64[ns]': ['datetime', 'date'],
            'bool': ['boolean', 'bool']
        }

        for base_type, compatible_types in compatibility_map.items():
            if actual_type == base_type:
                return expected_type.lower() in compatible_types

        return False

    def check_data_quality(self, data: Union[pd.DataFrame, np.ndarray],
                          fail_hard: bool = True) -> Dict[str, Any]:
        """Check data quality including NaN, inf, and other issues."""
        try:
            tprint_info("Performing comprehensive data quality check")
            quality_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'quality_metrics': {},
                'timestamp': format_datetime(get_current_datetime(), "%Y-%m-%d %H:%M:%S")
            }

            # Convert to DataFrame if needed using common operations
            if isinstance(data, np.ndarray):
                math_validate_numeric_array(data, "input_data")
                if data.ndim != 2:
                    error_msg = f"Array must be 2-dimensional, got {data.ndim}D"
                    quality_result['errors'].append(error_msg)
                    quality_result['is_valid'] = False
                    tprint_error(error_msg)
                    return quality_result
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Use common operations for data quality metrics
            total_cells = df.shape[0] * df.shape[1]
            data_quality = calculate_data_quality_metrics(df)

            # Check for NaN values using safe operations
            nan_counts = df.isnull().sum()
            nan_ratios = safe_divide(nan_counts, df.shape[0], 0.0)

            nan_issues = []
            for col, ratio in nan_ratios.items():
                if ratio > 0:
                    if ratio > self.max_nan_ratio:
                        if fail_hard:
                            error_msg = f"Column '{col}' has {ratio:.2%} NaN values (>{self.max_nan_ratio:.2%})"
                            nan_issues.append(error_msg)
                            tprint_error(error_msg)
                        else:
                            warning_msg = f"Column '{col}' has {ratio:.2%} NaN values"
                            quality_result['warnings'].append(warning_msg)
                            tprint_warning(warning_msg)
                    else:
                        warning_msg = f"Column '{col}' has {ratio:.2%} NaN values"
                        quality_result['warnings'].append(warning_msg)

            if nan_issues:
                quality_result['errors'].extend(nan_issues)
                quality_result['is_valid'] = False

            # Check for infinite values
            inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                inf_counts = inf_mask.sum()
                inf_ratios = inf_counts / df.shape[0]

                inf_issues = []
                for col in df.columns:
                    if col in inf_counts.index and inf_counts[col] > 0:
                        ratio = inf_ratios[col]
                        if ratio > self.max_inf_ratio:
                            if fail_hard:
                                inf_issues.append(f"Column '{col}' has {ratio:.2%} infinite values (>{self.max_inf_ratio:.2%})")
                            else:
                                quality_result['warnings'].append(f"Column '{col}' has {ratio:.2%} infinite values")
                        else:
                            quality_result['warnings'].append(f"Column '{col}' has {ratio:.2%} infinite values")

                if inf_issues:
                    quality_result['errors'].extend(inf_issues)
                    quality_result['is_valid'] = False

            # Check for constant columns
            constant_cols = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)

            if constant_cols:
                warning_msg = f"Constant columns found: {constant_cols}"
                quality_result['warnings'].append(warning_msg)

            # Check for duplicate columns
            duplicate_cols = df.columns[df.columns.duplicated()].unique()
            if len(duplicate_cols) > 0:
                warning_msg = f"Duplicate columns found: {list(duplicate_cols)}"
                quality_result['warnings'].append(warning_msg)

            # Calculate quality metrics using common operations utilities
            quality_result['quality_metrics'] = {
                'total_nan_count': safe_int(nan_counts.sum()),
                'total_inf_count': safe_int(inf_mask.sum().sum()),
                'constant_columns': len(constant_cols),
                'duplicate_columns': len(duplicate_cols),
                'overall_nan_ratio': safe_float(safe_divide(nan_counts.sum(), total_cells, 0.0)),
                'overall_inf_ratio': safe_float(safe_divide(inf_mask.sum().sum(), total_cells, 0.0)),
                'data_quality_score': calculate_data_quality_score(df, data_quality)
            }

            # Log summary using tprint
            tprint_structured({
                'validation_type': 'data_quality',
                'total_samples': df.shape[0],
                'total_features': df.shape[1],
                'quality_score': quality_result['quality_metrics']['data_quality_score'],
                'issues_found': len(quality_result['errors']) + len(quality_result['warnings'])
            })

            # Record validation with timestamp formatting
            self.validation_history.append({
                'validation_type': 'data_quality',
                'result': quality_result,
                'timestamp': format_datetime(get_current_datetime(), "%Y-%m-%d %H:%M:%S")
            })

            if quality_result['is_valid']:
                tprint_success(f"Data quality validation passed with score: {quality_result['quality_metrics']['data_quality_score']:.2f}")
            else:
                tprint_error(f"Data quality validation failed with {len(quality_result['errors'])} errors")

            return quality_result

        except Exception as e:
            self.logger.error(f"Failed to check data quality: {e}")
            return {
                'is_valid': False,
                'errors': [f"Data quality check failed: {str(e)}"],
                'warnings': [],
                'quality_metrics': {},
                'timestamp': datetime.now().isoformat()
            }

    def calculate_psi(self, base_data: Union[pd.DataFrame, np.ndarray],
                     current_data: Union[pd.DataFrame, np.ndarray],
                     bins: int = 10) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) between base and current data."""
        try:
            psi_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'psi_scores': {},
                'overall_psi': 0.0,
                'drift_detected': False,
                'timestamp': datetime.now().isoformat()
            }

            # Convert to DataFrames if needed
            if isinstance(base_data, np.ndarray):
                base_df = pd.DataFrame(base_data)
            else:
                base_df = base_data.copy()

            if isinstance(current_data, np.ndarray):
                current_df = pd.DataFrame(current_data)
            else:
                current_df = current_data.copy()

            # Ensure same number of features
            if base_df.shape[1] != current_df.shape[1]:
                psi_result['errors'].append(f"Feature count mismatch: base {base_df.shape[1]} vs current {current_df.shape[1]}")
                psi_result['is_valid'] = False
                return psi_result

            # Select numeric columns for PSI calculation
            numeric_cols_base = base_df.select_dtypes(include=[np.number]).columns
            numeric_cols_current = current_df.select_dtypes(include=[np.number]).columns
            common_numeric_cols = list(set(numeric_cols_base) & set(numeric_cols_current))

            if not common_numeric_cols:
                psi_result['warnings'].append("No common numeric columns found for PSI calculation")
                return psi_result

            # Calculate PSI for each numeric column
            column_psis = {}

            for col in common_numeric_cols:
                try:
                    psi_score = self._calculate_column_psi(
                        base_df[col].dropna(),
                        current_df[col].dropna(),
                        bins
                    )
                    column_psis[col] = psi_score

                    # Check for drift
                    if psi_score > self.psi_critical_threshold:
                        psi_result['drift_detected'] = True
                        psi_result['warnings'].append(f"Critical drift detected in column '{col}': PSI = {psi_score:.3f}")
                    elif psi_score > self.psi_warning_threshold:
                        psi_result['warnings'].append(f"Warning drift detected in column '{col}': PSI = {psi_score:.3f}")

                except Exception as e:
                    self.logger.warning(f"Failed to calculate PSI for column '{col}': {e}")
                    column_psis[col] = None

            # Filter out None values and calculate overall PSI
            valid_psis = [psi for psi in column_psis.values() if psi is not None]

            if valid_psis:
                # Overall PSI as average of column PSIs
                psi_result['overall_psi'] = float(np.mean(valid_psis))
                psi_result['psi_scores'] = {k: float(v) if v is not None else None for k, v in column_psis.items()}

                if psi_result['overall_psi'] > self.psi_critical_threshold:
                    psi_result['drift_detected'] = True

            # Record validation
            self.validation_history.append({
                'validation_type': 'psi_drift',
                'result': psi_result,
                'timestamp': datetime.now()
            })

            return psi_result

        except Exception as e:
            self.logger.error(f"Failed to calculate PSI: {e}")
            return {
                'is_valid': False,
                'errors': [f"PSI calculation failed: {str(e)}"],
                'warnings': [],
                'psi_scores': {},
                'overall_psi': 0.0,
                'drift_detected': False,
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_column_psi(self, base_series: pd.Series, current_series: pd.Series, bins: int) -> float:
        """Calculate PSI for a single column."""
        try:
            # Remove any remaining NaN/inf values
            base_clean = base_series.replace([np.inf, -np.inf], np.nan).dropna()
            current_clean = current_series.replace([np.inf, -np.inf], np.nan).dropna()

            if len(base_clean) == 0 or len(current_clean) == 0:
                return 0.0

            # Create bins based on base data quantiles
            try:
                # Use quantiles for bin edges
                bin_edges = np.quantile(base_clean, np.linspace(0, 1, bins + 1))

                # Ensure unique bin edges
                if len(np.unique(bin_edges)) < len(bin_edges):
                    # Fallback to equal-width bins
                    min_val = min(base_clean.min(), current_clean.min())
                    max_val = max(base_clean.max(), current_clean.max())
                    bin_edges = np.linspace(min_val, max_val, bins + 1)

            except Exception:
                # Fallback to equal-width bins
                min_val = min(base_clean.min(), current_clean.min())
                max_val = max(base_clean.max(), current_clean.max())
                bin_edges = np.linspace(min_val, max_val, bins + 1)

            # Calculate distributions
            base_hist, _ = np.histogram(base_clean, bins=bin_edges)
            current_hist, _ = np.histogram(current_clean, bins=bin_edges)

            # Convert to percentages
            base_pct = base_hist / len(base_clean)
            current_pct = current_hist / len(current_clean)

            # Avoid division by zero
            base_pct = np.where(base_pct == 0, 0.0001, base_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)

            # Calculate PSI
            psi_values = (current_pct - base_pct) * np.log(current_pct / base_pct)
            psi_score = np.sum(psi_values)

            return float(psi_score)

        except Exception as e:
            self.logger.warning(f"Failed to calculate column PSI: {e}")
            return 0.0

    def set_psi_baseline(self, data: Union[pd.DataFrame, np.ndarray], name: str) -> None:
        """Set PSI baseline for drift detection."""
        try:
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Store baseline statistics for numeric columns
            baseline_stats = {}

            for col in df.select_dtypes(include=[np.number]).columns:
                series = df[col].dropna()
                if len(series) > 0:
                    baseline_stats[col] = {
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'quantiles': series.quantile([0.25, 0.5, 0.75]).to_dict(),
                        'sample_size': len(series)
                    }

            self.psi_baselines[name] = {
                'statistics': baseline_stats,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'timestamp': datetime.now()
            }

            self.logger.info(f"PSI baseline '{name}' set for {len(baseline_stats)} columns")

        except Exception as e:
            self.logger.error(f"Failed to set PSI baseline: {e}")

    def validate_against_baseline(self, data: Union[pd.DataFrame, np.ndarray],
                                 baseline_name: str) -> Dict[str, Any]:
        """Validate data against a stored PSI baseline."""
        try:
            if baseline_name not in self.psi_baselines:
                return {
                    'is_valid': False,
                    'errors': [f"Baseline '{baseline_name}' not found"],
                    'warnings': [],
                    'drift_info': {},
                    'timestamp': datetime.now().isoformat()
                }

            baseline = self.psi_baselines[baseline_name]
            psi_result = self.calculate_psi(baseline['statistics'], data)

            # Add baseline information to result
            result = psi_result.copy()
            result['baseline_name'] = baseline_name
            result['baseline_timestamp'] = baseline['timestamp'].isoformat()

            return result

        except Exception as e:
            self.logger.error(f"Failed to validate against baseline: {e}")
            return {
                'is_valid': False,
                'errors': [f"Baseline validation failed: {str(e)}"],
                'warnings': [],
                'drift_info': {},
                'timestamp': datetime.now().isoformat()
            }

    def get_validation_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get summary of recent validations."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            recent_validations = [
                v for v in self.validation_history
                if v['timestamp'] >= cutoff_date
            ]

            if not recent_validations:
                return {'message': 'No recent validations found'}

            # Summarize by validation type
            summary_by_type = defaultdict(list)

            for validation in recent_validations:
                val_type = validation['validation_type']
                summary_by_type[val_type].append(validation['result'])

            # Calculate overall statistics
            total_validations = len(recent_validations)
            successful_validations = sum(1 for v in recent_validations if v['result'].get('is_valid', False))

            return {
                'period_days': days_back,
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
                'summary_by_type': dict(summary_by_type),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get validation summary: {e}")
            return {'error': str(e)}

    def log_validation_report(self, result: Dict[str, Any]) -> None:
        """Log validation results with appropriate level."""
        try:
            if not result.get('is_valid', True):
                self.logger.error("❌ DATA VALIDATION FAILED")
                for error in result.get('errors', []):
                    self.logger.error(f"   Error: {error}")

                for warning in result.get('warnings', []):
                    self.logger.warning(f"   Warning: {warning}")
            else:
                self.logger.info("✅ Data validation passed")

                # Log warnings even if validation passed
                for warning in result.get('warnings', []):
                    self.logger.warning(f"   Warning: {warning}")

        except Exception as e:
            self.logger.error(f"Failed to log validation report: {e}")

def validate_data_comprehensive(data: Union[pd.DataFrame, np.ndarray],
                              validator: Optional[DataValidator] = None,
                              fail_hard: bool = True) -> Dict[str, Any]:
    """Comprehensive data validation combining all checks."""
    if validator is None:
        validator = DataValidator()

    # Run all validation checks
    schema_result = validator.validate_schema(data, strict=fail_hard)
    quality_result = validator.check_data_quality(data, fail_hard=fail_hard)

    # Combine results
    combined_result = {
        'is_valid': schema_result['is_valid'] and quality_result['is_valid'],
        'errors': schema_result['errors'] + quality_result['errors'],
        'warnings': schema_result['warnings'] + quality_result['warnings'],
        'schema_result': schema_result,
        'quality_result': quality_result,
        'timestamp': datetime.now().isoformat()
    }

    return combined_result
