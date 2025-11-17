"""
Math Validation Functions

This module provides common math validation functions used across the feature engineering package.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import warnings

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

# Setup logging
logger = logging.getLogger(__name__)

def safe_divide(a, b, default=0.0):
    """Safely divide two numbers, returning default if division by zero or error occurs."""
    try:
        return a / b if b != 0 else default
    except:
        return default

def safe_log(x, default=0.0):
    """Safely calculate logarithm, returning default if x <= 0 or error occurs."""
    try:
        return np.log(x) if x > 0 else default
    except:
        return default

def safe_sqrt(x, default=0.0):
    """Safely calculate square root, returning default if x < 0 or error occurs."""
    try:
        return np.sqrt(x) if x >= 0 else default
    except:
        return default

def validate_positive(value, name="value"):
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def validate_range(value, min_val, max_val, name="value"):
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value

def validate_finite(value, name="value"):
    """Validate that a value is finite."""
    import numpy as np
    try:
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError(f"{name} cannot be empty")
            # Check for non-finite values using explicit boolean array handling
            finite_mask = np.isfinite(value)
            has_non_finite = not finite_mask.all()
            if has_non_finite:
                non_finite_count = np.sum(~finite_mask)
                raise ValueError(f"{name} contains {non_finite_count} non-finite values (NaN or inf)")
            return value

        # Handle scalar values - check if it's a single-element array first
        if hasattr(value, '__len__') and len(value) == 1:
            # Single-element array or list
            val = float(value[0])
        elif hasattr(value, '__len__') and len(value) > 1:
            # Multi-element array - convert to numpy array for validation
            val_array = np.array(value)
            finite_mask = np.isfinite(val_array)
            if not finite_mask.all():
                raise ValueError(f"{name} contains non-finite values")
            return val_array
        else:
            # Scalar value
            val = float(value)

        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        return val
    except Exception as e:
        raise ValueError(f"Invalid {name}: {e}")

def safe_percentage_change(old_value, new_value):
    """Safely calculate percentage change."""
    try:
        # Handle pandas Series inputs
        if hasattr(old_value, 'values') and hasattr(new_value, 'values'):
            # Both inputs are pandas Series or similar
            result = pd.Series(index=old_value.index, dtype=float)
            for i in range(len(old_value)):
                try:
                    current_old = old_value.iloc[i]
                    current_new = new_value.iloc[i]
                    # If either side is NaN or the denominator is zero, treat as 0% change
                    if pd.isna(current_old) or pd.isna(current_new) or current_old == 0:
                        result.iloc[i] = 0.0
                    else:
                        result.iloc[i] = ((current_new - current_old) / current_old) * 100
                except (ZeroDivisionError, ValueError, TypeError):
                    result.iloc[i] = 0.0
            return result
        else:
            # Handle scalar inputs
            if (
                old_value is None
                or new_value is None
                or (isinstance(old_value, float) and np.isnan(old_value))
                or (isinstance(new_value, float) and np.isnan(new_value))
                or old_value == 0
            ):
                return 0.0
            return ((new_value - old_value) / old_value) * 100
    except Exception:
        # Return appropriate type based on input
        if hasattr(old_value, 'values') and hasattr(new_value, 'values'):
            return pd.Series(0.0, index=old_value.index)
        return 0.0

class MathValidationError(Exception):
    """Math validation error."""
    pass

class FeatureValidationError(Exception):
    """Feature validation error."""
    pass

def validate_feature_quality(feature_data: Union[np.ndarray, pd.Series],
                           feature_name: str = "feature",
                           warn_on_issues: bool = True,
                           raise_on_critical: bool = False,
                           exclude_first_rows: int = 50) -> Dict[str, Any]:
    """
    Validate feature quality and return a comprehensive report.

    Args:
        feature_data: The feature data to validate
        feature_name: Name of the feature for logging
        warn_on_issues: Whether to issue warnings for quality issues
        raise_on_critical: Whether to raise exceptions for critical issues
        exclude_first_rows: Number of rows to exclude from validation (default: 50)

    Returns:
        Dictionary with validation results and statistics
    """
    validation_report = {
        'feature_name': feature_name,
        'total_values': len(feature_data),
        'valid_values': 0,
        'infinite_values': 0,
        'nan_values': 0,
        'zero_values': 0,
        'constant_feature': False,
        'unique_values': 0,
        'min_value': None,
        'max_value': None,
        'mean_value': None,
        'std_value': None,
        'issues': [],
        'critical_issues': [],
        'warnings': []
    }

    try:
        # Convert to numpy array for easier processing
        if isinstance(feature_data, pd.Series):
            data_array = feature_data.values
        else:
            data_array = feature_data

        # Basic statistics
        validation_report['total_values'] = len(data_array)

        # Exclude first N rows from validation (typically for warm-up period)
        if exclude_first_rows > 0 and len(data_array) > exclude_first_rows:
            data_array = data_array[exclude_first_rows:]
            validation_report['excluded_first_rows'] = exclude_first_rows
        else:
            validation_report['excluded_first_rows'] = 0

        # Check for infinite values
        infinite_mask = np.isinf(data_array)
        infinite_count = np.sum(infinite_mask)
        validation_report['infinite_values'] = infinite_count

        if infinite_count > 0:
            issue_msg = f"Feature '{feature_name}' contains {infinite_count} infinite values"
            validation_report['issues'].append(issue_msg)
            validation_report['critical_issues'].append(issue_msg)
            if warn_on_issues:
                warnings.warn(issue_msg, UserWarning)
            if raise_on_critical:
                raise FeatureValidationError(issue_msg)

        # Check for NaN values
        nan_mask = np.isnan(data_array)
        nan_count = np.sum(nan_mask)
        validation_report['nan_values'] = nan_count

        if nan_count > 0:
            issue_msg = f"Feature '{feature_name}' contains {nan_count} NaN values"
            validation_report['issues'].append(issue_msg)
            validation_report['critical_issues'].append(issue_msg)
            if warn_on_issues:
                warnings.warn(issue_msg, UserWarning)
            if raise_on_critical:
                raise FeatureValidationError(issue_msg)

        # Get valid (finite, non-NaN) values
        valid_mask = ~(infinite_mask | nan_mask)
        valid_data = data_array[valid_mask]
        validation_report['valid_values'] = len(valid_data)

        if len(valid_data) == 0:
            issue_msg = f"Feature '{feature_name}' has no valid values"
            validation_report['issues'].append(issue_msg)
            validation_report['critical_issues'].append(issue_msg)
            if warn_on_issues:
                warnings.warn(issue_msg, UserWarning)
            if raise_on_critical:
                raise FeatureValidationError(issue_msg)
            return validation_report

        # Check for constant features
        unique_values = np.unique(valid_data)
        validation_report['unique_values'] = len(unique_values)

        if len(unique_values) == 1:
            issue_msg = f"Feature '{feature_name}' is constant (all values = {unique_values[0]})"
            validation_report['issues'].append(issue_msg)
            validation_report['warnings'].append(issue_msg)
            validation_report['constant_feature'] = True
            if warn_on_issues:
                warnings.warn(issue_msg, UserWarning)

        # Check for zero values
        zero_count = np.sum(valid_data == 0)
        validation_report['zero_values'] = zero_count

        if zero_count > 0:
            zero_pct = (zero_count / len(valid_data)) * 100
            if zero_pct > 1:  # More than 1% zeros
                issue_msg = f"Feature '{feature_name}' has {zero_pct:.1f}% zero values"
                validation_report['issues'].append(issue_msg)
                validation_report['warnings'].append(issue_msg)
                if warn_on_issues:
                    warnings.warn(issue_msg, UserWarning)

        # Calculate statistics for valid data
        validation_report['min_value'] = float(np.min(valid_data))
        validation_report['max_value'] = float(np.max(valid_data))
        validation_report['mean_value'] = float(np.mean(valid_data))
        validation_report['std_value'] = float(np.std(valid_data))

        # Check for extreme values
        if validation_report['std_value'] == 0 and not validation_report['constant_feature']:
            issue_msg = f"Feature '{feature_name}' has zero standard deviation"
            validation_report['issues'].append(issue_msg)
            validation_report['warnings'].append(issue_msg)
            if warn_on_issues:
                warnings.warn(issue_msg, UserWarning)

        # Log summary with feature name
        logger.debug(f"Feature validation for '{feature_name}': "
                    f"{validation_report['valid_values']}/{validation_report['total_values']} valid, "
                    f"{len(validation_report['issues'])} issues")

        # Log each issue with feature name for better visibility
        if validation_report['issues']:
            logger.warning(f"Feature '{feature_name}' validation issues:")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")

    except Exception as e:
        error_msg = f"Error validating feature '{feature_name}': {str(e)}"
        validation_report['critical_issues'].append(error_msg)
        logger.error(error_msg)
        if raise_on_critical:
            raise FeatureValidationError(error_msg)

    return validation_report

def validate_features_dataframe(df: pd.DataFrame,
                              feature_columns: Optional[List[str]] = None,
                              warn_on_issues: bool = True,
                              raise_on_critical: bool = False,
                              exclude_first_rows: int = 50) -> Dict[str, Dict[str, Any]]:
    """
    Validate all features in a DataFrame.

    Args:
        df: DataFrame containing features
        feature_columns: List of column names to validate (if None, validates all numeric columns)
        warn_on_issues: Whether to issue warnings for quality issues
        raise_on_critical: Whether to raise exceptions for critical issues
        exclude_first_rows: Number of rows to exclude from validation (default: 50)

    Returns:
        Dictionary mapping feature names to their validation reports
    """
    if feature_columns is None:
        # Validate all numeric columns
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    validation_results = {}

    for column in feature_columns:
        if column in df.columns:
            validation_results[column] = validate_feature_quality(
                df[column],
                feature_name=column,
                warn_on_issues=warn_on_issues,
                raise_on_critical=raise_on_critical,
                exclude_first_rows=exclude_first_rows
            )
        else:
            logger.warning(f"Column '{column}' not found in DataFrame")

    return validation_results

def feature_validation_decorator(warn_on_issues: bool = True,
                                raise_on_critical: bool = False,
                                validate_output: bool = True,
                                validate_input: bool = False,
                                exclude_first_rows: int = 50):
    """
    Decorator to automatically validate features generated by functions.

    Args:
        warn_on_issues: Whether to issue warnings for quality issues
        raise_on_critical: Whether to raise exceptions for critical issues
        validate_output: Whether to validate the function's output
        validate_input: Whether to validate the function's input DataFrame
        exclude_first_rows: Number of rows to exclude from validation (default: 50)

    Returns:
        Decorated function with automatic feature validation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate input if requested
            if validate_input and args:
                # Assume first argument is the DataFrame
                input_df = args[0]
                if isinstance(input_df, pd.DataFrame):
                    logger.debug(f"Validating input DataFrame for function {func.__name__}")
                    input_validation = validate_features_dataframe(
                        input_df,
                        warn_on_issues=warn_on_issues,
                        raise_on_critical=raise_on_critical,
                        exclude_first_rows=exclude_first_rows
                    )

            # Execute the original function
            result = func(*args, **kwargs)

            # Validate output if requested and result is a DataFrame
            if validate_output and isinstance(result, pd.DataFrame):
                logger.debug(f"Validating output DataFrame for function {func.__name__}")
                output_validation = validate_features_dataframe(
                    result,
                    warn_on_issues=warn_on_issues,
                    raise_on_critical=raise_on_critical,
                    exclude_first_rows=exclude_first_rows
                )

                # Log summary of validation results with feature details
                total_issues = sum(len(report['issues']) for report in output_validation.values())
                total_critical = sum(len(report['critical_issues']) for report in output_validation.values())

                if total_critical > 0:
                    logger.error(f"Function {func.__name__} generated {total_critical} critical feature issues:")
                    for feature_name, report in output_validation.items():
                        if report['critical_issues']:
                            logger.error(f"  Feature '{feature_name}': {len(report['critical_issues'])} critical issues")
                            for issue in report['critical_issues']:
                                logger.error(f"    - {issue}")
                elif total_issues > 0:
                    logger.warning(f"Function {func.__name__} generated {total_issues} feature quality issues:")
                    for feature_name, report in output_validation.items():
                        if report['issues']:
                            logger.warning(f"  Feature '{feature_name}': {len(report['issues'])} issues")
                            for issue in report['issues']:
                                logger.warning(f"    - {issue}")
                else:
                    logger.debug(f"Function {func.__name__} generated features passed validation")

            return result

        return wrapper
    return decorator

# Convenience decorators for common use cases
def validate_generated_features(func: Callable) -> Callable:
    """Decorator to validate features generated by a function (output validation only)."""
    return feature_validation_decorator(
        warn_on_issues=True,
        raise_on_critical=False,
        validate_output=True,
        validate_input=False,
        exclude_first_rows=50
    )(func)

def validate_feature_pipeline(func: Callable) -> Callable:
    """Decorator to validate both input and output of a feature engineering pipeline."""
    return feature_validation_decorator(
        warn_on_issues=True,
        raise_on_critical=False,
        validate_output=True,
        validate_input=True,
        exclude_first_rows=50
    )(func)

def strict_feature_validation(func: Callable) -> Callable:
    """Decorator with strict validation that raises exceptions on critical issues."""
    return feature_validation_decorator(
        warn_on_issues=True,
        raise_on_critical=True,
        validate_output=True,
        validate_input=True,
        exclude_first_rows=50
    )(func)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
