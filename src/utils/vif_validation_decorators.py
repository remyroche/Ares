"""
VIF Validation Decorators

This module provides decorators specifically for validating VIF (Variance Inflation Factor)
calculations and handling edge cases like NaN, infinite, and zero values.
"""

import functools
import logging
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import signal
from contextlib import contextmanager

from src.utils.logger import system_logger


class VIFValidationError(Exception):
    """Custom exception for VIF validation errors."""
    pass


@contextmanager
def timeout_context(seconds: int, operation_name: str = "operation"):
    """Context manager for timeout handling."""
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore original handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_vif_inputs(
    check_nan: bool = True,
    check_infinite: bool = True,
    check_zero_variance: bool = True,
    check_duplicates: bool = True,
    log_level: str = "INFO"
):
    """
    Decorator to validate inputs before VIF calculation.

    Args:
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_zero_variance: Whether to check for zero variance features
        check_duplicates: Whether to check for duplicate features
        log_level: Logging level for validation messages
    """


def validate_vif_outputs(
    check_nan_vif: bool = True,
    check_infinite_vif: bool = True,
    check_zero_vif: bool = True,
    max_vif_threshold: float = 1000.0,
    log_level: str = "INFO"
):
    """
    Decorator to validate VIF calculation outputs.

    Args:
        check_nan_vif: Whether to check for NaN VIF values
        check_infinite_vif: Whether to check for infinite VIF values
        check_zero_vif: Whether to check for zero VIF values
        max_vif_threshold: Maximum acceptable VIF value
        log_level: Logging level for validation messages
    """


def safe_vif_calculation(
    timeout_seconds: int = 30,
    fallback_strategy: str = "ones",
    log_level: str = "INFO"
):
    """
    Decorator to safely calculate VIF with timeout protection and fallback strategies.

    Args:
        timeout_seconds: Timeout for VIF calculation in seconds
        fallback_strategy: Strategy to use when VIF calculation fails ("ones", "skip", "error")
        log_level: Logging level for validation messages
    """


def _extract_data_from_args(args: tuple, kwargs: dict) -> Optional[pd.DataFrame]:
    """Extract DataFrame from function arguments."""
    # Look for DataFrame in positional arguments
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg

    # Look for DataFrame in keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            return value

    return None


def _extract_vif_from_result(result: Any) -> Optional[pd.Series]:
    """Extract VIF values from function result."""
    if isinstance(result, pd.Series):
        return result
    elif isinstance(result, dict) and 'vif_values' in result:
        return result['vif_values']
    elif isinstance(result, dict) and 'vif' in result:
        return result['vif']
    elif hasattr(result, 'vif_values'):
        return result.vif_values
    elif hasattr(result, 'vif'):
        return result.vif

    return None


def _validate_nan_values(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for NaN values in the data."""
    nan_count = data.isna().sum().sum()
    nan_features = data.columns[data.isna().any()].tolist()

    return {
        'has_issues': nan_count > 0,
        'nan_count': nan_count,
        'nan_features': nan_features,
        'nan_percentage': (nan_count / (data.shape[0] * data.shape[1])) * 100
    }


def _validate_infinite_values(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for infinite values in the data."""
    numeric_data = data.select_dtypes(include=[np.number])
    infinite_count = np.isinf(numeric_data).sum().sum()
    infinite_features = numeric_data.columns[np.isinf(numeric_data).any()].tolist()

    return {
        'has_issues': infinite_count > 0,
        'infinite_count': infinite_count,
        'infinite_features': infinite_features,
        'infinite_percentage': (infinite_count / (numeric_data.shape[0] * numeric_data.shape[1])) * 100
    }


def _validate_zero_variance_features(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for zero variance features."""
    numeric_data = data.select_dtypes(include=[np.number])
    variances = numeric_data.var()
    zero_var_features = variances[variances == 0].index.tolist()

    return {
        'has_issues': len(zero_var_features) > 0,
        'zero_var_features': zero_var_features,
        'zero_var_count': len(zero_var_features)
    }


def _validate_duplicate_features(data: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for duplicate features."""
    # Check for exact duplicates
    duplicate_features = []
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns[i+1:], i+1):
            if data[col1].equals(data[col2]):
                duplicate_features.append((col1, col2))

    return {
        'has_issues': len(duplicate_features) > 0,
        'duplicate_features': duplicate_features,
        'duplicate_count': len(duplicate_features)
    }


def _validate_nan_vif_values(vif_values: pd.Series, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for NaN VIF values."""
    nan_vif_features = vif_values[vif_values.isna()].index.tolist()

    return {
        'has_issues': len(nan_vif_features) > 0,
        'nan_vif_features': nan_vif_features,
        'nan_vif_count': len(nan_vif_features)
    }


def _validate_infinite_vif_values(vif_values: pd.Series, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for infinite VIF values."""
    infinite_vif_features = vif_values[np.isinf(vif_values)].index.tolist()

    return {
        'has_issues': len(infinite_vif_features) > 0,
        'infinite_vif_features': infinite_vif_features,
        'infinite_vif_count': len(infinite_vif_features)
    }


def _validate_zero_vif_values(vif_values: pd.Series, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for zero VIF values."""
    zero_vif_features = vif_values[vif_values == 0].index.tolist()

    return {
        'has_issues': len(zero_vif_features) > 0,
        'zero_vif_features': zero_vif_features,
        'zero_vif_count': len(zero_vif_features)
    }


def _validate_high_vif_values(vif_values: pd.Series, max_threshold: float, logger: logging.Logger) -> Dict[str, Any]:
    """Validate for high VIF values."""
    high_vif_features = vif_values[vif_values > max_threshold].index.tolist()

    return {
        'has_issues': len(high_vif_features) > 0,
        'high_vif_features': high_vif_features,
        'high_vif_count': len(high_vif_features),
        'max_vif_value': float(vif_values.max()) if not vif_values.empty else 0.0
    }


def _log_validation_summary(validation_results: Dict[str, Any], logger: logging.Logger, log_level: str):
    """Log comprehensive validation summary."""
    if not validation_results:
        return

    logger.info("📊 VIF Input Validation Summary:")

    for validation_type, results in validation_results.items():
        if results.get('has_issues', False):
            if validation_type == 'nan':
                logger.warning(f"   ⚠️ NaN Values: {results['nan_count']} cells ({results['nan_percentage']:.2f}%)")
            elif validation_type == 'infinite':
                logger.warning(f"   ⚠️ Infinite Values: {results['infinite_count']} cells ({results['infinite_percentage']:.2f}%)")
            elif validation_type == 'zero_variance':
                logger.warning(f"   ⚠️ Zero Variance Features: {results['zero_var_count']} features")
            elif validation_type == 'duplicates':
                logger.warning(f"   ⚠️ Duplicate Features: {results['duplicate_count']} pairs")


def _log_vif_validation_summary(validation_results: Dict[str, Any], logger: logging.Logger, log_level: str):
    """Log comprehensive VIF validation summary."""
    if not validation_results:
        return

    logger.info("📊 VIF Output Validation Summary:")

    for validation_type, results in validation_results.items():
        if results.get('has_issues', False):
            if validation_type == 'nan_vif':
                logger.error(f"   ❌ NaN VIF Values: {results['nan_vif_count']} features")
            elif validation_type == 'infinite_vif':
                logger.error(f"   ❌ Infinite VIF Values: {results['infinite_vif_count']} features")
            elif validation_type == 'zero_vif':
                logger.warning(f"   ⚠️ Zero VIF Values: {results['zero_vif_count']} features")
            elif validation_type == 'high_vif':
                logger.warning(f"   ⚠️ High VIF Values: {results['high_vif_count']} features (max: {results['max_vif_value']:.2f})")


def _create_fallback_vif_result(args: tuple, kwargs: dict, fallback_value: Optional[float]) -> pd.Series:
    """Create fallback VIF result when calculation fails."""
    data = _extract_data_from_args(args, kwargs)
    if data is None:
        # Fallback implementation for data
        # Fallback implementation for data
        return pd.Series()

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if fallback_value is None:
        # Fallback implementation for fallback_value
        return pd.Series(dtype=float)
    else:
        return pd.Series([fallback_value] * len(numeric_cols), index=numeric_cols)


# Convenience decorator that combines all VIF validations