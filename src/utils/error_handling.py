"""
Error handling utilities for the Ares project.

This module provides common error handling functions used across different modules.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional, Any

logger = logging.getLogger(__name__)


def safe_diff(series: Union[pd.Series, np.ndarray, float, int], periods: int = 1) -> Union[pd.Series, np.ndarray]:
    """
    Safely compute differences, handling scalar returns from pandas operations.

    Args:
        series: Input series, array, or scalar
        periods: Number of periods to shift

    Returns:
        Series with differences, ensuring proper Series type
    """
    try:
        # Handle scalar inputs
        if isinstance(series, (int, float)):
            # For scalars, diff doesn't make sense, return the scalar itself
            return series

        # Handle Series or array inputs
        result = series.diff(periods=periods)

        # Handle case where diff returns a scalar
        if isinstance(result, (int, float)):
            if isinstance(series, pd.Series):
                return pd.Series([result] * len(series), index=series.index)
            else:
                return np.full(len(series), result)

        return result

    except Exception as e:
        logger.warning(f"Error in safe_diff: {e}")
        # Return zeros if diff fails
        if isinstance(series, (int, float)):
            return series  # Return the scalar as-is
        elif isinstance(series, pd.Series):
            return pd.Series(0.0, index=series.index)
        else:
            return np.zeros(len(series))


def safe_operation(operation_name: str, *args, **kwargs):
    """
    Safely execute an operation with error handling.

    Args:
        operation_name: Name of the operation for logging
        *args: Arguments to pass to the operation
        **kwargs: Keyword arguments to pass to the operation

    Returns:
        Result of the operation or None if it fails
    """
    try:
        # This is a placeholder - in a real implementation you'd call the actual operation
        logger.debug(f"Executing safe operation: {operation_name}")
        return None
    except Exception as e:
        logger.error(f"Error in safe operation '{operation_name}': {e}")
        return None


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ComputationError(Exception):
    """Raised when computation fails."""
    pass
