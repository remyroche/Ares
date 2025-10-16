"""
Common data preprocessing utilities for NAS and TAS regime detection systems.

This module provides standardized data preprocessing functions that ensure
consistent handling of categorical columns, data type validation, and ML-ready
data preparation across both NAS and TAS systems.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def prepare_ml_data(
    data: Union[pd.DataFrame, np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for ML processing by excluding categorical columns and ensuring numeric types.

    Args:
        data: Input data as DataFrame or numpy array
        timestamps: Optional timestamps array
        exclude_columns: Additional columns to exclude beyond standard categorical ones

    Returns:
        Tuple of (ml_data_array, timestamps)

    Raises:
        ValueError: If no numeric columns are found
        TypeError: If data cannot be converted to numeric format
    """
    try:
        # Default categorical columns to exclude
        default_exclude = ['timestamp', 'symbol', 'interval']
        if exclude_columns:
            exclude_columns = list(set(default_exclude + exclude_columns))
        else:
            exclude_columns = default_exclude

        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if timestamps is None:
                timestamps = np.arange(len(data))

            # Assume standard OHLCV columns if no column names provided
            columns = ['open', 'high', 'low', 'close', 'volume'] if data.shape[1] >= 5 else [f'col_{i}' for i in range(data.shape[1])]
            data_df = pd.DataFrame(data, columns=columns)
            data_df['timestamp'] = timestamps
        else:
            data_df = data.copy()
            if timestamps is None and 'timestamp' in data_df.columns:
                timestamps = data_df['timestamp'].values

        # Extract timestamps if present
        if 'timestamp' in data_df.columns:
            timestamps = data_df['timestamp'].values

        # Get ML columns (exclude categorical columns)
        ml_columns = [col for col in data_df.columns if col not in exclude_columns]

        if len(ml_columns) == 0:
            raise ValueError(f"No ML columns found after excluding: {exclude_columns}")

        # Extract ML data
        ml_data = data_df[ml_columns]

        # Ensure all columns are numeric
        numeric_columns = []
        for col in ml_columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(ml_data[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                logger.warning(f"Excluding non-numeric column: {col}")

        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for ML processing")

        # Convert to numpy array with proper data types
        ml_data_array = ml_data[numeric_columns].values.astype(np.float64)

        logger.info(f"ML data prepared: {ml_data_array.shape[0]} samples, {ml_data_array.shape[1]} features")
        logger.info(f"Excluded columns: {[col for col in ml_columns if col not in numeric_columns]}")

        return ml_data_array, timestamps

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

def validate_ml_data(data_array: np.ndarray, name: str = "data") -> np.ndarray:
    """
    Validate that data array is suitable for ML processing.

    Args:
        data_array: Data array to validate
        name: Name for logging purposes

    Returns:
        Validated data array

    Raises:
        ValueError: If data is not suitable for ML processing
    """
    try:
        # Check if array is numeric
        if not np.issubdtype(data_array.dtype, np.number):
            raise ValueError(f"{name} contains non-numeric data types")

        # Check for finite values
        finite_mask = np.isfinite(data_array)
        if not finite_mask.all():
            non_finite_count = np.sum(~finite_mask)
            logger.warning(f"{name} contains {non_finite_count} non-finite values")

            # Replace non-finite values with 0
            data_array = np.where(np.isfinite(data_array), data_array, 0.0)
            logger.info(f"Replaced non-finite values with 0 in {name}")

        # Check for reasonable shape
        if data_array.size == 0:
            raise ValueError(f"{name} is empty")

        if len(data_array.shape) != 2:
            raise ValueError(f"{name} must be 2D array, got shape: {data_array.shape}")

        logger.info(f"{name} validation passed: {data_array.shape}")
        return data_array

    except Exception as e:
        logger.error(f"Data validation failed for {name}: {e}")
        raise

def normalize_ml_data(data_array: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize data array for ML processing.

    Args:
        data_array: Data array to normalize
        method: Normalization method ('zscore', 'minmax', 'robust')

    Returns:
        Normalized data array
    """
    try:
        if method == "zscore":
            mean_vals = np.mean(data_array, axis=0)
            std_vals = np.std(data_array, axis=0)
            # Avoid division by zero
            std_vals = np.where(std_vals == 0, 1.0, std_vals)
            normalized = (data_array - mean_vals) / std_vals

        elif method == "minmax":
            min_vals = np.min(data_array, axis=0)
            max_vals = np.max(data_array, axis=0)
            range_vals = max_vals - min_vals
            # Avoid division by zero
            range_vals = np.where(range_vals == 0, 1.0, range_vals)
            normalized = (data_array - min_vals) / range_vals

        elif method == "robust":
            median_vals = np.median(data_array, axis=0)
            mad_vals = np.median(np.abs(data_array - median_vals), axis=0)
            # Avoid division by zero
            mad_vals = np.where(mad_vals == 0, 1.0, mad_vals)
            normalized = (data_array - median_vals) / mad_vals

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.info(f"Data normalized using {method} method")
        return normalized

    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        raise

def get_ml_columns_info(data: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> dict:
    """
    Get information about ML columns in the dataset.

    Args:
        data: Input DataFrame
        exclude_columns: Columns to exclude from analysis

    Returns:
        Dictionary with column information
    """
    default_exclude = ['timestamp', 'symbol', 'interval']
    if exclude_columns:
        exclude_columns = list(set(default_exclude + exclude_columns))
    else:
        exclude_columns = default_exclude

    ml_columns = [col for col in data.columns if col not in exclude_columns]

    info = {
        'total_columns': len(data.columns),
        'ml_columns': len(ml_columns),
        'excluded_columns': len(exclude_columns),
        'ml_column_names': ml_columns,
        'excluded_column_names': [col for col in data.columns if col in exclude_columns]
    }

    # Check data types
    numeric_columns = []
    non_numeric_columns = []

    for col in ml_columns:
        try:
            pd.to_numeric(data[col], errors='raise')
            numeric_columns.append(col)
        except (ValueError, TypeError):
            non_numeric_columns.append(col)

    info.update({
        'numeric_columns': len(numeric_columns),
        'non_numeric_columns': len(non_numeric_columns),
        'numeric_column_names': numeric_columns,
        'non_numeric_column_names': non_numeric_columns
    })

    return info
