# src/utils/data_validation.py

from typing import Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_pct_change(
    series: pd.Series,
    periods: int = 1,
    fill_method: str = "ffill",
    limit: int | None = None,
    freq: Any | None = None,
    **kwargs: Any,
) -> pd.Series:
    """
    Calculate percentage change with safe handling of infinite values.

    This function wraps pandas pct_change() and handles the infinite values
    that can occur when dividing by zero or very small numbers.

    Args:
        series: Input pandas Series
        periods: Number of periods to shift for calculating change
        fill_method: Method for filling NaN values before calculation
        limit: Maximum number of consecutive NaN values to fill
        freq: Frequency string for time series data
        **kwargs: Additional arguments passed to pct_change()

    Returns:
        pd.Series: Percentage change with infinite values replaced by 0
    """
    # Calculate percentage change
    pct_change = series.pct_change(
        periods=periods,
        fill_method=fill_method,
        limit=limit,
        freq=freq,
        **kwargs,
    )

    # Count infinite values for logging
    inf_count = np.isinf(pct_change).sum()
    if inf_count > 0:
        logger.warning(
            f"⚠️ Found {inf_count} infinite values in pct_change calculation - replacing with 0",
        )

    # Replace infinite values with 0
    pct_change = pct_change.replace([np.inf, -np.inf], 0)

    # Fill any remaining NaN values
    return pct_change.fillna(0)


def safe_log_returns(
    series: pd.Series,
    periods: int = 1,
    fill_method: str = "ffill",
    limit: int | None = None,
    freq: Any | None = None,
    **kwargs: Any,
) -> pd.Series:
    """
    Calculate log returns with safe handling of infinite values.

    Args:
        series: Input pandas Series
        periods: Number of periods to shift for calculating change
        fill_method: Method for filling NaN values before calculation
        limit: Maximum number of consecutive NaN values to fill
        freq: Frequency string for time series data
        **kwargs: Additional arguments passed to pct_change()

    Returns:
        pd.Series: Log returns with infinite values replaced by 0
    """
    # Calculate log returns using pct_change and log
    pct_change = series.pct_change(
        periods=periods,
        fill_method=fill_method,
        limit=limit,
        freq=freq,
        **kwargs,
    )

    # Add 1 to avoid log(0) and log(negative)
    log_returns = np.log(pct_change + 1)

    # Count infinite values for logging
    inf_count = np.isinf(log_returns).sum()
    if inf_count > 0:
        logger.warning(
            f"⚠️ Found {inf_count} infinite values in log_returns calculation - replacing with 0",
        )

    # Replace infinite values with 0
    log_returns = log_returns.replace([np.inf, -np.inf], 0)

    # Fill any remaining NaN values
    return log_returns.fillna(0)


def validate_dataframe_for_ml(
    df: pd.DataFrame,
    context: str = "unknown",
    clip_extreme_values: bool = True,
    max_abs_value: float = 1000.0,
) -> pd.DataFrame:
    """
    Validate and clean DataFrame for machine learning models.

    This function performs comprehensive validation to ensure the DataFrame
    is safe for use with scikit-learn and other ML libraries.

    Args:
        df: Input DataFrame
        context: Context string for logging
        clip_extreme_values: Whether to clip extreme values
        max_abs_value: Maximum absolute value for clipping

    Returns:
        pd.DataFrame: Cleaned DataFrame safe for ML
    """
    df_clean = df.copy()

    # Select only numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        logger.warning(
            f"⚠️ No numeric columns found in DataFrame for context: {context}",
        )
        return df_clean

    # Check for infinite values
    inf_count = np.isinf(df_clean[numeric_cols]).sum().sum()
    if inf_count > 0:
        logger.warning(
            f"⚠️ Found {inf_count} infinite values in {context} - replacing with 0",
        )
        df_clean[numeric_cols] = df_clean[numeric_cols].replace(
            [np.inf, -np.inf],
            0,
        )

    # Check for extreme values
    if clip_extreme_values:
        extreme_count = (np.abs(df_clean[numeric_cols]) > max_abs_value).sum().sum()
        if extreme_count > 0:
            logger.warning(
                f"⚠️ Found {extreme_count} extreme values (>±{max_abs_value}) in {context} - clipping",
            )
            df_clean[numeric_cols] = np.clip(
                df_clean[numeric_cols],
                -max_abs_value,
                max_abs_value,
            )

    # Check for NaN values
    nan_count = df_clean[numeric_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning(
            f"⚠️ Found {nan_count} NaN values in {context} - filling with 0",
        )
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    # Final validation
    final_inf_count = np.isinf(df_clean[numeric_cols]).sum().sum()
    final_nan_count = df_clean[numeric_cols].isna().sum().sum()

    if final_inf_count == 0 and final_nan_count == 0:
        logger.info(f"✅ Data validation passed for {context}: {df_clean.shape}")
    else:
        logger.error(
            f"🚨 CRITICAL: Data validation failed for {context}: {final_inf_count} inf = {final_nan_count} NaN",
        )

    return df_clean


def safe_division(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    fill_value: float = 0.0,
    context: str = "unknown",
) -> pd.Series | np.ndarray | float:
    """
    Perform safe division that handles division by zero and very small numbers.

    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero or very small
        context: Context string for logging

    Returns:
        Result of safe division
    """
    try:
        # Handle different input types
        if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
            # Both are pandas Series
            result = numerator / denominator
            zero_count = (denominator == 0).sum()
            small_count = ((denominator != 0) & (np.abs(denominator) < 1e-10)).sum()

            if zero_count > 0 or small_count > 0:
                logger.warning(
                    f"⚠️ Found {zero_count} zero and {small_count} very small denominators in {context}",
                )
                result = result.replace([np.inf, -np.inf], fill_value)
                result = result.fillna(fill_value)

            return result

        if isinstance(numerator, np.ndarray) and isinstance(
            denominator, np.ndarray,
        ):
            # Both are numpy arrays or scalars
            return np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, fill_value),
                where=denominator != 0,
            )

        # Mixed types - convert to numpy and handle
        num_array = np.array(numerator)
        den_array = np.array(denominator)
        return np.divide(
            num_array,
            den_array,
            out=np.full_like(num_array, fill_value),
            where=den_array != 0,
        )

    except Exception as e:
        logger.exception(f"Error in safe_division for {context}: {e}")
        # Return fill_value with same shape as numerator
        if isinstance(numerator, pd.Series):
            return pd.Series(fill_value, index=numerator.index)
        if isinstance(numerator, np.ndarray):
            return np.full_like(numerator, fill_value)
        return fill_value
