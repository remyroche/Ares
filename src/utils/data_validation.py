"""
Data Validation Utilities for Feature Engineering

This module provides utilities for validating and cleaning data in feature engineering pipelines.
"""

from typing import Any, Optional, Union, overload
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _coerce_series_numeric(series: pd.Series, copy: bool = True) -> pd.Series:
    """
    Coerce a pandas Series to numeric dtype safely.
    
    Args:
        series: Input Series
        copy: Whether to copy the data
        
    Returns:
        Numeric Series
    """
    try:
        s = series.copy() if copy else series
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        return s
    except Exception:
        return series

def safe_pct_change(
    series: pd.Series,
    periods: int = 1,
    freq: Optional[str] = None,
    fill_method: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs: Any,
) -> pd.Series:
    """
    Calculate percentage change safely, handling infinite values and NaNs.
    
    Args:
        series: Input Series
        periods: Number of periods to shift
        freq: Frequency string
        fill_method: Method to fill NaNs
        limit: Limit for forward/backward fill
        **kwargs: Additional arguments for pct_change
        
    Returns:
        Percentage change Series with infinite values replaced by 0
    """
    try:
        if fill_method:
            series = series.fillna(method=fill_method, limit=limit)
        
        s = _coerce_series_numeric(series)
        pct_change = s.pct_change(periods=periods, freq=freq, **kwargs)
        
        # Replace infinite values with 0
        inf_count = np.isinf(pct_change).sum()
        if int(inf_count) > 0:
            logger.warning(
                "Found %d infinite values in pct_change calculation - replacing with 0",
                int(inf_count),
            )
            pct_change = pct_change.replace([np.inf, -np.inf], 0)
        
        return pct_change.fillna(0)
    except Exception as e:
        logger.exception("Error in safe_pct_change: %s", e)
        return pd.Series(0, index=series.index, dtype="float64")

def safe_log_returns(
    series: pd.Series,
    periods: int = 1,
    freq: Optional[str] = None,
    fill_method: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs: Any,
) -> pd.Series:
    """
    Calculate log returns safely, handling infinite values and NaNs.
    
    Args:
        series: Input Series
        periods: Number of periods to shift
        freq: Frequency string
        fill_method: Method to fill NaNs
        limit: Limit for forward/backward fill
        **kwargs: Additional arguments for pct_change
        
    Returns:
        Log returns Series with infinite values replaced by 0
    """
    try:
        if fill_method:
            series = series.fillna(method=fill_method, limit=limit)
        
        s = _coerce_series_numeric(series)
        pct_change = s.pct_change(periods=periods, freq=freq, **kwargs)
        log_returns = np.log1p(pct_change)
        
        # Replace infinite values with 0
        inf_count = np.isinf(log_returns).sum()
        if int(inf_count) > 0:
            logger.warning(
                "Found %d infinite values in log_returns calculation - replacing with 0",
                int(inf_count),
            )
            log_returns = log_returns.replace([np.inf, -np.inf], 0)
        
        return log_returns.fillna(0)
    except Exception as e:
        logger.exception("Error in safe_log_returns: %s", e)
        return pd.Series(0, index=series.index, dtype="float64")

def validate_dataframe_for_ml(
    df: pd.DataFrame,
    context: str = "dataframe",
    clip_extreme_values: bool = True,
    max_abs_value: float = 1e6,
) -> pd.DataFrame:
    """
    Validate and clean DataFrame for machine learning, handling common issues.
    
    Args:
        df: Input DataFrame
        context: Context string for logging
        clip_extreme_values: Whether to clip extreme values
        max_abs_value: Maximum absolute value threshold
        
    Returns:
        Cleaned DataFrame suitable for ML
    """
    try:
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found in DataFrame for context: %s", context)
            return df_clean

        # Replace infinities
        inf_count = np.isinf(df_clean[numeric_cols]).sum().sum()
        if int(inf_count) > 0:
            logger.warning(
                "Found %d infinite values in %s - replacing with 0",
                int(inf_count),
                context,
            )
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], 0)

        # Clip extremes
        if clip_extreme_values:
            extreme_count = (np.abs(df_clean[numeric_cols]) > max_abs_value).sum().sum()
            if int(extreme_count) > 0:
                logger.warning(
                    "Found %d extreme values (>±%.3f) in %s - clipping",
                    int(extreme_count),
                    max_abs_value,
                    context,
                )
                df_clean[numeric_cols] = np.clip(df_clean[numeric_cols], -max_abs_value, max_abs_value)

        # Fill NaNs
        nan_count = df_clean[numeric_cols].isna().sum().sum()
        if int(nan_count) > 0:
            logger.warning("Found %d NaN values in %s - filling with 0", int(nan_count), context)
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

        # Verify final state
        final_inf_count = np.isinf(df_clean[numeric_cols]).sum().sum()
        final_nan_count = df_clean[numeric_cols].isna().sum().sum()
        
        if int(final_inf_count) == 0 and int(final_nan_count) == 0:
            logger.info("✅ Data validation passed for %s: %s", context, df_clean.shape)
        else:
            logger.error(
                "🚨 Data validation residuals for %s: %d inf, %d NaN",
                context,
                int(final_inf_count),
                int(final_nan_count),
            )
        
        return df_clean
    except Exception as e:
        logger.exception("Error in validate_dataframe_for_ml for %s: %s", context, e)
        return df

# Type alias for numeric-like objects
NumberLike = Union[pd.Series, np.ndarray, float, int]

def safe_division(
    numerator: NumberLike,
    denominator: NumberLike,
    fill_value: float = 0.0,
    context: str = "division",
) -> NumberLike:
    """
    Perform safe division, handling zero denominators and infinite results.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use for invalid results
        context: Context string for logging
        
    Returns:
        Division result with invalid values replaced
    """
    try:
        # Series / Series
        if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
            with np.errstate(divide="ignore", invalid="ignore"):
                result = numerator / denominator
            
            zeros = (denominator == 0).sum()
            smalls = ((denominator != 0) & (np.abs(denominator) < 1e-12)).sum()
            
            if int(zeros + smalls) > 0:
                logger.warning(
                    "Found %d zero and %d very small denominators in %s",
                    int(zeros),
                    int(smalls),
                    context,
                )
                result = result.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
            
            return result

        # ndarray / scalars
        elif isinstance(numerator, (np.ndarray, float, int)) and isinstance(
            denominator, (np.ndarray, float, int),
        ):
            num_arr = np.asarray(numerator)
            den_arr = np.asarray(denominator)
            safe_mask = np.abs(den_arr) > 1e-12
            
            out = np.full_like(num_arr, fill_value, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[safe_mask] = num_arr[safe_mask] / den_arr[safe_mask]
            
            return out if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray) else float(out)

        # Mixed types -> coerce to numpy and compute
        else:
            num_arr = np.asarray(numerator)
            den_arr = np.asarray(denominator)
            safe_mask = np.abs(den_arr) > 1e-12
            
            out = np.full_like(num_arr, fill_value, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[safe_mask] = num_arr[safe_mask] / den_arr[safe_mask]
            
            return out
            
    except Exception as e:
        logger.exception("Error in safe_division for %s: %s", context, e)
        
        # Return appropriate type
        if isinstance(numerator, pd.Series):
            return pd.Series(fill_value, index=numerator.index)
        if isinstance(numerator, np.ndarray):
            return np.full_like(numerator, fill_value, dtype=float)
        return fill_value

def validate_numeric_range(
    series: pd.Series,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    context: str = "series",
) -> pd.Series:
    """
    Validate numeric range and clip values if needed.
    
    Args:
        series: Input Series
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        context: Context string for logging
        
    Returns:
        Validated Series with clipped values
    """
    try:
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning("Series %s is not numeric, skipping range validation", context)
            return series
        
        original_count = len(series)
        clipped_count = 0
        
        if min_val is not None:
            below_min = (series < min_val).sum()
            if below_min > 0:
                logger.warning(
                    "Found %d values below minimum %.3f in %s - clipping",
                    int(below_min),
                    min_val,
                    context,
                )
                series = series.clip(lower=min_val)
                clipped_count += below_min
        
        if max_val is not None:
            above_max = (series > max_val).sum()
            if above_max > 0:
                logger.warning(
                    "Found %d values above maximum %.3f in %s - clipping",
                    int(above_max),
                    max_val,
                    context,
                )
                series = series.clip(upper=max_val)
                clipped_count += above_max
        
        if clipped_count > 0:
            logger.info(
                "Range validation for %s: clipped %d values (%.1f%%)",
                context,
                clipped_count,
                (clipped_count / original_count) * 100,
            )
        
        return series
        
    except Exception as e:
        logger.exception("Error in validate_numeric_range for %s: %s", context, e)
        return series

def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5,
    context: str = "series",
) -> pd.Series:
    """
    Detect outliers in a numeric series using various methods.
    
    Args:
        series: Input Series
        method: Detection method ("iqr", "zscore", "isolation_forest")
        threshold: Threshold for outlier detection
        context: Context string for logging
        
    Returns:
        Boolean Series indicating outliers
    """
    try:
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning("Series %s is not numeric, cannot detect outliers", context)
            return pd.Series(False, index=series.index)
        
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > threshold
            
        else:
            logger.warning("Unknown outlier detection method: %s, using IQR", method)
            return detect_outliers(series, method="iqr", threshold=threshold, context=context)
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.info(
                "Detected %d outliers (%.1f%%) in %s using %s method",
                int(outlier_count),
                (outlier_count / len(series)) * 100,
                context,
                method,
            )
        
        return outliers
        
    except Exception as e:
        logger.exception("Error in detect_outliers for %s: %s", context, e)
        return pd.Series(False, index=series.index)

def validate_categorical_consistency(
    series: pd.Series,
    expected_categories: Optional[list] = None,
    context: str = "series",
) -> pd.Series:
    """
    Validate categorical consistency and clean if needed.
    
    Args:
        series: Input Series
        expected_categories: Expected category values
        context: Context string for logging
        
    Returns:
        Cleaned categorical Series
    """
    try:
        if expected_categories is not None:
            # Check for unexpected categories
            unexpected = ~series.isin(expected_categories)
            unexpected_count = unexpected.sum()
            
            if unexpected_count > 0:
                logger.warning(
                    "Found %d unexpected category values in %s",
                    int(unexpected_count),
                    context,
                )
                # Replace unexpected values with NaN
                series = series.where(series.isin(expected_categories))
        
        # Convert to categorical if not already
        if not pd.api.types.is_categorical_dtype(series):
            series = series.astype("category")
            logger.info("Converted %s to categorical dtype", context)
        
        return series
        
    except Exception as e:
        logger.exception("Error in validate_categorical_consistency for %s: %s", context, e)
        return series
