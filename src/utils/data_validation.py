# src/utils/data_validation.py


from typing import Any, Optional, Union, overload
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _coerce_series_numeric(series: pd.Series, *, copy: bool = False) -> pd.Series:
    """Coerce a Series to numeric where possible, preserving index."""
    try:
        s = series.copy() if copy else series
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        return s
    except Exception:
        return series




def validate_dataframe_for_ml(
    df: pd.DataFrame,
    *,
    context: str = "unknown",
    clip_extreme_values: bool = True,
    max_abs_value: float = 1000.0,
) -> pd.DataFrame:
    """
    Validate and clean DataFrame for machine learning models:
    - Keep only numeric columns for cleaning operations
    - Replace infinite values with 0
    - Optionally clip extreme absolute values
    - Fill NaN values with 0
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


NumberLike = Union[pd.Series, np.ndarray, float, int]

