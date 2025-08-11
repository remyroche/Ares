import logging
from typing import Any

import pandas as pd

from src.utils.warning_symbols import (
    warning,
)


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "fill",
    fill_value: Any | None = 0,
) -> pd.DataFrame:
    """
    Handle missing data in a DataFrame with various strategies.
    Supported strategies: 'drop', 'fill', 'mean', 'median', 'mode', 'ffill', 'bfill', 'knn' (placeholder).
    Logs missing data rates and strategy used.
    """
    logger = logging.getLogger("data_cleaning")
    missing_rate = df.isna().mean().mean()
    logger.info(f"Missing data rate: {missing_rate:.2%} (strategy: {strategy})")
    if strategy == "drop":
        return df.dropna()
    if strategy == "fill":
        return df.fillna(fill_value)
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    if strategy == "mode":
        mode_vals = df.mode().iloc[0]
        return df.fillna(mode_vals)
    if strategy == "ffill":
        return df.fillna(method="ffill")
    if strategy == "bfill":
        return df.fillna(method="bfill")
    if strategy == "knn":
        print(warning("KNN imputation not implemented; falling back to mean."))
        return df.fillna(df.mean(numeric_only=True))
    return df
