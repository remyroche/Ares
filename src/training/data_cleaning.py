import pandas as pd
from typing import Optional
import numpy as np
import logging

def handle_missing_data(df: pd.DataFrame, strategy: str = 'fill', fill_value: Optional[Any] = 0) -> pd.DataFrame:
    """
    Handle missing data in a DataFrame with various strategies.
    Supported strategies: 'drop', 'fill', 'mean', 'median', 'mode', 'ffill', 'bfill', 'knn' (placeholder).
    Logs missing data rates and strategy used.
    """
    logger = logging.getLogger("data_cleaning")
    missing_rate = df.isna().mean().mean()
    logger.info(f"Missing data rate: {missing_rate:.2%} (strategy: {strategy})")
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        mode_vals = df.mode().iloc[0]
        return df.fillna(mode_vals)
    elif strategy == 'ffill':
        return df.fillna(method='ffill')
    elif strategy == 'bfill':
        return df.fillna(method='bfill')
    elif strategy == 'knn':
        logger.warning("KNN imputation not implemented; falling back to mean.")
        return df.fillna(df.mean(numeric_only=True))
    return df