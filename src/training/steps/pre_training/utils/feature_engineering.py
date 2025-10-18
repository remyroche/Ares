"""
Feature engineering utilities for pre-training steps.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def create_lag_features(df: pd.DataFrame, 
                       columns: List[str],
                       lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Create lagged features for time series data.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    df_lagged = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    return df_lagged

def create_rolling_features(df: pd.DataFrame,
                          columns: List[str],
                          windows: List[int] = [5, 10, 20, 50],
                          functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """Create rolling window features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create rolling features for
        windows: List of window sizes
        functions: List of functions to apply
        
    Returns:
        DataFrame with rolling features
    """
    df_rolling = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                for func in functions:
                    if func == 'mean':
                        df_rolling[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
                    elif func == 'std':
                        df_rolling[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
                    elif func == 'min':
                        df_rolling[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
                    elif func == 'max':
                        df_rolling[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
    
    return df_rolling
