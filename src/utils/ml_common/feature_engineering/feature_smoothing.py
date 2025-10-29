"""
Feature Smoothing Utilities for Regime Models

Provides smoothed features and rolling aggregates to reduce jittery predictions
in tree-based models.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any


def add_smoothed_features(
    X: np.ndarray,
    window_sizes: Optional[list] = None,
    feature_names: Optional[list] = None
) -> tuple[np.ndarray, list]:
    """
    Add smoothed features using rolling aggregates.
    
    Args:
        X: Input features (n_samples, n_features)
        window_sizes: List of window sizes for smoothing (default: [3, 5, 7])
        feature_names: Original feature names (optional)
        
    Returns:
        Tuple of (smoothed_X, updated_feature_names)
    """
    if window_sizes is None:
        window_sizes = [3, 5, 7]
    
    n_samples, n_features = X.shape
    
    # Convert to DataFrame for easier rolling operations
    df = pd.DataFrame(X)
    
    smoothed_features = []
    smoothed_names = []
    
    # Keep original features
    smoothed_features.append(X)
    if feature_names:
        smoothed_names.extend(feature_names)
    else:
        smoothed_names.extend([f'feature_{i}' for i in range(n_features)])
    
    # Add smoothed features for each window size
    for window in window_sizes:
        if window > n_samples:
            continue
        
        # Rolling mean
        rolling_mean = df.rolling(window=window, min_periods=1, center=True).mean().values
        smoothed_features.append(rolling_mean)
        if feature_names:
            smoothed_names.extend([f'{name}_ma{window}' for name in feature_names])
        else:
            smoothed_names.extend([f'feature_{i}_ma{window}' for i in range(n_features)])
        
        # Rolling std (volatility measure)
        rolling_std = df.rolling(window=window, min_periods=1, center=True).std().values
        # Fill NaN with 0 for first samples
        rolling_std = np.nan_to_num(rolling_std, nan=0.0)
        smoothed_features.append(rolling_std)
        if feature_names:
            smoothed_names.extend([f'{name}_std{window}' for name in feature_names])
        else:
            smoothed_names.extend([f'feature_{i}_std{window}' for i in range(n_features)])
    
    # Combine all features
    smoothed_X = np.hstack(smoothed_features)
    
    return smoothed_X, smoothed_names


def apply_ewm_smoothing(
    X: Union[np.ndarray, pd.DataFrame],
    alpha: float = 0.3,
    feature_names: Optional[list] = None
) -> tuple[np.ndarray, list]:
    """
    Apply exponential weighted moving average smoothing.
    
    Args:
        X: Input features
        alpha: Smoothing factor (0 < alpha <= 1), smaller = more smoothing
        feature_names: Original feature names (optional)
        
    Returns:
        Tuple of (smoothed_X, updated_feature_names)
    """
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X)
    else:
        df = X.copy()
    
    # Apply EWM smoothing
    ewm_features = df.ewm(alpha=alpha, adjust=False).mean().values
    
    # Combine original and smoothed
    smoothed_X = np.hstack([X, ewm_features])
    
    if feature_names:
        ewm_names = [f'{name}_ewm{alpha}' for name in feature_names]
        smoothed_names = feature_names + ewm_names
    else:
        n_features = X.shape[1]
        ewm_names = [f'feature_{i}_ewm{alpha}' for i in range(n_features)]
        smoothed_names = [f'feature_{i}' for i in range(n_features)] + ewm_names
    
    return smoothed_X, smoothed_names
