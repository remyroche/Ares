"""
Feature Smoothing Utilities for Regime Models

Provides smoothed features and rolling aggregates to reduce jittery predictions
in tree-based models.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
import re


def _is_smoothed_feature(feature_name: str) -> bool:
    """
    Check if a feature name indicates it's already smoothed.
    
    Args:
        feature_name: Feature name to check
        
    Returns:
        True if feature appears to be smoothed
    """
    if not feature_name:
        return False
    
    # Patterns that indicate smoothed features
    smoothed_patterns = [
        r'_ma\d+$',      # Moving average: feature_ma3, feature_ma5, etc.
        r'_std\d+$',     # Rolling std: feature_std3, feature_std5, etc.
        r'_ewm[\d.]+$',  # EWM: feature_ewm0.3, feature_ewm0.5, etc.
        r'_smooth',      # Generic smooth suffix
        r'_smoothed',    # Generic smoothed suffix
    ]
    
    for pattern in smoothed_patterns:
        if re.search(pattern, feature_name):
            return True
    
    return False


def add_smoothed_features(
    X: np.ndarray,
    window_sizes: Optional[list] = None,
    feature_names: Optional[list] = None
) -> tuple[np.ndarray, list]:
    """
    Add smoothed features using rolling aggregates.
    
    Only smooths original (non-smoothed) features to avoid double-smoothing.
    
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
    
    # Identify which features are already smoothed
    if feature_names:
        smoothed_indices = [
            i for i, name in enumerate(feature_names)
            if _is_smoothed_feature(name)
        ]
        original_indices = [
            i for i, name in enumerate(feature_names)
            if not _is_smoothed_feature(name)
        ]
        original_feature_names = [feature_names[i] for i in original_indices]
        smoothed_feature_names = [feature_names[i] for i in smoothed_indices]
    else:
        # If no feature names provided, assume all are original
        original_indices = list(range(n_features))
        smoothed_indices = []
        original_feature_names = None
        smoothed_feature_names = []
    
    # Extract original and already-smoothed features
    if original_indices:
        X_original = X[:, original_indices]
        n_original_features = len(original_indices)
    else:
        # All features are already smoothed, return as-is
        return X, feature_names if feature_names else [f'feature_{i}' for i in range(n_features)]
    
    X_smoothed_existing = X[:, smoothed_indices] if smoothed_indices else None
    
    # Convert original features to DataFrame for easier rolling operations
    df_original = pd.DataFrame(X_original)
    
    smoothed_features = []
    smoothed_names = []
    
    # Keep original features
    smoothed_features.append(X_original)
    if original_feature_names:
        smoothed_names.extend(original_feature_names)
    else:
        smoothed_names.extend([f'feature_{i}' for i in range(n_original_features)])
    
    # Keep already-smoothed features as-is
    if X_smoothed_existing is not None and len(X_smoothed_existing.shape) > 0:
        smoothed_features.append(X_smoothed_existing)
        smoothed_names.extend(smoothed_feature_names)
    
    # Add smoothed features for each window size (only for original features)
    for window in window_sizes:
        if window > n_samples:
            continue
        
        # Rolling mean (only for original features)
        rolling_mean = df_original.rolling(window=window, min_periods=1, center=True).mean().values
        smoothed_features.append(rolling_mean)
        if original_feature_names:
            smoothed_names.extend([f'{name}_ma{window}' for name in original_feature_names])
        else:
            smoothed_names.extend([f'feature_{i}_ma{window}' for i in range(n_original_features)])
        
        # Rolling std (volatility measure) - only for original features
        rolling_std = df_original.rolling(window=window, min_periods=1, center=True).std().values
        # Fill NaN with 0 for first samples
        rolling_std = np.nan_to_num(rolling_std, nan=0.0)
        smoothed_features.append(rolling_std)
        if original_feature_names:
            smoothed_names.extend([f'{name}_std{window}' for name in original_feature_names])
        else:
            smoothed_names.extend([f'feature_{i}_std{window}' for i in range(n_original_features)])
    
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
    
    Only smooths original (non-smoothed) features to avoid double-smoothing.
    
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
    
    # Identify which features are already smoothed
    if feature_names:
        smoothed_indices = [
            i for i, name in enumerate(feature_names)
            if _is_smoothed_feature(name)
        ]
        original_indices = [
            i for i, name in enumerate(feature_names)
            if not _is_smoothed_feature(name)
        ]
        original_feature_names = [feature_names[i] for i in original_indices]
        smoothed_feature_names = [feature_names[i] for i in smoothed_indices]
    else:
        # If no feature names provided, assume all are original
        original_indices = list(range(df.shape[1]))
        smoothed_indices = []
        original_feature_names = None
        smoothed_feature_names = []
    
    # Extract original and already-smoothed features
    if original_indices:
        df_original = df.iloc[:, original_indices]
    else:
        # All features are already smoothed, return as-is
        if isinstance(X, np.ndarray):
            return X, feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        else:
            return X.values, feature_names if feature_names else list(X.columns)
    
    df_smoothed_existing = df.iloc[:, smoothed_indices] if smoothed_indices else None
    
    # Apply EWM smoothing only to original features
    ewm_features = df_original.ewm(alpha=alpha, adjust=False).mean().values
    
    # Combine features: original + already-smoothed + new EWM smoothed
    features_list = [df_original.values]
    
    if df_smoothed_existing is not None and len(df_smoothed_existing.shape) > 0:
        features_list.append(df_smoothed_existing.values)
    
    features_list.append(ewm_features)
    smoothed_X = np.hstack(features_list)
    
    # Build feature names
    if original_feature_names:
        smoothed_names = original_feature_names.copy()
        if smoothed_feature_names:
            smoothed_names.extend(smoothed_feature_names)
        ewm_names = [f'{name}_ewm{alpha}' for name in original_feature_names]
        smoothed_names.extend(ewm_names)
    else:
        n_original = len(original_indices)
        smoothed_names = [f'feature_{i}' for i in range(n_original)]
        if smoothed_feature_names:
            smoothed_names.extend(smoothed_feature_names)
        ewm_names = [f'feature_{i}_ewm{alpha}' for i in range(n_original)]
        smoothed_names.extend(ewm_names)
    
    return smoothed_X, smoothed_names
