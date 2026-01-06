"""
Layer 3 Utility Functions

Helper functions and utilities for Layer 3 operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from functools import lru_cache
import hashlib
from numba import njit, prange

logger = logging.getLogger(__name__)

EPS = 1e-12

@njit
def calculate_sample_weights_efficient(returns: np.ndarray, volatility: np.ndarray,
                                     layer1_weights: np.ndarray = None,
                                     layer2_weights: np.ndarray = None) -> np.ndarray:
    """
    JIT-compiled efficient sample weight calculation combining multiple sources.

    Args:
        returns: Return series
        volatility: Volatility series
        layer1_weights: Optional layer 1 weights
        layer2_weights: Optional layer 2 weights

    Returns:
        Combined and normalized sample weights
    """
    n_samples = len(returns)

    # Base volatility weights (inverse variance)
    vol_safe = np.maximum(volatility, 1e-6)
    vol_weights = 1.0 / (vol_safe ** 2)

    # Initialize combined weights
    combined_weights = vol_weights.copy()

    # Add layer 1 weights if provided
    if layer1_weights is not None and len(layer1_weights) == n_samples:
        combined_weights *= layer1_weights

    # Add layer 2 weights if provided
    if layer2_weights is not None and len(layer2_weights) == n_samples:
        combined_weights *= layer2_weights

    # Apply finalize_sample_weights logic inline for efficiency
    # MAD scaling
    weights_median = np.median(combined_weights)
    weights_mad = np.median(np.abs(combined_weights - weights_median))

    if weights_mad > 0:
        scaled_weights = (combined_weights - weights_median) / weights_mad
    else:
        scaled_weights = combined_weights - weights_median

    # Center at 1.0 and ensure positive
    final_weights = np.maximum(scaled_weights + 1.0, 0.1)

    return final_weights

@njit
def finalize_sample_weights(weights: np.ndarray) -> np.ndarray:
    """
    Finalize sample weights using MAD scaling and centering at 1.0.
    
    This is the standard weight finalization used across layers.
    JIT compiled for performance.
    """
    # MAD scaling
    weights_median = np.median(weights)
    weights_mad = np.median(np.abs(weights - weights_median))
    
    if weights_mad > 0:
        scaled_weights = (weights - weights_median) / weights_mad
    else:
        scaled_weights = weights - weights_median
    
    # Center at 1.0 (add 1 to make mean around 1)
    final_weights = scaled_weights + 1.0
    
    # Ensure positive weights
    final_weights = np.maximum(final_weights, 0.1)
    
    return final_weights

@njit
def calculate_alpha_target(returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    Calculate volatility-standardized alpha target.
    
    Args:
        returns: Raw returns series
        volatility: Volatility series
        
    Returns:
        Volatility-standardized alpha target

    JIT compiled for performance.
    """
    # Clip volatility to avoid explosion
    vol_safe = np.clip(volatility, 1e-4, None)
    
    # Volatility-standardized returns
    alpha_target = returns / vol_safe
    
    # Clip extreme values
    alpha_target = np.clip(alpha_target, -10, 10)
    
    return alpha_target

def is_pareto_efficient_numba(costs: np.ndarray) -> np.ndarray:
    """
    Return the Pareto-efficient points.
    
    Args:
        costs: Cost matrix (n_points, n_objectives)
        
    Returns:
        Boolean array indicating Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[i] = not np.any(np.all(costs >= c, axis=1)) & np.any(np.any(costs > c, axis=1))
    return is_efficient

@njit
def validate_and_clean_features_jit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled feature validation and cleaning.
    Removes NaN/inf values and ensures data quality.

    Args:
        X: Feature matrix
        y: Target array

    Returns:
        Tuple of (cleaned_X, cleaned_y)
    """
    # Find valid rows (no NaN/inf in any column)
    valid_rows = np.ones(X.shape[0], dtype=np.bool_)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not np.isfinite(X[i, j]):
                valid_rows[i] = False
                break

    # Also check target
    for i in range(y.shape[0]):
        if not np.isfinite(y[i]):
            valid_rows[i] = False

    # Filter to valid rows
    if np.any(~valid_rows):
        X_clean = X[valid_rows, :]
        y_clean = y[valid_rows]
    else:
        X_clean = X
        y_clean = y

    return X_clean, y_clean

@njit
def apply_smart_activation(signal: np.ndarray, volatility: np.ndarray, activation_type: str) -> np.ndarray:
    """
    Apply activation function with volatility scaling.
    
    Args:
        signal: Input signal
        volatility: Volatility for scaling
        activation_type: Type of activation
        
    Returns:
        Activated signal

    JIT compiled for performance.
    """
    eps = 1e-12

    if activation_type == 'linear':
        return signal
    elif activation_type == 'tanh_dynamic':
        return np.tanh(signal / (volatility + eps))
    elif activation_type == 'cubic_regime':
        return np.tanh(signal**3 / (volatility + eps))
    elif activation_type == 'sigmoid':
        return 1 / (1 + np.exp(-signal / (volatility + eps)))
    else:
        return signal

def fast_cmi_proxy(
    X: pd.DataFrame,
    y: pd.Series,
    base_predictions: pd.Series,
    top_percentile: float = 0.5
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fast CMI approximation using correlation and mutual information.
    
    Args:
        X: Feature matrix
        y: Target series
        base_predictions: Base model predictions
        top_percentile: Percentile of features to keep
        
    Returns:
        Tuple of (filtered features, selected indices)
    """
    # Stage 1: Correlation-based pre-filtering (O(f))
    corr_scores = np.abs([np.corrcoef(X.iloc[:, i], y)[0, 1] for i in range(X.shape[1])])
    corr_scores = np.nan_to_num(corr_scores, nan=0.0)
    
    # Select top percentile by correlation
    top_corr_idx = np.argsort(corr_scores)[-int(top_percentile * X.shape[1]):]
    X_filtered = X.iloc[:, top_corr_idx]
    
    # Stage 2: Conditional correlation filtering
    conditional_scores = []
    for i in range(len(top_corr_idx)):
        feature_col = X_filtered.columns[i]
        feature_values = X_filtered[feature_col].values
        
        # Calculate conditional correlation
        residual_y = y - base_predictions
        residual_feature = feature_values - base_predictions
        
        if np.std(residual_feature) > EPS:
            conditional_corr = np.corrcoef(residual_feature, residual_y)[0, 1]
            conditional_scores.append(abs(conditional_corr))
        else:
            conditional_scores.append(0.0)
    
    # Select top features by conditional correlation
    final_top_idx = np.argsort(conditional_scores)[-int(0.4 * len(conditional_scores)):]
    selected_features = X_filtered.iloc[:, final_top_idx]
    selected_indices = top_corr_idx[final_top_idx]
    
    return selected_features, selected_indices

def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute hash of dataframe for caching purposes.
    
    Args:
        df: Input dataframe
        
    Returns:
        Hash string
    """
    # Use shape, columns, and first/last few rows for hash
    hash_data = (
        str(df.shape) +
        str(df.columns.tolist()) +
        str(df.head().values.tobytes()) +
        str(df.tail().values.tobytes())
    )
    return hashlib.md5(hash_data.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_correlation_matrix(data_hash: str, n_features: int) -> np.ndarray:
    """
    Cached correlation matrix computation.
    
    Args:
        data_hash: Hash of the data
        n_features: Number of features
        
    Returns:
        Identity matrix (placeholder - would compute actual correlation)
    """
    # In practice, this would compute and cache the actual correlation matrix
    return np.eye(n_features)

def validate_feature_matrix(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and clean feature matrix and target with optimized vectorized operations.
    
    Args:
        X: Feature matrix
        y: Target series
        
    Returns:
        Tuple of (cleaned X, cleaned y)
    """
    # Vectorized replacement of inf values
    X_values = X.values
    inf_mask = ~np.isfinite(X_values)
    if np.any(inf_mask):
        X_values[inf_mask] = np.nan
        X_clean = pd.DataFrame(X_values, index=X.index, columns=X.columns)
    else:
        X_clean = X

    # Vectorized NaN ratio calculation and feature filtering
    nan_threshold = 0.5  # Remove features with >50% NaNs
    nan_counts = np.isnan(X_clean.values).sum(axis=0)
    nan_ratios = nan_counts / len(X_clean)
    valid_features = nan_ratios <= nan_threshold
    
    if not np.all(valid_features):
        X_clean = X_clean.iloc[:, valid_features]
    
    # Optimized forward fill and NaN replacement
    if X_clean.isna().any().any():
        # Use vectorized operations for forward fill
        X_values = X_clean.values
        # Forward fill along rows (time dimension)
        for col in range(X_values.shape[1]):
            col_data = X_values[:, col]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask):
                # Forward fill implementation
                last_valid = np.nan
                for i in range(len(col_data)):
                    if not np.isnan(col_data[i]):
                        last_valid = col_data[i]
                    elif last_valid is not np.nan:
                        col_data[i] = last_valid

                # Fill any remaining NaNs with 0
                col_data[np.isnan(col_data)] = 0.0

        X_clean = pd.DataFrame(X_values, index=X_clean.index, columns=X_clean.columns)

    # Optimized index alignment using numpy operations
    x_index_set = set(X_clean.index)
    y_index_set = set(y.index)
    common_index = sorted(x_index_set & y_index_set)

    if len(common_index) < len(X_clean.index) or len(common_index) < len(y.index):
        # Create boolean mask for common indices
        x_mask = np.isin(X_clean.index, common_index)
        y_mask = np.isin(y.index, common_index)

        X_clean = X_clean.iloc[x_mask]
        y_clean = y.iloc[y_mask]
    else:
        y_clean = y

    # Vectorized target NaN removal
    if y_clean.isna().any():
        valid_mask = ~y_clean.isna()
        X_clean = X_clean.loc[valid_mask]
        y_clean = y_clean.loc[valid_mask]
    
    logger.info(f"Feature validation: {X.shape} -> {X_clean.shape}")
    
    return X_clean, y_clean

def calculate_feature_statistics(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive feature statistics.
    
    Args:
        X: Feature matrix
        
    Returns:
        Dictionary of feature statistics
    """
    stats = {
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'missing_ratio': (X.isna().sum() / len(X)).mean(),
        'feature_types': {
            'numeric': len(X.select_dtypes(include=[np.number]).columns),
            'categorical': len(X.select_dtypes(include=['object', 'category']).columns)
        },
        'correlation_stats': {}
    }
    
    # Correlation statistics
    numeric_X = X.select_dtypes(include=[np.number])
    if len(numeric_X.columns) > 1:
        corr_matrix = numeric_X.corr()
        
        # Remove diagonal
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        corr_values = corr_values[~np.isnan(corr_values)]
        
        if len(corr_values) > 0:
            stats['correlation_stats'] = {
                'mean_abs_correlation': np.mean(np.abs(corr_values)),
                'max_abs_correlation': np.max(np.abs(corr_values)),
                'high_corr_pairs': int(np.sum(np.abs(corr_values) > 0.9))
            }
    
    return stats

def create_feature_groups(features: List[str]) -> Dict[str, List[str]]:
    """
    Create feature groups for parallel processing.
    
    Args:
        features: List of feature names
        
    Returns:
        Dictionary of feature groups
    """
    groups = {
        'ensemble_features': [f for f in features if 'ensemble' in f],
        'momentum_features': [f for f in features if 'momentum' in f],
        'volatility_features': [f for f in features if 'vol' in f or 'volatility' in f],
        'regime_features': [f for f in features if 'regime' in f],
        'layer0_features': [f for f in features if any(x in f for x in ['unified', 'adaptive', 'noise', 'filter'])],
        'layer1_features': [f for f in features if 'layer1' in f],
        'technical_features': [f for f in features if any(x in f for x in ['rsi', 'macd', 'bb_', 'atr'])],
        'time_features': [f for f in features if any(x in f for x in ['hour', 'day', 'time'])],
        'other_features': []
    }
    
    # Assign uncategorized features
    all_grouped = []
    for group_features in groups.values():
        if group_features:  # Skip empty lists
            all_grouped.extend(group_features)
    
    groups['other_features'] = [f for f in features if f not in all_grouped]
    
    return groups

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Memory-optimized dataframe
    """
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != 'object':
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    optimized_df[col] = optimized_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
    
    return optimized_df
