"""
Layer 3 Utility Functions - Enhanced with Studentized HAR-Residual Logic

Helper functions and utilities for Layer 3 operations.
Includes Studentized HAR-Residual calculation for advanced target generation.
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

def calculate_studentized_har_target(
    returns: pd.Series,
    volatility: pd.Series,
    daily_period: int = 96,
    weekly_period: int = 480,
    monthly_period: int = 1920,
    ewma_span: int = 20
) -> pd.Series:
    """
    Calculate Studentized HAR-Residual Target.

    Step 1: HAR-Residualization (The "Surprise"):
    Strip out the expected return component based on Daily, Weekly, and Monthly variance.
    ϵ_t = Y_t - Ŷ_HAR

    Step 2: Studentization (The "Scale"):
    Divide the residual by the rolling volatility of the residuals.
    Y_Innovation = ϵ_t / σ_ϵ,t

    Args:
        returns: Return series (Y_t)
        volatility: Volatility series (σ_t) - used for generating variance features
        daily_period: Period for daily variance (short-term)
        weekly_period: Period for weekly variance (medium-term)
        monthly_period: Period for monthly variance (long-term)
        ewma_span: Span for EWMA volatility of residuals

    Returns:
        Studentized HAR-Residual Target series
    """
    # Ensure inputs are Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(volatility, pd.Series):
        volatility = pd.Series(volatility, index=returns.index)

    # --- Step 1: Feature Generation (Variance Components) ---
    # Using realized variance proxies (squared returns or squared volatility)
    # Volatility input is usually rolling std dev. Squared is variance.
    # We construct HAR components:
    # RV_d (Daily)
    # RV_w (Weekly)
    # RV_m (Monthly)

    # Construct lagged variance features to predict current return
    # We use realized variance over the past window
    var_d = (volatility ** 2).rolling(window=daily_period, min_periods=1).mean()
    var_w = (volatility ** 2).rolling(window=weekly_period, min_periods=1).mean()
    var_m = (volatility ** 2).rolling(window=monthly_period, min_periods=1).mean()

    # Lag features by 1 step to ensure no lookahead for prediction
    X = pd.DataFrame({
        'var_d': var_d.shift(1),
        'var_w': var_w.shift(1),
        'var_m': var_m.shift(1)
    }).fillna(0)

    # --- Step 2: Rolling OLS to estimate Expected Return ---
    # We use a rolling window OLS to estimate the relationship between past variance and return.
    # Using a large window (e.g., 2000 bars) or expanding window.
    # For efficiency and stability, we'll use Recursive Least Squares (RLS) via expanding window
    # or a Rolling OLS if feasible. Given performance constraints, we can use a simpler approach:
    # We'll use sklearn's LinearRegression in a rolling fashion or statsmodels RollingOLS.
    # To keep dependencies light and fast, we can use a simpler expanding window proxy
    # or just assume a global relationship if stationarity holds (but it rarely does).

    # Let's use a rolling window regression (window = monthly_period * 2 for stability)
    # Implementation using statsmodels if available, else expanding window loop (slow).
    # Faster approach: Rolling correlation/covariance math.

    # However, simpler robust approach for target generation in labeling (which happens once):
    # Use a large rolling window (e.g. 2000) for OLS.
    try:
        from statsmodels.regression.rolling import RollingOLS
        import statsmodels.api as sm

        # Add constant
        X_const = sm.add_constant(X)
        model = RollingOLS(returns, X_const, window=monthly_period * 2)
        params = model.fit(params_only=True).params

        # Compute predicted return: Y_hat = sum(X * params)
        y_hat = (X_const * params).sum(axis=1)

    except ImportError:
        # Fallback if statsmodels not installed: Simple expanding window mean subtraction (dumb proxy)
        # or just standardizing by volatility directly (fallback to old method)
        # But let's try to do a simple recursive calculation or block-based OLS.

        # Block-based fallback: Re-fit every N bars
        y_hat = pd.Series(0.0, index=returns.index)
        window = monthly_period * 2
        step = daily_period

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        # Initial fit
        # We can't do true rolling easily without library, so we'll just do a global fit
        # on past data to prevent lookahead?
        # Or just fit on the whole dataset? Fitting on whole dataset introduces lookahead bias
        # for the 'Surprise' component.
        # Let's use an expanding window approach with a stride.

        # FAST APPROXIMATION:
        # Expected return due to variance is likely small/noisy.
        # The main 'HAR' effect is usually on Volatility prediction, not Return prediction.
        # If the user insists on "strip out expected return based on variance",
        # and we lack fast rolling OLS, we might assume beta=0 (Efficient Market)
        # and just subtract moving average of returns?
        # But the prompt is specific.

        # Let's implement a simple expanding OLS:
        # Iterate in chunks.
        # This is slow in Python.

        # Alternative: Use the entire past expanding window mean of returns conditioned on variance?
        # Too complex.

        # Let's try to assume we can use the whole dataset for the *structure*
        # (betas constant) but locally adaptive?
        # No, strict HAR requires rolling.

        # Let's implement a simplified vectorised Rolling OLS using numpy stride tricks if needed,
        # but statsmodels is likely available in this environment.
        # If imports fail inside try, we log warning and fallback to 0 expected return (pure residual).
        logger.warning("Statsmodels RollingOLS not available/failed. Using raw returns as residual.")
        y_hat = pd.Series(0.0, index=returns.index)

    # Calculate Residual (The Surprise)
    # Fill NaNs in y_hat (start of window) with 0
    y_hat = y_hat.fillna(0)
    residuals = returns - y_hat

    # --- Step 3: Studentization (The Scale) ---
    # Divide residual by rolling volatility of residuals (20-period EWMA)
    # Note: The prompt says "20-period EWMA of the residuals themselves".
    # We should calculate the std dev of the residuals.
    # EWM std:
    resid_vol = residuals.ewm(span=ewma_span, adjust=False).std()

    # Avoid division by zero
    resid_vol = resid_vol.replace(0, 1e-6).fillna(1e-6)

    # Studentized Residual
    studentized_residuals = residuals / resid_vol

    # Clip extreme values
    studentized_residuals = studentized_residuals.clip(-10, 10)

    return studentized_residuals

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
