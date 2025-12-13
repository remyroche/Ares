"""
Distribution Shift Mitigation Utilities.

Provides features and utilities to detect and mitigate distribution shift
between training and production/filtered data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import warnings

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success
except ImportError:
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)


def compute_filter_propensity_scores(
    df: pd.DataFrame,
    filter_mask: pd.Series,
    feature_cols: List[str],
    cv_folds: int = 3,
) -> pd.Series:
    """
    Estimate the propensity of each sample to pass a filter.
    
    This enables the model to learn filter-aware patterns by including
    the propensity score as a feature. High propensity = likely to be
    in the filtered (training) set.
    
    Args:
        df: Full DataFrame (pre-filter)
        filter_mask: Boolean series indicating which rows pass the filter
        feature_cols: Feature columns to use for propensity estimation
        cv_folds: Number of cross-validation folds
        
    Returns:
        Series of propensity scores (0-1) for each row
    """
    if not SKLEARN_AVAILABLE:
        tprint_warning("sklearn not available - returning uniform propensity scores")
        return pd.Series(0.5, index=df.index)
    
    # Prepare features
    X = df[feature_cols].fillna(0).values
    y = filter_mask.astype(int).values
    
    if len(np.unique(y)) < 2:
        tprint_warning("Filter mask has only one class - returning uniform scores")
        return pd.Series(0.5, index=df.index)
    
    try:
        # Use Random Forest for propensity estimation (fast, handles nonlinear)
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        
        # Cross-validated predictions to avoid overfitting
        propensity_scores = cross_val_predict(
            clf, X, y, cv=cv_folds, method='predict_proba'
        )[:, 1]
        
        propensity_series = pd.Series(propensity_scores, index=df.index)
        
        tprint_success(f"Computed filter propensity scores (mean={propensity_scores.mean():.3f})")
        return propensity_series
        
    except Exception as e:
        tprint_warning(f"Failed to compute propensity scores: {e}")
        return pd.Series(0.5, index=df.index)


def compute_distribution_shift_metrics(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Compute metrics quantifying distribution shift between train and test sets.
    
    Args:
        df_train: Training DataFrame
        df_test: Test/production DataFrame
        feature_cols: Feature columns to compare
        
    Returns:
        Dictionary with shift metrics per feature and overall summary.
    """
    shift_metrics = {}
    
    for col in feature_cols:
        if col not in df_train.columns or col not in df_test.columns:
            continue
            
        train_vals = df_train[col].dropna()
        test_vals = df_test[col].dropna()
        
        if len(train_vals) < 10 or len(test_vals) < 10:
            continue
        
        # Mean shift
        mean_shift = abs(train_vals.mean() - test_vals.mean())
        
        # Std ratio
        std_train = train_vals.std()
        std_test = test_vals.std()
        std_ratio = std_test / (std_train + 1e-9)
        
        # KL-divergence approximation via histogram overlap
        try:
            bins = np.linspace(
                min(train_vals.min(), test_vals.min()),
                max(train_vals.max(), test_vals.max()),
                20
            )
            hist_train, _ = np.histogram(train_vals, bins=bins, density=True)
            hist_test, _ = np.histogram(test_vals, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            hist_train = hist_train + 1e-9
            hist_test = hist_test + 1e-9
            
            # Symmetric KL divergence
            kl_div = 0.5 * (
                np.sum(hist_train * np.log(hist_train / hist_test)) +
                np.sum(hist_test * np.log(hist_test / hist_train))
            )
        except Exception:
            kl_div = np.nan
        
        shift_metrics[col] = {
            "mean_shift": float(mean_shift),
            "std_ratio": float(std_ratio),
            "kl_divergence": float(kl_div) if not np.isnan(kl_div) else None,
        }
    
    # Summary statistics
    mean_shifts = [m["mean_shift"] for m in shift_metrics.values()]
    kl_divs = [m["kl_divergence"] for m in shift_metrics.values() if m["kl_divergence"] is not None]
    
    return {
        "per_feature": shift_metrics,
        "summary": {
            "n_features": len(shift_metrics),
            "avg_mean_shift": float(np.mean(mean_shifts)) if mean_shifts else 0.0,
            "max_mean_shift": float(np.max(mean_shifts)) if mean_shifts else 0.0,
            "avg_kl_divergence": float(np.mean(kl_divs)) if kl_divs else 0.0,
        }
    }


def add_temporal_dynamics_features(
    df: pd.DataFrame,
    prediction_col: str = "meta_probability",
    signal_col: str = "consensus",
) -> pd.DataFrame:
    """
    Add features capturing temporal dynamics that the model may be missing.
    
    These features address the high residual autocorrelation (0.27) found
    in the diagnostics, indicating sequential structure not captured.
    
    Args:
        df: DataFrame with predictions and signals
        prediction_col: Column with model predictions
        signal_col: Column with signal values
        
    Returns:
        DataFrame with additional temporal features.
    """
    result = df.copy()
    
    # 1. Lagged predictions (if prediction exists)
    if prediction_col in df.columns:
        result[f"{prediction_col}_lag1"] = df[prediction_col].shift(1)
        result[f"{prediction_col}_lag2"] = df[prediction_col].shift(2)
        result[f"{prediction_col}_diff"] = df[prediction_col].diff()
        
        # Rolling mean of predictions (trend in confidence)
        result[f"{prediction_col}_ma5"] = df[prediction_col].rolling(5).mean()
    
    # 2. Signal density features
    if signal_col in df.columns:
        signal_binary = (df[signal_col] != 0).astype(int)
        result["signals_in_last_10_bars"] = signal_binary.rolling(10).sum()
        result["signals_in_last_20_bars"] = signal_binary.rolling(20).sum()
        result["bars_since_last_signal"] = _compute_bars_since_signal(signal_binary)
    
    # 3. Regime persistence (consecutive same-direction bars)
    if "close" in df.columns:
        returns = df["close"].pct_change()
        result["regime_persistence"] = _compute_streak_length(returns > 0)
    
    # 4. Time decay feature (bars since start of data)
    result["time_index_normalized"] = np.linspace(0, 1, len(df))
    
    return result


def _compute_bars_since_signal(signal_binary: pd.Series) -> pd.Series:
    """Compute number of bars since last non-zero signal."""
    result = pd.Series(index=signal_binary.index, dtype=float)
    last_signal_idx = -1
    
    for i, (idx, val) in enumerate(signal_binary.items()):
        if val > 0:
            last_signal_idx = i
            result.iloc[i] = 0
        elif last_signal_idx >= 0:
            result.iloc[i] = i - last_signal_idx
        else:
            result.iloc[i] = np.nan
    
    return result


def _compute_streak_length(condition: pd.Series) -> pd.Series:
    """Compute length of current streak (consecutive True values)."""
    result = pd.Series(0, index=condition.index, dtype=int)
    streak = 0
    prev_val = None
    
    for i, (idx, val) in enumerate(condition.items()):
        if pd.isna(val):
            streak = 0
        elif prev_val is None or val == prev_val:
            streak += 1
        else:
            streak = 1
        
        result.iloc[i] = streak
        prev_val = val
    
    return result


def compute_sample_reweighting_for_shift(
    train_propensity: pd.Series,
    target_propensity: float = 0.5,
) -> pd.Series:
    """
    Compute sample weights to correct for distribution shift.
    
    Uses inverse propensity weighting (IPW) to reweight training samples
    to better match the target distribution.
    
    Args:
        train_propensity: Propensity scores for training samples
        target_propensity: Target propensity (default 0.5 = uniform)
        
    Returns:
        Series of sample weights.
    """
    # Clip propensity to avoid extreme weights
    propensity_clipped = train_propensity.clip(0.1, 0.9)
    
    # IPW: weight = target / estimated
    weights = target_propensity / propensity_clipped
    
    # Normalize to mean=1
    weights = weights / weights.mean()
    
    # Clip extreme weights
    weights = weights.clip(0.1, 10.0)
    
    return weights
