"""
Recency Weighting Utility for Time-Series Sample Weighting.

Provides exponential decay weights that prioritize recent samples in training,
helping models adapt to changing market conditions.

Usage:
    from src.utils.ml_common.recency_weighting import compute_recency_weights
    
    weights = compute_recency_weights(
        timestamps=df.index,
        decay_lambda=0.01,  # 1% decay per day
    )
"""

# =============================================================================
# SHARED CONFIG CONSTANTS - Single source of truth for all training modules
# =============================================================================

# Default recency decay rate (per day)
# 0.01 = 1% decay per day, meaning:
#   - 30 days ago: weight = 0.74
#   - 90 days ago: weight = 0.41
#   - 180 days ago: weight = 0.16
#   - 365 days ago: weight = 0.03
DEFAULT_RECENCY_DECAY_LAMBDA = 0.01

# Minimum weight floor to prevent near-zero weights
DEFAULT_RECENCY_MIN_WEIGHT = 0.1

# Config key name used across all training modules
RECENCY_CONFIG_KEY = "recency_decay_lambda"

# =============================================================================

import numpy as np
import pandas as pd
from typing import Optional, Union


def get_recency_decay_lambda(config: dict) -> float:
    """Get recency decay lambda from config, with centralized default.
    
    This function should be used by all training modules to ensure
    consistent recency weighting across HPO, backtest, and Analyst training.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Decay lambda value (0.0 = disabled, 0.01 = 1%/day default)
    """
    return config.get(RECENCY_CONFIG_KEY, DEFAULT_RECENCY_DECAY_LAMBDA)


def compute_recency_weights(
    timestamps: Union[pd.DatetimeIndex, pd.Series, np.ndarray],
    decay_lambda: float = 0.0,
    min_weight: float = 0.1,
    reference_time: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """Compute exponential decay weights based on recency.
    
    Weights are computed as: w(t) = clip(exp(-lambda * days_ago), min_weight, 1.0)
    
    Args:
        timestamps: Event timestamps (DatetimeIndex, Series, or array of timestamps)
        decay_lambda: Decay rate per day. Higher = more recency bias.
            - 0.0: No decay (all weights = 1.0)
            - 0.001: ~36% weight at 1000 days ago
            - 0.01: ~37% weight at 100 days ago
            - 0.02: ~37% weight at 50 days ago
        min_weight: Floor for weights to avoid near-zero values (default 0.1)
        reference_time: Reference timestamp for "now" (default: max timestamp)
    
    Returns:
        Array of weights in [min_weight, 1.0], same length as timestamps
    
    Example:
        >>> ts = pd.date_range('2024-01-01', periods=100, freq='D')
        >>> weights = compute_recency_weights(ts, decay_lambda=0.01)
        >>> weights[-1]  # Most recent = 1.0
        1.0
        >>> weights[0]   # 99 days ago = ~0.37
        0.372...
    """
    if decay_lambda <= 0:
        # No decay: return uniform weights
        return np.ones(len(timestamps), dtype=np.float64)
    
    # Convert to pandas DatetimeIndex if needed
    if isinstance(timestamps, np.ndarray):
        timestamps = pd.to_datetime(timestamps)
    elif isinstance(timestamps, pd.Series):
        timestamps = pd.to_datetime(timestamps.values)
    
    # Reference time (default: most recent timestamp)
    if reference_time is None:
        reference_time = timestamps.max()
    
    # Compute days ago
    time_deltas = reference_time - timestamps
    days_ago = time_deltas.total_seconds() / 86400.0
    
    # Handle negative days (future timestamps) by clipping to 0
    days_ago = np.maximum(days_ago, 0)
    
    # Compute exponential decay weights
    weights = np.exp(-decay_lambda * days_ago)
    
    # Clip to [min_weight, 1.0]
    weights = np.clip(weights, min_weight, 1.0)
    
    return weights


def combine_weights(
    base_weights: Optional[np.ndarray],
    recency_weights: np.ndarray,
    combination: str = "multiply",
) -> np.ndarray:
    """Combine base sample weights with recency weights.
    
    Args:
        base_weights: Existing sample weights (e.g., from class balancing).
            If None, uses uniform weights.
        recency_weights: Recency-based weights from compute_recency_weights()
        combination: How to combine:
            - "multiply": base * recency (default, preserves relative importance)
            - "add": base + recency (additive)
            - "replace": use recency_weights only
    
    Returns:
        Combined weights array
    """
    if base_weights is None:
        base_weights = np.ones_like(recency_weights)
    
    if combination == "multiply":
        combined = base_weights * recency_weights
    elif combination == "add":
        combined = base_weights + recency_weights
    elif combination == "replace":
        combined = recency_weights.copy()
    else:
        raise ValueError(f"Unknown combination method: {combination}")
    
    # Normalize to prevent numerical issues
    if combined.sum() > 0:
        combined = combined / combined.mean()
    
    return combined


def log_recency_stats(
    weights: np.ndarray,
    timestamps: Union[pd.DatetimeIndex, pd.Series],
    logger=None,
) -> dict:
    """Log recency weight statistics for debugging.
    
    Args:
        weights: Recency weights array
        timestamps: Corresponding timestamps
        logger: Logger instance (uses print if None)
    
    Returns:
        Dict with statistics
    """
    stats = {
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
        "std_weight": float(weights.std()),
        "n_samples": len(weights),
    }
    
    msg = (
        f"Recency weights: min={stats['min_weight']:.4f}, "
        f"max={stats['max_weight']:.4f}, "
        f"mean={stats['mean_weight']:.4f}, "
        f"n={stats['n_samples']}"
    )
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return stats
