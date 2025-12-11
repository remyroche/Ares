# src/training/steps/labeling/generate_weights_per_label.py
"""
Sample Weight Generation for Meta-Labeling HPO.

This module provides functions to compute sample weights based on:
- Magnitude: Risk-adjusted return magnitude
- Learnability: Trend consistency score via sigmoid gate
- Uniqueness: Event overlap/concurrency penalty
- Cross-term: Interaction between magnitude and learnability

These weights are used in the meta_labeling_hpo_sample_weighted step
to improve model training by emphasizing high-quality samples.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import optuna
from typing import Dict, Any, Optional, Union


# -------------------------------------------------------------------------
# 1. Mathematical Helpers
# -------------------------------------------------------------------------
def sigmoid_gate(x, threshold, sharpness=10.0, reverse=False):
    """Smooth sigmoid transition.
    
    Args:
        x: Input value
        threshold: Center point of sigmoid
        sharpness: Steepness of transition (higher = sharper)
        reverse: If True, returns 1-sigmoid instead of sigmoid
        
    Returns:
        Sigmoid-gated value in [0, 1]
    """
    exponent = np.clip(-sharpness * (x - threshold), -50, 50)
    val = 1 / (1 + np.exp(exponent))
    return (1.0 - val) if reverse else val


def magnitude_aware_spearman(weights, returns, magnitude_func=np.log1p):
    """
    Vectorized magnitude-aware Spearman correlation.
    Checks if weights rank-order the *magnitude* of returns correctly.
    
    Args:
        weights: Sample weights array
        returns: Returns array
        magnitude_func: Function to transform absolute returns (default: log1p)
        
    Returns:
        Correlation coefficient in [-1, 1]
    """
    # 1. Transform returns to magnitude
    abs_rets = np.abs(returns)
    mag_rets = magnitude_func(abs_rets)
    
    # 2. Compute Ranks
    # argsort(argsort(x)) gives the rank (0 to N-1)
    rank_weights = np.argsort(np.argsort(weights)).astype(np.float64)
    rank_mag = np.argsort(np.argsort(mag_rets)).astype(np.float64)
    
    # 3. Center Ranks
    rank_weights -= rank_weights.mean()
    rank_mag -= rank_mag.mean()
    
    # 4. Weighted Covariance (using Magnitude as the importance weight)
    # We want to know: Do the ranks match *specifically* on the big moves?
    weighted_cov = np.sum(rank_weights * rank_mag * mag_rets) / (np.sum(mag_rets) + 1e-12)
    
    # 5. Normalize
    # Approximate standard deviation normalization
    std_prod = np.std(rank_weights) * np.std(rank_mag)
    corr = weighted_cov / (std_prod + 1e-12)
    
    # Clip to valid correlation range [-1, 1]
    return np.clip(corr, -1.0, 1.0)


def compute_horizon_consistency(close_series: pd.Series, horizon: int = 12) -> pd.Series:
    """
    Calculates the Efficiency Ratio (Kaufman) as a proxy for trend consistency.
    ER = |Change| / Sum(|Changes|)
    
    Args:
        close_series: Close price series
        horizon: Lookback horizon in bars
        
    Returns:
        Series of efficiency ratios in [0, 1]
    """
    # 1. Net change over horizon
    net_change = close_series.diff(horizon).abs()
    
    # 2. Path length (Sum of absolute period-to-period changes)
    path_length = close_series.diff().abs().rolling(window=horizon).sum()
    
    # 3. Efficiency Ratio (Noise-dampened)
    # If path_length is 0 (flat line), ER is 0 (or 1? Usually 0 for trading).
    efficiency_ratio = net_change / (path_length + 1e-12)
    
    return efficiency_ratio.fillna(0.0)


def compute_uniqueness(
    t_events: pd.Series,
    price_index: pd.Index,
    lookahead: Optional[int] = None,
) -> pd.Series:
    """
    Calculates average uniqueness using the Fast Algorithm (Integral Image).
    Complexity: O(N) instead of O(N * Horizon).
    
    Args:
        t_events: Series with Index=Start Time, Values=End Time.
        price_index: Full timeline of bars (DatetimeIndex).
        lookahead: Optional lookahead parameter (currently unused, kept for API compat).
        
    Returns:
        Series of uniqueness scores in [0, 1] indexed by t_events.index
    """
    if t_events.empty:
        return pd.Series(index=t_events.index, dtype=float)
        
    # 1. Map timestamps to integer locations in the price_index
    # This allows us to work with fast numpy arrays
    if not price_index.is_monotonic_increasing:
        price_index = price_index.sort_values()
        
    n_bars = len(price_index)
    
    # Find start and end indices
    # searchsorted matches the timestamp to the array index
    start_idxs = price_index.searchsorted(t_events.index)
    
    # For end times, we want the event to cover the range [t_start, t_end].
    # 'side=right' gives the index *after* t_end, which is perfect for python slicing.
    end_idxs = price_index.searchsorted(t_events.values, side='right')
    
    # Clip to bounds
    end_idxs = np.minimum(end_idxs, n_bars)
    
    # 2. Compute Concurrency (How many trades active at each bar?)
    concurrency = np.zeros(n_bars + 1)  # +1 for safe indexing
    
    # Add +1 at start, Subtract -1 at end
    np.add.at(concurrency, start_idxs, 1)
    np.add.at(concurrency, end_idxs, -1)
    
    # Cumsum fills the gaps
    concurrency = np.cumsum(concurrency)[:n_bars]
    
    # Avoid division by zero (0 concurrency -> 1 for math safety)
    concurrency[concurrency == 0] = 1.0
    
    # 3. Compute Uniqueness (Average of 1/Concurrency)
    inv_concurrency = 1.0 / concurrency
    
    # Integral Image (Cumulative Sum of the Inverse)
    # cumsum[i] = sum(0..i)
    inv_cumsum = np.cumsum(inv_concurrency)
    inv_cumsum = np.insert(inv_cumsum, 0, 0.0)  # Prepend 0 for subtraction logic
    
    # Sum over [start, end) = cumsum[end] - cumsum[start]
    sums = inv_cumsum[end_idxs] - inv_cumsum[start_idxs]
    lengths = end_idxs - start_idxs
    
    # Handle instantaneous events (length 0)
    lengths[lengths == 0] = 1.0
    
    means = sums / lengths
    
    return pd.Series(means, index=t_events.index).fillna(1.0)


# -------------------------------------------------------------------------
# 2. The Weight Generator (CANONICAL VERSION)
# -------------------------------------------------------------------------
def generate_weights_per_label(
    returns: Union[np.ndarray, pd.Series],
    t_events: pd.Index,
    close_series: Optional[pd.Series],  # Kept for signature compatibility
    consistency_scores: np.ndarray,
    uniqueness_scores: np.ndarray,
    vol_proxy: np.ndarray,
    mag_compression: float,
    learn_slope: float,
    learn_center: float,
    uniq_intensity: float,
    exp_mag: float,
    exp_learn: float,
    exp_uniq: float,
    exp_cross: float,
    downside_multiplier: float,
    # Optional additional parameters for extended functionality
    y: Optional[np.ndarray] = None,
    class_balance_rate: float = 0.0,
    floor_weight: float = 1e-4,
    cap_weight: float = 50.0,
) -> np.ndarray:
    """
    Generates sample weights based on Magnitude, Learnability, Uniqueness, and Cross-term.
    
    This is the CANONICAL weight generation function used throughout the weighted
    meta-labeling pipeline.
    
    Args:
        returns: Array of realized returns per event
        t_events: Index of event timestamps
        close_series: Close price series (kept for API compat, unused)
        consistency_scores: Pre-computed horizon consistency scores
        uniqueness_scores: Pre-computed uniqueness scores
        vol_proxy: Volatility proxy for risk adjustment
        mag_compression: Compression exponent for magnitude (0-1)
        learn_slope: Sigmoid slope for learnability gate
        learn_center: Sigmoid center for learnability gate
        uniq_intensity: Exponent for uniqueness weighting
        exp_mag: Final exponent for magnitude component
        exp_learn: Final exponent for learnability component
        exp_uniq: Final exponent for uniqueness component
        exp_cross: Final exponent for cross-term (mag * learn)
        downside_multiplier: Multiplier for negative returns (emphasize losses)
        y: Optional class labels for class balancing
        class_balance_rate: Rate for inverse class frequency weighting (0 = no balancing)
        floor_weight: Minimum allowed weight
        cap_weight: Maximum allowed weight
        
    Returns:
        Array of sample weights, normalized to mean=1
    """
    # Convert to numpy if needed
    if hasattr(returns, 'values'):
        returns = returns.values
    returns = np.asarray(returns, dtype=float)
    
    # 1. Safety Fix: Infinite Volatility Trap
    # Force vol_proxy to be valid
    safe_vol = np.maximum(vol_proxy, 1e-6)

    # 2. Magnitude Weight (Risk Adjusted)
    risk_adjusted_ret = returns / safe_vol
    w_mag = np.abs(risk_adjusted_ret) ** mag_compression

    # Feature 1: Downside Multiplier
    # Apply multiplier where returns are negative
    if downside_multiplier != 1.0:
        w_mag = np.where(returns < 0, w_mag * downside_multiplier, w_mag)

    # 3. Learnability Weight (Consistency)
    # Inline Sigmoid Logic (to avoid external dependency in this function)
    # w_learn = 1 / (1 + exp(-slope * (x - center)))
    exponent = np.clip(-learn_slope * (consistency_scores - learn_center), -50, 50)
    w_learn = 1.0 / (1.0 + np.exp(exponent))

    # 4. Uniqueness Weight
    w_uniq = uniqueness_scores ** uniq_intensity

    # 5. Feature 2: Cross-term Interaction
    w_cross = w_mag * w_learn

    # 6. Class Balancing (optional)
    w_class = np.ones_like(returns, dtype=float)
    if y is not None and class_balance_rate > 0:
        y = np.asarray(y)
        classes = np.unique(y[~np.isnan(y)])
        n_samples = len(y)
        n_classes = len(classes)
        for c in classes:
            n_j = np.sum(y == c)
            if n_j > 0:
                base_w = n_samples / (n_classes * n_j)
                w_class[y == c] = np.power(base_w, class_balance_rate)

    # 7. Final Combination
    # Combine component weights with exponents
    raw_weights = (
        (w_mag ** exp_mag) *
        (w_learn ** exp_learn) *
        (w_uniq ** exp_uniq) *
        (w_cross ** exp_cross) *
        w_class
    )

    # 8. Bounds & Normalization
    raw_weights = np.clip(raw_weights, floor_weight, cap_weight)
    raw_weights = raw_weights / (np.mean(raw_weights) + 1e-12)

    return raw_weights


# -------------------------------------------------------------------------
# 3. The Core Scoring Engine
# -------------------------------------------------------------------------
def evaluate_weighting_scheme(
    weights: np.ndarray,
    returns: np.ndarray,
    consistency_scores: np.ndarray,
    vol_proxy: np.ndarray,
    thresholds: Dict[str, float],
) -> float:
    """
    Evaluates the 'Teacher Quality' of a weighting vector.
    
    Uses multiple metrics:
    - Information Coefficient (IC): Does weight predict magnitude?
    - ESS Stability: Are weights well-distributed?
    - Weighted Consistency: Are high-weight samples learnable?
    - Volatility Bias: Are we just weighting by vol?
    
    Args:
        weights: Sample weights array
        returns: Returns array
        consistency_scores: Trend consistency scores
        vol_proxy: Volatility proxy
        thresholds: Dict with 'ess_min' and 'cons_min' thresholds
        
    Returns:
        Composite quality score
    """
    # --- A. Data Hygiene ---
    weights = np.nan_to_num(weights, nan=1e-6)
    weights = weights / (np.mean(weights) + 1e-12)

    # --- B. Information Coefficient (IC) ---
    # Does weight predict Magnitude?
    ic = magnitude_aware_spearman(weights, returns)
    
    # Soft Floor: Keep gradient alive
    if ic <= 0:
        return 1e-6

    # --- C. Effective Sample Size (ESS) Stability ---
    n = len(weights)
    ess = (np.sum(weights) ** 2) / (np.sum(weights ** 2) + 1e-12)
    ess_ratio = ess / n
    
    score_ess = sigmoid_gate(
        ess_ratio, 
        threshold=thresholds['ess_min'], 
        sharpness=15.0
    )

    # --- D. Weighted Horizon Consistency (Quality) ---
    norm_w = weights / (np.sum(weights) + 1e-12)
    weighted_consistency = np.sum(norm_w * consistency_scores)
    
    score_consistency = sigmoid_gate(
        weighted_consistency, 
        threshold=thresholds['cons_min'], 
        sharpness=10.0
    )

    # --- E. Volatility Bias Correction (Safety) ---
    vol_corr, _ = spearmanr(weights, vol_proxy)
    score_vol_bias = sigmoid_gate(
        vol_corr, 
        threshold=0.85, 
        sharpness=15.0, 
        reverse=True
    )

    # --- F. Synthesis ---
    final_score = ic * score_ess * score_consistency * score_vol_bias
    
    return final_score


# -------------------------------------------------------------------------
# 4. The Optuna Objective Wrapper
# -------------------------------------------------------------------------
def objective_layer_1(
    trial,
    returns: np.ndarray,
    t_events: pd.Index,
    close_series: pd.Series,
    precalc_data: Dict[str, np.ndarray],
) -> float:
    """
    Optuna Objective for Layer 1 weight parameter optimization.
    
    Args:
        trial: Optuna trial object
        returns: Returns array aligned to t_events
        t_events: Event timestamps
        close_series: Close prices (for compatibility)
        precalc_data: Dict with 'consistency', 'volatility', 'uniqueness' arrays
        
    Returns:
        Weighting scheme quality score
    """
    # Unpack pre-calculated heavy data
    # NOTE: These should already be aligned to t_events by run_layer1_optimization
    consistency_scores = precalc_data['consistency']
    vol_proxy = precalc_data['volatility']
    uniqueness_scores = precalc_data['uniqueness']
    
    # --- 1. Suggest Parameters ---
    params = {
        'mag_compression': trial.suggest_float("mag_compression", 0.0, 1.0),
        'learn_slope': trial.suggest_float("learn_slope", 5.0, 20.0),
        'learn_center': trial.suggest_float("learn_center", 0.3, 0.5),
        'uniq_intensity': trial.suggest_float("uniq_intensity", 0.5, 1.5),
        'exp_mag': trial.suggest_float("exp_mag", 0.5, 2.0),
        'exp_learn': trial.suggest_float("exp_learn", 0.5, 2.0),
        'exp_uniq': trial.suggest_float("exp_uniq", 0.5, 1.5),
        # Feature 2: Cross-term interactions
        'exp_cross': trial.suggest_float("exp_cross", 0.5, 2.0),
        # Feature 1: Downside Multiplier
        'downside_multiplier': trial.suggest_float("downside_multiplier", 1.0, 1.5),
    }

    thresholds = {
        'ess_min': trial.suggest_float("eval_ess_min", 0.05, 0.15),
        'cons_min': trial.suggest_float("eval_cons_min", 0.55, 0.70)
    }

    # --- 2. Generate Weights ---
    # Call the generator with ALL pre-calc data
    weights = generate_weights_per_label(
        returns=returns,
        t_events=t_events,
        close_series=close_series,
        consistency_scores=consistency_scores,
        uniqueness_scores=uniqueness_scores,
        vol_proxy=vol_proxy,
        **params
    )
    
    # --- 3. Evaluate ---
    score = evaluate_weighting_scheme(
        weights, 
        returns, 
        consistency_scores, 
        vol_proxy, 
        thresholds
    )
    
    return score


# -------------------------------------------------------------------------
# 5. Driver Function
# -------------------------------------------------------------------------
def run_layer1_optimization(
    df: pd.DataFrame,
    returns: Union[np.ndarray, pd.Series],
    t_events: pd.Index,
    n_trials: int = 50,
    horizon: int = 12,
) -> Dict[str, Any]:
    """
    Pre-calculates data and runs the Layer 1 weighting parameter optimization.

    Args:
        df: Full dataframe with 'close' column.
        returns: Series of returns corresponding to t_events.
        t_events: Index/Series of event start times.
        n_trials: Number of Optuna trials (default: 50)
        horizon: Lookback horizon for consistency calculation (default: 12)
        
    Returns:
        Dict of best weighting parameters
    """
    print("Pre-calculating Layer 1 metrics...")
    
    # 1. Horizon Consistency (Full History)
    consistency_full = compute_horizon_consistency(df['close'], horizon=horizon)

    # 2. Volatility Proxy (Full History)
    # Fix: Infinite Volatility (will be handled in generator, but cleaner here too)
    volatility_full = df['close'].pct_change().rolling(20, min_periods=1).std().fillna(0)
    
    # 3. Uniqueness (Events)
    # Create t_events as a Series with end times for uniqueness calculation
    # If t_events is just an index, create end times as index + horizon bars
    if isinstance(t_events, pd.Series):
        t_events_series = t_events
    else:
        # Assume events end after 'horizon' bars
        try:
            t_events_series = pd.Series(
                index=t_events,
                data=t_events + pd.Timedelta(minutes=15 * horizon)  # Assuming 15m bars
            )
        except Exception:
            # Fallback: use same timestamp for start/end
            t_events_series = pd.Series(index=t_events, data=t_events)
    
    uniqueness_aligned = compute_uniqueness(t_events_series, df.index)
    
    # Feature 3: Fix "Array Shape" Crash
    # Reindex full-history metrics to t_events
    try:
        if isinstance(t_events, pd.Series):
            event_index = t_events.index
        else:
            event_index = t_events
            
        consistency_aligned = consistency_full.reindex(event_index).fillna(0).values
        volatility_aligned = volatility_full.reindex(event_index).fillna(0).values
        
        # Ensure uniqueness is aligned
        if isinstance(uniqueness_aligned, pd.Series):
            uniqueness_aligned = uniqueness_aligned.reindex(event_index).fillna(1.0).values
        
        # Ensure returns is also numpy array
        returns_values = returns.values if hasattr(returns, 'values') else np.asarray(returns)

    except Exception as e:
        print(f"Error reindexing metrics: {e}")
        # Fallback if t_events are not in index
        raise e
    
    precalc_data = {
        'consistency': consistency_aligned,
        'volatility': volatility_aligned,
        'uniqueness': uniqueness_aligned
    }
    
    # 4. Run Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_layer_1(
            trial,
            returns_values,
            event_index,
            df['close'],
            precalc_data
        ), 
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    print("Best Layer 1 Params:", study.best_params)
    
    # Extract only the weight generation params (exclude threshold params)
    weight_params = {
        k: v for k, v in study.best_params.items()
        if k not in ('eval_ess_min', 'eval_cons_min')
    }
    
    return weight_params
