# src/training/steps/labeling/generate_weights_per_label.py

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import optuna


# -------------------------------------------------------------------------
# 1. Mathematical Helpers
# -------------------------------------------------------------------------
def sigmoid_gate(x, threshold, sharpness=10.0, reverse=False):
    """Smooth sigmoid transition."""
    exponent = np.clip(-sharpness * (x - threshold), -50, 50)
    val = 1 / (1 + np.exp(exponent))
    return (1.0 - val) if reverse else val

def magnitude_aware_spearman(weights, returns, magnitude_func=np.log1p):
    """
    Vectorized magnitude-aware Spearman correlation.
    Checks if weights rank-order the *magnitude* of returns correctly.
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

def compute_horizon_consistency(close_series, horizon=12):
    """
    Calculates the Efficiency Ratio (Kaufman) as a proxy for trend consistency.
    ER = |Change| / Sum(|Changes|)
    """
    # 1. Net change over horizon
    net_change = close_series.diff(horizon).abs()
    
    # 2. Path length (Sum of absolute period-to-period changes)
    path_length = close_series.diff().abs().rolling(window=horizon).sum()
    
    # 3. Efficiency Ratio (Noise-dampened)
    # If path_length is 0 (flat line), ER is 0 (or 1? Usually 0 for trading).
    efficiency_ratio = net_change / (path_length + 1e-12)
    
    return efficiency_ratio.fillna(0.0)

def compute_uniqueness(t_events, price_index):
    """
    Calculates average uniqueness using the Fast Algorithm (Integral Image).
    Complexity: O(N) instead of O(N * Horizon).
    
    Args:
        t_events (pd.Series): Index=Start Time, Values=End Time.
        price_index (pd.Index): Full timeline of bars.
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
    concurrency = np.zeros(n_bars + 1) # +1 for safe indexing
    
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
    inv_cumsum = np.insert(inv_cumsum, 0, 0.0) # Prepend 0 for subtraction logic
    
    # Sum over [start, end) = cumsum[end] - cumsum[start]
    sums = inv_cumsum[end_idxs] - inv_cumsum[start_idxs]
    lengths = end_idxs - start_idxs
    
    # Handle instantaneous events (length 0)
    lengths[lengths == 0] = 1.0
    
    means = sums / lengths
    
    return pd.Series(means, index=t_events.index).fillna(1.0)

def generate_weights_per_label(
    returns,
    t_events,
    close_series,
    consistency_scores,
    uniqueness_scores,
    y=None,                 # NEW: Class labels
    vol_proxy=None,
    
    # Layer 1 HPO parameters
    mag_compression=0.8,
    learn_slope=10.0,
    learn_center=0.4,
    uniq_intensity=1.0,
    
    # Exponents
    exp_mag=1.0,
    exp_learn=1.0,
    exp_uniq=1.0,
    
    # Class Balance
    class_balance_rate=1.0, # Gamma
    
    downside_multiplier=1.0,
    floor_weight=1e-4,
    cap_weight=50.0
):
    """
    Vectorized weighting generator including Class Balancing.
    """
    # --- 0. Data Hygiene ---
    returns = np.nan_to_num(returns, nan=0.0)
    consistency_scores = np.nan_to_num(consistency_scores, nan=0.0)
    uniqueness_scores = np.nan_to_num(uniqueness_scores, nan=1.0)
    if vol_proxy is None: vol_proxy = np.ones_like(returns)
    vol_proxy = np.maximum(vol_proxy, 1e-8)

    # --- 1. Magnitude (Risk Adjusted) ---
    w_mag = np.power(np.abs(returns / vol_proxy), mag_compression)
    if downside_multiplier != 1.0:
        w_mag = np.where(returns < 0, w_mag * downside_multiplier, w_mag)

    # --- 2. Learnability (Consistency Gate) ---
    w_learn = 1 / (1 + np.exp(-learn_slope * (consistency_scores - learn_center)))

    # --- 3. Uniqueness ---
    w_uniq = np.power(uniqueness_scores, uniq_intensity)

    # --- 4. Class Balancing ---
    w_class = np.ones_like(returns)
    if y is not None and class_balance_rate > 0:
        classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(classes)
        for c in classes:
            n_j = np.sum(y == c)
            if n_j > 0:
                base_w = n_samples / (n_classes * n_j)
                w_class[y == c] = np.power(base_w, class_balance_rate)

    # --- 5. Synthesis ---
    w_final = (w_mag**exp_mag) * (w_learn**exp_learn) * (w_uniq**exp_uniq) * w_class

    # --- 6. Bounds & Norm ---
    w_final = np.clip(w_final, floor_weight, cap_weight)
    w_final /= (np.mean(w_final) + 1e-12)

    return w_final

# -------------------------------------------------------------------------
# 2. The Weight Generator
# -------------------------------------------------------------------------
def generate_weights_per_label(
    returns,
    t_events,
    close_series, # Kept for signature compatibility, though maybe unused if precalc passed
    consistency_scores,
    uniqueness_scores,
    vol_proxy,
    mag_compression,
    learn_slope,
    learn_center,
    uniq_intensity,
    exp_mag,
    exp_learn,
    exp_uniq,
    exp_cross,
    downside_multiplier
):
    """
    Generates sample weights based on Magnitude, Learnability, and Uniqueness.
    """
    # 1. Safety Fix: Infinite Volatility Trap
    # Force vol_proxy to be valid
    safe_vol = np.maximum(vol_proxy, 1e-6)

    # 2. Magnitude Weight (Risk Adjusted)
    risk_adjusted_ret = returns / safe_vol
    w_mag = np.abs(risk_adjusted_ret) ** mag_compression

    # Feature 1: Downside Multiplier
    # Apply multiplier where returns are negative
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

    # 6. Final Combination
    # Combine component weights with exponents
    raw_weights = (
        (w_mag ** exp_mag) *
        (w_learn ** exp_learn) *
        (w_uniq ** exp_uniq) *
        (w_cross ** exp_cross)
    )

    return raw_weights


# -------------------------------------------------------------------------
# 3. The Core Scoring Engine
# -------------------------------------------------------------------------
def evaluate_weighting_scheme(weights, returns, consistency_scores, vol_proxy, thresholds):
    """
    Evaluates the 'Teacher Quality' of a weighting vector.
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
def objective_layer_1(trial, returns, t_events, close_series, precalc_data):
    """
    Optuna Objective for Layer 1.
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
    
    # Removed 'exp_cons' (Phantom Parameter fix)

    thresholds = {
        'ess_min': trial.suggest_float("eval_ess_min", 0.05, 0.15),
        'cons_min': trial.suggest_float("eval_cons_min", 0.55, 0.70)
    }

    # --- 2. Generate Weights ---
    # Call the generator with ALL pre-calc data
    weights = generate_weights_per_label(
        returns,
        t_events,
        close_series,
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
def run_layer1_optimization(df, returns, t_events):
    """
    Pre-calculates data and runs the study.

    df: Full dataframe with 'close' column.
    returns: Series of returns corresponding to t_events (or full? Usually t_events).
             Wait, if returns has same index as t_events, we are good.
             If returns is full series, we need to reindex.
             Assumption: returns passed here are the Label Returns (aligned with t_events).
    t_events: Index/Series of event start times.
    """
    print("Pre-calculating Layer 1 metrics...")
    
    # 1. Horizon Consistency (Full History)
    consistency_full = compute_horizon_consistency(df['close'], horizon=12)

    # 2. Volatility Proxy (Full History)
    # Fix: Infinite Volatility (will be handled in generator, but cleaner here too)
    volatility_full = df['close'].pct_change().rolling(20, min_periods=1).std().fillna(0)
    
    # 3. Uniqueness (Events)
    # Calculated relative to full index but returns event-aligned array
    uniqueness_aligned = compute_uniqueness(t_events, df.index, lookahead=12)
    
    # Feature 3: Fix "Array Shape" Crash
    # Reindex full-history metrics to t_events
    try:
        consistency_aligned = consistency_full.reindex(t_events).fillna(0).values
        volatility_aligned = volatility_full.reindex(t_events).fillna(0).values
        # Ensure returns is also numpy array
        returns_values = returns.values if hasattr(returns, 'values') else returns

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
            t_events,
            df['close'],
            precalc_data
        ), 
        n_trials=50  # Reduced trial count for speed
    )
    
    print("Best Layer 1 Params:", study.best_params)
    return study.best_params
