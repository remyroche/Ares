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
    Calculates trend consistency (Efficiency Ratio) over the specified horizon.

    Efficiency Ratio = |Price_t - Price_{t-n}| / Sum(|Price_i - Price_{i-1}| over n)

    Returns:
        pd.Series: Consistency scores between 0.0 (choppy) and 1.0 (smooth trend).
    """
    # 1. Calculate net price change over the horizon
    net_change = close_series.diff(horizon).abs()

    # 2. Calculate period-to-period absolute changes (volatility/path length)
    period_change = close_series.diff(1).abs()

    # 3. Sum the path length over the horizon
    path_length = period_change.rolling(window=horizon).sum()

    # 4. Calculate Efficiency Ratio
    # Avoid division by zero
    efficiency_ratio = net_change / (path_length + 1e-12)

    # Fill NaN values (e.g., first 'horizon' rows) with 0.0
    return efficiency_ratio.fillna(0.0)

def compute_uniqueness(t_events, bar_index):
    """
    Calculates the average uniqueness of samples based on event overlaps.

    Args:
        t_events (pd.Series): Series where Index is event start time, Value is event end time.
        bar_index (pd.Index): The full timeline (index of the close prices).

    Returns:
        pd.Series: Uniqueness score for each event (aligned to t_events.index).
    """
    if t_events.empty:
        return pd.Series(index=t_events.index, dtype=float)

    # 1. Create a binary structure for event existence
    # We need to map events to the full timeline to count concurrency

    # Filter t_events to ensure they are within bar_index range
    # valid_events = t_events[t_events.isin(bar_index) & (t_events.index.isin(bar_index))]
    # Note: t1 might be slightly off bar_index if using timestamps, but we assume alignment or use searchsorted.
    # For simplicity and speed in this context, we'll map to the closest indices or assume alignment.

    # Build a Series of 1s on the bar_index
    concurrency = pd.Series(0, index=bar_index, dtype=float)

    # Iterate is slow, but robust for time ranges.
    # Optimization: Use start and end points
    # +1 at start, -1 after end. Cumsum gives concurrency.

    # Create a simplified timeline for the events
    # We only care about the span [min(start), max(end)]
    if hasattr(bar_index, 'tz'):
        # Ensure timezone awareness matches
        pass

    # Create change points
    # Start: +1
    start_counts = pd.Series(1, index=t_events.index)
    # End: -1 (We assume the event ends AT t1, so we subtract AFTER t1?
    # Usually standard is: event is active [t0, t1]. So at t1 it is still active.
    # So we subtract at the next bar after t1?
    # Or simplified: active from t0 to t1 inclusive.)

    # To avoid lookahead complexity, let's just iterate if dataset isn't massive.
    # Or use the standard MLDP snippet logic if available.

    # Vectorized approach using accumulation:
    timeline = pd.DataFrame(index=bar_index)
    timeline['count'] = 0

    # This part can be slow if we loop 100k events.
    # Faster:
    # 1. Melt t_events to (time, change)
    #    t0 -> +1
    #    t1 -> -1 (at next timestamp?)

    # Let's map t1 to the next available timestamp in bar_index to define the "end" for subtraction
    # But t1 IS the end time.

    # Let's stick to the definition: Uniqueness of event i = avg(1 / concurrency_t) for t in [t0, t1]

    # We can reconstruct concurrency:
    # c_t = sum(1 for event i if t0_i <= t <= t1_i)

    # If using standard pandas:
    # Create a long series of all bars involved in all events
    # This is memory intensive.

    # Approximate approach for speed (if events are many):
    # Just sum overlaps.

    # Let's try the looping approach, usually acceptable for <100k events in backtest.
    # If too slow, we'd need an interval tree.

    # However, let's use the standard "mp_num_concurrent_events" logic adapted.

    # Construct an indicator matrix? No, too big.

    # Let's use the 'start' and 'end' + cumsum approach.
    # Combine start times and end times into a single sorted index.

    # 1. Align t_events to bar_index to ensure we use valid timestamps
    # (Assuming t_events are subsets of bar_index)

    # Concurrency calculation:
    # We add 1 at t0, subtract 1 at t1 (or t1+1?).
    # If the interval is [t0, t1], then t1 is included. So we subtract at the step *after* t1.

    # Since we have the full bar_index, we can map t1 to the index location.

    # Get integer locations
    # searchsorted is efficient.

    # Ensure index is sorted
    if not bar_index.is_monotonic_increasing:
        bar_index = bar_index.sort_values()

    # Find integer indices of start times
    start_locs = bar_index.searchsorted(t_events.index)

    # Find integer indices of end times
    # We want the event to include t1. So we stop decreasing until after t1.
    # searchsorted(side='right') gives the index after t1.
    end_locs = bar_index.searchsorted(t_events.values, side='right')

    # Create an array of zeros for the full timeline
    counts = np.zeros(len(bar_index))

    # Add 1 at starts
    np.add.at(counts, start_locs, 1)

    # Subtract 1 at ends (if end_loc is within bounds)
    valid_ends = end_locs < len(bar_index)
    np.add.at(counts, end_locs[valid_ends], -1)

    # Cumulative sum to get concurrency at each bar
    concurrency_array = np.cumsum(counts)

    # Avoid division by zero (though concurrency should be >=1 for active events)
    concurrency_array[concurrency_array == 0] = 1.0 # Should be 0 where no events, but we only query at events

    # Now, for each event, calculate average uniqueness
    # Uniqueness = Average(1/Concurrency) over [start, end]

    # We can compute 1/Concurrency
    inv_concurrency = 1.0 / concurrency_array

    # We need sum(inv_concurrency) over [start_loc, end_loc) for each event
    # and divide by (end_loc - start_loc)

    # Use Integral Image (Cumulative Sum of the inverse) for O(1) query
    inv_conc_cumsum = np.cumsum(inv_concurrency)
    # Prepend 0 for easier indexing
    inv_conc_cumsum = np.insert(inv_conc_cumsum, 0, 0.0)

    # Sum(start, end) = CumSum[end] - CumSum[start]
    # In our case, end_loc is exclusive upper bound for the interval [start, end_loc)
    # which matches Python slicing.

    sums = inv_conc_cumsum[end_locs] - inv_conc_cumsum[start_locs]
    lengths = end_locs - start_locs

    # Handle zero lengths (should not happen if t1 > t0)
    lengths[lengths == 0] = 1.0

    means = sums / lengths

    return pd.Series(means, index=t_events.index).fillna(1.0)


def generate_weights_per_label(
    returns,
    t_events,
    close_series,
    consistency_scores,
    uniqueness_scores,
    y=None,                # NEW: Must pass binary labels [0, 0, 1, 0...]
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

    # NEW: Class Balance Intensity (Gamma)
    # 0.0 = Off, 1.0 = Standard Balanced, 1.5 = Aggressive
    class_balance_rate=1.0,

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

    # --- 1. Magnitude ---
    w_mag = np.power(np.abs(returns / vol_proxy), mag_compression)
    if downside_multiplier != 1.0:
        w_mag = np.where(returns < 0, w_mag * downside_multiplier, w_mag)

    # --- 2. Learnability ---
    w_learn = 1 / (1 + np.exp(-learn_slope * (consistency_scores - learn_center)))

    # --- 3. Uniqueness ---
    w_uniq = np.power(uniqueness_scores, uniq_intensity)

    # --- 4. NEW: Class Balancing ---
    w_class = np.ones_like(returns)

    if y is not None and class_balance_rate > 0:
        # Calculate standard inverse frequency weights
        # Note: We compute this solely based on the input 'y' vector
        classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(classes)

        for c in classes:
            # Logic: N / (k * n_j)
            n_j = np.sum(y == c)
            if n_j > 0:
                base_w = n_samples / (n_classes * n_j)
                # Apply Gamma (Intensity)
                w_c = np.power(base_w, class_balance_rate)
                # Assign to matching indices
                w_class[y == c] = w_c

    # --- 5. Synthesis ---
    # Combine all factors (Geometric Mean Logic)
    w_final = (w_mag**exp_mag) * (w_learn**exp_learn) * (w_uniq**exp_uniq) * w_class

    # --- 6. Bounds & Norm ---
    w_final = np.clip(w_final, floor_weight, cap_weight)
    w_final /= (np.mean(w_final) + 1e-12)

    return w_final


# -------------------------------------------------------------------------
# 2. The Core Scoring Engine
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
# 3. The Optuna Objective Wrapper
# -------------------------------------------------------------------------
def objective_layer_1(trial, returns, t_events, close_series, precalc_data):
    """
    Optuna Objective for Layer 1.
    """
    # Unpack pre-calculated heavy data
    consistency_scores = precalc_data['consistency']
    vol_proxy = precalc_data['volatility']
    uniqueness_scores = precalc_data['uniqueness'] # ADDED THIS
    
    # --- 1. Suggest Parameters ---
    params = {
        'mag_compression': trial.suggest_float("mag_compression", 0.0, 1.0),
        'learn_slope': trial.suggest_float("learn_slope", 5.0, 20.0),
        'learn_center': trial.suggest_float("learn_center", 0.3, 0.5),
        'uniq_intensity': trial.suggest_float("uniq_intensity", 0.5, 1.5),
        'exp_mag': trial.suggest_float("exp_mag", 0.5, 2.0),
        'exp_learn': trial.suggest_float("exp_learn", 0.5, 2.0),
        'exp_uniq': trial.suggest_float("exp_uniq", 0.5, 1.5),
        # exp_cons passed as exp_uniq in your generator logic, keeping consistent
    }
    
    thresholds = {
        'ess_min': trial.suggest_float("eval_ess_min", 0.05, 0.15),
        'cons_min': trial.suggest_float("eval_cons_min", 0.55, 0.70)
    }

    # --- 2. Generate Weights ---
    # Call the generator with ALL pre-calc data
    weights = generate_weights_per_label(
        returns, t_events, close_series,
        consistency_scores=consistency_scores,
        uniqueness_scores=uniqueness_scores,   # PASSED HERE
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
# 4. Driver Function
# -------------------------------------------------------------------------
def run_layer1_optimization(df, returns, t_events):
    """
    Pre-calculates data and runs the study.
    """
    print("Pre-calculating Layer 1 metrics...")
    
    # 1. Horizon Consistency
    # (Assuming compute_horizon_consistency is available in scope)
    consistency = compute_horizon_consistency(df['close'], horizon=12)
    
    # 2. Volatility Proxy
    volatility = returns.rolling(20, min_periods=1).std().fillna(0)
    
    # 3. Uniqueness (CRITICAL ADDITION)
    # (Assuming compute_uniqueness is available in scope)
    uniqueness = compute_uniqueness(t_events, df.index)
    
    precalc_data = {
        'consistency': consistency.values,
        'volatility': volatility.values,
        'uniqueness': uniqueness.values # Store it here
    }
    
    # 4. Run Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_layer_1(
            trial, returns.values, t_events, df['close'], precalc_data
        ), 
        n_trials=100
    )
    
    print("Best Layer 1 Params:", study.best_params)
    return study.best_params
