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

    Returns a Series aligned with close_series index.
    """
    # Net change over horizon
    change = close_series.diff(horizon).abs()

    # Sum of absolute changes (path length)
    path = close_series.diff().abs().rolling(window=horizon).sum()

    # Efficiency Ratio
    er = change / (path + 1e-12)
    return er.fillna(0.0)

def compute_uniqueness(t_events, price_index, lookahead=12):
    """
    Calculates label uniqueness based on concurrency.
    Assumes a fixed horizon 'lookahead' for each event if t1 is not available.

    t_events: Index of event start times.
    price_index: The full DatetimeIndex of the price series.
    lookahead: The duration of the label in bars.

    Returns: Array of uniqueness scores aligned with t_events.
    """
    # map t_events to integer indices in price_index
    # We need to find the integer locations of t_events in price_index
    # This can be slow if not optimized, but robust:

    # Create a Series to represent the timeline
    timeline = pd.Series(0, index=price_index)

    # We need to know where each event starts and ends (conceptually)
    # Since we work with timestamps, let's use searchsorted if monotonic
    # or just reindexing.

    # Optimization: Use integer indexing on the full array
    n_bars = len(price_index)
    concurrency = np.zeros(n_bars)

    # Get integer locations of start times
    # Assuming t_events is a subset of price_index
    # We can use searchsorted if sorted
    if not price_index.is_monotonic_increasing:
         # Fallback for unsorted index (unlikely for time series)
         # This is slow, but safe
         idxs = [price_index.get_loc(t) for t in t_events]
    else:
         idxs = price_index.searchsorted(t_events)

    # Increment concurrency counter
    for start_idx in idxs:
        end_idx = min(start_idx + lookahead, n_bars)
        concurrency[start_idx:end_idx] += 1

    # Average concurrency per event
    uniqueness = []
    for start_idx in idxs:
        end_idx = min(start_idx + lookahead, n_bars)
        # uniqueness = 1 / average_concurrency_over_lifespan
        avg_conc = concurrency[start_idx:end_idx].mean() if start_idx < end_idx else 1.0
        uniqueness.append(1.0 / (avg_conc + 1e-12))

    return np.array(uniqueness)

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
