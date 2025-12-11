# src/training/hpo/layer1_objective.py
import numpy as np
import pandas as pd
import optuna
from scipy.stats import spearmanr
from src.data.weighting import (
    generate_weights_per_label,
    compute_horizon_consistency,
    compute_uniqueness,
    sigmoid_gate
)

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
    # Matches src/data/weighting.py signature
    params = {
        'mag_compression': trial.suggest_float("mag_compression", 0.0, 1.0),
        'learn_slope': trial.suggest_float("learn_slope", 5.0, 20.0),
        'learn_center': trial.suggest_float("learn_center", 0.3, 0.5),
        'uniq_intensity': trial.suggest_float("uniq_intensity", 0.5, 1.5),
        'class_balance_rate': trial.suggest_float("class_balance_rate", 0.0, 2.0),
        'exp_mag': trial.suggest_float("exp_mag", 0.5, 2.0),
        'exp_learn': trial.suggest_float("exp_learn", 0.5, 2.0),
        'exp_uniq': trial.suggest_float("exp_uniq", 0.5, 1.5),
        'downside_multiplier': trial.suggest_float("downside_multiplier", 1.0, 1.5),
    }
    
    thresholds = {
        'ess_min': trial.suggest_float("eval_ess_min", 0.05, 0.15),
        'cons_min': trial.suggest_float("eval_cons_min", 0.55, 0.70)
    }

    # --- 2. Generate Weights ---
    # Call the shared generator
    weights = generate_weights_per_label(
        returns=returns,
        t_events=t_events,
        close_series=close_series,
        consistency_scores=consistency_scores,
        uniqueness_scores=uniqueness_scores,
        vol_proxy=vol_proxy,
        y=None, # Class balancing handled implicitly or ignored in layer 1
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

def run_layer1_optimization(df, returns, t_events):
    """
    Pre-calculates data and runs the study.

    df: Full dataframe with 'close' column.
    returns: Series of returns corresponding to t_events (or full? Usually t_events).
             Assumption: returns passed here are the Label Returns (aligned with t_events).
    t_events: Index/Series of event start times.
    """
    print("Pre-calculating Layer 1 metrics...")
    
    # 1. Horizon Consistency (Full History)
    consistency_full = compute_horizon_consistency(df['close'], horizon=12)

    # 2. Volatility Proxy (Full History)
    volatility_full = df['close'].pct_change().rolling(20, min_periods=1).std().fillna(0)
    
    # 3. Uniqueness (Events)
    uniqueness_aligned = compute_uniqueness(t_events, df.index, lookahead=12)
    
    # Reindex full-history metrics to t_events
    try:
        consistency_aligned = consistency_full.reindex(t_events).fillna(0).values
        volatility_aligned = volatility_full.reindex(t_events).fillna(0).values
        # Ensure returns is also numpy array
        returns_values = returns.values if hasattr(returns, 'values') else returns

    except Exception as e:
        print(f"Error reindexing metrics: {e}")
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
        n_trials=50
    )
    
    print("Best Layer 1 Params:", study.best_params)
    return study.best_params
