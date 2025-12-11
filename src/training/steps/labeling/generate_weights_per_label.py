# src/training/steps/labeling/generate_weights_per_label.py

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import optuna

# Assumes you have the generator saved in this path
# from src.data.weighting.generate_weights_per_label import generate_weights_per_label

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
