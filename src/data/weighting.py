# src/data/weighting.py
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. Mathematical Helpers
# -------------------------------------------------------------------------
def sigmoid_gate(x, threshold, sharpness=10.0, reverse=False):
    """Smooth sigmoid transition."""
    exponent = np.clip(-sharpness * (x - threshold), -50, 50)
    val = 1 / (1 + np.exp(exponent))
    return (1.0 - val) if reverse else val

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
    n_bars = len(price_index)
    concurrency = np.zeros(n_bars)

    # Get integer locations of start times
    if not price_index.is_monotonic_increasing:
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
        avg_conc = concurrency[start_idx:end_idx].mean() if start_idx < end_idx else 1.0
        uniqueness.append(1.0 / (avg_conc + 1e-12))

    return np.array(uniqueness)

# -------------------------------------------------------------------------
# 2. The Shared Weight Generator (Source of Truth)
# -------------------------------------------------------------------------
def generate_weights_per_label(
    returns,
    t_events,
    close_series,
    consistency_scores,    # Pre-calculated horizon consistency
    uniqueness_scores,     # Pre-calculated 1/N concurrency
    y=None,                # Labels (for class balancing)
    vol_proxy=None,        # Pre-calculated rolling volatility

    # Layer 1 HPO parameters (The Lens)
    mag_compression=0.8,   # 0=Linear, 1=Log-like
    learn_slope=10.0,
    learn_center=0.4,
    uniq_intensity=1.0,
    class_balance_rate=1.0,# Gamma for class weighting

    # Component exponents
    exp_mag=1.0,
    exp_learn=1.0,
    exp_uniq=1.0,

    # Optional Asymmetry
    downside_multiplier=1.0,

    # Bounds
    floor_weight=1e-4,
    cap_weight=50.0
):
    """
    Vectorized "Source of Truth" for Sample Weighting.
    Used by BOTH Layer 1 (Optimization) and Layer 2 (Training).
    """
    # --- 0. Data Hygiene ---
    returns = np.nan_to_num(returns, nan=0.0)
    consistency_scores = np.nan_to_num(consistency_scores, nan=0.0)
    uniqueness_scores = np.nan_to_num(uniqueness_scores, nan=1.0)

    if vol_proxy is None:
        vol_proxy = np.ones_like(returns)
    vol_proxy = np.maximum(vol_proxy, 1e-8)

    # --- 1. Magnitude Weighting (Risk-Adjusted) ---
    # Weight = |Return / Vol| ^ Compression
    risk_adjusted_ret = returns / vol_proxy
    w_mag = np.power(np.abs(risk_adjusted_ret), mag_compression)

    # Pain Bias (Optional)
    if downside_multiplier != 1.0:
        w_mag = np.where(returns < 0, w_mag * downside_multiplier, w_mag)

    # --- 2. Learnability (Consistency Gate) ---
    # Sigmoid Gate on Horizon Consistency
    w_learn = 1 / (1 + np.exp(-learn_slope * (consistency_scores - learn_center)))

    # --- 3. Uniqueness Weighting ---
    w_uniq = np.power(uniqueness_scores, uniq_intensity)

    # --- 4. Class Balancing (Gamma) ---
    w_class = np.ones_like(returns)
    if y is not None and class_balance_rate > 0:
        # Check if y is numpy array or series and has values
        y_vals = y.values if hasattr(y, 'values') else y
        classes = np.unique(y_vals)
        if len(classes) > 1:
            n_samples = len(y_vals)
            n_classes = len(classes)
            for c in classes:
                n_j = np.sum(y_vals == c)
                if n_j > 0:
                    base_w = n_samples / (n_classes * n_j)
                    # Apply Gamma Intensity
                    w_class[y_vals == c] = np.power(base_w, class_balance_rate)

    # --- 5. Composition (Geometric Mean Logic) ---
    # Combine orthogonal factors
    w_final = (w_mag**exp_mag) * (w_learn**exp_learn) * (w_uniq**exp_uniq) * w_class

    # --- 6. Bounds & Normalization ---
    w_final = np.clip(w_final, floor_weight, cap_weight)
    w_final /= (np.mean(w_final) + 1e-12)

    return w_final
