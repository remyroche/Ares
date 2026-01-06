"""
Signal Spacing Utilities for Meta-Labeling.

Provides functions to filter signals based on minimum spacing and priority,
reducing signal density while keeping the highest-quality opportunities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success
except ImportError:
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)


def compute_indicator_strength(
    signals_df: pd.DataFrame,
    momentum_col: str = "momentum_score",
    mr_col: str = "mr_score",
) -> pd.Series:
    """
    Compute indicator strength as priority for signal spacing.
    
    This is the absolute sum of momentum and mean-reversion scores,
    representing how strongly the indicators agree on a signal.
    
    This is a PRE-LABELING metric (no model predictions involved),
    making it safe to use for signal selection without circular dependency.
    
    Args:
        signals_df: DataFrame with signal scores
        momentum_col: Column name for momentum score
        mr_col: Column name for mean-reversion score
        
    Returns:
        Series of indicator strength values (higher = stronger signal)
    """
    momentum = signals_df[momentum_col] if momentum_col in signals_df.columns else 0
    mr = signals_df[mr_col] if mr_col in signals_df.columns else 0
    
    # Absolute sum: strong momentum OR strong MR = high priority
    strength = (momentum.abs() + mr.abs()).fillna(0)
    
    return strength


def compute_expected_signal_weight(
    df: pd.DataFrame,
    atr_col: str = "atr",
    close_col: str = "close",
    consistency_col: Optional[str] = None,
    vol_proxy_col: Optional[str] = None,
    # HPO Parameters
    mag_compression: float = 0.8,
    exp_mag: float = 1.0,
    exp_uniq: float = 1.0,
    exp_learn: float = 1.0,
    exp_cross: float = 1.0,  # New: Cross-interaction exponent
    learn_slope: float = 10.0,
    learn_center: float = 0.4,
    downside_multiplier: float = 1.0,
    uniq_intensity: float = 1.0,  # Included for completeness
    mag_clip_pct: float = 0.99,   # Now configurable
) -> pd.Series:
    """
    Compute expected signal weight BEFORE labeling using FULL HPO logic.
    
    Includes all requested factors:
        Weight ~ (Mag^p1 * Uniq^p2 * Time^p3 * Consistency * Volatility * DownsideRisk)
        
    Args:
        df: DataFrame with market data
        atr_col: ATR column name (Magnitude proxy)
        close_col: Close price column name
        consistency_col: Horizon consistency column name (optional)
        vol_proxy_col: Volatility proxy column name (optional)
        downside_multiplier: Penalty for high downside risk (approximated by vol)
        
    Returns:
        Series of expected weights, normalized to mean=1.0
    """
    n_samples = len(df)
    if n_samples == 0:
        return pd.Series(dtype=float)

    # 1. Magnitude Component (ATR% proxy)
    if atr_col in df.columns and close_col in df.columns:
        atr_pct = df[atr_col].abs() / (df[close_col].abs() + 1e-8)
        # Use configurable clip percentile
        clip_val = atr_pct.quantile(mag_clip_pct)
        norm_mag = np.clip(atr_pct, 0.0, clip_val) / (clip_val + 1e-9)
    else:
        norm_mag = pd.Series(0.01, index=df.index)

    comp_mag = np.power(norm_mag, mag_compression)

    # 2. Uniqueness Component (Default 1.0 for pre-filtering)
    comp_uniq = pd.Series(1.0, index=df.index)
    
    # 3. Time Component (Sigmoid)
    x = np.linspace(0.0, 1.0, n_samples)
    z = learn_slope * (x - learn_center)
    comp_time = pd.Series(1.0 / (1.0 + np.exp(-z)), index=df.index)
    
    # 4. Consistency Component (if available)
    # Higher consistency = stronger trend = higher weight
    if consistency_col and consistency_col in df.columns:
        cons = df[consistency_col].fillna(0.5)
        comp_cons = np.power(cons, exp_cross) # Use exp_cross for interaction terms
    else:
        comp_cons = pd.Series(1.0, index=df.index)
        
    # 5. Volatility Proxy & Downside Penalty
    # We use volatility to approximate downside risk.
    # Higher volatility = Higher potential risk -> apply downside_multiplier penalty
    # Penalty logic: Weight /= multiplier if Vol is High (above median)
    comp_risk = pd.Series(1.0, index=df.index)
    if vol_proxy_col and vol_proxy_col in df.columns:
        vol = df[vol_proxy_col].fillna(0)
        vol_median = vol.rolling(100, min_periods=10).median().fillna(vol.mean())
        
        # If Vol > Median, apply penalty
        high_vol_mask = vol > vol_median
        # Smooth penalty: 1.0 -> 1/multiplier
        penalty_factor = 1.0 / downside_multiplier
        comp_risk[high_vol_mask] = penalty_factor
        
        # Also scale magnitude by volatility regime? 
        # Actually HPO logic keeps them separate, so we just use it as a penalty multiplier.

    # 6. Geometric Mixing
    raw_weights = (
        np.power(comp_mag, exp_mag) * 
        np.power(comp_uniq, exp_uniq) * 
        np.power(comp_time, exp_learn) *
        comp_cons *  # Consistency boosts weight
        comp_risk    # Risk penalizes weight
    )
    
    # Normalize
    weight_mean = raw_weights.mean()
    if weight_mean > 0:
        expected_weight = raw_weights / weight_mean
    else:
        expected_weight = raw_weights.replace(0, 1.0)
    
    return expected_weight.fillna(1.0)


def apply_signal_spacing_filter(
    signals: pd.Series,
    min_spacing_bars: int = 4,
    priority_col: Optional[pd.Series] = None,
    max_signals_per_day: Optional[float] = None,
    bars_per_day: int = 96,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Apply minimum spacing between signals, keeping only the highest-priority signals.
    
    This function ensures signals are at least `min_spacing_bars` apart.
    When multiple signals fall within a spacing window, only the one with
    the highest priority is kept.
    
    IMPORTANT: To avoid circular dependencies, use PRE-LABELING priority metrics:
    - |momentum_score + mr_score| (raw indicator strength)
    - CUSUM threshold breach magnitude
    - Volatility-normalized signal strength
    - Simple time-based (None = chronological order)
    
    DO NOT use meta_probability or target_sample_weight as priority,
    as these involve post-hoc information or model predictions.
    
    Args:
        signals: Series with signal values (1, -1, or 0)
        min_spacing_bars: Minimum bars between signals (default: 4 = 1 hour for 15m)
        priority_col: Optional priority series (higher = more important to keep).
                     Recommended: abs(momentum_score) or indicator_strength.
                     If None, signals are kept in chronological order.
        max_signals_per_day: Optional max signals per day. If set, dynamically
                            increases min_spacing_bars to achieve target density.
        bars_per_day: Number of bars per day for density calculations (default: 96 for 15m)
        
    Returns:
        Tuple[Filtered signals (Series), Stats (Dict)]
    """
    empty_stats = {
        "original_count": 0,
        "final_count": 0,
        "reduction_pct": 0.0,
        "signals_per_day": 0.0,
    }

    if signals.empty or (signals == 0).all():
        return signals.copy(), empty_stats
    
    # Work with a copy
    filtered = pd.Series(0, index=signals.index, dtype=signals.dtype)
    
    # Get non-zero signal indices
    signal_mask = signals != 0
    signal_indices = signals.index[signal_mask].tolist()
    
    if len(signal_indices) == 0:
        return filtered, empty_stats
    
    # Calculate current density
    n_days = len(signals) / bars_per_day
    current_density = len(signal_indices) / max(n_days, 1)
    
    # Dynamically adjust spacing if max_signals_per_day is set
    effective_spacing = min_spacing_bars
    if max_signals_per_day is not None and max_signals_per_day > 0:
        target_density = max_signals_per_day
        if current_density > target_density:
            # Increase spacing to reduce density
            # spacing = bars_per_day / signals_per_day
            required_spacing = int(np.ceil(bars_per_day / target_density))
            effective_spacing = max(min_spacing_bars, required_spacing)
            tprint_info(f"Signal spacing adjusted from {min_spacing_bars} to {effective_spacing} "
                       f"to achieve {target_density:.1f} signals/day (current: {current_density:.1f})")
    
    # Convert to positional indices for faster processing
    idx_to_pos = {idx: pos for pos, idx in enumerate(signals.index)}
    pos_to_idx = {pos: idx for idx, pos in idx_to_pos.items()}
    
    # Create list of (position, index, signal, priority) tuples
    signal_data = []
    for idx in signal_indices:
        pos = idx_to_pos[idx]
        sig_val = signals.loc[idx]
        pri = priority_col.loc[idx] if priority_col is not None else 0.0
        signal_data.append((pos, idx, sig_val, pri))
    
    # Sort by position (chronological order)
    signal_data.sort(key=lambda x: x[0])
    
    # Greedy selection with spacing constraint
    # Strategy: Process signals in order, keep highest-priority within each window
    kept_signals = []
    i = 0
    
    while i < len(signal_data):
        pos, idx, sig_val, pri = signal_data[i]
        
        # Find all signals within the spacing window from this position
        window_signals = [(pos, idx, sig_val, pri)]
        j = i + 1
        while j < len(signal_data):
            next_pos = signal_data[j][0]
            if next_pos - pos < effective_spacing:
                window_signals.append(signal_data[j])
                j += 1
            else:
                break
        
        # Keep the signal with highest priority in this window
        best_signal = max(window_signals, key=lambda x: x[3])
        kept_signals.append(best_signal)
        
        # Move past all signals in this window
        i = j
    
    # Build the filtered series
    for pos, idx, sig_val, pri in kept_signals:
        filtered.loc[idx] = sig_val
    
    # Calculate statistics
    kept_count = len(kept_signals)
    original_count = len(signal_indices)
    filtered_density = kept_count / max(n_days, 1)
    
    tprint_info(f"Signal spacing filter: {original_count} → {kept_count} signals "
                   f"({kept_count/max(original_count,1)*100:.1f}% kept, "
                   f"{filtered_density:.1f} signals/day)")
    
    stats = {
        "original_count": original_count,
        "final_count": kept_count,
        "reduction_pct": (1.0 - kept_count/max(original_count, 1)) * 100,
        "signals_per_day": filtered_density,
    }

    return filtered, stats


def compute_signal_spacing_stats(
    signals: pd.Series,
    bars_per_day: int = 96,
) -> Dict[str, Any]:
    """
    Compute statistics about signal spacing and density.
    
    Args:
        signals: Series with signal values
        bars_per_day: Number of bars per day
        
    Returns:
        Dictionary with spacing statistics including:
        - signals_per_day: Average signals per day
        - mean_spacing: Average bars between consecutive signals
        - min_spacing: Minimum spacing observed
        - spacing_std: Standard deviation of spacing
    """
    signal_mask = signals != 0
    signal_positions = np.where(signal_mask)[0]
    
    n_signals = len(signal_positions)
    n_days = len(signals) / bars_per_day
    
    stats = {
        "n_signals": n_signals,
        "n_days": float(n_days),
        "signals_per_day": n_signals / max(n_days, 1),
    }
    
    if n_signals < 2:
        stats["mean_spacing"] = np.nan
        stats["min_spacing"] = np.nan
        stats["max_spacing"] = np.nan
        stats["spacing_std"] = np.nan
        return stats
    
    spacings = np.diff(signal_positions)
    
    stats["mean_spacing"] = float(np.mean(spacings))
    stats["min_spacing"] = int(np.min(spacings))
    stats["max_spacing"] = int(np.max(spacings))
    stats["spacing_std"] = float(np.std(spacings))
    
    return stats


def recommend_signal_spacing(
    target_signals_per_day: float = 15.0,
    timeframe_minutes: int = 15,
) -> Dict[str, Any]:
    """
    Recommend signal spacing parameters based on target density and timeframe.
    
    Args:
        target_signals_per_day: Desired number of signals per trading day
        timeframe_minutes: Timeframe in minutes
        
    Returns:
        Dictionary with recommended parameters.
    """
    bars_per_day = (24 * 60) / timeframe_minutes
    
    # Calculate required spacing
    if target_signals_per_day <= 0:
        min_spacing = int(bars_per_day)  # 1 signal per day
    else:
        min_spacing = int(bars_per_day / target_signals_per_day)
    
    # Ensure at least 1 bar spacing
    min_spacing = max(1, min_spacing)
    
    # Calculate effective signals per day with this spacing
    actual_signals_per_day = bars_per_day / min_spacing
    
    return {
        "target_signals_per_day": target_signals_per_day,
        "bars_per_day": bars_per_day,
        "recommended_min_spacing_bars": min_spacing,
        "effective_signals_per_day": actual_signals_per_day,
        "timeframe_minutes": timeframe_minutes,
        "human_readable_spacing": f"{min_spacing * timeframe_minutes} minutes",
    }
