"""
Generate Sample Weights per Label.

This module implements sample weighting based on label uniqueness and overlap,
as described in "Advances in Financial Machine Learning" by Marcos López de Prado.

The core idea is that overlapping labels contain redundant information. To prevent
the model from overweighting periods with high label density (concurrency), we
downweight samples proportional to their overlap.

Algorithm:
1. Count concurrent events at each timestamp.
2. Calculate uniqueness at each timestamp (1 / concurrency).
3. Compute average uniqueness for each event over its duration.

This ensures that highly overlapping events have lower individual weights,
providing a more balanced training signal.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning

_COERCE_NONFINITE_WARNED: set = set()

_WEIGHT_LOG_COUNTER = 0
_WEIGHT_LOG_LIMIT = 5

try:
    from scipy.stats import entropy as scipy_entropy, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_entropy = None
    spearmanr = None

def finalize_sample_weights(weights: np.ndarray) -> np.ndarray:
    """
    Robustly stabilizes weights for LightGBM.
    Process: MAD Clipping -> Mean-Centering.

    Args:
        weights: Raw combined weights array

    Returns:
        Processed weights array with mean 1.0 and clipped extremes
    """
    w = np.array(weights, dtype=float).copy()

    # 1. Robust Statistics (MAD)
    median = np.nanmedian(w)
    # 1.4826 scales MAD to be consistent with StdDev for normal distributions
    mad = np.nanmedian(np.abs(w - median)) * 1.4826

    # 2. Define Upper Bound
    # If distribution is flat (all weights equal), MAD is 0. Fallback to max.
    if mad <= 1e-12:
        upper_bound = np.nanmax(w)
    else:
        # k=5 is roughly equivalent to 5-sigma. Very safe.
        upper_bound = median + (5 * mad)

    # 3. Clip (Winsorize)
    # We clip high values. We also ensure no weight is exactly 0 (use epsilon)
    # unless it was originally 0 or negative (which we keep as 0 or clip to min).
    # Assuming input weights are >= 0.
    w_clipped = np.clip(w, a_min=1e-4, a_max=upper_bound)

    # Restore actual zeros if they were intentional (e.g. from Cost-Adjusted Magnitude)
    # If original weight was < 1e-6, it might be noise or intentional zero.
    # The user logic "Zeroes out the weight for Fake Opportunities" implies we allow 0.
    # But finalize_sample_weights says "ensure no weight is exactly 0 (use epsilon)".
    # However, LightGBM handles 0 weights fine (ignores sample).
    # If we want to zero out, we should allow 0.
    # The user's code snippet: "w_clipped = np.clip(w, a_min=1e-4, ...)" implies they WANT to lift zeros to 1e-4.
    # Maybe 1e-4 is "small enough to be ignored but safe for logs/division".

    # 4. Normalize (Mean=1.0)
    # This ensures the effective learning rate matches your params.
    mean_val = np.nanmean(w_clipped)
    w_final = w_clipped / (mean_val + 1e-9)

    return w_final

def compute_uniqueness(
    t1: pd.Series,
    events_index: Optional[pd.DatetimeIndex] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute sample weights based on label uniqueness (overlap).
    Alias for compute_uniqueness_weights for backwards compatibility.

    Args:
        t1: Series with event end times (indices), indexed by event start times.
            t1.index = t0 (start time)
            t1.values = t1 (end time)
        events_index: Optional index of events if t1 is not indexed by time.
        market_index: Optional full market index to align timestamps.

    Returns:
        Series of sample weights aligned with t1.index.
    """
    return compute_uniqueness_weights(t1, events_index, market_index)

def compute_uniqueness_weights(
    t1: pd.Series,
    events_index: Optional[pd.DatetimeIndex] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute sample weights based on label uniqueness (overlap).

    Args:
        t1: Series with event end times (indices), indexed by event start times.
            t1.index = t0 (start time)
            t1.values = t1 (end time)
        events_index: Optional index of events if t1 is not indexed by time.
        market_index: Optional full market index to align timestamps.

    Returns:
        Series of sample weights aligned with t1.index.
    """
    tprint(f"⚖️ Computing uniqueness weights for {len(t1)} events...", "INFO")

    if len(t1) == 0:
        tprint("⚠️ No events provided for weighting. Returning empty weights.", "WARNING")
        return pd.Series(dtype=float)

    # 1. Expand events to valid time range
    if events_index is not None:
        t1_aligned = pd.Series(t1.values, index=events_index)
    else:
        t1_aligned = t1.copy()

    # Sort by index to ensure proper processing
    t1_aligned = t1_aligned.sort_index()
    
    # 2. Derive concurrency (how many events are active at each time step)
    # We create a timeline of all start (t0) and end (t1) points
    start_times = pd.Series(1, index=t1_aligned.index)
    end_times = pd.Series(-1, index=t1_aligned.values)
    
    # Combine start and end times to track active count changes
    # Note: If an event ends strictly BEFORE the next one starts, this works. 
    # If using bars, ensure proper alignment. 
    # Here we assume continuous timestamps.
    timeline = pd.concat([start_times, end_times]).sort_index()
    
    # If multiple events start/end at same time, sum the changes
    timeline = timeline.groupby(timeline.index).sum()
    
    # Cumulative sum gives the number of active events at any point
    concurrency = timeline.cumsum()

    # 3. Calculate uniqueness based on average concurrency over event lifespan
    # Optimization: Reindex concurrency to a granular market index if provided, 
    # to account for duration more accurately
    if market_index is not None:
        # Realign concurrency to market bars to avoid gaps
        concurrency_aligned = concurrency.reindex(market_index, method='ffill').fillna(0)
    else:
        concurrency_aligned = concurrency

    # Ensure monotonically increasing index for searchsorted operations
    if not concurrency_aligned.index.is_monotonic_increasing:
        concurrency_aligned = concurrency_aligned.sort_index()

    index_values = concurrency_aligned.index.values
    conc_values = concurrency_aligned.astype(float).values

    # Inverse concurrency at each timestamp
    inv_conc = 1.0 / np.maximum(1.0, conc_values)

    # Cumulative sum of inverse concurrency
    cum_inv = np.cumsum(inv_conc)

    # Map event start/end times to integer positions in the concurrency index
    t0_values = t1_aligned.index.values
    t1_values = t1_aligned.values

    start_pos = concurrency_aligned.index.searchsorted(t0_values, side='left')
    end_pos = concurrency_aligned.index.searchsorted(t1_values, side='right') - 1

    n_points = len(concurrency_aligned)

    # Valid events have at least one overlapping timestamp in the concurrency index
    valid_mask = (start_pos < n_points) & (end_pos >= 0) & (start_pos <= end_pos)

    durations = np.zeros(len(t1_aligned), dtype=float)
    sum_inv = np.zeros(len(t1_aligned), dtype=float)

    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        sp_valid = start_pos[valid_idx]
        ep_valid = end_pos[valid_idx]

        durations[valid_idx] = (ep_valid - sp_valid + 1).astype(float)

        # Sum of inverse concurrency over each event window via prefix sums
        prev_cum = np.where(sp_valid > 0, cum_inv[sp_valid - 1], 0.0)
        sum_inv[valid_idx] = cum_inv[ep_valid] - prev_cum

    # Average uniqueness per event
    avg_uniqueness = np.ones(len(t1_aligned), dtype=float)
    valid_and_nonzero = valid_mask & (durations > 0)
    if valid_and_nonzero.any():
        avg_uniqueness[valid_and_nonzero] = sum_inv[valid_and_nonzero] / durations[valid_and_nonzero]

    weights = pd.Series(avg_uniqueness, index=t1_aligned.index, dtype=float)

    # Normalize weights to sum to count (preserve total effective sample size)
    if weights.sum() > 0:
        weights *= (len(weights) / weights.sum())
    
    tprint(f"✅ Uniqueness weights computed. Mean: {weights.mean():.4f}, Min: {weights.min():.4f}, Max: {weights.max():.4f}", "SUCCESS")
    return weights


def _coerce_numeric_array(
    arr: Optional[Union[np.ndarray, pd.Series, list]],
    expected_len: int,
    name: str,
    fill_value: float = 0.0,
    allow_negative: bool = True,
) -> Optional[np.ndarray]:
    """
    Convert an input array-like to a clean numpy array with finite values.

    - Verifies length matches expected_len; otherwise returns None.
    - Replaces non-finite values with fill_value.
    - Optionally enforces positivity with a floor.
    """
    if arr is None:
        return None

    arr_np = np.asarray(arr, dtype=float)
    if len(arr_np) != expected_len:
        tprint_warning(
            f"⚠️ {name} length ({len(arr_np)}) does not match returns ({expected_len}); ignoring {name}."
        )
        return None

    non_finite_mask = ~np.isfinite(arr_np)
    if non_finite_mask.any():
        warn_key = (name, "non_finite")
        if warn_key not in _COERCE_NONFINITE_WARNED:
            tprint_warning(
                f"⚠️ {name} contains {non_finite_mask.sum()} non-finite values; replacing with {fill_value}."
            )
            _COERCE_NONFINITE_WARNED.add(warn_key)
        arr_np = np.where(non_finite_mask, fill_value, arr_np)

    if not allow_negative:
        min_positive = 1e-12
        arr_np = np.abs(arr_np)
        arr_np = np.where(arr_np < min_positive, min_positive, arr_np)

    return arr_np

def compute_horizon_consistency(price_series: pd.Series, horizon: int = 12) -> pd.Series:
    """
    Compute consistency of price direction over the horizon.
    High consistency means price moved steadily in one direction.
    Low consistency means price chopped back and forth.
    
    Args:
        price_series: Series of prices
        horizon: Lookahead horizon in bars
        
    Returns:
        Series of consistency scores (0.0 to 1.0)
    """
    # Calculate returns
    returns = price_series.pct_change().fillna(0.0)
    
    # Absolute sum of returns over horizon (path length)
    abs_sum = returns.abs().rolling(horizon).sum()
    
    # Net return over horizon (displacement)
    net_ret = returns.rolling(horizon).sum().abs()
    
    # Consistency = Displacement / Path Length
    # If moved in straight line: Consistency = 1.0
    # If chopped and ended at same price: Consistency = 0.0
    consistency = net_ret / (abs_sum + 1e-8)
    
    return consistency


def compute_multi_horizon_consistency(
    price_series: pd.Series,
    horizons: Optional[List[int]] = None,
    aggregation: str = "mean",
) -> pd.Series:
    """
    Compute label consistency across multiple horizons.
    
    A sample is considered "consistent" if its label direction agrees
    across different lookahead horizons. This helps identify robust signals
    vs. noise that only appears at specific horizons.
    
    Args:
        price_series: Series of prices
        horizons: List of horizons to check (default: [6, 12, 24])
        aggregation: How to combine scores ("mean", "min", "geometric")
    
    Returns:
        Series of multi-horizon consistency scores (0.0 to 1.0)
    """
    if horizons is None:
        horizons = [6, 12, 24]
    
    # Compute consistency at each horizon
    consistency_scores = []
    for h in horizons:
        cons = compute_horizon_consistency(price_series, horizon=h)
        consistency_scores.append(cons)
    
    # Stack into DataFrame
    cons_df = pd.concat(consistency_scores, axis=1)
    cons_df.columns = [f"h{h}" for h in horizons]
    
    # Aggregate across horizons
    if aggregation == "mean":
        result = cons_df.mean(axis=1)
    elif aggregation == "min":
        result = cons_df.min(axis=1)
    elif aggregation == "geometric":
        # Geometric mean is more sensitive to low values
        result = cons_df.prod(axis=1) ** (1.0 / len(horizons))
    else:
        result = cons_df.mean(axis=1)
    
    return result.fillna(0.5)


def compute_label_agreement_consistency(
    labels_matrix: np.ndarray,
    returns_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute consistency based on agreement among multiple labeling schemes.
    
    For each sample, measures how much the different labeling experts agree.
    High agreement = high consistency = more reliable label.
    
    Args:
        labels_matrix: (n_samples, n_experts) matrix of labels (-1, 0, 1)
        returns_matrix: (n_samples, n_experts) matrix of realized returns
    
    Returns:
        Array of consistency scores (0.0 to 1.0)
    """
    n_samples, n_experts = labels_matrix.shape
    
    # Agreement score: fraction of experts that agree with majority
    consistency = np.zeros(n_samples, dtype=float)
    
    for i in range(n_samples):
        row = labels_matrix[i, :]
        # Count non-zero votes
        nonzero_mask = row != 0
        n_votes = nonzero_mask.sum()
        
        if n_votes == 0:
            consistency[i] = 0.0
            continue
        
        # Count positive and negative votes
        n_pos = (row > 0).sum()
        n_neg = (row < 0).sum()
        
        # Agreement = max(pos, neg) / total_votes
        majority = max(n_pos, n_neg)
        consistency[i] = float(majority) / float(n_votes) if n_votes > 0 else 0.0
    
    return consistency


def compute_return_sign_consistency(
    returns_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute consistency based on return sign agreement across experts.
    
    If all experts produce returns of the same sign, consistency is high.
    Mixed signs indicate an ambiguous/noisy sample.
    
    Args:
        returns_matrix: (n_samples, n_experts) matrix of realized returns
    
    Returns:
        Array of consistency scores (0.0 to 1.0)
    """
    n_samples, n_experts = returns_matrix.shape
    
    consistency = np.zeros(n_samples, dtype=float)
    
    for i in range(n_samples):
        row = returns_matrix[i, :]
        valid_mask = np.isfinite(row) & (row != 0)
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            consistency[i] = 0.0
            continue
        
        valid_returns = row[valid_mask]
        n_pos = (valid_returns > 0).sum()
        n_neg = (valid_returns < 0).sum()
        
        # Consistency = |pos - neg| / total
        consistency[i] = abs(n_pos - n_neg) / float(n_valid)
    
    return consistency

def generate_weights_per_label(
    returns: np.ndarray,
    t_events: pd.Index,
    close_series: Optional[pd.Series] = None,
    consistency_scores: Optional[np.ndarray] = None,
    label_quality_scores: Optional[np.ndarray] = None,
    uniqueness_scores: Optional[np.ndarray] = None,
    vol_proxy: Optional[np.ndarray] = None,
    mag_compression: float = 0.8,
    learn_slope: float = 0.0,
    learn_center: float = 0.5,
    uniq_intensity: float = 1.0,
    quality_intensity: float = 0.0,
    quality_floor: float = 0.2,
    exp_mag: float = 1.0,
    exp_learn: float = 1.0,
    exp_uniq: float = 1.0,
    exp_cross: float = 1.0,
    downside_multiplier: float = 1.0,
    time_decay_halflife: Optional[float] = None,
    mag_clip_pct: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Generate symmetric, bias-free sample weights combining magnitude, uniqueness,
    and time components.

    Structure:
        W = (Magnitude**exp_mag * Uniqueness**exp_uniq * Time**exp_learn)

    Properties:
    - Uses absolute returns only (no sign dependence).
    - Expects uniqueness_scores ≈ 1 / concurrency when provided.
    - Time component is a sigmoid over a normalized index [0, 1].
    - Weights are normalized so that mean(weight) = 1.0.

    Extra inputs like close_series, consistency_scores, vol_proxy, downside_multiplier
    are accepted for backwards compatibility but not used in the core geometry.
    """
    n_samples = len(returns)
    if n_samples == 0:
        tprint_info("⚠️ No returns provided; returning empty weights.")
        return np.array([])

    if len(t_events) != n_samples:
        tprint_warning(
            f"⚠️ t_events length ({len(t_events)}) does not match returns ({n_samples}); proceeding with positional alignment."
        )

    # 1. Clean core returns
    returns_clean = _coerce_numeric_array(
        returns, n_samples, "returns", fill_value=0.0, allow_negative=True
    )
    returns_clean = np.where(np.isfinite(returns_clean), returns_clean, 0.0)

    # 2. Magnitude component (Cost-Adjusted)
    # The Problem: Simply using abs(returns) weights dust moves (e.g. 0.1% gross, 0% net)
    # as having positive magnitude, leading the model to chase unprofitable noise.
    # The Fix: Net Magnitude. We subtract transaction costs before computing magnitude.
    # Formula: log(1 + max(0, abs(returns) - cost))

    # Check if transaction_cost was passed in kwargs, else assume default
    tx_cost = kwargs.get('transaction_cost', 0.003)
    try:
        tx_cost = float(tx_cost)
    except Exception:
        tx_cost = 0.003

    # We assume 'returns_clean' here are raw/gross returns. If they are net returns,
    # we should check if they can be negative. But 'abs(returns)' implies we care
    # about magnitude of move.
    # If returns are signed, we take abs first to get gross move size.
    abs_ret_raw = np.abs(returns_clean)

    # Zero out "fake opportunities" that don't cover the spread
    # max(0, abs(ret) - cost)
    net_mag_raw = np.maximum(0.0, abs_ret_raw - tx_cost)

    # If vol_proxy is available, prefer a volatility-normalized magnitude signal.
    # This keeps weights from being dominated by high-volatility regimes.
    vol_for_mag = _coerce_numeric_array(
        vol_proxy, n_samples, "vol_proxy", fill_value=np.nan, allow_negative=False
    )

    if vol_for_mag is not None:
        try:
            v = np.asarray(vol_for_mag, dtype=float)
            v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
            v_med = float(np.nanmedian(v)) if np.isfinite(np.nanmedian(v)) else np.nan
            if np.isfinite(v_med) and v_med > 0:
                v = np.where(np.isfinite(v), v, v_med)
                # Normalize the NET magnitude by volatility
                net_mag_raw = net_mag_raw / (v + 1e-12)
        except Exception:
            pass

    # Log-dampening: log(1 + x)
    # This compresses whale moves while preserving order.
    # Since we already subtracted cost, x is "excess magnitude".
    # We apply mag_compression as an exponent if desired, but user formula was specific.
    # Let's apply log1p first as the base "Magnitude" component.
    comp_mag_log = np.log1p(net_mag_raw)

    # Apply mag_compression power (defaults to 0.8 in optimization, or 1.0 if not)
    # If mag_compression is 1.0, it's just log1p.
    comp_mag = np.power(comp_mag_log, mag_compression)

    # 3. Uniqueness component (redundancy filter)
    cleaned_uniqueness = _coerce_numeric_array(
        uniqueness_scores, n_samples, "uniqueness_scores", fill_value=1.0, allow_negative=False
    )
    if cleaned_uniqueness is not None:
        comp_uniq = np.power(cleaned_uniqueness, uniq_intensity)
    else:
        comp_uniq = np.ones(n_samples, dtype=float)

    # 4. Time component (sigmoid over normalized time rank)
    x = np.linspace(0.0, 1.0, n_samples, dtype=float)
    if len(t_events) == n_samples:
        try:
            t_values = np.asarray(t_events)
            order = np.argsort(t_values)
            x_rank = np.linspace(0.0, 1.0, n_samples, dtype=float)
            x = np.empty(n_samples, dtype=float)
            x[order] = x_rank
        except Exception:
            pass

    z = learn_slope * (x - learn_center)
    z = np.clip(z, -60.0, 60.0)
    comp_time = 1.0 / (1.0 + np.exp(-z))

    # 5. Consistency component (interaction term)
    cleaned_consistency = _coerce_numeric_array(
        consistency_scores, n_samples, "consistency_scores", fill_value=1.0, allow_negative=False
    )
    if cleaned_consistency is not None:
        # Boost weight if consistent
        comp_cross = np.power(cleaned_consistency, exp_cross)
    else:
        comp_cross = np.ones(n_samples, dtype=float)

    # 6. Confident-learning label quality component
    # This is designed to be monotone and bounded:
    # quality_floor <= q_eff <= 1.0, then raised by quality_intensity.
    comp_quality = np.ones(n_samples, dtype=float)
    try:
        q_int = float(quality_intensity)
    except Exception:
        q_int = 0.0

    if q_int > 0.0:
        cleaned_quality = _coerce_numeric_array(
            label_quality_scores, n_samples, "label_quality_scores", fill_value=1.0, allow_negative=False
        )
        if cleaned_quality is not None:
            q = np.asarray(cleaned_quality, dtype=float)
            q = np.where(np.isfinite(q), q, 1.0)
            q = np.clip(q, 0.0, 1.0)
            try:
                q_floor = float(quality_floor)
            except Exception:
                q_floor = 0.2
            q_floor = float(np.clip(q_floor, 0.0, 1.0))
            q_eff = q_floor + (1.0 - q_floor) * q
            comp_quality = np.power(np.clip(q_eff, q_floor, 1.0), q_int)

    # 7. Downside risk penalty
    # Use vol_proxy to penalize high volatility periods if downside_multiplier > 1.0
    comp_risk = np.ones(n_samples, dtype=float)
    cleaned_vol = _coerce_numeric_array(
        vol_proxy, n_samples, "vol_proxy", fill_value=0.0, allow_negative=False
    )
    
    if cleaned_vol is not None and downside_multiplier > 1.0:
        vol_median = np.nanmedian(cleaned_vol)
        if vol_median > 0:
            high_vol_mask = cleaned_vol > vol_median
            if high_vol_mask.any():
                # Apply penalty: 1.0 -> 1.0 / multiplier
                comp_risk[high_vol_mask] = 1.0 / downside_multiplier

    # 8. Time-decay weighting (Phase 4.3: weight recent data higher)
    comp_decay = np.ones(n_samples, dtype=float)
    if time_decay_halflife is not None and time_decay_halflife > 0:
        try:
            # x is already normalized position [0, 1] where 1 = most recent
            # Exponential decay: weight = 2^((x - 1) / halflife)
            # This makes oldest sample = 2^(-1/halflife), newest = 1.0
            decay_power = (x - 1.0) / time_decay_halflife
            comp_decay = np.power(2.0, decay_power)
            comp_decay = np.clip(comp_decay, 0.1, 1.0)  # Floor at 10% of newest
        except Exception:
            pass

    # 9. Geometric mixing
    raw_weights = (
        np.power(comp_mag, exp_mag)
        * np.power(comp_uniq, exp_uniq)
        * np.power(comp_time, exp_learn)
        * comp_cross
        * comp_quality
        * comp_risk
        * comp_decay
    )

    raw_weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)

    if raw_weights.sum() > 0 and np.isfinite(raw_weights).all():
        # Use robust finalization (MAD clipping + Mean centering)
        final_weights = finalize_sample_weights(raw_weights)
    else:
        tprint_warning("⚠️ Combined weights invalid; falling back to uniform weights.")
        final_weights = np.ones(n_samples, dtype=float)

    global _WEIGHT_LOG_COUNTER
    try:
        if _WEIGHT_LOG_COUNTER < _WEIGHT_LOG_LIMIT:
            loss_mask = returns_clean < 0
            tprint_info(
                f"📊 Weights summary — mean: {final_weights.mean():.4f}, min: {final_weights.min():.4f}, "
                f"max: {final_weights.max():.4f}, neg share: {loss_mask.mean():.2%}, "
                f"neg weight avg: {final_weights[loss_mask].mean() if loss_mask.any() else 0:.4f}"
            )
            _WEIGHT_LOG_COUNTER += 1
    except Exception:
        pass

    return final_weights

