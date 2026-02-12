"""
Sample Uniqueness Weighting (AFML Chapter 4)

Addresses label overlap and information leakage by computing sample uniqueness
based on label time ranges and overlaps.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional


def concurrent_on_event_grid(label_times: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    """
    Compute number of concurrent labels at each point in the grid using sweep-line algorithm.
    """
    # grid must be sorted unique
    # Use int64 for speed
    starts = label_times["t_start"].values.astype("datetime64[ns]")
    ends = label_times["t_end"].values.astype("datetime64[ns]")

    # Handle NaNs
    valid = (~pd.isna(starts)) & (~pd.isna(ends))

    g = grid.values.astype("datetime64[ns]")

    # Find start and end indices in grid
    # i0: first index where grid >= start
    i0 = np.searchsorted(g, starts[valid], side="left")
    # i1: last index where grid <= end -> searchsorted(side="right") gives index where grid > end, so -1
    i1 = np.searchsorted(g, ends[valid], side="right") - 1

    i0 = np.clip(i0, 0, len(g)-1)
    i1 = np.clip(i1, 0, len(g)-1)

    # Difference array
    diff = np.zeros(len(g) + 1, dtype=np.int32)
    np.add.at(diff, i0, 1)
    np.add.at(diff, i1 + 1, -1)

    return diff[:-1].cumsum()


def avg_uniqueness_on_grid(label_times: pd.DataFrame, grid: pd.DatetimeIndex, concurrent: np.ndarray) -> pd.Series:
    """
    Compute average uniqueness for each label using prefix sums of inverse concurrency.
    """
    g = grid.values.astype("datetime64[ns]")
    starts = label_times["t_start"].values.astype("datetime64[ns]")
    ends = label_times["t_end"].values.astype("datetime64[ns]")

    valid = (~pd.isna(starts)) & (~pd.isna(ends))

    # We need indices for all labels to return aligned Series
    # For invalid labels, searchsorted result doesn't matter as we mask them
    i0 = np.searchsorted(g, starts, side="left")
    i1 = np.searchsorted(g, ends, side="right") - 1

    i0 = np.clip(i0, 0, len(g)-1)
    i1 = np.clip(i1, 0, len(g)-1)

    # Inverse concurrency
    inv = np.zeros_like(concurrent, dtype=np.float64)
    nz = concurrent > 0
    inv[nz] = 1.0 / concurrent[nz]

    # Prefix sum
    prefix = np.concatenate(([0.0], np.cumsum(inv)))

    out = np.zeros(len(label_times), dtype=np.float64)

    # Calculate average uniqueness for valid ranges
    # range is [i0, i1] inclusive
    # sum = prefix[i1+1] - prefix[i0]
    # count = i1 - i0 + 1

    v = valid & (i0 <= i1)

    if v.any():
        idx_v = np.where(v)[0]
        i0_v = i0[idx_v]
        i1_v = i1[idx_v]

        sums = prefix[i1_v + 1] - prefix[i0_v]
        counts = (i1_v - i0_v + 1).astype(np.float32)

        out[idx_v] = sums / counts

    return pd.Series(out, index=label_times.index)


def compute_avg_uniqueness(
    label_times: pd.DataFrame,
    num_concurrent_labels: Union[pd.Series, None] = None,
    time_grid: Optional[pd.DatetimeIndex] = None
) -> pd.Series:
    """
    Compute average uniqueness of samples based on label overlap.
    
    Args:
        label_times: DataFrame with columns ['t_start', 't_end'] indexed by sample
        num_concurrent_labels: Deprecated/Ignored if time_grid is used.
        time_grid: Optional DatetimeIndex representing the sampling grid (e.g. price bars).
                   If None, it is inferred from event boundaries.
        
    Returns:
        Series of uniqueness weights (0-1) indexed by sample
        
    AFML Chapter 4: u[i] = 1 / (# concurrent labels at each timestamp in label[i])
    averaged over the label's time span.
    """
    if label_times.empty:
        return pd.Series(dtype=float)
    
    # Use local copy to avoid mutating input index if it's not DatetimeIndex
    # We actually don't need index to be datetime for the algo,
    # but we need 't_start' and 't_end' columns to be datetime-like.
    
    # Ensure t_start/t_end are datetime
    # We rely on .values.astype("datetime64[ns]") in helper functions,
    # so we assume they are compatible.
    
    if time_grid is None:
        # Build event index (all unique timestamps where labels start or end)
        # Optimized construction
        t_starts = label_times['t_start'].dropna().values.astype("datetime64[ns]")
        t_ends = label_times['t_end'].dropna().values.astype("datetime64[ns]")
        
        # Use numpy unique
        all_times_ns = np.unique(np.concatenate([t_starts, t_ends]))
        time_grid = pd.DatetimeIndex(all_times_ns).sort_values()
    else:
        # Ensure sorted unique
        if not time_grid.is_monotonic_increasing:
             time_grid = time_grid.sort_values()
        # We assume grid is unique enough or searchsorted handles it gracefully
    
    # 1. Compute concurrency
    concurrent = concurrent_on_event_grid(label_times, time_grid)
    
    # 2. Compute uniqueness
    uniqueness = avg_uniqueness_on_grid(label_times, time_grid, concurrent)
    
    return uniqueness


def compute_sample_weights_with_uniqueness(
    label_times: pd.DataFrame,
    returns: Union[pd.Series, np.ndarray],
    base_weights: Union[pd.Series, np.ndarray, None] = None,
    time_grid: Optional[pd.DatetimeIndex] = None
) -> np.ndarray:
    """
    Compute sample weights combining uniqueness and return magnitude.
    
    Args:
        label_times: DataFrame with ['t_start', 't_end'] columns
        returns: Realized returns for weighting by magnitude
        base_weights: Optional base weights (e.g., from exhaustion probability)
        time_grid: Optional time grid for uniqueness calculation
        
    Returns:
        Combined sample weights as numpy array
    """
    # Compute uniqueness
    uniqueness_ser = compute_avg_uniqueness(label_times, time_grid=time_grid)
    uniqueness = uniqueness_ser.values
    
    # Prepare returns array
    if isinstance(returns, pd.Series):
        # Align if series
        # Assuming index alignment matches label_times
        if not returns.index.equals(label_times.index):
             returns = returns.reindex(label_times.index).fillna(0.0)
        returns_arr = returns.values
    else:
        returns_arr = np.asarray(returns)
        if len(returns_arr) != len(uniqueness):
            # Fallback or error? Assuming aligned if array
             pass

    # Return magnitude weighting (log scale)
    magnitude_weight = np.log1p(np.abs(returns_arr))
    
    # Combine: w = uniqueness * magnitude * base_weight
    combined = uniqueness * magnitude_weight
    
    if base_weights is not None:
        if isinstance(base_weights, pd.Series):
             if not base_weights.index.equals(label_times.index):
                 base_weights = base_weights.reindex(label_times.index).fillna(1.0)
             base_arr = base_weights.values
        else:
             base_arr = np.asarray(base_weights)

        combined = combined * base_arr
    
    # Normalize to sum to N (standard practice)
    # Handle degenerate case where sum is 0
    sum_w = combined.sum()
    if sum_w < 1e-12:
        weights = np.ones_like(combined) * (len(combined) / len(combined)) # basically 1.0
        # Wait, if everything is 0, we probably want uniform weights?
        # Or if 0 return and 0 uniqueness?
        # Let's stick to uniform if sum is 0
        weights = np.ones(len(combined))
    else:
        weights = combined * (len(combined) / sum_w)
    
    # Clip extremes
    weights = np.clip(weights, 0.1, 10.0)
    
    return weights


def build_label_time_ranges(
    entry_times: pd.DatetimeIndex,
    exit_times: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Build label time ranges DataFrame from entry/exit timestamps.
    
    Args:
        entry_times: Timestamps when labels start
        exit_times: Timestamps when labels end
        
    Returns:
        DataFrame with ['t_start', 't_end'] indexed by sample
    """
    if len(entry_times) != len(exit_times):
        raise ValueError("entry_times and exit_times must have same length")
    
    return pd.DataFrame({
        't_start': entry_times,
        't_end': exit_times
    }, index=range(len(entry_times)))


def compute_mfe_mae_weights(
    mfe: np.ndarray,
    mae: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    is_timeout: np.ndarray,
    touch_margin: Optional[np.ndarray] = None,
    w_min: float = 0.5,
    tau: float = 1.0,
    cost_floor: float = 0.001
) -> np.ndarray:
    """
    Compute sample weights based on MFE/MAE relative to barriers.
    
    Rationale:
    - r_mfe = MFE/TP: how much of TP was "within reach"
    - r_mae = MAE/SL: how close to SL did we get
    - d = max(r_mfe, r_mae): the more extreme excursion (normalized)
    - w_base = w_min + (1-w_min) * clip(d/tau, 0, 1)
    
    This weights samples by how "decisive" the price movement was,
    without weighting by speed (regime bias) or net R:R (trading policy).
    
    Args:
        mfe: Max Favorable Excursion (positive, as fraction of price)
        mae: Max Adverse Excursion (positive, as fraction of price)
        tp: Take-profit barrier distance (positive, as fraction of price)
        sl: Stop-loss barrier distance (positive, as fraction of price)
        is_timeout: Boolean array, True if label hit timeout (no TP/SL)
        touch_margin: How close to barrier when hit (optional, for cost floor check)
        w_min: Minimum weight floor (default 0.5)
        tau: Scaling factor for d/tau (default 1.0)
        cost_floor: Minimum touch margin to avoid cost-floor penalty (default 0.1%)
        
    Returns:
        Sample weights as numpy array
    """
    mfe = np.asarray(mfe, dtype=np.float64)
    mae = np.asarray(mae, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)
    sl = np.asarray(sl, dtype=np.float64)
    is_timeout = np.asarray(is_timeout, dtype=bool)
    
    # Ensure positive barriers
    tp = np.maximum(tp, 1e-8)
    sl = np.maximum(sl, 1e-8)
    
    # Normalized excursions
    r_mfe = np.maximum(mfe, 0.0) / tp  # How much of TP was within reach
    r_mae = np.maximum(mae, 0.0) / sl  # How close to SL
    
    # Take the more extreme normalized excursion
    d = np.maximum(r_mfe, r_mae)
    
    # Base weight: w_min + (1-w_min) * clip(d/tau, 0, 1)
    w_base = w_min + (1.0 - w_min) * np.clip(d / tau, 0.0, 1.0)
    
    # Cost floor penalty: if touch margin < cost_floor, halve the weight
    if touch_margin is not None:
        touch_margin = np.asarray(touch_margin, dtype=np.float64)
        cost_floor_mask = touch_margin < cost_floor
        w_base = np.where(cost_floor_mask, w_base * 0.5, w_base)
    
    # Timeout cap: if timeout, cap weight at 0.7
    w = np.where(is_timeout, np.minimum(w_base, 0.7), w_base)
    
    return w.astype(np.float32)
