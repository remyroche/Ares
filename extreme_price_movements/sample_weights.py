"""
Sample Uniqueness Weighting (AFML Chapter 4)

Addresses label overlap and information leakage by computing sample uniqueness
based on label time ranges and overlaps.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
from dataclasses import dataclass

# Import tprint for logging
try:
    from .utils import tprint
except ImportError:
    def tprint(msg):
        print(msg)


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
    time_grid: Optional[pd.DatetimeIndex] = None,
    selection_metric: Optional[Union[pd.Series, np.ndarray]] = None
) -> np.ndarray:
    """
    Compute sample weights combining uniqueness, event intensity, and MFE/MAE base weights.
    
    Args:
        label_times: DataFrame with ['t_start', 't_end'] columns
        returns: Realized returns for reference (not used in magnitude weighting)
        base_weights: MFE/MAE-based weights from compute_mfe_mae_weights() function
        time_grid: Optional time grid for uniqueness calculation
        selection_metric: Optional metric values used for candidate selection 
                         (e.g., dist_ema_fast). If provided, event scoring will be
                         based on this instead of realized returns.
        
    Returns:
        Combined sample weights as numpy array
        
    Weight Components:
    - Uniqueness: Addresses label overlap (AFML Chapter 4)
    - Event Score: 1 / (1 - percentile) of selection metric intensity, 
                   normalized to [0.8, 1.2], with percentile clamped to [0.01, 0.99]
    - Base Weights: MFE/MAE-based weights based on decisive price movement relative to barriers
    
    Final formula: w = uniqueness * event_score * mfe_mae_base_weight
    """
    # Compute uniqueness
    uniqueness_ser = compute_avg_uniqueness(label_times, time_grid=time_grid)
    uniqueness = uniqueness_ser.values
    
    # Compute event score based on selection metric intensity percentile
    # More intense events (higher absolute metric values) get higher scores
    if selection_metric is not None:
        # Use the same metric that was used for candidate selection
        if isinstance(selection_metric, pd.Series):
            if not selection_metric.index.equals(label_times.index):
                selection_metric = selection_metric.reindex(label_times.index).fillna(0.0)
            metric_values = np.abs(selection_metric.values)
        else:
            metric_values = np.abs(np.asarray(selection_metric))
        
        tprint(f"Using selection metric for event scoring: mean={metric_values.mean():.3f}, std={metric_values.std():.3f}")
    else:
        # Fallback to realized returns if no selection metric provided
        if isinstance(returns, pd.Series):
            if not returns.index.equals(label_times.index):
                returns = returns.reindex(label_times.index).fillna(0.0)
            returns_arr = returns.values
        else:
            returns_arr = np.asarray(returns)
        metric_values = np.abs(returns_arr)
        tprint(f"Using realized returns for event scoring (fallback)")
    
    # Vectorized percentile calculation
    # Get percentile ranks (0-1) for each metric value
    sorted_indices = np.argsort(metric_values)
    percentile_ranks = np.empty_like(metric_values)
    percentile_ranks[sorted_indices] = np.arange(1, len(metric_values) + 1) / len(metric_values)
    
    # Clamp percentiles to [0.01, 0.99] to prevent extreme scores
    percentile_ranks = np.clip(percentile_ranks, 0.01, 0.99)
    
    # Score = 1 / (1 - percentile), then normalize to [0.8, 1.2]
    # This way: high intensity (high percentile) = higher score
    raw_scores = 1.0 / (1.0 - percentile_ranks + 0.01)  # +0.01 to avoid division by zero
    # Normalize: map typical range [1, 100] to [0.8, 1.2]
    event_scores = 0.8 + 0.4 * (np.log1p(raw_scores - 1) / np.log1p(99))
    
    # Combine: w = uniqueness * event_score * base_weight
    combined = uniqueness * event_scores
    
    # Log weighting statistics for debugging
    tprint(f"Weight components - Uniqueness: mean={uniqueness.mean():.3f}, std={uniqueness.std():.3f}")
    tprint(f"Weight components - Event Score: mean={event_scores.mean():.3f}, std={event_scores.std():.3f}, min={event_scores.min():.3f}, max={event_scores.max():.3f}")
    tprint(f"Weight components - Combined (pre-normalization): mean={combined.mean():.3f}, std={combined.std():.3f}")
    
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

@dataclass(frozen=True)
class NegMassRenormCfg:
    # Timeout raw weight: clip((1 - Pto) / Pto, w_to_min, w_to_max)
    w_to_min: float = 0.2
    w_to_max: float = 1.0

    # StopLoss base raw weight inside negatives (usually 1.0)
    w_sl_raw: float = 1.0

    # Target pos:neg mass ratio rho = M_pos / M_neg_target
    # rho=1.0 means 50/50 mass (M_neg_target = M_pos)
    rho_pos_over_neg: float = 1.0

    # Optional clipping of the per-cell negative scalar alpha to avoid extreme rescaling
    alpha_min: Optional[float] = 0.5
    alpha_max: Optional[float] = 2.0

    # Numerical safety
    eps: float = 1e-12

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def compute_cell_weights_neg_mass_renorm(
    *,
    y: np.ndarray,
    cell_id: np.ndarray,
    base_w: Optional[np.ndarray],
    cfg: NegMassRenormCfg,
    tp_label,
    sl_label,
    to_label,
) -> np.ndarray:
    """
    Returns final per-sample weights w_final with negative mass renormalization.
    """
    n = len(y)
    if base_w is None:
        base_w = np.ones(n, dtype=np.float64)
    else:
        base_w = base_w.astype(np.float64, copy=False)

    w_final = base_w.copy()

    unique_cells = np.unique(cell_id)

    for c in unique_cells:
        idx = np.where(cell_id == c)[0]
        if idx.size == 0:
            continue

        y_c = y[idx]
        bw_c = base_w[idx]

        is_tp = (y_c == tp_label)
        is_sl = (y_c == sl_label)
        is_to = (y_c == to_label)

        # If a cell has no positives or no negatives, skip renorm
        n_pos = int(is_tp.sum())
        n_neg = int((is_sl | is_to).sum())
        if n_pos == 0 or n_neg == 0:
            continue

        # Timeout rate in THIS CELL among all labels
        Pto = float(is_to.sum()) / float(idx.size)
        Pto = min(max(Pto, cfg.eps), 1.0 - cfg.eps)

        # Raw intra-negative weights
        w_to_raw = _clip((1.0 - Pto) / Pto, cfg.w_to_min, cfg.w_to_max)
        w_sl_raw = cfg.w_sl_raw

        # Masses (include any existing base weights)
        M_pos = float(bw_c[is_tp].sum())
        M_neg_raw = float((bw_c[is_sl] * w_sl_raw).sum() + (bw_c[is_to] * w_to_raw).sum())

        if M_neg_raw <= cfg.eps:
            continue

        # Target negative mass to preserve pos:neg ratio
        rho = max(cfg.rho_pos_over_neg, cfg.eps)
        M_neg_target = M_pos / rho

        alpha = M_neg_target / M_neg_raw

        # Optional alpha clipping
        if cfg.alpha_min is not None:
            alpha = max(alpha, cfg.alpha_min)
        if cfg.alpha_max is not None:
            alpha = min(alpha, cfg.alpha_max)

        # Apply only to negatives; keep positives unchanged
        w_final[idx[is_sl]] = bw_c[is_sl] * (w_sl_raw * alpha)
        w_final[idx[is_to]] = bw_c[is_to] * (w_to_raw * alpha)
        # TP remain bw_c (already set via copy)

    return w_final
