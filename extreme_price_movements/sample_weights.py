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



def drawdown_aware_weights(dd: np.ndarray, k_dd: float = 5.0, k_early: float = 2.0, tau: float = 24.0) -> np.ndarray:
    """Upweight samples during drawdown and especially early in each episode."""
    dd = np.asarray(dd, dtype=float)
    if dd.size == 0:
        return np.array([], dtype=float)

    w = 1.0 + k_dd * np.clip(dd, 0.0, 1.0)

    starts = np.zeros_like(dd, dtype=float)
    starts[0] = 1.0 if dd[0] > 0 else 0.0
    if dd.size > 1:
        starts[1:] = ((dd[1:] > 0) & (dd[:-1] <= 0)).astype(float)

    # Vectorized early bonus computation using cumulative max for start positions
    # For each position, find the last drawdown start before it
    last_start = np.zeros_like(dd, dtype=np.int64)
    current_start = -1
    for t in range(dd.size):
        if starts[t] > 0:
            current_start = t
        last_start[t] = current_start
    
    # Vectorized bonus calculation
    bonus = np.zeros_like(dd, dtype=float)
    in_dd = dd > 0
    valid = (last_start >= 0) & in_dd
    if np.any(valid):
        t_indices = np.arange(dd.size)
        distances = t_indices - last_start
        distances = distances * valid.astype(np.int64)
        decay_tau = max(float(tau), 1e-6)
        bonus = np.exp(-distances / decay_tau) * valid

    return w * (1.0 + k_early * bonus)


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


def avg_uniqueness_on_grid_array(
    starts: np.ndarray, ends: np.ndarray, grid: np.ndarray, concurrent: np.ndarray
) -> np.ndarray:
    i0 = np.searchsorted(grid, starts, side="left")
    i1 = np.searchsorted(grid, ends, side="right") - 1

    i0 = np.clip(i0, 0, len(grid) - 1)
    i1 = np.clip(i1, 0, len(grid) - 1)

    inv = np.zeros_like(concurrent, dtype=np.float64)
    nz = concurrent > 0
    inv[nz] = 1.0 / concurrent[nz]
    prefix = np.concatenate(([0.0], np.cumsum(inv)))

    out = np.zeros(len(starts), dtype=np.float64)
    valid = (~np.isnat(starts)) & (~np.isnat(ends)) & (i0 <= i1)
    if np.any(valid):
        idx = np.where(valid)[0]
        sums = prefix[i1[idx] + 1] - prefix[i0[idx]]
        counts = (i1[idx] - i0[idx] + 1).astype(np.float64)
        out[idx] = sums / np.maximum(counts, 1.0)
    return out


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
    
    starts = pd.to_datetime(
        label_times["t_start"], utc=True, errors="coerce"
    ).values.astype("datetime64[ns]")
    ends = pd.to_datetime(
        label_times["t_end"], utc=True, errors="coerce"
    ).values.astype("datetime64[ns]")

    if time_grid is None:
        valid = (~np.isnat(starts)) & (~np.isnat(ends))
        all_times_ns = np.unique(np.concatenate([starts[valid], ends[valid]]))
        time_grid = pd.DatetimeIndex(all_times_ns).sort_values()
    else:
        if not time_grid.is_monotonic_increasing:
             time_grid = time_grid.sort_values()

    concurrent = concurrent_on_event_grid(label_times, time_grid)
    uniqueness = avg_uniqueness_on_grid_array(
        starts, ends, time_grid.values.astype("datetime64[ns]"), concurrent
    )

    return pd.Series(uniqueness, index=label_times.index)


def compute_avg_uniqueness_array(
    label_times: pd.DataFrame, time_grid: Optional[pd.DatetimeIndex] = None
) -> np.ndarray:
    if label_times.empty:
        return np.array([], dtype=float)
    return compute_avg_uniqueness(label_times, time_grid=time_grid).values


def _downsample_for_weight_opt(
    X_np: np.ndarray,
    qual_vals: np.ndarray,
    label_times: pd.DataFrame,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(X_np) <= max_samples:
        return X_np, qual_vals, label_times
    idx = np.linspace(0, len(X_np) - 1, max_samples, dtype=np.int32)
    return X_np[idx], qual_vals[idx], label_times.iloc[idx].reset_index(drop=True)


def _percentile_scores(metric_values: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(metric_values, kind="mergesort")
    percentile_ranks = np.empty_like(metric_values, dtype=np.float64)
    percentile_ranks[sorted_indices] = (
        np.arange(1, len(metric_values) + 1, dtype=np.float64) / max(len(metric_values), 1)
    )
    percentile_ranks = np.clip(percentile_ranks, 0.01, 0.99)
    raw_scores = 1.0 / (1.0 - percentile_ranks + 0.01)
    return 0.8 + 0.4 * (np.log1p(raw_scores - 1.0) / np.log1p(99.0))


def compute_sample_weights_with_uniqueness(
    label_times: pd.DataFrame,
    returns: Union[pd.Series, np.ndarray],
    base_weights: Union[pd.Series, np.ndarray, None] = None,
    time_grid: Optional[pd.DatetimeIndex] = None,
    selection_metric: Optional[Union[pd.Series, np.ndarray]] = None
) -> np.ndarray:
    """
    Compute sample weights combining uniqueness, event intensity, and MFE/MAE base weights.
    """
    uniqueness = compute_avg_uniqueness_array(label_times, time_grid=time_grid)

    if selection_metric is not None:
        if isinstance(selection_metric, pd.Series):
            if not selection_metric.index.equals(label_times.index):
                selection_metric = selection_metric.reindex(label_times.index).fillna(0.0)
            metric_values = np.abs(selection_metric.values)
        else:
            metric_values = np.abs(np.asarray(selection_metric))
        tprint(f"Using selection metric for event scoring: mean={metric_values.mean():.3f}, std={metric_values.std():.3f}")
    else:
        if isinstance(returns, pd.Series):
            if not returns.index.equals(label_times.index):
                returns = returns.reindex(label_times.index).fillna(0.0)
            returns_arr = returns.values
        else:
            returns_arr = np.asarray(returns)
        metric_values = np.abs(returns_arr)
        tprint("Using realized returns for event scoring (fallback)")

    event_scores = _percentile_scores(metric_values.astype(np.float64, copy=False))
    combined = uniqueness * event_scores

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

    sum_w = combined.sum()
    if sum_w < 1e-12:
        weights = np.ones(len(combined), dtype=np.float64)
    else:
        weights = combined * (len(combined) / sum_w)
    weights = np.clip(weights, 0.1, 10.0)
    return weights.astype(np.float32, copy=False)



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
    r_mfe = np.maximum(mfe, 0.0) / tp
    r_mae = np.maximum(mae, 0.0) / sl

    # Use a smooth intensity term plus an excursion-dominance term.
    # The previous hard clip on max(r_mfe, r_mae) saturated almost all rows at 1.0.
    tau_eff = max(float(tau), 1e-6)
    intensity = 1.0 - np.exp(-np.maximum(r_mfe, r_mae) / tau_eff)
    dominance = np.abs(r_mfe - r_mae) / (r_mfe + r_mae + 1e-9)
    quality = np.clip(0.65 * intensity + 0.35 * dominance, 0.0, 1.0)

    w_base = w_min + (1.0 - w_min) * quality
    
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


# =============================================================================
# OPTIMIZATION: Vectorized Single-Pass Weight Computation
# =============================================================================

def compute_all_weights_vectorized(
    n_samples: int,
    y_labels: np.ndarray,
    n_res: Optional[np.ndarray] = None,
    mfe: Optional[np.ndarray] = None,
    mae: Optional[np.ndarray] = None,
    tp: Optional[np.ndarray] = None,
    sl: Optional[np.ndarray] = None,
    barrier_pct: Optional[np.ndarray] = None,
    dd: Optional[np.ndarray] = None,
    selection_metric: Optional[np.ndarray] = None,
    w_min: float = 0.5,
    tau: float = 1.0,
    k_dd: float = 5.0,
    k_early: float = 2.0,
    dd_tau: float = 24.0,
) -> np.ndarray:
    """
    OPTIMIZATION: Compute all sample weights in a single vectorized pass.
    
    This combines:
    1. Uniqueness weights (via n_res resolution count)
    2. MFE/MAE quality weights
    3. Drawdown-aware weights
    4. Event intensity weights
    
    Args:
        n_samples: Number of samples
        y_labels: Label outcomes (2=TP, 1=TIMEOUT, 0=SL)
        n_res: Resolution count for uniqueness weighting
        mfe: Max Favorable Excursion
        mae: Max Adverse Excursion  
        tp: Take-profit barrier distance
        sl: Stop-loss barrier distance
        barrier_pct: Barrier percentage (ATR-based)
        dd: Drawdown values
        selection_metric: Metric for event intensity scoring
        w_min: Minimum MFE/MAE weight
        tau: MFE/MAE scaling factor
        k_dd: Drawdown weight multiplier
        k_early: Early drawdown bonus
        dd_tau: Drawdown decay tau
        
    Returns:
        Combined sample weights as float32 array
    """
    # Initialize with uniform weights
    w = np.ones(n_samples, dtype=np.float64)
    
    # 1. Uniqueness weights (via sqrt of resolution count)
    if n_res is not None:
        n_res = np.asarray(n_res, dtype=np.float64)
        # sqrt preserves relative ordering but compresses the tail
        w_uniqueness = np.sqrt(np.clip(n_res, 0.0, None))
        # Normalize to mean=1
        w_uniqueness = w_uniqueness / max(np.mean(w_uniqueness), 1e-12)
        w *= w_uniqueness
    
    # 2. MFE/MAE quality weights
    if mfe is not None and mae is not None and tp is not None and sl is not None:
        mfe = np.asarray(mfe, dtype=np.float64)
        mae = np.asarray(mae, dtype=np.float64)
        tp = np.asarray(tp, dtype=np.float64)
        sl = np.asarray(sl, dtype=np.float64)
        
        # Ensure positive barriers
        tp = np.maximum(tp, 1e-8)
        sl = np.maximum(sl, 1e-8)
        
        w_mfe_mae = compute_mfe_mae_weights(
            mfe=mfe,
            mae=mae,
            tp=tp,
            sl=sl,
            is_timeout=(np.asarray(y_labels, dtype=np.int8) == 1),
            touch_margin=None,
            w_min=w_min,
            tau=tau,
            cost_floor=0.0,
        ).astype(np.float64)

        is_timeout = np.asarray(y_labels, dtype=np.int8) == 1
        # Normalize to mean=1
        w_mfe_mae = w_mfe_mae / max(np.mean(w_mfe_mae), 1e-12)
        w *= w_mfe_mae
    
    # 3. Drawdown-aware weights
    if dd is not None:
        dd = np.asarray(dd, dtype=np.float64)
        
        # Base drawdown weight
        w_dd = 1.0 + k_dd * np.clip(dd, 0.0, 1.0)
        
        # Early drawdown bonus (vectorized approximation)
        # Find drawdown episode starts
        starts = np.zeros_like(dd, dtype=np.float64)
        starts[0] = 1.0 if dd[0] > 0 else 0.0
        if dd.size > 1:
            starts[1:] = ((dd[1:] > 0) & (dd[:-1] <= 0)).astype(np.float64)
        
        # Compute decay from last start (vectorized via cumulative max with decay)
        # This is an approximation - uses distance from last start
        episode_idx = np.cumsum(starts) - 1  # Episode index for each point
        episode_start_pos = np.maximum.accumulate(np.where(starts > 0, np.arange(len(dd)), 0))
        distance_from_start = np.arange(len(dd)) - episode_start_pos
        
        # Decay bonus
        decay_tau = max(float(dd_tau), 1e-6)
        bonus = np.where(dd > 0, np.exp(-distance_from_start / decay_tau), 0.0)
        
        w_dd = w_dd * (1.0 + k_early * bonus)
        
        # Normalize to mean=1
        w_dd = w_dd / max(np.mean(w_dd), 1e-12)
        w *= w_dd
    
    # 4. Event intensity weights (based on selection metric or barrier_pct)
    if selection_metric is not None:
        metric_values = np.abs(np.asarray(selection_metric, dtype=np.float64))
    elif barrier_pct is not None:
        metric_values = np.abs(np.asarray(barrier_pct, dtype=np.float64))
    else:
        metric_values = None
    
    if metric_values is not None and len(metric_values) == n_samples:
        # Vectorized percentile calculation
        sorted_indices = np.argsort(metric_values)
        percentile_ranks = np.empty_like(metric_values)
        percentile_ranks[sorted_indices] = np.arange(1, len(metric_values) + 1) / len(metric_values)
        
        # Clamp percentiles to [0.01, 0.99]
        percentile_ranks = np.clip(percentile_ranks, 0.01, 0.99)
        
        # Score = 1 / (1 - percentile), normalized to [0.8, 1.2]
        raw_scores = 1.0 / (1.0 - percentile_ranks + 0.01)
        event_scores = 0.8 + 0.4 * (np.log1p(raw_scores - 1) / np.log1p(99))
        
        # Normalize to mean=1
        event_scores = event_scores / max(np.mean(event_scores), 1e-12)
        w *= event_scores
    
    # Final normalization to mean=1
    w = w / max(np.mean(w), 1e-12)
    
    # Clip extremes
    w = np.clip(w, 0.1, 10.0)
    
    return w.astype(np.float32)
