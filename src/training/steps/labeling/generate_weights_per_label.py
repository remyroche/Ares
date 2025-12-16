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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)

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

    # 2. Magnitude component (absolute returns, clipped and compressed)
    abs_ret = np.abs(returns_clean)

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
                abs_ret = abs_ret / (v + 1e-12)
        except Exception:
            pass

    # Determine clipping percentile
    if mag_clip_pct is not None and 0.0 < mag_clip_pct < 1.0:
        clip_pct = float(mag_clip_pct)
    else:
        clip_pct = 0.99

    if abs_ret.size:
        try:
            clip_val = float(np.quantile(abs_ret, clip_pct))
        except Exception:
            clip_val = float(np.max(abs_ret)) if np.isfinite(abs_ret).any() else 0.0
    else:
        clip_val = 0.0

    if clip_val > 1e-9:
        norm_mag = np.clip(abs_ret, 0.0, clip_val) / clip_val
    else:
        norm_mag = np.zeros_like(abs_ret)

    comp_mag = np.power(norm_mag, mag_compression)

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
        final_weights = raw_weights * (n_samples / raw_weights.sum())
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


def safe_layer1_objective(
    weights: np.ndarray,
    returns: np.ndarray,
    concurrency: np.ndarray,
    volatility: np.ndarray,
    noise_threshold: float = 0.001,
) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return -10.0
    if not np.isfinite(w).all():
        return -10.0

    total = float(w.sum())
    if total <= 0:
        return -10.0

    w_norm = w / total

    r = np.asarray(returns, dtype=float)
    if r.size != w_norm.size:
        return -10.0

    n = w_norm.size
    if n < 2:
        return -10.0

    abs_returns = np.abs(r)

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2 or y.size < 2:
            return 0.0
        if not np.isfinite(x).any() or not np.isfinite(y).any():
            return 0.0
        try:
            if spearmanr is not None:
                corr, _ = spearmanr(x, y)
                if corr is None or not np.isfinite(corr):
                    return 0.0
                return float(corr)
        except Exception:
            pass

        try:
            x_center = x - np.nanmean(x)
            y_center = y - np.nanmean(y)
            denom = (
                np.sqrt(np.nanmean(x_center ** 2))
                * np.sqrt(np.nanmean(y_center ** 2))
            )
            if denom <= 0:
                return 0.0
            return float(np.nanmean(x_center * y_center) / denom)
        except Exception:
            return 0.0

    mas = _safe_corr(w, abs_returns)
    mas = max(0.0, mas)

    try:
        if SCIPY_AVAILABLE and scipy_entropy is not None:
            wes = float(scipy_entropy(w_norm) / np.log(float(n)))
        else:
            ent = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
            wes = ent / np.log(float(n)) if n > 1 else 0.0
    except Exception:
        wes = 0.0

    noise_mask = abs_returns < noise_threshold
    nwp = float(w_norm[noise_mask].sum()) if noise_mask.any() else 0.0

    concurrency_arr = np.asarray(concurrency, dtype=float)
    uop_corr = _safe_corr(w, concurrency_arr)
    uop_penalty = max(0.0, uop_corr)

    vol_arr = np.asarray(volatility, dtype=float)
    vdp_corr = _safe_corr(w, vol_arr)
    vdp_penalty = max(0.0, vdp_corr - 0.6)

    score = (
        1.0 * mas
        + 1.5 * wes
        - 2.0 * nwp
        - 1.0 * uop_penalty
        - 1.0 * vdp_penalty
    )

    if not np.isfinite(score):
        return -10.0
    return float(score)


def run_layer1_optimization(
    symbol: str, 
    timeframe: str,
    market_data: pd.DataFrame,
    labels: pd.Series,
    committee_agreement_scores: Optional[Union[np.ndarray, pd.Series, list]] = None,
    committee_mag_factors: Optional[Union[np.ndarray, pd.Series, list]] = None,
    n_trials: int = 60,
    objective_mode: str = "proxy",
) -> Dict[str, Any]:
    """
    Run lightweight optimization for weighting parameters (Layer 1).
    placeholder for full HPO logic.
    
    Returns:
        Dictionary of best parameters.
    """
    default_params: Dict[str, Any] = {
        'mag_compression': 0.8,
        'learn_slope': 0.0,
        'learn_center': 0.5,
        'uniq_intensity': 1.0,
        'quality_intensity': 0.0,
        'quality_floor': 0.2,
        'exp_mag': 1.0,
        'exp_learn': 1.0,
        'exp_uniq': 1.0,
        'exp_cross': 1.0,
        'downside_multiplier': 1.0,
        'committee_agreement_alpha': 0.5,
        'committee_mag_clip': 5.0,
    }

    tprint("⚙️ Running Layer 1 Weight Optimization (Placeholder)...", "INFO")
    tprint_info(
        f"⚙️ Running Layer 1 Weight Optimization for {symbol} {timeframe}...",
    )

    try:
        returns_series = labels.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns_series) < 50:
            tprint_warning(
                f"⚠️ Layer 1: insufficient events for optimization (n={len(returns_series)}). Using defaults.",
            )
            return default_params

        if 'close' not in market_data.columns:
            tprint_warning("⚠️ Layer 1: market_data missing 'close' column. Using defaults.")
            return default_params

        close_series = market_data['close'].astype(float)

        # Simple volatility proxy: rolling standard deviation of returns
        close_ret = close_series.pct_change()
        vol_series = close_ret.rolling(20).std().replace([np.inf, -np.inf], np.nan)

        t_events = returns_series.index
        vol_proxy = vol_series.reindex(t_events).astype(float).values
        if not np.isfinite(vol_proxy).any():
            vol_proxy = None

        returns_arr = returns_series.values.astype(float)

        n_samples = int(len(returns_arr))

        # Confident-learning style per-event label quality (out-of-sample probabilities)
        # This is used as an additional multiplicative component inside generate_weights_per_label.
        cl_quality_scores = None
        try:
            from src.training.steps.labeling.confident_learning import (
                get_cross_val_pred_probs,
                compute_label_quality_scores,
            )

            y_cl = (np.asarray(returns_arr, dtype=float) > 0.0).astype(int)
            if int(np.unique(y_cl).size) >= 2:
                # Simple, leakage-safe features from market_data available at t_events.
                close_ret_1 = close_series.pct_change(1)
                close_ret_3 = close_series.pct_change(3)
                close_ret_6 = close_series.pct_change(6)
                vol_20 = close_series.pct_change().rolling(20).std()
                feat_df = pd.DataFrame(
                    {
                        "ret1": close_ret_1.reindex(t_events),
                        "ret3": close_ret_3.reindex(t_events),
                        "ret6": close_ret_6.reindex(t_events),
                        "vol20": vol_20.reindex(t_events),
                    },
                    index=t_events,
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                pred_probs = get_cross_val_pred_probs(
                    feat_df.values,
                    y_cl,
                    model=None,
                    n_splits=3,
                    random_state=42,
                )
                cl_quality_scores = compute_label_quality_scores(y_cl, pred_probs, method="self_confidence")
                cl_quality_scores = np.asarray(cl_quality_scores, dtype=float)
                cl_quality_scores = np.where(np.isfinite(cl_quality_scores), cl_quality_scores, 1.0)
                cl_quality_scores = np.clip(cl_quality_scores, 0.0, 1.0)
        except Exception:
            cl_quality_scores = None
 
        # Heuristic floor for "small" returns (used in objective)
        finite_abs = np.abs(returns_arr[np.isfinite(returns_arr)])
        if finite_abs.size:
            small_ret_thr = float(np.quantile(finite_abs, 0.25))
        else:
            small_ret_thr = 0.0

        # Build per-event volatility for the objective
        if vol_proxy is None:
            event_volatility = np.zeros_like(returns_arr)
        else:
            event_volatility = np.asarray(vol_proxy, dtype=float)
            if event_volatility.shape[0] != returns_arr.shape[0]:
                event_volatility = np.resize(event_volatility, returns_arr.shape[0])
            non_finite_vol = ~np.isfinite(event_volatility)
            if non_finite_vol.all():
                event_volatility = np.zeros_like(returns_arr)
            else:
                median_vol = float(
                    np.nanmedian(event_volatility[~non_finite_vol])
                )
                event_volatility = np.where(
                    non_finite_vol, median_vol, event_volatility
                )

        # Approximate per-event concurrency using local event density
        horizon_bars = 12
        idx = market_data.index
        if len(idx) >= 2:
            try:
                bar_deltas = idx.to_series().diff().dropna()
                bar_delta = bar_deltas.median()
            except Exception:
                bar_delta = None
        else:
            bar_delta = None

        if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
            bar_delta = pd.Timedelta(minutes=15)

        window_span = horizon_bars * bar_delta
        t_events_arr = t_events.to_numpy()
        event_concurrency = np.ones(len(t_events_arr), dtype=float)
        try:
            window = window_span.to_timedelta64()
            order = np.argsort(t_events_arr)
            t_sorted = t_events_arr[order]
            left_bounds = t_sorted - window
            right_bounds = t_sorted + window
            left_idx = np.searchsorted(t_sorted, left_bounds, side='left')
            right_idx = np.searchsorted(t_sorted, right_bounds, side='right')
            concurrency_sorted = (right_idx - left_idx).astype(float)
            event_concurrency[order] = concurrency_sorted
        except Exception:
            event_concurrency = np.zeros(len(t_events_arr), dtype=float)
            for i, ts in enumerate(t_events_arr):
                left_ts = ts - window_span
                right_ts = ts + window_span
                mask = (t_events_arr >= left_ts) & (t_events_arr <= right_ts)
                event_concurrency[i] = float(mask.sum())

        # Convert concurrency into a simple uniqueness proxy (1 / concurrency)
        event_uniqueness = 1.0 / np.maximum(1.0, event_concurrency)

        try:
            objective_mode_local = str(objective_mode or "proxy").strip().lower()
        except Exception:
            objective_mode_local = "proxy"

        def _predictive_cv_score(
            weights: np.ndarray,
            returns_arr_local: np.ndarray,
            t_events_local: pd.Index,
            close_series_local: pd.Series,
            n_splits: int = 3,
        ) -> float:
            try:
                y = (np.asarray(returns_arr_local, dtype=float) > 0.0).astype(int)
                if int(np.unique(y).size) < 2:
                    return -10.0

                close = close_series_local.astype(float)
                ret1 = close.pct_change(1).reindex(t_events_local)
                ret3 = close.pct_change(3).reindex(t_events_local)
                ret6 = close.pct_change(6).reindex(t_events_local)
                vol20 = close.pct_change().rolling(20).std().reindex(t_events_local)
                X_df = (
                    pd.DataFrame(
                        {"ret1": ret1, "ret3": ret3, "ret6": ret6, "vol20": vol20},
                        index=t_events_local,
                    )
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                X = X_df.values.astype(float)

                w = np.asarray(weights, dtype=float)
                w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
                if float(np.sum(w)) <= 0.0:
                    w = np.ones_like(w, dtype=float)

                n_samples_local = int(len(y))
                if n_samples_local < 80:
                    n_splits = 2
                n_splits = int(max(2, min(int(n_splits), 5)))
                if n_samples_local < (n_splits + 1) * 10:
                    return -10.0

                tscv = TimeSeriesSplit(n_splits=n_splits)
                fold_aucs: List[float] = []
                fold_prs: List[float] = []

                for tr_idx, te_idx in tscv.split(X):
                    y_tr = y[tr_idx]
                    y_te = y[te_idx]
                    if int(np.unique(y_tr).size) < 2 or int(np.unique(y_te).size) < 2:
                        continue

                    model = LogisticRegression(
                        solver="lbfgs",
                        max_iter=500,
                        n_jobs=1,
                    )
                    model.fit(X[tr_idx], y_tr, sample_weight=w[tr_idx])
                    p_te = model.predict_proba(X[te_idx])[:, 1]

                    sw_te = w[te_idx]
                    try:
                        auc = float(roc_auc_score(y_te, p_te, sample_weight=sw_te))
                    except Exception:
                        auc = float(roc_auc_score(y_te, p_te))

                    try:
                        pr = float(average_precision_score(y_te, p_te, sample_weight=sw_te))
                    except Exception:
                        pr = float(average_precision_score(y_te, p_te))

                    if np.isfinite(auc):
                        fold_aucs.append(auc)
                    if np.isfinite(pr):
                        fold_prs.append(pr)

                if len(fold_aucs) < 1:
                    return -10.0

                mean_auc = float(np.mean(fold_aucs))
                mean_pr = float(np.mean(fold_prs)) if len(fold_prs) else 0.0
                score = mean_auc + 0.50 * mean_pr
                if not np.isfinite(score):
                    return -10.0
                return float(score)
            except Exception:
                return -10.0

        committee_components_available = False
        committee_agree_arr = _coerce_numeric_array(
            committee_agreement_scores,
            n_samples,
            "committee_agreement_scores",
            fill_value=0.0,
            allow_negative=False,
        )
        committee_mag_arr = _coerce_numeric_array(
            committee_mag_factors,
            n_samples,
            "committee_mag_factors",
            fill_value=1.0,
            allow_negative=False,
        )

        if committee_agree_arr is not None or committee_mag_arr is not None:
            committee_components_available = True
            if committee_agree_arr is None:
                committee_agree_arr = np.zeros(n_samples, dtype=float)
            if committee_mag_arr is None:
                committee_mag_arr = np.ones(n_samples, dtype=float)
            committee_agree_arr = np.where(np.isfinite(committee_agree_arr), committee_agree_arr, 0.0)
            committee_agree_arr = np.clip(committee_agree_arr, 0.0, 1.0)
            committee_mag_arr = np.where(np.isfinite(committee_mag_arr) & (committee_mag_arr > 0.0), committee_mag_arr, 1.0)

        if committee_components_available and committee_agree_arr is not None and committee_mag_arr is not None:
            try:
                def _safe_corr(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
                    x_arr = np.asarray(x_arr, dtype=float)
                    y_arr = np.asarray(y_arr, dtype=float)
                    if x_arr.size < 2 or y_arr.size < 2:
                        return 0.0
                    if not np.isfinite(x_arr).any() or not np.isfinite(y_arr).any():
                        return 0.0
                    try:
                        if spearmanr is not None:
                            corr, _ = spearmanr(x_arr, y_arr)
                            if corr is None or not np.isfinite(corr):
                                return 0.0
                            return float(corr)
                    except Exception:
                        pass

                    try:
                        x_center = x_arr - np.nanmean(x_arr)
                        y_center = y_arr - np.nanmean(y_arr)
                        denom = (
                            np.sqrt(np.nanmean(x_center ** 2))
                            * np.sqrt(np.nanmean(y_center ** 2))
                        )
                        if denom <= 0:
                            return 0.0
                        return float(np.nanmean(x_center * y_center) / denom)
                    except Exception:
                        return 0.0

                agree_v = np.asarray(committee_agree_arr, dtype=float)
                mag_v = np.asarray(committee_mag_arr, dtype=float)
                abs_ret_v = np.abs(np.asarray(returns_arr, dtype=float))

                agree_std = float(np.nanstd(agree_v)) if np.isfinite(agree_v).any() else 0.0
                tprint_info(
                    "   [Layer1 committee] agreement stats: "
                    f"mean={float(np.nanmean(agree_v)):.4f}, std={agree_std:.4f}, "
                    f"min={float(np.nanmin(agree_v)):.4f}, max={float(np.nanmax(agree_v)):.4f}"
                )
                tprint_info(
                    "   [Layer1 committee] magnitude stats: "
                    f"mean={float(np.nanmean(mag_v)):.4f}, std={float(np.nanstd(mag_v)):.4f}, "
                    f"min={float(np.nanmin(mag_v)):.4f}, max={float(np.nanmax(mag_v)):.4f}"
                )
                tprint_info(
                    "   [Layer1 committee] correlations (Spearman if available): "
                    f"corr(agree,|ret|)={_safe_corr(agree_v, abs_ret_v):.4f}, "
                    f"corr(mag,|ret|)={_safe_corr(mag_v, abs_ret_v):.4f}, "
                    f"corr(agree,concurrency)={_safe_corr(agree_v, event_concurrency):.4f}, "
                    f"corr(agree,vol)={_safe_corr(agree_v, event_volatility):.4f}"
                )
                if agree_std < 1e-3:
                    tprint_warning(
                        "   ⚠️ Layer1 committee agreement is nearly constant; committee_agreement_alpha may be weakly identified and can optimize to ~0."
                    )
            except Exception:
                pass

        search_space: Dict[str, Dict[str, Any]] = {
            # --- A. INFORMATION HANDLING (Magnitude & Uniqueness) ---
            # How much do we reward high returns?
            # 0.5 = Sqrt (Conservative), 1.0 = Linear, 1.5 = Convex
            'mag_compression': {
                'type': 'float',
                'low': 0.90,
                'high': 1.20,
                # step used only by grid utilities; TPE ignores it but it's harmless
                'step': 0.05,
                'log': False,
            },
            # How strictly do we punish concurrent/overlapping events?
            'uniq_intensity': {
                'type': 'float',
                'low': 1.00,
                'high': 3.00,
                'log': False,
            },

            # Confident-learning quality weight (optional; intensity 0 disables)
            'quality_intensity': {
                'type': 'float',
                'low': 0.0,
                'high': 3.0,
                'log': False,
            },
            'quality_floor': {
                'type': 'float',
                'low': 0.05,
                'high': 0.60,
                'log': False,
            },

            # --- C. COMPONENT MIXING (Exponents / Power Law) ---
            'exp_mag': {
                'type': 'float',
                'low': 1.0,
                'high': 1.5,
                'log': True,
            },
            'exp_uniq': {
                'type': 'float',
                'low': 1.0,
                'high': 1.5,
                'log': True,
            },

            # --- D. ASYMMETRY (Risk Management) ---
            'downside_multiplier': {
                'type': 'float',
                'low': 1.0,
                'high': 1.4,
                'log': False,
            },

            # --- E. NOISE CLIPPING ---
            'mag_clip_pct': {
                'type': 'float',
                'low': 0.95,
                'high': 0.99,
                'log': False,
            },
        }

        if committee_components_available:
            search_space.update(
                {
                    'committee_agreement_alpha': {
                        'type': 'float',
                        'low': 0.0,
                        'high': 2.0,
                        'log': False,
                    },
                    'committee_mag_clip': {
                        'type': 'float',
                        'low': 1.0,
                        'high': 10.0,
                        'log': False,
                    },
                }
            )

        def objective(params: Dict[str, Any]) -> float:
            try:
                weights = generate_weights_per_label(
                    returns=returns_arr,
                    t_events=t_events,
                    close_series=close_series,
                    consistency_scores=None,
                    label_quality_scores=cl_quality_scores,
                    uniqueness_scores=event_uniqueness,
                    vol_proxy=vol_proxy,
                    mag_compression=float(params.get('mag_compression', default_params['mag_compression'])),
                    learn_slope=float(default_params.get('learn_slope', 0.0)),
                    learn_center=float(default_params.get('learn_center', 0.5)),
                    uniq_intensity=float(params.get('uniq_intensity', default_params['uniq_intensity'])),
                    quality_intensity=float(params.get('quality_intensity', default_params['quality_intensity'])),
                    quality_floor=float(params.get('quality_floor', default_params['quality_floor'])),
                    exp_mag=float(params.get('exp_mag', default_params['exp_mag'])),
                    exp_learn=float(default_params.get('exp_learn', 1.0)),
                    exp_uniq=float(params.get('exp_uniq', default_params['exp_uniq'])),
                    exp_cross=float(params.get('exp_cross', default_params['exp_cross'])),
                    downside_multiplier=float(params.get('downside_multiplier', default_params['downside_multiplier'])),
                    mag_clip_pct=float(params.get('mag_clip_pct', 0.99)),
                )
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    return -10.0

                if committee_components_available and committee_agree_arr is not None and committee_mag_arr is not None:
                    try:
                        alpha = float(
                            params.get(
                                'committee_agreement_alpha',
                                default_params.get('committee_agreement_alpha', 0.5),
                            )
                        )
                    except Exception:
                        alpha = float(default_params.get('committee_agreement_alpha', 0.5))

                    try:
                        mag_clip = float(
                            params.get(
                                'committee_mag_clip',
                                default_params.get('committee_mag_clip', 5.0),
                            )
                        )
                    except Exception:
                        mag_clip = float(default_params.get('committee_mag_clip', 5.0))

                    alpha = float(np.clip(alpha, 0.0, 10.0))
                    mag_clip = float(np.clip(mag_clip, 0.5, 50.0))

                    cf = (1.0 + alpha * committee_agree_arr) * np.clip(committee_mag_arr, 0.0, mag_clip)
                    cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                    cf_mean = float(np.mean(cf)) if cf.size else 1.0
                    if np.isfinite(cf_mean) and cf_mean > 0:
                        cf = cf / cf_mean
                    else:
                        cf = np.ones_like(cf, dtype=float)

                    weights = np.asarray(weights, dtype=float) * cf
                    w_sum = float(np.sum(weights)) if weights.size else 0.0
                    if np.isfinite(w_sum) and w_sum > 0:
                        weights = weights * (len(weights) / w_sum)
                    else:
                        weights = np.ones(len(returns_arr), dtype=float)

                if objective_mode_local == "predictive_cv":
                    score = _predictive_cv_score(
                        weights=np.asarray(weights, dtype=float),
                        returns_arr_local=returns_arr,
                        t_events_local=t_events,
                        close_series_local=close_series,
                        n_splits=3,
                    )
                else:
                    score = safe_layer1_objective(
                        weights=weights,
                        returns=returns_arr,
                        concurrency=event_concurrency,
                        volatility=event_volatility,
                        noise_threshold=float(small_ret_thr) if small_ret_thr > 0 else 0.001,
                    )
                return float(score)
            except Exception as e:
                tprint_warning(f"⚠️ Layer 1 objective failure: {e}")
                return -10.0

        try:
            n_trials_i = int(n_trials)
        except Exception:
            n_trials_i = 60
        n_trials_i = max(5, min(n_trials_i, 250))

        opt_config = OptimizationConfig(
            n_trials=n_trials_i,
            execution_mode="light",
            direction="maximize",
            seed=42,
            enable_staged_optimization=False,
            coarse_grid_trials=0,
            fine_grid_trials=0,
            tpe_trials=n_trials_i,
        )

        optimizer = BayesianTPEOptimizer(config=opt_config)
        result = optimizer.optimize(objective=objective, search_space=search_space)

        best_params_raw = result.get('best_params') or {}
        best_value = result.get('best_value')

        best_params: Dict[str, Any] = default_params.copy()
        for key in default_params.keys():
            if key in best_params_raw:
                try:
                    best_params[key] = float(best_params_raw[key])
                except Exception:
                    continue

        if not committee_components_available:
            best_params.pop('committee_agreement_alpha', None)
            best_params.pop('committee_mag_clip', None)

        if 'mag_clip_pct' in best_params_raw:
            try:
                best_params['mag_clip_pct'] = float(best_params_raw['mag_clip_pct'])
            except Exception:
                pass

        if best_value is not None and np.isfinite(best_value):
            tprint_success(
                f"✅ Layer 1 optimization complete. Best score={best_value:.4f}",
            )
        else:
            tprint_success("✅ Layer 1 optimization complete.")

        tprint_info(f"   Best weighting params: {best_params}")

        # Persist Layer 1 trial metrics for correlation analysis
        try:
            def _compute_l1_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    weights = generate_weights_per_label(
                        returns=returns_arr,
                        t_events=t_events,
                        close_series=close_series,
                        consistency_scores=None,
                        label_quality_scores=cl_quality_scores,
                        uniqueness_scores=event_uniqueness,
                        vol_proxy=vol_proxy,
                        mag_compression=float(params.get('mag_compression', default_params['mag_compression'])),
                        learn_slope=float(params.get('learn_slope', default_params['learn_slope'])),
                        learn_center=float(params.get('learn_center', default_params['learn_center'])),
                        uniq_intensity=float(params.get('uniq_intensity', default_params['uniq_intensity'])),
                        quality_intensity=float(params.get('quality_intensity', default_params['quality_intensity'])),
                        quality_floor=float(params.get('quality_floor', default_params['quality_floor'])),
                        exp_mag=float(params.get('exp_mag', default_params['exp_mag'])),
                        exp_learn=float(params.get('exp_learn', default_params['exp_learn'])),
                        exp_uniq=float(params.get('exp_uniq', default_params['exp_uniq'])),
                        exp_cross=float(params.get('exp_cross', default_params['exp_cross'])),
                        downside_multiplier=float(params.get('downside_multiplier', default_params['downside_multiplier'])),
                        mag_clip_pct=float(params.get('mag_clip_pct', 0.99)),
                    )
                    if not np.isfinite(weights).all() or weights.sum() <= 0:
                        return {
                            "score": -10.0,
                            "weights_mean": np.nan,
                            "weights_min": np.nan,
                            "weights_max": np.nan,
                            "weights_entropy": np.nan,
                            "weights_entropy_norm": np.nan,
                            "mas": np.nan,
                            "wes": np.nan,
                            "nwp": np.nan,
                            "uop_penalty": np.nan,
                            "vdp_penalty": np.nan,
                            "committee_alpha": np.nan,
                            "committee_mag_clip": np.nan,
                            "committee_factor_mean": np.nan,
                            "committee_factor_min": np.nan,
                            "committee_factor_max": np.nan,
                            "n_events": int(len(returns_arr)),
                        }
                except Exception:
                    return {
                        "score": -10.0,
                        "weights_mean": np.nan,
                        "weights_min": np.nan,
                        "weights_max": np.nan,
                        "weights_entropy": np.nan,
                        "weights_entropy_norm": np.nan,
                        "mas": np.nan,
                        "wes": np.nan,
                        "nwp": np.nan,
                        "uop_penalty": np.nan,
                        "vdp_penalty": np.nan,
                        "committee_alpha": np.nan,
                        "committee_mag_clip": np.nan,
                        "committee_factor_mean": np.nan,
                        "committee_factor_min": np.nan,
                        "committee_factor_max": np.nan,
                        "n_events": int(len(returns_arr)),
                    }

                committee_alpha_val = np.nan
                committee_mag_clip_val = np.nan
                committee_factor_mean = np.nan
                committee_factor_min = np.nan
                committee_factor_max = np.nan

                if committee_components_available and committee_agree_arr is not None and committee_mag_arr is not None:
                    try:
                        committee_alpha_val = float(
                            params.get(
                                'committee_agreement_alpha',
                                default_params.get('committee_agreement_alpha', 0.5),
                            )
                        )
                    except Exception:
                        committee_alpha_val = float(default_params.get('committee_agreement_alpha', 0.5))

                    try:
                        committee_mag_clip_val = float(
                            params.get(
                                'committee_mag_clip',
                                default_params.get('committee_mag_clip', 5.0),
                            )
                        )
                    except Exception:
                        committee_mag_clip_val = float(default_params.get('committee_mag_clip', 5.0))

                    committee_alpha_val = float(np.clip(committee_alpha_val, 0.0, 10.0))
                    committee_mag_clip_val = float(np.clip(committee_mag_clip_val, 0.5, 50.0))

                    cf = (1.0 + committee_alpha_val * committee_agree_arr) * np.clip(
                        committee_mag_arr, 0.0, committee_mag_clip_val
                    )
                    cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                    cf_mean = float(np.mean(cf)) if cf.size else 1.0
                    if np.isfinite(cf_mean) and cf_mean > 0:
                        cf = cf / cf_mean
                    else:
                        cf = np.ones_like(cf, dtype=float)

                    try:
                        committee_factor_mean = float(np.mean(cf)) if cf.size else np.nan
                        committee_factor_min = float(np.min(cf)) if cf.size else np.nan
                        committee_factor_max = float(np.max(cf)) if cf.size else np.nan
                    except Exception:
                        committee_factor_mean = np.nan
                        committee_factor_min = np.nan
                        committee_factor_max = np.nan

                    weights = np.asarray(weights, dtype=float) * cf
                    w_sum = float(np.sum(weights)) if weights.size else 0.0
                    if np.isfinite(w_sum) and w_sum > 0:
                        weights = weights * (len(weights) / w_sum)
                    else:
                        weights = np.ones(len(returns_arr), dtype=float)

                # Recompute objective components for transparency
                w = np.asarray(weights, dtype=float)
                total = float(w.sum())
                w_norm = w / total if total > 0 else w

                r = np.asarray(returns_arr, dtype=float)
                abs_returns = np.abs(r)

                def _safe_corr(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
                    x_arr = np.asarray(x_arr, dtype=float)
                    y_arr = np.asarray(y_arr, dtype=float)
                    if x_arr.size < 2 or y_arr.size < 2:
                        return 0.0
                    if not np.isfinite(x_arr).any() or not np.isfinite(y_arr).any():
                        return 0.0
                    try:
                        if spearmanr is not None:
                            corr, _ = spearmanr(x_arr, y_arr)
                            if corr is None or not np.isfinite(corr):
                                return 0.0
                            return float(corr)
                    except Exception:
                        pass
                    try:
                        x_center = x_arr - np.nanmean(x_arr)
                        y_center = y_arr - np.nanmean(y_arr)
                        denom = (
                            np.sqrt(np.nanmean(x_center ** 2))
                            * np.sqrt(np.nanmean(y_center ** 2))
                        )
                        if denom <= 0:
                            return 0.0
                        return float(np.nanmean(x_center * y_center) / denom)
                    except Exception:
                        return 0.0

                mas = max(0.0, _safe_corr(w, abs_returns))

                try:
                    if SCIPY_AVAILABLE and scipy_entropy is not None:
                        wes = float(scipy_entropy(w_norm) / np.log(float(len(w_norm))))
                    else:
                        ent = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
                        wes = ent / np.log(float(len(w_norm))) if len(w_norm) > 1 else 0.0
                except Exception:
                    wes = 0.0

                noise_mask = abs_returns < (float(small_ret_thr) if small_ret_thr > 0 else 0.001)
                nwp = float(w_norm[noise_mask].sum()) if noise_mask.any() else 0.0

                concurrency_arr = np.asarray(event_concurrency, dtype=float)
                uop_penalty = max(0.0, _safe_corr(w, concurrency_arr))

                vol_arr = np.asarray(event_volatility, dtype=float)
                vdp_penalty = max(0.0, _safe_corr(w, vol_arr) - 0.6)

                score = (
                    1.0 * mas
                    + 1.5 * wes
                    - 2.0 * nwp
                    - 1.0 * uop_penalty
                    - 1.0 * vdp_penalty
                )

                w_valid = weights[np.isfinite(weights)]
                weights_mean = float(w_valid.mean()) if w_valid.size else np.nan
                weights_min = float(w_valid.min()) if w_valid.size else np.nan
                weights_max = float(w_valid.max()) if w_valid.size else np.nan
                weights_entropy = np.nan
                weights_entropy_norm = np.nan
                if w_valid.size > 1:
                    w_sum = float(w_valid.sum())
                    if w_sum > 0:
                        w_norm = w_valid / w_sum
                        entropy_val = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
                        max_entropy = np.log(float(len(w_norm)))
                        weights_entropy = entropy_val
                        weights_entropy_norm = float(entropy_val / max_entropy) if max_entropy > 0 else np.nan

                return {
                    "score": float(score),
                    "weights_mean": weights_mean,
                    "weights_min": weights_min,
                    "weights_max": weights_max,
                    "weights_entropy": weights_entropy,
                    "weights_entropy_norm": weights_entropy_norm,
                    "mas": float(mas),
                    "wes": float(wes),
                    "nwp": float(nwp),
                    "uop_penalty": float(uop_penalty),
                    "vdp_penalty": float(vdp_penalty),
                    "committee_alpha": committee_alpha_val,
                    "committee_mag_clip": committee_mag_clip_val,
                    "committee_factor_mean": committee_factor_mean,
                    "committee_factor_min": committee_factor_min,
                    "committee_factor_max": committee_factor_max,
                    "n_events": int(len(returns_arr)),
                }

            # Log best-score decomposition for the final best_params
            try:
                best_metrics = _compute_l1_metrics(best_params)
                tprint_info(
                    "   Layer 1 best score components: "
                    f"score={best_metrics.get('score', float('nan')):.4f}, "
                    f"MAS={best_metrics.get('mas', float('nan')):.4f}, "
                    f"WES={best_metrics.get('wes', float('nan')):.4f}, "
                    f"NWP={best_metrics.get('nwp', float('nan')):.4f}, "
                    f"UOP_penalty={best_metrics.get('uop_penalty', float('nan')):.4f}, "
                    f"VDP_penalty={best_metrics.get('vdp_penalty', float('nan')):.4f}, "
                    f"committee_alpha={best_metrics.get('committee_alpha', float('nan'))}, "
                    f"committee_mag_clip={best_metrics.get('committee_mag_clip', float('nan'))}, "
                    f"committee_factor_min={best_metrics.get('committee_factor_min', float('nan'))}, "
                    f"committee_factor_max={best_metrics.get('committee_factor_max', float('nan'))}"
                )
            except Exception:
                pass

            trial_rows = []
            for trial in result.get("history", []):
                params = trial.get("params", {}) if isinstance(trial, dict) else {}
                metrics = _compute_l1_metrics(params)
                row = {
                    **metrics,
                }
                for k, v in params.items():
                    row[f"param_{k}"] = v
                trial_rows.append(row)

            if trial_rows:
                ts_l1 = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                l1_trials_path = Path("outcomes") / f"hpo_layer1_trials_{symbol}_{timeframe}_{ts_l1}.csv"
                pd.DataFrame(trial_rows).to_csv(l1_trials_path, index=False)
                tprint_info(f"   💾 Saved Layer 1 trial metrics to {l1_trials_path}")
        except Exception as l1_trials_exc:
            tprint_warning(f"   ⚠️ Failed to save Layer 1 trial metrics: {l1_trials_exc}")

        return best_params

    except Exception as e:
        tprint_warning(f"⚠️ Layer 1 optimization failed, using defaults: {e}")
        return default_params

def apply_weights_decay(weights: pd.Series, decay_factor: float = 1.0) -> pd.Series:
    """
    Apply time-decay to weights (give more weight to recent events).
    
    Args:
        weights: Series of sample weights.
        decay_factor: Decay per sample (1.0 = no decay). 
                      < 1.0 means recent samples have higher weight? 
                      Usually implemented as linear ramp or exponential.
                      Here assumes simple linear ramp based on index position.
    
    Returns:
        Decayed weights.
    """
    if decay_factor == 1.0:
        return weights
    
    # Linear ramp from decay_factor to 1.0? 
    # Or Lopez de Prado's time decay? 
    # Let's implement a simple linear time decay where the oldest sample gets weight 'decay_factor'
    # and newest gets 1.0, relative to their original weight. 
    # (Assuming decay_factor in [0, 1])
    
    tprint(f"📉 Applying time decay (factor={decay_factor})...", "INFO")
    
    n_samples = len(weights)
    ramp = np.linspace(decay_factor, 1.0, n_samples)
    
    # Sort weights by time to apply ramp correctly
    sorted_idx = weights.index.argsort()
    
    # Create an array aligned with weights.iloc (original order) but containing ramp values based on time rank
    ramp_aligned = np.zeros(n_samples)
    ramp_aligned[sorted_idx] = ramp 
    
    decayed_weights = weights * ramp_aligned
    
    # Renormalize
    if decayed_weights.sum() > 0:
        decayed_weights *= (n_samples / decayed_weights.sum())
        
    return decayed_weights

