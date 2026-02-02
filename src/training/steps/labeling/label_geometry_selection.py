import numpy as np
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
import lightgbm as lgb
from scipy.stats import ks_2samp, entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Dummy jit decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False
    prange = range

# Try importing optimized uniqueness calculation
try:
    from src.utils.orthogonal_numba import _numba_get_uniqueness, _numba_build_indicator_matrix
    ORTHOGONAL_NUMBA_AVAILABLE = True
except ImportError:
    ORTHOGONAL_NUMBA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Economic Constraints for Geometry Pre-Filtering ---
MAX_HORIZON_BARS = 48       # 12h at 15m timeframe
MIN_SL_PCT = 0.004          # 0.4% floor
MAX_TP_PCT = 0.05           # 5% ceiling
MIN_TP_SL_RATIO = 1.5       # TP >= 1.5 * SL (positive expectancy)
MIN_SL_SIGMA = 0.5          # Minimum stop-loss in sigma units (prevent too-tight stops)
MAX_FINAL_GEOMETRIES = 18    # Increased from 12 to allow more diverse candidates
MIN_GEOMETRY_DISTANCE = 0.15  # Normalized distance threshold for deduplication

# --- 1. Data Structures ---

@dataclass(frozen=True)
class Geometry:
    sl_quantile: float  # Quantile of MAE distribution (0.0 - 1.0) instead of fixed sigma
    alpha: float        # Pain penalty (denominator exponent)
    beta: float         # Gain reward (numerator exponent)
    min_ratio: float = 1.0
    horizon: int = 120  # Horizon in bars
    sl_sigma: Optional[float] = None # Resolved Sigma threshold (populated after selection)
    
    @property
    def archetype(self) -> str:
        """Auto-classifies the geometry into a human-readable archetype."""
        # Adapted archetypes for quantile logic
        if self.sl_quantile < 0.25 and self.beta > 1.0:
            return "Sniper (Selective, High Reward)"
        elif self.alpha > 1.0:
            return "Pain Averse (High Penalty for Drawdown)"
        elif self.sl_quantile > 0.6 and self.beta < 0.8:
            return "Deep Value (Loose Tolerance, Low Target)"
        elif self.beta >= 1.0 and self.alpha <= 0.5:
            return "Momentum Surfer (Tolerates Volatility)"
        else:
            return "Balanced"
            
    @property
    def is_tail(self) -> bool:
        """
        Identifies 'Tail' geometries that demand very high Reward/Risk ratio.
        """
        return self.beta > 1.2 or self.min_ratio > 2.0

@dataclass
class Event:
    id: int
    entry_idx: int
    exit_idx: int          # The Vertical Barrier (time limit)
    direction: int         # +1 / -1
    returns_path: np.array # Cumulative returns relative to entry
    sigma: float           # Volatility at entry
    asset_id: Optional[int] = None # Cross-asset support

@dataclass
class LearnabilityMetrics:
    """Comprehensive learnability metrics for geometry evaluation."""
    feature_importance_sum: float = 0.0     # Total feature importance magnitude
    auc_early: float = 0.5                   # AUC on early split (first 70%)
    auc_late: float = 0.5                    # AUC on late split (last 30%)
    auc_stability: float = 0.0              # |early - late| penalty (lower = more stable)
    temporal_consistency: float = 0.0       # IC-IR style: mean_lift / std_lift
    n_survivors: int = 0                    # Absolute number of surviving events
    composite_score: float = 0.0            # Final combined learnability score

@dataclass
class GateDiagnostics:
    """Detailed report on why a geometry passed or failed."""
    passed: bool
    survival_rate: float
    avg_uniqueness: float
    avg_auc_lift: float
    avg_pr_lift: float
    ks_stat: float
    entropy_reduction: float
    learnability: Optional[LearnabilityMetrics] = None
    reasons: List[str] = field(default_factory=list)

# --- 2. Loss Function (TradingFocalLoss) ---

class TradingFocalLoss:
    def __init__(
        self,
        gamma_pos=1.5,
        gamma_neg=3.0,
        alpha=None,
        w_cap=3.0,
        label_smoothing=0.02,
        mix=0.5
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.w_cap = w_cap
        self.label_smoothing = label_smoothing
        self.mix = mix

    def __call__(self, preds, train_data):
        y = train_data.get_label()
        y = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-6, 1 - 1e-6)

        alpha = self.alpha
        if alpha is None:
            pos_rate = np.mean(y)
            alpha = min(0.5, 1 - pos_rate)

        gamma = y * self.gamma_pos + (1 - y) * self.gamma_neg
        focal = np.minimum((1 - p)**gamma, self.w_cap)

        # Log loss
        logloss_grad = p - y
        logloss_hess = p * (1 - p)

        # Focal-weighted
        grad = focal * logloss_grad
        hess = focal * logloss_hess

        # Hybrid
        grad = self.mix * grad + (1 - self.mix) * logloss_grad
        hess = self.mix * hess + (1 - self.mix) * logloss_hess

        return grad, hess

# --- 3. Vectorization & Pre-computation ---

@jit(nopython=True, nogil=True)
def _numba_calculate_event_metrics(
    returns_path_flat: np.ndarray,
    path_offsets: np.ndarray,
    sigmas: np.ndarray,
    horizons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized calculation of path metrics for multiple horizons.

    Args:
        returns_path_flat: Flattened array of all event return paths.
        path_offsets: Start index for each event in returns_path_flat.
        sigmas: Volatility for each event.
        horizons: Array of horizons to check.

    Returns:
        Tuple of (mae_matrix, mfe_matrix, duration_matrix)
        Shape: (n_events, n_horizons)
    """
    n_events = len(path_offsets) - 1
    n_horizons = len(horizons)

    mae_out = np.zeros((n_events, n_horizons), dtype=np.float32)
    mfe_out = np.zeros((n_events, n_horizons), dtype=np.float32)
    durations_out = np.zeros((n_events, n_horizons), dtype=np.int32)

    for i in range(n_events):
        start = path_offsets[i]
        end = path_offsets[i+1]
        path_len = end - start
        
        sigma = sigmas[i]
        if sigma < 1e-9: sigma = 1e-9
        
        curr_min = 0.0
        curr_max = 0.0

        # Process path step by step
        for k in range(path_len):
            val = returns_path_flat[start + k]
            if val < curr_min: curr_min = val
            if val > curr_max: curr_max = val

            # Check if this step matches any horizon
            # Assuming horizons are sorted, we can check efficiently
            # Steps are 0-based index k corresponds to duration k+1

            for h_idx in range(n_horizons):
                h = horizons[h_idx]
                if k == h - 1:
                    # Reached horizon h
                    mae_out[i, h_idx] = -curr_min / sigma
                    mfe_out[i, h_idx] = curr_max / sigma
                    durations_out[i, h_idx] = k + 1

        # Fill remaining horizons if path ended early
        for h_idx in range(n_horizons):
            h = horizons[h_idx]
            if path_len < h:
                # Path shorter than horizon, use final values
                mae_out[i, h_idx] = -curr_min / sigma
                mfe_out[i, h_idx] = curr_max / sigma
                durations_out[i, h_idx] = path_len

    return mae_out, mfe_out, durations_out

def events_to_dataframe(events: List[Event], horizons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Converts events to DataFrame and pre-calculates path metrics for multiple horizons.
    Vectorized for performance using Numba.
    """
    if not events:
        return pd.DataFrame()

    if horizons is None:
        horizons = [8, 12, 16, 20, 30, 40]

    horizons = sorted(horizons)
    horizons_arr = np.array(horizons, dtype=np.int32)

    # Pre-allocate arrays
    n_events = len(events)

    ids = np.empty(n_events, dtype=object) # Use object for flexibility or int
    entry_idxs = np.empty(n_events, dtype=np.int32)
    exit_idxs = np.empty(n_events, dtype=np.int32)
    directions = np.empty(n_events, dtype=np.int32)
    sigmas = np.empty(n_events, dtype=np.float32)
    asset_ids = np.empty(n_events, dtype=object) # Can be None

    # Flatten paths for Numba
    # Since paths are numpy arrays, we can concatenate them efficiently
    paths = []

    for i, e in enumerate(events):
        ids[i] = e.id
        entry_idxs[i] = e.entry_idx
        exit_idxs[i] = e.exit_idx
        directions[i] = e.direction
        sigmas[i] = e.sigma
        asset_ids[i] = e.asset_id

        # Apply direction to path here so Numba only deals with raw values
        paths.append(e.returns_path * e.direction)

    # Create flattened path array and offsets
    path_lens = np.array([len(p) for p in paths], dtype=np.int32)
    path_offsets = np.concatenate([[0], np.cumsum(path_lens)])

    if len(paths) > 0:
        returns_path_flat = np.concatenate(paths).astype(np.float32)
    else:
        returns_path_flat = np.array([], dtype=np.float32)

    # Numba Calculation with Fallback
    if NUMBA_AVAILABLE and len(returns_path_flat) > 0:
        mae_mat, mfe_mat, dur_mat = _numba_calculate_event_metrics(
            returns_path_flat, path_offsets, sigmas, horizons_arr
        )
    else:
        # Fallback: Python iteration
        mae_mat = np.zeros((n_events, len(horizons)), dtype=np.float32)
        mfe_mat = np.zeros((n_events, len(horizons)), dtype=np.float32)
        dur_mat = np.zeros((n_events, len(horizons)), dtype=np.int32)

        for i, path in enumerate(paths):
            path_len = len(path)
            sigma = sigmas[i]
            if sigma < 1e-9: sigma = 1e-9

            curr_min = 0.0
            curr_max = 0.0

            # Simple simulation
            for k in range(path_len):
                val = path[k]
                curr_min = min(curr_min, val)
                curr_max = max(curr_max, val)

                # Check horizons
                for h_idx, h in enumerate(horizons):
                    if k == h - 1:
                        mae_mat[i, h_idx] = -curr_min / sigma
                        mfe_mat[i, h_idx] = curr_max / sigma
                        dur_mat[i, h_idx] = k + 1

            # Fill remaining
            for h_idx, h in enumerate(horizons):
                if path_len < h:
                    mae_mat[i, h_idx] = -curr_min / sigma
                    mfe_mat[i, h_idx] = curr_max / sigma
                    dur_mat[i, h_idx] = path_len

    # Construct DataFrame
    data = {
        'id': ids,
        'entry_idx': entry_idxs,
        'exit_idx': exit_idxs,
        'direction': directions,
        'sigma': sigmas,
        'duration_bars': exit_idxs - entry_idxs,
    }

    if asset_ids[0] is not None:
        data['asset_id'] = asset_ids

    # Add horizon columns
    for h_idx, h in enumerate(horizons):
        norm_mae = mae_mat[:, h_idx]
        norm_mfe = mfe_mat[:, h_idx]
        duration_h = dur_mat[:, h_idx]

        # Time-scaled normalization
        sqrt_t = np.sqrt(np.maximum(1, duration_h))
        time_scaled_mae = norm_mae / sqrt_t
        time_scaled_mfe = norm_mfe / sqrt_t

        data[f'norm_mae_{h}'] = norm_mae
        data[f'norm_mfe_{h}'] = norm_mfe
        data[f'time_scaled_mae_{h}'] = time_scaled_mae
        data[f'time_scaled_mfe_{h}'] = time_scaled_mfe

        # Legacy fields map to max horizon
        if h == horizons[-1]:
            data['norm_mae'] = norm_mae
            data['norm_mfe'] = norm_mfe
            data['duration'] = exit_idxs - entry_idxs # Use real duration
            data['time_scaled_mae'] = time_scaled_mae
            data['time_scaled_mfe'] = time_scaled_mfe

    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index('id', inplace=True)
    return df

# --- 4. Advanced Metrics ---

def calculate_separation_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calculates KS Statistic and Entropy.
    KS: Max divergence between CDF of positives and CDF of negatives.
    Entropy: Normalized Shannon entropy of predictions.
    """
    # KS Statistic
    pos_preds = y_prob[y_true == 1]
    neg_preds = y_prob[y_true == 0]

    if len(pos_preds) == 0 or len(neg_preds) == 0:
        ks_stat = 0.0
    else:
        ks_result = ks_2samp(pos_preds, neg_preds)
        ks_stat = ks_result.statistic

    # Entropy
    # Clip probabilities for safety (Log safety)
    p = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    # Binary entropy per sample
    ent_samples = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    # Mean entropy
    avg_ent = np.mean(ent_samples)
    # Max possible entropy (log(2))
    max_ent = np.log(2)
    # Normalized entropy (0 to 1, where 1 is total uncertainty)
    norm_ent = avg_ent / max_ent

    return ks_stat, float(norm_ent)

@jit(nopython=True, nogil=True)
def _numba_indicator_matrix_global(
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
    max_idx: int
) -> np.ndarray:
    """
    Builds concurrency array for global indices (concatenated or single asset).
    This handles the case where assets are concatenated linearly.
    """
    indicator = np.zeros(max_idx + 1, dtype=np.float32)
    n_events = len(entry_indices)

    # Use diff array for O(N + T) complexity instead of O(N*W)
    diff = np.zeros(max_idx + 2, dtype=np.float32)

    for i in range(n_events):
        start = entry_indices[i]
        end = exit_indices[i]
        if start < 0 or start > max_idx: continue
        if end > max_idx: end = max_idx
        if end <= start: continue

        diff[start] += 1
        diff[end] -= 1

    current = 0.0
    for t in range(max_idx + 1):
        current += diff[t]
        indicator[t] = current

    return indicator

def get_average_uniqueness(selected_indices, all_events_df) -> float:
    """
    Calculates average uniqueness using time-weighted concurrency.
    Supports multi-asset if asset_id is present, or assumes concatenated global indices.
    """
    # Fix bug: ambiguous boolean check for Index
    if len(selected_indices) == 0:
        return 0.0
        
    subset = all_events_df.loc[list(selected_indices)]
    if subset.empty or 'entry_idx' not in subset or 'exit_idx' not in subset:
        return 0.0
    
    # Determine processing mode
    has_asset_id = 'asset_id' in subset.columns and subset['asset_id'].notna().any()
    
    if has_asset_id:
        # Compute uniqueness per asset and average
        uniqueness_scores = []
        for asset, group in subset.groupby('asset_id'):
            u = _compute_uniqueness_single_series(group)
            uniqueness_scores.append(u)
        return float(np.mean(uniqueness_scores)) if uniqueness_scores else 0.0
    else:
        # Global uniqueness (assuming concatenated indices)
        return _compute_uniqueness_single_series(subset)

def _compute_uniqueness_single_series(df_subset: pd.DataFrame) -> float:
    """Helper to compute uniqueness on a single index series."""
    starts = df_subset['entry_idx'].astype(np.int32).values
    ends = df_subset['exit_idx'].astype(np.int32).values

    if len(starts) == 0:
        return 0.0

    max_idx = int(ends.max())
    
    if NUMBA_AVAILABLE:
        # Build concurrency curve
        concurrency = _numba_indicator_matrix_global(starts, ends, max_idx)
        return _numba_variable_uniqueness(starts, ends, concurrency)
    else:
        # Python fallback using coordinate compression
        # This is the original logic moved to a helper
        return _python_variable_uniqueness(starts, ends)

def _python_variable_uniqueness(starts, ends):
    # Coordinate compression over all boundary points
    boundaries = np.unique(np.concatenate([starts, ends]))
    if len(boundaries) < 2:
        return 0.0
    
    boundary_to_pos = {val: idx for idx, val in enumerate(boundaries)}
    diff = np.zeros(len(boundaries), dtype=float)
    
    for s, e in zip(starts, ends):
        diff[boundary_to_pos[s]] += 1.0
        diff[boundary_to_pos[e]] -= 1.0
    
    concurrency = np.cumsum(diff)[:-1]  # concurrency per interval
    interval_lengths = np.diff(boundaries).astype(float)
    
    event_scores = []
    for s, e in zip(starts, ends):
        s_idx = boundary_to_pos[s]
        e_idx = boundary_to_pos[e]
        conc_slice = concurrency[s_idx:e_idx]
        len_slice = interval_lengths[s_idx:e_idx]
        
        mask = conc_slice > 0
        if not np.any(mask):
            continue
        
        weights = len_slice[mask]
        inv_conc = 1.0 / conc_slice[mask]
        event_scores.append(np.average(inv_conc, weights=weights))
    
    return float(np.mean(event_scores)) if event_scores else 0.0

@jit(nopython=True, nogil=True)
def _numba_variable_uniqueness(starts, ends, concurrency):
    n_events = len(starts)
    total_uniq = 0.0

    for i in range(n_events):
        s = starts[i]
        e = ends[i]
        if e <= s: continue

        # Average (1/c) over [s, e)
        sum_inv_c = 0.0
        count = 0
        for t in range(s, e):
            c = concurrency[t]
            if c > 0:
                sum_inv_c += 1.0 / c
            count += 1

        if count > 0:
            total_uniq += sum_inv_c / count

    return total_uniq / n_events if n_events > 0 else 0.0


def filter_informative_features(features_df: pd.DataFrame, event_ids: pd.Index, variance_threshold: float = 1e-12) -> Optional[pd.DataFrame]:
    """
    Aligns the feature matrix to the deduplicated events and removes columns
    with variance below the specified threshold.
    """
    if features_df is None or features_df.empty:
        return None
    
    aligned = features_df.reindex(event_ids)
    if aligned.isnull().any().any():
        aligned = aligned.fillna(0.0)
    
    variances = aligned.var()
    keep_cols = variances[variances > variance_threshold].index.tolist()
    
    if not keep_cols:
        logger.warning("All feature columns became constant after deduplication.")
        return None
    
    dropped = len(aligned.columns) - len(keep_cols)
    if dropped > 0:
        logger.info(f"Dropped {dropped} near-constant feature columns after dedup ({len(keep_cols)} remaining).")
    
    return aligned[keep_cols]


def run_logistic_probe(
    X: pd.DataFrame,
    y: pd.Series,
    min_samples: int = 15
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """
    Fit a regularized logistic regression as a lightweight probe.
    """
    if X is None or X.empty or len(X) < (min_samples * 3):
        return None
    
    if len(np.unique(y)) < 2:
        return None
    
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < min_samples or n_neg < min_samples:
        return None
    
    # Float32 conversion for memory/speed
    X = X.astype(np.float32)

    split_idx = int(len(X) * 0.8)
    if split_idx < min_samples or (len(X) - split_idx) < max(8, min_samples // 2):
        return None
    
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(np.unique(y_val)) < 2:
        return None
    
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_full_scaled = scaler.transform(X)
    except Exception:
        return None
    
    try:
        clf = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=500,
            class_weight='balanced',
            n_jobs=1
        )
        clf.fit(X_train_scaled, y_train)
    except Exception:
        return None
    
    try:
        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    except Exception:
        return None
    
    try:
        auc_val = roc_auc_score(y_val, val_probs)
        auc_lift = abs(auc_val - 0.5)
    except Exception:
        auc_lift = 0.0
    
    try:
        pr_val = average_precision_score(y_val, val_probs)
        pr_lift = pr_val - y_val.mean()
    except Exception:
        pr_lift = 0.0
    
    ks_stat, ent = calculate_separation_metrics(y_val.to_numpy(), val_probs)
    
    try:
        preds_full_prob = clf.predict_proba(X_full_scaled)[:, 1]
    except Exception:
        preds_full_prob = np.full(len(X), 0.5, dtype=float)
    
    metrics = {
        'auc_lift': auc_lift,
        'pr_lift': pr_lift,
        'ks_stat': ks_stat,
        'entropy': ent
    }
    
    return preds_full_prob, metrics

@jit(nopython=True, nogil=True)
def _numba_deduplicate_events(
    indices: np.ndarray,
    entry_idxs: np.ndarray,
    directions: np.ndarray,
    asset_ids: np.ndarray,
    min_gap: int
) -> List[int]:
    """
    Greedy deduplication logic in Numba.
    indices: original DataFrame indices to return
    """
    n = len(entry_idxs)
    keep_list = []

    # State tracking
    last_asset = -999 # Assuming asset_ids are ints. If None, we treat as 0.
    last_entry_dir_pos = -999999
    last_entry_dir_neg = -999999

    for i in range(n):
        curr_asset = asset_ids[i]
        curr_entry = entry_idxs[i]
        curr_dir = directions[i]

        if curr_asset != last_asset:
            # Reset state for new asset
            last_asset = curr_asset
            last_entry_dir_pos = -999999
            last_entry_dir_neg = -999999

        # Check gap
        if curr_dir >= 0: # Long or Neutral
            if curr_entry - last_entry_dir_pos >= min_gap:
                keep_list.append(indices[i])
                last_entry_dir_pos = curr_entry
        else: # Short
            if curr_entry - last_entry_dir_neg >= min_gap:
                keep_list.append(indices[i])
                last_entry_dir_neg = curr_entry

    return keep_list

def deduplicate_events(
    events_df: pd.DataFrame,
    min_gap_bars: int = 4  # Minimum bars between events with same direction
) -> pd.DataFrame:
    """
    Remove redundant events from CUSUM bursts using Numba.
    """
    if events_df.empty or 'direction' not in events_df.columns:
        return events_df
    
    # Prepare data for Numba
    # Sort by asset_id (if exists) and entry_idx
    sort_cols = ['entry_idx']
    if 'asset_id' in events_df.columns:
        sort_cols = ['asset_id', 'entry_idx']

    df_sorted = events_df.sort_values(sort_cols)
    
    indices = df_sorted.index.values
    entry_idxs = df_sorted['entry_idx'].values.astype(np.int32)
    directions = df_sorted['direction'].values.astype(np.int32)
    
    if 'asset_id' in df_sorted.columns:
        # Ensure asset_id is integer
        # If string/object, encode it
        if not pd.api.types.is_integer_dtype(df_sorted['asset_id']):
            asset_ids = df_sorted['asset_id'].astype('category').cat.codes.values.astype(np.int32)
        else:
            asset_ids = df_sorted['asset_id'].fillna(-1).values.astype(np.int32)
    else:
        asset_ids = np.zeros(len(df_sorted), dtype=np.int32)
        
    # Call Numba
    if NUMBA_AVAILABLE:
        keep_indices = _numba_deduplicate_events(indices, entry_idxs, directions, asset_ids, min_gap_bars)
    else:
        # Fallback to slow Python loop if Numba fails to load
        return _deduplicate_events_python(events_df, min_gap_bars)
    
    result = events_df.loc[keep_indices]
    
    if len(result) < len(events_df):
        logger.info(f"Event dedup: {len(events_df)} → {len(result)} events (min_gap={min_gap_bars} bars)")
    
    return result

def _deduplicate_events_python(events_df, min_gap_bars):
    # Legacy logic
    df = events_df.sort_values('entry_idx').copy()
    keep_indices = []
    last_entry_by_dir = {1: -float('inf'), -1: -float('inf')}
    for idx, row in df.iterrows():
        entry = row['entry_idx']
        direction = row.get('direction', 0)
        if entry - last_entry_by_dir.get(direction, -float('inf')) >= min_gap_bars:
            keep_indices.append(idx)
            last_entry_by_dir[direction] = entry
    return events_df.loc[keep_indices]

def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b: return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


# --- 4.5 Geometry Pre-Filtering Functions ---

def geometry_distance(g1: Geometry, g2: Geometry) -> float:
    """
    Compute normalized distance between two geometries.
    Used to ensure diversity in selected geometries.
    """
    sl_diff = abs(g1.sl_quantile - g2.sl_quantile)
    alpha_diff = abs(g1.alpha - g2.alpha) / 2.0  # Scale to ~[0,1]
    beta_diff = abs(g1.beta - g2.beta) / 2.0
    ratio_diff = abs(g1.min_ratio - g2.min_ratio) / 3.0
    horizon_diff = abs(g1.horizon - g2.horizon) / float(MAX_HORIZON_BARS)
    return (sl_diff + alpha_diff + beta_diff + ratio_diff + horizon_diff) / 5.0


def apply_hard_constraints(
    candidates: List[Geometry],
    thresholds_map: Dict[int, Dict[float, float]]
) -> List[Geometry]:
    """
    Filter geometry candidates by economic hard constraints BEFORE LGBM training.
    This is the key optimization to avoid wasted computation.
    
    Note: sl_sigma values are in SIGMA units (volatility-normalized), not raw percentages.
    So we filter by horizon, min_ratio, and minimum sl_sigma here.
    """
    valid = []
    for g in candidates:
        # 1. Horizon constraint (max 6h = 24 bars)
        if g.horizon > MAX_HORIZON_BARS:
            continue
        
        # 2. TP/SL ratio constraint (min 1.5x for positive expectancy)
        if g.min_ratio < MIN_TP_SL_RATIO:
            continue
        
        # 3. Minimum stop-loss constraint (prevent too-tight stops)
        if g.sl_sigma and g.sl_sigma < MIN_SL_SIGMA:
            continue
        
        # Note: MIN_SL_PCT and MAX_TP_PCT don't apply because sl_sigma is in sigma units
        # (typically 0.5 to 3.0), not raw percentage values (0.004 = 0.4%)
        
        valid.append(g)
    
    return valid


def parameter_diversity_penalty(new_geom: Geometry, selected_geoms: List[Geometry]) -> bool:
    """
    Check if new geometry is too similar to already selected ones in parameter space.
    Returns True if geometry should be rejected due to similarity.
    """
    for sel_geom in selected_geoms:
        # Check parameter similarity
        sl_diff = abs(new_geom.sl_quantile - sel_geom.sl_quantile)
        alpha_diff = abs(new_geom.alpha - sel_geom.alpha) / 2.0  # Normalized
        beta_diff = abs(new_geom.beta - sel_geom.beta) / 2.0    # Normalized
        ratio_diff = abs(new_geom.min_ratio - sel_geom.min_ratio) / 3.0  # Normalized
        horizon_diff = abs(new_geom.horizon - sel_geom.horizon) / 24.0   # Normalized
        
        # If all parameters are very similar, reject
        if (sl_diff < 0.1 and alpha_diff < 0.3 and beta_diff < 0.3 and 
            ratio_diff < 0.2 and horizon_diff < 0.2):
            return True
    
    return False


def deduplicate_by_distance(
    candidates: List[Geometry],
    min_dist: float = MIN_GEOMETRY_DISTANCE
) -> List[Geometry]:
    """
    Keep only geometries with sufficient distance from each other.
    Greedy selection: first candidate is always kept, subsequent ones
    must be far enough from all kept geometries.
    """
    if not candidates:
        return []
    
    kept = [candidates[0]]
    for c in candidates[1:]:
        if all(geometry_distance(c, k) >= min_dist for k in kept):
            kept.append(c)
        # Early exit if we have enough
        if len(kept) >= MAX_FINAL_GEOMETRIES * 20:  # Keep 20x target (200) for LGBM filtering
            break
    
    logger.info(f"Distance-based deduplication: {len(candidates)} -> {len(kept)} candidates")
    return kept


# --- 4.6 Learnability Scoring ---

def compute_learnability_metrics(
    survivor_ids: Set[int],
    all_event_ids: List[int],
    features_df: pd.DataFrame,
    temporal_split_ratio: float = 0.7,
    min_samples_per_split: int = 30,
    min_survivors_absolute: int = 50,
) -> LearnabilityMetrics:
    """
    Compute comprehensive learnability metrics for a geometry.
    """
    metrics = LearnabilityMetrics()
    metrics.n_survivors = len(survivor_ids)
    
    # Early exit if insufficient data
    if features_df.empty or len(survivor_ids) < min_survivors_absolute:
        return metrics
    
    # Build target
    target = pd.Series(0, index=all_event_ids)
    target.loc[list(survivor_ids)] = 1
    
    X = features_df.loc[all_event_ids].copy()
    y = target.loc[all_event_ids]
    
    # Drop low-variance features
    variances = X.var()
    informative_cols = variances[variances >= 1e-6].index.tolist()
    if len(informative_cols) < 3:
        return metrics
    X = X[informative_cols]
    
    # Temporal split: first 70% = early, last 30% = late
    n_total = len(X)
    split_idx = int(n_total * temporal_split_ratio)
    
    if split_idx < min_samples_per_split or (n_total - split_idx) < min_samples_per_split:
        return metrics
    
    X_early, X_late = X.iloc[:split_idx], X.iloc[split_idx:]
    y_early, y_late = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(np.unique(y_early)) < 2 or len(np.unique(y_late)) < 2:
        return metrics
    
    n_pos_early = int(y_early.sum())
    n_pos_late = int(y_late.sum())
    if n_pos_early < 10 or n_pos_late < 10:
        return metrics
    
    # Further split early data for train/val within early period
    early_train_idx = int(len(X_early) * 0.75)
    X_train = X_early.iloc[:early_train_idx]
    X_val = X_early.iloc[early_train_idx:]
    y_train = y_early.iloc[:early_train_idx]
    y_val = y_early.iloc[early_train_idx:]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return metrics
    
    try:
        train_data = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': 4,
            'num_leaves': 15,
            'learning_rate': 0.05,
            'verbose': -1,
            'verbosity': -1,
            'min_child_samples': 10,
            'reg_lambda': 1.0,
            'reg_alpha': 0.5,
            'n_jobs': 1,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            callbacks=[lgb.early_stopping(20, verbose=False)],
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
        )
        
        importances = model.feature_importance(importance_type='gain')
        metrics.feature_importance_sum = float(np.sum(importances))
        
        preds_early = model.predict(X_val)
        try:
            auc_early = roc_auc_score(y_val, preds_early)
            metrics.auc_early = float(auc_early)
        except Exception:
            metrics.auc_early = 0.5
        
        preds_late = model.predict(X_late)
        try:
            auc_late = roc_auc_score(y_late, preds_late)
            metrics.auc_late = float(auc_late)
        except Exception:
            metrics.auc_late = 0.5
        
        metrics.auc_stability = abs(metrics.auc_early - metrics.auc_late)
        
        lift_early = metrics.auc_early - 0.5
        lift_late = metrics.auc_late - 0.5
        mean_lift = (lift_early + lift_late) / 2.0
        std_lift = np.std([lift_early, lift_late]) + 0.01
        metrics.temporal_consistency = mean_lift / std_lift
        
        importance_score = min(1.0, metrics.feature_importance_sum / 1000.0)
        auc_lift_score = (mean_lift + 0.5) * 2.0
        stability_bonus = max(0, 0.2 - metrics.auc_stability)
        consistency_score = max(0, metrics.temporal_consistency)
        
        metrics.composite_score = (
            0.25 * importance_score +
            0.35 * auc_lift_score +
            0.20 * stability_bonus +
            0.20 * consistency_score
        )
        
    except Exception as e:
        logger.debug(f"Learnability computation failed: {e}")
        return metrics
    
    return metrics


# --- 5. Diagnostics-First Gates ---

def run_diagnostics_gates(
    survivor_ids: list,
    events_df: pd.DataFrame,
    fold_metrics: dict,
    geometry: Geometry,
    features_df: Optional[pd.DataFrame] = None,
    default_min_survival: float = 0.005,
    tail_min_survival: float = 0.005,
    min_uniqueness: float = 0.15,
    min_survivors_absolute: int = 50,
    min_learnability_score: float = 0.15,
    max_temporal_instability: float = 0.20,
) -> GateDiagnostics:
    """
    Enhanced diagnostics with learnability-based gating.
    """
    reasons = []
    is_passing = True
    learnability = None
    
    current_min_survival = tail_min_survival if geometry.is_tail else default_min_survival
    rate = len(survivor_ids) / len(events_df) if len(events_df) > 0 else 0.0
    
    if rate < current_min_survival:
        is_passing = False
        reasons.append(f"Low Survival ({rate:.2%} < {current_min_survival:.2%})")
        
    avg_u = get_average_uniqueness(survivor_ids, events_df)
    if avg_u < min_uniqueness:
        is_passing = False
        reasons.append(f"Low Uniqueness ({avg_u:.2f} < {min_uniqueness})")
    
    n_survivors = len(survivor_ids)
    if n_survivors < min_survivors_absolute:
        is_passing = False
        reasons.append(f"Too Few Survivors ({n_survivors} < {min_survivors_absolute})")
    
    if features_df is not None and not features_df.empty:
        learnability = compute_learnability_metrics(
            survivor_ids=set(survivor_ids),
            all_event_ids=list(events_df.index),
            features_df=features_df,
            min_survivors_absolute=min_survivors_absolute,
        )
        
        if learnability.composite_score < min_learnability_score:
            is_passing = False
            reasons.append(f"Low Learnability ({learnability.composite_score:.3f} < {min_learnability_score})")
        
        if learnability.auc_stability > max_temporal_instability:
            is_passing = False
            reasons.append(f"Temporal Instability ({learnability.auc_stability:.3f} > {max_temporal_instability})")
    
    avg_auc = 0.0
    avg_pr_lift = 0.0
    ks_stat = 0.0
    entropy_val = 1.0

    if fold_metrics:
        aucs = [m.get('auc_lift', 0.0) for m in fold_metrics.values() if isinstance(m, dict)]
        if aucs:
            avg_auc = np.mean(aucs)

        prs = [m.get('pr_lift', 0.0) for m in fold_metrics.values() if isinstance(m, dict)]
        if prs:
            avg_pr_lift = np.mean(prs)

        kss = [m.get('ks_stat', 0.0) for m in fold_metrics.values() if isinstance(m, dict)]
        if kss:
            ks_stat = np.mean(kss)

        ents = [m.get('entropy', 1.0) for m in fold_metrics.values() if isinstance(m, dict)]
        if ents:
            entropy_val = np.mean(ents)

    return GateDiagnostics(
        passed=is_passing,
        survival_rate=rate,
        avg_uniqueness=avg_u,
        avg_auc_lift=avg_auc,
        avg_pr_lift=avg_pr_lift,
        ks_stat=ks_stat,
        entropy_reduction=(1.0 - entropy_val),
        learnability=learnability,
        reasons=reasons
    )

# --- 6. Model Training ---

def train_model_for_geometry(
    survivor_ids: Set[int],
    all_event_ids: List[int],
    features_df: pd.DataFrame,
    min_positive_samples: int = 25,
    min_informative_features: int = 3,
    variance_threshold: float = 1e-6
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Trains a Weak Learner (max depth = 3) using TradingFocalLoss.
    """
    default_metrics = {'auc': 0.5, 'ks': 0.0, 'entropy': 1.0}
    
    if features_df.empty:
        return None, np.zeros(len(all_event_ids)), default_metrics

    target = pd.Series(0, index=all_event_ids)
    target.loc[list(survivor_ids)] = 1
    
    X = features_df.loc[all_event_ids].copy()
    y = target.loc[all_event_ids]
    
    variances = X.var()
    informative_cols = variances[variances >= variance_threshold].index.tolist()
    
    if len(informative_cols) < min_informative_features:
        logger.info(f"Model skipped: only {len(informative_cols)} informative features (need {min_informative_features})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    X = X[informative_cols]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(np.unique(y_train)) < 2:
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    n_positives = int(y_train.sum())
    n_negatives = len(y_train) - n_positives
    if n_positives < min_positive_samples:
        logger.info(f"Model skipped: only {n_positives} positive samples (need {min_positive_samples})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    if n_negatives < min_positive_samples:
        logger.info(f"Model skipped: only {n_negatives} negative samples (need {min_positive_samples})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    train_variances = X_train.var()
    informative_train_cols = train_variances[train_variances > 1e-6].index.tolist()
    
    if len(informative_train_cols) < min_informative_features:
        logger.info(f"Model skipped: only {len(informative_train_cols)} train-informative features (need {min_informative_features})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    X_train = X_train[informative_train_cols]
    X_val = X_val[informative_train_cols]
    X = X[informative_train_cols]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    focal_loss = TradingFocalLoss(
        gamma_pos=1.5,
        gamma_neg=3.0,
        alpha=None,
        w_cap=3.0,
        label_smoothing=0.02,
        mix=0.5
    )

    params = {
        'objective': focal_loss, 
        'metric': ['auc', 'average_precision'],
        'max_depth': 3,
        'verbose': -1,
        'verbosity': -1,
        'num_leaves': 7,
        'learning_rate': 0.05
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    preds_val_raw = model.predict(X_val)
    preds_val_prob = 1.0 / (1.0 + np.exp(-preds_val_raw))

    preds_full_raw = model.predict(X)
    preds_full_prob = 1.0 / (1.0 + np.exp(-preds_full_raw))

    prevalence = y_val.mean()
    try:
        auc_val = roc_auc_score(y_val, preds_val_prob)
        if auc_val < 0.5:
            logger.debug(f"Model shows anti-correlation AUC={auc_val:.3f} (likely noise, not inverse signal)")
        
        auc_lift = abs(auc_val - 0.5)

        pr_val = average_precision_score(y_val, preds_val_prob)
        pr_lift = pr_val - prevalence
    except:
        auc_lift = 0.0
        pr_lift = 0.0

    ks_stat, ent = calculate_separation_metrics(y_val.values, preds_val_prob)

    metrics = {
        'auc_lift': auc_lift,
        'pr_lift': pr_lift,
        'ks_stat': ks_stat,
        'entropy': ent
    }
    
    return model, preds_full_prob, metrics

# --- 7. Main Selection Loop ---

# Legacy geometry selection functions removed - replaced by orthogonal_label_generation.py
# which uses signal family generation instead of parameter sweeps
