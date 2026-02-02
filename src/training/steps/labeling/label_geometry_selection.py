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
from src.utils.numba_funcs import jit, NUMBA_AVAILABLE

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

def events_to_dataframe(events: List[Event], horizons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Converts events to DataFrame and pre-calculates path metrics for multiple horizons.
    Optimized for performance using dictionary of lists instead of list of dictionaries.

    Args:
        events: List of Event objects.
        horizons: List of horizons to calculate metrics for (e.g. [24, 48, 120]).
                  If None, defaults to [8, 12, 16, 20, 30, 40].
    """
    if not events:
        return pd.DataFrame()

    if horizons is None:
        horizons = [8, 12, 16, 20, 30, 40]

    # Pre-allocate lists
    ids = []
    entries = []
    exits = []
    directions = []
    durations = []
    sigmas = []

    # Horizon specific lists
    horizon_cols = {h: {
        'norm_mae': [],
        'norm_mfe': [],
        'time_scaled_mae': [],
        'time_scaled_mfe': []
    } for h in horizons}

    max_h = max(horizons) if horizons else None
    legacy_norm_mae = []
    legacy_norm_mfe = []
    legacy_ts_mae = []
    legacy_ts_mfe = []
    legacy_duration = []

    for e in events:
        full_path = e.returns_path * e.direction
        max_len = len(full_path)
        
        # FIX: Use REAL duration from exit_idx - entry_idx, not truncated path length
        # Ensure integer math
        real_duration_bars = int(e.exit_idx - e.entry_idx)
        
        ids.append(e.id)
        entries.append(e.entry_idx)
        exits.append(e.exit_idx)
        directions.append(e.direction)
        durations.append(real_duration_bars)
        sigmas.append(e.sigma)

        for h in horizons:
            limit = min(max_len, h)

            if limit > 0:
                # Numpy slice view
                path_view = full_path[:limit]
                # Inlined min/max for speed
                raw_mae = -np.min(path_view)
                raw_mfe = np.max(path_view)
            else:
                raw_mae = 0.0
                raw_mfe = 0.0

            duration_h = max(1, limit)

            # Avoid division by zero
            safe_sigma = e.sigma if e.sigma > 1e-12 else 1e-9

            # Standard normalization
            norm_mae = raw_mae / safe_sigma
            norm_mfe = raw_mfe / safe_sigma

            # Time-scaled normalization
            sqrt_t = np.sqrt(duration_h)
            time_scaled_mae = raw_mae / (safe_sigma * sqrt_t)
            time_scaled_mfe = raw_mfe / (safe_sigma * sqrt_t)

            cols = horizon_cols[h]
            cols['norm_mae'].append(norm_mae)
            cols['norm_mfe'].append(norm_mfe)
            cols['time_scaled_mae'].append(time_scaled_mae)
            cols['time_scaled_mfe'].append(time_scaled_mfe)

            if h == max_h:
                legacy_norm_mae.append(norm_mae)
                legacy_norm_mfe.append(norm_mfe)
                legacy_ts_mae.append(time_scaled_mae)
                legacy_ts_mfe.append(time_scaled_mfe)
                legacy_duration.append(real_duration_bars)

    # Construct DataFrame from dict of lists
    data_dict = {
        'id': ids,
        'entry_idx': entries,
        'exit_idx': exits,
        'direction': directions,
        'duration_bars': durations,
        'sigma': sigmas
    }

    for h in horizons:
        cols = horizon_cols[h]
        data_dict[f'norm_mae_{h}'] = cols['norm_mae']
        data_dict[f'norm_mfe_{h}'] = cols['norm_mfe']
        data_dict[f'time_scaled_mae_{h}'] = cols['time_scaled_mae']
        data_dict[f'time_scaled_mfe_{h}'] = cols['time_scaled_mfe']
    
    # Legacy columns
    data_dict['norm_mae'] = legacy_norm_mae
    data_dict['norm_mfe'] = legacy_norm_mfe
    data_dict['duration'] = legacy_duration
    data_dict['time_scaled_mae'] = legacy_ts_mae
    data_dict['time_scaled_mfe'] = legacy_ts_mfe

    df = pd.DataFrame(data_dict)
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
    # Clip probabilities for safety
    p = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    # Binary entropy per sample
    ent_samples = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    # Mean entropy
    avg_ent = np.mean(ent_samples)
    # Max possible entropy (log(2))
    max_ent = np.log(2)
    # Normalized entropy (0 to 1, where 1 is total uncertainty)
    norm_ent = avg_ent / max_ent

    # We want Entropy REDUCTION (i.e., lower is better).
    # Let's return the raw normalized entropy for now, diagnostics will interpret it.

    return ks_stat, norm_ent

@jit(nopython=True, cache=True)
def _numba_uniqueness_loop(
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    concurrency: np.ndarray,
    interval_lengths: np.ndarray
) -> float:
    total_score = 0.0
    count = 0
    n_events = len(start_indices)

    for i in range(n_events):
        s = start_indices[i]
        e = end_indices[i]

        sum_w = 0.0
        sum_val = 0.0

        # Iterate over the intervals covered by this event
        for k in range(s, e):
            c = concurrency[k]
            # Avoid division by zero, though concurrency should be >= 1 for active events
            if c > 0:
                l = interval_lengths[k]
                sum_w += l
                sum_val += (1.0 / c) * l

        if sum_w > 0:
            total_score += sum_val / sum_w
            count += 1

    if count == 0:
        return 0.0
    return total_score / count

def get_average_uniqueness(selected_indices, all_events_df) -> float:
    """
    Calculates average uniqueness using time-weighted concurrency.
    Uses coordinate compression on absolute entry/exit indices to avoid
    building huge arrays when there are gaps in timestamps.
    """
    if not selected_indices:
        return 0.0
        
    subset = all_events_df.loc[list(selected_indices)]
    if subset.empty or 'entry_idx' not in subset or 'exit_idx' not in subset:
        return 0.0
    
    starts = subset['entry_idx'].astype(int).to_numpy()
    ends = subset['exit_idx'].astype(int).to_numpy()
    valid = ends > starts
    if not np.any(valid):
        return 0.0
    
    starts = starts[valid]
    ends = ends[valid]
    
    # Coordinate compression over all boundary points
    boundaries = np.unique(np.concatenate([starts, ends]))
    if len(boundaries) < 2:
        return 0.0
    
    if NUMBA_AVAILABLE:
        try:
            # Map to indices using binary search
            start_indices = np.searchsorted(boundaries, starts)
            end_indices = np.searchsorted(boundaries, ends)

            n_boundaries = len(boundaries)
            diff = np.zeros(n_boundaries, dtype=np.float64)

            # Fast accumulation
            np.add.at(diff, start_indices, 1.0)
            np.add.at(diff, end_indices, -1.0)

            concurrency = np.cumsum(diff)[:-1]
            interval_lengths = np.diff(boundaries).astype(np.float64)

            return _numba_uniqueness_loop(start_indices, end_indices, concurrency, interval_lengths)
        except Exception as e:
            logger.warning(f"Numba uniqueness calc failed, falling back: {e}")
            # Fallback below
            pass

    # Legacy / Fallback
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
    Fit a regularized logistic regression as a lightweight probe to detect
    linear separability when LightGBM cannot train.
    Returns predictions over the full dataset and diagnostic metrics.
    """
    if X is None or X.empty or len(X) < (min_samples * 3):
        return None
    
    if len(np.unique(y)) < 2:
        return None
    
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < min_samples or n_neg < min_samples:
        return None
    
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
    except Exception as e:
        logger.warning(f"Logistic probe scaling failed: {e}")
        return None
    
    try:
        clf = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=500,
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train)
    except Exception as e:
        logger.warning(f"Logistic probe fitting failed: {e}")
        return None
    
    try:
        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    except Exception as e:
        logger.warning(f"Logistic probe prediction failed: {e}")
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


@jit(nopython=True, cache=True)
def _numba_deduplicate_events(
    entries: np.ndarray,
    directions: np.ndarray,
    min_gap_bars: int
) -> np.ndarray:
    n = len(entries)
    keep_mask = np.ones(n, dtype=np.bool_)

    # Track last accepted entry index for each direction
    # Initialize with a value far in the past
    last_entry_pos = -1e9  # For direction 1
    last_entry_neg = -1e9  # For direction -1
    last_entry_zero = -1e9 # For direction 0

    for i in range(n):
        entry = entries[i]
        d = directions[i]

        should_keep = True

        if d == 1:
            if entry - last_entry_pos < min_gap_bars:
                should_keep = False
            else:
                last_entry_pos = entry
        elif d == -1:
            if entry - last_entry_neg < min_gap_bars:
                should_keep = False
            else:
                last_entry_neg = entry
        elif d == 0:
            if entry - last_entry_zero < min_gap_bars:
                should_keep = False
            else:
                last_entry_zero = entry
        else:
            # Fallback for unexpected directions: keep them
            pass

        keep_mask[i] = should_keep

    return keep_mask


def deduplicate_events(
    events_df: pd.DataFrame,
    min_gap_bars: int = 4  # Minimum bars between events with same direction
) -> pd.DataFrame:
    """
    Remove redundant events from CUSUM bursts.
    Events within min_gap_bars of each other with the same direction are 
    likely from the same CUSUM regime and should be deduplicated to 
    reduce artificial overlap in uniqueness calculation.
    
    Returns a subset of events_df with reduced concurrency.
    """
    if events_df.empty or 'direction' not in events_df.columns:
        return events_df
    
    # Sort by entry_idx to process chronologically
    df = events_df.sort_values('entry_idx').copy()
    
    if NUMBA_AVAILABLE:
        try:
            entries = df['entry_idx'].values.astype(np.float64)
            directions = df['direction'].values.astype(np.int32)
            mask = _numba_deduplicate_events(entries, directions, min_gap_bars)
            result = df[mask]
        except Exception as e:
            logger.warning(f"Numba deduplication failed, falling back to Python: {e}")
            # Fallback
            result = _python_deduplicate_events(df, min_gap_bars, events_df)
    else:
        result = _python_deduplicate_events(df, min_gap_bars, events_df)

    if len(result) < len(events_df):
        logger.info(f"Event dedup: {len(events_df)} → {len(result)} events (min_gap={min_gap_bars} bars)")

    return result

def _python_deduplicate_events(sorted_df: pd.DataFrame, min_gap_bars: int, original_df: pd.DataFrame) -> pd.DataFrame:
    keep_indices = []
    last_entry_by_dir = {1: -float('inf'), -1: -float('inf')}
    
    for idx, row in sorted_df.iterrows():
        entry = row['entry_idx']
        direction = row.get('direction', 0)
        
        # Only keep if far enough from last event in same direction
        if entry - last_entry_by_dir.get(direction, -float('inf')) >= min_gap_bars:
            keep_indices.append(idx)
            last_entry_by_dir[direction] = entry
    
    return original_df.loc[keep_indices]

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
    
    Combines:
    1. Feature importance magnitude - higher = features matter
    2. Temporal stability - AUC in early vs late splits should be consistent
    3. Temporal consistency - IC-IR style (mean / std) across time
    4. Minimum survivor count - ensure enough samples for learning
    
    Returns LearnabilityMetrics with composite score.
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
    
    # Check class diversity in both splits
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
        # Train weak learner
        train_data = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': 4,
            'num_leaves': 15,
            'learning_rate': 0.05,
            'verbose': -1,
            'verbosity': -1,  # SILENCE WARNINGS
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
        
        # 1. Feature importance magnitude
        importances = model.feature_importance(importance_type='gain')
        metrics.feature_importance_sum = float(np.sum(importances))
        
        # 2. AUC on early validation
        preds_early = model.predict(X_val)
        try:
            auc_early = roc_auc_score(y_val, preds_early)
            metrics.auc_early = float(auc_early)
        except Exception:
            metrics.auc_early = 0.5
        
        # 3. AUC on late split (out-of-time validation)
        preds_late = model.predict(X_late)
        try:
            auc_late = roc_auc_score(y_late, preds_late)
            metrics.auc_late = float(auc_late)
        except Exception:
            metrics.auc_late = 0.5
        
        # 4. Temporal stability: penalize large drops between early and late
        metrics.auc_stability = abs(metrics.auc_early - metrics.auc_late)
        
        # 5. Temporal consistency: IC-IR style
        # If both AUC lifts are positive and stable, high consistency
        lift_early = metrics.auc_early - 0.5
        lift_late = metrics.auc_late - 0.5
        mean_lift = (lift_early + lift_late) / 2.0
        std_lift = np.std([lift_early, lift_late]) + 0.01
        metrics.temporal_consistency = mean_lift / std_lift
        
        # 6. Composite learnability score
        # Components:
        # - Feature importance (normalized, capped)
        # - AUC lift (mean of early and late)
        # - Stability bonus (low difference = bonus)
        # - Temporal consistency
        
        importance_score = min(1.0, metrics.feature_importance_sum / 1000.0)  # Normalize
        auc_lift_score = (mean_lift + 0.5) * 2.0  # Scale to [0, 2] roughly
        stability_bonus = max(0, 0.2 - metrics.auc_stability)  # Bonus if stable
        consistency_score = max(0, metrics.temporal_consistency)
        
        # Weighted combination
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
    # Tunable Thresholds (relaxed defaults for robustness)
    default_min_survival: float = 0.005,
    tail_min_survival: float = 0.005,
    min_uniqueness: float = 0.15,
    min_survivors_absolute: int = 50,         # Minimum absolute survivor count
    min_learnability_score: float = 0.15,     # Minimum composite learnability
    max_temporal_instability: float = 0.20,   # Max AUC difference between early/late (relaxed from 0.15)
) -> GateDiagnostics:
    """
    Enhanced diagnostics with learnability-based gating.
    
    Gates:
    1. Survival Rate - minimum percentage of events surviving
    2. Uniqueness - minimum average uniqueness (independent samples)
    3. Absolute Survivors - minimum count for learning
    4. Learnability Score - composite of feature importance + temporal stability
    5. Temporal Stability - AUC shouldn't drop significantly in late period
    """
    reasons = []
    is_passing = True
    learnability = None
    
    # 1. Survival Rate Gate
    current_min_survival = tail_min_survival if geometry.is_tail else default_min_survival
    rate = len(survivor_ids) / len(events_df) if len(events_df) > 0 else 0.0
    
    if rate < current_min_survival:
        is_passing = False
        reasons.append(f"Low Survival ({rate:.2%} < {current_min_survival:.2%})")
        
    # 2. Uniqueness Gate - ensures independent samples for learning
    avg_u = get_average_uniqueness(survivor_ids, events_df)
    if avg_u < min_uniqueness:
        is_passing = False
        reasons.append(f"Low Uniqueness ({avg_u:.2f} < {min_uniqueness})")
    
    # 3. Absolute Survivors Gate - ensures enough samples
    n_survivors = len(survivor_ids)
    if n_survivors < min_survivors_absolute:
        is_passing = False
        reasons.append(f"Too Few Survivors ({n_survivors} < {min_survivors_absolute})")
    
    # 4 & 5. Learnability Gates - feature importance + temporal stability
    if features_df is not None and not features_df.empty:
        learnability = compute_learnability_metrics(
            survivor_ids=set(survivor_ids),
            all_event_ids=list(events_df.index),
            features_df=features_df,
            min_survivors_absolute=min_survivors_absolute,
        )
        
        # Gate on composite learnability score
        if learnability.composite_score < min_learnability_score:
            is_passing = False
            reasons.append(f"Low Learnability ({learnability.composite_score:.3f} < {min_learnability_score})")
        
        # Gate on temporal stability (penalize regime-dependent geometries)
        if learnability.auc_stability > max_temporal_instability:
            is_passing = False
            reasons.append(f"Temporal Instability ({learnability.auc_stability:.3f} > {max_temporal_instability})")
    
    # Legacy metrics from fold_metrics (for backward compatibility)
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
    min_positive_samples: int = 25,  # Minimum positives needed
    min_informative_features: int = 3,  # Minimum features with variance
    variance_threshold: float = 1e-6  # Drop columns with var < this
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Trains a Weak Learner (max depth = 3) using TradingFocalLoss.
    Target: 1 if event is in survivor_ids, 0 otherwise.
    Returns: model, predictions, separation_metrics
    
    Pre-training checks:
    - Drops features with variance < threshold
    - Requires minimum positive samples in training set
    - Requires minimum informative features
    """
    default_metrics = {'auc': 0.5, 'ks': 0.0, 'entropy': 1.0}
    
    if features_df.empty:
        return None, np.zeros(len(all_event_ids)), default_metrics

    target = pd.Series(0, index=all_event_ids)
    target.loc[list(survivor_ids)] = 1
    
    X = features_df.loc[all_event_ids].copy()
    y = target.loc[all_event_ids]
    
    # PRE-CHECK 1: Drop low-variance features
    variances = X.var()
    informative_cols = variances[variances >= variance_threshold].index.tolist()
    
    if len(informative_cols) < min_informative_features:
        logger.info(f"Model skipped: only {len(informative_cols)} informative features (need {min_informative_features})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    X = X[informative_cols]
    
    # Validation split for metrics (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # PRE-CHECK 2: Minimum class diversity
    if len(np.unique(y_train)) < 2:
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    # PRE-CHECK 3: Minimum positive AND negative samples
    n_positives = int(y_train.sum())
    n_negatives = len(y_train) - n_positives
    if n_positives < min_positive_samples:
        logger.info(f"Model skipped: only {n_positives} positive samples (need {min_positive_samples})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    if n_negatives < min_positive_samples:
        logger.info(f"Model skipped: only {n_negatives} negative samples (need {min_positive_samples})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    # PRE-CHECK 4: Check variance in TRAINING split (not full data)
    # This catches cases where features are informative overall but constant in train
    train_variances = X_train.var()
    informative_train_cols = train_variances[train_variances > 1e-6].index.tolist()
    
    if len(informative_train_cols) < min_informative_features:
        logger.info(f"Model skipped: only {len(informative_train_cols)} train-informative features (need {min_informative_features})")
        return None, np.zeros(len(all_event_ids)), default_metrics
    
    # Keep only columns with variance in train split
    X_train = X_train[informative_train_cols]
    X_val = X_val[informative_train_cols]
    X = X[informative_train_cols]
    
    # Cast to float32 for LightGBM efficiency
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X = X.astype(np.float32)

    # PRE-CHECK 5: Removed covariance check (Fix 19)
    # LightGBM handles collinear features well; removing strict condition number check.


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
        'metric': ['auc', 'average_precision'], # Monitor AUC and PR-AUC (De Prado)
        'max_depth': 3, # Weak Learner Constraint
        'verbose': -1,
        'verbosity': -1, # SILENCE WARNINGS
        'num_leaves': 7, # 2^3 - 1
        'learning_rate': 0.05
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(period=0) # Disable print
        ]
    )
    
    # Predict on Validation set for metrics
    preds_val_raw = model.predict(X_val)
    preds_val_prob = 1.0 / (1.0 + np.exp(-preds_val_raw))

    # Predict on Full set for pruning correlation
    preds_full_raw = model.predict(X)
    preds_full_prob = 1.0 / (1.0 + np.exp(-preds_full_raw))

    # AUC Lift (Validation)
    # Baseline is naive prevalence
    prevalence = y_val.mean()
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_val = roc_auc_score(y_val, preds_val_prob)
        
        # Note: AUC < 0.5 indicates either:
        # 1. Random noise dominating weak signal (most common with shallow trees)
        # 2. Temporal regime shift between train/val
        # We use ABSOLUTE lift to measure discriminative power, but do NOT flip predictions
        # because "inverse signal" in this context is just noise, not real.
        if auc_val < 0.5:
            logger.debug(f"Model shows anti-correlation AUC={auc_val:.3f} (likely noise, not inverse signal)")
        
        auc_lift = abs(auc_val - 0.5)  # Absolute lift measures discriminative power

        # De Prado: PR-AUC is more informative for imbalanced classes
        pr_val = average_precision_score(y_val, preds_val_prob)
        pr_lift = pr_val - prevalence  # Can be negative if model is worse than random
    except:
        auc_lift = 0.0
        pr_lift = 0.0

    # Metrics
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
