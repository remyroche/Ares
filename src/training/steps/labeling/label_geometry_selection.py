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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Economic Constraints for Geometry Pre-Filtering ---
MAX_HORIZON_BARS = 48       # 12h at 15m timeframe
MIN_SL_PCT = 0.004          # 0.4% floor
MAX_TP_PCT = 0.05           # 5% ceiling
MIN_TP_SL_RATIO = 1.5       # TP >= 1.5 * SL (positive expectancy)
MIN_SL_SIGMA = 0.5          # Minimum stop-loss in sigma units (prevent too-tight stops)
MAX_FINAL_GEOMETRIES = 30    # Increased to allow more diverse candidates
MIN_GEOMETRY_DISTANCE = 0.05  # Relaxed distance threshold for deduplication (was 0.15)

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
    Vectorized for performance.

    Args:
        events: List of Event objects.
        horizons: List of horizons to calculate metrics for (e.g. [24, 48, 120]).
                  If None, defaults to [8, 12, 16, 20, 30, 40].
    """
    if horizons is None:
        horizons = [8, 12, 16, 20, 30, 40]

    data = []
    for e in events:
        full_path = e.returns_path * e.direction
        max_len = len(full_path)
        
        # FIX: Use REAL duration from exit_idx - entry_idx, not truncated path length
        real_duration_bars = e.exit_idx - e.entry_idx
        
        row = {
            'id': e.id,
            'entry_idx': e.entry_idx,
            'exit_idx': e.exit_idx,
            'direction': e.direction,  # FIX: Add direction for dedup
            'duration_bars': real_duration_bars,  # FIX: Real holding time
            'sigma': e.sigma,
        }

        for h in horizons:
            # Slice path to horizon
            # Note: returns_path is 0-based from entry
            limit = min(max_len, h)
            path = full_path[:limit]

            # Duration for THIS horizon (capped at horizon or actual path)
            duration_h = max(1, limit)

            raw_mae = -np.min(path) if len(path) > 0 else 0.0
            raw_mfe = np.max(path) if len(path) > 0 else 0.0

            # Standard normalization
            norm_mae = raw_mae / e.sigma
            norm_mfe = raw_mfe / e.sigma

            # Time-scaled normalization (Condition on Holding Time)
            # Assuming volatility scales with sqrt(t)
            sqrt_t = np.sqrt(duration_h)
            time_scaled_mae = raw_mae / (e.sigma * sqrt_t)
            time_scaled_mfe = raw_mfe / (e.sigma * sqrt_t)

            row[f'norm_mae_{h}'] = norm_mae
            row[f'norm_mfe_{h}'] = norm_mfe
            row[f'time_scaled_mae_{h}'] = time_scaled_mae
            row[f'time_scaled_mfe_{h}'] = time_scaled_mfe

            # Legacy fields (map to max horizon)
            if h == max(horizons):
                row['norm_mae'] = norm_mae
                row['norm_mfe'] = norm_mfe
                row['duration'] = real_duration_bars  # FIX: Use real duration here too
                row['time_scaled_mae'] = time_scaled_mae
                row['time_scaled_mfe'] = time_scaled_mfe

        data.append(row)
    
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
    except Exception:
        return None
    
    try:
        clf = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=500,
            class_weight='balanced'
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
    
    keep_indices = []
    last_entry_by_dir = {1: -float('inf'), -1: -float('inf')}
    
    for idx, row in df.iterrows():
        entry = row['entry_idx']
        direction = row.get('direction', 0)
        
        # Only keep if far enough from last event in same direction
        if entry - last_entry_by_dir.get(direction, -float('inf')) >= min_gap_bars:
            keep_indices.append(idx)
            last_entry_by_dir[direction] = entry
    
    result = events_df.loc[keep_indices]
    
    if len(result) < len(events_df):
        logger.info(f"Event dedup: {len(events_df)} → {len(result)} events (min_gap={min_gap_bars} bars)")
    
    return result

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
    min_uniqueness: float = 0.05,            # Relaxed from 0.15
    min_survivors_absolute: int = 50,         # Minimum absolute survivor count
    min_learnability_score: float = 0.05,     # Relaxed from 0.15 (Allow Weak Learners)
    max_temporal_instability: float = 0.30,   # Relaxed from 0.20
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

def select_geometries(
    events: List[Event],
    fold_metrics_map: Dict,
    features_df: pd.DataFrame
) -> List[Tuple[Geometry, Set[int], Dict[str, Any]]]:
    
    # Define horizons to sweep (constrained to max 6h = 24 bars)
    horizons = [12, 24, 48]

    rng = np.random.default_rng()

    logger.info(f"Vectorizing {len(events)} events for horizons {horizons}...")
    df = events_to_dataframe(events, horizons=horizons)
    if df.empty:
        logger.warning("No events to process.")
        return []
    
    # FIX: De-duplicate events to reduce artificial overlap from CUSUM bursts
    # This improves uniqueness scores by removing redundant events
    original_count = len(df)
    df = deduplicate_events(df, min_gap_bars=6)  # More spacing to reduce overlap
    if len(df) < original_count:
        logger.info(f"Events after dedup: {original_count} → {len(df)}")

    # Align and filter feature matrix to deduplicated events
    filtered_features = filter_informative_features(features_df, df.index, variance_threshold=1e-12)
    if filtered_features is None or filtered_features.empty:
        logger.warning("Feature matrix unavailable after deduplication; skipping geometry selection.")
        return []
    
    # Calculate quantile thresholds for MAE per horizon
    thresholds_map = {}
    quantiles = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for h in horizons:
        col_name = f'norm_mae_{h}'
        if col_name in df.columns:
            series = df[col_name]
            thresholds_map[h] = {q: series.quantile(q) for q in quantiles}
        else:
            logger.warning(f"Missing column {col_name}, skipping horizon {h}")

    # Generate candidates (only with valid min_ratio >= 1.5 upfront)
    candidates = []
    base_alphas = [0.3, 0.5, 1.0, 1.5, 2.0]           # Expanded from [0.5, 1.0, 1.5]
    base_betas = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Expanded from [0.5, 1.0, 1.5, 2.0]
    base_min_ratios = [1.5, 2.0, 2.5, 3.0]              # Expanded from [1.5, 2.0]

    for h in horizons:
        if h not in thresholds_map:
            continue
        for q in quantiles:
            for a in base_alphas:
                for b in base_betas:
                    for mr in base_min_ratios:
                        candidates.append(Geometry(sl_quantile=q, alpha=a, beta=b, min_ratio=mr, horizon=h))

    # Randomized jittered combinations to avoid identical survivor sets
    jitter_count = int(min(800, len(candidates))) or 500  # Increased for diversity
    for _ in range(jitter_count):
        h = int(rng.choice(horizons))
        q = float(np.clip(rng.normal(loc=0.5, scale=0.25), 0.1, 0.9))  # Expanded range
        a = float(np.clip(rng.normal(loc=1.0, scale=0.6), 0.2, 2.2))   # Expanded range  
        b = float(np.clip(rng.normal(loc=1.2, scale=0.7), 0.3, 2.8))   # Expanded range
        mr = float(rng.choice(base_min_ratios))
        candidates.append(Geometry(sl_quantile=q, alpha=a, beta=b, min_ratio=mr, horizon=h))

    # Deduplicate exact matches
    candidates = list(set(candidates))
    logger.info(f"Generated {len(candidates)} raw candidates")
    
    # === PRE-FILTERING (Key Optimization) ===
    # 1. Apply hard economic constraints (SL floor, TP ceiling)
    candidates = apply_hard_constraints(candidates, thresholds_map)
    logger.info(f"After hard constraints: {len(candidates)} candidates")
    
    # 2. Distance-based deduplication to reduce redundant LGBM training
    candidates = deduplicate_by_distance(candidates)
    
    if not candidates:
        logger.warning("No candidates passed pre-filtering constraints!")
        return []
    
    accepted_candidates = []
    
    logger.info(f"Training LGBM on {len(candidates)} pre-filtered candidates...")
    
    for geom in candidates:
        h = geom.horizon
        col_mae = f'norm_mae_{h}'
        col_mfe = f'norm_mfe_{h}'

        # B. Apply Quantile-Based Filters using horizon-specific norm_mae
        # Find nearest available quantile in thresholds_map for jittered candidates
        available_quantiles = sorted(thresholds_map[h].keys())
        nearest_q = min(available_quantiles, key=lambda x: abs(x - geom.sl_quantile))
        thresh = thresholds_map[h][nearest_q]
        mask_sl = df[col_mae] <= thresh
        
        # Score Calculation
        score = (df[col_mfe] ** geom.beta) / ((df[col_mae] + 1e-6) ** geom.alpha)
        mask_score = score >= geom.min_ratio
        
        survivors_df = df[mask_sl & mask_score]
        survivor_ids = set(survivors_df.index)
        
        if not survivor_ids:
            continue
        
        # Calculate survival rate for fallback
        survival_rate = len(survivor_ids) / len(df)
        
        target_series = pd.Series(0, index=df.index)
        target_series.loc[list(survivor_ids)] = 1
        
        # D. Train Weak Learner First to get Separation Metrics
        model, preds, metrics = train_model_for_geometry(
            survivor_ids,
            list(df.index),
            filtered_features
        )
        
        # DIAGNOSTIC: Use logistic probe for telemetry, not hard gating
        # Primary gating is via learnability composite score
        if model is None:
            # Log probe metrics for diagnostic purposes
            probe_output = run_logistic_probe(filtered_features, target_series)
            probe_auc = 0.0
            probe_preds = np.full(len(df), 0.5)
            if probe_output is not None:
                probe_preds, probe_metrics = probe_output
                probe_auc = probe_metrics.get('auc_lift', 0.0)
                logger.debug(f"Probe diagnostic: AUC_lift={probe_auc:.3f}, PR_lift={probe_metrics.get('pr_lift', 0.0):.3f}")
            
            # Run gates with learnability check (probe results are informational only)
            diag = run_diagnostics_gates(
                list(survivor_ids),
                df,
                {},  # No fold metrics - rely on learnability computation
                geom,
                features_df=filtered_features,
            )
            
            # Compute economic metrics for tie-breaking
            # NOTE: These require realized returns which we compute from MFE
            survivor_mfe = df.loc[list(survivor_ids), f'norm_mfe_{geom.horizon}'] if f'norm_mfe_{geom.horizon}' in df.columns else pd.Series(dtype=float)
            survivor_mae = df.loc[list(survivor_ids), f'norm_mae_{geom.horizon}'] if f'norm_mae_{geom.horizon}' in df.columns else pd.Series(dtype=float)
            
            # Sharpe proxy = mean(MFE - MAE) / std(MFE - MAE)
            if len(survivor_mfe) > 10:
                returns_proxy = survivor_mfe - survivor_mae
                sharpe_proxy = float(returns_proxy.mean() / (returns_proxy.std() + 1e-6))
                win_rate = float((returns_proxy > 0).mean())
            else:
                sharpe_proxy = 0.0
                win_rate = 0.5
            
            if diag.passed:
                learn_score = diag.learnability.composite_score if diag.learnability else 0.0
                resolved_geom = replace(geom, sl_sigma=float(thresh))
                accepted_candidates.append({
                    'geometry': resolved_geom,
                    'survivors': survivor_ids,
                    'preds': probe_preds,
                    'survival_rate': diag.survival_rate,
                    'metrics': {'auc_lift': probe_auc, 'sharpe_proxy': sharpe_proxy, 'win_rate': win_rate},
                    'separation_score': learn_score,
                    'sharpe_proxy': sharpe_proxy,
                    'win_rate': win_rate,
                    'learnability': diag.learnability,
                    'model': None
                })
                continue
            else:
                # Log rejection but don't require probe success
                logger.info(f"Gate rejected (no model): survival={survival_rate:.2%}, uniq={diag.avg_uniqueness:.2f}, reasons={diag.reasons}")
            continue

        # C. Run Diagnostics Gates with Learnability Check
        diag = run_diagnostics_gates(
            list(survivor_ids),
            df,
            {0: metrics},
            geom,
            features_df=filtered_features,
        )
        
        # Accept if gates pass - learnability gates now handle quality checks
        if not diag.passed:
            logger.info(f"Gate rejected: survival={survival_rate:.2%}, uniq={diag.avg_uniqueness:.2f}, reasons={diag.reasons}")
            continue

        # Compute economic metrics for tie-breaking
        h = geom.horizon
        survivor_mfe = df.loc[list(survivor_ids), f'norm_mfe_{h}'] if f'norm_mfe_{h}' in df.columns else pd.Series(dtype=float)
        survivor_mae = df.loc[list(survivor_ids), f'norm_mae_{h}'] if f'norm_mae_{h}' in df.columns else pd.Series(dtype=float)
        
        if len(survivor_mfe) > 10:
            returns_proxy = survivor_mfe - survivor_mae
            sharpe_proxy = float(returns_proxy.mean() / (returns_proxy.std() + 1e-6))
            win_rate = float((returns_proxy > 0).mean())
        else:
            sharpe_proxy = 0.0
            win_rate = 0.5

        # Store Candidate with learnability-based ranking
        # Primary: learnability.composite_score (if available)
        # Fallback: KS + entropy reduction
        if diag.learnability and diag.learnability.composite_score > 0:
            separation_score = diag.learnability.composite_score
        else:
            separation_score = (metrics['ks_stat'] + (1.0 - metrics['entropy'])) / 2.0
        
        # Create resolved geometry with concrete sl_sigma
        resolved_geom = replace(geom, sl_sigma=float(thresh))

        accepted_candidates.append({
            'geometry': resolved_geom,
            'survivors': survivor_ids,
            'preds': preds,
            'survival_rate': diag.survival_rate,
            'metrics': metrics,
            'separation_score': separation_score,
            'sharpe_proxy': sharpe_proxy,
            'win_rate': win_rate,
            'learnability': diag.learnability,
            'model': model
        })

    # E. Prediction-Correlation Pruning
    # Keep geometries with uncorrelated predictions (< 0.85 corr) for ensemble diversity
    # Rank by: learnability score (primary), then economic tie-breakers (sharpe, win_rate)
    
    logger.info(f"Pruning {len(accepted_candidates)} accepted geometries based on prediction correlation...")
    
    final_selection = []
    
    # Sort by: separation_score (learnability), sharpe_proxy, win_rate, survival_rate
    accepted_candidates.sort(
        key=lambda x: (x['separation_score'], x.get('sharpe_proxy', 0), x.get('win_rate', 0.5), x['survival_rate']),
        reverse=True
    )
    
    # Log top candidates with learnability details
    for i, cand in enumerate(accepted_candidates[:10]):
        learn = cand.get('learnability')
        if learn:
            logger.info(f"  Candidate {i+1}: {cand['geometry'].archetype} (H={cand['geometry'].horizon}) | "
                       f"Score={cand['separation_score']:.3f}, "
                       f"AUC_early={learn.auc_early:.3f}, AUC_late={learn.auc_late:.3f}, "
                       f"Stability={learn.auc_stability:.3f}, FeatImp={learn.feature_importance_sum:.1f}")
    
    if accepted_candidates:
        all_preds = np.array([c['preds'] for c in accepted_candidates])
        
        if len(accepted_candidates) > 1:
            # Handle NaN in predictions by filling with 0.5
            all_preds = np.nan_to_num(all_preds, nan=0.5)
            corr_matrix = np.corrcoef(all_preds)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        else:
            corr_matrix = np.array([[1.0]])
            
    
            
    # --- Diversity-Aware Selection (Horizon Round-Robin) ---
    # Instead of picking top N globally (which might favor one horizon),
    # we pick round-robin from each horizon bucket to ensure time-diversity.
    
    # 1. Bucket by Horizon
    by_horizon = {}
    for i, cand in enumerate(accepted_candidates):
        cand['_idx'] = i  # Store original index for correlation matrix lookups
        h = cand['geometry'].horizon
        if h not in by_horizon:
            by_horizon[h] = []
        by_horizon[h].append(cand)
        
    # 2. Sort within buckets by score
    for h in by_horizon:
        by_horizon[h].sort(
            key=lambda x: (x['separation_score'], x.get('sharpe_proxy', 0), x.get('win_rate', 0.5), x['survival_rate']),
            reverse=True
        )

    # 3. Round-Robin Selection
    final_selection = []
    active_horizons = sorted(by_horizon.keys())
    horizon_counts = {h: 0 for h in active_horizons}
    
    # RELAX: Increased from 0.85 to 0.95 to allow more "cousin" geometries
    # if they are high quality.
    CORR_THRESHOLD = 0.95
    
    while len(final_selection) < MAX_FINAL_GEOMETRIES and active_horizons:
        made_selection = False
        
        # Iterate through horizons (copy list to allow removal)
        for h in list(active_horizons):
            if not by_horizon[h] or horizon_counts[h] >= 4:
                if h in active_horizons:
                    active_horizons.remove(h)
                continue
                
            # Try top candidate from this horizon
            cand = by_horizon[h].pop(0)
            
            # Check Correlation against ALL currently selected
            is_correlated = False
            drop_msg = ""
            
            for selected in final_selection:
                # Use precomputed correlation matrix
                c_val = corr_matrix[cand['_idx'], selected['_idx']]
                
                if abs(c_val) > CORR_THRESHOLD:
                    is_correlated = True
                    drop_msg = f"corr {c_val:.2f} with {selected['geometry'].archetype} (H={selected['geometry'].horizon})"
                    break
            
            # Check parameter space diversity
            param_similar = False
            selected_geoms = [s['geometry'] for s in final_selection]
            if parameter_diversity_penalty(cand['geometry'], selected_geoms):
                param_similar = True
                drop_msg = f"parameter similarity with selected geometries"
            
            if not is_correlated and not param_similar:
                final_selection.append(cand)
                horizon_counts[h] += 1
                made_selection = True
                # Check limit immediately
                if len(final_selection) >= MAX_FINAL_GEOMETRIES:
                    break
            else:
                logger.info(f"Dropped {cand['geometry'].archetype} (H={cand['geometry'].horizon}, Score={cand['separation_score']:.3f}): {drop_msg}")
        
        # Determine if we should stop
        if len(final_selection) >= MAX_FINAL_GEOMETRIES:
            break
            
        if not made_selection:
            # If we iterate through all active horizons and pick nothing (all correlated),
            # we are done. Remaining candidates are all redundant.
            break

    result = []
    # Final selection: keep only top 6 geometries per horizon to ensure diversity
    horizon_geometries = {}
    for item in final_selection[:MAX_FINAL_GEOMETRIES]:
        h = item['geometry'].horizon
        if h not in horizon_geometries:
            horizon_geometries[h] = []
        horizon_geometries[h].append(item)
    
    # Keep top 6 per horizon (already sorted by score within each horizon)
    for h, items in horizon_geometries.items():
        for item in items[:6]:  # Top 6 per horizon (Increased from 2)
            # Pass full metadata for downstream analysis
            meta = {k: v for k, v in item.items() if k not in ['geometry', 'survivors', 'model', 'preds']}
            result.append((item['geometry'], item['survivors'], meta))
    
    best_score = final_selection[0]['separation_score'] if final_selection else 0.0
    logger.info(f"Final Selection: {len(result)} geometries (Best Learnability Score: {best_score:.3f})")
    logger.info(f"Horizon distribution: {[(h, len(items[:6])) for h, items in horizon_geometries.items()]}")
    return result
