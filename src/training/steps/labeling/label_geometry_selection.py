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
MAX_HORIZON_BARS = 24       # 6h at 15m timeframe
MIN_SL_PCT = 0.004          # 0.4% floor
MAX_TP_PCT = 0.05           # 5% ceiling
MIN_TP_SL_RATIO = 1.5       # TP >= 1.5 * SL (positive expectancy)
MAX_FINAL_GEOMETRIES = 10   # Target output count
MIN_GEOMETRY_DISTANCE = 0.10  # Normalized distance threshold for deduplication

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
class GateDiagnostics:
    """Detailed report on why a geometry passed or failed."""
    passed: bool
    survival_rate: float
    avg_uniqueness: float
    avg_auc_lift: float
    avg_pr_lift: float
    ks_stat: float
    entropy_reduction: float
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


def filter_informative_features(features_df: pd.DataFrame, event_ids: pd.Index, variance_threshold: float = 1e-8) -> Optional[pd.DataFrame]:
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
    So we only filter by horizon and min_ratio here.
    """
    valid = []
    for g in candidates:
        # 1. Horizon constraint (max 6h = 24 bars)
        if g.horizon > MAX_HORIZON_BARS:
            continue
        
        # 2. TP/SL ratio constraint (min 1.5x for positive expectancy)
        if g.min_ratio < MIN_TP_SL_RATIO:
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


# --- 5. Diagnostics-First Gates ---

def run_diagnostics_gates(
    survivor_ids: list,
    events_df: pd.DataFrame,
    fold_metrics: dict,
    geometry: Geometry,
    # Tunable Thresholds (relaxed defaults for robustness)
    default_min_survival: float = 0.005,  # Reduced from 0.01
    tail_min_survival: float = 0.005,
    min_uniqueness: float = 0.15,         # Reduced from 0.4 to match fallback
    min_auc_lift: float = 0.01,           # Adjusted to 0.01 as requested (Requires AUC >= 0.51)
    min_pr_lift: float = 0.0,
) -> GateDiagnostics:
    
    reasons = []
    is_passing = True
    
    # 1. Survival Rate Gate
    current_min_survival = tail_min_survival if geometry.is_tail else default_min_survival
    
    rate = len(survivor_ids) / len(events_df)
    if rate < current_min_survival:
        is_passing = False
        reasons.append(f"Low Survival ({rate:.2%} < {current_min_survival:.2%})")
        
    # 2. Uniqueness Gate
    avg_u = get_average_uniqueness(survivor_ids, events_df)
    if avg_u < min_uniqueness:
        is_passing = False
        reasons.append(f"Low Uniqueness ({avg_u:.2f} < {min_uniqueness})")
    
    # 3. Holding Time Gate - use real duration_bars if available
    subset = events_df.loc[survivor_ids]
    duration_col = 'duration_bars' if 'duration_bars' in subset.columns else 'duration'
    if duration_col in subset.columns:
        max_duration = events_df[duration_col].max()
        if subset[duration_col].quantile(0.95) >= max_duration * 0.99:
            # reasons.append("Warning: Hits Max Duration Limit frequently")
            # is_passing = False  # DO NOT REJECT BASED ON HOLDING TIME warning
            pass

    # 4. Fold Persistence / Learnability Gate
    avg_auc = 0.0
    avg_pr_lift = 0.0
    ks_stat = 0.0
    entropy_val = 1.0

    if fold_metrics:
        # fold_metrics is a dict with keys like 'auc', 'ks', 'entropy'
        aucs = [m.get('auc_lift', 0.0) for m in fold_metrics.values() if isinstance(m, dict)]
        if aucs:
            avg_auc = np.mean(aucs)
            # Gate on AUC
            if avg_auc < min_auc_lift:
                is_passing = False
                reasons.append(f"Low Learnability (AUC Lift {avg_auc:.3f})")

        prs = [m.get('pr_lift', 0.0) for m in fold_metrics.values() if isinstance(m, dict)]
        if prs:
            avg_pr_lift = np.mean(prs)
            # Gate on PR Lift - De Prado: Precision-Recall is critical for imbalanced datasets
            if avg_pr_lift < min_pr_lift:
                is_passing = False
                reasons.append(f"Low Precision Lift ({avg_pr_lift:.3f})")

        # We might not have KS/Entropy yet if this is pre-training check.
        # If we do (post-training check):
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
        entropy_reduction=(1.0 - entropy_val), # Higher is better
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
        'num_leaves': 7, # 2^3 - 1
        'learning_rate': 0.05
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(20, verbose=False)]
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
    horizons = [8, 12, 16, 20, 24]

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
    filtered_features = filter_informative_features(features_df, df.index, variance_threshold=1e-8)
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
    for h in horizons:
        if h not in thresholds_map: continue
        for q in quantiles:
            for a in [0.5, 1.0, 1.5]:
                for b in [0.5, 1.0, 1.5, 2.0]:
                    # Only generate candidates with min_ratio >= MIN_TP_SL_RATIO
                    for mr in [1.5, 2.0, 2.5, 3.0]:
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
        thresh = thresholds_map[h][geom.sl_quantile]
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
        
        # FALLBACK: Accept geometry based on survival rate if LGBM fails
        # Require both good survival AND uniqueness before accepting without model
        if model is None:
            probe_output = run_logistic_probe(filtered_features, target_series)
            if probe_output is not None:
                probe_preds, probe_metrics = probe_output
                diag = run_diagnostics_gates(
                    list(survivor_ids),
                    df,
                    {0: probe_metrics},
                    geom
                )
                if diag.passed:
                    separation_score = (probe_metrics['ks_stat'] + (1.0 - probe_metrics['entropy'])) / 2.0
                    resolved_geom = replace(geom, sl_sigma=float(thresh))
                    accepted_candidates.append({
                        'geometry': resolved_geom,
                        'survivors': survivor_ids,
                        'preds': probe_preds,
                        'survival_rate': diag.survival_rate,
                        'metrics': probe_metrics,
                        'separation_score': separation_score,
                        'model': None
                    })
                    continue
            
            # Compute uniqueness before fallback decision
            avg_uniqueness = get_average_uniqueness(survivor_ids, df)
            
            if survival_rate >= 0.10 and avg_uniqueness >= 0.25:
                logger.info(f"Fallback: Accepting geometry with survival={survival_rate:.2%}, uniq={avg_uniqueness:.2f} (no model)")
                resolved_geom = replace(geom, sl_sigma=float(thresh))
                accepted_candidates.append({
                    'geometry': resolved_geom,
                    'survivors': survivor_ids,
                    'preds': np.full(len(df), 0.5),  # Flat predictions
                    'survival_rate': survival_rate,
                    'metrics': {'auc_lift': 0.0, 'pr_lift': 0.0, 'ks_stat': 0.0, 'entropy': 1.0},
                    'separation_score': survival_rate,  # Use survival as score
                    'model': None
                })
            else:
                logger.info(f"Fallback rejected: survival={survival_rate:.2%}, uniq={avg_uniqueness:.2f} (need >=5%, >=0.15)")
            continue

        # C. Run Diagnostics Gates (Post-Training check included)
        diag = run_diagnostics_gates(
            list(survivor_ids),
            df,
            {0: metrics},
            geom
        )
        
        # Accept if gates pass - NO RELAXED MODE
        # Geometries must meet all diagnostic thresholds
        if not diag.passed:
            logger.info(f"Gate rejected: survival={survival_rate:.2%}, uniq={diag.avg_uniqueness:.2f}, reasons={diag.reasons}")
            continue

        # Store Candidate
        # We store the "Separation Score" for ranking
        # Combine KS and Entropy Reduction
        # KS is [0,1], Entropy Reduction is [0,1]
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
            'model': model
        })

    # E. Prediction-Correlation Pruning
    # "Optimize for Separation... Tie breaker: keep the one with higher survival rate?"
    # Let's prioritize Separation Score first, then Survival.
    
    logger.info(f"Pruning {len(accepted_candidates)} accepted geometries based on correlation...")
    
    final_selection = []
    
    # Sort by Separation Score descending
    accepted_candidates.sort(key=lambda x: (x['separation_score'], x['survival_rate']), reverse=True)
    
    if accepted_candidates:
        all_preds = np.array([c['preds'] for c in accepted_candidates])
        
        if len(accepted_candidates) > 1:
            corr_matrix = np.corrcoef(all_preds)
        else:
            corr_matrix = np.array([[1.0]])
            
        is_dropped = [False] * len(accepted_candidates)
        
        for i in range(len(accepted_candidates)):
            if is_dropped[i]:
                continue
            
            final_selection.append(accepted_candidates[i])
            
            for j in range(i + 1, len(accepted_candidates)):
                if is_dropped[j]:
                    continue
                
                corr = corr_matrix[i, j]
                if abs(corr) > 0.9:
                    is_dropped[j] = True
                    logger.info(f"Dropped {accepted_candidates[j]['geometry'].archetype} (Sep: {accepted_candidates[j]['separation_score']:.3f}) due to correlation {corr:.2f} with {accepted_candidates[i]['geometry'].archetype} (Sep: {accepted_candidates[i]['separation_score']:.3f})")

    result = []
    for item in final_selection[:MAX_FINAL_GEOMETRIES]:  # Limit to target count
        # Combine metrics into a single dict for return
        out_metrics = item['metrics'].copy()
        out_metrics['separation_score'] = item['separation_score']
        out_metrics['survival_rate'] = item['survival_rate']
        result.append((item['geometry'], item['survivors'], out_metrics))
        
    best_score_str = f"{final_selection[0]['separation_score']:.3f}" if final_selection else "N/A"
    logger.info(f"Final Selection: {len(result)} geometries (Best Separation Score: {best_score_str}, Max: {MAX_FINAL_GEOMETRIES})")
    return result
