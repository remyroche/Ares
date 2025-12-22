import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Data Structures ---

@dataclass(frozen=True)
class Geometry:
    sl_sigma: float     # Stop Loss in sigma units
    alpha: float        # Pain penalty (denominator exponent)
    beta: float         # Gain reward (numerator exponent)
    min_ratio: float = 1.0 
    
    @property
    def archetype(self) -> str:
        """Auto-classifies the geometry into a human-readable archetype."""
        if self.sl_sigma < 1.0 and self.beta > 1.0:
            return "Sniper (Tight SL, High Reward)"
        elif self.alpha > 1.0:
            return "Pain Averse (High Penalty for Drawdown)"
        elif self.sl_sigma > 2.0 and self.beta < 0.8:
            return "Deep Value (Loose SL, Low Target)"
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

def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    """
    Converts events to DataFrame and pre-calculates path metrics.
    Vectorized for performance.
    """
    data = []
    for e in events:
        path = e.returns_path * e.direction
        
        duration = e.exit_idx - e.entry_idx
        
        raw_mae = -np.min(path) if len(path) > 0 else 0.0
        raw_mfe = np.max(path) if len(path) > 0 else 0.0
        
        data.append({
            'id': e.id,
            'entry_idx': e.entry_idx,
            'exit_idx': e.exit_idx,
            'duration': duration,
            'sigma': e.sigma,
            'norm_mae': raw_mae / e.sigma if e.sigma > 0 else 0.0,
            'norm_mfe': raw_mfe / e.sigma if e.sigma > 0 else 0.0
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index('id', inplace=True)
    return df

# --- 4. Advanced Metrics (De Prado) ---

def get_average_uniqueness(selected_indices, all_events_df) -> float:
    """
    Calculates Average Uniqueness to prevent 'label p-hacking' via overlaps.
    """
    if not len(selected_indices):
        return 0.0
        
    subset = all_events_df.loc[list(selected_indices)]
    if subset.empty:
        return 0.0
        
    max_idx = int(all_events_df['exit_idx'].max())
    
    concurrency = np.zeros(max_idx + 1)
    for _, row in subset.iterrows():
        start = int(row['entry_idx'])
        end = int(row['exit_idx'])
        concurrency[start:end] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        uniqueness_t = 1.0 / concurrency
        uniqueness_t[~np.isfinite(uniqueness_t)] = 0
    
    event_scores = []
    for _, row in subset.iterrows():
        start = int(row['entry_idx'])
        end = int(row['exit_idx'])
        if end > start:
            u = uniqueness_t[start:end].mean()
            event_scores.append(u)
            
    return np.mean(event_scores) if event_scores else 0.0

def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b: return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

# --- 5. Diagnostics-First Gates ---

def run_diagnostics_gates(
    survivor_ids: list,
    events_df: pd.DataFrame,
    fold_metrics: dict,
    geometry: Geometry,
    # Tunable Thresholds
    default_min_survival: float = 0.15,
    tail_min_survival: float = 0.01,
    min_uniqueness: float = 0.5,
    min_auc_lift: float = 0.02,
) -> GateDiagnostics:
    
    reasons = []
    is_passing = True
    
    # 1. Survival Rate Gate
    # "3.1 min_survival = 0.05 Is Too Low for Default"
    # "allow 0.05 (or 0.01) only for explicitly tagged tail geometries"
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
    
    # 3. Holding Time Gate
    subset = events_df.loc[survivor_ids]
    if subset['duration'].quantile(0.95) >= events_df['duration'].max() * 0.99:
        reasons.append("Warning: Hits Max Duration Limit frequently")

    # 4. Fold Persistence Gate (Meta-Model readiness)
    # "3.3 Fold Persistence Should Use a Majority Rule" -> Code says: if avg_auc < min_auc_lift
    avg_auc = 0.0
    if fold_metrics:
        # Assuming fold_metrics is a dict of metrics per fold
        aucs = [m['auc_lift'] for m in fold_metrics.values()]

        # Majority Rule Logic
        pass_rate = np.mean([a >= min_auc_lift for a in aucs])
        avg_auc = np.mean(aucs) # Keep calculating average for reporting

        if pass_rate < 0.6:
            is_passing = False
            reasons.append(f"Fold Pass Rate {pass_rate:.0%} < 60% (Avg AUC Lift {avg_auc:.3f})")
    else:
        # If no metrics provided, we assume we are in 'discovery' mode
        pass

    return GateDiagnostics(
        passed=is_passing,
        survival_rate=rate,
        avg_uniqueness=avg_u,
        avg_auc_lift=avg_auc,
        reasons=reasons
    )

# --- 6. Model Training ---

def train_model_for_geometry(
    survivor_ids: Set[int],
    all_event_ids: List[int],
    features_df: pd.DataFrame
) -> Tuple[Any, np.ndarray]:
    """
    Trains a simple LGBM model (max depth = 3) using TradingFocalLoss.
    Target: 1 if event is in survivor_ids, 0 otherwise.

    # Models trained here are not evaluated for performance. They are used solely for correlation pruning.
    """
    # Align features and target
    # features_df index should match event ids
    if features_df.empty:
        return None, np.zeros(len(all_event_ids))

    target = pd.Series(0, index=all_event_ids)
    target.loc[list(survivor_ids)] = 1
    
    # Ensure features_df has rows for all_event_ids
    X = features_df.loc[all_event_ids]
    y = target.loc[all_event_ids]
    
    train_data = lgb.Dataset(X, label=y)
    
    focal_loss = TradingFocalLoss(
        gamma_pos=1.5,
        gamma_neg=3.0,
        alpha=None, # Will be calculated data-driven in __call__
        w_cap=3.0,
        label_smoothing=0.02,
        mix=0.5
    )

    params = {
        'objective': focal_loss, 
        'metric': 'binary_logloss',
        'max_depth': 3, # Changed from 4 to 3 as requested
        'verbose': -1,
        'num_leaves': 7, # constrained by depth 3 (2^3 - 1)
        'learning_rate': 0.05
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=50
    )
    
    # Predict
    preds = model.predict(X)
    # Apply sigmoid if custom objective returns raw scores (fobj usually needs raw)
    # With custom fobj, predictions are raw margins.
    preds_proba = 1.0 / (1.0 + np.exp(-preds))
    
    return model, preds_proba

# --- 7. Main Selection Loop ---

def select_geometries(
    events: List[Event],
    fold_metrics_map: Dict,
    features_df: pd.DataFrame
) -> List[Tuple[Geometry, Set[int]]]:
    
    logger.info(f"Vectorizing {len(events)} events...")
    df = events_to_dataframe(events)
    if df.empty:
        logger.warning("No events to process.")
        return []

    # Generate candidates
    candidates = [
        Geometry(sl, a, b, min_ratio=mr) 
        for sl in [0.5, 1.0, 2.0]
        for a in [0.5, 1.0, 1.5]
        for b in [0.5, 1.0, 1.5]
        for mr in [1.0, 1.5, 2.0] 
    ]
    # Deduplicate candidates if any
    candidates = list(set(candidates))
    
    accepted_candidates = [] # List of dicts: {'geom': g, 'survivors': s, 'preds': p, 'survival_rate': r}
    
    logger.info(f"Evaluating {len(candidates)} geometric candidates...")
    
    for geom in candidates:
        # B. Apply Geometry Filters
        mask_sl = df['norm_mae'] <= geom.sl_sigma
        
        score = (df['norm_mfe'] ** geom.beta) / ((df['norm_mae'] + 1e-6) ** geom.alpha)
        mask_score = score >= geom.min_ratio
        
        survivors_df = df[mask_sl & mask_score]
        survivor_ids = set(survivors_df.index)
        
        if not survivor_ids:
            continue

        # C. Run Diagnostics
        diag = run_diagnostics_gates(
            list(survivor_ids),
            df,
            fold_metrics_map.get(geom, {}),
            geom
        )
        
        if not diag.passed:
            continue

        # D. Train Model (The "Weak Learner")
        # "train a simple (max depth = 3) LGBM model... then Add Prediction-Correlation Pruning"
        model, preds = train_model_for_geometry(
            survivor_ids,
            list(df.index),
            features_df
        )
        
        accepted_candidates.append({
            'geometry': geom,
            'survivors': survivor_ids,
            'preds': preds,
            'survival_rate': diag.survival_rate,
            'model': model
        })

    # E. Prediction-Correlation Pruning
    # "After training base models: Compute correlation... Drop geometries with >0.9 correlation"
    
    logger.info(f"Pruning {len(accepted_candidates)} accepted geometries based on correlation...")
    
    final_selection = []
    
    # Sort by survival rate descending (to prioritize high survival in tie-breaking)
    # or process in order.
    # User said: "Correlation pruning tie breaker: keep the one with higher survival rate"
    
    # We can use a greedy approach:
    # 1. Sort candidates by survival rate (descending)
    # 2. Pick top, discard any that are highly correlated (>0.9) with it.
    # 3. Repeat.
    
    accepted_candidates.sort(key=lambda x: x['survival_rate'], reverse=True)
    
    # Using indices to track what is kept
    kept_indices = []
    
    if accepted_candidates:
        # Convert preds to a matrix for fast correlation
        # shape: (n_candidates, n_events)
        all_preds = np.array([c['preds'] for c in accepted_candidates])
        
        # We need correlation between rows.
        # np.corrcoef does that.
        if len(accepted_candidates) > 1:
            corr_matrix = np.corrcoef(all_preds)
        else:
            corr_matrix = np.array([[1.0]])
            
        is_dropped = [False] * len(accepted_candidates)
        
        for i in range(len(accepted_candidates)):
            if is_dropped[i]:
                continue
            
            # Keep candidate i
            final_selection.append(accepted_candidates[i])
            
            # Check for correlations with remaining candidates
            for j in range(i + 1, len(accepted_candidates)):
                if is_dropped[j]:
                    continue
                
                corr = corr_matrix[i, j]
                # Improved: Use absolute correlation
                if abs(corr) > 0.9:
                    # Drop j because i has higher survival rate (since we sorted)
                    is_dropped[j] = True
                    # logger.info(f"Dropped {accepted_candidates[j]['geometry'].archetype} due to correlation {corr:.2f} with {accepted_candidates[i]['geometry'].archetype}")

    # Return formatted list
    
    result = []
    for item in final_selection:
        result.append((item['geometry'], item['survivors']))
        
    logger.info(f"Final Selection: {len(result)} geometries.")
    return result
