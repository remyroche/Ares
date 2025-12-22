import numpy as np
import pandas as pd
from dataclasses import dataclass, field, replace
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
import lightgbm as lgb
from scipy.stats import ks_2samp, entropy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Data Structures ---

@dataclass(frozen=True)
class Geometry:
    sl_quantile: float  # Quantile of MAE distribution (0.0 - 1.0) instead of fixed sigma
    alpha: float        # Pain penalty (denominator exponent)
    beta: float         # Gain reward (numerator exponent)
    min_ratio: float = 1.0 
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

def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    """
    Converts events to DataFrame and pre-calculates path metrics.
    Vectorized for performance.
    """
    data = []
    for e in events:
        path = e.returns_path * e.direction
        
        duration = max(1, e.exit_idx - e.entry_idx)
        
        raw_mae = -np.min(path) if len(path) > 0 else 0.0
        raw_mfe = np.max(path) if len(path) > 0 else 0.0
        
        # Standard normalization
        norm_mae = raw_mae / e.sigma
        norm_mfe = raw_mfe / e.sigma

        # Time-scaled normalization (Condition on Holding Time)
        # Assuming volatility scales with sqrt(t)
        # We normalize by sigma * sqrt(duration) to treat long/short horizons equally
        sqrt_t = np.sqrt(duration)
        time_scaled_mae = raw_mae / (e.sigma * sqrt_t)
        time_scaled_mfe = raw_mfe / (e.sigma * sqrt_t)

        data.append({
            'id': e.id,
            'entry_idx': e.entry_idx,
            'exit_idx': e.exit_idx,
            'duration': duration,
            'sigma': e.sigma,
            'norm_mae': norm_mae,
            'norm_mfe': norm_mfe,
            'time_scaled_mae': time_scaled_mae,
            'time_scaled_mfe': time_scaled_mfe
        })
    
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
    default_min_survival: float = 0.01, # Relaxed from 0.15 to allow quantiles to work (0.2 quantile -> 20% max)
    tail_min_survival: float = 0.005,
    min_uniqueness: float = 0.5,
    min_auc_lift: float = 0.02,
    min_pr_lift: float = 0.0, # Require at least some improvement over baseline precision
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
    
    # 3. Holding Time Gate
    subset = events_df.loc[survivor_ids]
    if subset['duration'].quantile(0.95) >= events_df['duration'].max() * 0.99:
        reasons.append("Warning: Hits Max Duration Limit frequently")

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
    features_df: pd.DataFrame
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Trains a Weak Learner (max depth = 3) using TradingFocalLoss.
    Target: 1 if event is in survivor_ids, 0 otherwise.
    Returns: model, predictions, separation_metrics
    """
    if features_df.empty:
        return None, np.zeros(len(all_event_ids)), {'auc': 0.5, 'ks': 0.0, 'entropy': 1.0}

    target = pd.Series(0, index=all_event_ids)
    target.loc[list(survivor_ids)] = 1
    
    X = features_df.loc[all_event_ids]
    y = target.loc[all_event_ids]
    
    # Validation split for metrics (simple 80/20 for this diagnostic probe)
    # To be rigorous, we should use OOF, but this is a fast selection loop.
    # We'll train on 80% and eval on 20% to get "out of sample" separation metrics.
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(np.unique(y_train)) < 2:
        # Degenerate case
        return None, np.zeros(len(all_event_ids)), {'auc': 0.5, 'ks': 0.0, 'entropy': 1.0}

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

    # Metrics
    ks_stat, ent = calculate_separation_metrics(y_val.values, preds_val_prob)

    # Predict on Full set for pruning correlation
    preds_full_raw = model.predict(X)
    preds_full_prob = 1.0 / (1.0 + np.exp(-preds_full_raw))

    # AUC Lift (Validation)
    # Baseline is naive prevalence
    prevalence = y_val.mean()
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_val = roc_auc_score(y_val, preds_val_prob)
        auc_lift = auc_val - 0.5

        # De Prado: PR-AUC is more informative for imbalanced classes
        pr_val = average_precision_score(y_val, preds_val_prob)
        pr_lift = pr_val - prevalence
    except:
        auc_lift = 0.0
        pr_lift = 0.0

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
) -> List[Tuple[Geometry, Set[int]]]:
    
    logger.info(f"Vectorizing {len(events)} events (Standard normalization for Fixed Barrier compatibility)...")
    df = events_to_dataframe(events)
    if df.empty:
        logger.warning("No events to process.")
        return []

    # Calculate global quantile thresholds for MAE
    # We switch to 'norm_mae' (raw_mae / sigma) instead of 'time_scaled_mae'
    # because LabelBasedLayer2 executes Fixed Barriers (sl_mult * vol).
    # 'norm_mae' aligns exactly with 'sl_mult'.
    mae_series = df['norm_mae']

    # Quantiles to test: keeping top 50%, top 30%, top 20%, top 10% tightest stops
    # Lower MAE = Tighter stop relative to vol
    # "Quantile-Based Selection (Never Fixed Cutoffs)"
    quantiles = [0.2, 0.3, 0.4, 0.5]

    # Pre-calculate threshold values
    thresholds = {q: mae_series.quantile(q) for q in quantiles}

    # Generate candidates
    candidates = []
    for q in quantiles:
        for a in [0.5, 1.0, 1.5]:
            for b in [0.5, 1.0, 1.5]:
                for mr in [1.0, 1.5, 2.0]:
                    candidates.append(Geometry(sl_quantile=q, alpha=a, beta=b, min_ratio=mr))

    # Deduplicate
    candidates = list(set(candidates))
    
    accepted_candidates = []
    
    logger.info(f"Evaluating {len(candidates)} geometric candidates using Separation Objectives...")
    
    for geom in candidates:
        # B. Apply Quantile-Based Filters using norm_mae (Fixed Barrier Logic)
        thresh = thresholds[geom.sl_quantile]
        mask_sl = df['norm_mae'] <= thresh
        
        # Score Calculation
        # Use norm_mfe / norm_mae to align with LabelBasedLayer2 logic
        # score = (norm_mfe ** beta) / (norm_mae ** alpha)

        score = (df['norm_mfe'] ** geom.beta) / ((df['norm_mae'] + 1e-6) ** geom.alpha)
        mask_score = score >= geom.min_ratio
        
        survivors_df = df[mask_sl & mask_score]
        survivor_ids = set(survivors_df.index)
        
        if not survivor_ids:
            continue

        # D. Train Weak Learner First to get Separation Metrics
        # We need these metrics for the "Diagnostics Gates" now if we want to gate on KS
        model, preds, metrics = train_model_for_geometry(
            survivor_ids,
            list(df.index),
            features_df
        )

        if model is None:
            continue

        # C. Run Diagnostics Gates (Post-Training check included)
        # We pass the just-computed metrics as a "fold" metric for the current check
        diag = run_diagnostics_gates(
            list(survivor_ids),
            df,
            {0: metrics}, # Fake fold dict
            geom
        )
        
        if not diag.passed:
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
    for item in final_selection:
        result.append((item['geometry'], item['survivors']))
        
    best_score_str = f"{final_selection[0]['separation_score']:.3f}" if final_selection else "N/A"
    logger.info(f"Final Selection: {len(result)} geometries (Best Separation Score: {best_score_str})")
    return result
