import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union

def prepare_huber_teacher_outputs(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    vol_proxy: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    epsilons: List[float] = [1.1, 1.35, 1.75],
    alphas: List[float] = [1e-4, 1e-3, 1e-2],
    pruning_percentile: int = 15,
    corr_threshold: float = 0.7,
    n_jobs: int = -1
) -> Dict:
    """
    Advanced Huber Orchestrator for 15m Crypto Specialists.
    Includes: Vol-Weighting, Parallel Grid Fitting, and Named Interaction Constraints.
    """
    # 1. Sample Weighting (De Prado alignment)
    # Priority 1: Direct sample_weight (e.g. from Sequential Bootstrap)
    # Priority 2: Inverse Volatility (if vol_proxy provided)
    if sample_weight is not None:
        actual_weights = np.asarray(sample_weight)
        actual_weights /= actual_weights.mean() # Normalize to preserve scale
    elif vol_proxy is not None:
        actual_weights = (1.0 / vol_proxy).fillna(vol_proxy.median()).values
        actual_weights /= actual_weights.mean() # Normalize to preserve scale
    else:
        actual_weights = None

    # 2. Robust Scaling (NumPy-first for speed)
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    feature_names = np.asarray(X_train.columns)

    # 3. Vectorized Ensemble Training (Parallel Grid Fit)
    # 3x3 Grid: 3 Epsilons x 3 Alphas = 9 Teachers
    def _fit_huber(eps, alpha):
        h = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=5000)  # Increased max_iter for better convergence
        if actual_weights is not None:
            h.fit(X_tr_scaled, y_train, sample_weight=actual_weights)
        else:
            h.fit(X_tr_scaled, y_train)
        return h.coef_, h.predict(X_tr_scaled), h

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_huber)(e, a) for e in epsilons for a in alphas
    )
    
    ensemble_coeffs, ensemble_preds_tr, models = zip(*results)

    # 4. Consensus Logic (Median Coeffs for structural logic)
    avg_coeffs = np.median(ensemble_coeffs, axis=0)
    abs_avg_coeffs = np.abs(avg_coeffs)
    warm_start_tr = np.mean(ensemble_preds_tr, axis=0)

    # 5. O(n) Feature Pruning
    kth_val = np.percentile(abs_avg_coeffs, pruning_percentile)
    keep_mask = abs_avg_coeffs > kth_val
    selected_feats = feature_names[keep_mask]
    
    # 6. Monotonicity via Local Residual Scale
    local_scale = np.mean(abs_avg_coeffs[keep_mask])
    mono_cst = np.where(abs_avg_coeffs[keep_mask] > max(0.15 * local_scale, np.percentile(abs_avg_coeffs[keep_mask], 10)),
                        np.sign(avg_coeffs[keep_mask]), 0)

    # 7. Interaction Constraints (Named Output for Tree Learners)
    # Efficiency: Use Rank-based correlation (O(n log n))
    imp_mask = abs_avg_coeffs[keep_mask] > np.median(abs_avg_coeffs[keep_mask])
    imp_feat_names = selected_feats[imp_mask]
    
    interaction_constraints = []
    if imp_feat_names.size > 1:
        # Vectorized Rank correlation
        X_imp_ranks = pd.DataFrame(X_tr_scaled[:, keep_mask][:, imp_mask]).rank().values
        corr_matrix = np.corrcoef(X_imp_ranks.T)
        
        D = np.clip(1 - np.abs(corr_matrix), 0, 1)
        Z = linkage(squareform(D, checks=False), method='complete')
        
        labels = fcluster(Z, corr_threshold, criterion='distance')
        
        # Save as feature names to prevent indexing breaks in HPO
        for l in np.unique(labels):
            group = imp_feat_names[labels == l].tolist()
            interaction_constraints.append(group)
    else:
        interaction_constraints = None

    # 8. Consensus Inference for Validation/Test
    def get_consensus_pred(df):
        if df is None: return None
        df_s = scaler.transform(df)
        preds = [m.predict(df_s) for m in models]
        return np.mean(preds, axis=0)

    return {
        'selected_features': selected_feats.tolist(),
        'monotonic_constraints': dict(zip(selected_feats, mono_cst.astype(int))),
        'interaction_constraints': interaction_constraints,
        'warm_start': {
            'train': warm_start_tr,
            'val': get_consensus_pred(X_val),
            'test': get_consensus_pred(X_test)
        },
        'huber_models': models, # For future inspection and prediction
        'quantile_meta_targets': y_train - warm_start_tr,
        'scaler': scaler
    }

# Backward compatibility alias
def prepare_huber_production_orchestrator(*args, **kwargs):
    """Deprecated alias for prepare_huber_teacher_outputs"""
    return prepare_huber_teacher_outputs(*args, **kwargs)
