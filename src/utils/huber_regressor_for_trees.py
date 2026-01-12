import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional

def prepare_huber_ensemble_teacher(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    epsilons: List[float] = [1.1, 1.35, 1.75],
    pruning_percentile: int = 15,
    corr_threshold: float = 0.7
) -> Dict:
    """
    Multi-Alpha Huber Ensemble & Quantile Meta-Target Infrastructure.
    Optimized for memory efficiency, causal consistency, and residual-student training.
    """
    # 1. Robust Scaling (Focus on IQR to neutralize wicks)
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    feature_names = np.asarray(X_train.columns)

    # 2. Multi-Alpha Huber Ensembling
    # Fits multiple epsilons to capture signals across different volatility regimes
    ensemble_coeffs = []
    ensemble_preds_tr = []
    
    # We maintain the list of models for validation/test inference
    models = []

    for eps in epsilons:
        # Varying alpha (L2 penalty) relative to epsilon for stability
        h = HuberRegressor(epsilon=eps, alpha=0.0001 * eps, max_iter=1000)
        h.fit(X_tr_scaled, y_train)
        
        ensemble_coeffs.append(h.coef_)
        ensemble_preds_tr.append(h.predict(X_tr_scaled))
        models.append(h)

    # 3. Consensus Logic (Median Coeffs for structural logic)
    # Using the Median coefficient across regimes provides a 'Structural Anchor'
    avg_coeffs = np.median(ensemble_coeffs, axis=0)
    abs_avg_coeffs = np.abs(avg_coeffs)
    
    # Warm-start consensus (Mean prediction)
    warm_start_tr = np.mean(ensemble_preds_tr, axis=0)

    # 4. Vectorized Feature Pruning (O(n) partitioning)
    kth_val = np.percentile(abs_avg_coeffs, pruning_percentile)
    keep_mask = abs_avg_coeffs > kth_val
    selected_feats = feature_names[keep_mask]
    
    # 5. Monotonicity via Local Residual Scale
    # Ties monotonicity to the structural signal floor
    local_scale = np.mean(abs_avg_coeffs[keep_mask])
    mono_cst = np.where(abs_avg_coeffs[keep_mask] > (0.15 * local_scale), 
                        np.sign(avg_coeffs[keep_mask]), 0)

    # 6. Interaction Constraints (Spearman-lite Clustering)
    # Filter for above-median importance to keep linkage computationally light
    imp_mask = abs_avg_coeffs[keep_mask] > np.median(abs_avg_coeffs[keep_mask])
    imp_indices = np.where(imp_mask)[0]
    
    interaction_constraints = []
    if imp_indices.size > 1:
        # Rank-based correlation (Spearman) is more robust for crypto feature tails
        X_imp_ranks = pd.DataFrame(X_tr_scaled[:, keep_mask][:, imp_indices]).rank()
        corr_matrix = np.corrcoef(X_imp_ranks.values.T)
        
        # Dissimilarity: D = 1 - |rho|
        D = np.clip(1 - np.abs(corr_matrix), 0, 1)
        Z = linkage(squareform(D, checks=False), method='complete')
        
        labels = fcluster(Z, corr_threshold, criterion='distance')
        interaction_constraints = [imp_indices[labels == l].tolist() for l in np.unique(labels)]

    # 7. Multi-Set Inference (Consensus)
    def get_consensus_pred(df):
        if df is None: return None
        df_s = scaler.transform(df)
        preds = [m.predict(df_s) for m in models]
        return np.mean(preds, axis=0)

    # 8. Residual Generation for Quantile Meta-Targets
    # The student should now fit the 'Unexplained Alpha' of the consensus
    residuals_tr = y_train - warm_start_tr

    return {
        'selected_features': selected_feats.tolist(),
        'monotonic_constraints': tuple(mono_cst.astype(int).tolist()),
        'interaction_constraints': interaction_constraints,
        'warm_start': {
            'train': warm_start_tr,
            'val': get_consensus_pred(X_val),
            'test': get_consensus_pred(X_test)
        },
        'quantile_meta_targets': residuals_tr,
        'scaler': scaler,
        'models': models
    }
