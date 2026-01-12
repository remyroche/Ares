import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler  # Better for Huber logic
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, Tuple, Optional, List

def prepare_huber_teacher_outputs(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: Optional[pd.DataFrame] = None, 
    X_test: Optional[pd.DataFrame] = None, 
    sample_weight: Optional[np.ndarray] = None,
    pruning_percentile: int = 15, 
    corr_threshold: float = 0.7,
    epsilon: float = 1.35  # 1.35 for 95% efficiency, 1.1 for higher robustness
) -> Dict:
    """
    Optimized Teacher Script (2026 Edition).
    Refined for speed, robustness to 'fat-tailed' data, and native model integration.
    """
    # 1. Robust Scaling: Aligns with Huber logic by focusing on medians/IQRs
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    # 2. Vectorized Huber Training
    # warm_start=True allows iterative updates if this is called in a loop
    huber = HuberRegressor(epsilon=epsilon, alpha=0.0001, max_iter=1000, warm_start=False)
    huber.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    
    coeffs = huber.coef_
    abs_coeffs = np.abs(coeffs)
    
    # 3. Dynamic Feature Pruning
    prune_limit = np.percentile(abs_coeffs, pruning_percentile)
    selected_mask = abs_coeffs > prune_limit
    selected_features = X_train.columns[selected_mask].tolist()
    
    if not selected_features:
        raise ValueError("Pruning percentile too high; 0 features selected.")

    # 4. Monotonic Constraints (Vectorized)
    # 2026 Best Practice: Using 15% of mean importance as the neutral '0' floor
    sig_threshold = np.mean(abs_coeffs) * 0.15
    monotonic_cst = np.where(coeffs[selected_mask] > sig_threshold, 1, 
                             np.where(coeffs[selected_mask] < -sig_threshold, -1, 0))
    
    # 5. Optimized Interaction Constraints
    # Prune interaction search to only 'High Signal' features to speed up linkage
    important_mask = abs_coeffs[selected_mask] > np.median(abs_coeffs[selected_mask])
    important_idx = np.where(important_mask)[0]
    
    interaction_constraints = []
    if len(important_idx) > 1:
        # Use Spearman correlation if data is highly non-linear/outlier-heavy
        corr = X_train[selected_features].iloc[:, important_idx].corr(method='spearman').abs()
        dissimilarity = np.clip(1 - corr.fillna(0).values, 0, 1)
        
        # Linkage is the bottleneck; complete method prevents 'chaining'
        hierarchy = linkage(squareform(dissimilarity, checks=False), method='complete')
        cluster_labels = fcluster(hierarchy, corr_threshold, criterion='distance')
        
        groups = {}
        for i, label in enumerate(cluster_labels):
            feat_idx = int(important_idx[i])
            groups.setdefault(label, []).append(feat_idx)
        interaction_constraints = list(groups.values())
    
    # 6. Optimized Baseline Generation
    def get_pred(df):
        return huber.predict(scaler.transform(df)) if df is not None else None

    return {
        "selected_features": selected_features,
        "monotonic_constraints": tuple(monotonic_cst.tolist()),
        "interaction_constraints": interaction_constraints,
        "warm_start": {
            "train": huber.predict(X_train_scaled),
            "val": get_pred(X_val),
            "test": get_pred(X_test)
        },
        "scaler": scaler,
        "huber_model": huber
    }
