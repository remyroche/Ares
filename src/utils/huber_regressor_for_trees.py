import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def prepare_huber_teacher_outputs(X_train, y_train, X_val=None, X_test=None, 
                                 pruning_percentile=15, corr_threshold=0.7):
    """
    Unified Teacher Script (2026 Strategy)
    Uses HuberRegressor to generate: Pruned Features, Monotonic Constraints, 
    Interaction Groups, and Warm-Start Baselines.
    """
    # --- 1. Robust Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    
    # --- 2. Train Huber Teacher ---
    # epsilon=1.35 is robust to crypto wicks while staying efficient
    huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)
    huber.fit(X_train_scaled, y_train)
    
    coeffs = huber.coef_
    abs_coeffs = np.abs(coeffs)
    
    # --- 3. Feature Pruning ---
    # Drop features that Huber identifies as having near-zero structural signal
    prune_limit = np.percentile(abs_coeffs, pruning_percentile)
    selected_mask = abs_coeffs > prune_limit
    selected_features = X_train.columns[selected_mask].tolist()
    
    # Filter coefficients and data for selected features
    pruned_coeffs = coeffs[selected_mask]
    X_train_pruned = X_train[selected_features]
    
    # --- 4. Monotonic Constraints ---
    # 1: Increasing, -1: Decreasing, 0: Neutral (below significance threshold)
    sig_threshold = np.mean(abs_coeffs) * 0.1
    monotonic_cst = [1 if c > sig_threshold else -1 if c < -sig_threshold else 0 
                     for c in pruned_coeffs]
    
    # --- 5. Automated Interaction Constraints ---
    # Group important features by correlation to prevent spurious inter-domain links
    important_mask = abs_coeffs[selected_mask] > np.mean(abs_coeffs[selected_mask])
    important_idx = np.where(important_mask)[0]
    
    if len(important_idx) > 1:
        corr = X_train_pruned.iloc[:, important_idx].corr().abs()
        dissimilarity = 1 - corr.fillna(0)
        hierarchy = linkage(squareform(dissimilarity, checks=False), method='complete')
        cluster_labels = fcluster(hierarchy, corr_threshold, criterion='distance')
        
        interaction_groups = {}
        for i, label in enumerate(cluster_labels):
            idx = int(important_idx[i])
            interaction_groups.setdefault(label, []).append(idx)
        interaction_constraints = list(interaction_groups.values())
    else:
        interaction_constraints = None

    # --- 6. Warm-Start Baselines ---
    # Generate the 'initial guess' for the Tree models
    warm_start_train = huber.predict(X_train_scaled)
    warm_start_val = huber.predict(scaler.transform(X_val)) if X_val is not None else None
    warm_start_test = huber.predict(scaler.transform(X_test)) if X_test is not None else None

    return {
        "selected_features": selected_features,
        "monotonic_constraints": tuple(monotonic_cst),
        "interaction_constraints": interaction_constraints,
        "warm_start": {
            "train": warm_start_train,
            "val": warm_start_val,
            "test": warm_start_test
        },
        "huber_model": huber, # For future inspection
        "scaler": scaler
    }

# --- EXAMPLE USAGE ---
# outputs = prepare_huber_teacher_outputs(X_train, y_train, X_val, X_test)
# xgb_params['interaction_constraints'] = outputs['interaction_constraints']
# xgb_params['monotone_constraints'] = outputs['monotonic_constraints']
