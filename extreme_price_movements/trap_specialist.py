"""
Trap Specialist: GMM-based quality filter for identifying hollow/trap moves.

Uses a compact set of 8 liquidity/flow features to classify moves into
quality clusters: [Trap, Weak, Clean, Premium]
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.utils import tprint
from extreme_price_movements.fast_funcs import compute_quality_labels


# Compact feature set for GMM (8 features instead of 18)
TRAP_FEATURE_KEYS = [
    "flow_persistence",      # Flow alignment
    "churn",                 # Volume / Price change (high = fighting)
    "efficiency",            # Net move / Gross path
    "wick_ratio",            # Rejection signals
    "vol_z",                 # Volume conviction
    "climax_decay",          # Volume sustainability
    "vol_range_shock",       # Price-volume interaction
    "delta_stall_6",         # Flow-price correlation
]


def train_trap_specialist(panel, feats, cfg, syms, ts_end):
    """
    Train Trap Specialist GMM model.
    
    Args:
        panel: Dictionary with OHLCV DataFrames
        feats: Dictionary of feature DataFrames
        cfg: Configuration dictionary
        syms: List of symbols to train on
        ts_end: End timestamp for training window
    
    Returns:
        Dictionary with trained GMM model, scaler, and metadata
    """
    tprint("Training Trap Specialist (GMM)...")
    
    # 1. Generate quality labels
    horizon = cfg.get("trap_horizon", 6)
    tprint(f"  Computing quality labels (horizon={horizon})...")
    quality_scores = compute_quality_labels(panel, horizon=horizon)
    
    # 2. Build feature matrix
    tprint(f"  Building feature matrix ({len(TRAP_FEATURE_KEYS)} features)...")
    X_list = []
    y_list = []
    
    for sym in syms:
        if sym not in quality_scores.columns:
            continue
        
        # Get quality scores for this symbol
        y_sym = quality_scores[sym].dropna()
        
        if len(y_sym) < 100:
            continue
        
        # Extract features for this symbol
        X_sym_list = []
        valid_idx = []
        
        for idx in y_sym.index:
            if idx not in feats[TRAP_FEATURE_KEYS[0]].index:
                continue
            
            row = []
            valid = True
            for feat_key in TRAP_FEATURE_KEYS:
                if feat_key not in feats or sym not in feats[feat_key].columns:
                    valid = False
                    break
                val = feats[feat_key].loc[idx, sym]
                if np.isnan(val) or np.isinf(val):
                    valid = False
                    break
                row.append(val)
            
            if valid:
                X_sym_list.append(row)
                valid_idx.append(idx)
        
        if len(X_sym_list) > 0:
            X_sym = np.array(X_sym_list, dtype=np.float32)
            y_sym_aligned = y_sym.loc[valid_idx].values
            
            X_list.append(X_sym)
            y_list.append(y_sym_aligned)
    
    if not X_list:
        tprint("  ERROR: No valid training data for Trap Specialist")
        return None
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    tprint(f"  Training data: {len(X)} samples, {X.shape[1]} features")
    
    # 3. Bin quality scores into 4 clusters for GMM
    tprint("  Binning quality scores into 4 clusters...")
    y_binned = pd.qcut(y, q=4, labels=False, duplicates='drop')
    
    # 4. Fit GMM
    tprint("  Fitting GMM (4 components)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        max_iter=200,
        n_init=10,
        random_state=cfg.get("random_state", 42),
        verbose=0
    )
    gmm.fit(X_scaled)
    
    # 5. Semantic Sorting: Map GMM clusters to quality levels
    tprint("  Performing semantic sorting...")
    cluster_labels = gmm.predict(X_scaled)
    cluster_means = []
    
    for k in range(4):
        mask = (cluster_labels == k)
        if mask.sum() > 0:
            cluster_means.append(y[mask].mean())
        else:
            cluster_means.append(0.0)
    
    # Sort clusters by mean quality (0=Trap, 3=Premium)
    cluster_order = np.argsort(cluster_means)
    
    tprint(f"  Cluster quality means: {[f'{m:.3f}' for m in sorted(cluster_means)]}")
    tprint(f"  Cluster mapping: {cluster_order}")
    
    # 6. Validation metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    silhouette = silhouette_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    tprint(f"  Silhouette Score: {silhouette:.3f} (higher is better)")
    tprint(f"  Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
    
    tprint("✅ Trap Specialist training complete")
    
    return {
        "gmm": gmm,
        "scaler": scaler,
        "columns": TRAP_FEATURE_KEYS,
        "cluster_order": cluster_order,
        "n_samples": len(X),
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
    }
