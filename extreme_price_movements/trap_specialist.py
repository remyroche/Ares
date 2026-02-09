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


def build_trap_dataset(panel, feats, cfg, syms):
    """
    Build training dataset for Trap Specialist.
    Returns: DataFrame with features and 'y_quality' column.
    """
    tprint(f"Building Trap Specialist dataset...")
    
    # 1. Generate quality labels
    horizon = cfg.get("trap_horizon", 6)
    tprint(f"  Computing quality labels (horizon={horizon})...")
    quality_scores = compute_quality_labels(panel, horizon=horizon)
    
    # 2. Build feature matrix
    tprint(f"  Building feature matrix ({len(TRAP_FEATURE_KEYS)} features)...")
    data_list = []
    
    for sym in syms:
        if sym not in quality_scores.columns:
            continue
        
        # Get quality scores for this symbol
        y_sym = quality_scores[sym].dropna()
        
        if len(y_sym) < 100:
            continue
        
        # Extract features for this symbol
        # Use pandas reindexing for speed instead of loop
        valid_idx = y_sym.index.intersection(feats[TRAP_FEATURE_KEYS[0]].index)
        if len(valid_idx) < 100: continue
        
        y_sym = y_sym.loc[valid_idx]

        # Check all features exist
        # We can stack features into a DataFrame for this symbol
        X_df_list = []
        valid_feats = True
        for k in TRAP_FEATURE_KEYS:
            if k not in feats or sym not in feats[k].columns:
                valid_feats = False
                break
            X_df_list.append(feats[k][sym].reindex(valid_idx))
            
        if not valid_feats: continue
        
        X_sym = pd.concat(X_df_list, axis=1)
        X_sym.columns = TRAP_FEATURE_KEYS

        # Drop NaNs
        combined = X_sym.copy()
        combined["y_quality"] = y_sym.values
        combined["symbol"] = sym
        combined = combined.dropna()

        if len(combined) > 0:
            data_list.append(combined)
            
    if not data_list:
        tprint("  ERROR: No valid training data for Trap Specialist")
        return None

    full_df = pd.concat(data_list)
    # Ensure index name is preserved or reset?
    # Usually index is timestamp. Let's reset index to keep timestamp as column if needed.
    full_df.index.name = "ts"
    full_df = full_df.reset_index()
    
    tprint(f"  Trap dataset: {len(full_df)} samples")
    return full_df


def train_trap_from_dataset(dataset, cfg):
    """
    Train Trap Specialist from pre-built dataset.
    Args:
        dataset: DataFrame with features and 'y_quality' column.
        cfg: Config dict.
    Returns:
        Trained model dict.
    """
    tprint("Training Trap Specialist (GMM) from dataset...")

    if dataset is None or dataset.empty:
        tprint("  ERROR: Dataset is empty.")
        return None

    X = dataset[TRAP_FEATURE_KEYS].values.astype(np.float32)
    y = dataset["y_quality"].values.astype(np.float32)
    
    tprint(f"  Training data: {len(X)} samples, {X.shape[1]} features")
    
    # 3. Bin quality scores into 4 clusters for GMM
    # Binning is only for analysis/check, GMM is unsupervised/semi-supervised on features
    # Wait, GMM is trained on X (features), NOT y.
    # But we use y for semantic sorting.
    
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
    
    # Compute on subset if too large for silhouette
    if len(X_scaled) > 10000:
        idx = np.random.choice(len(X_scaled), 10000, replace=False)
        X_val = X_scaled[idx]
        lbl_val = cluster_labels[idx]
    else:
        X_val = X_scaled
        lbl_val = cluster_labels
    
    try:
        silhouette = silhouette_score(X_val, lbl_val)
        davies_bouldin = davies_bouldin_score(X_val, lbl_val)
        tprint(f"  Silhouette Score: {silhouette:.3f} (higher is better)")
        tprint(f"  Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
    except Exception as e:
        tprint(f"  Metrics failed: {e}")
        silhouette = 0.0
        davies_bouldin = 0.0
    
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


def train_trap_specialist(panel, feats, cfg, syms, ts_end):
    """
    Train Trap Specialist GMM model (Legacy wrapper).
    """
    ds = build_trap_dataset(panel, feats, cfg, syms)
    return train_trap_from_dataset(ds, cfg)
