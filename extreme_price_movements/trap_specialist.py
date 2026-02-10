"""
Trap Specialist: GMM-based quality filter for identifying hollow/trap moves.

Uses a compact set of 8 liquidity/flow features to classify moves into
quality clusters: [Trap, Weak, Clean, Premium]
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from extreme_price_movements.utils import tprint
from extreme_price_movements.fast_funcs import compute_quality_labels
from extreme_price_movements.purged_cv import PurgedKFold


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
    
    # Subsample for GMM fitting (full covariance on 8M rows is prohibitive)
    max_gmm_samples = cfg.get("trap_max_gmm_samples", 200_000)
    if len(X) > max_gmm_samples:
        rng = np.random.RandomState(cfg.get("random_state", 42))
        idx_sub = rng.choice(len(X), max_gmm_samples, replace=False)
        X_fit = X[idx_sub]
        y_fit = y[idx_sub]
        tprint(f"  Subsampled {max_gmm_samples} / {len(X)} for GMM fitting")
    else:
        X_fit = X
        y_fit = y
    
    # 4. Fit GMM
    tprint("  Fitting GMM (4 components)...")
    scaler = StandardScaler()
    X_scaled_fit = scaler.fit_transform(X_fit)
    
    gmm = GaussianMixture(
        n_components=4,
        covariance_type='diag',
        max_iter=200,
        n_init=3,
        random_state=cfg.get("random_state", 42),
        verbose=0
    )
    gmm.fit(X_scaled_fit)
    
    # Scale full dataset for prediction
    X_scaled = scaler.transform(X)
    
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

def compute_trap_oof_predictions(X, y, cfg):
    """
    Compute OOF quality scores for Trap Specialist.
    Returns: array of scores.
    """
    tprint("  TrapSpecialist: Computing OOF scores...")

    # Using PurgedKFold(n_splits=5)
    kf = PurgedKFold(n_splits=5, purge=2, embargo=0)

    oof_scores = np.full(len(y), np.nan, dtype=np.float32)

    # Ensure array
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(np.float32)
    else:
        X_arr = X

    y_arr = np.array(y, dtype=np.float32)

    # Since we need scaling, we must scale inside fold

    for i, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train = y_arr[train_idx]

        # Subsample train fold for GMM speed
        max_fold = 200_000
        if len(X_train) > max_fold:
            rng = np.random.RandomState(cfg.get("random_state", 42) + i)
            sub = rng.choice(len(X_train), max_fold, replace=False)
            X_train_sub = X_train[sub]
            y_train_sub = y_train[sub]
        else:
            X_train_sub = X_train
            y_train_sub = y_train

        # Fit Fold Model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sub)

        try:
            gmm = GaussianMixture(
                n_components=4,
                covariance_type='diag',
                max_iter=100,
                n_init=2,
                random_state=cfg.get("random_state", 42) + i,
                verbose=0
            )
            gmm.fit(X_train_scaled)

            # Semantic Sorting on Fold
            train_labels = gmm.predict(X_train_scaled)
            cluster_means = []
            for k in range(4):
                mask = (train_labels == k)
                if mask.sum() > 0:
                    cluster_means.append(y_train_sub[mask].mean())
                else:
                    cluster_means.append(0.0)

            # Map cluster ID to quality score
            # We can use the mean quality as the score directly
            # prediction = mean_quality[cluster]

            X_test_scaled = scaler.transform(X_test)
            test_labels = gmm.predict(X_test_scaled)

            # Vectorized mapping
            scores = np.array([cluster_means[l] for l in test_labels], dtype=np.float32)
            oof_scores[test_idx] = scores

        except Exception:
            # Fallback if GMM fails
            continue

    return oof_scores


def train_trap_specialist(panel, feats, cfg, syms, ts_end):
    """
    Train Trap Specialist GMM model (Legacy wrapper).
    """
    ds = build_trap_dataset(panel, feats, cfg, syms)
    return train_trap_from_dataset(ds, cfg)
