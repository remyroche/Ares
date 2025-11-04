#!/usr/bin/env python3
"""
Optimized Variational / Fast HMM test
- Uses hmmlearn.vhmm.VariationalGaussianHMM when available (finite VB HMM)
- Fallbacks: sklearn.mixture.BayesianGaussianMixture (VB-GMM) or KMeans
- Optimized for macOS / Apple Silicon (reduce threads, float32, randomized PCA)
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd

# --- set sensible thread limits for macOS / M1 (adjust to your machine) ---
# NOTE: set before importing heavy numeric libs if you want it honored by BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

# lightweight logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fast_vb_hmm")

def parse_args():
    p = argparse.ArgumentParser(description="Fast VB/EM HMM test (optimized).")
    p.add_argument("alpha", type=float)
    p.add_argument("kappa", type=float)
    p.add_argument("gamma", type=float)
    p.add_argument("n_iterations", type=int, nargs="?", default=30)
    p.add_argument("--cache", type=str, default="hdp_hmm_features_cache.npy")
    p.add_argument("--pca-components", type=int, default=15)
    p.add_argument("--seed", type=int, default=789)
    return p.parse_args()

def safe_load_features(path: str):
    """
    Load features from cache.
    Handles both old format (single array) and new format (dict with categorized features).
    Returns structural features for HMM training.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    
    # Try new format first (pickle with categorization)
    pkl_path = path.replace('.npy', '.pkl')
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if this is the new format with categorized features
            if isinstance(cache_data, dict) and 'structural_features' in cache_data:
                logger.info("Loading structural features (new format - prevents HMM 'cheating')")
                return cache_data['structural_features'].values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not load new format cache: {e}. Falling back to numpy.")
    
    # Fallback to old format (single numpy array)
    logger.info("Loading features (old format - all features)")
    arr = np.load(path, mmap_mode='r')
    return np.asarray(arr, dtype=np.float32)

def compute_metrics(X: np.ndarray, labels: np.ndarray, sample_for_silhouette=5000):
    from sklearn.metrics import silhouette_score
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return {
            "n_clusters": n_clusters,
            "silhouette": 0.0,
            "temporal_smoothness": 0.0,
            "balance_entropy": 0.0,
            "between_dispersion": 0.0,
            "within_dispersion": 0.0
        }
    # silhouette (sampled for speed on big datasets)
    sample_size = min(sample_for_silhouette, X.shape[0])
    silhouette = silhouette_score(X, labels, sample_size=sample_size, random_state=0)

    # temporal smoothness: fraction of positions that are same as previous (0..1)
    changes = np.sum(labels[1:] != labels[:-1])
    temporal = 1.0 - (changes / max(1, (len(labels) - 1)))

    # balance: normalized entropy across clusters
    sizes = np.bincount(labels)
    probs = sizes / sizes.sum()
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
    balance = entropy / np.log(n_clusters)

    # within-cluster dispersion (mean Frobenius norm of centered clusters)
    centroids = np.vstack([X[labels == k].mean(axis=0) for k in range(n_clusters)])
    within = 0.0
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            diffs = X[mask] - centroids[k]
            # sum of squared deviations per cluster normalized by cluster size
            within += np.sum(diffs * diffs) / mask.sum()
    within_disp = within / n_clusters
    # between-cluster dispersion: variance of centroids
    between_disp = np.var(centroids, axis=0).sum()

    return {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette),
        "temporal_smoothness": float(temporal),
        "balance_entropy": float(balance),
        "between_dispersion": float(between_disp),
        "within_dispersion": float(within_disp)
    }

def main():
    args = parse_args()
    alpha = args.alpha
    kappa = args.kappa
    gamma = args.gamma
    n_iter = args.n_iterations
    seed = args.seed
    rng = np.random.default_rng(seed)

    try:
        X = safe_load_features(args.cache)  # float32
    except Exception as exc:
        print(f"ERROR|{alpha}|{kappa}|{gamma}|Cache not found: {exc}", flush=True)
        sys.exit(1)

    # PCA: Skip if features are already in PCA space (new format saves PCA'd features)
    # Check if number of features matches expected PCA components (indicates already PCA'd)
    from sklearn.decomposition import PCA
    n_components = min(args.pca_components, X.shape[1])
    
    if X.shape[1] == args.pca_components or X.shape[1] < 20:
        # Features are already in PCA space (new format) - skip PCA
        logger.info(f"Features already in PCA space ({X.shape[1]} components) - skipping PCA")
        Xp = X.astype(np.float32)
    else:
        # Raw features (old format or fallback) - apply PCA
        logger.info(f"Applying PCA to reduce {X.shape[1]} features to {n_components} components")
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
        Xp = pca.fit_transform(X.astype(np.float32))

    # ============================================================================
    # NEW: Normalize & rank-transform features to address high CV ratio issues
    # ============================================================================
    # 1. Apply rolling window normalization to ALL PCA components
    logger.info("Applying rolling window normalization (window=500)...")
    df_features = pd.DataFrame(Xp)
    
    rolling_mean = df_features.rolling(500, min_periods=50).mean()
    rolling_std = df_features.rolling(500, min_periods=50).std()
    features_normalized = (df_features - rolling_mean) / (rolling_std + 1e-9)
    
    # Fill NaNs from rolling window: forward fill, then backward fill, then zero (mean of z-score)
    features_normalized = features_normalized.ffill().bfill().fillna(0)
    
    # 2. Rank-transform volatility-like features to neutralize their scale
    # Since these are PCA components, we'll identify high-variance components
    # (which often capture volatility) and rank-transform them
    component_std = features_normalized.std()
    vol_threshold = component_std.quantile(0.75)  # Top 25% most volatile components
    vol_cols = component_std[component_std > vol_threshold].index.tolist()
    
    if not vol_cols:
        logger.warning("Could not find high-variance PCA components to rank-transform.")
    else:
        logger.info(f"Rank-transforming {len(vol_cols)} high-variance PCA components...")
        
        # Store the non-transformed columns for proper NaN handling later
        non_vol_cols = [col for col in features_normalized.columns if col not in vol_cols]
        
        # Use a long-term rolling rank to get stable percentiles
        # This converts extreme values into a clean 0.0 to 1.0 scale
        for col in vol_cols:
            # .rank(pct=True) calculates the percentile rank from 0.0 to 1.0
            features_normalized[col] = features_normalized[col].rolling(500, min_periods=50).rank(pct=True)
        
        # Balanced NaN filling strategy:
        # - Forward/backward fill first (preserves temporal structure)
        # - Rank-transformed columns: fill remaining with column median (typically ~0.5)
        # - Non-transformed columns: fill remaining with 0 (mean of z-score)
        features_normalized = features_normalized.ffill().bfill()
        
        # Fill any remaining NaNs with column-specific appropriate values
        for col in vol_cols:
            # For rank-transformed: use actual column median (more robust than fixed 0.5)
            col_median = features_normalized[col].median()
            features_normalized[col].fillna(col_median if pd.notna(col_median) else 0.5, inplace=True)
        
        for col in non_vol_cols:
            # For z-scored features: use 0 (the mean)
            features_normalized[col].fillna(0, inplace=True)
    
    # Convert back to numpy array
    Xp = features_normalized.values.astype(np.float32)
    logger.info("Normalization and rank-transform complete")
    # ============================================================================

    # tiny jitter reproducibly
    Xp += rng.normal(0.0, 1e-6, size=Xp.shape).astype(np.float32)

    start = time.time()

    # Prefer hmmlearn's VariationalGaussianHMM if available (finite VB)
    use_vhmm = False
    try:
        # attempt to import variational HMM class
        import importlib
        hmmlearn_vhmm = importlib.import_module("hmmlearn.vhmm")
        VariationalGaussianHMM = getattr(hmmlearn_vhmm, "VariationalGaussianHMM", None)
        if VariationalGaussianHMM is not None:
            use_vhmm = True
    except Exception:
        use_vhmm = False

    # choose K with a slightly smarter rule: capped, but at least 3 and scale with alpha and data dimensionality
    K = int(min(50, max(3, np.clip(int(alpha * 2), 3, 20))))
    # but don't ask for more components than data points
    K = min(K, max(3, Xp.shape[0] // 5))

    labels = None
    if use_vhmm:
        try:
            # finite variational HMM (this is true VB - uses variational objective)
            model = VariationalGaussianHMM(
                n_components=K,
                covariance_type="diag",
                n_iter=n_iter,
                random_state=seed,
                # set initialization behavior
                init_params="stmc",
                params="stmc",
                tol=1e-4,
                verbose=False
            )
            model.fit(Xp)
            labels = model.predict(Xp)
            used_method = "VariationalGaussianHMM"
        except Exception:
            # fallback if vhmm fails
            labels = None

    if labels is None:
        # Try BayesianGaussianMixture (VB for GMM) as a faster VB fallback (non-sequential)
        try:
            from sklearn.mixture import BayesianGaussianMixture
            bgm = BayesianGaussianMixture(
                n_components=K,
                covariance_type="diag",
                max_iter=n_iter,
                random_state=seed,
                weight_concentration_prior_type='dirichlet_distribution'
            )
            labels = bgm.fit_predict(Xp)
            used_method = "BayesianGaussianMixture"
        except Exception:
            # final fallback: KMeans (fast)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, random_state=seed, n_init=10)
            labels = km.fit_predict(Xp)
            used_method = "KMeans"

    elapsed = time.time() - start

    metrics = compute_metrics(Xp, labels)

    out = ("SUCCESS|{alpha}|{kappa}|{gamma}|{n_clusters}|{silhouette:.6f}|"
           "{temporal:.6f}|{balance:.6f}|{between:.6e}|{within:.6e}|{method}|{elapsed:.3f}")
    print(out.format(
        alpha=alpha,
        kappa=kappa,
        gamma=gamma,
        n_clusters=metrics["n_clusters"],
        silhouette=metrics["silhouette"],
        temporal=metrics["temporal_smoothness"],
        balance=metrics["balance_entropy"],
        between=metrics["between_dispersion"],
        within=metrics["within_dispersion"],
        method=used_method,
        elapsed=elapsed
    ), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        # include traceback in error to help debugging
        print(f"ERROR|{sys.argv[1] if len(sys.argv)>1 else 'NA'}|"
              f"{sys.argv[2] if len(sys.argv)>2 else 'NA'}|"
              f"{sys.argv[3] if len(sys.argv)>3 else 'NA'}|{str(e)}\\n{tb}", flush=True)
        sys.exit(1)
#!/usr/bin/env python3
"""
FAST Variational HDP-HMM Test (Truncated + Diagonal Covariance)
Uses EM algorithm instead of Gibbs sampling for 10-100x speedup
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

# Accept parameters
if len(sys.argv) < 4:
    print("Usage: python hdp_hmm_single_test_fast_vb.py <alpha> <kappa> <gamma> [n_iterations]")
    sys.exit(1)

alpha = float(sys.argv[1])
kappa = float(sys.argv[2])
gamma = float(sys.argv[3])
n_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 30

try:
    # Try hmmlearn first (supports diagonal covariance + EM)
    try:
        from hmmlearn import hmm as hmmlearn_hmm
        USE_HMMLEARN = True
    except:
        USE_HMMLEARN = False
    
    # Load cached features
    cache_file = "hdp_hmm_features_cache.npy"
    if not os.path.exists(cache_file):
        print(f"ERROR|{alpha}|{kappa}|{gamma}|Cache not found", flush=True)
        sys.exit(1)
    
    feature_array = np.load(cache_file)
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=15)
    data_pca = pca.fit_transform(feature_array)
    
    # ============================================================================
    # NEW: Normalize & rank-transform features to address high CV ratio issues
    # ============================================================================
    # 1. Apply rolling window normalization to ALL PCA components
    print(f"   🔧 Applying rolling window normalization (window=500)...", flush=True)
    df_features = pd.DataFrame(data_pca)
    
    rolling_mean = df_features.rolling(500, min_periods=50).mean()
    rolling_std = df_features.rolling(500, min_periods=50).std()
    features_normalized = (df_features - rolling_mean) / (rolling_std + 1e-9)
    
    # Fill NaNs from rolling window: forward fill, then backward fill, then zero (mean of z-score)
    features_normalized = features_normalized.ffill().bfill().fillna(0)
    
    # 2. Rank-transform volatility-like features to neutralize their scale
    # Since these are PCA components, we'll identify high-variance components
    # (which often capture volatility) and rank-transform them
    component_std = features_normalized.std()
    vol_threshold = component_std.quantile(0.75)  # Top 25% most volatile components
    vol_cols = component_std[component_std > vol_threshold].index.tolist()
    
    if not vol_cols:
        print(f"   ⚠️  Could not find high-variance PCA components to rank-transform.", flush=True)
    else:
        print(f"   🔧 Rank-transforming {len(vol_cols)} high-variance PCA components...", flush=True)
        
        # Store the non-transformed columns for proper NaN handling later
        non_vol_cols = [col for col in features_normalized.columns if col not in vol_cols]
        
        # Use a long-term rolling rank to get stable percentiles
        # This converts extreme values into a clean 0.0 to 1.0 scale
        for col in vol_cols:
            # .rank(pct=True) calculates the percentile rank from 0.0 to 1.0
            features_normalized[col] = features_normalized[col].rolling(500, min_periods=50).rank(pct=True)
        
        # Balanced NaN filling strategy:
        # - Forward/backward fill first (preserves temporal structure)
        # - Rank-transformed columns: fill remaining with column median (typically ~0.5)
        # - Non-transformed columns: fill remaining with 0 (mean of z-score)
        features_normalized = features_normalized.ffill().bfill()
        
        # Fill any remaining NaNs with column-specific appropriate values
        for col in vol_cols:
            # For rank-transformed: use actual column median (more robust than fixed 0.5)
            col_median = features_normalized[col].median()
            features_normalized[col].fillna(col_median if pd.notna(col_median) else 0.5, inplace=True)
        
        for col in non_vol_cols:
            # For z-scored features: use 0 (the mean)
            features_normalized[col].fillna(0, inplace=True)
    
    # Convert back to numpy array
    data_pca = features_normalized.values
    print(f"   ✅ Normalization and rank-transform complete", flush=True)
    # ============================================================================
    
    # Add noise jitter to duplicate features (required for diagonal cov)
    data_pca = data_pca + np.random.normal(0, 1e-6, data_pca.shape)
    
    start_time = datetime.now()
    
    if USE_HMMLEARN:
        # FAST: Use hmmlearn with diagonal covariance + EM (variational)
        # Truncated: Fix K based on parameter exploration
        K = min(10, max(3, int(alpha * 2)))  # K scales with alpha
        
        model = hmmlearn_hmm.GaussianHMM(
            n_components=K,
            covariance_type="diag",  # DIAGONAL - much faster than full!
            n_iter=n_iterations,
            init_params="stmc",  # Initialize all params
            params="stmc",       # Update all params
            random_state=789
        )
        
        # Fit using EM (variational inference, much faster than Gibbs!)
        model.fit(data_pca)
        labels = model.predict(data_pca)
        
    else:
        # Fallback: Use simplified K-means
        from sklearn.cluster import KMeans
        K = min(10, max(3, int(alpha * 2)))
        kmeans = KMeans(n_clusters=K, random_state=789, n_init=10)
        labels = kmeans.fit_predict(data_pca)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Calculate quick metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    n_clusters = len(np.unique(labels))
    
    if n_clusters > 1:
        silhouette = silhouette_score(data_pca, labels)
        # Simple temporal smoothness (adjacent label changes)
        changes = np.sum(labels[1:] != labels[:-1])
        temporal = 1.0 - (changes / len(labels))
        
        # Simple balance (entropy of cluster sizes)
        sizes = np.bincount(labels)
        probs = sizes / sizes.sum()
        balance = -np.sum(probs * np.log(probs + 1e-10)) / np.log(n_clusters + 1e-10)
        
        # CV metrics
        between_cv = 0.0
        within_cv = 0.0
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.sum() > 1:
                cluster_features = data_pca[mask]
                within_cv += cluster_features.std()
        within_cv /= n_clusters
        between_cv = data_pca.mean(axis=0).std()
        
    else:
        silhouette = 0.0
        temporal = 0.0
        balance = 0.0
        between_cv = 0.0
        within_cv = 1.0
    
    # ============================================================================
    # Regime-Conditioned P&L Evaluation
    # ============================================================================
    sharpe_per_regime = {}
    sharpe_differences_valid = True
    
    try:
        # Load price data for signal returns calculation
        price_cache_file = "hdp_hmm_price_cache.pkl"
        if os.path.exists(price_cache_file):
            import pickle
            with open(price_cache_file, 'rb') as f:
                price_data = pickle.load(f)
            
            close_prices = price_data['close']
            
            # Calculate forward returns (signal returns) - using 5-bar forward returns
            # This represents the P&L from entering a position based on the regime signal
            forward_returns = np.zeros(len(close_prices))
            for i in range(len(close_prices) - 5):
                forward_returns[i] = (close_prices[i+5] / close_prices[i]) - 1.0
            
            # Calculate Sharpe ratio per regime
            unique_regimes = np.unique(labels)
            sharpe_values = []
            
            for r in unique_regimes:
                regime_mask = labels == r
                signal_returns_r = forward_returns[regime_mask]
                
                # Filter out NaNs and ensure we have enough data points
                signal_returns_r = signal_returns_r[~np.isnan(signal_returns_r)]
                
                if len(signal_returns_r) < 2:
                    sharpe_r = 0.0
                else:
                    mean_return = np.mean(signal_returns_r)
                    std_return = np.std(signal_returns_r)
                    
                    if std_return > 1e-10:
                        # Annualize Sharpe: assuming hourly data, sqrt(24 * 365) for hourly
                        sharpe_r = (mean_return / std_return) * np.sqrt(24 * 365)
                    else:
                        sharpe_r = 0.0
                
                sharpe_per_regime[int(r)] = sharpe_r
                sharpe_values.append(sharpe_r)
            
            # Check if Sharpe differences are economically meaningful
            # If all Sharpe ratios are similar (< 0.3 difference), segmentation is not meaningful
            if len(sharpe_values) > 1:
                sharpe_range = max(sharpe_values) - min(sharpe_values)
                sharpe_differences_valid = sharpe_range >= 0.3
            else:
                sharpe_differences_valid = False
                
        else:
            # Price cache not available - set defaults
            sharpe_per_regime = {}
            sharpe_differences_valid = False
            
    except Exception as e:
        # If evaluation fails, continue without regime P&L metrics
        sharpe_per_regime = {}
        sharpe_differences_valid = False
    
    # Format Sharpe per regime for output (comma-separated values)
    sharpe_str = ",".join([f"{r}:{sharpe_per_regime.get(r, 0.0):.4f}" for r in sorted(sharpe_per_regime.keys())]) if sharpe_per_regime else ""
    sharpe_diffs_valid_int = 1 if sharpe_differences_valid else 0
    
    # Output result (extended format with regime P&L metrics)
    # Format: SUCCESS|α|κ|γ|clusters|silhouette|temporal|balance|between_cv|within_cv|economic_cv|elapsed|sharpe_per_regime|sharpe_diffs_valid
    print(f"SUCCESS|{alpha}|{kappa}|{gamma}|{n_clusters}|{silhouette:.6f}|"
          f"{temporal:.6f}|{balance:.6f}|{between_cv:.6e}|{within_cv:.6e}|0.0|{elapsed:.3f}|"
          f"{sharpe_str}|{sharpe_diffs_valid_int}", flush=True)
    
except Exception as e:
    print(f"ERROR|{alpha}|{kappa}|{gamma}|{str(e)}", flush=True)
    sys.exit(1)

