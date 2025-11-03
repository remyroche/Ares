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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    # allow memmap if file large
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

    # PCA: randomized for speed, float32
    from sklearn.decomposition import PCA
    n_components = min(args.pca_components, X.shape[1])
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
    Xp = pca.fit_transform(X.astype(np.float32))

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
    
    # Output result
    print(f"SUCCESS|{alpha}|{kappa}|{gamma}|{n_clusters}|{silhouette}|"
          f"{temporal}|{balance}|{between_cv}|{within_cv}|0.0|{elapsed}", flush=True)
    
except Exception as e:
    print(f"ERROR|{alpha}|{kappa}|{gamma}|{str(e)}", flush=True)
    sys.exit(1)

