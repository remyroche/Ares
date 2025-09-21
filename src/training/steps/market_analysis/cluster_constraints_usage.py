"""
Usage helpers to apply 4D-safe clustering constraints in the training pipeline.

This module provides functions to apply clustering with balanced distribution:
- `apply_constraints_to_hdbscan`: Original HDBSCAN approach
- `apply_optimized_centroid_clustering`: New centroid-based approach for 3-8% distribution
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .cluster_constraints import (
    fit_hdbscan_with_noise_target,
    enforce_cluster_constraints,
    split_giant_clusters,
    summarize_distribution,
)

# Import the optimized clustering approach
from .optimal_regime_clustering.optimized_clustering import OptimizedClustering


def apply_constraints_to_hdbscan(
    X_4d: np.ndarray,
    *,
    initial_labels: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    hdbscan_params: Optional[Dict[str, object]] = None,
    noise_target: Tuple[float, float] = (0.05, 0.10),
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Fit HDBSCAN (with optional noise tuning) and enforce distribution constraints in-place.

    Args:
        X_4d: Feature array in 4D (or existing embedding) of shape (N, 4 or D).
        initial_labels: If provided, skip fitting and only enforce constraints on these labels.
        metric: Distance metric for both HDBSCAN and constraint steps ("euclidean" or "cosine").
        hdbscan_params: Parameters for HDBSCAN. You can omit min_samples; this function tunes it.
        noise_target: Desired noise fraction range.
        random_state: Random seed for deterministic splits.

    Returns:
        Dict with keys: labels, metrics, model (if fitted), and params used.
    """
    try:
        import hdbscan
    except Exception as e:  # pragma: no cover
        raise ImportError("hdbscan is required for this function. Please install hdbscan.") from e

    X = np.asarray(X_4d)
    params = dict(hdbscan_params or {})
    params.setdefault("metric", metric)
    params.setdefault("cluster_selection_method", "leaf")
    params.setdefault("min_cluster_size", 5)  # Ensure min_cluster_size is set to prevent errors

    # Set additional parameters for maximum balance and to avoid oversized clusters
    params.setdefault("cluster_selection_epsilon", 0.01)  # Very tight cluster boundaries for many clusters
    params.setdefault("min_samples", None)  # Will be tuned by fit_hdbscan_with_noise_target
    params.setdefault("max_cluster_size", int(len(X) * 0.05))  # Limit maximum cluster size to 5% of data
    params.setdefault("cluster_selection_method", "eom")  # Use excess of mass for better cluster selection
    params.setdefault("allow_single_cluster", False)  # Prevent single large clusters
    params.setdefault("min_cluster_size", 2)  # Allow very small clusters to be created
    params.setdefault("clusterer_kwargs", {"approx_min_span_tree": False})  # More accurate clustering

    if initial_labels is None:
        # Fit with noise tuning
        try:
            clusterer = fit_hdbscan_with_noise_target(X, params, target_range=noise_target)
            labels = clusterer.labels_
            if labels is None or len(labels) != len(X):
                labels = clusterer.fit_predict(X)
            model = clusterer

            # Simple approach: use HDBSCAN results as-is, fallback to simple K-means if needed

        except Exception as e:
            print(f"DEBUG: HDBSCAN failed: {e}, using simple K-means fallback")
            # Simple fallback to K-means with reasonable cluster count
            labels = _simple_kmeans_fallback(X, n_clusters=25, random_state=random_state)
            model = None
    else:
        labels = np.asarray(initial_labels).copy()
        model = None

    # Allow large clusters through for coverage, then split them afterwards
    print(f"DEBUG: Before constraint enforcement: {len(np.unique(labels[labels >= 0]))} clusters")
    labels = enforce_cluster_constraints(
        X,
        labels,
        noise_label=-1,
        max_legit_prop=0.45,  # Allow large clusters through for coverage
        target_topk=50,  # Allow all reasonable clusters
        coverage_target=(0.90, 1.00),  # Target high coverage
        target_range=(0.01, 0.15),  # Allow larger clusters temporarily
        metric=metric,
        random_state=random_state,
    )
    print(f"DEBUG: After constraint enforcement: {len(np.unique(labels[labels >= 0]))} clusters")

    # Aggressive splitting: split any cluster >8% into smaller pieces
    labels = split_giant_clusters(
        X,
        labels,
        max_prop=0.08,  # Split any cluster >8%
        target_range=(0.02, 0.05),  # Split into 2-5% sub-clusters
        metric=metric,
        random_state=random_state,
    )

    # Summarize
    dist = summarize_distribution(labels, topk=20)
    result = {
        "labels": labels,
        "metrics": dist,
        "model": model,
        "params": params,
    }
    return result


def _check_cluster_balance(labels: np.ndarray, max_prop: float = 0.12) -> bool:
    """Check if clusters are reasonably balanced (no cluster > max_prop of data)."""
    if len(labels) == 0:
        return True

    N = len(labels)
    positive = [c for c in np.unique(labels) if c >= 0]

    # Check if any cluster is too large
    for c in positive:
        prop = np.sum(labels == c) / N
        if prop > max_prop:
            print(f"DEBUG: Found unbalanced cluster {c} with proportion {prop:.3f}")
            return False

    return True


def _simple_kmeans_fallback(X: np.ndarray, n_clusters: int = 20, random_state: int = 42) -> np.ndarray:
    """Simple K-means fallback when HDBSCAN fails."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print(f"DEBUG: Simple K-means fallback with {n_clusters} clusters")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simple K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        init='k-means++'
    )
    labels = kmeans.fit_predict(X_scaled)

    # Quick check of cluster sizes
    N = len(labels)
    for c in np.unique(labels):
        size = np.sum(labels == c)
        prop = size / N
        print(f"DEBUG: K-means cluster {c}: {size} samples ({prop:.3f}%)")

    return labels


def apply_optimized_centroid_clustering(
    X_4d: np.ndarray,
    *,
    metric: str = "euclidean",
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Apply optimized centroid-based clustering for 3-8% cluster distribution.

    This function uses the OptimizedClustering approach specifically designed
    to create 20 clusters with balanced 3-8% distribution, which is ideal
    for market analysis where we want uniform cluster sizes.

    Args:
        X_4d: Feature array in 4D (or existing embedding) of shape (N, 4 or D).
        metric: Distance metric ("euclidean" or "cosine").
        random_state: Random seed for deterministic results.

    Returns:
        Dict with keys: labels, metrics, model, and params used.
    """
    try:
        print(f"🎯 Starting optimized centroid clustering for 20 balanced clusters (3-8% distribution)...")

        # Create optimized clustering instance
        optimized_clusterer = OptimizedClustering(
            n_clusters=20,
            metric=metric,
            random_state=random_state,
            force_n_clusters=True,
            target_n_clusters=20,
        )

        # Perform centroid-based clustering
        labels = optimized_clusterer._calculate_centroid_based_clusters(X_4d)

        # Calculate distribution metrics
        N = len(labels)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Calculate cluster sizes and percentages
        cluster_sizes = {}
        cluster_percentages = {}

        for label in unique_labels:
            size = np.sum(labels == label)
            percentage = (size / N) * 100
            cluster_sizes[int(label)] = size
            cluster_percentages[int(label)] = percentage

        # Sort by size for analysis
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

        print(f"✅ Optimized clustering completed: {n_clusters} clusters created")
        print(f"📊 Cluster size distribution:")

        for cluster_id, size in sorted_clusters[:5]:  # Show top 5 largest
            percentage = cluster_percentages[cluster_id]
            print(f"   Cluster {cluster_id}: {size} samples ({percentage:.1f}%)")

        # Calculate coverage and balance metrics
        coverage = 100.0  # All samples are assigned
        largest_cluster_pct = cluster_percentages[sorted_clusters[0][0]]
        smallest_cluster_pct = cluster_percentages[sorted_clusters[-1][0]]

        # Check if we achieved the target distribution
        balanced_count = sum(1 for pct in cluster_percentages.values() if 3.0 <= pct <= 8.0)
        balanced_pct = (balanced_count / n_clusters) * 100

        print(f"📈 Balance metrics: {balanced_count}/{n_clusters} clusters in 3-8% range ({balanced_pct:.1f}%)")
        print(f"📈 Largest cluster: {largest_cluster_pct:.1f}%, Smallest cluster: {smallest_cluster_pct:.1f}%")

        return {
            'labels': labels,
            'model': optimized_clusterer,
            'metrics': {
                'n_clusters': n_clusters,
                'total_samples': N,
                'coverage_percentage': coverage,
                'largest_cluster_percentage': largest_cluster_pct,
                'smallest_cluster_percentage': smallest_cluster_pct,
                'balanced_clusters_count': balanced_count,
                'balanced_clusters_percentage': balanced_pct,
                'cluster_sizes': cluster_sizes,
                'cluster_percentages': cluster_percentages,
            },
            'params': {
                'method': 'optimized_centroid_clustering',
                'metric': metric,
                'random_state': random_state,
                'target_n_clusters': 20,
                'target_range': (3.0, 8.0),
            }
        }

    except Exception as e:
        print(f"❌ Error in optimized centroid clustering: {e}")
        # Fallback to simple K-means if centroid approach fails
        print("🔄 Falling back to simple K-means...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=20, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_4d)

        N = len(labels)
        return {
            'labels': labels,
            'model': kmeans,
            'metrics': {
                'n_clusters': 20,
                'total_samples': N,
                'coverage_percentage': 100.0,
                'method': 'fallback_kmeans',
            },
            'params': {
                'method': 'fallback_kmeans',
                'random_state': random_state,
                'n_clusters': 20,
            }
        }
