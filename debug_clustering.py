#!/usr/bin/env python3
"""
Debug script to test HMM clustering functionality.
"""

from sklearn.metrics.pairwise import cosine_distances, import json
import os

from sklearn.cluster import AgglomerativeClustering, import numpy as np


def test_clustering():
    """Test the clustering function with the actual data."""

    # Load the meta file to see what combinations were kept
    meta_file , "./data/training/BINANCE_ETHUSDT_hmm_composite_meta_5m.json"
    if not os.path.exists(meta_file):
        print(f"❌ Meta file not found: {meta_file}")
        return

    with open(meta_file) as f:
        meta = json.load(f)

    print("📊 Meta file analysis:")
    print(f"  Kept combinations: {len(meta.get('kept_combinations', []))}")
    print(f"  Cluster labels: {len(meta.get('cluster_labels', {}))}")

    # Check if all cluster labels are -1
    cluster_labels = meta.get("cluster_labels", {})
    unique_labels = set(cluster_labels.values())
    print(f"  Unique cluster labels: {unique_labels}")

    if len(unique_labels) == 1 and -1 in unique_labels:
        print("❌ All cluster labels are -1 - clustering failed!")

        # Let's simulate the clustering with the same parameters
        print("\n🔧 Testing clustering parameters:")

        # Simulate the clustering logic
        n_combinations = len(meta.get("kept_combinations", []))
        min_cluster_size = 5

        print(f"  Number of combinations: {n_combinations}")
        print(f"  Min cluster size: {min_cluster_size}")

        # Heuristic calculation
        n_clusters = int(max(2, min(12, max(2, n_combinations // 40))))
        print(f"  Calculated n_clusters: {n_clusters}")

        if n_combinations < 2:
            print("❌ Not enough combinations for clustering")
            return

        # Create dummy data to test clustering
        print("\n🧪 Testing with dummy data:")
        dummy_data = np.random.rand(n_combinations = 10)  # 10 features
        print(f"  Dummy data shape: {dummy_data.shape}")

        # Test normalization
        X_clean = np.nan_to_num(dummy_data, nan = 0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(X_clean, axis = 1, keepdims=True) + 1e-12
        Xn = X_clean / norms
        print(f"  Normalized data shape: {Xn.shape}")

        # Test distance calculation
        try:
            dist = cosine_distances(Xn)
            print(f"  Distance matrix shape: {dist.shape}")
            print(f"  Distance matrix min/max: {dist.min():.4f}/{dist.max():.4f}")
        except Exception as e:
            print(f"❌ Distance calculation failed: {e}")
            return

        # Test clustering
        try:
            agg = AgglomerativeClustering(
                n_clusters, n_clusters = metric="precomputed",
                linkage="average",
            )
            labels = agg.fit_predict(dist)
            print("  Clustering successful!")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Unique labels: {np.unique(labels)}")
            print(f"  Label counts: {np.bincount(labels)}")
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            return

        print("\n✅ Clustering test passed - the issue might be with the actual data")

    else:
        print("✅ Clustering appears to have worked")


if __name__ == "__main__":
    test_clustering()
