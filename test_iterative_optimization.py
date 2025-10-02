#!/usr/bin/env python3
"""
Test script for the new 3-step iterative clustering optimization.

This script tests the advanced iterative optimization system with:
1. Local frontier moves (CV-focused)
2. Global reallocation (capacity-aware)
3. Break large clusters (size-aware quality thresholds)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the path
sys.path.append('/workspace/src')

from training.steps.market_analysis.clusters.iterative_optimization import (
    IterativeOptimization, ClusteringStats
)
from training.steps.market_analysis.clusters.step1_feature_preparation import ClusteringContext


def create_test_data(n_samples=1000, n_features=10, n_clusters=5, noise=0.1):
    """Create synthetic test data for clustering."""
    # Generate synthetic data with known clusters
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        random_state=42,
        cluster_std=noise
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true


def create_initial_clustering(features, n_clusters):
    """Create initial clustering using K-means."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(features)
    
    return assignments


async def test_iterative_optimization():
    """Test the iterative optimization system."""
    print("=== Testing 3-Step Iterative Clustering Optimization ===\n")
    
    # Create test data
    print("Creating test data...")
    features, true_labels = create_test_data(n_samples=500, n_features=8, n_clusters=4)
    print(f"Data shape: {features.shape}")
    print(f"True clusters: {len(np.unique(true_labels))}")
    
    # Create initial clustering
    print("\nCreating initial clustering...")
    initial_assignments = create_initial_clustering(features, n_clusters=4)
    print(f"Initial clusters: {len(np.unique(initial_assignments))}")
    
    # Create clustering context
    context = ClusteringContext(
        optimized_features=features,
        initial_assignments=initial_assignments,
        feature_names=[f"feature_{i}" for i in range(features.shape[1])],
        optimization_metrics={}
    )
    
    # Test ClusteringStats
    print("\nTesting ClusteringStats...")
    stats = ClusteringStats(features, initial_assignments)
    print(f"Initial CV ratio: {stats.get_cv_ratio():.4f}")
    print(f"Initial balance: {stats.get_balance_score():.4f}")
    print(f"Cluster sizes: {stats.cluster_sizes}")
    
    # Test move delta calculation
    print("\nTesting move delta calculation...")
    if len(features) > 10:
        test_point = 0
        current_cluster = stats.assignments[test_point]
        target_cluster = (current_cluster + 1) % stats.n_clusters
        
        delta_info = stats.calculate_move_delta(test_point, current_cluster, target_cluster)
        print(f"Move delta for point {test_point}: {delta_info}")
    
    # Test boundary point identification
    print("\nTesting boundary point identification...")
    boundary_points = IterativeOptimization()._identify_boundary_points(features, stats)
    print(f"Found {len(boundary_points)} boundary points")
    print(f"First 10 boundary points: {boundary_points[:10]}")
    
    # Test the full optimization loop
    print("\nTesting full optimization loop...")
    optimizer = IterativeOptimization(verbose=True)
    
    try:
        # Run optimization
        result_context = await optimizer.execute_optimization_loop(
            context, 
            config=None, 
            max_iterations=5  # Limit for testing
        )
        
        print(f"\nOptimization completed!")
        print(f"Final clusters: {len(np.unique(result_context.optimized_assignments))}")
        print(f"Final CV ratio: {stats.get_cv_ratio():.4f}")
        print(f"Final balance: {stats.get_balance_score():.4f}")
        
        # Show step reports
        if optimizer.step_reports:
            print(f"\nStep reports generated: {len(optimizer.step_reports)}")
            for i, report in enumerate(optimizer.step_reports):
                print(f"Round {i+1}: CV={report['final_cv']:.4f}, "
                      f"Balance={report['final_balance']:.4f}, "
                      f"Silhouette={report['final_silhouette']:.4f}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_iterative_optimization())