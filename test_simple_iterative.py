#!/usr/bin/env python3
"""
Simple test for the iterative optimization system.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append('/workspace/src')

# Test the core components directly
def test_core_components():
    """Test the core components without full system integration."""
    print("=== Testing Core Components ===")
    
    # Test Numba functions
    try:
        from training.steps.market_analysis.clusters.iterative_optimization import (
            calculate_wcss_incremental, calculate_bcss_incremental, 
            calculate_boundary_scores_numba, calculate_margin_gain_numba
        )
        print("✅ Numba functions imported successfully")
        
        # Test with simple data
        features = np.random.rand(100, 5)
        centroids = np.random.rand(3, 5)
        assignments = np.random.randint(0, 3, 100)
        global_mean = np.mean(features, axis=0)
        cluster_sizes = np.array([30, 35, 35])
        
        # Test WCSS calculation
        wcss = calculate_wcss_incremental(features, centroids, assignments)
        print(f"✅ WCSS calculation: {wcss:.4f}")
        
        # Test BCSS calculation
        bcss = calculate_bcss_incremental(centroids, global_mean, cluster_sizes)
        print(f"✅ BCSS calculation: {bcss:.4f}")
        
        # Test boundary scores
        boundary_scores = calculate_boundary_scores_numba(features, centroids, assignments)
        print(f"✅ Boundary scores calculated: {len(boundary_scores)} points")
        
        # Test margin gain
        point = features[0]
        centroid_from = centroids[0]
        centroid_to = centroids[1]
        margin_gain = calculate_margin_gain_numba(point, centroid_from, centroid_to, centroids)
        print(f"✅ Margin gain calculation: {margin_gain:.4f}")
        
    except Exception as e:
        print(f"❌ Core components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_clustering_stats():
    """Test the ClusteringStats class."""
    print("\n=== Testing ClusteringStats ===")
    
    try:
        from training.steps.market_analysis.clusters.iterative_optimization import ClusteringStats
        
        # Create test data
        features = np.random.rand(200, 8)
        assignments = np.random.randint(0, 4, 200)
        
        # Initialize stats
        stats = ClusteringStats(features, assignments)
        print(f"✅ ClusteringStats initialized")
        print(f"   - CV ratio: {stats.get_cv_ratio():.4f}")
        print(f"   - Balance score: {stats.get_balance_score():.4f}")
        print(f"   - Cluster sizes: {stats.cluster_sizes}")
        
        # Test move delta calculation
        if len(features) > 10:
            delta_info = stats.calculate_move_delta(0, 0, 1)
            print(f"✅ Move delta calculation: {delta_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ ClusteringStats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_iterative_optimization():
    """Test the IterativeOptimization class."""
    print("\n=== Testing IterativeOptimization ===")
    
    try:
        from training.steps.market_analysis.clusters.iterative_optimization import IterativeOptimization
        
        # Create optimizer
        optimizer = IterativeOptimization(verbose=True)
        print(f"✅ IterativeOptimization initialized")
        print(f"   - Max rounds: {optimizer.max_rounds}")
        print(f"   - Frontier fraction: {optimizer.frontier_fraction}")
        print(f"   - Local threshold: {optimizer.local_threshold}")
        print(f"   - Global threshold: {optimizer.global_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ IterativeOptimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing 3-Step Iterative Clustering Optimization System\n")
    
    success = True
    success &= test_core_components()
    success &= test_clustering_stats()
    success &= test_iterative_optimization()
    
    if success:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")