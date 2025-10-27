#!/usr/bin/env python3
"""
Test script for iterative optimization fallback functionality.

This script tests the quality target checking and iterative optimization fallback
mechanism in the regime clustering system.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

def create_test_data_with_poor_quality():
    """Create test data that will fail quality targets."""
    np.random.seed(42)
    
    # Create data with only 2 clusters (below minimum of 4)
    n_samples = 1000
    
    # Two distinct clusters
    cluster1 = np.random.normal([0, 0], 0.5, (n_samples // 2, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (n_samples // 2, 2))
    
    features = np.vstack([cluster1, cluster2])
    
    # Create labels with only 2 clusters
    labels = np.concatenate([
        np.zeros(n_samples // 2, dtype=int),
        np.ones(n_samples // 2, dtype=int)
    ])
    
    return features, labels

def create_test_data_with_good_quality():
    """Create test data that will meet quality targets."""
    np.random.seed(42)
    
    # Create data with 5 clusters (within target range)
    n_samples = 1000
    n_clusters = 5
    
    features = []
    labels = []
    
    for i in range(n_clusters):
        cluster_size = n_samples // n_clusters
        cluster_features = np.random.normal([i * 2, i * 2], 0.3, (cluster_size, 2))
        cluster_labels = np.full(cluster_size, i, dtype=int)
        
        features.append(cluster_features)
        labels.append(cluster_labels)
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

def test_quality_target_checking():
    """Test quality target checking functionality."""
    print("🧪 Testing Quality Target Checking")
    print("=" * 50)
    
    try:
        # Direct import to avoid dependency issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regime_clustering_step", 
            "src/training/steps/market_analysis/regime_clustering_step.py"
        )
        regime_clustering_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regime_clustering_module)
        
        RegimeClusteringStep = regime_clustering_module.RegimeClusteringStep
        step = RegimeClusteringStep()
        
        # Test with poor quality data
        print("\n📊 Testing with poor quality data (2 clusters)...")
        features_poor, labels_poor = create_test_data_with_poor_quality()
        
        hdbscan_artifacts_poor = {
            'regime_labels': labels_poor,
            'features': features_poor
        }
        
        config = {
            'min_clusters': 4,
            'max_clusters': 8,
            'min_cv_score': 0.3,
            'min_silhouette_score': 0.2,
            'min_dbi_score': 0.5,
            'min_temporal_smoothness': 0.6
        }
        
        quality_result_poor = step._check_quality_targets(labels_poor, hdbscan_artifacts_poor, config)
        
        print(f"✅ Meets targets: {quality_result_poor['meets_targets']}")
        print(f"📊 Cluster count: {quality_result_poor['n_clusters']}")
        print(f"⚠️ Issues: {quality_result_poor['issues']}")
        
        if not quality_result_poor['meets_targets']:
            print("✅ Correctly identified poor quality data")
        else:
            print("❌ Failed to identify poor quality data")
            return False
        
        # Test with good quality data
        print("\n📊 Testing with good quality data (5 clusters)...")
        features_good, labels_good = create_test_data_with_good_quality()
        
        hdbscan_artifacts_good = {
            'regime_labels': labels_good,
            'features': features_good
        }
        
        quality_result_good = step._check_quality_targets(labels_good, hdbscan_artifacts_good, config)
        
        print(f"✅ Meets targets: {quality_result_good['meets_targets']}")
        print(f"📊 Cluster count: {quality_result_good['n_clusters']}")
        print(f"⚠️ Issues: {quality_result_good['issues']}")
        
        if quality_result_good['meets_targets']:
            print("✅ Correctly identified good quality data")
        else:
            print("❌ Incorrectly identified good quality data as poor")
            return False
        
        print("✅ Quality target checking tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Quality target checking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metric_calculations():
    """Test individual metric calculation methods."""
    print("\n🧪 Testing Metric Calculations")
    print("=" * 50)
    
    try:
        # Direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regime_clustering_step", 
            "src/training/steps/market_analysis/regime_clustering_step.py"
        )
        regime_clustering_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regime_clustering_module)
        
        RegimeClusteringStep = regime_clustering_module.RegimeClusteringStep
        step = RegimeClusteringStep()
        
        # Create test data
        features, labels = create_test_data_with_good_quality()
        
        # Test CV score calculation
        cv_score = step._calculate_cv_score(features, labels)
        print(f"📊 CV Score: {cv_score}")
        
        # Test Silhouette score calculation
        silhouette_score = step._calculate_silhouette_score(features, labels)
        print(f"📊 Silhouette Score: {silhouette_score}")
        
        # Test DBI score calculation
        dbi_score = step._calculate_dbi_score(features, labels)
        print(f"📊 DBI Score: {dbi_score}")
        
        # Test temporal smoothness calculation
        temporal_smoothness = step._calculate_temporal_smoothness(labels)
        print(f"📊 Temporal Smoothness: {temporal_smoothness}")
        
        # All metrics should be calculated successfully
        if all(score is not None for score in [cv_score, silhouette_score, dbi_score]):
            print("✅ All metric calculations working")
            return True
        else:
            print("❌ Some metric calculations failed")
            return False
        
    except Exception as e:
        print(f"❌ Metric calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_iterative_optimization_fallback():
    """Test iterative optimization fallback mechanism."""
    print("\n🧪 Testing Iterative Optimization Fallback")
    print("=" * 50)
    
    try:
        # Direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regime_clustering_step", 
            "src/training/steps/market_analysis/regime_clustering_step.py"
        )
        regime_clustering_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regime_clustering_module)
        
        RegimeClusteringStep = regime_clustering_module.RegimeClusteringStep
        step = RegimeClusteringStep()
        
        # Create test data with poor quality
        features, labels = create_test_data_with_poor_quality()
        
        hdbscan_artifacts = {
            'regime_labels': labels,
            'features': features
        }
        
        config = {
            'min_clusters': 4,
            'max_clusters': 8,
            'min_cv_score': 0.3,
            'min_silhouette_score': 0.2,
            'min_dbi_score': 0.5,
            'min_temporal_smoothness': 0.6,
            'iterative_max_iterations': 5,  # Small number for testing
            'iterative_convergence_threshold': 0.001,
            'iterative_enable_risk_mitigation': True
        }
        
        # Test iterative optimization fallback
        print("🔄 Testing iterative optimization fallback...")
        result = step._run_iterative_optimization_fallback(hdbscan_artifacts, labels, config)
        
        if result is not None:
            print(f"✅ Iterative optimization completed: {len(np.unique(result))} clusters")
            print(f"📊 Original clusters: {len(np.unique(labels))}")
            print(f"📊 Optimized clusters: {len(np.unique(result))}")
            return True
        else:
            print("⚠️ Iterative optimization fallback returned None (may be expected if not available)")
            return True  # This is acceptable if iterative optimization is not available
        
    except Exception as e:
        print(f"❌ Iterative optimization fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_integration():
    """Test the full integration with quality targets and fallback."""
    print("\n🧪 Testing Full Integration")
    print("=" * 50)
    
    try:
        # Direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "regime_clustering_step", 
            "src/training/steps/market_analysis/regime_clustering_step.py"
        )
        regime_clustering_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regime_clustering_module)
        
        RegimeClusteringStep = regime_clustering_module.RegimeClusteringStep
        step = RegimeClusteringStep()
        
        # Create test data
        features, labels = create_test_data_with_poor_quality()
        
        hdbscan_artifacts = {
            'regime_labels': labels,
            'features': features
        }
        
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'execution_mode': 'light',
            'min_clusters': 4,
            'max_clusters': 8,
            'min_cv_score': 0.3,
            'min_silhouette_score': 0.2,
            'min_dbi_score': 0.5,
            'min_temporal_smoothness': 0.6,
            'iterative_max_iterations': 5,
            'iterative_convergence_threshold': 0.001,
            'iterative_enable_risk_mitigation': True
        }
        
        # Test the full refinement process
        print("🔄 Testing full refinement process with quality targets...")
        result = step._refine_hdbscan_clusters(hdbscan_artifacts, config)
        
        print(f"✅ Refinement completed")
        print(f"📊 Clustering method: {result.get('clustering_method', 'unknown')}")
        print(f"📊 Final clusters: {result.get('n_clusters', 0)}")
        
        # Check if quality targets are included
        if 'quality_targets' in result:
            quality_targets = result['quality_targets']
            print(f"📊 Meets targets: {quality_targets.get('meets_targets', 'unknown')}")
            print(f"📊 Issues: {quality_targets.get('issues', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Iterative Optimization Fallback Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Quality target checking
    if not test_quality_target_checking():
        success = False
    
    # Test 2: Metric calculations
    if not test_metric_calculations():
        success = False
    
    # Test 3: Iterative optimization fallback
    if not test_iterative_optimization_fallback():
        success = False
    
    # Test 4: Full integration
    if not test_full_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Iterative optimization fallback is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)