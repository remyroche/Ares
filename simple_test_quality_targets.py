#!/usr/bin/env python3
"""
Simple test for quality target checking functionality.

This script tests the quality target checking without heavy dependencies.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

def test_quality_targets():
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
        
        # Test with poor quality data (2 clusters)
        print("\n📊 Testing with poor quality data (2 clusters)...")
        np.random.seed(42)
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
            'min_temporal_smoothness': 0.6
        }
        
        quality_result = step._check_quality_targets(labels, hdbscan_artifacts, config)
        
        print(f"✅ Meets targets: {quality_result['meets_targets']}")
        print(f"📊 Cluster count: {quality_result['n_clusters']}")
        print(f"⚠️ Issues: {quality_result['issues']}")
        
        if not quality_result['meets_targets']:
            print("✅ Correctly identified poor quality data")
        else:
            print("❌ Failed to identify poor quality data")
            return False
        
        # Test with good quality data (5 clusters)
        print("\n📊 Testing with good quality data (5 clusters)...")
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
        
        hdbscan_artifacts = {
            'regime_labels': labels,
            'features': features
        }
        
        quality_result = step._check_quality_targets(labels, hdbscan_artifacts, config)
        
        print(f"✅ Meets targets: {quality_result['meets_targets']}")
        print(f"📊 Cluster count: {quality_result['n_clusters']}")
        print(f"⚠️ Issues: {quality_result['issues']}")
        
        if quality_result['meets_targets']:
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
        np.random.seed(42)
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

def test_temporal_smoothness():
    """Test temporal smoothness calculation specifically."""
    print("\n🧪 Testing Temporal Smoothness")
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
        
        # Test with very smooth data (all same label)
        smooth_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        smoothness = step._calculate_temporal_smoothness(smooth_labels)
        print(f"📊 Smooth labels smoothness: {smoothness}")
        
        if smoothness == 1.0:
            print("✅ Perfect smoothness detected correctly")
        else:
            print("❌ Perfect smoothness not detected")
            return False
        
        # Test with very rough data (alternating labels)
        rough_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        smoothness = step._calculate_temporal_smoothness(rough_labels)
        print(f"📊 Rough labels smoothness: {smoothness}")
        
        if smoothness == 0.0:
            print("✅ Perfect roughness detected correctly")
        else:
            print("❌ Perfect roughness not detected")
            return False
        
        # Test with mixed data
        mixed_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        smoothness = step._calculate_temporal_smoothness(mixed_labels)
        print(f"📊 Mixed labels smoothness: {smoothness}")
        
        if 0.0 < smoothness < 1.0:
            print("✅ Mixed smoothness calculated correctly")
        else:
            print("❌ Mixed smoothness calculation failed")
            return False
        
        print("✅ Temporal smoothness tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal smoothness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Quality Target Testing")
    print("=" * 60)
    
    success = True
    
    # Test 1: Quality target checking
    if not test_quality_targets():
        success = False
    
    # Test 2: Metric calculations
    if not test_metric_calculations():
        success = False
    
    # Test 3: Temporal smoothness
    if not test_temporal_smoothness():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Quality target checking is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 60)