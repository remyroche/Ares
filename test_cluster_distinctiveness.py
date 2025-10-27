#!/usr/bin/env python3
"""
Test script for cluster distinctiveness metrics.

This script demonstrates how to use the cluster distinctiveness metrics
to select features for regime clustering.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from feature_generation.utils.cluster_distinctiveness_metrics import (
    ClusterDistinctivenessCalculator, ClusterDistinctivenessConfig,
    calculate_cluster_distinctiveness, rank_features_by_cluster_distinctiveness
)

def create_sample_data():
    """Create sample data with different feature types."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create 3 distinct clusters
    cluster_0 = np.random.normal(0, 1, 300)
    cluster_1 = np.random.normal(5, 1, 400)
    cluster_2 = np.random.normal(10, 1, 300)
    
    # Create cluster labels
    cluster_labels = np.concatenate([
        np.zeros(300, dtype=int),
        np.ones(400, dtype=int),
        np.full(300, 2, dtype=int)
    ])
    
    # Create features with different distinctiveness levels
    features = {}
    
    # High distinctiveness feature (separates clusters well)
    features['high_distinctiveness'] = np.concatenate([cluster_0, cluster_1, cluster_2])
    
    # Medium distinctiveness feature (some separation)
    features['medium_distinctiveness'] = np.concatenate([
        np.random.normal(0, 2, 300),
        np.random.normal(3, 2, 400),
        np.random.normal(6, 2, 300)
    ])
    
    # Low distinctiveness feature (poor separation)
    features['low_distinctiveness'] = np.random.normal(0, 1, n_samples)
    
    # Constant feature (no distinctiveness)
    features['constant'] = np.ones(n_samples)
    
    # Noisy feature (high variance but no cluster structure)
    features['noisy'] = np.random.normal(0, 5, n_samples)
    
    return features, cluster_labels

def test_cluster_distinctiveness():
    """Test cluster distinctiveness calculation."""
    print("🧪 Testing Cluster Distinctiveness Metrics")
    print("=" * 50)
    
    # Create sample data
    features, cluster_labels = create_sample_data()
    
    print(f"Created sample data with {len(features)} features and {len(cluster_labels)} samples")
    print(f"Number of clusters: {len(set(cluster_labels))}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    print()
    
    # Test different configurations
    configs = [
        ("Basic", ClusterDistinctivenessConfig(enable_advanced_metrics=False)),
        ("Advanced", ClusterDistinctivenessConfig(enable_advanced_metrics=True)),
        ("Scaled", ClusterDistinctivenessConfig(enable_scaling=True, enable_advanced_metrics=True))
    ]
    
    for config_name, config in configs:
        print(f"📊 Testing {config_name} Configuration")
        print("-" * 30)
        
        # Calculate distinctiveness metrics
        calculator = ClusterDistinctivenessCalculator(config)
        metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
        
        # Display results
        for feature_name, feature_metrics in metrics.items():
            print(f"\n{feature_name}:")
            for metric_name, value in feature_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # Rank features
        ranked_features = calculator.rank_features_by_distinctiveness(features, cluster_labels)
        print(f"\n🏆 Feature Ranking ({config_name}):")
        for i, (feature_name, score) in enumerate(ranked_features, 1):
            print(f"  {i}. {feature_name}: {score:.4f}")
        
        print("\n" + "=" * 50)

def test_feature_selection():
    """Test feature selection using cluster distinctiveness."""
    print("🎯 Testing Feature Selection")
    print("=" * 50)
    
    # Create sample data
    features, cluster_labels = create_sample_data()
    
    # Test selecting top 3 features
    config = ClusterDistinctivenessConfig(enable_advanced_metrics=True, enable_scaling=True)
    calculator = ClusterDistinctivenessCalculator(config)
    
    selected_features = calculator.get_top_distinctive_features(features, cluster_labels, 3)
    
    print(f"Selected {len(selected_features)} features:")
    for feature_name in selected_features.keys():
        print(f"  - {feature_name}")
    
    # Show the distinctiveness scores for selected features
    print("\nDistinctiveness scores for selected features:")
    metrics = calculator.calculate_feature_distinctiveness(selected_features, cluster_labels)
    for feature_name, feature_metrics in metrics.items():
        print(f"\n{feature_name}:")
        print(f"  Combined Score: {feature_metrics['combined_score']:.4f}")
        print(f"  F-ratio: {feature_metrics['f_ratio']:.4f}")
        print(f"  Separation Strength: {feature_metrics['separation_strength']:.4f}")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("⚠️ Testing Edge Cases")
    print("=" * 50)
    
    config = ClusterDistinctivenessConfig()
    calculator = ClusterDistinctivenessCalculator(config)
    
    # Test with insufficient clusters
    print("Testing with insufficient clusters...")
    features = {'test': np.random.normal(0, 1, 100)}
    cluster_labels = np.zeros(100)  # All same cluster
    metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
    print(f"Result: {metrics}")
    
    # Test with invalid feature
    print("\nTesting with invalid feature...")
    features = {'constant': np.ones(100)}  # Constant values
    cluster_labels = np.random.choice([0, 1], 100)
    metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
    print(f"Result: {metrics}")
    
    # Test with empty features
    print("\nTesting with empty features...")
    features = {}
    cluster_labels = np.random.choice([0, 1], 100)
    metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
    print(f"Result: {metrics}")

if __name__ == "__main__":
    print("🚀 Cluster Distinctiveness Metrics Test Suite")
    print("=" * 60)
    
    try:
        test_cluster_distinctiveness()
        test_feature_selection()
        test_edge_cases()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()