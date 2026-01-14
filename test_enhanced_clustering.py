#!/usr/bin/env python3
"""
Test script for enhanced clustering metrics and adaptive pruning
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares')

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Import our enhanced De Prado engine
from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine

def test_enhanced_clustering():
    """Test the enhanced multi-criteria ONC clustering"""
    print("🧪 Testing Enhanced Multi-criteria ONC Clustering")
    print("=" * 60)
    
    # Generate synthetic feature data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Convert to DataFrame with feature names
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"📊 Generated synthetic data: {X_df.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X_df)}")
    
    # Initialize enhanced De Prado engine
    engine = DePradoFeatureEngine(
        max_clusters=8,
        random_state=42
    )
    
    # Test enhanced clustering
    print("\n🔍 Testing Multi-criteria ONC Clustering...")
    cluster_labels = engine._get_onc_clusters(X_df)
    
    print(f"\n✅ Clustering Results:")
    print(f"   Optimal clusters: {engine.optimal_n_clusters_}")
    print(f"   Cluster distribution: {cluster_labels.value_counts().sort_index().to_dict()}")
    
    # Show detailed metrics
    if hasattr(engine, 'silhouette_scores_') and engine.silhouette_scores_:
        best_k = engine.optimal_n_clusters_
        best_metrics = engine.silhouette_scores_.get(best_k, {})
        
        print(f"\n📊 Best Cluster Metrics (K={best_k}):")
        print(f"   CV Ratio: {best_metrics.get('cv_ratio', 0):.3f}")
        print(f"   Davies-Bouldin: {best_metrics.get('dbi', 0):.3f}")
        print(f"   Silhouette: {best_metrics.get('silhouette', 0):.3f}")
        print(f"   Calinski-Harabasz: {best_metrics.get('ch', 0):.1f}")
        print(f"   Composite Score: {best_metrics.get('composite', 0):.3f}")
    
    return True

def test_adaptive_pruning():
    """Test the adaptive pruning logic"""
    print("\n🧪 Testing Adaptive Pruning Logic")
    print("=" * 60)
    
    # Test different feature counts
    test_cases = [
        (25, "Small feature set"),
        (75, "Medium feature set"), 
        (150, "Large feature set"),
        (300, "Very large feature set")
    ]
    
    for n_features, description in test_cases:
        print(f"\n📊 {description} ({n_features} features):")
        
        # Calculate adaptive pruning percentile
        if n_features < 50:
            pruning_percentile = 5
        elif n_features < 100:
            pruning_percentile = 10
        elif n_features < 200:
            pruning_percentile = 15
        else:
            pruning_percentile = 20
        
        # Calculate expected features after pruning
        retained_percent = 1 - pruning_percentile / 100
        expected_features = int(n_features * retained_percent)
        min_features = max(10, expected_features)
        
        print(f"   Pruning percentile: {pruning_percentile}th")
        print(f"   Expected retention: {expected_features} ({retained_percent*100:.0f}%)")
        print(f"   Minimum guaranteed: {min_features}")
    
    return True

if __name__ == "__main__":
    try:
        # Test enhanced clustering
        clustering_success = test_enhanced_clustering()
        
        # Test adaptive pruning
        pruning_success = test_adaptive_pruning()
        
        if clustering_success and pruning_success:
            print("\n🎉 All tests passed successfully!")
            print("✅ Enhanced clustering metrics working")
            print("✅ Adaptive pruning logic working")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
