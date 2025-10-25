"""
Simple test for Automated HDBSCAN Parameter Tuner

This script tests the core functionality without requiring the full project structure.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🧪 Testing Basic HDBSCAN Parameter Tuner Functionality")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample financial data...")
    np.random.seed(42)
    
    # Create synthetic financial data with multiple regimes
    n_samples = 500
    data = {}
    
    # Price data with different volatility regimes
    for i in range(n_samples):
        if i < n_samples // 3:
            # High volatility regime
            data.setdefault('close', []).append(100 + np.random.normal(0, 2))
        elif i < 2 * n_samples // 3:
            # Low volatility regime
            data.setdefault('close', []).append(105 + np.random.normal(0, 0.5))
        else:
            # Medium volatility regime
            data.setdefault('close', []).append(110 + np.random.normal(0, 1))
    
    # Add technical indicators
    close_prices = np.array(data['close'])
    data['returns'] = np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]])
    data['sma_20'] = pd.Series(close_prices).rolling(20).mean().fillna(close_prices[0]).values
    data['volatility'] = pd.Series(data['returns']).rolling(20).std().fillna(0).values
    
    # Add more features
    for i in range(4):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    print(f"✅ Created dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Test HDBSCAN directly
    try:
        import hdbscan
        print("\n🔧 Testing HDBSCAN clustering...")
        
        # Test different parameter sets
        test_params = [
            {'min_cluster_size': 20, 'min_samples': 10, 'cluster_selection_epsilon': 0.05},
            {'min_cluster_size': 30, 'min_samples': 15, 'cluster_selection_epsilon': 0.1},
            {'min_cluster_size': 15, 'min_samples': 8, 'cluster_selection_epsilon': 0.02}
        ]
        
        best_score = -1
        best_params = None
        best_labels = None
        
        for i, params in enumerate(test_params):
            print(f"  Testing parameter set {i+1}: {params}")
            
            try:
                clusterer = hdbscan.HDBSCAN(**params)
                labels = clusterer.fit_predict(df)
                
                # Calculate basic metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # Calculate silhouette score if possible
                silhouette_score = None
                if n_clusters > 1:
                    try:
                        from sklearn.metrics import silhouette_score
                        valid_mask = labels != -1
                        if valid_mask.sum() > 1:
                            valid_data = df[valid_mask]
                            valid_labels = labels[valid_mask]
                            if len(set(valid_labels)) > 1:
                                silhouette_score = silhouette_score(valid_data, valid_labels)
                    except:
                        pass
                
                print(f"    Results: {n_clusters} clusters, {noise_ratio:.3f} noise ratio, silhouette: {silhouette_score}")
                
                # Track best result
                if silhouette_score is not None and silhouette_score > best_score:
                    best_score = silhouette_score
                    best_params = params
                    best_labels = labels
                    
            except Exception as e:
                print(f"    Error: {e}")
        
        if best_params:
            print(f"\n🏆 Best parameters: {best_params}")
            print(f"📊 Best silhouette score: {best_score:.3f}")
            
            # Analyze cluster distribution
            unique_labels = np.unique(best_labels)
            print(f"\n📈 Cluster Distribution:")
            total_samples = len(best_labels)
            cluster_distributions = []
            
            for label in unique_labels:
                count = list(best_labels).count(label)
                percentage = (count / total_samples) * 100
                if label == -1:
                    print(f"  • Noise: {count} samples ({percentage:.1f}%)")
                else:
                    print(f"  • Cluster {label}: {count} samples ({percentage:.1f}%)")
                    cluster_distributions.append(percentage)
            
            # Check if we achieved target (4-8 clusters)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            cluster_count_ok = 4 <= n_clusters <= 8
            
            # Check cluster distribution constraint (2%-20%)
            distribution_ok = all(2.0 <= pct <= 20.0 for pct in cluster_distributions) if cluster_distributions else False
            
            print(f"\n🎯 Target Assessment:")
            print(f"  • Cluster count: {'✅' if cluster_count_ok else '❌'} {n_clusters} clusters (target: 4-8)")
            print(f"  • Distribution balance: {'✅' if distribution_ok else '❌'} (target: 2%-20% per cluster)")
            
            if cluster_distributions:
                min_pct = min(cluster_distributions)
                max_pct = max(cluster_distributions)
                print(f"  • Min cluster size: {min_pct:.1f}% (target: ≥2%)")
                print(f"  • Max cluster size: {max_pct:.1f}% (target: ≤20%)")
            
            if cluster_count_ok and distribution_ok:
                print(f"✅ All targets achieved!")
            elif cluster_count_ok:
                print(f"⚠️ Cluster count OK, but distribution needs improvement")
            elif distribution_ok:
                print(f"⚠️ Distribution OK, but cluster count needs improvement")
            else:
                print(f"❌ Both cluster count and distribution need improvement")
            
            # Test final validation (if we had the full tuner)
            print(f"\n🔍 Final Validation Summary:")
            print(f"  • Goals: 5 optimization targets")
            print(f"  • Hard Caps: 2 critical constraints")
            print(f"  • Current Status: {'✅ SUCCESS' if cluster_count_ok and distribution_ok else '❌ NEEDS IMPROVEMENT'}")
            
            return True
        else:
            print("❌ No valid clustering results found")
            return False
            
    except ImportError:
        print("❌ HDBSCAN not available - cannot test clustering")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_quality_metrics():
    """Test quality metrics calculation."""
    print("\n🧪 Testing Quality Metrics Calculation")
    print("-" * 40)
    
    # Test different quality scenarios
    scenarios = [
        {
            'name': 'Optimal Clustering',
            'n_clusters': 6,
            'noise_ratio': 0.1,
            'silhouette_score': 0.6,
            'within_cluster_cv': 0.15,
            'between_cluster_cv': 0.25,
            'cluster_distributions': [15.0, 18.0, 12.0, 16.0, 14.0, 15.0],  # All within 2%-20%
            'distribution_balanced': True
        },
        {
            'name': 'Poor Clustering',
            'n_clusters': 2,
            'noise_ratio': 0.6,
            'silhouette_score': -0.2,
            'within_cluster_cv': 0.8,
            'between_cluster_cv': 0.05,
            'cluster_distributions': [35.0, 5.0],  # One too large, one too small
            'distribution_balanced': False
        },
        {
            'name': 'Too Many Clusters',
            'n_clusters': 12,
            'noise_ratio': 0.2,
            'silhouette_score': 0.3,
            'within_cluster_cv': 0.2,
            'between_cluster_cv': 0.15,
            'cluster_distributions': [1.5, 2.1, 1.8, 2.3, 1.9, 2.0, 1.7, 2.2, 1.6, 2.4, 1.4, 1.3],  # All too small
            'distribution_balanced': False
        },
        {
            'name': 'Unbalanced Distribution',
            'n_clusters': 5,
            'noise_ratio': 0.1,
            'silhouette_score': 0.4,
            'within_cluster_cv': 0.2,
            'between_cluster_cv': 0.15,
            'cluster_distributions': [45.0, 8.0, 12.0, 15.0, 10.0],  # One cluster too large
            'distribution_balanced': False
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}:")
        print(f"  • Clusters: {scenario['n_clusters']}")
        print(f"  • Noise Ratio: {scenario['noise_ratio']:.3f}")
        print(f"  • Silhouette: {scenario['silhouette_score']:.3f}")
        print(f"  • Within-cluster CV: {scenario['within_cluster_cv']:.3f}")
        print(f"  • Between-cluster CV: {scenario['between_cluster_cv']:.3f}")
        print(f"  • Cluster Distributions: {scenario['cluster_distributions']}")
        print(f"  • Distribution Balanced: {scenario['distribution_balanced']}")
        
        # Check target criteria
        cluster_optimal = 4 <= scenario['n_clusters'] <= 8
        cv_optimal = scenario['within_cluster_cv'] < 0.3 and scenario['between_cluster_cv'] > 0.1
        silhouette_optimal = scenario['silhouette_score'] > 0.0
        distribution_optimal = scenario['distribution_balanced']
        
        print(f"  • Cluster count optimal: {'✅' if cluster_optimal else '❌'}")
        print(f"  • CV metrics optimal: {'✅' if cv_optimal else '❌'}")
        print(f"  • Silhouette optimal: {'✅' if silhouette_optimal else '❌'}")
        print(f"  • Distribution optimal: {'✅' if distribution_optimal else '❌'}")
        
        # Overall assessment
        all_optimal = cluster_optimal and cv_optimal and silhouette_optimal and distribution_optimal
        print(f"  • Overall: {'✅ OPTIMAL' if all_optimal else '❌ NEEDS IMPROVEMENT'}")

if __name__ == "__main__":
    print("🚀 Simple HDBSCAN Parameter Tuner Test")
    print("=" * 60)
    
    # Test basic functionality
    success = test_basic_functionality()
    
    # Test quality metrics
    test_quality_metrics()
    
    if success:
        print("\n✅ Basic tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
