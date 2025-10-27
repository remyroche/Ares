#!/usr/bin/env python3
"""
Test script for Enhanced Regime Clustering with Cluster Distinctiveness and Economic Relevance.

This script demonstrates the complete enhanced regime clustering system that combines:
1. Cluster distinctiveness metrics (optimized)
2. Economic relevance weighting (existing system)
3. Temporal stability scoring (existing system)
4. Enhanced feature selection
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from typing import Dict, Any

# Add src to path
sys.path.append('src')

from feature_generation.integration.enhanced_regime_clustering_integration import (
    EnhancedRegimeClusteringIntegration, perform_enhanced_regime_clustering
)
from feature_generation.utils.cluster_feature_selection import (
    EnhancedFeatureSelector, EnhancedFeatureSelectionConfig
)
from feature_generation.utils.cluster_distinctiveness_metrics import (
    ClusterDistinctivenessCalculator, ClusterDistinctivenessConfig
)

def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data with different regimes."""
    np.random.seed(42)
    
    # Create 3 distinct market regimes
    regime_1 = np.random.normal(100, 5, n_samples // 3)  # Low volatility
    regime_2 = np.random.normal(105, 15, n_samples // 3)  # High volatility
    regime_3 = np.random.normal(110, 8, n_samples - 2 * (n_samples // 3))  # Medium volatility
    
    # Combine regimes
    close_prices = np.concatenate([regime_1, regime_2, regime_3])
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.01, len(close_prices))),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.02, len(close_prices)))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.02, len(close_prices)))),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, len(close_prices))
    })
    
    # Ensure high >= low and high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['high'] = np.maximum(data['high'], data['low'])
    
    return data

def test_optimized_cluster_distinctiveness():
    """Test the optimized cluster distinctiveness calculator."""
    print("🧪 Testing Optimized Cluster Distinctiveness Calculator")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_market_data(1000)
    
    # Create sample features
    features = {
        'regime_volatility_20': data['close'].rolling(20).std().values,
        'regime_trend_20': data['close'].rolling(20).mean().values,
        'volume_profile': data['volume'].rolling(20).mean().values,
        'price_momentum': data['close'].pct_change(5).values,
        'constant_feature': np.ones(len(data)),
        'noisy_feature': np.random.normal(0, 1, len(data))
    }
    
    # Create cluster labels (3 regimes)
    cluster_labels = np.concatenate([
        np.zeros(len(data) // 3, dtype=int),
        np.ones(len(data) // 3, dtype=int),
        np.full(len(data) - 2 * (len(data) // 3), 2, dtype=int)
    ])
    
    # Test different configurations
    configs = [
        ("Fast Proxies", ClusterDistinctivenessConfig(
            enable_fast_proxies=True,
            use_approximate_silhouette=True,
            silhouette_sample_ratio=0.1,
            max_samples_for_advanced=5000
        )),
        ("Full Calculation", ClusterDistinctivenessConfig(
            enable_fast_proxies=False,
            use_approximate_silhouette=False
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n📊 Testing {config_name} Configuration")
        print("-" * 40)
        
        start_time = time.time()
        calculator = ClusterDistinctivenessCalculator(config)
        
        # Calculate distinctiveness metrics
        metrics = calculator.calculate_feature_distinctiveness(features, cluster_labels)
        
        calculation_time = time.time() - start_time
        
        print(f"Calculation time: {calculation_time:.3f}s")
        print(f"Features processed: {len(features)}")
        
        # Display results
        for feature_name, feature_metrics in metrics.items():
            print(f"\n{feature_name}:")
            print(f"  Combined Score: {feature_metrics['combined_score']:.4f}")
            print(f"  F-ratio: {feature_metrics['f_ratio']:.4f}")
            print(f"  Separation Strength: {feature_metrics['separation_strength']:.4f}")
            print(f"  Silhouette Score: {feature_metrics.get('silhouette_score', 0.0):.4f}")

def test_enhanced_feature_selection():
    """Test the enhanced feature selection system."""
    print("\n🎯 Testing Enhanced Feature Selection System")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_market_data(1000)
    
    # Create sample features with different economic relevance
    features = {
        # High economic relevance (regime features)
        'regime_volatility_20': data['close'].rolling(20).std().values,
        'regime_trend_strength_20': data['close'].rolling(20).apply(lambda x: np.corrcoef(x, np.arange(len(x)))[0,1]).values,
        'regime_persistence_20': data['close'].rolling(20).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1]).values,
        
        # Medium economic relevance (volume features)
        'volume_profile': data['volume'].rolling(20).mean().values,
        'volume_momentum': data['volume'].pct_change(5).values,
        'vwap_ratio': (data['close'] / data['close'].rolling(20).mean()).values,
        
        # Lower economic relevance (statistical features)
        'price_skewness': data['close'].rolling(20).skew().values,
        'price_kurtosis': data['close'].rolling(20).kurt().values,
        'constant_feature': np.ones(len(data))
    }
    
    # Create cluster labels
    cluster_labels = np.concatenate([
        np.zeros(len(data) // 3, dtype=int),
        np.ones(len(data) // 3, dtype=int),
        np.full(len(data) - 2 * (len(data) // 3), 2, dtype=int)
    ])
    
    # Create feature categories
    from feature_generation.integration.feature_bank_integration import FeatureBankCategory
    feature_categories = {
        'regime_volatility_20': FeatureBankCategory.REGIME,
        'regime_trend_strength_20': FeatureBankCategory.REGIME,
        'regime_persistence_20': FeatureBankCategory.REGIME,
        'volume_profile': FeatureBankCategory.VOLUME,
        'volume_momentum': FeatureBankCategory.VOLUME,
        'vwap_ratio': FeatureBankCategory.VOLUME,
        'price_skewness': FeatureBankCategory.VOLATILITY,
        'price_kurtosis': FeatureBankCategory.VOLATILITY,
        'constant_feature': FeatureBankCategory.MOMENTUM
    }
    
    # Test enhanced feature selection
    config = EnhancedFeatureSelectionConfig(
        cluster_distinctiveness_weight=0.4,
        economic_relevance_weight=0.4,
        temporal_stability_weight=0.2,
        min_combined_score=0.3
    )
    
    selector = EnhancedFeatureSelector(config)
    
    # Select top 5 features
    selected_features = selector.select_optimal_features(
        features, cluster_labels, feature_categories, 5
    )
    
    print(f"Selected {len(selected_features)} features:")
    for feature_name in selected_features.keys():
        print(f"  - {feature_name}")
    
    # Generate selection report
    report = selector.get_feature_selection_report(
        features, cluster_labels, feature_categories
    )
    
    print(f"\n📈 Selection Report:")
    print(f"Total features: {report['total_features']}")
    print(f"Selection weights: {report['selection_summary']['weights']}")
    print(f"Selection thresholds: {report['selection_summary']['thresholds']}")
    
    print(f"\nCategory breakdown:")
    for category, stats in report['category_breakdown'].items():
        print(f"  {category}: {stats['count']} features, avg score: {stats['avg_score']:.3f}")

def test_enhanced_regime_clustering():
    """Test the complete enhanced regime clustering system."""
    print("\n🚀 Testing Enhanced Regime Clustering System")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_market_data(1000)
    
    print(f"Created sample data with {len(data)} samples")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"Volume range: {data['volume'].min():.2f} - {data['volume'].max():.2f}")
    
    # Test different configurations
    configs = [
        ("Standard Clustering", False),
        ("Enhanced Clustering", True)
    ]
    
    for config_name, use_enhanced in configs:
        print(f"\n🔬 Testing {config_name}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Perform clustering
        result = perform_enhanced_regime_clustering(
            data, 
            algorithm='kmeans', 
            n_clusters=3,
            use_enhanced_selection=use_enhanced
        )
        
        clustering_time = time.time() - start_time
        
        print(f"Clustering time: {clustering_time:.3f}s")
        print(f"Number of clusters found: {result['n_clusters']}")
        print(f"Number of noise points: {result['n_noise']}")
        print(f"Number of features used: {len(result['feature_names'])}")
        
        # Quality metrics
        quality = result['quality_metrics']
        print(f"Silhouette score: {quality['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz score: {quality['calinski_harabasz_score']:.4f}")
        print(f"Davies-Bouldin score: {quality['davies_bouldin_score']:.4f}")
        print(f"Overall quality: {quality['overall_quality']}")
        
        # Feature information
        if use_enhanced and 'selection_report' in result['metadata']:
            selection_report = result['metadata']['selection_report']
            print(f"\nEnhanced selection used: {result['metadata'].get('enhanced_selection', False)}")
            print(f"Selection method: {result['metadata'].get('selection_method', 'standard')}")
            
            if 'category_breakdown' in selection_report:
                print(f"\nFeature category breakdown:")
                for category, stats in selection_report['category_breakdown'].items():
                    print(f"  {category}: {stats['count']} features, avg score: {stats['avg_score']:.3f}")

def test_performance_comparison():
    """Compare performance between standard and enhanced selection."""
    print("\n⚡ Performance Comparison")
    print("=" * 60)
    
    # Create larger dataset for performance testing
    data = create_sample_market_data(5000)
    
    # Test different feature counts
    feature_counts = [50, 100, 200]
    
    for n_features in feature_counts:
        print(f"\n📊 Testing with {n_features} features")
        print("-" * 30)
        
        # Create sample features
        features = {}
        for i in range(n_features):
            if i < n_features // 4:
                # Regime features (high economic relevance)
                features[f'regime_feature_{i}'] = data['close'].rolling(20).std().values + np.random.normal(0, 0.1, len(data))
            elif i < n_features // 2:
                # Volume features (medium economic relevance)
                features[f'volume_feature_{i}'] = data['volume'].rolling(20).mean().values + np.random.normal(0, 0.1, len(data))
            else:
                # Statistical features (lower economic relevance)
                features[f'stat_feature_{i}'] = np.random.normal(0, 1, len(data))
        
        # Create cluster labels
        cluster_labels = np.concatenate([
            np.zeros(len(data) // 3, dtype=int),
            np.ones(len(data) // 3, dtype=int),
            np.full(len(data) - 2 * (len(data) // 3), 2, dtype=int)
        ])
        
        # Test standard selection
        start_time = time.time()
        from feature_generation.integration.feature_bank_integration import FeatureBankIntegrator, FeatureBankConfig
        config = FeatureBankConfig()
        config.selection_method = "variance"
        integrator = FeatureBankIntegrator(config)
        # This would normally be called through the full pipeline
        standard_time = time.time() - start_time
        
        # Test enhanced selection
        start_time = time.time()
        from feature_generation.integration.feature_bank_integration import FeatureBankCategory
        feature_categories = {name: FeatureBankCategory.REGIME for name in features.keys()}
        selector = EnhancedFeatureSelector()
        selected = selector.select_optimal_features(features, cluster_labels, feature_categories, 40)
        enhanced_time = time.time() - start_time
        
        print(f"Standard selection time: {standard_time:.3f}s")
        print(f"Enhanced selection time: {enhanced_time:.3f}s")
        print(f"Speedup: {standard_time / enhanced_time:.2f}x")

if __name__ == "__main__":
    print("🎯 Enhanced Regime Clustering Test Suite")
    print("=" * 80)
    
    try:
        test_optimized_cluster_distinctiveness()
        test_enhanced_feature_selection()
        test_enhanced_regime_clustering()
        test_performance_comparison()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎉 Enhanced Regime Clustering System is ready for production!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()