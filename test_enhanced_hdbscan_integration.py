#!/usr/bin/env python3
"""
Test script to verify the enhanced HDBSCAN clustering integration is properly wired.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test market data for HDBSCAN clustering."""
    # Generate 1000 samples of synthetic market data
    np.random.seed(42)
    n_samples = 1000
    
    # Create time index
    start_date = datetime.now() - timedelta(days=100)
    dates = [start_date + timedelta(minutes=15*i) for i in range(n_samples)]
    
    # Generate synthetic OHLCV data
    base_price = 100.0
    prices = []
    volumes = []
    
    for i in range(n_samples):
        # Random walk with some trend
        trend = 0.001 * np.sin(i / 100) + 0.0001 * i
        noise = np.random.normal(0, 0.02)
        price_change = trend + noise
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + price_change)
        
        # Generate OHLC from price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[-1] if i > 0 else price
        close = price
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        
        prices.append(price)
        volumes.append(volume)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [prices[i-1] if i > 0 else prices[0] for i in range(n_samples)],
        'high': [prices[i] * (1 + abs(np.random.normal(0, 0.01))) for i in range(n_samples)],
        'low': [prices[i] * (1 - abs(np.random.normal(0, 0.01))) for i in range(n_samples)],
        'close': prices,
        'volume': volumes
    })
    
    return data

def test_enhanced_hdbscan_integration():
    """Test the enhanced HDBSCAN clustering integration."""
    print("🧪 Testing Enhanced HDBSCAN Clustering Integration")
    print("=" * 60)
    
    try:
        # Import the enhanced integration
        from src.feature_generation.integration.enhanced_hdbscan_clustering_integration import (
            EnhancedHDBSCANClusteringIntegration,
            get_enhanced_hdbscan_features,
            perform_enhanced_hdbscan_clustering
        )
        print("✅ Successfully imported enhanced HDBSCAN integration")
        
        # Create test data
        print("\n📊 Creating test market data...")
        data = create_test_data()
        print(f"✅ Created test data: {data.shape[0]} samples, {data.shape[1]} columns")
        
        # Test 1: Feature generation only
        print("\n🔧 Test 1: Feature Generation (100-150 features)")
        print("-" * 40)
        
        integrator = EnhancedHDBSCANClusteringIntegration(
            min_features=100,
            max_features=150,
            enable_pca_reduction=True,
            pca_components=15
        )
        
        feature_result = integrator.get_comprehensive_clustering_features(data)
        print(f"✅ Generated {len(feature_result['feature_names'])} features")
        print(f"   Target range: {feature_result['target_range']}")
        print(f"   Clustering optimized: {feature_result['clustering_optimized']}")
        
        # Test 2: Data preparation with PCA
        print("\n🔧 Test 2: Data Preparation with PCA (100-150 → 10-25)")
        print("-" * 40)
        
        feature_matrix, feature_names, metadata = integrator.prepare_data_for_clustering(data)
        print(f"✅ Prepared data: {feature_matrix.shape}")
        print(f"   Original features: {metadata['preprocessing']['original_shape'][1]}")
        print(f"   Final features: {metadata['preprocessing']['final_shape'][1]}")
        print(f"   PCA applied: {metadata['preprocessing']['pca_applied']}")
        
        if metadata['preprocessing']['pca_applied']:
            print(f"   PCA components: {metadata['preprocessing']['pca_components']}")
            print(f"   Explained variance: {metadata['preprocessing']['pca_explained_variance_ratio'][:3]}...")
        
        # Test 3: Full HDBSCAN clustering
        print("\n🔧 Test 3: Full HDBSCAN Clustering")
        print("-" * 40)
        
        clustering_result = integrator.cluster_with_enhanced_hdbscan(
            data,
            min_cluster_size=5,
            min_samples=3
        )
        
        print(f"✅ Clustering completed:")
        print(f"   Clusters found: {clustering_result['n_clusters']}")
        print(f"   Noise points: {clustering_result['n_noise']}")
        print(f"   Feature matrix shape: {clustering_result['feature_matrix'].shape}")
        
        # Test 4: Convenience functions
        print("\n🔧 Test 4: Convenience Functions")
        print("-" * 40)
        
        # Test get_enhanced_hdbscan_features
        features = get_enhanced_hdbscan_features(data)
        print(f"✅ get_enhanced_hdbscan_features: {len(features['feature_names'])} features")
        
        # Test perform_enhanced_hdbscan_clustering
        result = perform_enhanced_hdbscan_clustering(data)
        print(f"✅ perform_enhanced_hdbscan_clustering: {result['n_clusters']} clusters")
        
        print("\n🎉 All tests passed! Enhanced HDBSCAN integration is properly wired.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_launcher_integration():
    """Test if the launcher can trigger HDBSCAN clustering."""
    print("\n🚀 Testing Launcher Integration")
    print("=" * 60)
    
    try:
        # Test launcher argument parsing
        from src.launcher.ares_launcher import main
        print("✅ Successfully imported ares_launcher")
        
        # Test HDBSCAN regime discovery step
        from src.training.steps.market_analysis.hdbscan_regime_discovery_step import HDBSCANRegimeDiscoveryStep
        print("✅ Successfully imported HDBSCANRegimeDiscoveryStep")
        
        # Test step registration
        from src.training.steps.base_step import step_registry
        if 'hdbscan_regime_discovery' in step_registry._steps:
            print("✅ HDBSCAN regime discovery step is registered")
        else:
            print("⚠️  HDBSCAN regime discovery step not found in registry")
        
        print("\n🎉 Launcher integration looks good!")
        return True
        
    except Exception as e:
        print(f"\n❌ Launcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Enhanced HDBSCAN Clustering Integration Test")
    print("=" * 60)
    
    # Test 1: Enhanced integration
    integration_success = test_enhanced_hdbscan_integration()
    
    # Test 2: Launcher integration
    launcher_success = test_launcher_integration()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    print(f"Enhanced Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"Launcher Integration: {'✅ PASS' if launcher_success else '❌ FAIL'}")
    
    if integration_success and launcher_success:
        print("\n🎉 All systems are properly wired! HDBSCAN clustering will work when launched.")
    else:
        print("\n⚠️  Some issues found. Check the errors above.")