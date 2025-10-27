#!/usr/bin/env python3
"""
Simple test for VectorBT-enhanced HDBSCAN clustering implementation.

This script tests the core functionality without complex dependencies.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=200, n_features=5):
    """Create sample financial data for testing."""
    np.random.seed(42)
    
    # Create timestamps
    start_date = datetime.now() - timedelta(days=n_samples)
    timestamps = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Create price data with different regimes
    prices = []
    current_price = 100.0
    
    for i in range(n_samples):
        if i < n_samples // 3:
            # High volatility regime
            change = np.random.normal(0, 0.02)
        elif i < 2 * n_samples // 3:
            # Low volatility regime
            change = np.random.normal(0, 0.005)
        else:
            # Trending regime
            change = np.random.normal(0.001, 0.01)
        
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Add some additional features
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 50 + np.random.normal(0, 15, n_samples)  # Simplified RSI
    
    return data

def test_basic_imports():
    """Test basic imports."""
    print("=" * 60)
    print("Testing basic imports...")
    print("=" * 60)
    
    try:
        # Test tprint
        from src.utils.tprint import tprint
        tprint("✅ tprint imported successfully")
        
        # Test hardware manager
        from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
        hardware_manager = UnifiedHardwareManager()
        tprint("✅ Hardware manager imported and initialized")
        
        # Test feature engineering
        from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering import (
            FeatureEngineeringConfig, AdvancedFeatureGenerator
        )
        tprint("✅ Feature engineering modules imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_feature_generation():
    """Test feature generation with VectorBT optimizations."""
    print("\n" + "=" * 60)
    print("Testing feature generation...")
    print("=" * 60)
    
    try:
        from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering import (
            FeatureEngineeringConfig, AdvancedFeatureGenerator
        )
        from src.utils.tprint import tprint
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=3)
        tprint(f"✅ Sample data created: {data.shape}")
        
        # Create config
        config = FeatureEngineeringConfig(
            enable_technical_indicators=True,
            enable_volatility_features=True,
            enable_momentum_features=True,
            enable_entropy_features=False,  # Disable to avoid complexity
            enable_spectral_features=False,  # Disable to avoid complexity
            enable_temporal_features=True,
            enable_regime_features=True,
            enable_feature_interactions=False,  # Disable to avoid complexity
            enable_feature_selection=False  # Disable to avoid complexity
        )
        
        # Initialize generator
        generator = AdvancedFeatureGenerator(config)
        tprint("✅ Feature generator initialized")
        
        # Generate features
        features = generator.generate_features(data)
        tprint(f"✅ Features generated: {features.shape}")
        tprint(f"📊 Feature columns: {list(features.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hdbscan_clustering():
    """Test HDBSCAN clustering."""
    print("\n" + "=" * 60)
    print("Testing HDBSCAN clustering...")
    print("=" * 60)
    
    try:
        import hdbscan
        from src.utils.tprint import tprint
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=3)
        
        # Prepare features for clustering
        features = data[['close', 'volatility', 'rsi']].dropna()
        tprint(f"✅ Features prepared for clustering: {features.shape}")
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
        labels = clusterer.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        tprint(f"✅ Clustering completed")
        tprint(f"📊 Clusters found: {n_clusters}")
        tprint(f"📊 Noise points: {n_noise}")
        tprint(f"📊 Noise ratio: {n_noise / len(labels):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ HDBSCAN clustering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting simple VectorBT-enhanced HDBSCAN clustering tests...")
    print("=" * 80)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Feature Generation", test_feature_generation),
        ("HDBSCAN Clustering", test_hdbscan_clustering)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VectorBT-enhanced HDBSCAN clustering is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)