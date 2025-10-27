#!/usr/bin/env python3
"""
Test script for VectorBT-enhanced HDBSCAN clustering implementation.

This script tests the enhanced HDBSCAN clustering with VectorBT optimizations
and comprehensive tprint logging.
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

def create_sample_data(n_samples=1000, n_features=10):
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

def test_vectorbt_imports():
    """Test VectorBT imports and availability."""
    print("=" * 60)
    print("Testing VectorBT imports and availability...")
    print("=" * 60)
    
    try:
        from src.vectorbt import (
            vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
            rolling_sum, rolling_apply, VECTORBT_AVAILABLE
        )
        
        print(f"✅ VectorBT available: {VECTORBT_AVAILABLE}")
        if VECTORBT_AVAILABLE:
            print("✅ All VectorBT functions imported successfully")
        else:
            print("⚠️ VectorBT not available - will use pandas fallback")
            
    except ImportError as e:
        print(f"❌ VectorBT import failed: {e}")
        return False
    
    return True

def test_tprint_imports():
    """Test tprint system imports."""
    print("\n" + "=" * 60)
    print("Testing tprint system imports...")
    print("=" * 60)
    
    try:
        from src.utils.tprint import (
            tprint, tprint_data_preview, tprint_data_format, 
            tprint_performance
        )
        
        print("✅ tprint system imported successfully")
        
        # Test basic tprint functionality
        tprint("🧪 Testing tprint functionality")
        tprint_data_format("Test data", {"key": "value", "number": 42})
        
    except ImportError as e:
        print(f"❌ tprint import failed: {e}")
        return False
    
    return True

def test_hardware_imports():
    """Test hardware optimization imports."""
    print("\n" + "=" * 60)
    print("Testing hardware optimization imports...")
    print("=" * 60)
    
    try:
        from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
        from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
        
        print("✅ Hardware optimization modules imported successfully")
        
        # Test hardware manager initialization
        hardware_manager = UnifiedHardwareManager()
        print(f"✅ Hardware manager initialized: {type(hardware_manager).__name__}")
        
    except ImportError as e:
        print(f"❌ Hardware optimization import failed: {e}")
        return False
    
    return True

def test_enhanced_feature_engineering():
    """Test the enhanced feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("Testing enhanced feature engineering pipeline...")
    print("=" * 60)
    
    try:
        from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering import (
            EnhancedFeatureEngineeringPipeline, FeatureEngineeringConfig
        )
        
        # Create sample data
        data = create_sample_data(n_samples=500, n_features=5)
        print(f"✅ Sample data created: {data.shape}")
        
        # Create feature engineering config
        config = FeatureEngineeringConfig(
            enable_technical_indicators=True,
            enable_volatility_features=True,
            enable_momentum_features=True,
            enable_entropy_features=True,
            enable_spectral_features=True,
            enable_temporal_features=True,
            enable_regime_features=True,
            enable_feature_interactions=True,
            enable_feature_selection=True,
            max_features=20
        )
        
        # Initialize pipeline
        pipeline = EnhancedFeatureEngineeringPipeline(config)
        print("✅ Enhanced feature engineering pipeline initialized")
        
        # Process features
        processed_features, processing_info = pipeline.process_features(data)
        print(f"✅ Features processed: {processed_features.shape}")
        print(f"📊 Processing info: {processing_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_regime_discovery():
    """Test the enhanced regime discovery system."""
    print("\n" + "=" * 60)
    print("Testing enhanced regime discovery system...")
    print("=" * 60)
    
    try:
        from src.training.steps.market_analysis.hdbscan_clustering.enhanced_regime_discovery import (
            EnhancedHDBSCANRegimeDiscovery
        )
        
        # Create sample data
        data = create_sample_data(n_samples=300, n_features=5)
        print(f"✅ Sample data created: {data.shape}")
        
        # Initialize regime discovery
        regime_discovery = EnhancedHDBSCANRegimeDiscovery()
        print("✅ Enhanced regime discovery initialized")
        
        # Discover regimes
        result = regime_discovery.discover_regimes(data)
        print(f"✅ Regime discovery completed")
        print(f"📊 Clusters found: {result.n_clusters}")
        print(f"📊 Noise ratio: {result.noise_ratio:.3f}")
        print(f"📊 Quality validation passed: {result.validation_results['overall_passed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Regime discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting VectorBT-enhanced HDBSCAN clustering tests...")
    print("=" * 80)
    
    tests = [
        ("VectorBT Imports", test_vectorbt_imports),
        ("tprint System", test_tprint_imports),
        ("Hardware Optimizations", test_hardware_imports),
        ("Feature Engineering", test_enhanced_feature_engineering),
        ("Regime Discovery", test_enhanced_regime_discovery)
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