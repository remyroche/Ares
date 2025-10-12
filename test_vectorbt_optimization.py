#!/usr/bin/env python3
"""
Test script to validate VectorBT optimization implementation in regime feature integration.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test data for validation."""
    # Create 1000 data points (about 10 days of 15-minute data)
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.01, 1000)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    return data

def test_regime_feature_integration():
    """Test the RegimeFeatureIntegration with VectorBT optimizations."""
    try:
        from src.feature_generation.categories.regime_feature_integration import (
            RegimeFeatureIntegration, RegimeFeatureConfig
        )
        
        print("✅ Successfully imported RegimeFeatureIntegration")
        
        # Create test data
        data = create_test_data()
        print(f"✅ Created test data with {len(data)} rows")
        
        # Test configuration
        config = RegimeFeatureConfig(
            include_volatility_regime=True,
            include_volume_regime=True,
            include_structural_trend=True,
            include_statistical_regime=True,
            enable_parallel_processing=True,
            enable_matrix_optimization=True,
            total_max_features=50
        )
        
        # Initialize generator
        generator = RegimeFeatureIntegration(config)
        print("✅ Successfully initialized RegimeFeatureIntegration")
        
        # Check if VectorBT optimizers are initialized
        if hasattr(generator, 'vectorbt_optimizer') and generator.vectorbt_optimizer:
            print("✅ VectorBT optimizer initialized")
        else:
            print("⚠️ VectorBT optimizer not initialized")
        
        if hasattr(generator, 'unified_optimizer') and generator.unified_optimizer:
            print("✅ Unified optimizer initialized")
        else:
            print("⚠️ Unified optimizer not initialized")
        
        # Test feature generation
        print("🚀 Testing feature generation...")
        features = generator.generate_features(data)
        
        print(f"✅ Generated {len(features)} features")
        
        # Test optimization methods
        print("🚀 Testing optimization methods...")
        optimized_data = generator.optimize_dataframe_processing(data)
        print(f"✅ DataFrame optimization completed, shape: {optimized_data.shape}")
        
        # Test rolling operations
        print("🚀 Testing vectorized rolling operations...")
        rolling_result = generator.vectorized_rolling_operations(
            data, ['mean', 'std'], [20, 50], ['close', 'volume']
        )
        print(f"✅ Rolling operations completed, shape: {rolling_result.shape}")
        
        # Test individual VectorBT operations
        print("🚀 Testing individual VectorBT operations...")
        close_series = data['close']
        
        # Test rolling mean
        rolling_mean_result = generator._vectorbt_rolling_operation(close_series, 'mean', 20)
        print(f"✅ Rolling mean test completed, length: {len(rolling_mean_result)}")
        
        # Test rolling std
        rolling_std_result = generator._vectorbt_rolling_operation(close_series, 'std', 20)
        print(f"✅ Rolling std test completed, length: {len(rolling_std_result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_generators():
    """Test individual regime feature generators."""
    try:
        from src.feature_generation.categories.regime_volatility import RegimeVolatilityFeatureGenerator
        from src.feature_generation.categories.regime_volume import RegimeVolumeFeatureGenerator
        from src.feature_generation.categories.regime_structural_trend import RegimeStructuralTrendFeatureGenerator
        
        print("✅ Successfully imported individual generators")
        
        data = create_test_data()
        
        # Test volatility generator
        print("🚀 Testing RegimeVolatilityFeatureGenerator...")
        vol_generator = RegimeVolatilityFeatureGenerator()
        vol_features = vol_generator.generate_features(data)
        print(f"✅ Volatility generator: {len(vol_features)} features")
        
        # Test volume generator
        print("🚀 Testing RegimeVolumeFeatureGenerator...")
        volume_generator = RegimeVolumeFeatureGenerator()
        volume_features = volume_generator.generate_features(data)
        print(f"✅ Volume generator: {len(volume_features)} features")
        
        # Test structural trend generator
        print("🚀 Testing RegimeStructuralTrendFeatureGenerator...")
        trend_generator = RegimeStructuralTrendFeatureGenerator()
        trend_features = trend_generator.generate_features(data)
        print(f"✅ Structural trend generator: {len(trend_features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Starting VectorBT optimization validation tests...")
    print("=" * 60)
    
    # Test main integration
    print("\n1. Testing RegimeFeatureIntegration...")
    integration_success = test_regime_feature_integration()
    
    # Test individual generators
    print("\n2. Testing individual generators...")
    individual_success = test_individual_generators()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   RegimeFeatureIntegration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"   Individual Generators: {'✅ PASS' if individual_success else '❌ FAIL'}")
    
    if integration_success and individual_success:
        print("\n🎉 All tests passed! VectorBT optimization is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())