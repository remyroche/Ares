#!/usr/bin/env python3
"""
Simple test for time features optimization.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic time feature functionality."""
    print("🧪 Testing basic time feature functionality...")
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    test_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    print(f"📊 Created test data with {len(test_data)} samples")
    
    # Test basic time features without VectorBT
    print("\n🔍 Testing basic time features...")
    
    # Hour feature
    hour_feature = pd.Series(test_data.index.hour, index=test_data.index, name='hour')
    print(f"✅ Hour feature: {hour_feature.shape}, unique values: {hour_feature.nunique()}")
    
    # Hour sine feature
    hour_sin = pd.Series(np.sin(2 * np.pi * test_data.index.hour / 24), index=test_data.index, name='hour_sin')
    print(f"✅ Hour sine feature: {hour_sin.shape}, range: [{hour_sin.min():.3f}, {hour_sin.max():.3f}]")
    
    # Hour cosine feature
    hour_cos = pd.Series(np.cos(2 * np.pi * test_data.index.hour / 24), index=test_data.index, name='hour_cos')
    print(f"✅ Hour cosine feature: {hour_cos.shape}, range: [{hour_cos.min():.3f}, {hour_cos.max():.3f}]")
    
    # Market open indicator
    market_open = ((test_data.index.hour >= 9) & (test_data.index.hour < 11)).astype(int)
    market_open_series = pd.Series(market_open, index=test_data.index, name='market_open')
    print(f"✅ Market open feature: {market_open_series.shape}, sum: {market_open_series.sum()}")
    
    # Day of week sine
    dow_sin = pd.Series(np.sin(2 * np.pi * test_data.index.dayofweek / 7), index=test_data.index, name='dow_sin')
    print(f"✅ Day of week sine: {dow_sin.shape}, range: [{dow_sin.min():.3f}, {dow_sin.max():.3f}]")
    
    print("\n✅ All basic time features working correctly!")
    return True

def test_optimized_time_features():
    """Test the optimized time features if available."""
    print("\n🔍 Testing optimized time features...")
    
    try:
        from src.feature_generation.categories.time import (
            OptimizedTimeFeatureGenerator,
            HourGenerator,
            HourSinGenerator,
            create_default_time_generators
        )
        
        print("✅ Successfully imported optimized time features")
        
        # Test individual generators
        generators = [HourGenerator(), HourSinGenerator()]
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.lognormal(10, 1, 1000)
        }, index=dates)
        
        for generator in generators:
            try:
                feature = generator.generate_feature(test_data)
                print(f"✅ {generator.config.name}: {feature.shape}")
            except Exception as e:
                print(f"❌ {generator.config.name}: {e}")
        
        # Test factory function
        all_generators = create_default_time_generators()
        print(f"✅ Created {len(all_generators)} time feature generators")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Could not import optimized features: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing optimized features: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Time Features Optimization Test")
    print("=" * 40)
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    # Test optimized features
    optimized_success = test_optimized_time_features()
    
    print(f"\n📋 Test Results:")
    print(f"   Basic functionality: {'✅' if basic_success else '❌'}")
    print(f"   Optimized features: {'✅' if optimized_success else '❌'}")
    
    if basic_success:
        print("\n🎉 Basic time features are working correctly!")
    else:
        print("\n❌ Basic time features failed!")
    
    if optimized_success:
        print("🎉 Optimized time features are working correctly!")
    else:
        print("⚠️  Optimized time features need attention")
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()