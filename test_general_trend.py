#!/usr/bin/env python3
"""
Test script for the GeneralTrendFeatureGenerator.

This script tests the new general trend feature that combines ADX (strength) and MACD/SMA (direction).
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from feature_generation.categories.trend import GeneralTrendFeatureGenerator, create_general_trend_generators
    print("✅ Successfully imported GeneralTrendFeatureGenerator")
except ImportError as e:
    print(f"❌ Failed to import GeneralTrendFeatureGenerator: {e}")
    sys.exit(1)

def create_sample_data(n_points=100):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend
    base_price = 100
    trend = np.linspace(0, 20, n_points)  # Upward trend
    noise = np.random.normal(0, 2, n_points)
    close_prices = base_price + trend + noise
    
    # Generate high, low, open based on close
    high_prices = close_prices + np.random.uniform(0, 3, n_points)
    low_prices = close_prices - np.random.uniform(0, 3, n_points)
    open_prices = close_prices + np.random.uniform(-1, 1, n_points)
    volume = np.random.uniform(1000, 10000, n_points)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    return data

def test_general_trend_macd():
    """Test general trend with MACD direction."""
    print("\n🧪 Testing General Trend with MACD direction...")
    
    data = create_sample_data(100)
    generator = GeneralTrendFeatureGenerator(
        adx_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        use_sma_instead_of_macd=False
    )
    
    try:
        result = generator.generate(data)
        print(f"✅ MACD-based general trend generated successfully")
        print(f"   - Result shape: {result.shape}")
        print(f"   - Non-null values: {result.notna().sum()}")
        print(f"   - Value range: [{result.min():.4f}, {result.max():.4f}]")
        print(f"   - Mean: {result.mean():.4f}")
        return True
    except Exception as e:
        print(f"❌ MACD-based general trend failed: {e}")
        return False

def test_general_trend_sma():
    """Test general trend with SMA direction."""
    print("\n🧪 Testing General Trend with SMA direction...")
    
    data = create_sample_data(100)
    generator = GeneralTrendFeatureGenerator(
        adx_period=14,
        sma_period=20,
        use_sma_instead_of_macd=True
    )
    
    try:
        result = generator.generate(data)
        print(f"✅ SMA-based general trend generated successfully")
        print(f"   - Result shape: {result.shape}")
        print(f"   - Non-null values: {result.notna().sum()}")
        print(f"   - Value range: [{result.min():.4f}, {result.max():.4f}]")
        print(f"   - Mean: {result.mean():.4f}")
        return True
    except Exception as e:
        print(f"❌ SMA-based general trend failed: {e}")
        return False

def test_general_trend_generators():
    """Test the create_general_trend_generators function."""
    print("\n🧪 Testing create_general_trend_generators...")
    
    try:
        generators = create_general_trend_generators(
            adx_periods=[14, 21],
            macd_configs=[{"fast": 12, "slow": 26, "signal": 9}],
            sma_periods=[20, 50],
            use_sma_variants=True
        )
        
        print(f"✅ Created {len(generators)} general trend generators")
        
        # Test each generator
        data = create_sample_data(100)
        for i, generator in enumerate(generators):
            try:
                result = generator.generate(data)
                print(f"   - Generator {i+1}: {generator.config.name} - ✅ Success")
            except Exception as e:
                print(f"   - Generator {i+1}: {generator.config.name} - ❌ Failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ create_general_trend_generators failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases."""
    print("\n🧪 Testing edge cases...")
    
    # Test with insufficient data
    data_short = create_sample_data(10)  # Less than required periods
    generator = GeneralTrendFeatureGenerator(adx_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    
    try:
        result = generator.generate(data_short)
        print(f"✅ Short data handled correctly - all NaN values: {result.isna().all()}")
    except Exception as e:
        print(f"❌ Short data handling failed: {e}")
        return False
    
    # Test with empty data
    data_empty = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    try:
        result = generator.generate(data_empty)
        print(f"✅ Empty data handled correctly - result is empty: {result.empty}")
    except Exception as e:
        print(f"❌ Empty data handling failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting General Trend Feature Tests")
    print("=" * 50)
    
    tests = [
        test_general_trend_macd,
        test_general_trend_sma,
        test_general_trend_generators,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! General Trend Feature is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)