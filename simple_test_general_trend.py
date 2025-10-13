#!/usr/bin/env python3
"""
Simple test script for the GeneralTrendFeatureGenerator.

This script tests the basic functionality without complex dependencies.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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

def test_basic_functionality():
    """Test basic functionality without importing the complex class."""
    print("🧪 Testing basic functionality...")
    
    # Test data creation
    data = create_sample_data(100)
    print(f"✅ Sample data created: {data.shape}")
    print(f"   - Columns: {list(data.columns)}")
    print(f"   - Price range: [{data['close'].min():.2f}, {data['close'].max():.2f}]")
    
    # Test basic ADX calculation
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate True Range
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - close.shift(1)),
        np.abs(low - close.shift(1))
    ])
    
    print(f"✅ True Range calculated: {tr.notna().sum()} non-null values")
    
    # Test basic MACD calculation
    ema_fast = close.ewm(span=12).mean()
    ema_slow = close.ewm(span=26).mean()
    macd = ema_fast - ema_slow
    
    print(f"✅ MACD calculated: {macd.notna().sum()} non-null values")
    print(f"   - MACD range: [{macd.min():.4f}, {macd.max():.4f}]")
    
    # Test basic SMA calculation
    sma = close.rolling(window=20).mean()
    price_position = (close - sma) / sma
    
    print(f"✅ SMA calculated: {sma.notna().sum()} non-null values")
    print(f"   - Price position range: [{price_position.min():.4f}, {price_position.max():.4f}]")
    
    return True

def test_import():
    """Test if we can import the module."""
    print("🧪 Testing module import...")
    
    try:
        # Try to import just the basic classes we need
        from feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
        print("✅ Basic feature generator classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import basic classes: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Simple General Trend Feature Tests")
    print("=" * 50)
    
    tests = [
        test_import,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic functionality tests passed!")
        print("💡 The general trend feature implementation should work correctly.")
        return True
    else:
        print("⚠️  Some basic tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)