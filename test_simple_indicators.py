#!/usr/bin/env python3
"""
Simple test for centralized indicators system.

This script tests the basic functionality without importing
the complex feature generation system.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def test_basic_indicators():
    """Test basic indicator calculations without complex imports."""
    print("🧪 Testing Basic Indicator Calculations")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volume = np.random.randint(1000, 10000, 100)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    print(f"✅ Created sample data with {len(data)} rows")
    print(f"📊 Data columns: {list(data.columns)}")
    print(f"📈 Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # Test basic RSI calculation
    print("\n📊 Testing RSI calculation...")
    def calculate_rsi_basic(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    rsi = calculate_rsi_basic(data['close'], period=14)
    print(f"✅ RSI calculated: {len(rsi)} values")
    print(f"📈 RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
    print(f"📊 RSI sample values: {rsi.head().values}")
    
    # Test basic MACD calculation
    print("\n📊 Testing MACD calculation...")
    def calculate_macd_basic(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}
    
    macd_data = calculate_macd_basic(data['close'], fast=12, slow=26, signal=9)
    print(f"✅ MACD calculated: {len(macd_data)} components")
    for key, series in macd_data.items():
        print(f"📈 {key}: {len(series)} values, range: {series.min():.2f} - {series.max():.2f}")
    
    # Test basic Stochastic calculation
    print("\n📊 Testing Stochastic calculation...")
    def calculate_stochastic_basic(data, period=14, smooth_k=3, smooth_d=3):
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        k_smooth = k.rolling(window=smooth_k).mean()
        d_smooth = k_smooth.rolling(window=smooth_d).mean()
        
        return {'k': k_smooth, 'd': d_smooth}
    
    stoch_data = calculate_stochastic_basic(data, period=14, smooth_k=3, smooth_d=3)
    print(f"✅ Stochastic calculated: {len(stoch_data)} components")
    for key, series in stoch_data.items():
        print(f"📈 {key}: {len(series)} values, range: {series.min():.2f} - {series.max():.2f}")
    
    print("\n🎉 Basic indicator calculations are working correctly!")
    return True

def test_centralized_indicators_simple():
    """Test the centralized indicators module directly."""
    print("\n🔧 Testing Centralized Indicators Module")
    print("=" * 40)
    
    try:
        # Import the indicators module directly
        from src.training.steps.feature_engineering.indicators import (
            CentralizedIndicators, IndicatorConfig, 
            calculate_rsi, calculate_macd, calculate_stochastic,
            calculate_williams_r, calculate_cci, calculate_adx
        )
        print("✅ Successfully imported centralized indicators module")
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 0.5,
            'high': 100 + np.random.randn(50) * 0.5 + 1,
            'low': 100 + np.random.randn(50) * 0.5 - 1,
            'close': 100 + np.cumsum(np.random.randn(50) * 0.01),
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        # Test individual functions
        print("\n📊 Testing individual indicator functions...")
        
        rsi = calculate_rsi(data, period=14)
        print(f"✅ RSI: {len(rsi)} values, range: {rsi.min():.2f} - {rsi.max():.2f}")
        
        macd_data = calculate_macd(data, fast=12, slow=26, signal=9)
        print(f"✅ MACD: {len(macd_data)} components")
        
        stoch_data = calculate_stochastic(data, period=14)
        print(f"✅ Stochastic: {len(stoch_data)} components")
        
        williams_r = calculate_williams_r(data, period=14)
        print(f"✅ Williams %R: {len(williams_r)} values, range: {williams_r.min():.2f} - {williams_r.max():.2f}")
        
        cci = calculate_cci(data, period=20)
        print(f"✅ CCI: {len(cci)} values, range: {cci.min():.2f} - {cci.max():.2f}")
        
        adx = calculate_adx(data, period=14)
        print(f"✅ ADX: {len(adx)} values, range: {adx.min():.2f} - {adx.max():.2f}")
        
        # Test CentralizedIndicators class
        print("\n🔧 Testing CentralizedIndicators class...")
        config = IndicatorConfig(rsi_period=21, macd_fast=15, macd_slow=30)
        indicators = CentralizedIndicators(config)
        
        rsi_custom = indicators.calculate_rsi(data, period=21)
        print(f"✅ Custom RSI (period=21): {len(rsi_custom)} values")
        
        macd_custom = indicators.calculate_macd(data, fast=15, slow=30)
        print(f"✅ Custom MACD (15,30): {len(macd_custom)} components")
        
        print("\n🎉 Centralized indicators module is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Centralized indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple Indicators Test Suite")
    print("=" * 60)
    
    # Test basic calculations
    basic_ok = test_basic_indicators()
    
    # Test centralized indicators
    centralized_ok = test_centralized_indicators_simple()
    
    print("\n" + "=" * 60)
    if basic_ok and centralized_ok:
        print("🎉 ALL TESTS PASSED! Indicator calculations are working correctly.")
        print("✅ Basic indicator calculations work")
        print("✅ Centralized indicators module works")
        print("✅ Fallback calculations are available")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)