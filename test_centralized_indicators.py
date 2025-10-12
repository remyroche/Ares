#!/usr/bin/env python3
"""
Test script to verify that centralized indicators work correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_centralized_indicators():
    """Test all centralized indicator calculators."""
    print("🧪 Testing centralized indicator calculators...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    data = pd.DataFrame({
        'close': close_prices,
        'high': close_prices + np.random.rand(100) * 2,
        'low': close_prices - np.random.rand(100) * 2,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    try:
        # Test RSI Calculator
        print("📊 Testing RSI Calculator...")
        from src.feature_generation.indicators import RSICalculator
        rsi = RSICalculator.calculate(data['close'], 14)
        print(f"✅ RSI calculation successful. Last value: {rsi.iloc[-1]:.2f}")
        
        # Test MACD Calculator
        print("📊 Testing MACD Calculator...")
        from src.feature_generation.indicators import MACDCalculator
        macd_line, signal_line, histogram = MACDCalculator.calculate(data['close'], 12, 26, 9)
        print(f"✅ MACD calculation successful. Last MACD: {macd_line.iloc[-1]:.2f}")
        
        # Test SMA Calculator
        print("📊 Testing SMA Calculator...")
        from src.feature_generation.indicators import SMACalculator
        sma = SMACalculator.calculate(data['close'], 20)
        print(f"✅ SMA calculation successful. Last value: {sma.iloc[-1]:.2f}")
        
        # Test EMA Calculator
        print("📊 Testing EMA Calculator...")
        from src.feature_generation.indicators import EMACalculator
        ema = EMACalculator.calculate(data['close'], 20)
        print(f"✅ EMA calculation successful. Last value: {ema.iloc[-1]:.2f}")
        
        # Test Stochastic Calculator
        print("📊 Testing Stochastic Calculator...")
        from src.feature_generation.indicators import StochasticCalculator
        k_percent, d_percent = StochasticCalculator.calculate(data['high'], data['low'], data['close'], 14, 3)
        print(f"✅ Stochastic calculation successful. Last %K: {k_percent.iloc[-1]:.2f}")
        
        # Test Bollinger Bands Calculator
        print("📊 Testing Bollinger Bands Calculator...")
        from src.feature_generation.indicators import BollingerBandsCalculator
        upper, middle, lower = BollingerBandsCalculator.calculate(data['close'], 20, 2.0)
        print(f"✅ Bollinger Bands calculation successful. Last upper: {upper.iloc[-1]:.2f}")
        
        print("\n🎉 All centralized indicator tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_updated_files():
    """Test that updated files work with centralized indicators."""
    print("\n🔍 Testing updated files...")
    
    try:
        # Test trading utils helpers
        print("📊 Testing trading utils helpers...")
        from src.trading.utils.helpers import compute_rsi
        
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 101, 100, 99, 98, 99, 100, 101]
        })
        
        rsi_value = compute_rsi(data, 3)
        print(f"✅ Trading utils RSI calculation successful. Value: {rsi_value:.2f}")
        
        print("\n🎉 All updated file tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Updated file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting centralized indicators test...")
    
    success1 = test_centralized_indicators()
    success2 = test_updated_files()
    
    if success1 and success2:
        print("\n✅ All tests passed! Centralization successful.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)