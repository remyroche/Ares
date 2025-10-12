#!/usr/bin/env python3
"""
Test script to verify that centralized indicators can be imported correctly.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all centralized indicator modules can be imported."""
    print("🧪 Testing centralized indicator imports...")
    
    try:
        # Test individual indicator imports
        print("📊 Testing individual indicator imports...")
        from src.feature_generation.indicators.rsi import RSICalculator
        print("✅ RSI Calculator imported successfully")
        
        from src.feature_generation.indicators.macd import MACDCalculator
        print("✅ MACD Calculator imported successfully")
        
        from src.feature_generation.indicators.sma import SMACalculator
        print("✅ SMA Calculator imported successfully")
        
        from src.feature_generation.indicators.ema import EMACalculator
        print("✅ EMA Calculator imported successfully")
        
        from src.feature_generation.indicators.stochastic import StochasticCalculator
        print("✅ Stochastic Calculator imported successfully")
        
        from src.feature_generation.indicators.bollinger_bands import BollingerBandsCalculator
        print("✅ Bollinger Bands Calculator imported successfully")
        
        # Test main indicators module import
        print("📊 Testing main indicators module import...")
        from src.feature_generation.indicators import (
            RSICalculator as RSI,
            MACDCalculator as MACD,
            SMACalculator as SMA,
            EMACalculator as EMA,
            StochasticCalculator as STOCH,
            BollingerBandsCalculator as BB
        )
        print("✅ Main indicators module imported successfully")
        
        print("\n🎉 All import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting import test...")
    
    success = test_imports()
    
    if success:
        print("\n✅ All imports successful! Centralization working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Import test failed. Check the output above.")
        sys.exit(1)