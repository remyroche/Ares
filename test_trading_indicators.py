#!/usr/bin/env python3
"""
Test Trading Indicators Functionality

This script tests the new trading indicators functionality in the unified matrix operations module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_trading_indicators_import():
    """Test that trading indicators can be imported."""
    try:
        from src.utils.matrix_operations import (
            compute_trading_indicators,
            compute_moving_averages,
            compute_momentum_indicators,
            compute_volatility_indicators,
            compute_volume_indicators,
            compute_trend_indicators,
            compute_oscillator_indicators,
            compute_pattern_indicators
        )
        print("✅ Trading indicators imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import trading indicators: {e}")
        return False

def test_vectorized_core_import():
    """Test that vectorized core can be imported."""
    try:
        from src.utils.matrix_operations import get_vectorized_processing_core
        core = get_vectorized_processing_core()
        print("✅ Vectorized processing core imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import vectorized core: {e}")
        return False

def test_trading_indicators_methods():
    """Test that trading indicator methods exist."""
    try:
        from src.utils.matrix_operations import get_vectorized_processing_core
        core = get_vectorized_processing_core()
        
        # Check if methods exist
        methods_to_check = [
            'compute_trading_indicators',
            '_compute_moving_averages',
            '_compute_momentum_indicators',
            '_compute_volatility_indicators',
            '_compute_volume_indicators',
            '_compute_trend_indicators',
            '_compute_oscillator_indicators',
            '_compute_pattern_indicators',
            '_get_default_indicator_config'
        ]
        
        for method_name in methods_to_check:
            if hasattr(core, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error checking methods: {e}")
        return False

def test_convenience_functions():
    """Test that convenience functions are available."""
    try:
        from src.utils.matrix_operations import (
            compute_trading_indicators,
            compute_moving_averages,
            compute_momentum_indicators,
            compute_volatility_indicators,
            compute_volume_indicators,
            compute_trend_indicators,
            compute_oscillator_indicators,
            compute_pattern_indicators
        )
        
        # Check if functions are callable
        functions_to_check = [
            compute_trading_indicators,
            compute_moving_averages,
            compute_momentum_indicators,
            compute_volatility_indicators,
            compute_volume_indicators,
            compute_trend_indicators,
            compute_oscillator_indicators,
            compute_pattern_indicators
        ]
        
        for func in functions_to_check:
            if callable(func):
                print(f"✅ Function {func.__name__} is callable")
            else:
                print(f"❌ Function {func.__name__} is not callable")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error checking convenience functions: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 TRADING INDICATORS FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Import Trading Indicators", test_trading_indicators_import),
        ("Import Vectorized Core", test_vectorized_core_import),
        ("Check Trading Indicator Methods", test_trading_indicators_methods),
        ("Check Convenience Functions", test_convenience_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All trading indicators tests passed!")
        print("\n✨ Trading indicators are ready for use!")
        print("\n📖 Usage Example:")
        print("""
from src.utils.matrix_operations import compute_trading_indicators
import pandas as pd

# Create sample OHLCV data
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Compute all trading indicators
indicators = compute_trading_indicators(data)
print(f"Computed {len(indicators.columns)} indicators")
        """)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())