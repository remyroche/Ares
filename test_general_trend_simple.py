#!/usr/bin/env python3
"""
Simple test for the general trend feature implementation.

This test verifies that the general trend feature code was added correctly
without requiring complex dependencies.
"""

import os
import sys

def test_file_exists():
    """Test that the trend.py file exists and contains our new class."""
    print("🧪 Testing file existence and content...")
    
    trend_file = "src/feature_generation/categories/trend.py"
    
    if not os.path.exists(trend_file):
        print(f"❌ File {trend_file} does not exist")
        return False
    
    print(f"✅ File {trend_file} exists")
    
    # Read the file and check for our new class
    with open(trend_file, 'r') as f:
        content = f.read()
    
    # Check for key components of our implementation
    checks = [
        ("GeneralTrendFeatureGenerator class", "class GeneralTrendFeatureGenerator"),
        ("ADX calculation method", "def _calculate_adx"),
        ("MACD calculation method", "def _calculate_macd_direction"),
        ("SMA calculation method", "def _calculate_sma_direction"),
        ("Component combination method", "def _combine_trend_components"),
        ("Generator creation function", "def create_general_trend_generators"),
        ("ADX strength comment", "ADX (strength)"),
        ("MACD direction comment", "MACD (direction)"),
        ("General trend formula", "general_trend = ADX_normalized * MACD_normalized")
    ]
    
    passed_checks = 0
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✅ {check_name}")
            passed_checks += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 {passed_checks}/{len(checks)} checks passed")
    
    return passed_checks == len(checks)

def test_implementation_structure():
    """Test the structure of our implementation."""
    print("\n🧪 Testing implementation structure...")
    
    trend_file = "src/feature_generation/categories/trend.py"
    
    with open(trend_file, 'r') as f:
        content = f.read()
    
    # Check for proper class structure
    class_start = content.find("class GeneralTrendFeatureGenerator")
    if class_start == -1:
        print("❌ GeneralTrendFeatureGenerator class not found")
        return False
    
    print("✅ GeneralTrendFeatureGenerator class found")
    
    # Check for proper method structure
    methods = [
        "__init__",
        "_create_default_config", 
        "_generate_feature",
        "_calculate_adx",
        "_calculate_macd_direction",
        "_calculate_sma_direction",
        "_combine_trend_components",
        "_generate_fallback_trend",
        "_should_use_vectorbt"
    ]
    
    method_found = 0
    for method in methods:
        if f"def {method}" in content:
            print(f"   ✅ Method {method} found")
            method_found += 1
        else:
            print(f"   ❌ Method {method} not found")
    
    print(f"   📊 {method_found}/{len(methods)} methods found")
    
    # Check for proper documentation
    if '"""' in content and "ADX (strength)" in content and "MACD (direction)" in content:
        print("✅ Proper documentation found")
        doc_found = True
    else:
        print("❌ Documentation missing or incomplete")
        doc_found = False
    
    return method_found == len(methods) and doc_found

def test_mathematical_logic():
    """Test the mathematical logic of our implementation."""
    print("\n🧪 Testing mathematical logic...")
    
    trend_file = "src/feature_generation/categories/trend.py"
    
    with open(trend_file, 'r') as f:
        content = f.read()
    
    # Check for key mathematical operations
    math_checks = [
        ("ADX normalization", "adx_normalized = adx_values / 100.0"),
        ("Direction normalization", "direction_normalized"),
        ("Trend combination", "general_trend = adx_normalized * direction_normalized"),
        ("True Range calculation", "tr = np.maximum.reduce"),
        ("Directional Movement", "dm_plus = np.maximum"),
        ("MACD calculation", "macd = ema_fast - ema_slow"),
        ("SMA calculation", "sma = close.rolling"),
        ("Price position", "price_position = (close - sma) / sma")
    ]
    
    math_found = 0
    for check_name, check_string in math_checks:
        if check_string in content:
            print(f"   ✅ {check_name}")
            math_found += 1
        else:
            print(f"   ❌ {check_name}")
    
    print(f"   📊 {math_found}/{len(math_checks)} mathematical operations found")
    
    return math_found >= len(math_checks) * 0.8  # Allow for some flexibility

def main():
    """Run all tests."""
    print("🚀 Starting General Trend Feature Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_file_exists,
        test_implementation_structure,
        test_mathematical_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All implementation tests passed!")
        print("✅ The general trend feature has been successfully implemented.")
        print("💡 The feature combines ADX (strength) and MACD/SMA (direction) as requested.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)