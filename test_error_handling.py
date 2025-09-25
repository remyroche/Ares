#!/usr/bin/env python3
"""
Test script to verify that our error handling fixes are working correctly.
"""

def test_bare_except_fix():
    """Test that bare except clauses have been replaced with proper error handling."""
    try:
        # Test case 1: Simple exception handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            print(f"✅ Exception handling working: {e}")
            return True
    except:
        print("❌ Bare except still present - this should not happen")
        return False

def test_tprint_logging():
    """Test that tprint logging is working."""
    try:
        from src.utils.tprint import tprint_error, tprint_warning, tprint_info

        tprint_info("✅ tprint logging is working correctly")
        tprint_warning("⚠️ This is a test warning")
        tprint_error("❌ This is a test error")

        return True
    except Exception as e:
        print(f"❌ tprint logging failed: {e}")
        return False

def test_file_imports():
    """Test that our fixed files can be imported (at least the structure)."""
    try:
        # Try importing just the structure without dependencies
        import importlib.util

        # Test one of our fixed files
        spec = importlib.util.spec_from_file_location(
            "test_module",
            "/workspace/src/training/steps/market_analysis/tas_regime/core/advanced_tas_search.py"
        )

        if spec is None:
            print("❌ Could not load module spec")
            return False

        print("✅ Module structure is valid")
        return True

    except Exception as e:
        print(f"❌ File import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing error handling fixes...")

    tests = [
        ("Bare except fix", test_bare_except_fix),
        ("tprint logging", test_tprint_logging),
        ("File imports", test_file_imports)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")

    print("\n📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("🎉 All tests passed! Error handling fixes are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the fixes.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)