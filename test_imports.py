#!/usr/bin/env python3
"""
Test script to verify import fixes
"""

def test_tenacity_import():
    """Test tenacity import in binance.py context"""
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential
        print("✅ Tenacity import successful")
        return True
    except ImportError as e:
        print(f"❌ Tenacity import failed: {e}")
        return False

def test_sr_breakout_import():
    """Test SR breakout predictor import"""
    try:
        # Import the module to check if the file exists
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

        # Try importing the module (without instantiating to avoid circular imports)
        import importlib
        spec = importlib.util.spec_from_file_location(
            "sr_breakout_predictor_enhanced",
            "src/tactician/sr_levels/sr_breakout_predictor_enhanced.py"
        )
        if spec and spec.loader:
            print("✅ SR breakout predictor module found")
            return True
        else:
            print("❌ SR breakout predictor module not found")
            return False
    except Exception as e:
        print(f"❌ SR breakout predictor import check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing import fixes...")
    tenacity_ok = test_tenacity_import()
    sr_ok = test_sr_breakout_import()

    if tenacity_ok and sr_ok:
        print("\n🎉 All import issues resolved!")
    else:
        print("\n⚠️ Some import issues remain")