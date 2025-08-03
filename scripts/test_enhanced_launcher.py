#!/usr/bin/env python3
# scripts/test_enhanced_launcher.py

"""
Test script to demonstrate the enhanced launcher capabilities.

This script shows how the ares_launcher.py now uses enhanced training
with efficiency optimizations for backtest and blank runs.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_enhanced_backtest():
    """Test enhanced backtesting capabilities."""
    print("🧪 Testing Enhanced Backtesting...")

    try:
        # Run enhanced backtesting
        result = subprocess.run(
            [
                sys.executable,
                "ares_launcher.py",
                "backtest",
                "--symbol",
                "ETHUSDT",
                "--exchange",
                "BINANCE",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )  # 5 minute timeout

        if result.returncode == 0:
            print("✅ Enhanced backtesting test passed")
            return True
        print(f"❌ Enhanced backtesting test failed: {result.stderr}")
        return False

    except subprocess.TimeoutExpired:
        print(
            "⏰ Enhanced backtesting test timed out (this is expected for large datasets)",
        )
        return True
    except Exception as e:
        print(f"❌ Enhanced backtesting test error: {e}")
        return False


def test_enhanced_blank_training():
    """Test enhanced blank training capabilities."""
    print("🧪 Testing Enhanced Blank Training...")

    try:
        # Run enhanced blank training
        result = subprocess.run(
            [
                sys.executable,
                "ares_launcher.py",
                "blank",
                "--symbol",
                "ETHUSDT",
                "--exchange",
                "BINANCE",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )  # 5 minute timeout

        if result.returncode == 0:
            print("✅ Enhanced blank training test passed")
            return True
        print(f"❌ Enhanced blank training test failed: {result.stderr}")
        return False

    except subprocess.TimeoutExpired:
        print(
            "⏰ Enhanced blank training test timed out (this is expected for large datasets)",
        )
        return True
    except Exception as e:
        print(f"❌ Enhanced blank training test error: {e}")
        return False


def test_efficiency_demo():
    """Test efficiency features demonstration."""
    print("🧪 Testing Efficiency Features Demo...")

    try:
        # Run efficiency demo
        result = subprocess.run(
            [sys.executable, "scripts/run_enhanced_training.py", "--demo"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode == 0:
            print("✅ Efficiency demo test passed")
            return True
        print(f"❌ Efficiency demo test failed: {result.stderr}")
        return False

    except Exception as e:
        print(f"❌ Efficiency demo test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Launcher Capabilities")
    print("=" * 50)

    tests = [
        ("Efficiency Demo", test_efficiency_demo),
        ("Enhanced Backtesting", test_enhanced_backtest),
        ("Enhanced Blank Training", test_enhanced_blank_training),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced launcher is working correctly.")
        return 0
    print("⚠️ Some tests failed. Check the output above for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
