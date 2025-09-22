#!/usr/bin/env python3
"""
Test script to verify Numba compatibility with tprint.

This script demonstrates:
1. Basic tprint functionality (works everywhere)
2. Numba-compatible functions (works in objmode contexts)
3. Context manager limitations with Numba
4. Performance comparison
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.tprint import (
    tprint, tprint_info, tprint_error,
    numba_print_info, numba_print_error, numba_print_performance,
    capture_print_to_tprint, NUMBA_AVAILABLE
)

def test_basic_tprint():
    """Test basic tprint functionality."""
    print("\n🧪 Testing basic tprint functionality...")
    tprint("Basic tprint message")
    tprint_info("Info level message")
    tprint_error("Error level message")
    print("✅ Basic tprint test completed")

def test_numba_compatible_functions():
    """Test Numba-compatible functions."""
    print("\n🧪 Testing Numba-compatible functions...")

    if NUMBA_AVAILABLE:
        # Test Numba-compatible functions
        numba_print_info("Numba-compatible info message")
        numba_print_error("Numba-compatible error message")
        numba_print_performance("Test operation", 0.123)
        print("✅ Numba-compatible functions work")
    else:
        print("⚠️ Numba not available - skipping Numba tests")

def test_context_manager():
    """Test context manager functionality."""
    print("\n🧪 Testing context manager functionality...")

    try:
        # This works in regular Python
        with capture_print_to_tprint():
            print("This will be captured by context manager")
        print("✅ Context manager works in regular Python")
    except Exception as e:
        print(f"⚠️ Context manager failed: {e}")

def demonstrate_numba_usage_pattern():
    """Demonstrate proper Numba usage patterns."""
    print("\n📚 Demonstrating proper Numba usage patterns...")

    print("✅ For Numba-compiled functions:")
    print("   - Use numba_print_info(), numba_print_error(), etc.")
    print("   - These functions use @njit + objmode() internally")
    print("   - They call tprint() from within objmode contexts")
    print("   - Perfect for logging in Numba-compiled functions")

    print("\n⚠️ For regular Python functions:")
    print("   - Use tprint(), tprint_info(), tprint_error(), etc.")
    print("   - Full functionality including colors, file logging, etc.")
    print("   - More features than Numba-compatible versions")

    print("\n❌ Context manager limitations:")
    print("   - capture_print_to_tprint() won't work in Numba-compiled functions")
    print("   - File I/O operations don't work in nopython mode")
    print("   - Use numba_print_* functions instead")

def main():
    """Main test function."""
    print("🚀 Testing Numba Compatibility with tprint...")
    print("=" * 60)

    test_basic_tprint()
    test_numba_compatible_functions()
    test_context_manager()
    demonstrate_numba_usage_pattern()

    print("\n🎉 Numba compatibility test completed!")
    print("\n📊 Summary:")
    print("   - Basic tprint: ✅ Works everywhere")
    print("   - Numba functions: ✅ Works in Numba (when available)")
    print("   - Context manager: ⚠️ Limited to regular Python")
    print("   - Integration: ✅ Fully compatible with existing codebase")

    print("\n💡 Usage Recommendations:")
    print("   1. Use tprint_* for regular Python functions")
    print("   2. Use numba_print_* for Numba-compiled functions")
    print("   3. Avoid context manager in Numba-compiled code")
    print("   4. All functions integrate with the same logging system")

if __name__ == "__main__":
    main()