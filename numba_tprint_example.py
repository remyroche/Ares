#!/usr/bin/env python3
"""
Example demonstrating proper Numba compatibility with tprint.

This example shows:
1. How to use tprint in regular Python functions
2. How to use numba_print_* functions in Numba-compiled functions
3. The difference between the two approaches
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.tprint import (
    tprint, tprint_info, tprint_error,
    numba_print_info, numba_print_error, numba_print_performance,
    numba_timer_start, numba_timer_elapsed, NUMBA_AVAILABLE
)

def regular_python_function():
    """Example of using tprint in regular Python functions."""
    print("\n🔍 Regular Python function with tprint:")

    tprint("Starting regular Python processing")
    tprint_info("This is an info message with full tprint features")
    tprint_info("Including colors, timestamps, and structured logging")

    # Simulate some work
    import time
    time.sleep(0.1)

    tprint("Completed regular Python processing")
    return "Regular Python Result"

def example_numba_function():
    """Example of how Numba functions would use numba_print_* functions.

    Note: This is a regular Python function that simulates what a
    Numba-compiled function would do. In real Numba code, you would
    use @njit decorator and call numba_print_* functions from within
    objmode() contexts.
    """
    print("\n🔍 Simulated Numba function pattern:")

    # This simulates what would happen in a real Numba function
    if NUMBA_AVAILABLE:
        print("📝 In real Numba code, you would use:")
        print("   from numba import njit, objmode")
        print("   @njit")
        print("   def my_numba_function(data):")
        print("       with objmode():")
        print("           numba_print_info('Processing data in Numba')")
        print("           # ... processing ...")
        print("           numba_print_performance('Data processing', elapsed)")
    else:
        print("📝 Numba not available, but the pattern would be:")
        print("   Use numba_print_info() instead of tprint_info()")
        print("   Use numba_print_error() instead of tprint_error()")
        print("   These functions work in Numba's objmode contexts")

    # Demonstrate the Numba-compatible functions (which work in regular Python too)
    numba_print_info("This numba_print_info works in regular Python too")
    numba_print_error("This numba_print_error works in regular Python too")

    return "Numba Pattern Result"

def main():
    """Main demonstration function."""
    print("🚀 Numba Compatibility Demonstration")
    print("=" * 50)

    print("\n✅ Both approaches integrate with the same tprint system:")
    print("   - Regular Python: Use tprint_* functions")
    print("   - Numba-compiled: Use numba_print_* functions")
    print("   - Both output to the same console and logging system")

    # Test regular Python function
    result1 = regular_python_function()

    # Test Numba pattern
    result2 = example_numba_function()

    print("\n🎉 Demonstration completed!")
    print(f"Results: {result1} | {result2}")

    print("\n💡 Key Points:")
    print("   • tprint_* functions: Full features, Python only")
    print("   • numba_print_* functions: Numba compatible, limited features")
    print("   • Both integrate seamlessly with your existing logging setup")
    print("   • Choose based on where you're calling from")

if __name__ == "__main__":
    main()