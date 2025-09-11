#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test script for numba-friendly timestamps

This script tests the numba-friendly timestamp functionality
to ensure it works correctly in both numba and non-numba environments.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic timestamp functionality."""
    tprint("Testing basic numba-friendly timestamp functionality...")
    
    try:
        from src.utils.numba_timestamps import (
            numba_print_with_timestamp,
            numba_print_info,
            numba_print_warning,
            numba_print_error,
            get_numba_timestamp,
            NUMBA_AVAILABLE
        )
        
        tprint(f"Numba available: {NUMBA_AVAILABLE}")
        
        # Test basic printing
        numba_print_with_timestamp("This is a test message")
        numba_print_info("This is an info message")
        numba_print_warning("This is a warning message")
        numba_print_error("This is an error message")
        
        # Test timestamp generation
        timestamp = get_numba_timestamp()
        tprint(f"Generated timestamp: {timestamp}")
        
        tprint("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        tprint(f"❌ Basic functionality test failed: {e}")
        return False


def test_numba_compilation():
    """Test numba compilation with timestamps."""
    tprint("\nTesting numba compilation with timestamps...")
    
    try:
        from src.utils.numba_timestamps import (
            numba_print_info,
            numba_print_progress,
            numba_timer_start,
            numba_timer_elapsed,
            numba_print_timing
        )
        
        # Test if we can create a numba-compiled function
        try:
            import numba
            from numba import njit
            
            @njit
            def test_numba_function(data):
                """Test function with numba timestamps."""
                numba_print_info("Starting numba function")
                
                start_time = numba_timer_start()
                
                result = 0.0
                for i, item in enumerate(data):
                    if i % 100 == 0:
                        numba_print_progress(i, len(data), f"Processing item {i}")
                    result += item * 2.0
                
                numba_print_timing("Numba processing", start_time)
                numba_print_info("Numba function completed")
                
                return result
            
            # Test the function
            test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = test_numba_function(test_data)
            tprint(f"Numba function result: {result}")
            tprint("✅ Numba compilation test passed!")
            return True
            
        except ImportError:
            tprint("⚠️ Numba not available, skipping compilation test")
            return True
            
    except Exception as e:
        tprint(f"❌ Numba compilation test failed: {e}")
        return False


def test_logger_integration():
    """Test integration with the logger module."""
    tprint("\nTesting logger integration...")
    
    try:
        from src.utils.logger import (
            numba_print_with_timestamp,
            numba_print_info,
            get_numba_timestamp,
            NUMBA_TIMESTAMPS_AVAILABLE
        )
        
        tprint(f"Numba timestamps available in logger: {NUMBA_TIMESTAMPS_AVAILABLE}")
        
        # Test logger integration
        numba_print_with_timestamp("Logger integration test")
        numba_print_info("Logger integration info")
        
        timestamp = get_numba_timestamp()
        tprint(f"Logger timestamp: {timestamp}")
        
        tprint("✅ Logger integration test passed!")
        return True
        
    except Exception as e:
        tprint(f"❌ Logger integration test failed: {e}")
        return False


def test_performance():
    """Test performance of timestamp functions."""
    tprint("\nTesting performance...")
    
    try:
        from src.utils.numba_timestamps import (
            numba_print_with_timestamp,
            get_numba_timestamp,
            numba_timer_start,
            numba_timer_elapsed
        )
        
        # Test timestamp generation performance
        start_time = numba_timer_start()
        
        for i in range(1000):
            timestamp = get_numba_timestamp()
        
        elapsed = numba_timer_elapsed(start_time)
        tprint(f"Generated 1000 timestamps in {elapsed:.4f} seconds")
        tprint(f"Average time per timestamp: {elapsed/1000*1000:.4f} ms")
        
        # Test printing performance
        start_time = numba_timer_start()
        
        for i in range(100):
            numba_print_with_timestamp(f"Performance test message {i}")
        
        elapsed = numba_timer_elapsed(start_time)
        tprint(f"Printed 100 messages in {elapsed:.4f} seconds")
        tprint(f"Average time per print: {elapsed/100*1000:.4f} ms")
        
        tprint("✅ Performance test passed!")
        return True
        
    except Exception as e:
        tprint(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    tprint("Numba-Friendly Timestamps Test Suite")
    tprint("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_numba_compilation,
        test_logger_integration,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    tprint(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        tprint("🎉 All tests passed!")
        return True
    else:
        tprint("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
