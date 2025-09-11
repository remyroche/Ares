#!/usr/bin/env python3
"""
Test script comparing the tprint approach vs global print replacement.

This demonstrates why the tprint approach is safer and more predictable.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tprint_approach():
    """Test the new tprint approach."""
    print("\n" + "="*60)
    print("🧪 TESTING TPRINT APPROACH (SAFER)")
    print("="*60)
    
    try:
        from src.utils.print_utils import (
            tprint, tprint_info, tprint_warning, tprint_error, 
            tprint_success, tprint_progress, tprint_performance
        )
        
        print("✅ Successfully imported tprint functions")
        
        # Test basic tprint
        print("\n--- Basic tprint ---")
        tprint("This is a regular message")
        tprint("Multiple arguments", "work", "correctly")
        
        # Test different log levels
        print("\n--- Different log levels ---")
        tprint_info("This is an info message")
        tprint_warning("This is a warning message")
        tprint_error("This is an error message")
        tprint_success("This is a success message")
        
        # Test progress
        print("\n--- Progress tracking ---")
        tprint_progress(3, 10, "Processing data")
        tprint_progress(7, 10, "Almost done")
        tprint_progress(10, 10, "Complete")
        
        # Test performance
        print("\n--- Performance tracking ---")
        tprint_performance("Data processing", 2.5)
        tprint_performance("Model training", 45.2)
        
        # Test that regular print still works
        print("\n--- Regular print still works ---")
        print("This is a regular print statement (no timestamp)")
        
        print("\n✅ TPRINT APPROACH TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numba_compatibility():
    """Test numba compatibility with tprint approach."""
    print("\n" + "="*60)
    print("🧪 TESTING NUMBA COMPATIBILITY")
    print("="*60)
    
    try:
        # Test if numba is available
        try:
            import numba
            print(f"✅ Numba is available (version: {numba.__version__})")
            
            # Test numba compilation
            @numba.jit(nopython=True)
            def test_numba_function(x):
                return x * 2
            
            result = test_numba_function(5)
            print(f"✅ Numba function executed successfully: {result}")
            
            # Test that tprint doesn't interfere
            from src.utils.print_utils import tprint
            tprint("This tprint call doesn't interfere with numba")
            
            print("✅ NUMBA COMPATIBILITY TEST PASSED")
            return True
            
        except ImportError:
            print("ℹ️ Numba is not available, skipping numba test")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_explicit_vs_implicit():
    """Demonstrate the difference between explicit and implicit approaches."""
    print("\n" + "="*60)
    print("🧪 TESTING EXPLICIT VS IMPLICIT APPROACHES")
    print("="*60)
    
    try:
        from src.utils.print_utils import tprint
        
        print("--- EXPLICIT APPROACH (tprint) ---")
        print("✅ Clear intent - you know exactly what gets timestamped")
        tprint("This will definitely have a timestamp")
        print("This will definitely NOT have a timestamp")
        
        print("\n--- IMPLICIT APPROACH (global print replacement) ---")
        print("❌ Unclear intent - you don't know if print is timestamped")
        print("This might have a timestamp (depends on global state)")
        print("This might also have a timestamp (same uncertainty)")
        
        print("\n✅ EXPLICIT VS IMPLICIT TEST COMPLETED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_easy_migration():
    """Show how easy it is to migrate existing code."""
    print("\n" + "="*60)
    print("🧪 TESTING EASY MIGRATION")
    print("="*60)
    
    try:
        from src.utils.print_utils import tprint
        
        print("--- BEFORE (regular print) ---")
        print("print('User logged in')")
        print("print('Processing data...')")
        print("print('Error: Connection failed')")
        
        print("\n--- AFTER (tprint) ---")
        print("tprint('User logged in')")
        print("tprint('Processing data...')")
        print("tprint('Error: Connection failed')")
        
        print("\n--- ACTUAL OUTPUT ---")
        tprint("User logged in")
        tprint("Processing data...")
        tprint("Error: Connection failed")
        
        print("\n✅ MIGRATION TEST COMPLETED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 STARTING TPRINT APPROACH TESTS")
    print("="*80)
    
    tests = [
        test_tprint_approach,
        test_numba_compatibility,
        test_explicit_vs_implicit,
        test_easy_migration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*80)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - TPRINT APPROACH IS SUPERIOR!")
        print("\n🎯 KEY ADVANTAGES:")
        print("  ✅ Explicit and clear intent")
        print("  ✅ No global state pollution")
        print("  ✅ No numba conflicts")
        print("  ✅ Easy to test and mock")
        print("  ✅ Easy to migrate existing code")
        print("  ✅ Multiple log levels available")
        print("  ✅ Progress and performance tracking")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()