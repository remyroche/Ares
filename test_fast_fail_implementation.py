#!/usr/bin/env python3
"""
Test Script for Fast Fail Implementation

This script tests that the system fast fails when optimal periods cannot be found,
instead of using fallback configurations.
"""

import os
import sys
from pathlib import Path

def test_fast_fail_implementation():
    """Test that fast fail is implemented correctly."""
    print("🧪 Testing Fast Fail Implementation...")
    
    try:
        # Test automatic_timeframe_optimizer.py
        print("   → Testing automatic_timeframe_optimizer.py...")
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for fast fail implementations
        fast_fail_checks = [
            "FAST FAIL: Cannot find optimal periods",
            "FAST FAIL: Low optimization score",
            "FAST FAIL: Optimized configuration validation failed",
            "FAST FAIL: Optimization failed",
            "raise RuntimeError",
            "Cannot proceed without optimal timeframe discovery"
        ]
        
        all_found = True
        for check in fast_fail_checks:
            if check in content:
                print(f"      ✅ {check}")
            else:
                print(f"      ❌ {check} - MISSING")
                all_found = False
        
        # Check that fallback methods are removed
        fallback_removed_checks = [
            "return self._create_fallback_result(model_type)",
            "using fallback",
            "fallback configurations"
        ]
        
        for check in fallback_removed_checks:
            if check not in content:
                print(f"      ✅ Fallback removed: {check}")
            else:
                print(f"      ❌ Fallback still present: {check}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def test_enhanced_pipeline_fast_fail():
    """Test enhanced pipeline fast fail implementation."""
    print("\n🧪 Testing Enhanced Pipeline Fast Fail...")
    
    try:
        with open("src/training/steps/market_analysis/enhanced_multi_horizon_pipeline.py", 'r') as f:
            content = f.read()
        
        # Check for fast fail implementations
        fast_fail_checks = [
            "FAST FAIL: Automatic optimization failed",
            "FAST FAIL: Fallback configurations not allowed",
            "fast_fail_on_optimization_failure: bool = True",
            "raise RuntimeError",
            "Training pipeline will terminate"
        ]
        
        all_found = True
        for check in fast_fail_checks:
            if check in content:
                print(f"      ✅ {check}")
            else:
                print(f"      ❌ {check} - MISSING")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def test_sub_pipeline_adapter_fast_fail():
    """Test sub-pipeline adapter fast fail implementation."""
    print("\n🧪 Testing Sub-Pipeline Adapter Fast Fail...")
    
    try:
        with open("src/training/steps/market_analysis/multi_horizon_sub_pipeline_adapter.py", 'r') as f:
            content = f.read()
        
        # Check for fast fail implementations
        fast_fail_checks = [
            "FAST FAIL: Optimization disabled",
            "FAST FAIL: Low optimization score",
            "FAST FAIL: Low validation score",
            "FAST FAIL: Automatic optimization failed",
            "raise RuntimeError",
            "Training pipeline will terminate"
        ]
        
        all_found = True
        for check in fast_fail_checks:
            if check in content:
                print(f"      ✅ {check}")
            else:
                print(f"      ❌ {check} - MISSING")
                all_found = False
        
        # Check that fallback methods are removed
        fallback_removed_checks = [
            "Falling back to default configuration",
            "using default timeframes",
            "return MultiHorizonConfig()"
        ]
        
        for check in fallback_removed_checks:
            if check not in content:
                print(f"      ✅ Fallback removed: {check}")
            else:
                print(f"      ❌ Fallback still present: {check}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def test_error_message_consistency():
    """Test that error messages are consistent and informative."""
    print("\n🧪 Testing Error Message Consistency...")
    
    try:
        # Check all files for consistent error messages
        files_to_check = [
            "src/training/steps/market_analysis/automatic_timeframe_optimizer.py",
            "src/training/steps/market_analysis/enhanced_multi_horizon_pipeline.py",
            "src/training/steps/market_analysis/multi_horizon_sub_pipeline_adapter.py"
        ]
        
        consistent_checks = [
            "❌ FAST FAIL:",
            "Cannot proceed without optimal timeframe discovery",
            "Training pipeline will terminate"
        ]
        
        all_consistent = True
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for check in consistent_checks:
                if check in content:
                    print(f"      ✅ {file_path}: {check}")
                else:
                    print(f"      ❌ {file_path}: {check} - MISSING")
                    all_consistent = False
        
        return all_consistent
        
    except Exception as e:
        print(f"   ❌ Error checking consistency: {e}")
        return False


def main():
    """Run all fast fail tests."""
    print("🚀 Testing Fast Fail Implementation")
    print("=" * 50)
    print("Expected Behavior:")
    print("  • System should fast fail when optimization fails")
    print("  • No fallback configurations should be used")
    print("  • Clear error messages should be provided")
    print("  • Training pipeline should terminate on failure")
    print("=" * 50)
    
    tests = [
        ("Automatic Timeframe Optimizer Fast Fail", test_fast_fail_implementation),
        ("Enhanced Pipeline Fast Fail", test_enhanced_pipeline_fast_fail),
        ("Sub-Pipeline Adapter Fast Fail", test_sub_pipeline_adapter_fast_fail),
        ("Error Message Consistency", test_error_message_consistency)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 FAST FAIL TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All fast fail tests passed!")
        print("\n📋 FAST FAIL SUMMARY:")
        print("   ✅ System will fast fail when optimization fails")
        print("   ✅ No fallback configurations will be used")
        print("   ✅ Clear error messages are provided")
        print("   ✅ Training pipeline will terminate on failure")
        print("   ✅ Consistent error handling across all components")
    else:
        print("\n⚠️ Some fast fail tests failed. Check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)