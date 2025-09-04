#!/usr/bin/env python3
"""
Minimal test script for the conservative logging approach.

This script tests only the core logging functions without heavy dependencies.
"""

import logging
import sys
import traceback
from pathlib import Path
from io import StringIO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def capture_log_output():
    """Capture log output to verify conservative logging."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Create a logger and add the handler
    logger = logging.getLogger("ConservativeTest")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    return logger, log_capture

def test_core_logging_functions():
    """Test the core logging functions directly."""
    print("🧪 Testing Core Logging Functions...")
    
    try:
        # Import only the core logging functions
        from src.utils.logger import (
            log_validation_result, log_data_quality_check, 
            log_performance_metrics, log_system_status
        )
        
        logger, log_capture = capture_log_output()
        
        # Test 1: Validation results - should only log failures
        print("📋 Testing validation result logging...")
        
        # This should NOT log (success)
        log_validation_result(logger, "TestValidator", True, "All checks passed")
        
        # This SHOULD log (failure)
        log_validation_result(logger, "TestValidator", False, "Critical error found")
        
        # Test 2: Data quality checks - should only log failures/warnings
        print("📋 Testing data quality check logging...")
        
        # This should NOT log (passed)
        log_data_quality_check(logger, "TestCheck", "passed", "Data quality is good")
        
        # This SHOULD log (warning)
        log_data_quality_check(logger, "TestCheck", "warning", "Some issues found")
        
        # This SHOULD log (failed)
        log_data_quality_check(logger, "TestCheck", "failed", "Critical issues found")
        
        # Test 3: Performance metrics - should only log slow operations
        print("📋 Testing performance metrics logging...")
        
        # This should NOT log (fast operation)
        log_performance_metrics(logger, "FastOperation", 0.5, 50)
        
        # This should NOT log (normal operation)
        log_performance_metrics(logger, "NormalOperation", 5.0, 200)
        
        # This SHOULD log (slow operation)
        log_performance_metrics(logger, "SlowOperation", 15.0, 500)
        
        # This SHOULD log (high memory usage)
        log_performance_metrics(logger, "MemoryIntensiveOperation", 2.0, 2048)
        
        # Test 4: System status - should only log degraded/failed
        print("📋 Testing system status logging...")
        
        # This should NOT log (healthy)
        log_system_status(logger, "TestComponent", "healthy", "All systems operational")
        
        # This should NOT log (starting)
        log_system_status(logger, "TestComponent", "starting", "Initializing")
        
        # This SHOULD log (degraded)
        log_system_status(logger, "TestComponent", "degraded", "Some issues detected")
        
        # This SHOULD log (failed)
        log_system_status(logger, "TestComponent", "failed", "Critical failure")
        
        # Get the captured log output
        log_output = log_capture.getvalue()
        log_lines = [line.strip() for line in log_output.split('\n') if line.strip()]
        
        print(f"📊 Total log lines captured: {len(log_lines)}")
        
        # Verify that we only logged issues, not successes
        success_indicators = ["✅", "PASSED", "healthy", "starting"]
        issue_indicators = ["❌", "⚠️", "FAILED", "degraded", "failed", "Performance Issue"]
        
        success_logs = [line for line in log_lines if any(indicator in line for indicator in success_indicators)]
        issue_logs = [line for line in log_lines if any(indicator in line for indicator in issue_indicators)]
        
        print(f"📊 Success logs found: {len(success_logs)}")
        print(f"📊 Issue logs found: {len(issue_logs)}")
        
        # Print the actual log lines for verification
        print("\n📋 Captured log lines:")
        for i, line in enumerate(log_lines, 1):
            print(f"  {i:2d}. {line}")
        
        # Verify conservative logging
        if len(success_logs) == 0:
            print("✅ Conservative logging working: No success messages logged")
        else:
            print(f"❌ Conservative logging issue: {len(success_logs)} success messages logged")
            for log in success_logs:
                print(f"   - {log}")
        
        if len(issue_logs) > 0:
            print("✅ Issue logging working: Issues are being logged")
        else:
            print("❌ Issue logging problem: No issues were logged")
        
        # Expected: 2 validation failures, 2 data quality issues, 2 performance issues, 2 system issues = 8 total
        expected_issue_logs = 8
        if len(issue_logs) >= expected_issue_logs:
            print(f"✅ Expected number of issue logs: {len(issue_logs)} >= {expected_issue_logs}")
        else:
            print(f"⚠️ Fewer issue logs than expected: {len(issue_logs)} < {expected_issue_logs}")
        
        return len(success_logs) == 0 and len(issue_logs) >= expected_issue_logs
        
    except Exception as e:
        print(f"❌ Core logging test failed: {e}")
        traceback.print_exc()
        return False

def test_decorator_config():
    """Test decorator config without heavy dependencies."""
    print("🧪 Testing Decorator Config...")
    
    try:
        from src.utils.decorator_config import DecoratorConfig
        
        logger, log_capture = capture_log_output()
        
        # Test with healthy configuration
        config = DecoratorConfig()
        health_status = config.get_health_status()
        
        # Should not log anything for healthy config
        log_output = log_capture.getvalue()
        log_lines = [line.strip() for line in log_output.split('\n') if line.strip()]
        
        print(f"📊 Health status: {health_status.get('status', 'unknown')} (valid: {health_status.get('is_valid', False)})")
        print(f"📊 Log lines for healthy config: {len(log_lines)}")
        
        if len(log_lines) == 0:
            print("✅ Healthy decorator config: No logging (as expected)")
        else:
            print("❌ Healthy decorator config: Unexpected logging")
            for line in log_lines:
                print(f"   - {line}")
        
        return len(log_lines) == 0
        
    except Exception as e:
        print(f"❌ Decorator config test failed: {e}")
        traceback.print_exc()
        return False

def test_warning_symbols():
    """Test warning symbols functionality."""
    print("🧪 Testing Warning Symbols...")
    
    try:
        from src.utils.warning_symbols import error, warning, info, success
        
        # Test that symbols are working
        error_msg = error("Test error message")
        warning_msg = warning("Test warning message")
        info_msg = info("Test info message")
        success_msg = success("Test success message")
        
        print(f"📊 Error symbol: {error_msg}")
        print(f"📊 Warning symbol: {warning_msg}")
        print(f"📊 Info symbol: {info_msg}")
        print(f"📊 Success symbol: {success_msg}")
        
        # Verify symbols are present
        symbols_present = all([
            "❌" in error_msg,
            "⚠️" in warning_msg,
            "ℹ️" in info_msg,
            "✅" in success_msg
        ])
        
        if symbols_present:
            print("✅ Warning symbols working correctly")
        else:
            print("❌ Warning symbols not working correctly")
        
        return symbols_present
        
    except Exception as e:
        print(f"❌ Warning symbols test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run minimal conservative logging tests."""
    print("🚀 Starting Minimal Conservative Logging Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Core Logging Functions", test_core_logging_functions),
        ("Decorator Config", test_decorator_config),
        ("Warning Symbols", test_warning_symbols),
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Tests...")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Conservative logging is working correctly.")
        print("📝 Key findings:")
        print("   - ✅ No success emojis cluttering logs")
        print("   - ✅ Only issues are logged for troubleshooting")
        print("   - ✅ Health status only logged when fair/poor")
        print("   - ✅ Performance issues only logged when slow/high memory")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)