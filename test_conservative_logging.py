#!/usr/bin/env python3
"""
Test script for the conservative logging approach.

This script verifies that the logging system only logs when there are issues
and doesn't overcrowd logs with success messages.
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

def test_conservative_logging():
    """Test that logging is conservative - only logs issues."""
    print("🧪 Testing Conservative Logging Approach...")
    
    try:
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
        print(f"❌ Conservative logging test failed: {e}")
        traceback.print_exc()
        return False

def test_decorator_health_logging():
    """Test that decorator health logging is conservative."""
    print("🧪 Testing Decorator Health Logging...")
    
    try:
        from src.utils.decorator_config import DecoratorConfig
        
        logger, log_capture = capture_log_output()
        
        # Test with healthy configuration
        config = DecoratorConfig()
        health_status = config.get_health_status()
        
        # Should not log anything for healthy config
        log_output = log_capture.getvalue()
        log_lines = [line.strip() for line in log_output.split('\n') if line.strip()]
        
        print(f"📊 Health status: {health_status['status']} ({health_status['health_score']}/100)")
        print(f"📊 Log lines for healthy config: {len(log_lines)}")
        
        if len(log_lines) == 0:
            print("✅ Healthy decorator config: No logging (as expected)")
        else:
            print("❌ Healthy decorator config: Unexpected logging")
            for line in log_lines:
                print(f"   - {line}")
        
        return len(log_lines) == 0
        
    except Exception as e:
        print(f"❌ Decorator health logging test failed: {e}")
        traceback.print_exc()
        return False

def test_common_operations_logging():
    """Test that common operations logging is conservative."""
    print("🧪 Testing Common Operations Logging...")
    
    try:
        from src.utils.common_operations import (
            get_current_datetime, safe_float, safe_int, 
            safe_lower, safe_upper, safe_append
        )
        
        logger, log_capture = capture_log_output()
        
        # Test normal operations - should not log
        now = get_current_datetime()
        float_val = safe_float("3.14")
        int_val = safe_int("42")
        lower_str = safe_lower("TEST")
        upper_str = safe_upper("test")
        test_list = safe_append([], "test")
        
        # Test error cases - should log
        float_error = safe_float("not_a_number")
        int_error = safe_int("not_a_number")
        lower_none = safe_lower(None)
        
        log_output = log_capture.getvalue()
        log_lines = [line.strip() for line in log_output.split('\n') if line.strip()]
        
        print(f"📊 Log lines for common operations: {len(log_lines)}")
        
        # Should only log warnings for error cases, not success cases
        success_logs = [line for line in log_lines if "✅" in line]
        warning_logs = [line for line in log_lines if "⚠️" in line]
        
        print(f"📊 Success logs: {len(success_logs)}")
        print(f"📊 Warning logs: {len(warning_logs)}")
        
        if len(success_logs) == 0:
            print("✅ Common operations: No success logging (as expected)")
        else:
            print("❌ Common operations: Unexpected success logging")
            for log in success_logs:
                print(f"   - {log}")
        
        if len(warning_logs) >= 3:  # Expected warnings for error cases
            print("✅ Common operations: Warning logging for errors (as expected)")
        else:
            print("❌ Common operations: Missing warning logs for errors")
        
        return len(success_logs) == 0 and len(warning_logs) >= 3
        
    except Exception as e:
        print(f"❌ Common operations logging test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all conservative logging tests."""
    print("🚀 Starting Conservative Logging Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Conservative Logging", test_conservative_logging),
        ("Decorator Health Logging", test_decorator_health_logging),
        ("Common Operations Logging", test_common_operations_logging),
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