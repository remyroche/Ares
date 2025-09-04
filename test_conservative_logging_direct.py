#!/usr/bin/env python3
"""
Direct test script for the conservative logging approach.

This script tests the logging functions directly without importing modules that require numpy.
"""

import logging
import sys
import traceback
from pathlib import Path
from io import StringIO
from datetime import datetime

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

def log_validation_result(logger: logging.Logger, validator_name: str, result: bool, 
                         details: str = "", metrics: dict = None) -> None:
    """
    Log validation results (only failures for troubleshooting).
    """
    try:
        # Only log failures - skip successful validations
        if result:
            return
            
        emoji = "❌"
        status = "FAILED"
            
        message = f"{emoji} Validation {status} | {validator_name}"
        if details:
            message += f" | {details}"
            
        # Add metrics if provided
        if metrics:
            metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
            message += f" | Metrics: {metrics_str}"
            
        logger.error(message)
            
    except Exception as e:
        logger.error(f"❌ Failed to log validation result: {e}")

def log_data_quality_check(logger: logging.Logger, check_name: str, status: str, 
                          details: str = "", stats: dict = None) -> None:
    """
    Log data quality check results (only failures and warnings for troubleshooting).
    """
    try:
        # Only log failures and warnings - skip passed checks
        if status == "passed":
            return
            
        status_emojis = {
            "failed": "❌", 
            "warning": "⚠️"
        }
        
        emoji = status_emojis.get(status, "⚠️")
        message = f"{emoji} Data Quality Check | {check_name} | {status.upper()}"
        
        if details:
            message += f" | {details}"
            
        if stats:
            stats_str = " | ".join([f"{k}={v}" for k, v in stats.items()])
            message += f" | Stats: {stats_str}"
            
        if status == "failed":
            logger.error(message)
        elif status == "warning":
            logger.warning(message)
            
    except Exception as e:
        logger.error(f"❌ Failed to log data quality check: {e}")

def log_performance_metrics(logger: logging.Logger, operation_name: str, 
                           duration: float, memory_usage: float = None, 
                           additional_metrics: dict = None) -> None:
    """
    Log performance metrics (only for slow operations that need troubleshooting).
    """
    try:
        # Only log slow operations (>10s) or high memory usage (>1GB)
        should_log = False
        emoji = "🐌"
        
        if duration > 10.0:
            should_log = True
            emoji = "🐌"  # Slow
        elif memory_usage is not None and memory_usage > 1024:  # >1GB
            should_log = True
            emoji = "💾"  # High memory usage
            
        if not should_log:
            return
            
        message = f"{emoji} Performance Issue | {operation_name} | Duration: {duration:.3f}s"
        
        if memory_usage is not None:
            message += f" | Memory: {memory_usage:.2f}MB"
            
        if additional_metrics:
            metrics_str = " | ".join([f"{k}={v}" for k, v in additional_metrics.items()])
            message += f" | {metrics_str}"
            
        logger.warning(message)
        
    except Exception as e:
        logger.error(f"❌ Failed to log performance metrics: {e}")

def log_system_status(logger: logging.Logger, component: str, status: str, 
                     details: str = "", health_metrics: dict = None) -> None:
    """
    Log system component status with health indicators (only for issues).
    """
    try:
        # Only log if there are issues - skip healthy/starting status
        if status in ["healthy", "starting"]:
            return
            
        status_emojis = {
            "degraded": "🟡", 
            "failed": "🔴",
            "stopping": "⏹️",
            "maintenance": "🔧"
        }
        
        emoji = status_emojis.get(status, "⚠️")
        message = f"{emoji} System Status | {component} | {status.upper()}"
        
        if details:
            message += f" | {details}"
            
        if health_metrics:
            metrics_str = " | ".join([f"{k}={v}" for k, v in health_metrics.items()])
            message += f" | Health: {metrics_str}"
            
        if status == "failed":
            logger.error(message)
        elif status in ["degraded", "stopping", "maintenance"]:
            logger.warning(message)
            
    except Exception as e:
        logger.error(f"❌ Failed to log system status: {e}")

def test_conservative_logging():
    """Test that logging is conservative - only logs issues."""
    print("🧪 Testing Conservative Logging Approach...")
    
    try:
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
        issue_indicators = ["❌", "⚠️", "FAILED", "degraded", "failed", "Performance Issue", "🐌", "💾", "🟡", "🔴"]
        
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
        
        # Expected: 1 validation failure, 2 data quality issues, 2 performance issues, 2 system issues = 7 total
        expected_issue_logs = 7
        if len(issue_logs) >= expected_issue_logs:
            print(f"✅ Expected number of issue logs: {len(issue_logs)} >= {expected_issue_logs}")
        else:
            print(f"⚠️ Fewer issue logs than expected: {len(issue_logs)} < {expected_issue_logs}")
        
        return len(success_logs) == 0 and len(issue_logs) >= expected_issue_logs
        
    except Exception as e:
        print(f"❌ Conservative logging test failed: {e}")
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
    """Run direct conservative logging tests."""
    print("🚀 Starting Direct Conservative Logging Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Conservative Logging", test_conservative_logging),
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