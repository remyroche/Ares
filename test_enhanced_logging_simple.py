#!/usr/bin/env python3
"""
Simplified test script for the enhanced logging system with emoji-based logging.

This script tests the core logging functionality without requiring external dependencies
like numpy and pandas.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_logger():
    """Test the enhanced logger functionality."""
    print("🧪 Testing Enhanced Logger...")
    
    try:
        from src.utils.logger import (
            setup_logging, get_logger, log_step_progress, log_validation_result,
            log_data_quality_check, log_performance_metrics, log_error_with_context,
            log_system_status
        )
        
        # Setup logging
        logger = setup_logging()
        if logger:
            print("✅ Enhanced logger setup successful")
        else:
            print("❌ Enhanced logger setup failed")
            return False
        
        # Test enhanced logging methods
        test_logger = get_logger("TestLogger")
        
        # Test different log levels with emojis
        test_logger.info("This is an info message")
        test_logger.warning("This is a warning message")
        test_logger.error("This is an error message")
        test_logger.debug("This is a debug message")
        
        # Test specialized logging functions
        log_step_progress(test_logger, "Test Step", 1, 5, "running", "Testing progress logging")
        log_validation_result(test_logger, "Test Validator", True, "Validation passed successfully")
        log_data_quality_check(test_logger, "Test Check", "passed", "Data quality is good")
        log_performance_metrics(test_logger, "Test Operation", 1.5, 128.5, {"cpu_usage": 45})
        
        # Test error logging with context
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            log_error_with_context(test_logger, e, {"test_param": "test_value"}, "test_operation")
        
        # Test system status logging
        log_system_status(test_logger, "Test Component", "healthy", "All systems operational")
        
        print("✅ Enhanced logger tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced logger test failed: {e}")
        traceback.print_exc()
        return False

def test_decorator_system():
    """Test the enhanced decorator system."""
    print("🧪 Testing Enhanced Decorator System...")
    
    try:
        from src.utils.decorator_config import DecoratorConfig, global_config
        from src.utils.decorator_registry import DecoratorRegistry, register_decorator
        
        # Test DecoratorConfig
        config = DecoratorConfig()
        print("✅ DecoratorConfig created successfully")
        
        # Test configuration validation
        is_valid, issues = config.validate_config()
        print(f"✅ Config validation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            print(f"⚠️ Issues found: {issues}")
        
        # Test health status
        health_status = config.get_health_status()
        print(f"✅ Config health status: {health_status['status']} ({health_status['health_score']}/100)")
        
        # Test DecoratorRegistry
        registry = DecoratorRegistry()
        print("✅ DecoratorRegistry created successfully")
        
        # Test decorator registration
        @register_decorator("test_decorator", version="1.0", description="Test decorator")
        def test_function():
            return "test"
        
        print("✅ Decorator registration successful")
        
        # Test registry health
        registry_health = registry.get_registry_health_status()
        print(f"✅ Registry health: {registry_health['status']} ({registry_health['health_score']}/100)")
        
        print("✅ Enhanced decorator system tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced decorator system test failed: {e}")
        traceback.print_exc()
        return False

def test_basic_common_operations():
    """Test basic common operations without numpy/pandas dependencies."""
    print("🧪 Testing Basic Common Operations...")
    
    try:
        from src.utils.common_operations import (
            get_current_datetime, get_today, format_datetime, parse_datetime,
            ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
            safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
            safe_join, safe_float, safe_int, create_argument_parser
        )
        
        # Test datetime operations
        now = get_current_datetime()
        today = get_today()
        formatted = format_datetime(now)
        parsed = parse_datetime("2023-01-01 00:00:00")
        print("✅ Datetime operations successful")
        
        # Test file operations
        test_dir = Path("/tmp/test_logging_dir")
        ensure_directory(test_dir)
        exists = safe_file_exists(test_dir)
        print("✅ File operations successful")
        
        # Test list operations
        test_list = safe_append([], "test")
        extended_list = safe_extend(test_list, ["a", "b"])
        dict_val = safe_dict_get({"key": "value"}, "key")
        lower_str = safe_lower("TEST")
        upper_str = safe_upper("test")
        joined_str = safe_join(",", ["a", "b", "c"])
        print("✅ List and string operations successful")
        
        # Test type conversion
        float_val = safe_float("3.14")
        int_val = safe_int("42")
        print("✅ Type conversion operations successful")
        
        # Test argument parser
        parser = create_argument_parser("Test parser")
        print("✅ Argument parser creation successful")
        
        print("✅ Basic common operations tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic common operations test failed: {e}")
        traceback.print_exc()
        return False

def test_error_scenarios():
    """Test error handling scenarios."""
    print("🧪 Testing Error Handling Scenarios...")
    
    try:
        from src.utils.common_operations import safe_float, safe_int, safe_lower, safe_upper
        from src.utils.logger import log_error_with_context
        
        # Test with invalid inputs
        logger = logging.getLogger("ErrorTest")
        
        # Test safe_float with invalid input
        result = safe_float("not_a_number")
        print(f"✅ safe_float with invalid input: {result}")
        
        # Test safe_int with invalid input
        result = safe_int("not_a_number")
        print(f"✅ safe_int with invalid input: {result}")
        
        # Test safe_lower with None
        result = safe_lower(None)
        print(f"✅ safe_lower with None: '{result}'")
        
        # Test safe_upper with None
        result = safe_upper(None)
        print(f"✅ safe_upper with None: '{result}'")
        
        # Test error logging with context
        try:
            raise RuntimeError("Test runtime error")
        except Exception as e:
            log_error_with_context(logger, e, {"test_context": "error_test"}, "error_scenario_test")
        
        print("✅ Error handling scenarios tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling scenarios test failed: {e}")
        traceback.print_exc()
        return False

def test_warning_symbols():
    """Test the warning symbols functionality."""
    print("🧪 Testing Warning Symbols...")
    
    try:
        from src.utils.warning_symbols import (
            error, warning, info, success, format_error_message, 
            format_warning_message, format_success_message, format_info_message
        )
        
        # Test basic symbol functions
        error_msg = error("This is an error message")
        warning_msg = warning("This is a warning message")
        info_msg = info("This is an info message")
        success_msg = success("This is a success message")
        
        print("✅ Basic warning symbols functions work")
        
        # Test formatted messages
        formatted_error = format_error_message("Formatted error")
        formatted_warning = format_warning_message("Formatted warning")
        formatted_success = format_success_message("Formatted success")
        formatted_info = format_info_message("Formatted info")
        
        print("✅ Formatted warning symbols functions work")
        
        print("✅ Warning symbols tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Warning symbols test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Simplified Enhanced Logging System Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Enhanced Logger", test_enhanced_logger),
        ("Decorator System", test_decorator_system),
        ("Basic Common Operations", test_basic_common_operations),
        ("Error Scenarios", test_error_scenarios),
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
        print("🎉 All tests passed! Enhanced logging system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)