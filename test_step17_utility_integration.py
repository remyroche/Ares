"""
Test Script for Step17 Utility Integration

This script tests the integration of all specified utility modules:
- src/utils/common_operations.py
- src/utils/math_validation.py
- src/utils/parquet_utils.py
- src/core/decorators/
- src/core/errors/
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('/workspace')

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("Step17UtilityIntegrationTest")

# Test utility imports
def test_utility_imports():
    """Test that all utility modules can be imported."""
    print("\n🧪 Testing Utility Module Imports...")
    
    try:
        # Test common_operations imports
        from src.utils.common_operations import (
            safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
            safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
            format_datetime, safe_dict_get, safe_dict_items, safe_append,
            safe_extend, get_logger, setup_basic_logging, generate_hash,
            generate_cache_key, safe_deepcopy, safe_copy, validate_dataframe,
            validate_numeric_range, safe_sleep, safe_gather, create_async_task,
            timed_operation, format_bytes, chunked_iterable, parallel_map
        )
        print("✅ src/utils/common_operations.py imported successfully")
        
        # Test math_validation imports
        from src.utils.math_validation import (
            safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
            validate_positive, validate_range, safe_kelly_calculation,
            safe_weighted_average, safe_percentage_change, validate_correlation_matrix,
            safe_matrix_inverse, math_safe, MathValidationError
        )
        print("✅ src/utils/math_validation.py imported successfully")
        
        # Test parquet_utils imports
        from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
        print("✅ src/utils/parquet_utils.py imported successfully")
        
        # Test core decorators imports
        from src.core.decorators import (
            handles_errors, error_boundary, retry, timeout, circuit_breaker,
            log_call, log_execution_time, traced, cached, memoize,
            validate_dataframe as validate_df_decorator, validates
        )
        print("✅ src/core/decorators/ imported successfully")
        
        # Test core errors imports
        from src.core.errors import (
            AppError, ValidationError, NotFoundError, TimeoutError,
            ServiceUnavailableError, BusinessRuleError, DataIntegrityError,
            ErrorCode, ErrorMapper, error_mapper
        )
        print("✅ src/core/errors/ imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_common_operations_integration():
    """Test common operations utility integration."""
    print("\n🧪 Testing Common Operations Integration...")
    
    try:
        from src.utils.common_operations import (
            safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
            safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
            format_datetime, safe_dict_get, safe_dict_items, safe_append,
            safe_extend, get_logger, generate_hash, generate_cache_key
        )
        
        # Test safe operations
        test_data = {'test': 'data', 'number': 42}
        
        # Test JSON operations
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        safe_json_dump(test_data, temp_file)
        loaded_data = safe_json_load(temp_file)
        assert loaded_data == test_data, "JSON operations failed"
        
        # Test file operations
        assert safe_file_exists(temp_file), "File existence check failed"
        
        # Test directory operations
        test_dir = tempfile.mkdtemp()
        ensure_directory(test_dir)
        assert os.path.exists(test_dir), "Directory creation failed"
        
        # Test math operations
        test_values = [1, 2, 3, 4, 5]
        mean_val = safe_mean(test_values)
        std_val = safe_std(test_values)
        assert abs(mean_val - 3.0) < 0.01, "Mean calculation failed"
        assert std_val > 0, "Standard deviation calculation failed"
        
        # Test type conversions
        float_val = safe_float("3.14", 0.0)
        int_val = safe_int("42", 0)
        assert float_val == 3.14, "Float conversion failed"
        assert int_val == 42, "Int conversion failed"
        
        # Test datetime operations
        current_time = get_current_datetime()
        formatted_time = format_datetime(current_time)
        assert isinstance(current_time, datetime), "Datetime operation failed"
        assert isinstance(formatted_time, str), "Datetime formatting failed"
        
        # Test dictionary operations
        test_dict = {'key1': 'value1', 'key2': 'value2'}
        retrieved_value = safe_dict_get(test_dict, 'key1', 'default')
        assert retrieved_value == 'value1', "Dictionary get failed"
        
        # Test list operations
        test_list = [1, 2, 3]
        appended_list = safe_append(test_list, 4)
        assert len(appended_list) == 4, "List append failed"
        
        extended_list = safe_extend(appended_list, [5, 6])
        assert len(extended_list) == 6, "List extend failed"
        
        # Test hash generation
        hash_val = generate_hash("test_string", 'md5')
        assert isinstance(hash_val, str), "Hash generation failed"
        assert len(hash_val) == 32, "MD5 hash length incorrect"
        
        # Cleanup
        os.unlink(temp_file)
        os.rmdir(test_dir)
        
        print("✅ Common operations integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Common operations integration test failed: {e}")
        return False

async def test_math_validation_integration():
    """Test math validation utility integration."""
    print("\n🧪 Testing Math Validation Integration...")
    
    try:
        from src.utils.math_validation import (
            safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
            validate_positive, validate_range, safe_kelly_calculation,
            safe_weighted_average, safe_percentage_change, MathValidationError
        )
        
        # Test safe division
        result = safe_divide(10, 2, 0.0)
        assert result == 5.0, "Safe division failed"
        
        result = safe_divide(10, 0, 0.0)
        assert result == 0.0, "Division by zero handling failed"
        
        # Test safe logarithm
        result = safe_log(10, math.e, 0.0)
        assert result > 0, "Safe logarithm failed"
        
        result = safe_log(0, math.e, 0.0)
        assert result == 0.0, "Log of zero handling failed"
        
        # Test safe square root
        result = safe_sqrt(16, 0.0)
        assert result == 4.0, "Safe square root failed"
        
        result = safe_sqrt(-4, 0.0)
        assert result == 0.0, "Square root of negative handling failed"
        
        # Test safe power
        result = safe_power(2, 3, 1.0)
        assert result == 8.0, "Safe power failed"
        
        # Test validation functions
        try:
            validate_finite(3.14, "test_value")
            validate_positive(5.0, "test_value")
            validate_range(0.5, 0.0, 1.0, "test_value")
            print("✅ Validation functions work correctly")
        except MathValidationError:
            print("❌ Validation functions failed")
            return False
        
        # Test Kelly calculation
        kelly_result = safe_kelly_calculation(0.6, 100, 50, 1.0)
        assert kelly_result >= 0, "Kelly calculation failed"
        
        # Test weighted average
        values = [1, 2, 3, 4, 5]
        weights = [1, 1, 1, 1, 1]
        weighted_avg = safe_weighted_average(values, weights, 0.0)
        assert abs(weighted_avg - 3.0) < 0.01, "Weighted average failed"
        
        # Test percentage change
        pct_change = safe_percentage_change(100, 120, 0.0)
        assert abs(pct_change - 20.0) < 0.01, "Percentage change failed"
        
        print("✅ Math validation integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Math validation integration test failed: {e}")
        return False

async def test_parquet_utils_integration():
    """Test parquet utils integration."""
    print("\n🧪 Testing Parquet Utils Integration...")
    
    try:
        from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
        
        # Test ParquetUtils instantiation
        parquet_utils = get_parquet_utils()
        assert isinstance(parquet_utils, ParquetUtils), "ParquetUtils instantiation failed"
        
        # Test file validation (with non-existent file)
        validation_result = parquet_utils.validate_parquet_file("/nonexistent/file.parquet")
        assert not validation_result["valid"], "File validation should fail for non-existent file"
        assert not validation_result["file_exists"], "File existence check failed"
        
        print("✅ Parquet utils integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Parquet utils integration test failed: {e}")
        return False

async def test_core_decorators_integration():
    """Test core decorators integration."""
    print("\n🧪 Testing Core Decorators Integration...")
    
    try:
        from src.core.decorators import (
            handles_errors, error_boundary, retry, timeout, circuit_breaker,
            log_call, log_execution_time, traced, cached, memoize
        )
        
        # Test handles_errors decorator
        @handles_errors(default_return="error", context="test")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success", "handles_errors decorator failed"
        
        # Test error handling
        @handles_errors(default_return="error", context="test")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "error", "Error handling decorator failed"
        
        # Test log_call decorator
        @log_call
        def logged_function():
            return "logged"
        
        result = logged_function()
        assert result == "logged", "log_call decorator failed"
        
        # Test cached decorator
        call_count = 0
        
        @cached
        def cached_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = cached_function(5)
        result2 = cached_function(5)
        assert result1 == result2 == 10, "Cached decorator failed"
        assert call_count == 1, "Caching didn't work"
        
        print("✅ Core decorators integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Core decorators integration test failed: {e}")
        return False

async def test_core_errors_integration():
    """Test core errors integration."""
    print("\n🧪 Testing Core Errors Integration...")
    
    try:
        from src.core.errors import (
            AppError, ValidationError, NotFoundError, TimeoutError,
            ServiceUnavailableError, BusinessRuleError, DataIntegrityError,
            ErrorCode, ErrorMapper, error_mapper
        )
        
        # Test custom error creation
        validation_error = ValidationError("Test validation error")
        assert str(validation_error) == "Test validation error", "ValidationError creation failed"
        
        not_found_error = NotFoundError("Test not found error")
        assert str(not_found_error) == "Test not found error", "NotFoundError creation failed"
        
        business_rule_error = BusinessRuleError("Test business rule error")
        assert str(business_rule_error) == "Test business rule error", "BusinessRuleError creation failed"
        
        # Test error inheritance
        assert isinstance(validation_error, AppError), "Error inheritance failed"
        assert isinstance(not_found_error, AppError), "Error inheritance failed"
        assert isinstance(business_rule_error, AppError), "Error inheritance failed"
        
        print("✅ Core errors integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Core errors integration test failed: {e}")
        return False

async def test_step17_utility_integration():
    """Test the integrated Step17 implementation."""
    print("\n🧪 Testing Step17 Utility Integration...")
    
    try:
        # Test importing the integrated implementation
        from src.training.steps.optimisation.step17_optimized_with_utils import (
            ThreadSafeConfigManager, ParameterResultCache, AdvancedOptimizationStrategies,
            IntelligentParameterGrouper, ResourceValidator, InputValidator, ResultValidator,
            ValidationResult, OptimizationMetrics, ParameterGroup,
            Step17ValidationError, Step17ResourceError, Step17OptimizationError,
            memory_efficient_context
        )
        
        # Test component instantiation
        logger = logging.getLogger("test")
        
        config_manager = ThreadSafeConfigManager()
        assert config_manager is not None, "ThreadSafeConfigManager instantiation failed"
        
        parameter_cache = ParameterResultCache(max_size=100)
        assert parameter_cache is not None, "ParameterResultCache instantiation failed"
        
        optimization_strategies = AdvancedOptimizationStrategies(logger)
        assert optimization_strategies is not None, "AdvancedOptimizationStrategies instantiation failed"
        
        parameter_grouper = IntelligentParameterGrouper(logger)
        assert parameter_grouper is not None, "IntelligentParameterGrouper instantiation failed"
        
        resource_validator = ResourceValidator(logger)
        assert resource_validator is not None, "ResourceValidator instantiation failed"
        
        input_validator = InputValidator(logger)
        assert input_validator is not None, "InputValidator instantiation failed"
        
        result_validator = ResultValidator(logger)
        assert result_validator is not None, "ResultValidator instantiation failed"
        
        # Test custom exceptions
        validation_error = Step17ValidationError("Test validation error")
        assert isinstance(validation_error, ValidationError), "Step17ValidationError inheritance failed"
        
        resource_error = Step17ResourceError("Test resource error")
        assert isinstance(resource_error, ServiceUnavailableError), "Step17ResourceError inheritance failed"
        
        optimization_error = Step17OptimizationError("Test optimization error")
        assert isinstance(optimization_error, BusinessRuleError), "Step17OptimizationError inheritance failed"
        
        # Test memory efficient context
        async with memory_efficient_context(max_memory_gb=0.1):
            # Simulate some work
            await asyncio.sleep(0.01)
        
        print("✅ Step17 utility integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Step17 utility integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all utility integration tests."""
    print("🚀 Starting Step17 Utility Integration Tests")
    print("=" * 60)
    
    test_functions = [
        test_utility_imports,
        test_common_operations_integration,
        test_math_validation_integration,
        test_parquet_utils_integration,
        test_core_decorators_integration,
        test_core_errors_integration,
        test_step17_utility_integration
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All utility integration tests passed!")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed. Please review the integration.")
    
    return passed_tests == total_tests

def main():
    """Main test function."""
    print("🧪 Step17 Utility Integration Test Suite")
    print("Testing integration of all specified utility modules")
    print()
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ All utility modules are properly integrated!")
        print("\n📋 Integrated Modules:")
        print("   ✅ src/utils/common_operations.py")
        print("   ✅ src/utils/math_validation.py")
        print("   ✅ src/utils/parquet_utils.py")
        print("   ✅ src/core/decorators/")
        print("   ✅ src/core/errors/")
        
        print("\n🚀 The Step17 implementation is ready with full utility integration!")
    else:
        print("\n❌ Some integration tests failed. Please review the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    import math
    main()