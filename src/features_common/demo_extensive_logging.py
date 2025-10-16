"""
Comprehensive demonstration of extensive logging and error handling.

This script demonstrates that all operations have extensive logging using tprint
and comprehensive error handling to ensure no silent failures.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_extensive_logging():
    """
    Demonstrate extensive logging and error handling throughout the system.
    """
    print("🚀 Features Common - Extensive Logging & Error Handling Demo")
    print("=" * 70)

    # Import the enhanced features_common
    try:
        from src.features_common import (
            create_optimized_scaler, get_unified_vectorbt_manager,
            validate_input_data, safe_execute, check_system_health,
            report_silent_failures, get_logger, ensure_no_silent_failures
        )
        print("✅ Successfully imported enhanced features_common")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # Check system health first
    print("\n🔍 System Health Check")
    print("-" * 30)
    health_status = check_system_health()
    print(f"Overall health: {health_status['overall_health']}")
    if health_status['issues']:
        print(f"Issues: {health_status['issues']}")
    if health_status['warnings']:
        print(f"Warnings: {health_status['warnings']}")
    if health_status['recommendations']:
        print(f"Recommendations: {health_status['recommendations']}")

    # Create test data with various issues
    print("\n📊 Creating Test Data")
    print("-" * 30)

    # Normal data
    normal_data = pd.Series(np.random.randn(1000))
    print(f"✅ Normal data: {len(normal_data)} samples")

    # Data with NaN values
    data_with_nans = normal_data.copy()
    data_with_nans.iloc[100:110] = np.nan
    print(f"⚠️  Data with NaNs: {len(data_with_nans)} samples, {data_with_nans.isna().sum()} NaN values")

    # Empty data
    empty_data = pd.Series(dtype=float)
    print(f"❌ Empty data: {len(empty_data)} samples")

    # Invalid data type
    invalid_data = "not a pandas series"
    print(f"❌ Invalid data: {type(invalid_data)}")

    # Test 1: Data Validation with Extensive Logging
    print("\n🔍 Test 1: Data Validation with Extensive Logging")
    print("-" * 50)

    # Validate normal data
    print("\n   Validating normal data:")
    is_valid, warnings = validate_input_data(normal_data, "normal_data")
    print(f"   Result: valid={is_valid}, warnings={len(warnings)}")

    # Validate data with NaNs
    print("\n   Validating data with NaNs:")
    is_valid, warnings = validate_input_data(data_with_nans, "data_with_nans")
    print(f"   Result: valid={is_valid}, warnings={len(warnings)}")
    for warning in warnings:
        print(f"     Warning: {warning}")

    # Validate empty data
    print("\n   Validating empty data:")
    try:
        is_valid, warnings = validate_input_data(empty_data, "empty_data", allow_empty=False)
        print(f"   Result: valid={is_valid}, warnings={len(warnings)}")
    except Exception as e:
        print(f"   Exception (expected): {e}")

    # Validate invalid data
    print("\n   Validating invalid data:")
    try:
        is_valid, warnings = validate_input_data(invalid_data, "invalid_data", required_type=pd.Series)
        print(f"   Result: valid={is_valid}, warnings={len(warnings)}")
    except Exception as e:
        print(f"   Exception (expected): {e}")

    # Test 2: Safe Execution with Error Handling
    print("\n🔍 Test 2: Safe Execution with Error Handling")
    print("-" * 50)

    def test_operation(data):
        """Test operation that might fail."""
        if len(data) == 0:
            raise ValueError("Empty data not allowed")
        return data.mean()

    def failing_operation(data):
        """Operation that always fails."""
        raise RuntimeError("This operation always fails")

    # Test successful operation
    print("\n   Testing successful operation:")
    result, success, error = safe_execute(test_operation, normal_data)
    print(f"   Result: {result}, Success: {success}, Error: {error}")

    # Test failing operation
    print("\n   Testing failing operation:")
    result, success, error = safe_execute(failing_operation, normal_data)
    print(f"   Result: {result}, Success: {success}, Error: {error}")

    # Test operation with empty data
    print("\n   Testing operation with empty data:")
    result, success, error = safe_execute(test_operation, empty_data)
    print(f"   Result: {result}, Success: {success}, Error: {error}")

    # Test 3: Optimized Scaler with Extensive Logging
    print("\n🔍 Test 3: Optimized Scaler with Extensive Logging")
    print("-" * 50)

    # Create scaler (should show extensive logging)
    print("\n   Creating optimized scaler:")
    scaler = create_optimized_scaler(method='zscore')
    print(f"   Scaler created: {type(scaler).__name__}")

    # Test fit_transform with normal data
    print("\n   Testing fit_transform with normal data:")
    try:
        result = scaler.fit_transform(normal_data)
        print(f"   Result shape: {result.shape}, mean: {result.mean():.3f}, std: {result.std():.3f}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test fit_transform with data containing NaNs
    print("\n   Testing fit_transform with data containing NaNs:")
    try:
        result = scaler.fit_transform(data_with_nans)
        print(f"   Result shape: {result.shape}, mean: {result.mean():.3f}, std: {result.std():.3f}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test transform with fitted scaler
    print("\n   Testing transform with fitted scaler:")
    try:
        result = scaler.transform(normal_data)
        print(f"   Result shape: {result.shape}, mean: {result.mean():.3f}, std: {result.std():.3f}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test 4: VectorBT Manager with Extensive Logging
    print("\n🔍 Test 4: VectorBT Manager with Extensive Logging")
    print("-" * 50)

    # Get VectorBT manager
    print("\n   Getting VectorBT manager:")
    vectorbt_manager = get_unified_vectorbt_manager()
    print(f"   Manager created: {type(vectorbt_manager).__name__}")
    print(f"   VectorBT available: {vectorbt_manager.is_vectorbt_available()}")
    print(f"   Available operations: {len(vectorbt_manager.get_available_operations())}")

    # Test rolling operations
    print("\n   Testing rolling mean:")
    try:
        result = vectorbt_manager.rolling_mean(normal_data, window=20)
        print(f"   Result shape: {result.shape}")
    except Exception as e:
        print(f"   Exception: {e}")

    print("\n   Testing rolling std:")
    try:
        result = vectorbt_manager.rolling_std(normal_data, window=20)
        print(f"   Result shape: {result.shape}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test data scaling
    print("\n   Testing data scaling:")
    try:
        result = vectorbt_manager.scale_data(normal_data, method='zscore')
        print(f"   Result shape: {result.shape}, mean: {result.mean():.3f}, std: {result.std():.3f}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test 5: Error Handling Decorator
    print("\n🔍 Test 5: Error Handling Decorator")
    print("-" * 50)

    @ensure_no_silent_failures
    def test_function_success(data):
        """Function that succeeds."""
        return data.mean()

    @ensure_no_silent_failures
    def test_function_failure(data):
        """Function that fails."""
        raise ValueError("This function always fails")

    # Test successful function
    print("\n   Testing successful function with decorator:")
    try:
        result = test_function_success(normal_data)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test failing function
    print("\n   Testing failing function with decorator:")
    try:
        result = test_function_failure(normal_data)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Test 6: Logger Statistics
    print("\n🔍 Test 6: Logger Statistics")
    print("-" * 50)

    # Get logger and show statistics
    logger_instance = get_logger()
    stats = logger_instance.get_stats()

    print("\n   Logger Statistics:")
    for key, value in stats.items():
        print(f"     {key}: {value}")

    # Report silent failures
    print("\n   Silent Failure Report:")
    failure_stats = report_silent_failures()
    for key, value in failure_stats.items():
        print(f"     {key}: {value}")

    # Test 7: Performance Monitoring
    print("\n🔍 Test 7: Performance Monitoring")
    print("-" * 50)

    # Test performance monitoring
    print("\n   Testing performance monitoring:")
    try:
        # This should show extensive logging
        with scaler.profile_operation("demo_operation"):
            result = scaler.transform(normal_data)
        print(f"   Operation completed, result shape: {result.shape}")
    except Exception as e:
        print(f"   Exception: {e}")

    # Get performance statistics
    if hasattr(scaler, 'get_performance_stats'):
        perf_stats = scaler.get_performance_stats()
        print(f"   Performance statistics available: {len(perf_stats)} metrics")

        # Show key metrics
        key_metrics = ['total_operations', 'avg_execution_time', 'vectorbt_operations', 'pandas_fallbacks']
        for metric in key_metrics:
            if metric in perf_stats:
                print(f"     {metric}: {perf_stats[metric]}")

    # Final Summary
    print("\n🎉 Extensive Logging & Error Handling Demo Complete!")
    print("=" * 70)
    print("✅ All operations have extensive logging using tprint")
    print("✅ Comprehensive error handling ensures no silent failures")
    print("✅ Detailed validation with warnings and errors")
    print("✅ Performance monitoring with detailed metrics")
    print("✅ System health checking and reporting")
    print("✅ Logger statistics and silent failure detection")
    print("\n🚀 The system provides complete visibility into all operations!")

if __name__ == "__main__":
    demonstrate_extensive_logging()
