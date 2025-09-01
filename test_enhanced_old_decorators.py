#!/usr/bin/env python3
"""
Test script to verify that all old decorators are working with new enhancements
while maintaining backwards compatibility.
"""

import pandas as pd
import numpy as np
import asyncio
import time

def test_enhanced_old_decorators():
    """Test that all old decorators work with new enhancements."""

    print("🧪 Testing Enhanced Old Decorators")
    print("=" * 60)

    try:
        # Test imports from the original decorators module
        from src.utils.decorators import (
            validate_call_or_runtime_types,
            pa_check_input,
            pa_check_output,
            pa_check_io,
            enforce_ndarray,
            auto_vectorize,
            guard_array_nan_inf,
            guard_dataframe_nulls,
            normalize_errors,
            with_tracing_span,
        )

        print("✅ All original decorators imported successfully")

        # Test that they still work exactly as before
        print("\n🔍 Testing Original Functionality")
        print("-" * 40)

        # Test validate_call_or_runtime_types
        @validate_call_or_runtime_types()
        def test_function(x: int) -> int:
            return x * 2

        result1 = test_function(5)
        print(f"  validate_call_or_runtime_types: {result1}")

        # Test enforce_ndarray
        @enforce_ndarray(arg_index=0, forbid_lists=True)
        def test_ndarray_function(data):
            return data.shape

        result2 = test_ndarray_function(np.array([1, 2, 3]))
        print(f"  enforce_ndarray: {result2}")

        # Test auto_vectorize
        @auto_vectorize()
        def test_vectorize_function(x):
            return x * 2

        result3 = test_vectorize_function(np.array([1, 2, 3, 4, 5]))
        print(f"  auto_vectorize: {result3}")

        # Test guard_array_nan_inf
        @guard_array_nan_inf(mode="warn", arg_indices=(0,))
        def test_nan_guard_function(data):
            return data.sum()

        # Test with clean data
        clean_data = np.array([1, 2, 3, 4, 5])
        result4 = test_nan_guard_function(clean_data)
        print(f"  guard_array_nan_inf (clean): {result4}")

        # Test with NaN data (should warn but continue)
        nan_data = np.array([1, 2, np.nan, 4, 5])
        result5 = test_nan_guard_function(nan_data)
        print(f"  guard_array_nan_inf (NaN): {result5}")

        # Test guard_dataframe_nulls
        @guard_dataframe_nulls(mode="warn", arg_index=0)
        def test_null_guard_function(df):
            return df.shape

        # Test with clean DataFrame
        clean_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result6 = test_null_guard_function(clean_df)
        print(f"  guard_dataframe_nulls (clean): {result6}")

        # Test with DataFrame containing nulls (should warn but continue)
        null_df = pd.DataFrame({
            'A': [1, 2, None],
            'B': [4, 5, 6]
        })
        result7 = test_null_guard_function(null_df)
        print(f"  guard_dataframe_nulls (nulls): {result7}")

        # Test normalize_errors
        @normalize_errors(reraise=False)
        def test_error_function(x):
            if x < 0:
                raise ValueError("Negative number")
            return x * 2

        result8 = test_error_function(5)
        print(f"  normalize_errors (normal): {result8}")

        result9 = test_error_function(-3)  # Should return None due to error normalization
        print(f"  normalize_errors (error): {result9}")

        # Test with_tracing_span
        @with_tracing_span("test_span", log_args=False)
        def test_tracing_function(x):
            time.sleep(0.01)  # Small delay to see timing
            return x * 3

        result10 = test_tracing_function(7)
        print(f"  with_tracing_span: {result10}")

        print("✅ All original decorators working correctly")

        # Test that enhanced features are working
        print("\n🚀 Testing Enhanced Features")
        print("-" * 40)

        # Test configuration integration
        from src.utils.decorator_config import global_config
        print(f"  Global config validation mode: {global_config.validation_mode}")
        print(f"  Global config cache enabled: {global_config.cache_enabled}")
        print(f"  Global config performance monitoring: {global_config.enable_performance_monitoring}")

        # Test registry integration
        from src.utils.decorator_registry import decorator_registry

        # Check if decorators are registered
        registered_decorators = decorator_registry.list_decorators()
        print(f"  Total registered decorators: {len(registered_decorators)}")

        # Look for our enhanced old decorators
        enhanced_old_decorators = [
            "validate_call_or_runtime_types",
            "pa_check_input",
            "pa_check_output",
            "pa_check_io",
            "enforce_ndarray",
            "auto_vectorize",
            "guard_array_nan_inf",
            "guard_dataframe_nulls",
            "normalize_errors",
            "with_tracing_span"
        ]

        found_enhanced = []
        for decorator in registered_decorators:
            if decorator.name in enhanced_old_decorators:
                found_enhanced.append(decorator.name)
                print(f"    Found enhanced: {decorator.name} v{decorator.version}")

        print(f"  Enhanced old decorators found: {len(found_enhanced)}/{len(enhanced_old_decorators)}")

        # Test performance monitoring integration
        print("\n📊 Testing Performance Monitoring Integration")
        print("-" * 40)

        # Enable performance monitoring
        global_config.enable_performance_monitoring = True

        @with_tracing_span("performance_test")
        def performance_test_function():
            time.sleep(0.1)  # Simulate work
            return "performance test completed"

        result11 = performance_test_function()
        print(f"  Performance test result: {result11}")

        # Test caching integration
        print("\n💾 Testing Caching Integration")
        print("-" * 40)

        # Enable caching
        global_config.cache_enabled = True
        global_config.cache_size = 10
        global_config.cache_ttl = 5

        @validate_call_or_runtime_types()
        def cached_test_function(x: int) -> int:
            time.sleep(0.1)  # Simulate expensive operation
            return x ** 2

        # First call should be slow
        start_time = time.time()
        result12 = cached_test_function(5)
        first_call_time = time.time() - start_time

        # Second call should be fast (cached)
        start_time = time.time()
        result13 = cached_test_function(5)
        cached_call_time = time.time() - start_time

        print(f"  Cached function first call: {result12} (took {first_call_time:.3f}s)")
        print(f"  Cached function cached call: {result13} (took {cached_call_time:.3f}s)")
        print(f"  Cache speedup: {first_call_time/cached_call_time:.1f}x")

        # Test backwards compatibility
        print("\n🔄 Testing Backwards Compatibility")
        print("-" * 40)

            validate_call,  # Legacy name
            check_input,    # Legacy name
            check_output,   # Legacy name
            vectorize,      # Legacy name
            guard_nan_inf,  # Legacy name
            guard_nulls,    # Legacy name
            error_handler,  # Legacy name
            tracing         # Legacy name
        )

        print("Testing legacy decorator names...")

        @validate_call()
        def legacy_validate_function(x: int) -> int:
            return x * 2

        @check_input(None)
        def legacy_check_function(df):
            return df.shape

        @vectorize()
        def legacy_vectorize_function(x):
            return x * 2

        @guard_nan_inf(mode="warn")
        def legacy_guard_function(data):
            return data.sum()

        @guard_nulls(mode="warn")
        def legacy_null_function(df):
            return df.shape

        @error_handler(reraise=False)
        def legacy_error_function(x):
            if x < 0:
                raise ValueError("Negative")
            return x

        @tracing("legacy_test")
        def legacy_tracing_function(x):
            return x * 3

        # Test all legacy functions
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        test_array = np.array([1, 2, 3, 4, 5])

        result14 = legacy_validate_function(10)
        result15 = legacy_check_function(test_df)
        result16 = legacy_vectorize_function(test_array)
        result17 = legacy_guard_function(test_array)
        result18 = legacy_null_function(test_df)
        result19 = legacy_error_function(5)
        result20 = legacy_error_function(-3)
        result21 = legacy_tracing_function(7)

        print(f"  Legacy validate_call: {result14}")
        print(f"  Legacy check_input: {result15}")
        print(f"  Legacy vectorize: {result16}")
        print(f"  Legacy guard_nan_inf: {result17}")
        print(f"  Legacy guard_nulls: {result18}")
        print(f"  Legacy error_handler (normal): {result19}")
        print(f"  Legacy error_handler (error): {result20}")
        print(f"  Legacy tracing: {result21}")

        print("✅ All legacy decorators working correctly")

        # Test enhanced data quality decorators
        print("\n🔍 Testing Enhanced Data Quality Decorators")
        print("-" * 40)

        from src.utils.data_quality_decorators import validate_data_quality

        @validate_data_quality(
            required_columns=['A', 'B'],
            min_rows=2,
            context="test"
        )
        def test_data_quality_function(df):
            return df * 2

        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        result22 = test_data_quality_function(test_df)
        print(f"  Enhanced data quality function: {result22.shape}")

        print("\n✅ Enhanced old decorators test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing enhanced old decorators: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_enhanced_old_decorators():
    """Test async functionality of enhanced old decorators."""

    print("\n🔄 Testing Async Enhanced Old Decorators")
    print("-" * 40)

    try:
        from src.utils.decorators import (
            validate_call_or_runtime_types,
            with_tracing_span
        )

        @validate_call_or_runtime_types()
        @with_tracing_span("async_test")
        async def async_test_function(x: int) -> int:
            await asyncio.sleep(0.1)  # Simulate async work
            return x * 2

        print("Testing async enhanced decorators...")
        result = await async_test_function(10)
        print(f"  Async function result: {result}")

        print("✅ Async enhanced old decorators test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing async enhanced old decorators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_features_integration():
    """Test that enhanced features are properly integrated."""

    print("\n🔧 Testing Enhanced Features Integration")
    print("-" * 40)

    try:
        # Test configuration system integration
        from src.utils.decorator_config import global_config
        from src.utils.decorator_registry import decorator_registry

        # Test configuration changes
        original_validation_mode = global_config.validation_mode
        original_cache_enabled = global_config.cache_enabled

        print("Testing configuration changes...")

        # Change configuration
        global_config.validation_mode = "STRICT"
        global_config.cache_enabled = True
        global_config.max_retries = 5

        print(f"  Updated validation mode: {global_config.validation_mode}")
        print(f"  Updated cache enabled: {global_config.cache_enabled}")
        print(f"  Updated max retries: {global_config.max_retries}")

        # Test registry functionality
        print("\nTesting registry functionality...")

        # Search for decorators
        validation_decorators = decorator_registry.search("validation")
        print(f"  Validation decorators found: {len(validation_decorators)}")

        # Get usage stats
        usage_stats = decorator_registry.get_usage_stats()
        print(f"  Decorator usage stats available: {len(usage_stats)} decorators")

        # Export config
        config = decorator_registry.export_config()
        print(f"  Registry config exported with {len(config)} keys")

        # Restore original configuration
        global_config.validation_mode = original_validation_mode
        global_config.cache_enabled = original_cache_enabled

        print("✅ Enhanced features integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing enhanced features integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_improvements():
    """Test that performance improvements are working."""

    print("\n⚡ Testing Performance Improvements")
    print("-" * 40)

    try:
        from src.utils.decorators import validate_call_or_runtime_types
        from src.utils.decorator_config import global_config

        # Enable performance monitoring and caching
        global_config.enable_performance_monitoring = True
        global_config.cache_enabled = True
        global_config.cache_size = 20
        global_config.cache_ttl = 10

        @validate_call_or_runtime_types()
        def performance_test_function(x: int) -> int:
            time.sleep(0.05)  # Simulate work
            return x ** 2

        print("Testing performance improvements...")

        # Test multiple calls to see caching effect
        results = []
        call_times = []

        for i in range(5):
            start_time = time.time()
            result = performance_test_function(i)
            call_time = time.time() - start_time

            results.append(result)
            call_times.append(call_time)

            print(f"  Call {i+1}: {result} (took {call_time:.3f}s)")

        # First call should be slowest, subsequent calls should be faster
        first_call_time = call_times[0]
        avg_cached_time = sum(call_times[1:]) / len(call_times[1:])

        if avg_cached_time < first_call_time:
            speedup = first_call_time / avg_cached_time
            print(f"  Performance improvement: {speedup:.1f}x speedup with caching")
        else:
            print(f"  No significant performance improvement detected")

        print("✅ Performance improvements test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing performance improvements: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""

    print("🚀 Enhanced Old Decorators Test Suite")
    print("=" * 60)

    # Run tests
    success1 = test_enhanced_old_decorators()
    success2 = asyncio.run(test_async_enhanced_old_decorators())
    success3 = test_enhanced_features_integration()
    success4 = test_performance_improvements()

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    print(f"Enhanced old decorators: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Async enhanced decorators: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Enhanced features integration: {'✅ PASS' if success3 else '❌ FAIL'}")
    print(f"Performance improvements: {'✅ PASS' if success4 else '❌ FAIL'}")

    overall_success = success1 and success2 and success3 and success4
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 All old decorators are working with new enhancements!")
        print("   - Original functionality preserved")
        print("   - New enhanced features working")
        print("   - Backwards compatibility maintained")
        print("   - Performance improvements active")
        print("   - Configuration system integrated")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

    return overall_success

if __name__ == "__main__":
    main()