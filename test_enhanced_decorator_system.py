#!/usr/bin/env python3
"""
Test script for the enhanced decorator system
Demonstrates new functionality while maintaining backwards compatibility
"""

import pandas as pd
import numpy as np
import asyncio
import time
import warnings
from datetime import datetime, timedelta

def test_enhanced_decorator_system():
    """Test the enhanced decorator system."""

    print("🧪 Testing Enhanced Decorator System")
    print("=" * 60)

    try:
        # Test imports from the new system
        from src.utils.decorator_config import global_config, ValidationMode, PerformanceMode
        from src.utils.decorator_registry import decorator_registry
        from src.utils.enhanced_decorators import (
            smart_error_recovery,
            cached_validation,
            enhanced_validation,
            performance_monitor_v2,
            ValidationResult,
            ValidatableData
        )
        from src.utils.decorator_compatibility import (
            get_decorator_config,
            set_decorator_config,
            list_available_decorators,
            get_decorator_usage_stats,
            search_decorators
        )
        from src.utils.centralized_decorators_v2 import (
            validate_data_quality_v2,
            quality_gate_v2,
            step_specific_ml_validation_v2,
            auto_fix_data_quality_issues_v2,
            monitor_feature_engineering_v2,
            monitor_data_collection_v2
        )

        print("✅ All enhanced decorators imported successfully")

        # Test configuration system
        print("\n📋 Testing Configuration System")
        print("-" * 40)

        # Test global config
        print(f"Default validation mode: {global_config.validation_mode}")
        print(f"Default performance mode: {global_config.performance_mode}")
        print(f"Cache enabled: {global_config.cache_enabled}")
        print(f"Max retries: {global_config.max_retries}")

        # Test config modification
        set_decorator_config(validation_mode="STRICT", max_retries=5)
        updated_config = get_decorator_config()
        print(f"Updated validation mode: {updated_config.validation_mode}")
        print(f"Updated max retries: {updated_config.max_retries}")

        # Test decorator registry
        print("\n📚 Testing Decorator Registry")
        print("-" * 40)

        available_decorators = list_available_decorators()
        print(f"Available decorators: {len(available_decorators)}")

        # List some decorators
        for decorator in available_decorators[:5]:
            print(f"  - {decorator.name} v{decorator.version} ({', '.join(decorator.tags)})")

        # Test search functionality
        validation_decorators = search_decorators("validation")
        print(f"Validation decorators found: {len(validation_decorators)}")

        # Test enhanced decorators
        print("\n🚀 Testing Enhanced Decorators")
        print("-" * 40)

        # Test smart error recovery
        @smart_error_recovery(max_retries=2, fallback_strategy="default_return")
        def function_with_errors(x):
            if x < 0:
                raise ValueError("Negative number not allowed")
            return x * 2

        print("Testing smart error recovery...")
        result1 = function_with_errors(5)
        print(f"  Normal execution: {result1}")

        result2 = function_with_errors(-3)
        print(f"  Error recovery: {result2}")

        # Test cached validation
        @cached_validation(cache_size=10, ttl_seconds=5)
        def expensive_calculation(x):
            time.sleep(0.1)  # Simulate expensive operation
            return x ** 2

        print("\nTesting cached validation...")
        start_time = time.time()
        result3 = expensive_calculation(5)
        first_call_time = time.time() - start_time

        start_time = time.time()
        result4 = expensive_calculation(5)  # Should use cache
        cached_call_time = time.time() - start_time

        print(f"  First call: {result3} (took {first_call_time:.3f}s)")
        print(f"  Cached call: {result4} (took {cached_call_time:.3f}s)")
        print(f"  Cache speedup: {first_call_time/cached_call_time:.1f}x")

        # Test performance monitoring
        @performance_monitor_v2(level="detailed", track_memory=True, track_cpu=True)
        def sample_function():
            time.sleep(0.1)
            return "Hello, World!"

        print("\nTesting performance monitoring...")
        result5 = sample_function()
        print(f"  Function result: {result5}")

        # Test enhanced validation decorators
        print("\n🔍 Testing Enhanced Validation Decorators")
        print("-" * 40)

        # Test data quality validation v2
        @validate_data_quality_v2(validation_level="WARNING", auto_fix=True, context="test")
        def data_processing_function(df):
            return df * 2

        # Create test data
        test_df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, 6, 7, 8]
        })

        print("Testing data quality validation v2...")
        result6 = data_processing_function(test_df)
        print(f"  Processed data shape: {result6.shape}")

        # Test quality gate v2
        @quality_gate_v2(min_quality_score=0.6, required_grade="C", action_on_failure="warn")
        def quality_test_function():
            return "high_quality_result"

        print("\nTesting quality gate v2...")
        result7 = quality_test_function()
        print(f"  Quality gate result: {result7}")

        # Test step-specific ML validation v2
        @step_specific_ml_validation_v2("feature_engineering", adaptive_thresholds=True)
        def ml_step_function(data):
            return data + 1

        print("\nTesting step-specific ML validation v2...")
        test_data = np.array([1, 2, 3, 4, 5])
        result8 = ml_step_function(test_data)
        print(f"  ML step result: {result8}")

        # Test auto-fix decorator v2
        @auto_fix_data_quality_issues_v2(context="test", max_fix_attempts=2)
        def problematic_function(x):
            if x < 0:
                raise ValueError("Negative input")
            return x

        print("\nTesting auto-fix decorator v2...")
        result9 = problematic_function(5)
        print(f"  Normal input: {result9}")

        # Test monitoring decorators v2
        @monitor_feature_engineering_v2(track_feature_stats=True, track_memory_usage=True)
        def feature_engineering_function(data):
            return data * 2

        @monitor_data_collection_v2(track_data_volume=True, track_quality_metrics=True)
        def data_collection_function():
            return pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        print("\nTesting monitoring decorators v2...")
        result10 = feature_engineering_function(np.array([1, 2, 3]))
        result11 = data_collection_function()
        print(f"  Feature engineering result: {result10}")
        print(f"  Data collection result shape: {result11.shape}")

        # Test backwards compatibility
        print("\n🔄 Testing Backwards Compatibility")
        print("-" * 40)

        from src.utils.decorator_compatibility import (
            validate_call,  # Legacy name
            check_input,    # Legacy name
            check_output,   # Legacy name
            smart_recovery, # Alias
            cached,         # Alias
            validation,     # Alias
            performance     # Alias
        )

        print("Testing legacy decorator names...")

        @validate_call()
        def legacy_function(x: int) -> int:
            return x * 2

        @check_input(None)
        def legacy_check_function(df):
            return df

        @smart_recovery(max_retries=1)
        def legacy_recovery_function(x):
            if x < 0:
                raise ValueError("Negative")
            return x

        @cached(cache_size=5)
        def legacy_cached_function(x):
            return x ** 2

        # Test legacy functions
        result12 = legacy_function(10)
        result13 = legacy_check_function(test_df)
        result14 = legacy_recovery_function(5)
        result15 = legacy_cached_function(6)

        print(f"  Legacy validate_call: {result12}")
        print(f"  Legacy check_input: {result13.shape}")
        print(f"  Legacy recovery: {result14}")
        print(f"  Legacy cached: {result15}")

        # Test usage statistics
        print("\n📊 Testing Usage Statistics")
        print("-" * 40)

        usage_stats = get_decorator_usage_stats()
        print("Decorator usage statistics:")
        for decorator_name, usage_count in usage_stats.items():
            print(f"  {decorator_name}: {usage_count} uses")

        print("\n✅ Enhanced decorator system test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing enhanced decorator system: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_enhanced_decorators():
    """Test async enhanced decorators."""

    print("\n🔄 Testing Async Enhanced Decorators")
    print("-" * 40)

    try:
        from src.utils.enhanced_decorators import smart_error_recovery, cached_validation

        @smart_error_recovery(max_retries=2, fallback_strategy="graceful_degradation")
        async def async_function_with_errors(x):
            if x < 0:
                raise ValueError("Negative number not allowed")
            await asyncio.sleep(0.1)  # Simulate async work
            return x * 2

        @cached_validation(cache_size=5, ttl_seconds=10)
        async def async_expensive_calculation(x):
            await asyncio.sleep(0.1)  # Simulate expensive async operation
            return x ** 2

        print("Testing async smart error recovery...")
        result1 = await async_function_with_errors(5)
        print(f"  Normal async execution: {result1}")

        result2 = await async_function_with_errors(-3)
        print(f"  Async error recovery: {result2}")

        print("Testing async cached validation...")
        start_time = time.time()
        result3 = await async_expensive_calculation(5)
        first_call_time = time.time() - start_time

        start_time = time.time()
        result4 = await async_expensive_calculation(5)  # Should use cache
        cached_call_time = time.time() - start_time

        print(f"  First async call: {result3} (took {first_call_time:.3f}s)")
        print(f"  Cached async call: {result4} (took {cached_call_time:.3f}s)")
        print(f"  Async cache speedup: {first_call_time/cached_call_time:.1f}x")

        print("✅ Async enhanced decorators test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing async enhanced decorators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decorator_registry_features():
    """Test advanced decorator registry features."""

    print("\n🏗️ Testing Decorator Registry Features")
    print("-" * 40)

    try:
        from src.utils.decorator_registry import decorator_registry

        # Test decorator discovery
        print("Testing decorator discovery...")

        # Search by tags
        performance_decorators = decorator_registry.search("performance")
        print(f"  Performance decorators: {len(performance_decorators)}")
        for decorator in performance_decorators:
            print(f"    - {decorator.name}: {decorator.description}")

        # Search by name
        validation_decorators = decorator_registry.search("validation")
        print(f"  Validation decorators: {len(validation_decorators)}")

        # Test filtering
        active_decorators = decorator_registry.list_decorators(include_deprecated=False)
        deprecated_decorators = decorator_registry.list_decorators(include_deprecated=True)

        print(f"  Active decorators: {len(active_decorators)}")
        print(f"  Total decorators (including deprecated): {len(deprecated_decorators)}")

        # Test configuration export
        config = decorator_registry.export_config()
        print(f"  Registry config keys: {list(config.keys())}")

        print("✅ Decorator registry features test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing decorator registry features: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""

    print("🚀 Enhanced Decorator System Test Suite")
    print("=" * 60)

    # Test the main system
    success1 = test_enhanced_decorator_system()

    # Test async decorators
    success2 = asyncio.run(test_async_enhanced_decorators())

    # Test registry features
    success3 = test_decorator_registry_features()

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    print(f"Enhanced decorator system: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Async decorators: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Registry features: {'✅ PASS' if success3 else '❌ FAIL'}")

    overall_success = success1 and success2 and success3
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 The enhanced decorator system is working correctly!")
        print("   - New enhanced decorators are functional")
        print("   - Backwards compatibility is maintained")
        print("   - Configuration system is working")
        print("   - Registry system is operational")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

    return overall_success

if __name__ == "__main__":
    main()