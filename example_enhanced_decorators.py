#!/usr/bin/env python3
"""
Example script demonstrating the enhanced decorator system
"""

import pandas as pd
import numpy as np
import time
import asyncio

def example_basic_usage():
    """Demonstrate basic usage of enhanced decorators."""

    print("🚀 Basic Enhanced Decorator Usage")
    print("=" * 50)

    # Import enhanced decorators
    from src.utils.enhanced_decorators import (
        smart_error_recovery,
        cached_validation,
        performance_monitor_v2
    )

    # Smart error recovery example
    @smart_error_recovery(max_retries=2, fallback_strategy="default_return")
    def risky_function(x):
        if x < 0:
            raise ValueError("Negative numbers not allowed")
        return x * 2

    print("Testing smart error recovery:")
    print(f"  Normal case: {risky_function(5)}")
    print(f"  Error case: {risky_function(-3)}")

    # Cached validation example
    @cached_validation(cache_size=10, ttl_seconds=5)
    def expensive_calculation(x):
        time.sleep(0.1)  # Simulate expensive operation
        return x ** 2

    print("\nTesting cached validation:")
    start_time = time.time()
    result1 = expensive_calculation(5)
    first_call_time = time.time() - start_time

    start_time = time.time()
    result2 = expensive_calculation(5)  # Should use cache
    cached_call_time = time.time() - start_time

    print(f"  First call: {result1} (took {first_call_time:.3f}s)")
    print(f"  Cached call: {result2} (took {cached_call_time:.3f}s)")
    print(f"  Speedup: {first_call_time/cached_call_time:.1f}x")

    # Performance monitoring example
    @performance_monitor_v2(level="detailed", track_memory=True, track_cpu=True)
    def sample_function():
        time.sleep(0.1)
        return "Hello, World!"

    print("\nTesting performance monitoring:")
    result = sample_function()
    print(f"  Function result: {result}")

def example_advanced_validation():
    """Demonstrate advanced validation decorators."""

    print("\n🔍 Advanced Validation Decorators")
    print("=" * 50)

    from src.utils.centralized_decorators_v2 import (
        validate_data_quality_v2,
        quality_gate_v2,
        step_specific_ml_validation_v2
    )

    # Data quality validation v2
    @validate_data_quality_v2(
        validation_level="WARNING",
        auto_fix=True,
        context="data processing"
    )
    def process_dataframe(df):
        return df * 2

    # Create test data with some issues
    test_df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    })

    print("Testing data quality validation:")
    result = process_dataframe(test_df)
    print(f"  Processed data shape: {result.shape}")
    print(f"  Has NaN values: {result.isna().any().any()}")

    # Quality gate v2
    @quality_gate_v2(
        min_quality_score=0.6,
        required_grade="C",
        action_on_failure="warn"
    )
    def quality_assessment():
        return "high_quality_result"

    print("\nTesting quality gate:")
    result = quality_assessment()
    print(f"  Quality assessment result: {result}")

    # Step-specific ML validation v2
    @step_specific_ml_validation_v2(
        step_name="feature_engineering",
        adaptive_thresholds=True
    )
    def feature_engineering_step(data):
        return data + 1

    print("\nTesting ML step validation:")
    test_data = np.array([1, 2, 3, 4, 5])
    result = feature_engineering_step(test_data)
    print(f"  Feature engineering result: {result}")

def example_configuration_management():
    """Demonstrate configuration management."""

    print("\n⚙️ Configuration Management")
    print("=" * 50)

    from src.utils.decorator_compatibility import (
        get_decorator_config,
        set_decorator_config
    )

    # Show current configuration
    print("Current configuration:")
    print(f"  Validation mode: {global_config.validation_mode}")
    print(f"  Performance mode: {global_config.performance_mode}")
    print(f"  Cache enabled: {global_config.cache_enabled}")
    print(f"  Max retries: {global_config.max_retries}")

    # Modify configuration
    print("\nModifying configuration...")
    set_decorator_config(
        validation_mode="STRICT",
        max_retries=5,
        cache_enabled=True
    )

    # Show updated configuration
    updated_config = get_decorator_config()
    print("Updated configuration:")
    print(f"  Validation mode: {updated_config.validation_mode}")
    print(f"  Max retries: {updated_config.max_retries}")
    print(f"  Cache enabled: {updated_config.cache_enabled}")

def example_registry_discovery():
    """Demonstrate decorator registry features."""

    print("\n📚 Decorator Registry Discovery")
    print("=" * 50)

    from src.utils.decorator_compatibility import (
        list_available_decorators,
        get_decorator_usage_stats,
        search_decorators
    )

    # List available decorators
    decorators = list_available_decorators()
    print(f"Available decorators: {len(decorators)}")

    # Show some decorators
    print("\nSample decorators:")
    for decorator in decorators[:5]:
        print(f"  - {decorator.name} v{decorator.version}")
        print(f"    Tags: {', '.join(decorator.tags)}")
        print(f"    Description: {decorator.description}")
        print()

    # Search by functionality
    print("Searching for validation decorators:")
    validation_decorators = search_decorators("validation")
    for decorator in validation_decorators:
        print(f"  - {decorator.name}")

    # Get usage statistics
    print("\nUsage statistics:")
    usage_stats = get_decorator_usage_stats()
    for decorator_name, usage_count in usage_stats.items():
        print(f"  {decorator_name}: {usage_count} uses")

def example_backwards_compatibility():
    """Demonstrate backwards compatibility."""

    print("\n🔄 Backwards Compatibility")
    print("=" * 50)

    from src.utils.decorator_compatibility import (
        validate_call,      # Legacy name
        check_input,        # Legacy name
        smart_recovery,     # Alias
        cached             # Alias
    )

    # Legacy decorators (with deprecation warnings)
    @validate_call()
    @check_input(None)
    def legacy_check_function(df):
        return df

    # Enhanced decorator aliases
    @smart_recovery(max_retries=1)
    @cached(cache_size=5)
    # Test legacy functions
    print("Testing legacy decorator names:")
    test_df = pd.DataFrame({'A': [1, 2, 3]})

    result1 = legacy_function(10)
    result2 = legacy_check_function(test_df)
    result3 = legacy_recovery_function(5)
    result4 = legacy_cached_function(6)

    print(f"  Legacy validate_call: {result1}")
    print(f"  Legacy check_input: {result2.shape}")
    print(f"  Legacy recovery: {result3}")
    print(f"  Legacy cached: {result4}")

async def example_async_decorators():
    """Demonstrate async enhanced decorators."""

    print("\n🔄 Async Enhanced Decorators")
    print("=" * 50)

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

    print("Testing async smart error recovery:")
    result1 = await async_function_with_errors(5)
    print(f"  Normal async execution: {result1}")

    result2 = await async_function_with_errors(-3)
    print(f"  Async error recovery: {result2}")

    print("\nTesting async cached validation:")
    start_time = time.time()
    result3 = await async_expensive_calculation(5)
    first_call_time = time.time() - start_time

    start_time = time.time()
    result4 = await async_expensive_calculation(5)  # Should use cache
    cached_call_time = time.time() - start_time

    print(f"  First async call: {result3} (took {first_call_time:.3f}s)")
    print(f"  Cached async call: {result4} (took {cached_call_time:.3f}s)")
    print(f"  Async cache speedup: {first_call_time/cached_call_time:.1f}x")

def main():
    """Run all examples."""

    print("🎯 Enhanced Decorator System Examples")
    print("=" * 60)

    try:
        # Run examples
        example_basic_usage()
        example_advanced_validation()
        example_configuration_management()
        example_registry_discovery()
        example_backwards_compatibility()

        # Run async examples
        asyncio.run(example_async_decorators())

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nThe enhanced decorator system provides:")
        print("  - Better performance with intelligent caching")
        print("  - Enhanced error handling with automatic recovery")
        print("  - Unified configuration management")
        print("  - Complete backwards compatibility")
        print("  - Centralized decorator discovery")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()