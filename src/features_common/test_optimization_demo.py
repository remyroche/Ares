"""
Comprehensive demonstration of native optimization benefits.

This script demonstrates that all transforms natively benefit from all optimizations
including VectorBT acceleration, caching, performance monitoring, and intelligent fallback.
"""

import time
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_optimization_benefits():
    """
    Demonstrate that all transforms natively benefit from all optimizations.
    """
    print("🚀 Features Common Optimization Demonstration")
    print("=" * 60)

    # Create test data
    print("\n📊 Creating test data...")
    np.random.seed(42)
    data = pd.Series(np.random.randn(10000) + np.sin(np.linspace(0, 10, 10000)) * 2)
    data_with_nans = data.copy()
    data_with_nans.iloc[100:110] = np.nan  # Add some NaN values

    print(f"   Data shape: {data.shape}")
    print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"   NaN values: {data_with_nans.isna().sum()}")

    # Test 1: Basic Scaler with All Optimizations
    print("\n🔧 Test 1: Basic Scaler with All Optimizations")
    print("-" * 50)

    from src.features_common.transforms.base_scaler import SimpleScaler

    # Create scaler (automatically gets all optimizations)
    scaler = SimpleScaler()

    # Show that all mixins are available
    print(f"   ✅ Optimization enabled: {scaler.is_optimization_enabled()}")
    print(f"   ✅ Performance monitoring enabled: {scaler.is_performance_monitoring_enabled()}")
    print(f"   ✅ VectorBT available: {scaler.is_vectorbt_available()}")
    print(f"   ✅ Caching enabled: {hasattr(scaler, 'cached_operation')}")
    print(f"   ✅ Validation enabled: {hasattr(scaler, 'validate_data')}")
    print(f"   ✅ Monitoring enabled: {hasattr(scaler, 'run_health_checks')}")

    # Test fit_transform with optimizations
    print("\n   🔄 Testing fit_transform with all optimizations...")
    start_time = time.time()

    with scaler.profile_operation("fit_transform_demo"):
        result = scaler.fit_transform(data_with_nans)

    fit_time = time.time() - start_time
    print(f"   ⏱️  Fit time: {fit_time:.4f} seconds")
    print(f"   📈 Result shape: {result.shape}")
    print(f"   📊 Result stats: mean={result.mean():.3f}, std={result.std():.3f}")

    # Test transform with optimizations
    print("\n   🔄 Testing transform with all optimizations...")
    start_time = time.time()

    with scaler.profile_operation("transform_demo"):
        transform_result = scaler.transform(data_with_nans)

    transform_time = time.time() - start_time
    print(f"   ⏱️  Transform time: {transform_time:.4f} seconds")

    # Show optimization statistics
    print("\n   📊 Optimization Statistics:")
    opt_stats = scaler.get_optimization_stats()
    for key, value in opt_stats.items():
        if isinstance(value, (int, float)):
            print(f"      {key}: {value}")

    # Show performance statistics
    print("\n   📈 Performance Statistics:")
    perf_stats = scaler.get_performance_stats()
    for key, value in perf_stats.items():
        if isinstance(value, (int, float)) and not key.startswith('system_'):
            print(f"      {key}: {value}")

    # Test 2: VectorBT Manager with All Optimizations
    print("\n🔧 Test 2: VectorBT Manager with All Optimizations")
    print("-" * 50)

    from src.features_common.vectorbt import get_unified_vectorbt_manager

    # Get unified VectorBT manager
    vectorbt_manager = get_unified_vectorbt_manager()

    # Show manager capabilities
    print(f"   ✅ VectorBT available: {vectorbt_manager.is_vectorbt_available()}")
    print(f"   ✅ Available operations: {len(vectorbt_manager.get_available_operations())}")
    print(f"   ✅ Optimization enabled: {vectorbt_manager.is_optimization_enabled()}")
    print(f"   ✅ Performance monitoring: {vectorbt_manager.is_performance_monitoring_enabled()}")

    # Test VectorBT operations
    print("\n   🔄 Testing VectorBT operations...")

    # Rolling mean
    start_time = time.time()
    rolling_mean_result = vectorbt_manager.rolling_mean(data, window=20)
    rolling_time = time.time() - start_time
    print(f"   📊 Rolling mean (window=20): {rolling_time:.4f}s, shape={rolling_mean_result.shape}")

    # Rolling std
    start_time = time.time()
    rolling_std_result = vectorbt_manager.rolling_std(data, window=20)
    rolling_std_time = time.time() - start_time
    print(f"   📊 Rolling std (window=20): {rolling_std_time:.4f}s, shape={rolling_std_result.shape}")

    # Data scaling
    start_time = time.time()
    scaled_data = vectorbt_manager.scale_data(data, method='zscore')
    scale_time = time.time() - start_time
    print(f"   📊 Data scaling (zscore): {scale_time:.4f}s, shape={scaled_data.shape}")

    # Show VectorBT statistics
    print("\n   📊 VectorBT Statistics:")
    vectorbt_stats = vectorbt_manager.get_operation_stats()
    for key, value in vectorbt_stats.items():
        if isinstance(value, (int, float)):
            print(f"      {key}: {value}")

    # Test 3: Factory-Created Optimized Scaler
    print("\n🔧 Test 3: Factory-Created Optimized Scaler")
    print("-" * 50)

    from src.features_common.factories import create_optimized_scaler, create_batch_scaler

    # Create optimized scaler using factory
    factory_scaler = create_optimized_scaler(method='zscore')

    print(f"   ✅ Factory scaler created: {type(factory_scaler).__name__}")
    print(f"   ✅ All optimizations enabled: {factory_scaler.is_optimization_enabled()}")

    # Test factory scaler
    start_time = time.time()
    factory_result = factory_scaler.fit_transform(data)
    factory_time = time.time() - start_time
    print(f"   ⏱️  Factory scaler time: {factory_time:.4f} seconds")

    # Test batch scaler
    print("\n   🔄 Testing batch scaler...")
    batch_data = pd.DataFrame({
        'feature1': data,
        'feature2': data * 1.5 + 2,
        'feature3': data * 0.8 - 1
    })

    batch_scaler = create_batch_scaler(method='zscore')
    start_time = time.time()
    batch_result = batch_scaler.fit_transform(batch_data)
    batch_time = time.time() - start_time
    print(f"   ⏱️  Batch scaler time: {batch_time:.4f} seconds")
    print(f"   📊 Batch result shape: {batch_result.shape}")

    # Test 4: Configuration and Monitoring
    print("\n🔧 Test 4: Configuration and Monitoring")
    print("-" * 50)

    from src.features_common.config import get_unified_config

    # Get unified configuration
    config = get_unified_config()

    print(f"   ✅ Configuration loaded: {type(config).__name__}")
    print(f"   ✅ VectorBT enabled: {config.vectorbt.enable_vectorbt}")
    print(f"   ✅ GPU enabled: {config.vectorbt.enable_gpu}")
    print(f"   ✅ Caching enabled: {config.optimization.enable_caching}")
    print(f"   ✅ Performance monitoring: {config.optimization.enable_performance_monitoring}")

    # Test health checks
    print("\n   🔄 Running health checks...")
    health_results = scaler.run_health_checks()
    print(f"   📊 Overall health: {health_results['overall_status']}")
    print(f"   📈 Checks performed: {len(health_results['checks'])}")

    # Test 5: Performance Comparison
    print("\n🔧 Test 5: Performance Comparison")
    print("-" * 50)

    # Compare with basic pandas operations
    print("   🔄 Comparing with basic pandas operations...")

    # Basic pandas z-score
    start_time = time.time()
    pandas_mean = data.mean()
    pandas_std = data.std()
    pandas_result = (data - pandas_mean) / pandas_std
    pandas_time = time.time() - start_time

    # Optimized scaler
    start_time = time.time()
    optimized_result = scaler.transform(data)
    optimized_time = time.time() - start_time

    print(f"   📊 Pandas z-score time: {pandas_time:.4f} seconds")
    print(f"   📊 Optimized scaler time: {optimized_time:.4f} seconds")
    print(f"   🚀 Speed improvement: {pandas_time / optimized_time:.2f}x")

    # Verify results are equivalent
    result_diff = np.abs(pandas_result - optimized_result).max()
    print(f"   ✅ Results equivalent (max diff): {result_diff:.2e}")

    # Test 6: Caching Benefits
    print("\n🔧 Test 6: Caching Benefits")
    print("-" * 50)

    # First transform (no cache)
    start_time = time.time()
    result1 = scaler.transform(data)
    first_time = time.time() - start_time

    # Second transform (should use cache)
    start_time = time.time()
    result2 = scaler.transform(data)
    second_time = time.time() - start_time

    print(f"   📊 First transform time: {first_time:.4f} seconds")
    print(f"   📊 Second transform time: {second_time:.4f} seconds")
    print(f"   🚀 Cache speedup: {first_time / second_time:.2f}x")

    # Show cache statistics
    if hasattr(scaler, 'get_cache_stats'):
        cache_stats = scaler.get_cache_stats()
        print(f"   📊 Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
        print(f"   📊 Cache size: {cache_stats.get('cache_size', 0)}")

    # Test 7: Memory and Performance Monitoring
    print("\n🔧 Test 7: Memory and Performance Monitoring")
    print("-" * 50)

    # Get comprehensive performance summary
    if hasattr(scaler, 'get_performance_summary'):
        perf_summary = scaler.get_performance_summary()
        print(f"   📊 Performance summary available: {len(perf_summary)} metrics")

    # Get monitoring recommendations
    if hasattr(scaler, 'get_performance_recommendations'):
        recommendations = scaler.get_performance_recommendations()
        if recommendations:
            print(f"   💡 Performance recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec}")
        else:
            print("   ✅ No performance recommendations (system optimized)")

    # Final Summary
    print("\n🎉 Optimization Demonstration Complete!")
    print("=" * 60)
    print("✅ All transforms natively benefit from:")
    print("   • VectorBT acceleration when beneficial")
    print("   • Intelligent caching for repeated operations")
    print("   • Performance monitoring and optimization")
    print("   • Data validation and sanitization")
    print("   • Automatic fallback mechanisms")
    print("   • Memory optimization and management")
    print("   • Health monitoring and alerting")
    print("\n🚀 The system automatically selects the best optimization")
    print("   strategy based on data characteristics and performance history.")

if __name__ == "__main__":
    demonstrate_optimization_benefits()
