"""
Test script for the hardware-optimized caching system.

This script demonstrates the enhanced caching capabilities with:
- Hardware-aware memory management
- Predictive optimization
- Adaptive cleanup strategies
- Performance monitoring
- Real-time optimization
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import the optimized caching system
from src.utils.feature_common.caching_optimized import (
    HardwareAwareCache,
    OptimizedFeatureCache,
    OptimizedCacheConfig,
    CacheStrategy,
    CompressionType,
    get_shared_cache,
    get_feature_cache,
    cache_context,
    optimize_cache_for_operation
)
from src.utils.hardware.unified_hardware_manager import WorkloadType, OptimizationLevel
from src.utils.hardware.adaptive_optimization_engine import OptimizationTarget

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 10000, n_features: int = 50) -> pd.DataFrame:
    """Create test DataFrame for caching experiments."""
    np.random.seed(42)
    
    data = {
        f'feature_{i}': np.random.normal(0, 1, n_samples) 
        for i in range(n_features)
    }
    
    # Add some correlation
    data['feature_0'] = data['feature_1'] + np.random.normal(0, 0.1, n_samples)
    data['feature_2'] = data['feature_3'] * 2 + np.random.normal(0, 0.1, n_samples)
    
    # Add target variable
    data['target'] = (
        data['feature_0'] * 0.3 + 
        data['feature_1'] * 0.2 + 
        data['feature_2'] * 0.1 + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    return pd.DataFrame(data)

def expensive_computation(data: pd.DataFrame, operation: str = 'complex') -> pd.DataFrame:
    """Simulate expensive computation for caching."""
    time.sleep(0.1)  # Simulate computation time
    
    if operation == 'complex':
        # Complex feature engineering
        result = data.copy()
        for col in data.columns:
            if col != 'target':
                result[f'{col}_squared'] = data[col] ** 2
                result[f'{col}_log'] = np.log1p(np.abs(data[col]))
                result[f'{col}_rolling_mean'] = data[col].rolling(window=5).mean()
        
        # Add some statistical features
        result['feature_sum'] = data.select_dtypes(include=[np.number]).sum(axis=1)
        result['feature_mean'] = data.select_dtypes(include=[np.number]).mean(axis=1)
        result['feature_std'] = data.select_dtypes(include=[np.number]).std(axis=1)
        
    elif operation == 'correlation':
        # Correlation matrix computation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data[numeric_cols].corr()
        
    elif operation == 'statistical':
        # Statistical analysis
        result = {}
        for col in data.columns:
            if col != 'target':
                result[f'{col}_mean'] = data[col].mean()
                result[f'{col}_std'] = data[col].std()
                result[f'{col}_skew'] = data[col].skew()
                result[f'{col}_kurtosis'] = data[col].kurtosis()
        result = pd.DataFrame([result])
    
    return result

def test_basic_caching():
    """Test basic caching functionality."""
    print("\n" + "="*60)
    print("🧪 Testing Basic Caching Functionality")
    print("="*60)
    
    # Create test data
    data = create_test_data(1000, 10)
    
    # Initialize cache
    config = OptimizedCacheConfig(
        max_size=100,
        max_memory_mb=50,
        ttl_seconds=60,
        enable_hardware_optimization=True
    )
    cache = HardwareAwareCache(config)
    
    # Test caching
    start_time = time.time()
    
    # First computation (should be slow)
    result1 = cache.get_or_compute(expensive_computation, data, 'complex')
    first_time = time.time() - start_time
    
    # Second computation (should be fast - cached)
    start_time = time.time()
    result2 = cache.get_or_compute(expensive_computation, data, 'complex')
    second_time = time.time() - start_time
    
    print(f"✅ First computation: {first_time:.3f}s")
    print(f"✅ Second computation (cached): {second_time:.3f}s")
    print(f"✅ Speedup: {first_time/second_time:.1f}x")
    
    # Verify results are identical
    assert result1.equals(result2), "Cached result differs from original"
    print("✅ Results are identical")
    
    # Test cache stats
    stats = cache.get_stats()
    print(f"✅ Cache size: {stats['cache_size']}")
    print(f"✅ Memory usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"✅ Hit rate: {stats['hit_rate']:.2%}")
    
    cache.shutdown()

def test_memory_optimization():
    """Test memory optimization features."""
    print("\n" + "="*60)
    print("🧪 Testing Memory Optimization")
    print("="*60)
    
    # Create large test data
    data = create_test_data(5000, 20)
    
    # Initialize cache with memory optimization
    config = OptimizedCacheConfig(
        max_size=50,
        max_memory_mb=100,
        enable_compression=True,
        compression_threshold_mb=0.5,
        enable_memory_pooling=True
    )
    cache = HardwareAwareCache(config)
    
    # Test different operations
    operations = ['complex', 'correlation', 'statistical']
    
    for i, operation in enumerate(operations):
        print(f"\n--- Testing {operation} operation ---")
        
        start_time = time.time()
        result = cache.get_or_compute(expensive_computation, data, operation)
        compute_time = time.time() - start_time
        
        stats = cache.get_stats()
        print(f"✅ Computation time: {compute_time:.3f}s")
        print(f"✅ Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"✅ Cache size: {stats['cache_size']}")
        
        if 'entry_analysis' in stats:
            analysis = stats['entry_analysis']
            print(f"✅ Compressed entries: {analysis['compressed_entries']}")
            print(f"✅ Avg entry size: {analysis['avg_entry_size_mb']:.2f} MB")
    
    cache.shutdown()

def test_adaptive_cleanup():
    """Test adaptive cleanup strategies."""
    print("\n" + "="*60)
    print("🧪 Testing Adaptive Cleanup Strategies")
    print("="*60)
    
    # Create cache with aggressive cleanup
    config = OptimizedCacheConfig(
        max_size=20,
        max_memory_mb=10,
        aggressive_cleanup_threshold=0.5,
        predictive_cleanup_threshold=0.3,
        enable_adaptive_cleanup=True
    )
    cache = HardwareAwareCache(config)
    
    # Fill cache beyond capacity
    data = create_test_data(1000, 5)
    
    print("Filling cache beyond capacity...")
    for i in range(30):
        test_data = data.sample(100)  # Different subset each time
        result = cache.get_or_compute(expensive_computation, test_data, 'complex')
        
        if i % 5 == 0:
            stats = cache.get_stats()
            print(f"  Iteration {i}: Cache size={stats['cache_size']}, Memory={stats['memory_usage_mb']:.2f}MB")
    
    # Final stats
    final_stats = cache.get_stats()
    print(f"✅ Final cache size: {final_stats['cache_size']}")
    print(f"✅ Final memory usage: {final_stats['memory_usage_mb']:.2f} MB")
    print(f"✅ Memory pressure: {final_stats['memory_pressure']:.2%}")
    
    cache.shutdown()

def test_workload_optimization():
    """Test workload-specific optimization."""
    print("\n" + "="*60)
    print("🧪 Testing Workload-Specific Optimization")
    print("="*60)
    
    data = create_test_data(2000, 15)
    
    # Test different workload types
    workloads = [
        (WorkloadType.FEATURE_ENGINEERING, OptimizationTarget.BALANCED),
        (WorkloadType.ML_TRAINING, OptimizationTarget.PERFORMANCE),
        (WorkloadType.DATA_PROCESSING, OptimizationTarget.EFFICIENCY)
    ]
    
    for workload_type, target in workloads:
        print(f"\n--- Testing {workload_type.value} ({target.value}) ---")
        
        with cache_context(workload_type, target) as cache:
            start_time = time.time()
            result = cache.get_or_compute(expensive_computation, data, 'complex')
            compute_time = time.time() - start_time
            
            stats = cache.get_stats()
            print(f"✅ Computation time: {compute_time:.3f}s")
            print(f"✅ Memory usage: {stats['memory_usage_mb']:.2f} MB")
            print(f"✅ Strategy: {stats['config']['strategy']}")

def test_feature_cache():
    """Test optimized feature cache."""
    print("\n" + "="*60)
    print("🧪 Testing Optimized Feature Cache")
    print("="*60)
    
    # Create test data
    data = create_test_data(1000, 10)
    feature_cache = get_feature_cache(max_size=100, enable_hardware_optimization=True)
    
    # Test rolling statistics
    print("Testing rolling statistics...")
    series = data['feature_0']
    
    for window in [5, 10, 20]:
        for stat_type in ['mean', 'std', 'var']:
            start_time = time.time()
            result = feature_cache.get_rolling_stat(series, window, stat_type)
            compute_time = time.time() - start_time
            
            print(f"  Rolling {stat_type} (window={window}): {compute_time:.3f}s")
    
    # Test correlation matrix
    print("\nTesting correlation matrix...")
    start_time = time.time()
    corr_matrix = feature_cache.get_correlation_matrix(data)
    compute_time = time.time() - start_time
    print(f"  Correlation matrix: {compute_time:.3f}s")
    
    # Test statistical tests
    print("\nTesting statistical tests...")
    for test_type in ['ttest', 'correlation']:
        start_time = time.time()
        result = feature_cache.get_statistical_test(data['feature_0'], data['target'], test_type)
        compute_time = time.time() - start_time
        print(f"  {test_type}: {compute_time:.3f}s, result={result:.4f}")
    
    # Get cache stats
    stats = feature_cache.get_stats()
    print(f"\n✅ Feature cache stats: {stats}")

def test_performance_monitoring():
    """Test performance monitoring and learning."""
    print("\n" + "="*60)
    print("🧪 Testing Performance Monitoring")
    print("="*60)
    
    config = OptimizedCacheConfig(
        max_size=50,
        max_memory_mb=100,
        enable_hardware_optimization=True
    )
    cache = HardwareAwareCache(config)
    
    # Perform various operations
    data = create_test_data(1000, 10)
    
    print("Performing operations for monitoring...")
    for i in range(20):
        # Vary the data slightly
        test_data = data.sample(800)
        operation = ['complex', 'correlation', 'statistical'][i % 3]
        
        result = cache.get_or_compute(expensive_computation, test_data, operation)
        
        if i % 5 == 0:
            stats = cache.get_stats()
            print(f"  Iteration {i}: Hit rate={stats['hit_rate']:.2%}, Memory={stats['memory_usage_mb']:.2f}MB")
    
    # Final performance analysis
    final_stats = cache.get_stats()
    print(f"\n✅ Final performance metrics:")
    print(f"  Hit rate: {final_stats['hit_rate']:.2%}")
    print(f"  Memory usage: {final_stats['memory_usage_mb']:.2f} MB")
    print(f"  Cache size: {final_stats['cache_size']}")
    
    if 'recent_performance' in final_stats:
        recent = final_stats['recent_performance']
        print(f"  Recent performance samples: {len(recent)}")
    
    cache.shutdown()

def test_compression():
    """Test compression functionality."""
    print("\n" + "="*60)
    print("🧪 Testing Compression")
    print("="*60)
    
    # Test different compression types
    compression_types = [CompressionType.NONE, CompressionType.LZ4, CompressionType.ZLIB]
    
    for comp_type in compression_types:
        print(f"\n--- Testing {comp_type.value} compression ---")
        
        config = OptimizedCacheConfig(
            max_size=20,
            max_memory_mb=50,
            compression_type=comp_type,
            enable_compression=True,
            compression_threshold_mb=0.1
        )
        cache = HardwareAwareCache(config)
        
        # Create data that will trigger compression
        data = create_test_data(2000, 15)
        
        start_time = time.time()
        result = cache.get_or_compute(expensive_computation, data, 'complex')
        compute_time = time.time() - start_time
        
        stats = cache.get_stats()
        print(f"✅ Computation time: {compute_time:.3f}s")
        print(f"✅ Memory usage: {stats['memory_usage_mb']:.2f} MB")
        
        if 'entry_analysis' in stats:
            analysis = stats['entry_analysis']
            print(f"✅ Compressed entries: {analysis['compressed_entries']}")
            print(f"✅ Total entries: {analysis['total_entries']}")
        
        cache.shutdown()

def benchmark_comparison():
    """Benchmark optimized cache vs basic cache."""
    print("\n" + "="*60)
    print("🧪 Benchmarking: Optimized vs Basic Cache")
    print("="*60)
    
    data = create_test_data(2000, 20)
    operations = ['complex', 'correlation', 'statistical']
    
    # Test optimized cache
    print("Testing optimized cache...")
    config = OptimizedCacheConfig(
        max_size=100,
        max_memory_mb=200,
        enable_hardware_optimization=True,
        enable_compression=True,
        enable_memory_pooling=True
    )
    optimized_cache = HardwareAwareCache(config)
    
    optimized_times = []
    for i in range(10):
        operation = operations[i % len(operations)]
        start_time = time.time()
        result = optimized_cache.get_or_compute(expensive_computation, data, operation)
        compute_time = time.time() - start_time
        optimized_times.append(compute_time)
    
    optimized_stats = optimized_cache.get_stats()
    optimized_cache.shutdown()
    
    # Test basic cache (simulated)
    print("Testing basic cache...")
    basic_times = []
    for i in range(10):
        operation = operations[i % len(operations)]
        start_time = time.time()
        result = expensive_computation(data, operation)
        compute_time = time.time() - start_time
        basic_times.append(compute_time)
    
    # Compare results
    print(f"\n✅ Performance Comparison:")
    print(f"  Optimized cache avg time: {np.mean(optimized_times):.3f}s")
    print(f"  Basic computation avg time: {np.mean(basic_times):.3f}s")
    print(f"  Speedup: {np.mean(basic_times)/np.mean(optimized_times):.1f}x")
    print(f"  Optimized hit rate: {optimized_stats['hit_rate']:.2%}")

def main():
    """Run all tests."""
    print("🚀 Starting Hardware-Optimized Caching Tests")
    print("="*60)
    
    try:
        # Run all tests
        test_basic_caching()
        test_memory_optimization()
        test_adaptive_cleanup()
        test_workload_optimization()
        test_feature_cache()
        test_performance_monitoring()
        test_compression()
        benchmark_comparison()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()