"""
Simple test script for the hardware-optimized caching system.

This script demonstrates the enhanced caching capabilities without relying on
the __init__.py file that has missing dependencies.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the optimized caching system directly
from utils.feature_common.caching_optimized import (
    HardwareAwareCache,
    OptimizedFeatureCache,
    OptimizedCacheConfig,
    CacheStrategy,
    CompressionType,
    get_shared_cache,
    get_feature_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
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
    time.sleep(0.05)  # Simulate computation time
    
    if operation == 'complex':
        # Complex feature engineering
        result = data.copy()
        for col in data.columns:
            if col != 'target':
                result[f'{col}_squared'] = data[col] ** 2
                result[f'{col}_log'] = np.log1p(np.abs(data[col]))
        
        # Add some statistical features
        result['feature_sum'] = data.select_dtypes(include=[np.number]).sum(axis=1)
        result['feature_mean'] = data.select_dtypes(include=[np.number]).mean(axis=1)
        
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
        result = pd.DataFrame([result])
    
    return result

def test_basic_caching():
    """Test basic caching functionality."""
    print("\n" + "="*60)
    print("🧪 Testing Basic Caching Functionality")
    print("="*60)
    
    # Create test data
    data = create_test_data(500, 5)
    
    # Initialize cache
    config = OptimizedCacheConfig(
        max_size=50,
        max_memory_mb=25,
        ttl_seconds=60,
        enable_hardware_optimization=False  # Disable for this test
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
    
    # Create test data
    data = create_test_data(1000, 8)
    
    # Initialize cache with memory optimization
    config = OptimizedCacheConfig(
        max_size=30,
        max_memory_mb=50,
        enable_compression=True,
        compression_threshold_mb=0.1,
        enable_memory_pooling=False  # Disable for this test
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
            max_memory_mb=30,
            compression_type=comp_type,
            enable_compression=True,
            compression_threshold_mb=0.05
        )
        cache = HardwareAwareCache(config)
        
        # Create data that will trigger compression
        data = create_test_data(800, 6)
        
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

def test_feature_cache():
    """Test optimized feature cache."""
    print("\n" + "="*60)
    print("🧪 Testing Optimized Feature Cache")
    print("="*60)
    
    # Create test data
    data = create_test_data(500, 5)
    feature_cache = get_feature_cache(max_size=50, enable_hardware_optimization=False)
    
    # Test rolling statistics
    print("Testing rolling statistics...")
    series = data['feature_0']
    
    for window in [5, 10]:
        for stat_type in ['mean', 'std']:
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

def benchmark_comparison():
    """Benchmark optimized cache vs basic computation."""
    print("\n" + "="*60)
    print("🧪 Benchmarking: Optimized vs Basic Computation")
    print("="*60)
    
    data = create_test_data(800, 8)
    operations = ['complex', 'correlation', 'statistical']
    
    # Test optimized cache
    print("Testing optimized cache...")
    config = OptimizedCacheConfig(
        max_size=50,
        max_memory_mb=100,
        enable_hardware_optimization=False,
        enable_compression=True,
        enable_memory_pooling=False
    )
    optimized_cache = HardwareAwareCache(config)
    
    optimized_times = []
    for i in range(8):
        operation = operations[i % len(operations)]
        start_time = time.time()
        result = optimized_cache.get_or_compute(expensive_computation, data, operation)
        compute_time = time.time() - start_time
        optimized_times.append(compute_time)
    
    optimized_stats = optimized_cache.get_stats()
    optimized_cache.shutdown()
    
    # Test basic computation (simulated)
    print("Testing basic computation...")
    basic_times = []
    for i in range(8):
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
        test_compression()
        test_feature_cache()
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
