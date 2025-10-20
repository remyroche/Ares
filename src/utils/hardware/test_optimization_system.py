"""
Test Suite for Enhanced Caching and Optimization System.

This module provides comprehensive tests for the enhanced caching and optimization
system to ensure it works correctly throughout the codebase.
"""

import unittest
import pandas as pd
import numpy as np
import time
from typing import Dict, Any

from .enhanced_caching_system import (
    EnhancedCacheSystem, CacheConfig, DataTypeOptimization, CacheStrategy,
    get_global_cache, optimize_dataframe_default, optimize_numpy_array_default
)
from .optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from .integrated_hardware_manager import (
    get_integrated_hardware_manager, process_market_data
)

class TestEnhancedCachingSystem(unittest.TestCase):
    """Test cases for enhanced caching system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_config = CacheConfig(
            max_memory_mb=100.0,
            max_items=1000,
            strategy=CacheStrategy.LRU,
            data_type_optimization=DataTypeOptimization.AGGRESSIVE,
            enable_compression=True,
            auto_optimize_dtypes=True
        )
        self.cache = EnhancedCacheSystem(self.cache_config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()
        self.cache.shutdown()
    
    def test_basic_caching(self):
        """Test basic caching functionality."""
        # Test put and get
        test_data = {'key1': 'value1', 'key2': 42}
        self.assertTrue(self.cache.put('test_key', test_data))
        
        retrieved_data = self.cache.get('test_key')
        self.assertEqual(retrieved_data, test_data)
        
        # Test cache miss
        self.assertIsNone(self.cache.get('nonexistent_key'))
    
    def test_dataframe_optimization(self):
        """Test DataFrame optimization."""
        # Create DataFrame with inefficient types
        df = pd.DataFrame({
            'id': np.arange(1000, dtype=np.int64),
            'price': np.random.uniform(0, 100, 1000).astype(np.float64),
            'category': ['A', 'B', 'C'] * 333 + ['A']
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize DataFrame
        optimized_df = optimize_dataframe_default(df)
        
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Check that memory usage decreased
        self.assertLess(optimized_memory, original_memory)
        
        # Check data types
        self.assertEqual(optimized_df['id'].dtype, np.int32)
        self.assertEqual(optimized_df['price'].dtype, np.float32)
        self.assertEqual(optimized_df['category'].dtype.name, 'category')
    
    def test_numpy_array_optimization(self):
        """Test NumPy array optimization."""
        # Create array with inefficient type
        arr = np.random.rand(1000, 100).astype(np.float64)
        original_size = arr.nbytes
        
        # Optimize array
        optimized_arr = optimize_numpy_array_default(arr)
        optimized_size = optimized_arr.nbytes
        
        # Check that size decreased
        self.assertLess(optimized_size, original_size)
        self.assertEqual(optimized_arr.dtype, np.float32)
    
    def test_compression(self):
        """Test data compression."""
        # Create large data
        large_data = np.random.rand(10000, 1000)
        
        # Store in cache (should be compressed)
        self.assertTrue(self.cache.put('large_data', large_data))
        
        # Retrieve from cache
        retrieved_data = self.cache.get('large_data')
        
        # Check that data is the same
        np.testing.assert_array_equal(retrieved_data, large_data)
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache beyond capacity
        for i in range(1500):  # More than max_items (1000)
            self.cache.put(f'key_{i}', f'value_{i}')
        
        # Check that some items were evicted
        self.assertLessEqual(len(self.cache._cache), 1000)
        
        # Check that most recent items are still there
        self.assertIsNotNone(self.cache.get('key_1499'))
        self.assertIsNotNone(self.cache.get('key_1498'))
    
    def test_statistics(self):
        """Test cache statistics."""
        # Perform some operations
        self.cache.put('key1', 'value1')
        self.cache.put('key2', 'value2')
        self.cache.get('key1')
        self.cache.get('nonexistent')
        
        stats = self.cache.get_statistics()
        
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)

class TestOptimizationDecorators(unittest.TestCase):
    """Test cases for optimization decorators."""
    
    def test_smart_cache_decorator(self):
        """Test smart cache decorator."""
        call_count = 0
        
        @smart_cache(ttl=1.0)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Should not increment
        
        # Different argument (should compute)
        result3 = expensive_function(10)
        self.assertEqual(result3, 20)
        self.assertEqual(call_count, 2)
    
    def test_auto_optimize_decorator(self):
        """Test auto optimize decorator."""
        @auto_optimize(optimize_inputs=True, optimize_outputs=True)
        def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            return df.copy()
        
        # Create DataFrame with inefficient types
        df = pd.DataFrame({
            'id': np.arange(100, dtype=np.int64),
            'value': np.random.rand(100).astype(np.float64)
        })
        
        original_dtypes = df.dtypes.to_dict()
        
        # Process with optimization
        result = process_dataframe(df)
        
        # Check that types were optimized
        self.assertEqual(result['id'].dtype, np.int32)
        self.assertEqual(result['value'].dtype, np.float32)
    
    def test_memory_efficient_decorator(self):
        """Test memory efficient decorator."""
        @memory_efficient(memory_threshold_mb=1.0)
        def memory_intensive_function(size: int) -> np.ndarray:
            return np.random.rand(size, size)
        
        # This should work without issues
        result = memory_intensive_function(100)
        self.assertEqual(result.shape, (100, 100))
    
    def test_performance_tracked_decorator(self):
        """Test performance tracked decorator."""
        @performance_tracked(log_performance=True)
        def tracked_function(duration: float) -> str:
            time.sleep(duration)
            return "completed"
        
        # Execute function
        result = tracked_function(0.01)
        self.assertEqual(result, "completed")

class TestIntegratedHardwareManager(unittest.TestCase):
    """Test cases for integrated hardware manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = get_integrated_hardware_manager()
    
    def test_process_market_data(self):
        """Test market data processing."""
        market_data = {
            'prices': pd.DataFrame({
                'symbol': ['BTCUSDT', 'ETHUSDT'],
                'price': [50000.0, 3000.0],
                'volume': [1000000, 2000000]
            }),
            'features': np.random.rand(100, 10)
        }
        
        # Process data
        result = process_market_data(market_data)
        
        # Check that data was processed
        self.assertIn('prices', result)
        self.assertIn('features', result)
        
        # Check that DataFrames were optimized
        if isinstance(result['prices'], pd.DataFrame):
            self.assertEqual(result['prices']['price'].dtype, np.float32)
    
    def test_optimization_report(self):
        """Test optimization report generation."""
        report = self.manager.get_optimization_report()
        
        # Check that report contains expected keys
        self.assertIn('hardware_status', report)
        self.assertIn('cache_statistics', report)
        self.assertIn('performance_metrics', report)
    
    def test_memory_report(self):
        """Test memory report generation."""
        report = self.manager.get_memory_report()
        
        # Check that report contains expected keys
        self.assertIn('hardware_memory', report)
        self.assertIn('cache_memory', report)
        self.assertIn('performance_metrics', report)

class TestOptimizationPatches(unittest.TestCase):
    """Test cases for optimization patches."""
    
    def test_dataframe_optimization_patch(self):
        """Test DataFrame optimization patch."""
        from src.utils.common_operations import optimize_dataframe_memory
        
        # Create DataFrame with inefficient types
        df = pd.DataFrame({
            'id': np.arange(1000, dtype=np.int64),
            'price': np.random.rand(1000).astype(np.float64)
        })
        
        # Optimize using patched function
        optimized_df = optimize_dataframe_memory(df)
        
        # Check that optimization was applied
        self.assertEqual(optimized_df['id'].dtype, np.int32)
        self.assertEqual(optimized_df['price'].dtype, np.float32)

def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("Running Performance Benchmarks...")
    
    # Benchmark DataFrame optimization
    print("\n=== DataFrame Optimization Benchmark ===")
    
    # Create large DataFrame
    large_df = pd.DataFrame({
        'id': np.arange(100000, dtype=np.int64),
        'price': np.random.uniform(0, 1000, 100000).astype(np.float64),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100000)
    })
    
    original_memory = large_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Original memory usage: {original_memory:.2f} MB")
    
    # Time optimization
    start_time = time.time()
    optimized_df = optimize_dataframe_default(large_df)
    optimization_time = time.time() - start_time
    
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_saved = original_memory - optimized_memory
    
    print(f"Optimized memory usage: {optimized_memory:.2f} MB")
    print(f"Memory saved: {memory_saved:.2f} MB ({memory_saved/original_memory*100:.1f}%)")
    print(f"Optimization time: {optimization_time:.3f} seconds")
    
    # Benchmark caching
    print("\n=== Caching Benchmark ===")
    
    @smart_cache(ttl=60.0)
    def expensive_computation(data: np.ndarray) -> np.ndarray:
        time.sleep(0.1)  # Simulate expensive computation
        return np.dot(data.T, data)
    
    test_data = np.random.rand(1000, 100)
    
    # First call (computation)
    start_time = time.time()
    result1 = expensive_computation(test_data)
    first_call_time = time.time() - start_time
    
    # Second call (cached)
    start_time = time.time()
    result2 = expensive_computation(test_data)
    second_call_time = time.time() - start_time
    
    speedup = first_call_time / second_call_time
    
    print(f"First call (computation): {first_call_time:.3f} seconds")
    print(f"Second call (cached): {second_call_time:.3f} seconds")
    print(f"Speedup: {speedup:.1f}x")
    
    # Verify results are the same
    np.testing.assert_array_almost_equal(result1, result2)

def run_all_tests():
    """Run all tests."""
    print("Running Enhanced Caching and Optimization Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEnhancedCachingSystem))
    test_suite.addTest(unittest.makeSuite(TestOptimizationDecorators))
    test_suite.addTest(unittest.makeSuite(TestIntegratedHardwareManager))
    test_suite.addTest(unittest.makeSuite(TestOptimizationPatches))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmarks
    run_performance_benchmark()
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)