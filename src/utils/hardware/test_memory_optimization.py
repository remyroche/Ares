"""
Test Suite for Advanced Memory Management and Optimization.

This module provides comprehensive tests for the advanced memory management
system including garbage collection, chunking, and memory optimization.
"""

import unittest
import pandas as pd
import numpy as np
import time
import gc
from typing import Dict, Any

from .advanced_memory_manager import (
    AdvancedMemoryManager, MemoryConfig, MemoryPressureLevel, ChunkingStrategy,
    get_advanced_memory_manager, memory_efficient_processing, chunked_processing
)
from .memory_optimized_decorators import (
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    force_garbage_collection, cleanup_all_memory, get_memory_optimization_stats
)

class TestAdvancedMemoryManager(unittest.TestCase):
    """Test cases for advanced memory manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MemoryConfig(
            enable_aggressive_gc=True,
            gc_threshold_mb=50.0,
            enable_memory_pressure_detection=True,
            enable_chunking=True,
            default_chunk_size_mb=10.0,
            enable_memory_pools=True,
            pool_size_mb=20.0,
            enable_weak_references=True
        )
        self.memory_manager = AdvancedMemoryManager(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.memory_manager.shutdown()
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        stats = self.memory_manager.get_memory_stats()
        
        # Check that stats contain expected fields
        self.assertIn('total_memory_mb', stats.__dict__)
        self.assertIn('used_memory_mb', stats.__dict__)
        self.assertIn('available_memory_mb', stats.__dict__)
        self.assertIn('memory_percent', stats.__dict__)
        self.assertIn('pressure_level', stats.__dict__)
        self.assertIn('gc_count', stats.__dict__)
        self.assertIn('objects_tracked', stats.__dict__)
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection."""
        # Create memory pressure
        large_objects = []
        for i in range(10):
            large_objects.append(np.random.rand(1000, 1000))
        
        # Check pressure level
        stats = self.memory_manager.get_memory_stats()
        self.assertIn(stats.pressure_level, MemoryPressureLevel)
        
        # Clean up
        del large_objects
        self.memory_manager.cleanup_all()
    
    def test_chunking_dataframe(self):
        """Test DataFrame chunking."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'id': np.arange(10000),
            'value': np.random.rand(10000)
        })
        
        # Test chunking
        chunks = list(self.memory_manager.chunk_data(large_df, chunk_size_bytes=1024))
        
        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that all data is preserved
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, len(large_df))
    
    def test_chunking_numpy_array(self):
        """Test NumPy array chunking."""
        # Create large array
        large_array = np.random.rand(10000, 100)
        
        # Test chunking
        chunks = list(self.memory_manager.chunk_data(large_array, chunk_size_bytes=1024))
        
        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that all data is preserved
        total_elements = sum(chunk.size for chunk in chunks)
        self.assertEqual(total_elements, large_array.size)
    
    def test_memory_context(self):
        """Test memory context manager."""
        with self.memory_manager.memory_context("test_operation"):
            # Create some data
            data = np.random.rand(1000, 1000)
            result = np.dot(data.T, data)
            
            # Check that context works
            self.assertIsNotNone(result)
    
    def test_weak_reference_tracking(self):
        """Test weak reference tracking."""
        # Create large object
        large_obj = np.random.rand(1000, 1000)
        
        # Track with weak reference
        weak_ref = self.memory_manager.track_object(large_obj)
        
        # Check that reference exists
        self.assertIsNotNone(weak_ref())
        
        # Delete object
        del large_obj
        
        # Force garbage collection
        gc.collect()
        
        # Check that reference is now dead
        self.assertIsNone(weak_ref())
    
    def test_memory_pool(self):
        """Test memory pool functionality."""
        memory_pool = self.memory_manager.get_memory_pool()
        
        if memory_pool:
            # Get array from pool
            arr1 = memory_pool.get_numpy_array((100, 100), np.float32)
            self.assertEqual(arr1.shape, (100, 100))
            self.assertEqual(arr1.dtype, np.float32)
            
            # Return array to pool
            memory_pool.return_numpy_array(arr1)
            
            # Get array again (should reuse from pool)
            arr2 = memory_pool.get_numpy_array((100, 100), np.float32)
            self.assertEqual(arr2.shape, (100, 100))
            self.assertEqual(arr2.dtype, np.float32)
    
    def test_cleanup_all(self):
        """Test comprehensive cleanup."""
        # Create some data
        large_objects = [np.random.rand(1000, 1000) for _ in range(5)]
        
        # Perform cleanup
        self.memory_manager.cleanup_all()
        
        # Check that cleanup completed without error
        self.assertTrue(True)  # If we get here, cleanup succeeded

class TestMemoryOptimizedDecorators(unittest.TestCase):
    """Test cases for memory-optimized decorators."""
    
    def test_memory_optimized_decorator(self):
        """Test memory optimized decorator."""
        @memory_optimized(
            optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
            enable_aggressive_gc=True,
            log_memory_usage=False
        )
        def process_data(data):
            return data * 2
        
        # Test with DataFrame
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = process_data(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(df))
    
    def test_gc_optimized_decorator(self):
        """Test GC optimized decorator."""
        @gc_optimized(gc_after_function=True)
        def create_large_data():
            return np.random.rand(1000, 1000)
        
        # Test function
        result = create_large_data()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1000, 1000))
    
    def test_chunked_processing_auto_decorator(self):
        """Test chunked processing auto decorator."""
        @chunked_processing_auto(chunk_size_mb=0.1)  # Very small chunks for testing
        def process_large_data(data):
            return data.describe()
        
        # Create large DataFrame
        large_df = pd.DataFrame({
            'value': np.random.rand(10000)
        })
        
        # Process with chunking
        result = process_large_data(large_df)
        
        # Check result
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_comprehensive_memory_optimization(self):
        """Test comprehensive memory optimization decorator."""
        @comprehensive_memory_optimization(
            optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
            enable_caching=True,
            enable_chunking=True,
            enable_gc=True
        )
        def process_complex_data(data):
            return data * 2
        
        # Test with DataFrame
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = process_complex_data(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(df))

class TestMemoryOptimizationFunctions(unittest.TestCase):
    """Test cases for memory optimization functions."""
    
    def test_force_garbage_collection(self):
        """Test force garbage collection function."""
        # Create some objects
        large_objects = [np.random.rand(1000, 1000) for _ in range(5)]
        
        # Force garbage collection
        force_garbage_collection()
        
        # Check that function completed without error
        self.assertTrue(True)
    
    def test_cleanup_all_memory(self):
        """Test cleanup all memory function."""
        # Create some objects
        large_objects = [np.random.rand(1000, 1000) for _ in range(5)]
        
        # Cleanup all memory
        cleanup_all_memory()
        
        # Check that function completed without error
        self.assertTrue(True)
    
    def test_get_memory_optimization_stats(self):
        """Test get memory optimization stats function."""
        stats = get_memory_optimization_stats()
        
        # Check that stats contain expected fields
        self.assertIn('memory_stats', stats)
        self.assertIn('gc_stats', stats)
        
        # Check memory stats
        memory_stats = stats['memory_stats']
        self.assertIn('total_mb', memory_stats)
        self.assertIn('used_mb', memory_stats)
        self.assertIn('available_mb', memory_stats)
        self.assertIn('percent', memory_stats)
        self.assertIn('pressure_level', memory_stats)

class TestMemoryEfficientProcessing(unittest.TestCase):
    """Test cases for memory efficient processing."""
    
    def test_memory_efficient_processing_decorator(self):
        """Test memory efficient processing decorator."""
        @memory_efficient_processing
        def process_data(data):
            return data * 2
        
        # Test with DataFrame
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = process_data(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(df))
    
    def test_chunked_processing_decorator(self):
        """Test chunked processing decorator."""
        @chunked_processing(chunk_size_mb=0.1)
        def process_large_data(data):
            return data.describe()
        
        # Create large DataFrame
        large_df = pd.DataFrame({
            'value': np.random.rand(10000)
        })
        
        # Process with chunking
        result = process_large_data(large_df)
        
        # Check result
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_track_memory_usage_decorator(self):
        """Test track memory usage decorator."""
        @track_memory_usage
        def process_data(data):
            return data * 2
        
        # Test with DataFrame
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = process_data(df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(df))

def run_memory_optimization_benchmarks():
    """Run memory optimization benchmarks."""
    print("Running Memory Optimization Benchmarks...")
    
    # Benchmark 1: Memory optimization with GC
    print("\n=== Memory Optimization with GC Benchmark ===")
    
    # Create large DataFrame
    large_df = pd.DataFrame({
        'id': np.arange(100000, dtype=np.int64),
        'price': np.random.uniform(0, 1000, 100000).astype(np.float64),
        'category': ['A', 'B', 'C', 'D'] * 25000
    })
    
    original_memory = large_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Original memory usage: {original_memory:.2f} MB")
    
    # Test without optimization
    start_time = time.time()
    start_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.NONE)
    def process_without_optimization(df):
        return df.copy()
    
    result1 = process_without_optimization(large_df)
    end_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    time1 = time.time() - start_time
    memory_delta1 = end_memory - start_memory
    
    # Test with optimization
    start_time = time.time()
    start_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def process_with_optimization(df):
        return df.copy()
    
    result2 = process_with_optimization(large_df)
    end_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    time2 = time.time() - start_time
    memory_delta2 = end_memory - start_memory
    
    print(f"Without optimization: {time1:.3f}s, Memory delta: {memory_delta1:+.1f}MB")
    print(f"With optimization: {time2:.3f}s, Memory delta: {memory_delta2:+.1f}MB")
    print(f"Memory improvement: {memory_delta1 - memory_delta2:.1f}MB")
    
    # Benchmark 2: Chunked processing
    print("\n=== Chunked Processing Benchmark ===")
    
    # Create very large DataFrame
    very_large_df = pd.DataFrame({
        'id': np.arange(1000000),
        'value': np.random.rand(1000000)
    })
    
    # Test without chunking
    start_time = time.time()
    
    def process_without_chunking(df):
        return df.describe()
    
    result1 = process_without_chunking(very_large_df)
    time1 = time.time() - start_time
    
    # Test with chunking
    start_time = time.time()
    
    @chunked_processing_auto(chunk_size_mb=50.0)
    def process_with_chunking(df):
        return df.describe()
    
    result2 = process_with_chunking(very_large_df)
    time2 = time.time() - start_time
    
    print(f"Without chunking: {time1:.3f}s")
    print(f"With chunking: {time2:.3f}s")
    print(f"Time difference: {time1 - time2:+.3f}s")

def run_all_memory_tests():
    """Run all memory optimization tests."""
    print("Running Memory Optimization Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAdvancedMemoryManager))
    test_suite.addTest(unittest.makeSuite(TestMemoryOptimizedDecorators))
    test_suite.addTest(unittest.makeSuite(TestMemoryOptimizationFunctions))
    test_suite.addTest(unittest.makeSuite(TestMemoryEfficientProcessing))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmarks
    run_memory_optimization_benchmarks()
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_memory_tests()
    if success:
        print("\n✅ All memory optimization tests passed!")
    else:
        print("\n❌ Some memory optimization tests failed!")
        exit(1)