"""
Comprehensive Tests for M1 Hardware Optimizations.

This module provides comprehensive tests for all M1/M2/M3/M4 hardware optimizations
including unified memory management, CPU optimization, GPU acceleration, and Neural Engine integration.
"""

import unittest
import time
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional
import tempfile
import os

# Import all M1 optimization modules
from .m1_comprehensive_optimizer import (
    get_comprehensive_optimizer, ComprehensiveConfig, OptimizationStrategy,
    WorkloadCategory, m1_optimized
)
from .m1_unified_memory_manager import (
    get_unified_memory_manager, optimize_for_unified_memory, unified_memory_optimized
)
from .m1_advanced_cpu_optimizer import (
    get_advanced_cpu_optimizer, optimize_cpu_execution, parallel_cpu_execution
)
from .m1_enhanced_gpu_manager import (
    get_enhanced_gpu_manager, gpu_accelerated, GPUOperationType
)
from .m1_neural_engine_manager import (
    get_neural_engine_manager, neural_engine_optimized, NeuralEngineOperation
)

# Suppress logging during tests
logging.getLogger().setLevel(logging.CRITICAL)

class TestM1UnifiedMemoryManager(unittest.TestCase):
    """Test unified memory management."""
    
    def setUp(self):
        self.memory_manager = get_unified_memory_manager()
    
    def test_memory_allocation(self):
        """Test memory allocation and deallocation."""
        # Allocate memory
        allocation_id = self.memory_manager.allocate_for_operation(
            'matrix_operations', 100.0, 'test'
        )
        
        self.assertIsNotNone(allocation_id)
        self.assertIn(allocation_id, self.memory_manager.memory_pool.allocations)
        
        # Free memory
        success = self.memory_manager.memory_pool.free_memory(allocation_id)
        self.assertTrue(success)
        self.assertNotIn(allocation_id, self.memory_manager.memory_pool.allocations)
    
    def test_memory_optimization(self):
        """Test memory optimization for different data types."""
        # Test NumPy array optimization
        large_array = np.random.random((1000, 1000)).astype(np.float64)
        optimized_array = optimize_for_unified_memory(large_array, 'matrix_operations', 'gpu')
        
        self.assertEqual(optimized_array.shape, large_array.shape)
        self.assertEqual(optimized_array.dtype, np.float32)  # Should be optimized to float32
        
        # Test DataFrame optimization
        large_df = pd.DataFrame(np.random.random((10000, 100)))
        optimized_df = optimize_for_unified_memory(large_df, 'data_processing', 'cpu')
        
        self.assertEqual(optimized_df.shape, large_df.shape)
    
    def test_memory_stats(self):
        """Test memory statistics."""
        stats = self.memory_manager.get_comprehensive_stats()
        
        self.assertIn('allocations', stats)
        self.assertIn('current_usage_mb', stats)
        self.assertIn('peak_usage_mb', stats)
        self.assertIn('system', stats)
    
    def test_unified_memory_decorator(self):
        """Test unified memory decorator."""
        @unified_memory_optimized('matrix_operations', 'gpu')
        def test_function(data):
            return data * 2
        
        test_data = np.random.random((100, 100))
        result = test_function(test_data)
        
        self.assertEqual(result.shape, test_data.shape)
        np.testing.assert_array_almost_equal(result, test_data * 2)

class TestM1AdvancedCPUOptimizer(unittest.TestCase):
    """Test advanced CPU optimization."""
    
    def setUp(self):
        self.cpu_optimizer = get_advanced_cpu_optimizer()
    
    def test_cpu_execution_optimization(self):
        """Test CPU execution optimization."""
        @optimize_cpu_execution(WorkloadType.CPU_INTENSIVE)
        def cpu_intensive_task(data):
            return np.sum(data ** 2)
        
        test_data = np.random.random((1000, 1000))
        
        start_time = time.time()
        result = cpu_intensive_task(test_data)
        execution_time = time.time() - start_time
        
        self.assertIsNotNone(result)
        self.assertGreater(execution_time, 0)
    
    def test_parallel_execution(self):
        """Test parallel CPU execution."""
        @parallel_cpu_execution(WorkloadType.MIXED)
        def parallel_task(data_list):
            return [np.sum(data) for data in data_list]
        
        test_data_list = [np.random.random((100, 100)) for _ in range(4)]
        
        start_time = time.time()
        results = parallel_task(test_data_list)
        execution_time = time.time() - start_time
        
        self.assertEqual(len(results), len(test_data_list))
        self.assertGreater(execution_time, 0)
    
    def test_performance_metrics(self):
        """Test CPU performance metrics."""
        metrics = self.cpu_optimizer.get_performance_metrics()
        
        self.assertIn('cpu_metrics', metrics)
        self.assertIn('thermal_state', metrics)
        self.assertIn('core_utilization', metrics)
        self.assertIn('available_cores', metrics)

class TestM1EnhancedGPUManager(unittest.TestCase):
    """Test enhanced GPU manager."""
    
    def setUp(self):
        self.gpu_manager = get_enhanced_gpu_manager()
    
    def test_gpu_availability(self):
        """Test GPU availability detection."""
        is_available = self.gpu_manager.is_available()
        self.assertIsInstance(is_available, bool)
    
    def test_gpu_matrix_operations(self):
        """Test GPU matrix operations."""
        if not self.gpu_manager.is_available():
            self.skipTest("GPU not available")
        
        A = np.random.random((500, 500)).astype(np.float32)
        B = np.random.random((500, 500)).astype(np.float32)
        
        result = self.gpu_manager.execute_matrix_multiply(A, B)
        
        self.assertEqual(result.shape, (500, 500))
        self.assertEqual(result.dtype, np.float32)
    
    def test_gpu_accelerated_decorator(self):
        """Test GPU accelerated decorator."""
        @gpu_accelerated(GPUOperationType.MATRIX_MULTIPLICATION)
        def gpu_matrix_multiply(A, B):
            return np.dot(A, B)
        
        A = np.random.random((100, 100)).astype(np.float32)
        B = np.random.random((100, 100)).astype(np.float32)
        
        result = gpu_matrix_multiply(A, B)
        
        self.assertEqual(result.shape, (100, 100))
    
    def test_gpu_performance_metrics(self):
        """Test GPU performance metrics."""
        metrics = self.gpu_manager.get_performance_metrics()
        
        self.assertIn('gpu_metrics', metrics)
        self.assertIn('memory_stats', metrics)
        self.assertIn('queue_status', metrics)
        self.assertIn('mps_available', metrics)

class TestM1NeuralEngineManager(unittest.TestCase):
    """Test Neural Engine manager."""
    
    def setUp(self):
        self.neural_manager = get_neural_engine_manager()
    
    def test_neural_engine_availability(self):
        """Test Neural Engine availability detection."""
        is_available = self.neural_manager.is_available()
        self.assertIsInstance(is_available, bool)
    
    def test_neural_engine_capabilities(self):
        """Test Neural Engine capabilities."""
        capabilities = self.neural_manager.get_capabilities()
        
        self.assertIn('is_available', capabilities)
        self.assertIn('neural_engine_count', capabilities)
        self.assertIn('max_ops_per_second', capabilities)
        self.assertIn('supported_operations', capabilities)
    
    def test_neural_engine_decorator(self):
        """Test Neural Engine decorator."""
        @neural_engine_optimized(NeuralEngineOperation.INFERENCE)
        def neural_inference(model, data):
            return np.random.random(data.shape[0])
        
        test_data = np.random.random((100, 10))
        dummy_model = "dummy_model"
        
        result = neural_inference(dummy_model, test_data)
        
        self.assertEqual(len(result), test_data.shape[0])
    
    def test_neural_engine_metrics(self):
        """Test Neural Engine metrics."""
        metrics = self.neural_engine_manager.get_performance_metrics()
        
        self.assertIn('neural_engine_available', metrics)
        self.assertIn('capabilities', metrics)
        self.assertIn('executor_metrics', metrics)

class TestM1ComprehensiveOptimizer(unittest.TestCase):
    """Test comprehensive M1 optimizer."""
    
    def setUp(self):
        self.optimizer = get_comprehensive_optimizer()
    
    def test_optimization_strategies(self):
        """Test different optimization strategies."""
        strategies = [
            OptimizationStrategy.MAXIMUM_PERFORMANCE,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.POWER_EFFICIENT,
            OptimizationStrategy.MEMORY_OPTIMIZED,
            OptimizationStrategy.NEURAL_OPTIMIZED
        ]
        
        for strategy in strategies:
            config = ComprehensiveConfig(optimization_strategy=strategy)
            optimizer = get_comprehensive_optimizer(config)
            self.assertIsNotNone(optimizer)
    
    def test_workload_categories(self):
        """Test different workload categories."""
        categories = [
            WorkloadCategory.MACHINE_LEARNING,
            WorkloadCategory.DATA_PROCESSING,
            WorkloadCategory.FINANCIAL_MODELING,
            WorkloadCategory.BACKTESTING,
            WorkloadCategory.REAL_TIME_TRADING,
            WorkloadCategory.BATCH_PROCESSING,
            WorkloadCategory.STREAMING
        ]
        
        for category in categories:
            config = ComprehensiveConfig(workload_category=category)
            optimizer = get_comprehensive_optimizer(config)
            self.assertIsNotNone(optimizer)
    
    def test_comprehensive_optimization(self):
        """Test comprehensive optimization."""
        @m1_optimized("matrix_operations", WorkloadCategory.MACHINE_LEARNING)
        def optimized_function(data):
            return np.dot(data, data.T)
        
        test_data = np.random.random((200, 200)).astype(np.float32)
        
        start_time = time.time()
        result = optimized_function(test_data)
        execution_time = time.time() - start_time
        
        self.assertEqual(result.shape, (200, 200))
        self.assertGreater(execution_time, 0)
    
    def test_optimization_result(self):
        """Test optimization result."""
        result = self.optimizer.optimize_operation("test_operation", np.random.random((100, 100)))
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.execution_time, float)
        self.assertIsInstance(result.memory_used_mb, float)
        self.assertIsInstance(result.optimization_applied, list)
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics."""
        metrics = self.optimizer.get_comprehensive_metrics()
        
        self.assertIn('overall_metrics', metrics)
        self.assertIn('unified_memory', metrics)
        self.assertIn('cpu_optimizer', metrics)
        self.assertIn('gpu_manager', metrics)
        self.assertIn('neural_engine', metrics)
        self.assertIn('adaptive_optimizer', metrics)

class TestM1Integration(unittest.TestCase):
    """Test M1 integration scenarios."""
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization scenario."""
        # Create test data
        large_matrix = np.random.random((1000, 1000)).astype(np.float32)
        large_dataframe = pd.DataFrame(np.random.random((10000, 100)))
        
        # Test unified memory optimization
        optimized_matrix = optimize_for_unified_memory(large_matrix, 'matrix_operations', 'gpu')
        optimized_dataframe = optimize_for_unified_memory(large_dataframe, 'data_processing', 'cpu')
        
        # Test CPU optimization
        @optimize_cpu_execution(WorkloadType.CPU_INTENSIVE)
        def cpu_task(data):
            return np.sum(data ** 2)
        
        cpu_result = cpu_task(optimized_matrix)
        
        # Test GPU optimization (if available)
        gpu_manager = get_enhanced_gpu_manager()
        if gpu_manager.is_available():
            gpu_result = gpu_manager.execute_matrix_multiply(optimized_matrix, optimized_matrix.T)
            self.assertEqual(gpu_result.shape, (1000, 1000))
        
        # Test comprehensive optimization
        @m1_optimized("comprehensive_test", WorkloadCategory.DATA_PROCESSING)
        def comprehensive_task(data):
            return np.mean(data, axis=0)
        
        comprehensive_result = comprehensive_task(optimized_dataframe)
        
        # Verify results
        self.assertIsNotNone(cpu_result)
        self.assertIsNotNone(comprehensive_result)
        self.assertEqual(len(comprehensive_result), optimized_dataframe.shape[1])
    
    def test_memory_pressure_handling(self):
        """Test memory pressure handling."""
        memory_manager = get_unified_memory_manager()
        
        # Allocate large amounts of memory to trigger pressure
        allocations = []
        for i in range(10):
            allocation_id = memory_manager.allocate_for_operation(
                'test_operation', 100.0, f'test_{i}'
            )
            allocations.append(allocation_id)
        
        # Check memory stats
        stats = memory_manager.get_comprehensive_stats()
        self.assertGreater(stats['current_usage_mb'], 0)
        
        # Cleanup
        for allocation_id in allocations:
            memory_manager.memory_pool.free_memory(allocation_id)
    
    def test_thermal_management(self):
        """Test thermal management."""
        cpu_optimizer = get_advanced_cpu_optimizer()
        
        # Get thermal state
        thermal_state = cpu_optimizer.thermal_manager.get_thermal_state()
        self.assertIsNotNone(thermal_state)
        
        # Get temperature history
        temp_history = cpu_optimizer.thermal_manager.get_temperature_history()
        self.assertIsInstance(temp_history, list)
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        # Get metrics from all components
        memory_metrics = get_unified_memory_manager().get_comprehensive_stats()
        cpu_metrics = get_advanced_cpu_optimizer().get_performance_metrics()
        gpu_metrics = get_enhanced_gpu_manager().get_performance_metrics()
        neural_metrics = get_neural_engine_manager().get_performance_metrics()
        comprehensive_metrics = get_comprehensive_optimizer().get_comprehensive_metrics()
        
        # Verify all metrics are present
        self.assertIsInstance(memory_metrics, dict)
        self.assertIsInstance(cpu_metrics, dict)
        self.assertIsInstance(gpu_metrics, dict)
        self.assertIsInstance(neural_metrics, dict)
        self.assertIsInstance(comprehensive_metrics, dict)

class TestM1Performance(unittest.TestCase):
    """Test M1 performance optimizations."""
    
    def test_memory_optimization_performance(self):
        """Test memory optimization performance."""
        # Create large dataset
        large_data = np.random.random((2000, 2000)).astype(np.float64)
        
        # Test without optimization
        start_time = time.time()
        result_unoptimized = large_data * 2
        unoptimized_time = time.time() - start_time
        
        # Test with optimization
        start_time = time.time()
        optimized_data = optimize_for_unified_memory(large_data, 'matrix_operations', 'gpu')
        result_optimized = optimized_data * 2
        optimized_time = time.time() - start_time
        
        # Verify results are equivalent
        np.testing.assert_array_almost_equal(result_unoptimized, result_optimized, decimal=5)
        
        # Memory optimization should reduce memory usage
        self.assertLess(optimized_data.nbytes, large_data.nbytes)
    
    def test_cpu_optimization_performance(self):
        """Test CPU optimization performance."""
        test_data = np.random.random((1000, 1000))
        
        # Test without optimization
        def unoptimized_task(data):
            return np.sum(data ** 2)
        
        start_time = time.time()
        result_unoptimized = unoptimized_task(test_data)
        unoptimized_time = time.time() - start_time
        
        # Test with optimization
        @optimize_cpu_execution(WorkloadType.CPU_INTENSIVE)
        def optimized_task(data):
            return np.sum(data ** 2)
        
        start_time = time.time()
        result_optimized = optimized_task(test_data)
        optimized_time = time.time() - start_time
        
        # Verify results are equivalent
        self.assertAlmostEqual(result_unoptimized, result_optimized, places=5)
    
    def test_gpu_acceleration_performance(self):
        """Test GPU acceleration performance."""
        gpu_manager = get_enhanced_gpu_manager()
        
        if not gpu_manager.is_available():
            self.skipTest("GPU not available")
        
        A = np.random.random((1000, 1000)).astype(np.float32)
        B = np.random.random((1000, 1000)).astype(np.float32)
        
        # Test CPU matrix multiplication
        start_time = time.time()
        cpu_result = np.dot(A, B)
        cpu_time = time.time() - start_time
        
        # Test GPU matrix multiplication
        start_time = time.time()
        gpu_result = gpu_manager.execute_matrix_multiply(A, B)
        gpu_time = time.time() - start_time
        
        # Verify results are equivalent
        np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=5)
        
        # GPU should be faster for large matrices
        if gpu_time < cpu_time:
            improvement = (cpu_time - gpu_time) / cpu_time * 100
            print(f"GPU acceleration: {improvement:.1f}% improvement")

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestM1UnifiedMemoryManager,
        TestM1AdvancedCPUOptimizer,
        TestM1EnhancedGPUManager,
        TestM1NeuralEngineManager,
        TestM1ComprehensiveOptimizer,
        TestM1Integration,
        TestM1Performance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")