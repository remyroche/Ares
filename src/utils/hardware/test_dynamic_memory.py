"""
Test Suite for Dynamic Memory Allocation System.

This module provides comprehensive tests for the dynamic memory allocation
system including system detection, workload-based allocation, and adaptive learning.
"""

import unittest
import pandas as pd
import numpy as np
import time
import psutil
from typing import Dict, Any

from .dynamic_memory_allocator import (
    DynamicMemoryAllocator, SystemTier, WorkloadType, MemoryAllocation,
    get_dynamic_allocator, get_optimal_memory_allocation, get_system_recommendations,
    update_memory_usage
)

class TestDynamicMemoryAllocator(unittest.TestCase):
    """Test cases for dynamic memory allocator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.allocator = DynamicMemoryAllocator()
    
    def test_system_detection(self):
        """Test system resource detection."""
        resources = self.allocator.system_resources
        
        # Check that resources are detected
        self.assertGreater(resources.total_memory_gb, 0)
        self.assertGreater(resources.cpu_cores, 0)
        self.assertIsInstance(resources.system_tier, SystemTier)
        self.assertIsInstance(resources.is_m1_chip, bool)
        self.assertIsInstance(resources.is_ssd, bool)
    
    def test_workload_based_allocation(self):
        """Test memory allocation based on workload type."""
        # Test different workload types
        workloads = [
            WorkloadType.LIGHT,
            WorkloadType.MODERATE,
            WorkloadType.HEAVY,
            WorkloadType.EXTREME,
            WorkloadType.STREAMING
        ]
        
        allocations = []
        for workload in workloads:
            allocation = self.allocator.get_optimal_allocation(workload)
            allocations.append(allocation)
            
            # Check allocation properties
            self.assertGreater(allocation.cache_memory_mb, 0)
            self.assertGreater(allocation.processing_memory_mb, 0)
            self.assertGreater(allocation.buffer_memory_mb, 0)
            self.assertGreater(allocation.total_allocated_mb, 0)
            self.assertIsInstance(allocation.allocation_strategy, str)
        
        # Check that different workloads get different allocations
        cache_sizes = [a.cache_memory_mb for a in allocations]
        self.assertGreater(len(set(cache_sizes)), 1)  # Should have different sizes
    
    def test_data_size_adaptation(self):
        """Test memory allocation adaptation based on data size."""
        # Test different data sizes
        data_sizes = [100, 1000, 5000, 20000]  # MB
        
        allocations = []
        for data_size in data_sizes:
            allocation = self.allocator.get_optimal_allocation(
                WorkloadType.MODERATE, 
                data_size_mb=data_size
            )
            allocations.append(allocation)
        
        # Check that larger data sizes get more memory
        cache_sizes = [a.cache_memory_mb for a in allocations]
        processing_sizes = [a.processing_memory_mb for a in allocations]
        
        # Generally, larger data should get more memory
        self.assertGreaterEqual(cache_sizes[-1], cache_sizes[0])
        self.assertGreaterEqual(processing_sizes[-1], processing_sizes[0])
    
    def test_user_preferences(self):
        """Test memory allocation with user preferences."""
        # Conservative preferences
        conservative_prefs = {'memory_usage_factor': 0.7}
        conservative_allocation = self.allocator.get_optimal_allocation(
            WorkloadType.MODERATE,
            user_preferences=conservative_prefs
        )
        
        # Aggressive preferences
        aggressive_prefs = {'memory_usage_factor': 1.3}
        aggressive_allocation = self.allocator.get_optimal_allocation(
            WorkloadType.MODERATE,
            user_preferences=aggressive_prefs
        )
        
        # Aggressive should allocate more memory
        self.assertGreater(
            aggressive_allocation.total_allocated_mb,
            conservative_allocation.total_allocated_mb
        )
    
    def test_adaptive_learning(self):
        """Test adaptive learning functionality."""
        # Simulate memory pressure scenarios
        pressure_scenarios = [
            (0.3, "low"),
            (0.6, "medium"),
            (0.8, "high"),
            (0.95, "critical")
        ]
        
        for pressure_ratio, pressure_level in pressure_scenarios:
            total_memory = psutil.virtual_memory().total / (1024 * 1024)
            used_memory = total_memory * pressure_ratio
            
            # Update memory usage
            self.allocator.update_memory_usage(used_memory, pressure_level)
        
        # Check that adaptive factors were updated
        self.assertIn('memory_pressure', self.allocator.adaptive_factors)
        self.assertIn('workload_intensity', self.allocator.adaptive_factors)
        self.assertIn('success_rate', self.allocator.adaptive_factors)
        self.assertIn('performance_score', self.allocator.adaptive_factors)
    
    def test_allocation_history(self):
        """Test allocation history tracking."""
        # Make several allocations
        for i in range(5):
            allocation = self.allocator.get_optimal_allocation(WorkloadType.MODERATE)
            time.sleep(0.01)  # Small delay
        
        # Check history
        self.assertEqual(len(self.allocator.allocation_history), 5)
        
        # Check history structure
        for record in self.allocator.allocation_history:
            self.assertIn('timestamp', record)
            self.assertIn('allocation', record)
            self.assertIn('workload_type', record)
            self.assertIn('factors', record)
    
    def test_system_recommendations(self):
        """Test system recommendations."""
        recommendations = self.allocator.get_system_recommendations()
        
        # Check required fields
        self.assertIn('system_tier', recommendations)
        self.assertIn('total_memory_gb', recommendations)
        self.assertIn('recommended_cache_mb', recommendations)
        self.assertIn('recommended_processing_mb', recommendations)
        self.assertIn('optimization_tips', recommendations)
        
        # Check data types
        self.assertIsInstance(recommendations['system_tier'], str)
        self.assertIsInstance(recommendations['total_memory_gb'], float)
        self.assertIsInstance(recommendations['recommended_cache_mb'], float)
        self.assertIsInstance(recommendations['optimization_tips'], list)
    
    def test_allocation_stats(self):
        """Test allocation statistics."""
        # Make some allocations
        for i in range(10):
            self.allocator.get_optimal_allocation(WorkloadType.MODERATE)
        
        stats = self.allocator.get_allocation_stats()
        
        # Check required fields
        self.assertIn('total_allocations', stats)
        self.assertIn('recent_allocations', stats)
        self.assertIn('average_cache_mb', stats)
        self.assertIn('average_processing_mb', stats)
        self.assertIn('average_total_mb', stats)
        self.assertIn('adaptive_factors', stats)
        
        # Check values
        self.assertEqual(stats['total_allocations'], 10)
        self.assertGreater(stats['average_total_mb'], 0)

class TestDynamicMemoryFunctions(unittest.TestCase):
    """Test cases for dynamic memory functions."""
    
    def test_get_optimal_memory_allocation(self):
        """Test get_optimal_memory_allocation function."""
        allocation = get_optimal_memory_allocation(WorkloadType.MODERATE)
        
        self.assertIsInstance(allocation, MemoryAllocation)
        self.assertGreater(allocation.cache_memory_mb, 0)
        self.assertGreater(allocation.processing_memory_mb, 0)
        self.assertGreater(allocation.total_allocated_mb, 0)
    
    def test_get_system_recommendations(self):
        """Test get_system_recommendations function."""
        recommendations = get_system_recommendations()
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('system_tier', recommendations)
        self.assertIn('total_memory_gb', recommendations)
    
    def test_update_memory_usage(self):
        """Test update_memory_usage function."""
        # This should not raise an exception
        update_memory_usage(1000.0, 'medium')
        update_memory_usage(5000.0, 'high')
        update_memory_usage(8000.0, 'critical')

def run_dynamic_memory_benchmarks():
    """Run dynamic memory allocation benchmarks."""
    print("Running Dynamic Memory Allocation Benchmarks...")
    
    # Benchmark 1: System detection performance
    print("\n=== System Detection Performance ===")
    
    start_time = time.time()
    allocator = DynamicMemoryAllocator()
    detection_time = time.time() - start_time
    
    print(f"System detection time: {detection_time:.3f}s")
    print(f"System tier: {allocator.system_resources.system_tier.value}")
    print(f"Total memory: {allocator.system_resources.total_memory_gb:.1f}GB")
    print(f"CPU cores: {allocator.system_resources.cpu_cores}")
    
    # Benchmark 2: Allocation performance
    print("\n=== Allocation Performance ===")
    
    workloads = [WorkloadType.LIGHT, WorkloadType.MODERATE, WorkloadType.HEAVY]
    data_sizes = [100, 1000, 5000]  # MB
    
    for workload in workloads:
        for data_size in data_sizes:
            start_time = time.time()
            allocation = get_optimal_memory_allocation(workload, data_size)
            allocation_time = time.time() - start_time
            
            print(f"{workload.value} + {data_size}MB: {allocation_time:.4f}s "
                  f"({allocation.total_allocated_mb:.0f}MB)")
    
    # Benchmark 3: Adaptive learning performance
    print("\n=== Adaptive Learning Performance ===")
    
    # Simulate memory pressure updates
    start_time = time.time()
    for i in range(100):
        pressure_levels = ['low', 'medium', 'high', 'critical']
        pressure = pressure_levels[i % len(pressure_levels)]
        used_memory = 1000 + i * 100
        update_memory_usage(used_memory, pressure)
    
    learning_time = time.time() - start_time
    print(f"100 memory updates: {learning_time:.3f}s")
    
    # Get final stats
    stats = allocator.get_allocation_stats()
    print(f"Total allocations: {stats['total_allocations']}")
    print(f"Adaptive factors: {stats['adaptive_factors']}")

def run_all_dynamic_memory_tests():
    """Run all dynamic memory allocation tests."""
    print("Running Dynamic Memory Allocation Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDynamicMemoryAllocator))
    test_suite.addTest(unittest.makeSuite(TestDynamicMemoryFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmarks
    run_dynamic_memory_benchmarks()
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_dynamic_memory_tests()
    if success:
        print("\n✅ All dynamic memory allocation tests passed!")
    else:
        print("\n❌ Some dynamic memory allocation tests failed!")
        exit(1)