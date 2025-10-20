"""
Examples of Dynamic Memory Allocation and Management.

This module demonstrates the intelligent, dynamic memory allocation system
that adapts to system resources, workload characteristics, and real-time usage patterns.
"""

import pandas as pd
import numpy as np
import time
import psutil
from typing import Dict, Any, Optional

from .dynamic_memory_allocator import (
    get_dynamic_allocator, get_optimal_memory_allocation, get_system_recommendations,
    update_memory_usage, WorkloadType, SystemTier, MemoryAllocation
)
from .enhanced_caching_system import get_global_cache, CacheConfig
from .integrated_hardware_manager import get_integrated_hardware_manager

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_performance
)

def example_system_detection():
    """Example of system resource detection and tier classification."""
    tprint_info("=== System Detection and Classification ===")
    
    # Get system recommendations
    recommendations = get_system_recommendations()
    
    tprint_success(f"System Tier: {recommendations['system_tier']}")
    tprint_success(f"Total Memory: {recommendations['total_memory_gb']:.1f}GB")
    tprint_success(f"Recommended Cache: {recommendations['recommended_cache_mb']:.0f}MB")
    tprint_success(f"Recommended Processing: {recommendations['recommended_processing_mb']:.0f}MB")
    
    tprint_info("Optimization Tips:")
    for tip in recommendations['optimization_tips']:
        tprint_info(f"  • {tip}")

def example_workload_based_allocation():
    """Example of memory allocation based on different workload types."""
    tprint_info("=== Workload-Based Memory Allocation ===")
    
    # Test different workload types
    workloads = [
        (WorkloadType.LIGHT, "Small datasets, simple operations"),
        (WorkloadType.MODERATE, "Medium datasets, standard ML"),
        (WorkloadType.HEAVY, "Large datasets, complex ML"),
        (WorkloadType.EXTREME, "Very large datasets, deep learning"),
        (WorkloadType.STREAMING, "Continuous data processing")
    ]
    
    for workload_type, description in workloads:
        allocation = get_optimal_memory_allocation(workload_type)
        
        tprint_info(f"\n{workload_type.value.upper()} Workload ({description}):")
        tprint_info(f"  Cache Memory: {allocation.cache_memory_mb:.0f}MB")
        tprint_info(f"  Processing Memory: {allocation.processing_memory_mb:.0f}MB")
        tprint_info(f"  Buffer Memory: {allocation.buffer_memory_mb:.0f}MB")
        tprint_info(f"  Total Allocated: {allocation.total_allocated_mb:.0f}MB")
        tprint_info(f"  Strategy: {allocation.allocation_strategy}")

def example_data_size_adaptation():
    """Example of memory allocation adaptation based on data size."""
    tprint_info("=== Data Size-Based Memory Adaptation ===")
    
    # Test different data sizes
    data_sizes = [
        (100, "Small dataset (100MB)"),
        (1000, "Medium dataset (1GB)"),
        (5000, "Large dataset (5GB)"),
        (20000, "Very large dataset (20GB)"),
        (50000, "Extreme dataset (50GB)")
    ]
    
    for data_size_mb, description in data_sizes:
        allocation = get_optimal_memory_allocation(
            WorkloadType.MODERATE, 
            data_size_mb=data_size_mb
        )
        
        tprint_info(f"\n{description}:")
        tprint_info(f"  Cache Memory: {allocation.cache_memory_mb:.0f}MB")
        tprint_info(f"  Processing Memory: {allocation.processing_memory_mb:.0f}MB")
        tprint_info(f"  Total Allocated: {allocation.total_allocated_mb:.0f}MB")
        tprint_info(f"  Data/Total Ratio: {data_size_mb / allocation.total_allocated_mb:.2f}")

def example_user_preferences():
    """Example of memory allocation with user preferences."""
    tprint_info("=== User Preference-Based Memory Allocation ===")
    
    # Conservative user preferences
    conservative_prefs = {
        'memory_usage_factor': 0.7,  # Use 70% of recommended memory
        'prioritize_stability': True,
        'enable_aggressive_cleanup': True
    }
    
    # Aggressive user preferences
    aggressive_prefs = {
        'memory_usage_factor': 1.3,  # Use 130% of recommended memory
        'prioritize_performance': True,
        'enable_aggressive_cleanup': False
    }
    
    # Balanced user preferences
    balanced_prefs = {
        'memory_usage_factor': 1.0,  # Use 100% of recommended memory
        'balance_performance_stability': True
    }
    
    preferences = [
        (conservative_prefs, "Conservative (70% memory usage)"),
        (balanced_prefs, "Balanced (100% memory usage)"),
        (aggressive_prefs, "Aggressive (130% memory usage)")
    ]
    
    for prefs, description in preferences:
        allocation = get_optimal_memory_allocation(
            WorkloadType.MODERATE,
            data_size_mb=2000,
            user_preferences=prefs
        )
        
        tprint_info(f"\n{description}:")
        tprint_info(f"  Cache Memory: {allocation.cache_memory_mb:.0f}MB")
        tprint_info(f"  Processing Memory: {allocation.processing_memory_mb:.0f}MB")
        tprint_info(f"  Total Allocated: {allocation.total_allocated_mb:.0f}MB")

def example_adaptive_learning():
    """Example of adaptive learning based on memory usage patterns."""
    tprint_info("=== Adaptive Learning and Memory Usage Patterns ===")
    
    allocator = get_dynamic_allocator()
    
    # Simulate different memory pressure scenarios
    pressure_scenarios = [
        (0.3, "low", "Low memory pressure"),
        (0.6, "medium", "Medium memory pressure"),
        (0.8, "high", "High memory pressure"),
        (0.95, "critical", "Critical memory pressure")
    ]
    
    tprint_info("Simulating memory pressure scenarios:")
    
    for pressure_ratio, pressure_level, description in pressure_scenarios:
        # Simulate memory usage
        total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        used_memory = total_memory * pressure_ratio
        
        # Update memory usage
        update_memory_usage(used_memory, pressure_level)
        
        tprint_info(f"  {description}: {used_memory:.0f}MB used ({pressure_ratio:.1%})")
    
    # Get allocation stats
    stats = allocator.get_allocation_stats()
    tprint_info(f"\nAllocation Statistics:")
    tprint_info(f"  Total Allocations: {stats['total_allocations']}")
    tprint_info(f"  Average Total MB: {stats['average_total_mb']:.0f}")
    tprint_info(f"  Max Total MB: {stats['max_total_mb']:.0f}")
    tprint_info(f"  Min Total MB: {stats['min_total_mb']:.0f}")
    tprint_info(f"  Adaptive Factors: {stats['adaptive_factors']}")

def example_dynamic_cache_creation():
    """Example of creating caches with dynamic memory allocation."""
    tprint_info("=== Dynamic Cache Creation ===")
    
    # Create cache for light workload
    light_allocation = get_optimal_memory_allocation(WorkloadType.LIGHT)
    light_cache_config = CacheConfig(
        max_memory_mb=light_allocation.cache_memory_mb,
        strategy=CacheStrategy.LRU,
        data_type_optimization=DataTypeOptimization.AGGRESSIVE,
        enable_compression=True,
        auto_optimize_dtypes=True
    )
    light_cache = get_global_cache(light_cache_config)
    
    # Create cache for heavy workload
    heavy_allocation = get_optimal_memory_allocation(WorkloadType.HEAVY)
    heavy_cache_config = CacheConfig(
        max_memory_mb=heavy_allocation.cache_memory_mb,
        strategy=CacheStrategy.LRU,
        data_type_optimization=DataTypeOptimization.AGGRESSIVE,
        enable_compression=True,
        auto_optimize_dtypes=True
    )
    heavy_cache = get_global_cache(heavy_cache_config)
    
    tprint_success(f"Light workload cache: {light_allocation.cache_memory_mb:.0f}MB")
    tprint_success(f"Heavy workload cache: {heavy_allocation.cache_memory_mb:.0f}MB")
    
    # Test cache operations
    test_data = pd.DataFrame({
        'id': range(1000),
        'value': np.random.rand(1000)
    })
    
    # Store in light cache
    light_cache.put('test_data_light', test_data)
    tprint_info("Stored data in light workload cache")
    
    # Store in heavy cache
    heavy_cache.put('test_data_heavy', test_data)
    tprint_info("Stored data in heavy workload cache")
    
    # Get cache statistics
    light_stats = light_cache.get_statistics()
    heavy_stats = heavy_cache.get_statistics()
    
    tprint_info(f"Light cache stats: {light_stats.total_memory_used_mb:.1f}MB used")
    tprint_info(f"Heavy cache stats: {heavy_stats.total_memory_used_mb:.1f}MB used")

def example_integrated_hardware_with_dynamic_allocation():
    """Example of integrated hardware manager with dynamic allocation."""
    tprint_info("=== Integrated Hardware Manager with Dynamic Allocation ===")
    
    # Get integrated hardware manager (uses dynamic allocation internally)
    hardware_manager = get_integrated_hardware_manager()
    
    # Get current allocation info
    if hasattr(hardware_manager, 'current_allocation'):
        allocation = hardware_manager.current_allocation
        tprint_success(f"Current allocation: {allocation.total_allocated_mb:.0f}MB total")
        tprint_success(f"Cache: {allocation.cache_memory_mb:.0f}MB")
        tprint_success(f"Processing: {allocation.processing_memory_mb:.0f}MB")
        tprint_success(f"Buffer: {allocation.buffer_memory_mb:.0f}MB")
        tprint_success(f"Strategy: {allocation.allocation_strategy}")
    
    # Test processing with different data sizes
    test_sizes = [100, 1000, 5000]  # MB
    
    for size_mb in test_sizes:
        # Create test data
        test_data = pd.DataFrame({
            'id': range(size_mb * 1000),  # Roughly size_mb rows
            'value': np.random.rand(size_mb * 1000)
        })
        
        tprint_info(f"\nProcessing {size_mb}MB dataset:")
        
        # Process with hardware manager
        start_time = time.time()
        result = hardware_manager.process_data_with_optimization(
            test_data, 
            workload_type=WorkloadType.MODERATE
        )
        processing_time = time.time() - start_time
        
        tprint_success(f"Processed in {processing_time:.2f}s")
        tprint_success(f"Result shape: {result.shape}")
        
        # Get memory report
        memory_report = hardware_manager.get_memory_report()
        total_usage = memory_report['total_memory_usage_mb']
        tprint_info(f"Total memory usage: {total_usage:.1f}MB")

def example_real_time_adaptation():
    """Example of real-time memory adaptation during processing."""
    tprint_info("=== Real-Time Memory Adaptation ===")
    
    allocator = get_dynamic_allocator()
    
    # Simulate a long-running process with varying memory needs
    tprint_info("Simulating long-running process with varying memory needs...")
    
    for iteration in range(10):
        # Simulate different memory pressure levels
        pressure_levels = ['low', 'medium', 'high', 'critical']
        pressure = pressure_levels[iteration % len(pressure_levels)]
        
        # Simulate memory usage
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        pressure_ratios = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 0.95}
        used_memory = total_memory * pressure_ratios[pressure]
        
        # Update memory usage
        update_memory_usage(used_memory, pressure)
        
        # Get new allocation recommendation
        new_allocation = get_optimal_memory_allocation(
            WorkloadType.MODERATE,
            data_size_mb=1000
        )
        
        tprint_info(f"Iteration {iteration + 1}: {pressure} pressure, "
                   f"allocation: {new_allocation.total_allocated_mb:.0f}MB")
        
        # Small delay to simulate processing
        time.sleep(0.1)
    
    # Get final allocation statistics
    stats = allocator.get_allocation_stats()
    tprint_success(f"Final adaptive factors: {stats['adaptive_factors']}")

def run_all_dynamic_memory_examples():
    """Run all dynamic memory allocation examples."""
    tprint_info("🚀 Running Dynamic Memory Allocation Examples")
    
    try:
        example_system_detection()
        print()
        
        example_workload_based_allocation()
        print()
        
        example_data_size_adaptation()
        print()
        
        example_user_preferences()
        print()
        
        example_adaptive_learning()
        print()
        
        example_dynamic_cache_creation()
        print()
        
        example_integrated_hardware_with_dynamic_allocation()
        print()
        
        example_real_time_adaptation()
        print()
        
        tprint_success("✅ All dynamic memory allocation examples completed successfully!")
        
    except Exception as e:
        tprint_error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    run_all_dynamic_memory_examples()