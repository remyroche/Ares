"""
Examples of Enhanced Caching and Optimization Usage.

This module demonstrates how to use the enhanced caching and optimization
system throughout the codebase.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .enhanced_caching_system import (
    get_global_cache, CacheConfig, DataTypeOptimization, CacheStrategy,
    optimize_dataframe_default, optimize_numpy_array_default
)
from .optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    cache_dataframe_result, cache_numpy_result, optimize_heavy_computation,
    memory_aware, optimize_all_dataframes, optimize_all_arrays
)
from .integrated_hardware_manager import (
    get_integrated_hardware_manager, process_market_data,
    process_ml_training_data, process_backtesting_data
)

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_performance
)

# Example 1: Basic DataFrame Optimization
def example_dataframe_optimization():
    """Example of automatic DataFrame optimization."""
    tprint_info("=== DataFrame Optimization Example ===")
    
    # Create a large DataFrame with inefficient data types
    data = {
        'id': np.arange(1000000, dtype=np.int64),  # int64 -> int32
        'price': np.random.uniform(0, 1000, 1000000).astype(np.float64),  # float64 -> float32
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000000),  # object -> category
        'active': np.random.choice([True, False], 1000000)
    }
    
    df = pd.DataFrame(data)
    original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    tprint_info(f"Original DataFrame memory: {original_memory:.2f} MB")
    
    # Optimize with default settings
    optimized_df = optimize_dataframe_default(df)
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    tprint_success(f"Optimized DataFrame memory: {optimized_memory:.2f} MB")
    tprint_success(f"Memory saved: {original_memory - optimized_memory:.2f} MB ({(original_memory - optimized_memory) / original_memory * 100:.1f}%)")

# Example 2: Cached Function with Optimization
@smart_cache(ttl=3600)  # Cache for 1 hour
@auto_optimize(optimize_inputs=True, optimize_outputs=True)
def expensive_calculation(data: pd.DataFrame, param1: float, param2: int) -> pd.DataFrame:
    """Example expensive calculation that benefits from caching and optimization."""
    # Simulate expensive computation
    time.sleep(0.1)
    
    # Process data
    result = data.copy()
    result['calculated'] = result['price'] * param1 + param2
    result['normalized'] = (result['calculated'] - result['calculated'].mean()) / result['calculated'].std()
    
    return result

def example_cached_calculation():
    """Example of cached calculation with optimization."""
    tprint_info("=== Cached Calculation Example ===")
    
    # Create test data
    data = pd.DataFrame({
        'price': np.random.uniform(0, 100, 10000),
        'volume': np.random.uniform(0, 1000, 10000)
    })
    
    # First call - will compute and cache
    start_time = time.time()
    result1 = expensive_calculation(data, 1.5, 10)
    first_call_time = time.time() - start_time
    
    # Second call - will use cache
    start_time = time.time()
    result2 = expensive_calculation(data, 1.5, 10)
    second_call_time = time.time() - start_time
    
    tprint_success(f"First call (computation): {first_call_time:.3f}s")
    tprint_success(f"Second call (cached): {second_call_time:.3f}s")
    tprint_success(f"Speedup: {first_call_time / second_call_time:.1f}x")

# Example 3: Memory-Efficient Processing
@memory_efficient(memory_threshold_mb=50.0, auto_cleanup=True)
def process_large_dataset(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Example of memory-efficient processing."""
    results = {}
    
    for key, df in data.items():
        # Process each DataFrame
        processed_df = df.copy()
        processed_df['processed'] = processed_df['price'] * 2
        processed_df = processed_df.dropna()
        
        results[key] = processed_df
    
    return results

def example_memory_efficient_processing():
    """Example of memory-efficient processing."""
    tprint_info("=== Memory-Efficient Processing Example ===")
    
    # Create large dataset
    large_data = {}
    for i in range(5):
        large_data[f'dataset_{i}'] = pd.DataFrame({
            'price': np.random.uniform(0, 100, 50000),
            'volume': np.random.uniform(0, 1000, 50000)
        })
    
    # Process with memory efficiency
    results = process_large_dataset(large_data)
    
    tprint_success(f"Processed {len(results)} datasets efficiently")

# Example 4: Performance Tracking
@performance_tracked(log_performance=True, track_memory=True)
def tracked_operation(data: np.ndarray) -> np.ndarray:
    """Example operation with performance tracking."""
    # Simulate some computation
    result = np.dot(data.T, data)
    result = np.sqrt(result)
    return result

def example_performance_tracking():
    """Example of performance tracking."""
    tprint_info("=== Performance Tracking Example ===")
    
    # Create test data
    data = np.random.rand(1000, 1000)
    
    # Execute with tracking
    result = tracked_operation(data)
    
    tprint_success("Performance tracking completed - check logs for details")

# Example 5: Integrated Hardware Management
def example_integrated_processing():
    """Example of integrated hardware management."""
    tprint_info("=== Integrated Hardware Management Example ===")
    
    # Get integrated manager
    manager = get_integrated_hardware_manager()
    
    # Process different types of data
    market_data = {
        'prices': pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'price': [50000.0, 3000.0, 0.5],
            'volume': [1000000, 2000000, 5000000]
        }),
        'features': np.random.rand(100, 50)
    }
    
    # Process with automatic optimization
    optimized_market_data = process_market_data(market_data)
    
    # Process ML training data
    ml_data = np.random.rand(10000, 100)
    optimized_ml_data = process_ml_training_data(ml_data)
    
    # Get optimization report
    report = manager.get_optimization_report()
    
    tprint_success("Integrated processing completed")
    tprint_success(f"Cache hit rate: {report['cache_statistics']['hit_rate']:.2%}")
    tprint_success(f"Memory optimizations: {report['performance_metrics']['optimizations_applied']}")

# Example 6: Custom Cache Configuration
def example_custom_cache_config():
    """Example of custom cache configuration."""
    tprint_info("=== Custom Cache Configuration Example ===")
    
    # Create custom cache configuration
    custom_config = CacheConfig(
        max_memory_mb=1024.0,  # 1GB cache
        max_items=50000,
        strategy=CacheStrategy.LFU,  # Least Frequently Used
        data_type_optimization=DataTypeOptimization.MAXIMUM,
        enable_compression=True,
        compression_threshold_mb=0.5,
        ttl_seconds=7200  # 2 hours
    )
    
    # Get cache with custom config
    cache = get_global_cache(custom_config)
    
    # Use cache
    large_data = np.random.rand(10000, 1000)
    cache.put('large_array', large_data)
    
    # Retrieve from cache
    retrieved_data = cache.get('large_array')
    
    tprint_success("Custom cache configuration example completed")
    tprint_success(f"Cache statistics: {cache.get_statistics()}")

# Example 7: Batch Processing with Optimization
@cache_dataframe_result(ttl=1800)  # Cache for 30 minutes
def process_batch_data(batch_id: str, data: pd.DataFrame) -> Dict[str, Any]:
    """Process batch data with caching."""
    # Simulate batch processing
    results = {
        'batch_id': batch_id,
        'row_count': len(data),
        'mean_price': data['price'].mean(),
        'std_price': data['price'].std(),
        'processed_data': data.copy()
    }
    
    return results

def example_batch_processing():
    """Example of batch processing with optimization."""
    tprint_info("=== Batch Processing Example ===")
    
    # Create multiple batches
    batches = []
    for i in range(10):
        batch_data = pd.DataFrame({
            'price': np.random.uniform(0, 100, 1000),
            'volume': np.random.uniform(0, 1000, 1000)
        })
        batches.append((f'batch_{i}', batch_data))
    
    # Process batches
    results = []
    for batch_id, data in batches:
        result = process_batch_data(batch_id, data)
        results.append(result)
    
    tprint_success(f"Processed {len(results)} batches")
    
    # Process same batches again (should use cache)
    for batch_id, data in batches:
        result = process_batch_data(batch_id, data)
    
    tprint_success("Batch processing with caching completed")

# Example 8: Data Pipeline with Full Optimization
def example_optimized_pipeline():
    """Example of a fully optimized data pipeline."""
    tprint_info("=== Optimized Data Pipeline Example ===")
    
    # Step 1: Load and optimize data
    raw_data = {
        'prices': pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='1min'),
            'price': np.random.uniform(100, 200, 10000),
            'volume': np.random.uniform(1000, 10000, 10000)
        }),
        'features': np.random.rand(10000, 20)
    }
    
    # Step 2: Process with full optimization
    @memory_efficient()
    @auto_optimize()
    @performance_tracked()
    def pipeline_step(data: Dict[str, Any]) -> Dict[str, Any]:
        # Optimize all data
        optimized_data = optimize_all_dataframes(optimize_all_arrays(data))
        
        # Process data
        processed_data = {}
        for key, value in optimized_data.items():
            if isinstance(value, pd.DataFrame):
                processed_data[key] = value.describe()
            else:
                processed_data[key] = {
                    'mean': np.mean(value),
                    'std': np.std(value),
                    'shape': value.shape
                }
        
        return processed_data
    
    # Execute pipeline
    result = pipeline_step(raw_data)
    
    tprint_success("Optimized pipeline completed")
    tprint_success(f"Result keys: {list(result.keys())}")

def run_all_examples():
    """Run all optimization examples."""
    tprint_info("🚀 Running Enhanced Caching and Optimization Examples")
    
    try:
        example_dataframe_optimization()
        print()
        
        example_cached_calculation()
        print()
        
        example_memory_efficient_processing()
        print()
        
        example_performance_tracking()
        print()
        
        example_integrated_processing()
        print()
        
        example_custom_cache_config()
        print()
        
        example_batch_processing()
        print()
        
        example_optimized_pipeline()
        print()
        
        tprint_success("✅ All examples completed successfully!")
        
    except Exception as e:
        tprint_error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    run_all_examples()