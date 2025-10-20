"""
Examples of Advanced Memory Management with Garbage Collection and Chunking.

This module demonstrates how to use the advanced memory management system
with efficient garbage collection, chunking, and memory optimization.
"""

import pandas as pd
import numpy as np
import time
import gc
from typing import Dict, List, Any, Optional

from .advanced_memory_manager import (
    get_advanced_memory_manager, memory_efficient_processing,
    chunked_processing, track_memory_usage, MemoryConfig
)
from .memory_optimized_decorators import (
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_large_dataframes, optimize_large_arrays, optimize_memory_intensive,
    force_garbage_collection, cleanup_all_memory, get_memory_optimization_stats
)
from .enhanced_caching_system import get_global_cache

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_performance
)

# Example 1: Basic Memory Optimization
def example_basic_memory_optimization():
    """Example of basic memory optimization with garbage collection."""
    tprint_info("=== Basic Memory Optimization Example ===")
    
    # Create large dataset
    large_df = pd.DataFrame({
        'id': np.arange(100000, dtype=np.int64),
        'price': np.random.uniform(0, 1000, 100000).astype(np.float64),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100000)
    })
    
    original_memory = large_df.memory_usage(deep=True).sum() / (1024 * 1024)
    tprint_info(f"Original DataFrame memory: {original_memory:.2f} MB")
    
    # Optimize with memory management
    @memory_optimized(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_aggressive_gc=True,
        log_memory_usage=True
    )
    def process_dataframe(df):
        # Simulate processing
        result = df.copy()
        result['processed'] = result['price'] * 2
        result = result[result['processed'] > 100]
        return result
    
    # Process with optimization
    optimized_df = process_dataframe(large_df)
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    tprint_success(f"Optimized DataFrame memory: {optimized_memory:.2f} MB")
    tprint_success(f"Memory saved: {original_memory - optimized_memory:.2f} MB")

# Example 2: Chunked Processing
def example_chunked_processing():
    """Example of chunked processing for large datasets."""
    tprint_info("=== Chunked Processing Example ===")
    
    # Create very large dataset
    very_large_df = pd.DataFrame({
        'id': np.arange(1000000, dtype=np.int64),
        'value': np.random.uniform(0, 100, 1000000).astype(np.float64),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000000)
    })
    
    original_memory = very_large_df.memory_usage(deep=True).sum() / (1024 * 1024)
    tprint_info(f"Original DataFrame memory: {original_memory:.2f} MB")
    
    # Process with chunking
    @chunked_processing_auto(
        chunk_size_mb=50.0,
        chunking_mode='memory_aware',
        combine_results=True
    )
    def process_large_dataframe(df):
        # Simulate expensive processing
        time.sleep(0.01)  # Simulate processing time
        
        result = df.copy()
        result['processed'] = result['value'] * np.sin(result['id'] / 1000)
        result = result[result['processed'] > 0]
        
        return result
    
    # Process with chunking
    start_time = time.time()
    processed_df = process_large_dataframe(very_large_df)
    processing_time = time.time() - start_time
    
    tprint_success(f"Processed in {processing_time:.2f} seconds")
    tprint_success(f"Result shape: {processed_df.shape}")

# Example 3: Garbage Collection Optimization
def example_gc_optimization():
    """Example of garbage collection optimization."""
    tprint_info("=== Garbage Collection Optimization Example ===")
    
    # Function without GC optimization
    def process_without_gc(data):
        results = []
        for i in range(10):
            # Create large temporary objects
            temp_array = np.random.rand(10000, 1000)
            temp_df = pd.DataFrame(temp_array)
            result = temp_df.describe()
            results.append(result)
        return results
    
    # Function with GC optimization
    @gc_optimized(gc_after_function=True, gc_after_chunks=True)
    def process_with_gc(data):
        results = []
        for i in range(10):
            # Create large temporary objects
            temp_array = np.random.rand(10000, 1000)
            temp_df = pd.DataFrame(temp_array)
            result = temp_df.describe()
            results.append(result)
        return results
    
    # Test without GC optimization
    start_time = time.time()
    start_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    results1 = process_without_gc(None)
    end_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    time1 = time.time() - start_time
    memory_delta1 = end_memory - start_memory
    
    # Force cleanup
    cleanup_all_memory()
    
    # Test with GC optimization
    start_time = time.time()
    start_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    results2 = process_with_gc(None)
    end_memory = get_memory_optimization_stats()['memory_stats']['used_mb']
    time2 = time.time() - start_time
    memory_delta2 = end_memory - start_memory
    
    tprint_success(f"Without GC: {time1:.2f}s, Memory delta: {memory_delta1:+.1f}MB")
    tprint_success(f"With GC: {time2:.2f}s, Memory delta: {memory_delta2:+.1f}MB")
    tprint_success(f"Memory improvement: {memory_delta1 - memory_delta2:.1f}MB")

# Example 4: Comprehensive Memory Optimization
def example_comprehensive_optimization():
    """Example of comprehensive memory optimization."""
    tprint_info("=== Comprehensive Memory Optimization Example ===")
    
    # Create large dataset
    large_data = {
        'prices': pd.DataFrame({
            'symbol': ['BTCUSDT'] * 100000,
            'price': np.random.uniform(10000, 70000, 100000).astype(np.float64),
            'volume': np.random.uniform(1000, 100000, 100000).astype(np.float64)
        }),
        'features': np.random.rand(100000, 50).astype(np.float64)
    }
    
    # Process with comprehensive optimization
    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.MAXIMUM,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True,
        enable_weak_refs=True
    )
    def process_complex_data(data):
        # Process prices
        prices = data['prices'].copy()
        prices['price_change'] = prices['price'].pct_change()
        prices['volume_ma'] = prices['volume'].rolling(10).mean()
        prices = prices.dropna()
        
        # Process features
        features = data['features'].copy()
        features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
        
        return {
            'processed_prices': prices,
            'processed_features': features_normalized,
            'summary': {
                'price_mean': prices['price'].mean(),
                'volume_mean': prices['volume'].mean(),
                'feature_std': features_normalized.std()
            }
        }
    
    # Process data
    start_time = time.time()
    result = process_complex_data(large_data)
    processing_time = time.time() - start_time
    
    tprint_success(f"Comprehensive processing completed in {processing_time:.2f} seconds")
    tprint_success(f"Processed prices shape: {result['processed_prices'].shape}")
    tprint_success(f"Processed features shape: {result['processed_features'].shape}")

# Example 5: Memory Pool Usage
def example_memory_pool_usage():
    """Example of memory pool usage for object reuse."""
    tprint_info("=== Memory Pool Usage Example ===")
    
    # Get memory manager
    memory_manager = get_advanced_memory_manager()
    memory_pool = memory_manager.get_memory_pool()
    
    if memory_pool:
        # Create arrays using memory pool
        arrays = []
        for i in range(5):
            arr = memory_pool.get_numpy_array((1000, 1000), np.float32)
            arr.fill(i)  # Fill with data
            arrays.append(arr)
        
        tprint_success(f"Created {len(arrays)} arrays using memory pool")
        
        # Return arrays to pool
        for arr in arrays:
            memory_pool.return_numpy_array(arr)
        
        # Get pool statistics
        pool_stats = memory_pool.get_stats()
        tprint_success(f"Memory pool stats: {pool_stats}")
    else:
        tprint_info("Memory pool not available")

# Example 6: Weak Reference Management
def example_weak_reference_management():
    """Example of weak reference management for large objects."""
    tprint_info("=== Weak Reference Management Example ===")
    
    # Get memory manager
    memory_manager = get_advanced_memory_manager()
    
    # Create large objects
    large_objects = []
    for i in range(10):
        large_obj = np.random.rand(10000, 1000)
        weak_ref = memory_manager.track_object(large_obj, 
                                             callback=lambda ref: tprint_debug(f"Object {id(ref)} cleaned up"))
        large_objects.append((large_obj, weak_ref))
    
    tprint_success(f"Created {len(large_objects)} large objects with weak references")
    
    # Get weak reference stats
    weak_stats = memory_manager.weak_ref_manager.get_stats()
    tprint_success(f"Weak reference stats: {weak_stats}")
    
    # Delete some objects
    for i in range(5):
        del large_objects[i][0]  # Delete the object but keep weak reference
    
    # Force garbage collection
    force_garbage_collection()
    
    # Check stats after cleanup
    weak_stats_after = memory_manager.weak_ref_manager.get_stats()
    tprint_success(f"Weak reference stats after cleanup: {weak_stats_after}")

# Example 7: Memory Pressure Detection
def example_memory_pressure_detection():
    """Example of memory pressure detection and adaptive cleanup."""
    tprint_info("=== Memory Pressure Detection Example ===")
    
    # Get memory manager
    memory_manager = get_advanced_memory_manager()
    
    # Create memory pressure by allocating large objects
    large_objects = []
    for i in range(20):
        large_obj = np.random.rand(5000, 1000)
        large_objects.append(large_obj)
        
        # Check memory pressure
        stats = memory_manager.get_memory_stats()
        tprint_info(f"Allocated object {i+1}, Memory: {stats.used_memory_mb:.1f}MB, "
                   f"Pressure: {stats.pressure_level.value}")
        
        if stats.pressure_level.value in ['high', 'critical']:
            tprint_warning("High memory pressure detected!")
            break
    
    # Clean up
    del large_objects
    cleanup_all_memory()
    
    # Check final stats
    final_stats = memory_manager.get_memory_stats()
    tprint_success(f"Final memory stats: {final_stats.used_memory_mb:.1f}MB, "
                  f"Pressure: {final_stats.pressure_level.value}")

# Example 8: Streaming Data Processing
def example_streaming_processing():
    """Example of streaming data processing with memory optimization."""
    tprint_info("=== Streaming Data Processing Example ===")
    
    # Simulate streaming data
    def data_stream():
        for i in range(100):
            yield pd.DataFrame({
                'id': np.arange(i * 1000, (i + 1) * 1000),
                'value': np.random.rand(1000),
                'timestamp': pd.Timestamp.now()
            })
    
    # Process streaming data with memory optimization
    @memory_optimized(
        optimization_level=MemoryOptimizationLevel.MODERATE,
        enable_chunking=True,
        chunking_mode='streaming',
        enable_aggressive_gc=True
    )
    def process_streaming_data(data_chunk):
        # Process chunk
        result = data_chunk.copy()
        result['processed'] = result['value'] * 2
        result = result[result['processed'] > 0.5]
        return result
    
    # Process streaming data
    processed_chunks = []
    for chunk in data_stream():
        processed_chunk = process_streaming_data(chunk)
        processed_chunks.append(processed_chunk)
        
        # Log progress
        if len(processed_chunks) % 10 == 0:
            tprint_info(f"Processed {len(processed_chunks)} chunks")
    
    # Combine results
    final_result = pd.concat(processed_chunks, ignore_index=True)
    tprint_success(f"Streaming processing completed: {len(final_result)} total records")

# Example 9: Memory-Aware Data Pipeline
def example_memory_aware_pipeline():
    """Example of a memory-aware data processing pipeline."""
    tprint_info("=== Memory-Aware Data Pipeline Example ===")
    
    # Create pipeline with memory awareness
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def load_data():
        return pd.DataFrame({
            'id': np.arange(50000, dtype=np.int64),
            'price': np.random.uniform(0, 1000, 50000).astype(np.float64),
            'volume': np.random.uniform(0, 10000, 50000).astype(np.float64)
        })
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def clean_data(df):
        return df.dropna().copy()
    
    @chunked_processing_auto(chunk_size_mb=25.0)
    def process_data(df):
        result = df.copy()
        result['price_change'] = result['price'].pct_change()
        result['volume_ma'] = result['volume'].rolling(5).mean()
        return result.dropna()
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def aggregate_data(df):
        return df.groupby('id').agg({
            'price': 'mean',
            'volume': 'sum',
            'price_change': 'std'
        }).reset_index()
    
    # Execute pipeline
    with get_advanced_memory_manager().memory_context("data_pipeline"):
        # Load data
        raw_data = load_data()
        tprint_info(f"Loaded data: {raw_data.shape}")
        
        # Clean data
        clean_data_result = clean_data(raw_data)
        tprint_info(f"Cleaned data: {clean_data_result.shape}")
        
        # Process data
        processed_data = process_data(clean_data_result)
        tprint_info(f"Processed data: {processed_data.shape}")
        
        # Aggregate data
        final_result = aggregate_data(processed_data)
        tprint_info(f"Final result: {final_result.shape}")
    
    tprint_success("Memory-aware pipeline completed successfully")

def run_all_memory_examples():
    """Run all memory optimization examples."""
    tprint_info("🚀 Running Advanced Memory Management Examples")
    
    try:
        example_basic_memory_optimization()
        print()
        
        example_chunked_processing()
        print()
        
        example_gc_optimization()
        print()
        
        example_comprehensive_optimization()
        print()
        
        example_memory_pool_usage()
        print()
        
        example_weak_reference_management()
        print()
        
        example_memory_pressure_detection()
        print()
        
        example_streaming_processing()
        print()
        
        example_memory_aware_pipeline()
        print()
        
        tprint_success("✅ All memory optimization examples completed successfully!")
        
    except Exception as e:
        tprint_error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    run_all_memory_examples()