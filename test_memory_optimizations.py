#!/usr/bin/env python3
"""
Test script to demonstrate the memory optimization improvements.
"""

import pandas as pd
import numpy as np
import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.feature_generation.utils.optimized_feature_pipeline import OptimizedFeaturePipeline, PipelineConfig

def test_memory_optimizations():
    """Test the memory optimization features."""
    
    print("🧪 Testing Memory Optimizations")
    print("=" * 50)
    
    # Create a large test dataset
    np.random.seed(42)
    n_rows = 50000  # Large dataset to test memory optimizations
    
    print(f"📊 Creating test dataset with {n_rows:,} rows...")
    
    # Create test data with various data types
    data = {
        'close': np.random.randn(n_rows).astype(np.float64),
        'volume': np.random.randint(1000, 100000, n_rows).astype(np.int64),
        'high': np.random.randn(n_rows).astype(np.float64),
        'low': np.random.randn(n_rows).astype(np.float64),
        'open': np.random.randn(n_rows).astype(np.float64),
        'day': np.random.randint(1, 32, n_rows).astype(np.int64),
        'hour': np.random.randint(0, 24, n_rows).astype(np.int64),
        'day_of_week': np.random.randint(0, 7, n_rows).astype(np.int64),
    }
    
    df = pd.DataFrame(data)
    original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"📈 Original dataset memory usage: {original_memory:.2f} MB")
    
    # Configure pipeline with memory optimizations
    config = PipelineConfig(
        enable_streaming_processing=True,
        streaming_chunk_size=5000,
        memory_threshold_mb=100.0,  # Low threshold to trigger streaming
        enable_adaptive_chunking=True,
        enable_aggressive_memory_cleanup=True,
        enable_advanced_data_type_optimization=True,
        max_memory_usage_mb=500.0
    )
    
    # Initialize pipeline
    pipeline = OptimizedFeaturePipeline(config)
    
    print("\n🔧 Testing Data Type Optimization...")
    start_time = time.time()
    
    # Test data type optimization
    optimized_df = pipeline._optimize_data_types(df)
    optimization_time = time.time() - start_time
    
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_saved = original_memory - optimized_memory
    reduction_percentage = (memory_saved / original_memory) * 100
    
    print(f"✅ Data type optimization completed in {optimization_time:.3f}s")
    print(f"💾 Memory saved: {memory_saved:.2f} MB ({reduction_percentage:.1f}% reduction)")
    print(f"📊 Optimized memory usage: {optimized_memory:.2f} MB")
    
    print("\n🌊 Testing Streaming Processing...")
    start_time = time.time()
    
    # Test streaming processing
    try:
        streaming_result = pipeline._streaming_feature_generation(optimized_df)
        streaming_time = time.time() - start_time
        
        print(f"✅ Streaming processing completed in {streaming_time:.3f}s")
        print(f"📊 Result shape: {streaming_result.shape}")
        
    except Exception as e:
        print(f"⚠️ Streaming processing test failed: {e}")
    
    print("\n🧹 Testing Memory Cleanup...")
    start_time = time.time()
    
    # Test memory cleanup
    initial_memory = pipeline._get_memory_usage()
    collected, freed = pipeline._enhanced_memory_cleanup()
    final_memory = pipeline._get_memory_usage()
    cleanup_time = time.time() - start_time
    
    print(f"✅ Memory cleanup completed in {cleanup_time:.3f}s")
    print(f"🧹 Objects collected: {collected}")
    print(f"💾 Memory freed: {freed:.2f} MB")
    print(f"📊 Memory before: {initial_memory:.2f} MB")
    print(f"📊 Memory after: {final_memory:.2f} MB")
    
    print("\n📈 Testing Memory Monitoring...")
    
    # Test memory monitoring
    memory_status = pipeline._monitor_memory_usage()
    current_memory = pipeline._get_memory_usage()
    
    print(f"📊 Current memory usage: {current_memory:.2f} MB")
    print(f"🔍 Memory monitoring triggered cleanup: {memory_status}")
    
    print("\n✅ Memory optimization tests completed!")
    print("=" * 50)
    
    return {
        'original_memory': original_memory,
        'optimized_memory': optimized_memory,
        'memory_saved': memory_saved,
        'reduction_percentage': reduction_percentage,
        'current_memory': current_memory
    }

if __name__ == "__main__":
    results = test_memory_optimizations()
    
    print("\n📊 Summary:")
    print(f"   Original memory: {results['original_memory']:.2f} MB")
    print(f"   Optimized memory: {results['optimized_memory']:.2f} MB")
    print(f"   Memory saved: {results['memory_saved']:.2f} MB")
    print(f"   Reduction: {results['reduction_percentage']:.1f}%")
    print(f"   Current memory: {results['current_memory']:.2f} MB")
