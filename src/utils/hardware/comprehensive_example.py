"""
Comprehensive Example of Enhanced Hardware Optimizations.

This example demonstrates all the new hardware optimization features
with backward compatibility and real-world usage patterns.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List

# Import all hardware optimization modules
from .backward_compatibility import (
    gpu_vectorbt_optimization, cpu_optimized_feature_correlation,
    unified_memory_feature_processing, adaptive_feature_selection,
    gpu_accelerated, optimize_cpu_execution, unified_memory_optimized,
    adaptive_optimization, smart_cache, performance_tracked,
    comprehensive_memory_optimization, get_hardware_optimization_status
)

def demonstrate_gpu_vectorbt_optimization():
    """Demonstrate GPU-accelerated VectorBT optimization."""
    print("🎮 GPU VectorBT Optimization Demo")
    print("=" * 50)
    
    # Generate sample financial data
    np.random.seed(42)
    n_days = 1000
    n_assets = 10
    
    # Generate price returns
    returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
    price_data = np.cumprod(1 + returns, axis=0)
    
    # Generate features
    features = {
        'weights': np.random.dirichlet(np.ones(n_assets)),
        'risk_free_rate': 0.02,
        'lookback_period': 252
    }
    
    print(f"Data shape: {price_data.shape}")
    print(f"Memory usage: {price_data.nbytes / (1024**2):.1f} MB")
    
    # Time the optimization
    start_time = time.time()
    result = gpu_vectorbt_optimization(price_data, features)
    execution_time = time.time() - start_time
    
    print(f"Execution time: {execution_time:.3f}s")
    print(f"Results: {result}")
    print()

def demonstrate_cpu_optimization():
    """Demonstrate enhanced CPU optimization."""
    print("⚡ Enhanced CPU Optimization Demo")
    print("=" * 50)
    
    # Generate sample feature data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate correlated features
    base_features = np.random.randn(n_samples, 10)
    features = np.hstack([
        base_features,
        base_features @ np.random.randn(10, 20),
        np.random.randn(n_samples, 20)
    ])
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Memory usage: {features.nbytes / (1024**2):.1f} MB")
    
    # Time the correlation calculation
    start_time = time.time()
    correlation_matrix = cpu_optimized_feature_correlation(features)
    execution_time = time.time() - start_time
    
    print(f"Execution time: {execution_time:.3f}s")
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Memory usage: {correlation_matrix.nbytes / (1024**2):.1f} MB")
    print()

def demonstrate_unified_memory_optimization():
    """Demonstrate unified memory optimization."""
    print("🧠 Unified Memory Optimization Demo")
    print("=" * 50)
    
    # Generate large dataset
    np.random.seed(42)
    n_samples = 10000
    n_features = 100
    
    data = np.random.randn(n_samples, n_features).astype(np.float64)
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data type: {data.dtype}")
    print(f"Original memory usage: {data.nbytes / (1024**2):.1f} MB")
    
    # Apply unified memory optimization
    start_time = time.time()
    optimized_data = unified_memory_feature_processing(
        data, 
        operation_type='feature_selection', 
        component='gpu'
    )
    execution_time = time.time() - start_time
    
    print(f"Optimized data shape: {optimized_data.shape}")
    print(f"Optimized data type: {optimized_data.dtype}")
    print(f"Optimized memory usage: {optimized_data.nbytes / (1024**2):.1f} MB")
    print(f"Memory savings: {(data.nbytes - optimized_data.nbytes) / data.nbytes * 100:.1f}%")
    print(f"Execution time: {execution_time:.3f}s")
    print()

def demonstrate_adaptive_optimization():
    """Demonstrate adaptive optimization."""
    print("🧠 Adaptive Optimization Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(1000, 20)
    
    print(f"Data shape: {data.shape}")
    
    # Apply adaptive optimization
    start_time = time.time()
    optimized_data = adaptive_feature_selection(data, learn_from_execution=True)
    execution_time = time.time() - start_time
    
    print(f"Optimized data shape: {optimized_data.shape}")
    print(f"Execution time: {execution_time:.3f}s")
    print()

def demonstrate_decorators():
    """Demonstrate hardware optimization decorators."""
    print("🎯 Hardware Optimization Decorators Demo")
    print("=" * 50)
    
    # GPU-accelerated function
    @gpu_accelerated("matrix_multiplication")
    def matrix_multiply(A, B):
        return np.dot(A, B)
    
    # CPU-optimized function
    @optimize_cpu_execution("cpu_intensive")
    def feature_correlation(data):
        return np.corrcoef(data.T)
    
    # Unified memory optimized function
    @unified_memory_optimized('feature_selection', 'gpu')
    def process_features(data):
        return data * 2
    
    # Adaptive optimization function
    @adaptive_optimization(learn_from_execution=True)
    def adaptive_processing(data):
        return np.sum(data, axis=1)
    
    # Smart caching function
    @smart_cache(ttl=3600, max_size=100, compression=True)
    def expensive_calculation(data):
        time.sleep(0.1)  # Simulate expensive operation
        return np.sum(data)
    
    # Performance tracked function
    @performance_tracked(['execution_time', 'memory_usage', 'cpu_utilization'])
    def tracked_function(data):
        return np.mean(data, axis=0)
    
    # Comprehensive memory optimization function
    @comprehensive_memory_optimization(
        int64_to_int32=True,
        float64_to_float32=True,
        object_to_category=True,
        compression_ratio=0.7
    )
    def memory_optimized_function(data):
        return data * 2
    
    # Test data
    A = np.random.randn(100, 100).astype(np.float64)
    B = np.random.randn(100, 100).astype(np.float64)
    data = np.random.randn(1000, 50).astype(np.float64)
    
    print("Testing decorators...")
    
    # Test each decorator
    start_time = time.time()
    result1 = matrix_multiply(A, B)
    print(f"GPU matrix multiply: {time.time() - start_time:.3f}s")
    
    start_time = time.time()
    result2 = feature_correlation(data)
    print(f"CPU feature correlation: {time.time() - start_time:.3f}s")
    
    start_time = time.time()
    result3 = process_features(data)
    print(f"Unified memory processing: {time.time() - start_time:.3f}s")
    
    start_time = time.time()
    result4 = adaptive_processing(data)
    print(f"Adaptive processing: {time.time() - start_time:.3f}s")
    
    start_time = time.time()
    result5 = expensive_calculation(data)
    print(f"Smart cached calculation: {time.time() - start_time:.3f}s")
    
    start_time = time.time()
    result6 = tracked_function(data)
    print(f"Performance tracked function: {time.time() - start_time:.3f}s")
    
    start_time = time.time()
    result7 = memory_optimized_function(data)
    print(f"Memory optimized function: {time.time() - start_time:.3f}s")
    
    print()

def demonstrate_real_world_workflow():
    """Demonstrate a real-world financial analysis workflow."""
    print("💰 Real-World Financial Analysis Workflow")
    print("=" * 50)
    
    # Generate realistic financial data
    np.random.seed(42)
    n_days = 252 * 5  # 5 years of daily data
    n_assets = 20
    
    # Generate price data with realistic characteristics
    returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
    prices = np.cumprod(1 + returns, axis=0)
    
    print(f"Generated {n_days} days of data for {n_assets} assets")
    print(f"Data shape: {prices.shape}")
    print(f"Memory usage: {prices.nbytes / (1024**2):.1f} MB")
    
    # Step 1: GPU-accelerated portfolio analysis
    print("\n1. GPU-Accelerated Portfolio Analysis")
    start_time = time.time()
    
    portfolio_weights = np.random.dirichlet(np.ones(n_assets))
    portfolio_analysis = gpu_vectorbt_optimization(returns, {
        'weights': portfolio_weights,
        'risk_free_rate': 0.02
    })
    
    print(f"   Portfolio analysis completed in {time.time() - start_time:.3f}s")
    print(f"   Sharpe ratio: {portfolio_analysis.get('sharpe_ratio', 0):.3f}")
    
    # Step 2: CPU-optimized feature correlation analysis
    print("\n2. CPU-Optimized Feature Correlation Analysis")
    start_time = time.time()
    
    # Calculate technical indicators as features
    features = np.column_stack([
        returns,  # Returns
        np.roll(returns, 1, axis=0),  # Lagged returns
        np.roll(returns, 5, axis=0),  # 5-day lagged returns
        np.roll(returns, 20, axis=0),  # 20-day lagged returns
    ])
    
    correlation_matrix = cpu_optimized_feature_correlation(features)
    
    print(f"   Feature correlation completed in {time.time() - start_time:.3f}s")
    print(f"   Correlation matrix shape: {correlation_matrix.shape}")
    
    # Step 3: Unified memory optimization for large dataset
    print("\n3. Unified Memory Optimization")
    start_time = time.time()
    
    # Create large feature matrix
    large_features = np.random.randn(n_days, 100).astype(np.float64)
    optimized_features = unified_memory_feature_processing(
        large_features, 
        operation_type='feature_selection', 
        component='gpu'
    )
    
    print(f"   Memory optimization completed in {time.time() - start_time:.3f}s")
    print(f"   Memory savings: {(large_features.nbytes - optimized_features.nbytes) / large_features.nbytes * 100:.1f}%")
    
    # Step 4: Adaptive optimization for feature selection
    print("\n4. Adaptive Feature Selection")
    start_time = time.time()
    
    selected_features = adaptive_feature_selection(optimized_features, learn_from_execution=True)
    
    print(f"   Adaptive feature selection completed in {time.time() - start_time:.3f}s")
    print(f"   Selected features shape: {selected_features.shape}")
    
    print("\n✅ Real-world workflow completed successfully!")
    print()

def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("📊 Performance Monitoring Demo")
    print("=" * 50)
    
    # Get comprehensive hardware status
    status = get_hardware_optimization_status()
    
    print("Hardware Optimization Status:")
    print(f"  VectorBT GPU metrics: {status.get('vectorbt_gpu_metrics', {}).get('gpu_available', False)}")
    print(f"  Enhanced CPU metrics: {status.get('enhanced_cpu_metrics', {}).get('enhanced_allocations', 0)}")
    print(f"  Unified memory metrics: {status.get('unified_memory_metrics', {}).get('enhanced_allocations', 0)}")
    print(f"  Adaptive optimization metrics: {status.get('adaptive_optimization_metrics', {}).get('learning_enabled', False)}")
    
    print("\nDetailed Metrics:")
    for category, metrics in status.items():
        if isinstance(metrics, dict):
            print(f"  {category}:")
            for key, value in list(metrics.items())[:3]:  # Show first 3 items
                print(f"    {key}: {value}")
    
    print()

def main():
    """Run all demonstrations."""
    print("🚀 Enhanced Hardware Optimizations - Comprehensive Demo")
    print("=" * 70)
    print()
    
    try:
        # Run all demonstrations
        demonstrate_gpu_vectorbt_optimization()
        demonstrate_cpu_optimization()
        demonstrate_unified_memory_optimization()
        demonstrate_adaptive_optimization()
        demonstrate_decorators()
        demonstrate_real_world_workflow()
        demonstrate_performance_monitoring()
        
        print("🎉 All demonstrations completed successfully!")
        print("\nKey Benefits:")
        print("  ✅ GPU acceleration for VectorBT operations")
        print("  ✅ Enhanced CPU optimization with thermal management")
        print("  ✅ Unified memory architecture with cross-component sharing")
        print("  ✅ Adaptive optimization with machine learning")
        print("  ✅ Intelligent caching with compression")
        print("  ✅ Comprehensive performance monitoring")
        print("  ✅ Full backward compatibility")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()