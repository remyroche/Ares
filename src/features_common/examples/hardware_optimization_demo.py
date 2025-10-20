"""
Hardware Optimization Demo for features_common.

This module demonstrates the integration of hardware utilities with features_common
for optimizing demanding processes.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Import hardware-optimized components
from ..vectorbt_extensions.hardware_optimized_manager import get_hardware_optimized_vectorbt_manager
from ..transforms.hardware_optimized_scaler import (
    create_hardware_optimized_scaler,
    create_hardware_optimized_batch_scaler
)
from ..optimization.performance_benchmark import get_performance_benchmark, run_quick_benchmark

logger = logging.getLogger(__name__)

def demonstrate_hardware_optimization():
    """Demonstrate hardware optimization capabilities."""
    print("🚀 Hardware Optimization Demo for features_common")
    print("=" * 60)
    
    # Generate test data
    print("\n📊 Generating test data...")
    test_data = generate_test_data()
    print(f"Generated DataFrame with shape: {test_data.shape}")
    
    # Demonstrate VectorBT operations
    print("\n🔧 Demonstrating VectorBT operations with hardware optimization...")
    demonstrate_vectorbt_operations(test_data)
    
    # Demonstrate scaling operations
    print("\n📏 Demonstrating scaling operations with hardware optimization...")
    demonstrate_scaling_operations(test_data)
    
    # Demonstrate batch operations
    print("\n🔄 Demonstrating batch operations with hardware optimization...")
    demonstrate_batch_operations(test_data)
    
    # Run performance benchmark
    print("\n📈 Running performance benchmark...")
    run_performance_benchmark()
    
    print("\n✅ Hardware optimization demo completed!")

def generate_test_data(rows: int = 10000, columns: int = 10) -> pd.DataFrame:
    """Generate test data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic financial data
    data = {}
    for i in range(columns):
        # Generate price-like data with trend and volatility
        trend = np.linspace(100, 120, rows) + np.random.randn(rows) * 2
        data[f'price_{i}'] = trend
        
        # Generate volume data
        data[f'volume_{i}'] = np.random.exponential(1000, rows)
        
        # Generate returns
        data[f'returns_{i}'] = np.random.normal(0, 0.02, rows)
    
    return pd.DataFrame(data)

def demonstrate_vectorbt_operations(data: pd.DataFrame) -> None:
    """Demonstrate VectorBT operations with hardware optimization."""
    # Get hardware-optimized manager
    manager = get_hardware_optimized_vectorbt_manager()
    
    # Test different operations
    operations = [
        ('rolling_mean', {'window': 20}),
        ('rolling_std', {'window': 20}),
        ('rolling_var', {'window': 20}),
        ('rolling_min', {'window': 20}),
        ('rolling_max', {'window': 20}),
        ('rolling_sum', {'window': 20})
    ]
    
    for op_name, kwargs in operations:
        print(f"\n  🔧 Testing {op_name}...")
        
        # Test on first column
        test_column = data.iloc[:, 0]
        
        # Time the operation
        start_time = time.time()
        result = manager.execute_operation(op_name, test_column, **kwargs)
        execution_time = time.time() - start_time
        
        print(f"    ✅ {op_name} completed in {execution_time:.4f}s")
        print(f"    📊 Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        # Show hardware stats
        hardware_stats = manager.get_hardware_stats()
        print(f"    🖥️  Hardware operations: {hardware_stats.get('hardware_operations', 0)}")
        print(f"    💾 Memory optimizations: {hardware_stats.get('memory_optimizations', 0)}")
        print(f"    🚀 GPU operations: {hardware_stats.get('gpu_operations', 0)}")

def demonstrate_scaling_operations(data: pd.DataFrame) -> None:
    """Demonstrate scaling operations with hardware optimization."""
    # Test different scaling methods
    scaling_methods = ['zscore', 'minmax', 'robust', 'quantile']
    
    for method in scaling_methods:
        print(f"\n  📏 Testing {method} scaling...")
        
        # Create hardware-optimized scaler
        scaler = create_hardware_optimized_scaler(
            method=method,
            enable_hardware_optimization=True
        )
        
        # Test on first column
        test_column = data.iloc[:, 0]
        
        # Time the operation
        start_time = time.time()
        scaled_data = scaler.fit_transform(test_column)
        execution_time = time.time() - start_time
        
        print(f"    ✅ {method} scaling completed in {execution_time:.4f}s")
        print(f"    📊 Scaled data stats: mean={scaled_data.mean():.4f}, std={scaled_data.std():.4f}")
        
        # Show hardware stats
        hardware_stats = scaler.get_hardware_stats()
        print(f"    🖥️  Hardware operations: {hardware_stats.get('hardware_operations', 0)}")
        print(f"    💾 Memory optimizations: {hardware_stats.get('memory_optimizations', 0)}")

def demonstrate_batch_operations(data: pd.DataFrame) -> None:
    """Demonstrate batch operations with hardware optimization."""
    print(f"\n  🔄 Testing batch scaling on {data.shape[1]} columns...")
    
    # Create hardware-optimized batch scaler
    batch_scaler = create_hardware_optimized_batch_scaler(
        method='zscore',
        enable_hardware_optimization=True
    )
    
    # Time the batch operation
    start_time = time.time()
    scaled_data = batch_scaler.fit_transform(data)
    execution_time = time.time() - start_time
    
    print(f"    ✅ Batch scaling completed in {execution_time:.4f}s")
    print(f"    📊 Scaled data shape: {scaled_data.shape}")
    
    # Show hardware stats
    hardware_stats = batch_scaler.get_hardware_stats()
    print(f"    🖥️  Hardware operations: {hardware_stats.get('hardware_operations', 0)}")
    print(f"    💾 Memory optimizations: {hardware_stats.get('memory_optimizations', 0)}")
    print(f"    🔄 Chunked operations: {hardware_stats.get('chunked_operations', 0)}")

def run_performance_benchmark() -> None:
    """Run performance benchmark to compare optimization effectiveness."""
    print("\n  📈 Running performance benchmark...")
    
    try:
        # Run quick benchmark
        benchmark = get_performance_benchmark()
        report = benchmark.run_comprehensive_benchmark()
        
        # Display key results
        print(f"    📊 Total benchmarks: {report.summary_stats.get('total_benchmarks', 0)}")
        print(f"    ✅ Success rate: {report.summary_stats.get('success_rate', 0):.2%}")
        
        # Display hardware optimization effectiveness
        if 'hardware_stats' in report and 'speedup_factor' in report.hardware_stats:
            speedup = report.hardware_stats['speedup_factor']
            print(f"    🚀 Hardware optimization speedup: {speedup:.2f}x")
            print(f"    📈 Performance improvement: {report.hardware_stats.get('performance_improvement_percent', 0):.1f}%")
        
        # Display recommendations
        if report.recommendations:
            print(f"    💡 Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):  # Show first 3
                print(f"      {i}. {rec}")
        
        print(f"    📄 Full report saved to benchmark_results/")
        
    except Exception as e:
        print(f"    ⚠️  Benchmark failed: {e}")

def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("\n💾 Demonstrating memory optimization...")
    
    # Generate large dataset
    large_data = generate_test_data(rows=100000, columns=20)
    print(f"Generated large dataset: {large_data.shape}")
    
    # Test memory optimization
    from ..optimization.hardware_optimized_mixin import HardwareOptimizedMixin
    
    class MemoryDemoMixin(HardwareOptimizedMixin):
        def __init__(self):
            super().__init__()
            self._memory_optimization_enabled = True
    
    mixin = MemoryDemoMixin()
    
    # Test memory-efficient operation
    def memory_intensive_operation(data):
        # Simulate memory-intensive operation
        result = data.rolling(50).mean()
        result = result.rolling(20).std()
        return result
    
    print("  🔧 Testing memory-efficient operation...")
    start_time = time.time()
    result = mixin.memory_efficient_operation(memory_intensive_operation, large_data)
    execution_time = time.time() - start_time
    
    print(f"    ✅ Memory-efficient operation completed in {execution_time:.4f}s")
    print(f"    📊 Result shape: {result.shape}")
    
    # Show memory stats
    memory_stats = mixin.get_hardware_stats()
    print(f"    💾 Memory optimizations: {memory_stats.get('memory_optimizations', 0)}")
    print(f"    🔄 Chunked operations: {memory_stats.get('chunked_operations', 0)}")

def demonstrate_adaptive_optimization():
    """Demonstrate adaptive optimization capabilities."""
    print("\n🧠 Demonstrating adaptive optimization...")
    
    # Get hardware-optimized manager
    manager = get_hardware_optimized_vectorbt_manager()
    
    # Test with different data characteristics
    test_cases = [
        ("Small dataset", generate_test_data(1000, 5)),
        ("Medium dataset", generate_test_data(10000, 10)),
        ("Large dataset", generate_test_data(100000, 20))
    ]
    
    for case_name, test_data in test_cases:
        print(f"\n  📊 Testing {case_name}...")
        
        # Get optimal strategy
        strategy = manager.get_optimal_strategy('data_processing', test_data)
        print(f"    🎯 Optimal strategy: {strategy.get('optimization_target', 'unknown')}")
        print(f"    🖥️  Use GPU: {strategy.get('use_gpu', False)}")
        print(f"    📦 Batch size: {strategy.get('batch_size', 'N/A')}")
        
        # Test operation with adaptive optimization
        start_time = time.time()
        result = manager.rolling_mean(test_data.iloc[:, 0], window=20)
        execution_time = time.time() - start_time
        
        print(f"    ✅ Operation completed in {execution_time:.4f}s")
        
        # Show adaptive decisions
        hardware_stats = manager.get_hardware_stats()
        print(f"    🧠 Adaptive decisions: {hardware_stats.get('adaptive_decisions', 0)}")

def main():
    """Main demonstration function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run main demonstration
        demonstrate_hardware_optimization()
        
        # Run additional demonstrations
        demonstrate_memory_optimization()
        demonstrate_adaptive_optimization()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Demonstration failed")

if __name__ == "__main__":
    main()