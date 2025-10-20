"""
Enhanced features_common Demo with Hardware Optimization

This example demonstrates the enhanced features_common system with
hardware utility integration for maximum performance optimization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import enhanced components
from ..vectorbt_extensions.unified_manager import get_unified_vectorbt_manager
from ..transforms.vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from ..transforms.base_scaler import BaseScaler
from ..mixins.optimization_mixin import OptimizationMixin
from ..vectorbt_extensions.performance_monitor import get_performance_monitor
from ..vectorbt_extensions.optimization_engine import get_optimization_engine
from ..vectorbt_extensions.gpu_accelerator import get_gpu_accelerator

logger = logging.getLogger(__name__)

def demonstrate_enhanced_features():
    """Demonstrate the enhanced features_common with hardware optimization."""
    print("🚀 Enhanced features_common with Hardware Optimization")
    print("=" * 60)
    
    # Generate test data
    print("\n📊 Generating test data...")
    test_data = generate_test_data()
    print(f"Generated DataFrame with shape: {test_data.shape}")
    
    # Demonstrate enhanced VectorBT operations
    print("\n🔧 Demonstrating enhanced VectorBT operations...")
    demonstrate_enhanced_vectorbt_operations(test_data)
    
    # Demonstrate enhanced scaling operations
    print("\n📏 Demonstrating enhanced scaling operations...")
    demonstrate_enhanced_scaling_operations(test_data)
    
    # Demonstrate hardware optimization mixin
    print("\n🧠 Demonstrating hardware optimization mixin...")
    demonstrate_hardware_optimization_mixin(test_data)
    
    # Demonstrate performance monitoring
    print("\n📈 Demonstrating enhanced performance monitoring...")
    demonstrate_enhanced_performance_monitoring()
    
    # Demonstrate optimization engine
    print("\n⚙️ Demonstrating optimization engine...")
    demonstrate_optimization_engine(test_data)
    
    # Demonstrate GPU accelerator
    print("\n🚀 Demonstrating GPU accelerator...")
    demonstrate_gpu_accelerator(test_data)
    
    print("\n✅ Enhanced features demonstration completed!")

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

def demonstrate_enhanced_vectorbt_operations(data: pd.DataFrame) -> None:
    """Demonstrate enhanced VectorBT operations with hardware optimization."""
    # Get enhanced manager
    manager = get_unified_vectorbt_manager()
    
    # Enable hardware optimization
    manager.enable_hardware_optimization()
    
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
        
        # Execute operation
        result = manager.execute_operation(op_name, test_column, **kwargs)
        
        print(f"    ✅ {op_name} completed")
        print(f"    📊 Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        # Show enhanced stats
        stats = manager.get_performance_summary()
        print(f"    🖥️  Hardware operations: {stats.get('hardware_optimization', {}).get('hardware_operations', 0)}")
        print(f"    💾 Memory optimizations: {stats.get('hardware_optimization', {}).get('memory_optimizations', 0)}")
        print(f"    🚀 GPU operations: {stats.get('hardware_optimization', {}).get('gpu_operations', 0)}")

def demonstrate_enhanced_scaling_operations(data: pd.DataFrame) -> None:
    """Demonstrate enhanced scaling operations with hardware optimization."""
    # Test different scaling methods
    scaling_methods = ['zscore', 'minmax', 'robust', 'quantile']
    
    for method in scaling_methods:
        print(f"\n  📏 Testing {method} scaling...")
        
        # Create enhanced scaler with hardware optimization
        scaler = VectorBTScaler(
            method=method,
            enable_hardware_optimization=True,
            memory_efficient=True
        )
        
        # Test on first column
        test_column = data.iloc[:, 0]
        
        # Fit and transform
        scaled_data = scaler.fit_transform(test_column)
        
        print(f"    ✅ {method} scaling completed")
        print(f"    📊 Scaled data stats: mean={scaled_data.mean():.4f}, std={scaled_data.std():.4f}")
        
        # Show hardware stats
        hardware_stats = scaler.get_hardware_stats()
        print(f"    🖥️  Hardware operations: {hardware_stats.get('hardware_operations', 0)}")
        print(f"    💾 Memory optimizations: {hardware_stats.get('memory_optimizations', 0)}")

def demonstrate_hardware_optimization_mixin(data: pd.DataFrame) -> None:
    """Demonstrate hardware optimization mixin capabilities."""
    print(f"\n  🧠 Testing hardware optimization mixin...")
    
    # Create a class with hardware optimization mixin
    class TestClass(OptimizationMixin):
        def __init__(self):
            super().__init__()
        
        def test_operation(self, data):
            return data.rolling(20).mean()
    
    # Create instance
    test_instance = TestClass()
    
    # Enable hardware optimization
    test_instance.enable_hardware_optimization()
    
    # Test operation with hardware optimization
    test_column = data.iloc[:, 0]
    result = test_instance.auto_optimize_operation(
        test_instance.test_operation, 
        test_column, 
        operation_type='data_processing'
    )
    
    print(f"    ✅ Hardware-optimized operation completed")
    print(f"    📊 Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    
    # Show hardware stats
    hardware_stats = test_instance.get_hardware_stats()
    print(f"    🖥️  Hardware operations: {hardware_stats.get('hardware_operations', 0)}")
    print(f"    💾 Memory optimizations: {hardware_stats.get('memory_optimizations', 0)}")
    print(f"    🧠 Adaptive decisions: {hardware_stats.get('adaptive_decisions', 0)}")

def demonstrate_enhanced_performance_monitoring() -> None:
    """Demonstrate enhanced performance monitoring capabilities."""
    print(f"\n  📈 Testing enhanced performance monitoring...")
    
    # Get performance monitor
    monitor = get_performance_monitor()
    
    # Simulate some operations
    operations = [
        ('rolling_mean', 1000, 'hardware_optimized', True),
        ('rolling_std', 2000, 'standard', False),
        ('scaling', 1500, 'memory_optimized', True),
        ('ranking', 800, 'adaptive', True)
    ]
    
    for op_name, data_size, strategy, hardware_optimized in operations:
        # Start monitoring
        operation_id = monitor.start_monitoring(
            operation=op_name,
            data_size=data_size,
            optimization_strategy=strategy,
            hardware_optimized=hardware_optimized
        )
        
        # Simulate operation
        import time
        time.sleep(0.1)  # Simulate processing time
        
        # Stop monitoring
        monitor.stop_monitoring(operation_id, success=True)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"    📊 Total operations: {summary.total_operations}")
    print(f"    ✅ Successful operations: {summary.successful_operations}")
    print(f"    📈 Average duration: {summary.avg_duration:.4f}s")
    print(f"    🖥️  Hardware optimization rate: {summary.hardware_optimization_rate:.2%}")
    print(f"    📊 Performance trend: {summary.performance_trend}")
    
    # Show recommendations
    if summary.recommendations:
        print(f"    💡 Recommendations:")
        for i, rec in enumerate(summary.recommendations[:3], 1):
            print(f"      {i}. {rec}")

def demonstrate_optimization_engine(data: pd.DataFrame) -> None:
    """Demonstrate optimization engine capabilities."""
    print(f"\n  ⚙️ Testing optimization engine...")
    
    # Get optimization engine
    engine = get_optimization_engine()
    
    # Test different operations
    operations = ['rolling_mean', 'scaling', 'ranking']
    
    for operation in operations:
        print(f"\n    🔧 Testing {operation} optimization...")
        
        # Test on first column
        test_column = data.iloc[:, 0]
        
        # Optimize operation
        result = engine.optimize_operation(operation, test_column, workload_type='data_processing')
        
        print(f"      ✅ {operation} optimization completed")
        print(f"      📊 Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    
    # Get optimization summary
    summary = engine.get_optimization_summary()
    print(f"\n    📊 Optimization Summary:")
    print(f"      Total optimizations: {summary['total_optimizations']}")
    print(f"      Hardware optimization rate: {summary['hardware_optimization_rate']:.2%}")
    print(f"      Memory optimization rate: {summary['memory_optimization_rate']:.2%}")
    print(f"      Hardware available: {summary['hardware_available']}")
    
    # Show recommendations
    if summary['recommendations']:
        print(f"      💡 Recommendations:")
        for i, rec in enumerate(summary['recommendations'][:3], 1):
            print(f"        {i}. {rec}")

def demonstrate_gpu_accelerator(data: pd.DataFrame) -> None:
    """Demonstrate GPU accelerator capabilities."""
    print(f"\n  🚀 Testing GPU accelerator...")
    
    # Get GPU accelerator
    accelerator = get_gpu_accelerator()
    
    # Test different operations
    operations = ['rolling_mean', 'scaling', 'ranking', 'matrix_multiplication']
    
    for operation in operations:
        print(f"\n    🔧 Testing {operation} GPU acceleration...")
        
        # Test on first column
        test_column = data.iloc[:, 0]
        
        # Accelerate operation
        result = accelerator.accelerate_operation(operation, test_column, operation_type='data_processing')
        
        print(f"      ✅ {operation} acceleration completed")
        print(f"      📊 Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    
    # Get acceleration summary
    summary = accelerator.get_acceleration_summary()
    print(f"\n    📊 Acceleration Summary:")
    print(f"      Total operations: {summary['total_operations']}")
    print(f"      GPU acceleration rate: {summary['gpu_acceleration_rate']:.2%}")
    print(f"      GPU available: {summary['gpu_available']}")
    print(f"      Hardware available: {summary['hardware_available']}")
    
    # Show recommendations
    if summary['recommendations']:
        print(f"      💡 Recommendations:")
        for i, rec in enumerate(summary['recommendations'][:3], 1):
            print(f"        {i}. {rec}")

def demonstrate_batch_operations(data: pd.DataFrame) -> None:
    """Demonstrate enhanced batch operations."""
    print(f"\n  🔄 Testing enhanced batch operations...")
    
    # Create batch scaler
    batch_scaler = VectorBTBatchScaler(
        method='zscore',
        enable_hardware_optimization=True,
        memory_efficient=True
    )
    
    # Test batch scaling
    scaled_data = batch_scaler.fit_transform(data)
    
    print(f"    ✅ Batch scaling completed")
    print(f"    📊 Scaled data shape: {scaled_data.shape}")
    
    # Show hardware stats
    hardware_stats = batch_scaler.get_hardware_stats()
    print(f"    🖥️  Hardware operations: {hardware_stats.get('hardware_operations', 0)}")
    print(f"    💾 Memory optimizations: {hardware_stats.get('memory_optimizations', 0)}")
    print(f"    🔄 Chunked operations: {hardware_stats.get('chunked_operations', 0)}")

def main():
    """Main demonstration function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run main demonstration
        demonstrate_enhanced_features()
        
        # Run additional demonstrations
        print("\n🔄 Additional Demonstrations")
        print("-" * 40)
        
        # Generate larger dataset for batch operations
        large_data = generate_test_data(rows=50000, columns=20)
        demonstrate_batch_operations(large_data)
        
        print("\n🎉 All enhanced features demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Demonstration failed")

if __name__ == "__main__":
    main()