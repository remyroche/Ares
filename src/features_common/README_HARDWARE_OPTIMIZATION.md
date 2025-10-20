# Hardware Optimization Integration for features_common

## Overview

This document describes the comprehensive integration of hardware utilities from `src/utils/hardware/` into the `features_common/` system to optimize demanding processes. The integration provides intelligent memory management, adaptive optimization, GPU acceleration, and performance monitoring.

## Key Features

### 🚀 Hardware-Optimized Components

- **HardwareOptimizedVectorBTManager**: Enhanced VectorBT operations with hardware optimization
- **HardwareOptimizedVectorBTScaler**: Intelligent scaling with memory and GPU optimization
- **HardwareOptimizedVectorBTBatchScaler**: Batch processing with hardware acceleration
- **HardwareOptimizedMixin**: Reusable optimization capabilities for any component

### 🧠 Adaptive Optimization

- **Machine Learning-Based Tuning**: Automatic parameter optimization based on performance patterns
- **Context-Aware Decisions**: Optimization strategies adapt to data characteristics and workload
- **Continuous Learning**: System improves performance over time through experience

### 💾 Advanced Memory Management

- **Intelligent Chunking**: Automatic data chunking for large datasets
- **Memory Pooling**: Efficient object reuse and memory allocation
- **Garbage Collection**: Automatic memory cleanup and pressure detection
- **Data Type Optimization**: Automatic optimization of data types for memory efficiency

### 🖥️ GPU Acceleration

- **GPU Memory Pooling**: Efficient GPU memory management
- **Compute Pipelines**: Parallel operation execution
- **Batch Operations**: Optimized batch processing for GPU
- **M1 Neural Engine**: Specialized optimizations for Apple Silicon

### 📊 Performance Monitoring

- **Real-Time Metrics**: Live performance tracking and optimization feedback
- **Comprehensive Benchmarking**: Detailed performance analysis and comparison
- **Optimization Effectiveness**: Measurement of optimization impact
- **Automated Recommendations**: Intelligent suggestions for performance improvement

## Quick Start

### Basic Usage

```python
from src.features_common.vectorbt_extensions.hardware_optimized_manager import get_hardware_optimized_vectorbt_manager
from src.features_common.transforms.hardware_optimized_scaler import create_hardware_optimized_scaler

# Get hardware-optimized manager
manager = get_hardware_optimized_vectorbt_manager()

# Perform optimized operations
result = manager.rolling_mean(data, window=20)
result = manager.scale_data(data, method='zscore')

# Create hardware-optimized scaler
scaler = create_hardware_optimized_scaler(method='zscore')
scaled_data = scaler.fit_transform(data)
```

### Advanced Usage

```python
# Enable specific optimizations
manager.enable_hardware_optimization()
manager.set_workload_type(WorkloadType.DATA_PROCESSING)

# Get performance statistics
stats = manager.get_comprehensive_performance_summary()
print(f"Hardware operations: {stats['hardware_optimization']['hardware_operations']}")
print(f"Memory optimizations: {stats['hardware_optimization']['memory_optimizations']}")

# Run performance benchmark
from src.features_common.optimization.performance_benchmark import run_quick_benchmark
report = run_quick_benchmark()
```

## Architecture

### Component Hierarchy

```
features_common/
├── optimization/
│   ├── hardware_optimized_mixin.py      # Base optimization capabilities
│   ├── performance_benchmark.py         # Performance monitoring
│   └── hardware_integration_strategy.md # Integration strategy
├── vectorbt_extensions/
│   ├── hardware_optimized_manager.py    # Enhanced VectorBT manager
│   └── unified_manager.py               # Base VectorBT manager
├── transforms/
│   ├── hardware_optimized_scaler.py     # Enhanced scalers
│   └── vectorbt_scaler.py               # Base scalers
└── examples/
    └── hardware_optimization_demo.py    # Demonstration script
```

### Integration Points

1. **Memory Management**: Integration with `AdvancedMemoryManager`
2. **Adaptive Optimization**: Integration with `AdaptiveOptimizationEngine`
3. **GPU Acceleration**: Integration with `EnhancedGPUManager`
4. **Unified Hardware**: Integration with `IntegratedHardwareManager`

## Performance Improvements

### Expected Gains

- **Memory Usage**: 30-50% reduction through intelligent optimization
- **Large Dataset Processing**: 40-60% improvement with chunking
- **VectorBT Operations**: 20-40% improvement with hardware optimization
- **GPU Operations**: 50-80% improvement with GPU acceleration
- **Adaptive Optimization**: 10-25% improvement through learned optimizations

### Benchmarking

Run comprehensive benchmarks to measure performance:

```python
from src.features_common.optimization.performance_benchmark import get_performance_benchmark

benchmark = get_performance_benchmark()
report = benchmark.run_comprehensive_benchmark()

# View results
print(f"Speedup factor: {report.hardware_stats['speedup_factor']:.2f}x")
print(f"Performance improvement: {report.hardware_stats['performance_improvement_percent']:.1f}%")
```

## Configuration

### Hardware Optimization Settings

```python
# Enable/disable specific optimizations
manager.enable_hardware_optimization()    # Enable all optimizations
manager.disable_hardware_optimization()   # Disable all optimizations

# Set workload type for optimization
manager.set_workload_type(WorkloadType.DATA_PROCESSING)
manager.set_workload_type(WorkloadType.ML_TRAINING)
manager.set_workload_type(WorkloadType.BACKTESTING)
```

### Memory Optimization

```python
# Enable memory optimization
scaler = create_hardware_optimized_scaler(
    method='zscore',
    memory_efficient=True,
    enable_hardware_optimization=True
)

# Use memory-efficient decorators
from src.features_common.optimization.hardware_optimized_mixin import memory_efficient_processing

@memory_efficient_processing
def my_operation(data):
    return data.rolling(20).mean()
```

### GPU Acceleration

```python
# Enable GPU acceleration
scaler = create_hardware_optimized_scaler(
    method='zscore',
    enable_gpu=True,
    enable_hardware_optimization=True
)

# Check GPU availability
manager = get_hardware_optimized_vectorbt_manager()
gpu_info = manager.gpu_manager.get_enhanced_gpu_info()
print(f"GPU available: {gpu_info['mps_available']}")
```

## Monitoring and Debugging

### Performance Statistics

```python
# Get comprehensive performance summary
summary = manager.get_comprehensive_performance_summary()

# View hardware optimization stats
hardware_stats = summary['hardware_optimization']
print(f"Hardware operations: {hardware_stats['hardware_operations']}")
print(f"Memory optimizations: {hardware_stats['memory_optimizations']}")
print(f"GPU operations: {hardware_stats['gpu_operations']}")

# View optimization effectiveness
effectiveness = summary['optimization_effectiveness']
for strategy, stats in effectiveness.items():
    print(f"{strategy}: {stats['avg_execution_time']:.4f}s")
```

### Health Assessment

```python
# Check system health
health = summary['overall_health']
print(f"System health: {health}")

# Get recommendations
recommendations = summary.get('recommendations', [])
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

## Error Handling and Fallbacks

### Automatic Fallbacks

The system includes comprehensive fallback mechanisms:

1. **Hardware Unavailable**: Falls back to standard operations
2. **Optimization Failure**: Falls back to basic implementation
3. **GPU Failure**: Falls back to CPU operations
4. **Memory Pressure**: Automatically adjusts chunking strategy

### Error Monitoring

```python
# Check for errors in performance summary
summary = manager.get_comprehensive_performance_summary()
if 'errors' in summary:
    print(f"Errors detected: {summary['errors']}")

# Monitor fallback usage
stats = manager.get_hardware_stats()
fallback_rate = stats.get('fallback_rate', 0)
print(f"Fallback rate: {fallback_rate:.2%}")
```

## Best Practices

### 1. Workload Type Selection

Choose appropriate workload types for optimal performance:

```python
# For data processing tasks
manager.set_workload_type(WorkloadType.DATA_PROCESSING)

# For machine learning tasks
manager.set_workload_type(WorkloadType.ML_TRAINING)

# For backtesting tasks
manager.set_workload_type(WorkloadType.BACKTESTING)
```

### 2. Memory Management

Use memory-efficient operations for large datasets:

```python
# Enable memory optimization
scaler = create_hardware_optimized_scaler(
    memory_efficient=True,
    enable_hardware_optimization=True
)

# Use chunked processing for very large datasets
from src.features_common.optimization.hardware_optimized_mixin import chunked_processing

@chunked_processing(chunk_size_mb=50.0)
def process_large_dataset(data):
    return data.rolling(20).mean()
```

### 3. Performance Monitoring

Regularly monitor performance and adjust settings:

```python
# Run periodic benchmarks
benchmark = get_performance_benchmark()
report = benchmark.run_comprehensive_benchmark()

# Adjust settings based on results
if report.hardware_stats['speedup_factor'] < 1.1:
    manager.optimize_hardware_settings()
```

## Troubleshooting

### Common Issues

1. **Hardware utilities not available**
   - Check if `src/utils/hardware/` is properly installed
   - Verify import paths are correct

2. **Performance not improving**
   - Check if hardware optimization is enabled
   - Verify workload type is appropriate
   - Run benchmarks to identify bottlenecks

3. **Memory issues**
   - Enable memory optimization
   - Use chunked processing for large datasets
   - Check memory pressure levels

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run operations with debug output
manager = get_hardware_optimized_vectorbt_manager()
result = manager.rolling_mean(data, window=20)
```

## Examples

### Complete Example

```python
import pandas as pd
import numpy as np
from src.features_common.vectorbt_extensions.hardware_optimized_manager import get_hardware_optimized_vectorbt_manager
from src.features_common.transforms.hardware_optimized_scaler import create_hardware_optimized_scaler

# Generate test data
np.random.seed(42)
data = pd.DataFrame({
    'price': np.random.randn(10000).cumsum() + 100,
    'volume': np.random.exponential(1000, 10000),
    'returns': np.random.normal(0, 0.02, 10000)
})

# Get hardware-optimized manager
manager = get_hardware_optimized_vectorbt_manager()

# Set workload type
manager.set_workload_type(WorkloadType.DATA_PROCESSING)

# Perform optimized operations
rolling_mean = manager.rolling_mean(data['price'], window=20)
rolling_std = manager.rolling_std(data['returns'], window=20)
scaled_data = manager.scale_data(data['price'], method='zscore')

# Create hardware-optimized scaler
scaler = create_hardware_optimized_scaler(method='zscore')
scaled_returns = scaler.fit_transform(data['returns'])

# Get performance summary
summary = manager.get_comprehensive_performance_summary()
print(f"Hardware operations: {summary['hardware_optimization']['hardware_operations']}")
print(f"Memory optimizations: {summary['hardware_optimization']['memory_optimizations']}")
print(f"System health: {summary['overall_health']}")
```

## Contributing

When contributing to the hardware optimization system:

1. **Follow the existing patterns** in the codebase
2. **Add comprehensive tests** for new features
3. **Update benchmarks** to include new optimizations
4. **Document performance impacts** of changes
5. **Maintain backward compatibility** with existing APIs

## License

This hardware optimization integration follows the same license as the main features_common system.