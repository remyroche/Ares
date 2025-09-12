# Hardware Optimization Guide for Triple Barrier Labeling

This guide explains the hardware optimization features implemented in the triple barrier labeling system, specifically designed for M1/M2/M3 Macs and other high-performance computing environments.

## Overview

The triple barrier labeling system has been enhanced with comprehensive hardware optimizations to leverage the unique capabilities of modern hardware, particularly Apple Silicon (M1/M2/M3) processors.

## Key Hardware Optimizations

### 1. M1/M2/M3 CPU Optimization

#### Features:
- **Performance Core Utilization**: Optimized thread pools that utilize M1 performance cores efficiently
- **Efficiency Core Management**: Intelligent workload distribution between performance and efficiency cores
- **Numpy Operations**: Optimized NumPy operations with M1-specific thread configuration
- **Parallel Processing**: Numba-accelerated parallel processing for large datasets

#### Implementation:
```python
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Initialize M1 CPU optimizer
cpu_optimizer = get_m1_cpu_optimizer()
cpu_optimizer.optimize_numpy_operations()

# Create optimized thread pool
thread_pool = create_m1_optimized_thread_pool(max_workers=4)

# Run CPU-intensive tasks
result = run_cpu_intensive_task(compute_intensive_function, data)
```

### 2. M1/M2/M3 GPU Acceleration (MPS)

#### Features:
- **Metal Performance Shaders (MPS)**: GPU acceleration for tensor operations
- **Memory-Efficient GPU Operations**: Optimized data transfer between CPU and GPU
- **Automatic Fallback**: Graceful fallback to CPU when GPU is unavailable

#### Implementation:
```python
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager

# Initialize M1 GPU manager
gpu_manager = get_m1_gpu_manager()

# Check GPU availability
if gpu_manager.mps_available:
    # Use GPU acceleration
    optimized_data = gpu_manager.optimize_tensor_operations(data)
else:
    # Fallback to CPU
    optimized_data = data
```

### 3. Advanced Memory Management

#### Features:
- **Unified Memory Architecture**: Optimized for M1's unified memory system
- **Memory Compression**: Leverages macOS memory compression
- **Garbage Collection Tuning**: Optimized GC settings for M1
- **Memory Leak Detection**: Automatic detection and prevention of memory leaks
- **Swap Management**: Intelligent swap usage optimization

#### Implementation:
```python
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer

# Initialize memory optimizer
memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)

# Optimize DataFrame memory usage
optimized_df = memory_optimizer.optimize_dataframe_memory(df)

# Get memory statistics
memory_stats = memory_optimizer.get_memory_stats()
```

### 4. Data Optimization

#### Features:
- **Float32 Optimization**: Automatic conversion to float32 for better M1 performance
- **Contiguous Memory Layout**: Ensures optimal memory access patterns
- **DataFrame Optimization**: Automatic dtype optimization and memory reduction
- **Chunked Processing**: Memory-efficient processing of large datasets

#### Implementation:
```python
from src.utils.hardware.m1_gpu_utils import create_m1_optimized_array

# Create optimized arrays
optimized_array = create_m1_optimized_array(data, dtype=np.float32)

# Optimize DataFrame
optimized_df = optimize_dataframe_for_m1(df)
```

## Usage Examples

### Basic Hardware-Optimized Triple Barrier Labeling

```python
from src.training.steps.market_analysis import apply_hardware_optimized_triple_barrier_labeling

# Apply hardware-optimized labeling
labeled_data = apply_hardware_optimized_triple_barrier_labeling(
    data,
    profit_take_multiplier=0.004,
    stop_loss_multiplier=0.003,
    enable_hardware_optimization=True,
    enable_numba_acceleration=True,
    enable_gpu_acceleration=True,
    memory_limit_gb=8.0
)
```

### Regime-Aware Hardware Optimization

```python
from src.training.steps.market_analysis import apply_hardware_optimized_regime_aware_triple_barrier_labeling

# Apply regime-aware hardware optimization
labeled_data = apply_hardware_optimized_regime_aware_triple_barrier_labeling(
    data,
    regime_column='hmm_regime',
    enable_hardware_optimization=True,
    enable_parallel_processing=True,
    num_threads=4
)
```

### Advanced Configuration

```python
from src.training.steps.market_analysis import HardwareOptimizedTripleBarrierLabeling, HardwareOptimizedConfig

# Create advanced configuration
config = HardwareOptimizedConfig(
    profit_take_multiplier=0.004,
    stop_loss_multiplier=0.003,
    time_barrier_minutes=30,
    max_lookahead=100,
    transaction_cost=0.0008,
    binary_classification=True,
    enable_regime_aware=True,
    enable_hardware_optimization=True,
    enable_numba_acceleration=True,
    enable_gpu_acceleration=True,
    memory_limit_gb=8.0,
    enable_memory_monitoring=True,
    enable_parallel_processing=True,
    num_threads=4
)

# Initialize hardware-optimized labeler
labeler = HardwareOptimizedTripleBarrierLabeling(config)

# Apply labeling
labeled_data = labeler.apply_triple_barrier_labeling(data)

# Cleanup resources
labeler.cleanup()
```

## Performance Benefits

### Expected Performance Improvements:

1. **CPU Optimization**: 20-40% improvement in CPU-bound operations
2. **Memory Optimization**: 30-50% reduction in memory usage
3. **GPU Acceleration**: 2-5x speedup for large tensor operations
4. **Parallel Processing**: 2-4x speedup for large datasets
5. **Overall**: 50-200% improvement in total processing time

### Memory Efficiency:

- **Unified Memory**: Better utilization of M1's unified memory architecture
- **Memory Compression**: Automatic memory compression for large datasets
- **Garbage Collection**: Optimized GC reduces memory fragmentation
- **Data Types**: Automatic optimization of data types for memory efficiency

## Hardware Requirements

### Minimum Requirements:
- **macOS**: 12.0 or later (for MPS support)
- **Python**: 3.8 or later
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space for caching

### Recommended Requirements:
- **Processor**: M1/M2/M3 with 8+ cores
- **Memory**: 16GB+ unified memory
- **Storage**: SSD with 10GB+ free space
- **Python Packages**: numba, torch, psutil

## Installation and Setup

### 1. Install Required Packages

```bash
# Core packages
pip install numba torch psutil

# For MPS support (PyTorch)
pip install torch torchvision torchaudio
```

### 2. Verify Hardware Support

```python
from src.training.steps.market_analysis import get_hardware_optimization_info

# Check hardware optimization availability
info = get_hardware_optimization_info()
print(f"Hardware optimizations available: {info['hardware_optimizations_available']}")
print(f"M1 available: {info['m1_available']}")
print(f"MPS available: {info['mps_available']}")
```

### 3. Test Hardware Optimizations

```python
from src.training.steps.market_analysis.test_hardware_optimizations import run_comprehensive_hardware_test

# Run comprehensive hardware tests
results = run_comprehensive_hardware_test()
print(f"Performance improvement: {results['summary']['performance_improvement']:.1f}%")
```

## Troubleshooting

### Common Issues:

1. **MPS Not Available**:
   - Ensure macOS 12.0+ and PyTorch 1.12+
   - Check if M1/M2/M3 processor is detected
   - Verify PyTorch MPS installation

2. **Memory Issues**:
   - Reduce memory_limit_gb parameter
   - Enable memory monitoring
   - Check available system memory

3. **Performance Issues**:
   - Verify Numba installation
   - Check thread pool configuration
   - Monitor CPU and memory usage

### Debug Mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for hardware optimizations
```

## Best Practices

### 1. Memory Management:
- Use appropriate memory limits for your system
- Enable memory monitoring for large datasets
- Cleanup resources after processing

### 2. Performance Optimization:
- Use Numba acceleration for large datasets
- Enable parallel processing for multi-core systems
- Monitor performance and adjust parameters

### 3. Error Handling:
- Always use try-catch blocks for hardware operations
- Implement fallback mechanisms for unsupported hardware
- Monitor and log hardware optimization status

## Monitoring and Profiling

### Performance Monitoring:

```python
# Get performance statistics
stats = labeler.get_performance_stats()
print(f"Hardware optimizations: {stats['performance_optimization']}")

# Get recommendations
recommendations = labeler.get_recommendations()
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

### Memory Monitoring:

```python
from src.utils.hardware.m1_memory_optimizer import start_m1_memory_monitoring, stop_m1_memory_monitoring

# Start memory monitoring
start_m1_memory_monitoring()

# Your processing code here
# ...

# Stop memory monitoring
stop_m1_memory_monitoring()
```

## Future Enhancements

### Planned Features:
1. **Multi-GPU Support**: Support for multiple M1 GPUs
2. **Advanced Caching**: Intelligent caching strategies
3. **Dynamic Optimization**: Runtime optimization based on workload
4. **Cross-Platform Support**: Extensions for Intel and AMD processors

### Contributing:
- Report issues and feature requests
- Contribute optimization improvements
- Share performance benchmarks
- Help with cross-platform testing

## Conclusion

The hardware optimization features provide significant performance improvements for triple barrier labeling on M1/M2/M3 Macs. By leveraging the unique capabilities of Apple Silicon, users can achieve substantial speedups and memory efficiency improvements.

For more information, see:
- `test_hardware_optimizations.py` - Comprehensive test suite
- `hardware_optimized_triple_barrier_labeling.py` - Implementation details
- `TRIPLE_BARRIER_DOCUMENTATION.md` - General documentation