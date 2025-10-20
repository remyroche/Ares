# M1 Comprehensive Hardware Optimizations for Apple Silicon

This module provides comprehensive hardware optimization specifically designed for M1/M2/M3/M4 Apple Silicon chips, including unified memory management, advanced CPU optimization, enhanced GPU acceleration, Neural Engine integration, and intelligent adaptive optimization.

## 🚀 Comprehensive M1 Features

### 🧠 Unified Memory Architecture
- **Cross-Component Sharing**: Intelligent memory sharing between CPU, GPU, and Neural Engine
- **Memory Tier Optimization**: Automatic tier selection (CPU-only, Shared, GPU-optimized, Neural Engine, Compressed)
- **Memory Compression**: Automatic compression for large datasets with configurable ratios
- **Memory Pool Management**: Efficient memory pooling with adaptive sizing
- **Pressure Management**: Real-time memory pressure detection and response

### ⚡ Advanced CPU Optimization
- **Performance/Efficiency Core Management**: Intelligent workload distribution across M1 cores
- **Thread Affinity**: Automatic thread-to-core assignment for optimal performance
- **Thermal Management**: Real-time thermal monitoring and throttling prevention
- **Workload Balancing**: Dynamic load balancing across available cores
- **Power Management**: Intelligent power scaling based on workload requirements

### 🎮 Enhanced GPU Acceleration
- **Metal Performance Shaders**: Native Metal compute pipeline integration
- **Unified Memory Optimization**: Seamless data sharing between CPU and GPU
- **Batch Processing**: Intelligent batching for improved throughput
- **Memory Pool Management**: GPU memory pooling with compression
- **Async Execution**: Non-blocking GPU operations with callback support

### 🧠 Neural Engine Integration
- **Model Optimization**: Automatic model optimization for Neural Engine execution
- **Quantization Support**: 8-bit and 16-bit quantization for efficiency
- **Batch Inference**: High-throughput batch processing
- **Model Caching**: Intelligent model caching and management
- **Fallback Support**: Automatic fallback to CPU/GPU when Neural Engine unavailable

### 🔄 Adaptive Optimization Engine
- **Performance Learning**: Learns from execution patterns and adapts optimization strategies
- **Bottleneck Detection**: Automatically identifies and resolves performance bottlenecks
- **Strategy Selection**: Chooses optimal execution strategy based on workload characteristics
- **Real-time Adaptation**: Continuously adjusts optimization parameters
- **Comprehensive Monitoring**: Detailed performance metrics and analysis

### 💾 Advanced Memory Management
- **int64 → int32**: Reduces memory usage by 50% for integer data
- **float64 → float32**: Reduces memory usage by 50% for floating-point data
- **object → category**: Converts repeated strings to efficient categorical data
- **Automatic detection**: Intelligently determines when optimization is safe

### 🗄️ Intelligent Caching System
- **LRU Eviction**: Least Recently Used eviction policy
- **Compression**: Automatic compression for large data (>1MB)
- **TTL Support**: Time-to-live for cached items
- **Memory Monitoring**: Real-time memory usage tracking
- **Statistics**: Comprehensive hit/miss rate tracking

## Quick Start

### Basic M1 Optimization

```python
from src.utils.hardware import (
    m1_optimized, get_comprehensive_optimizer, 
    optimize_for_unified_memory, WorkloadCategory
)

# Comprehensive M1 optimization
@m1_optimized("matrix_operations", WorkloadCategory.MACHINE_LEARNING)
def optimized_matrix_operations(data):
    return np.dot(data, data.T)

# Unified memory optimization
optimized_data = optimize_for_unified_memory(large_array, 'matrix_operations', 'gpu')

# Get comprehensive optimizer
optimizer = get_comprehensive_optimizer()
result = optimizer.optimize_operation("neural_inference", model_data)
```

### Advanced M1 Features

```python
from src.utils.hardware import (
    unified_memory_optimized, optimize_cpu_execution, 
    gpu_accelerated, neural_engine_optimized,
    WorkloadType, GPUOperationType, NeuralEngineOperation
)

# Unified memory optimization
@unified_memory_optimized('matrix_operations', 'gpu')
def gpu_optimized_function(data):
    return data * 2

# CPU optimization with thermal management
@optimize_cpu_execution(WorkloadType.CPU_INTENSIVE)
def cpu_intensive_task(data):
    return np.sum(data ** 2)

# GPU acceleration
@gpu_accelerated(GPUOperationType.MATRIX_MULTIPLICATION)
def gpu_matrix_multiply(A, B):
    return np.dot(A, B)

# Neural Engine optimization
@neural_engine_optimized(NeuralEngineOperation.INFERENCE)
def neural_inference(model, data):
    return model.predict(data)
```

### Comprehensive M1 Optimization

```python
from src.utils.hardware import (
    get_comprehensive_optimizer, ComprehensiveConfig, 
    OptimizationStrategy, WorkloadCategory, m1_optimized
)

# Configure for maximum performance
config = ComprehensiveConfig(
    optimization_strategy=OptimizationStrategy.MAXIMUM_PERFORMANCE,
    workload_category=WorkloadCategory.MACHINE_LEARNING
)
optimizer = get_comprehensive_optimizer(config)

# Financial modeling with M1 optimization
@m1_optimized("monte_carlo", WorkloadCategory.FINANCIAL_MODELING)
def monte_carlo_simulation(returns, num_simulations=10000):
    scenarios = np.random.normal(0, 0.02, (num_simulations, len(returns)))
    portfolio_values = np.sum(returns * (1 + scenarios), axis=1)
    return {
        'mean': np.mean(portfolio_values),
        'var_95': np.percentile(portfolio_values, 5)
    }

# Real-time trading optimization
@m1_optimized("signal_processing", WorkloadCategory.REAL_TIME_TRADING)
def process_trading_signals(market_data):
    # Optimized for low-latency processing
    return calculate_signals(market_data)
```

### Memory Management Examples

```python
from src.utils.hardware import (
    get_unified_memory_manager, allocate_unified_memory,
    memory_tier_aware, MemoryTier
)

# Unified memory allocation
memory_manager = get_unified_memory_manager()
allocation_id = memory_manager.allocate_for_operation(
    'neural_network', 512.0, 'inference'
)

# Memory tier aware processing
@memory_tier_aware(MemoryTier.NEURAL_ENGINE)
def neural_processing(data):
    return process_with_neural_engine(data)

# Get comprehensive memory stats
stats = memory_manager.get_comprehensive_stats()
print(f"Memory usage: {stats['current_usage_mb']:.1f}MB")
```

### Dynamic Memory Allocation

```python
from src.utils.hardware import (
    get_optimal_memory_allocation, get_system_recommendations,
    WorkloadType, update_memory_usage
)

# Get system recommendations
recommendations = get_system_recommendations()
print(f"System tier: {recommendations['system_tier']}")
print(f"Total memory: {recommendations['total_memory_gb']:.1f}GB")

# Get optimal allocation for specific workload and data size
allocation = get_optimal_memory_allocation(
    workload_type=WorkloadType.HEAVY,
    data_size_mb=5000,  # 5GB dataset
    user_preferences={'memory_usage_factor': 1.2}
)
print(f"Cache memory: {allocation.cache_memory_mb:.0f}MB")
print(f"Processing memory: {allocation.processing_memory_mb:.0f}MB")
print(f"Total allocated: {allocation.total_allocated_mb:.0f}MB")

# Update memory usage for adaptive learning
update_memory_usage(used_memory_mb=8000, pressure_level='high')
```

## Configuration

### Cache Configuration

```python
from src.utils.hardware import CacheConfig, DataTypeOptimization, CacheStrategy

config = CacheConfig(
    max_memory_mb=1024.0,           # Maximum cache memory
    max_items=10000,                # Maximum number of items
    strategy=CacheStrategy.LRU,     # Eviction strategy
    data_type_optimization=DataTypeOptimization.AGGRESSIVE,
    enable_compression=True,        # Enable compression
    auto_optimize_dtypes=True,      # Auto-optimize data types
    prefer_int32=True,              # Prefer int32 over int64
    prefer_float32=True,            # Prefer float32 over float64
    ttl_seconds=3600.0             # Default TTL
)
```

### Hardware Configuration

```python
from src.utils.hardware import HardwareConfig, OptimizationLevel

config = HardwareConfig(
    memory_limit_gb=8.0,
    enable_adaptive_optimization=True,
    performance_monitoring_enabled=True,
    cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
    gpu_optimization_level=OptimizationLevel.BALANCED,
    memory_optimization_level=OptimizationLevel.AGGRESSIVE
)
```

## Decorators

### Caching Decorators

```python
# Basic caching
@cache_result(ttl=3600)
def my_function(x):
    return x * 2

# Smart caching with optimization
@smart_cache(ttl=1800, key_func=lambda x: f"key_{x}")
def expensive_function(data):
    return process_data(data)

# DataFrame-specific caching
@cache_dataframe_result(ttl=3600)
def process_dataframe(df):
    return df.describe()

# NumPy array-specific caching
@cache_numpy_result(ttl=1800)
def process_array(arr):
    return np.dot(arr.T, arr)
```

### Optimization Decorators

```python
# Auto-optimize inputs and outputs
@auto_optimize(optimize_inputs=True, optimize_outputs=True)
def process_data(data):
    return data * 2

# Memory-efficient processing
@memory_efficient(memory_threshold_mb=100.0, auto_cleanup=True)
def memory_intensive_function(data):
    return np.dot(data.T, data)

# Heavy computation optimization
@optimize_heavy_computation()
def heavy_calculation(data):
    return complex_computation(data)

# Memory-aware processing
@memory_aware()
def memory_aware_function(data):
    return process_large_data(data)
```

### Performance Tracking

```python
# Track performance metrics
@performance_tracked(log_performance=True, track_memory=True)
def tracked_function(data):
    return process_data(data)
```

## Data Processing

### DataFrame Optimization

```python
from src.utils.hardware import optimize_dataframe_default

# Optimize DataFrame
df_optimized = optimize_dataframe_default(df)

# Check memory savings
original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
optimized_memory = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
savings = original_memory - optimized_memory
print(f"Memory saved: {savings:.2f} MB")
```

### NumPy Array Optimization

```python
from src.utils.hardware import optimize_numpy_array_default

# Optimize NumPy array
arr_optimized = optimize_numpy_array_default(arr)

# Check size reduction
original_size = arr.nbytes / (1024 * 1024)
optimized_size = arr_optimized.nbytes / (1024 * 1024)
reduction = (original_size - optimized_size) / original_size * 100
print(f"Size reduced by: {reduction:.1f}%")
```

### Batch Processing

```python
from src.utils.hardware import optimize_all_dataframes, optimize_all_arrays

# Optimize all DataFrames in a dictionary
data_optimized = optimize_all_dataframes(data)

# Optimize all NumPy arrays in a dictionary
data_optimized = optimize_all_arrays(data)
```

## Hardware Management

### Integrated Hardware Manager

```python
from src.utils.hardware import get_integrated_hardware_manager

# Get integrated manager
manager = get_integrated_hardware_manager()

# Process different types of data
market_data = process_market_data(raw_market_data)
ml_data = process_ml_training_data(training_data)
backtest_data = process_backtesting_data(backtest_data)

# Get optimization report
report = manager.get_optimization_report()
print(f"Cache hit rate: {report['cache_statistics']['hit_rate']:.2%}")
print(f"Memory optimizations: {report['performance_metrics']['optimizations_applied']}")
```

### Memory Management

```python
from src.utils.hardware import get_memory_report, clear_all_caches

# Get memory report
memory_report = get_memory_report()
print(f"Total memory usage: {memory_report['total_memory_usage_mb']:.2f} MB")

# Clear all caches
clear_all_caches()
```

## Performance Monitoring

### Cache Statistics

```python
from src.utils.hardware import get_optimization_stats

stats = get_optimization_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory used: {stats['total_memory_used_mb']:.2f} MB")
```

### System Status

```python
from src.utils.hardware import get_system_optimization_status

status = get_system_optimization_status()
print(f"Hardware status: {status['hardware_status']}")
print(f"Cache statistics: {status['cache_statistics']}")
print(f"Performance metrics: {status['performance_metrics']}")
```

## Examples

### Complete Data Pipeline

```python
from src.utils.hardware import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    process_market_data, get_integrated_hardware_manager
)

@smart_cache(ttl=3600)
@auto_optimize()
@memory_efficient(memory_threshold_mb=200.0)
@performance_tracked(log_performance=True)
def complete_data_pipeline(raw_data):
    """Complete data processing pipeline with full optimization."""
    
    # Step 1: Load and optimize data
    optimized_data = process_market_data(raw_data)
    
    # Step 2: Process data
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

# Use the pipeline
result = complete_data_pipeline(raw_data)
```

### Custom Cache Configuration

```python
from src.utils.hardware import CacheConfig, get_global_cache

# Create custom cache configuration
custom_config = CacheConfig(
    max_memory_mb=2048.0,           # 2GB cache
    max_items=50000,                # 50K items
    strategy=CacheStrategy.LFU,     # Least Frequently Used
    data_type_optimization=DataTypeOptimization.MAXIMUM,
    enable_compression=True,
    compression_threshold_mb=0.5,   # Compress items > 0.5MB
    ttl_seconds=7200               # 2 hours TTL
)

# Get cache with custom configuration
cache = get_global_cache(custom_config)

# Use cache
cache.put('large_data', large_array)
retrieved_data = cache.get('large_data')
```

## Best Practices

### 1. Use Appropriate Decorators
- Use `@smart_cache` for functions with expensive computations
- Use `@auto_optimize` for data processing functions
- Use `@memory_efficient` for memory-intensive operations
- Use `@performance_tracked` for monitoring critical functions

### 2. Configure Cache Appropriately
- Set appropriate TTL based on data freshness requirements
- Use custom key functions for complex cache keys
- Monitor cache hit rates and adjust configuration accordingly

### 3. Monitor Performance
- Regularly check optimization statistics
- Monitor memory usage and adjust limits as needed
- Use performance tracking for critical functions

### 4. Data Type Optimization
- Let the system automatically optimize data types
- Use `prefer_int32=True` and `prefer_float32=True` for maximum memory savings
- Consider using categorical data for repeated strings

### 5. Memory Management
- Use `memory_efficient` decorator for large data processing
- Monitor memory usage and clear caches when needed
- Use appropriate memory thresholds for your hardware

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce cache size limits
   - Enable more aggressive compression
   - Use `memory_efficient` decorator

2. **Low Cache Hit Rate**
   - Check cache key generation
   - Increase TTL for stable data
   - Use appropriate cache strategies

3. **Performance Issues**
   - Enable performance tracking
   - Check optimization statistics
   - Adjust hardware configuration

### Debug Mode

```python
import logging
logging.getLogger('src.utils.hardware').setLevel(logging.DEBUG)
```

## API Reference

See the individual module documentation for detailed API reference:
- `enhanced_caching_system.py` - Core caching functionality
- `optimization_decorators.py` - Decorators for optimization
- `integrated_hardware_manager.py` - Integrated hardware management
- `optimization_patches.py` - Patches for existing code
- `optimization_examples.py` - Usage examples

## License

This module is part of the Ares Trading System and is subject to the same license terms.