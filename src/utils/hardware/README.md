# Enhanced Hardware Utilities with Caching and Optimization

This module provides a comprehensive hardware optimization and caching system that automatically optimizes data types, implements LRU caching, and provides memory-efficient operations throughout the codebase.

## Features

### 🚀 Automatic Data Type Optimization
- **int64 → int32**: Reduces memory usage by 50% for integer data
- **float64 → float32**: Reduces memory usage by 50% for floating-point data
- **object → category**: Converts repeated strings to efficient categorical data
- **Automatic detection**: Intelligently determines when optimization is safe

### 💾 Advanced Caching System
- **LRU Eviction**: Least Recently Used eviction policy
- **Compression**: Automatic compression for large data (>1MB)
- **TTL Support**: Time-to-live for cached items
- **Memory Monitoring**: Real-time memory usage tracking
- **Statistics**: Comprehensive hit/miss rate tracking

### 🔧 Hardware Integration
- **M1 Optimization**: Apple Silicon specific optimizations
- **GPU Acceleration**: Automatic GPU detection and acceleration
- **Memory Management**: Intelligent memory cleanup and optimization
- **Performance Monitoring**: Real-time performance tracking

## Quick Start

### Basic Usage

```python
from src.utils.hardware import optimize_dataframe, cache_result, auto_optimize

# Optimize a DataFrame
df_optimized = optimize_dataframe(df)

# Cache a function result
@cache_result(ttl=3600)  # Cache for 1 hour
def expensive_calculation(data):
    return process_data(data)

# Auto-optimize function inputs and outputs
@auto_optimize()
def process_data(data):
    return data * 2
```

### Advanced Usage

```python
from src.utils.hardware import (
    smart_cache, memory_efficient, performance_tracked,
    get_integrated_hardware_manager, process_market_data
)

# Smart caching with optimization
@smart_cache(ttl=1800, key_func=lambda x: f"data_{x.shape}")
@auto_optimize()
def process_large_dataset(data):
    return data.describe()

# Memory-efficient processing
@memory_efficient(memory_threshold_mb=200.0)
def process_memory_intensive_data(data):
    return np.dot(data.T, data)

# Performance tracking
@performance_tracked(log_performance=True)
def tracked_operation(data):
    return expensive_computation(data)

# Integrated hardware management
manager = get_integrated_hardware_manager()
optimized_data = process_market_data(raw_data)
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