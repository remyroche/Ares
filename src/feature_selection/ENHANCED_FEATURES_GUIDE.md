# Enhanced Feature Selection Features Guide

## 🚀 Overview

The feature selection module has been significantly enhanced with performance optimizations, error handling, caching, and hardware acceleration. This guide covers all the new capabilities and how to use them.

## 📋 New Features

### 1. Intelligent Caching System 💾
- **Hardware-optimized caching** using M1 memory management
- **Automatic cache invalidation** based on data changes
- **Memory-efficient storage** with compression
- **Performance monitoring** and statistics

### 2. Enhanced Error Handling 🛡️
- **Comprehensive error recovery** with fallback strategies
- **Detailed error logging** using tprint
- **Automatic retry mechanisms** with exponential backoff
- **Graceful degradation** for edge cases

### 3. Memory Efficiency 🧠
- **Chunked processing** for large datasets
- **Sparse matrix support** for memory optimization
- **Hardware-accelerated memory management**
- **Adaptive memory allocation**

### 4. Parallel Processing ⚡
- **Multi-threaded selection methods**
- **Hardware-optimized parallel execution**
- **Cross-validation in parallel**
- **Method comparison in parallel**

### 5. Vectorized Operations 🚀
- **Optimized correlation filtering**
- **Vectorized variance computation**
- **Hardware-accelerated operations**
- **Memory-efficient algorithms**

### 6. Sparse Matrix Support 📊
- **Sparse-aware feature selection**
- **Memory-efficient sparse operations**
- **Automatic sparse/dense conversion**
- **Optimized sparse algorithms**

### 7. Chunked Processing 📦
- **Large dataset handling**
- **Adaptive chunk sizing**
- **Memory-aware processing**
- **Progress tracking**

## 🎯 Quick Start

### Basic Enhanced Selection

```python
from src.feature_selection import enhanced_select_features

# Automatic optimization selection
result = enhanced_select_features(X, y, method='comprehensive')
print(f"Selected {len(result['selected_features'])} features")
print(f"Optimizations used: {result['optimizations_used']}")
```

### Using Specific Optimizations

```python
from src.feature_selection import (
    FeatureSelectionCacheManager, 
    MemoryEfficientFeatureSelector,
    ParallelFeatureSelector,
    VectorizedFeatureSelector
)

# Cached selection
cache_manager = FeatureSelectionCacheManager()
result = cache_manager.get_cached_selection(X, y, 'comprehensive', {})

# Memory-efficient selection
memory_selector = MemoryEfficientFeatureSelector()
result = memory_selector.select_features_chunked(X, y, 'comprehensive')

# Parallel selection
parallel_selector = ParallelFeatureSelector()
result = parallel_selector.parallel_selection(X, y, ['comprehensive', 'regularization'])

# Vectorized selection
vectorized_selector = VectorizedFeatureSelector()
result = vectorized_selector.vectorized_feature_selection(X, y, 'comprehensive')
```

### Sparse Matrix Support

```python
from scipy.sparse import csr_matrix
from src.feature_selection import SparseFeatureSelector

# Create sparse matrix
X_sparse = csr_matrix(X)

# Sparse-aware selection
sparse_selector = SparseFeatureSelector()
result = sparse_selector.select_features_sparse(X_sparse, y, 'comprehensive')
```

### Error Handling

```python
from src.feature_selection import robust_feature_selection, EnhancedErrorHandler

# Robust selection with error handling
@robust_feature_selection
def my_selection_func(X, y, method='comprehensive'):
    return select_features(X, y, method)

# Custom error handler
error_handler = EnhancedErrorHandler(enable_recovery=True)
result = error_handler.handle_error(error, context, selection_func, X, y)
```

## ⚙️ Configuration

### Enhanced Framework Configuration

```python
from src.feature_selection import EnhancedFeatureSelectionConfig

config = EnhancedFeatureSelectionConfig(
    enable_caching=True,
    enable_memory_optimization=True,
    enable_parallel_processing=True,
    enable_vectorization=True,
    enable_sparse_support=True,
    enable_chunked_processing=True,
    memory_limit_gb=16.0
)

result = enhanced_select_features(X, y, config=config)
```

### Component-Specific Configuration

```python
from src.feature_selection import CacheConfig, MemoryConfig, ParallelConfig

# Cache configuration
cache_config = CacheConfig(
    enable_caching=True,
    cache_dir="data_cache/feature_selection",
    max_memory_mb=2048,
    default_ttl_seconds=3600
)

# Memory configuration
memory_config = MemoryConfig(
    memory_limit_gb=8.0,
    chunk_size=10000,
    enable_memory_monitoring=True
)

# Parallel configuration
parallel_config = ParallelConfig(
    max_workers=8,
    enable_hardware_optimization=True
)
```

## 📊 Performance Monitoring

### Get Performance Statistics

```python
from src.feature_selection import get_enhanced_framework

framework = get_enhanced_framework()
stats = framework.get_performance_summary()

print("Performance Summary:")
for component, stats in stats['components'].items():
    print(f"{component}: {stats}")
```

### Component-Specific Stats

```python
# Cache statistics
cache_stats = cache_manager.get_performance_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

# Memory statistics
memory_stats = memory_selector.get_performance_stats()
print(f"Memory optimizations: {memory_stats['memory_optimizations']}")

# Parallel statistics
parallel_stats = parallel_selector.get_performance_stats()
print(f"Average speedup: {parallel_stats['avg_speedup']:.1f}x")
```

## 🔧 Advanced Usage

### Custom Processing Pipeline

```python
from src.feature_selection import (
    ChunkedFeatureProcessor,
    SparseFeatureSelector,
    VectorizedFeatureSelector
)

def custom_processing_pipeline(X, y):
    # Step 1: Check if sparse processing is beneficial
    sparse_selector = SparseFeatureSelector()
    if sparse_selector._is_sparse_beneficial(X):
        result = sparse_selector.select_features_sparse(X, y, 'comprehensive')
    else:
        # Step 2: Use vectorized processing
        vectorized_selector = VectorizedFeatureSelector()
        result = vectorized_selector.vectorized_feature_selection(X, y, 'comprehensive')
    
    return result
```

### Parallel Method Comparison

```python
from src.feature_selection import ParallelSelectionManager

# Compare multiple methods in parallel
parallel_manager = ParallelSelectionManager()
methods = ['comprehensive', 'regularization', 'adaptive', 'directional']

comparison_result = parallel_manager.compare_methods(X, y, methods)

print("Method Comparison Results:")
for method, result in comparison_result['results'].items():
    print(f"{method}: {len(result['selected_features'])} features, "
          f"{result['execution_time']:.3f}s")
```

### Memory-Efficient Large Dataset Processing

```python
from src.feature_selection import ChunkedFeatureProcessor

# Process very large dataset
chunked_processor = ChunkedFeatureProcessor()

def process_chunk(X_chunk, y_chunk):
    # Your custom processing logic here
    return select_features(X_chunk, y_chunk, method='comprehensive')

result = chunked_processor.process_large_dataset(X, y, process_chunk)
```

## 🐛 Error Handling and Recovery

### Automatic Error Recovery

```python
from src.feature_selection import robust_feature_selection

@robust_feature_selection
def safe_feature_selection(X, y, method='comprehensive'):
    return select_features(X, y, method)

# This will automatically handle errors and attempt recovery
result = safe_feature_selection(X, y)
```

### Custom Error Handling

```python
from src.feature_selection import EnhancedErrorHandler, FeatureSelectionError

error_handler = EnhancedErrorHandler(enable_recovery=True)

try:
    result = select_features(X, y, method='comprehensive')
except FeatureSelectionError as e:
    # Custom error handling
    recovery_result = error_handler.handle_error(e, context, select_features, X, y)
    result = recovery_result
```

## 📈 Performance Tips

### 1. Use Appropriate Optimizations
- **Small datasets** (< 1K samples): Use basic selection
- **Medium datasets** (1K-10K samples): Use vectorization and caching
- **Large datasets** (> 10K samples): Use chunked processing and memory optimization
- **Sparse data**: Use sparse matrix support

### 2. Memory Management
- Monitor memory usage with `memory_optimizer.get_memory_pressure()`
- Use chunked processing for datasets > 50K samples
- Enable garbage collection for memory-intensive operations

### 3. Parallel Processing
- Use parallel processing for method comparison
- Enable hardware optimization for M1/M2 Macs
- Adjust worker count based on available cores

### 4. Caching Strategy
- Enable caching for repeated operations
- Use appropriate TTL based on data stability
- Monitor cache hit rates for optimization

## 🔍 Troubleshooting

### Common Issues

1. **Memory errors**: Enable chunked processing or reduce chunk size
2. **Slow performance**: Enable vectorization and parallel processing
3. **Cache misses**: Check data consistency and TTL settings
4. **Sparse matrix errors**: Ensure proper sparse matrix format

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from src.utils.tprint import tprint_debug
tprint_debug("Debug information will be shown")
```

## 📚 Examples

### Complete Example

```python
import numpy as np
from src.feature_selection import enhanced_select_features, get_enhanced_framework

# Generate sample data
X = np.random.rand(10000, 100)
y = np.random.rand(10000)

# Enhanced selection with automatic optimization
result = enhanced_select_features(X, y, method='comprehensive')

print(f"Selected {len(result['selected_features'])} features")
print(f"Execution time: {result['execution_time']:.3f}s")
print(f"Optimizations used: {result['optimizations_used']}")

# Get performance summary
framework = get_enhanced_framework()
stats = framework.get_performance_summary()
print("Performance stats:", stats)
```

### Sparse Data Example

```python
from scipy.sparse import csr_matrix
from src.feature_selection import SparseFeatureSelector

# Create sparse data
X_sparse = csr_matrix(np.random.rand(1000, 500))
y = np.random.rand(1000)

# Sparse-aware selection
sparse_selector = SparseFeatureSelector()
result = sparse_selector.select_features_sparse(X_sparse, y, 'comprehensive')

print(f"Sparse selection: {len(result['selected_features'])} features")
print(f"Memory saved: {result['memory_saved_mb']:.1f}MB")
```

## 🎉 Conclusion

The enhanced feature selection framework provides significant performance improvements and robust error handling while maintaining ease of use. The automatic optimization selection ensures optimal performance for different data characteristics without requiring manual configuration.

For more information, see the individual module documentation and examples in the `src/feature_selection/` directory.