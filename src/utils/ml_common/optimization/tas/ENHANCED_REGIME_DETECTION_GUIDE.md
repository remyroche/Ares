# Enhanced TAS Regime Detection Guide

## Overview

The Enhanced TAS Regime Detection system provides advanced regime detection capabilities with comprehensive performance optimizations and validation features. This system is designed to handle large-scale financial data efficiently while maintaining high accuracy and reliability.

## Key Features

### 1. Performance Optimizations

#### Memory Management
- **Memory-efficient processing** for large datasets
- **Automatic memory optimization** using M1 memory optimizer
- **Chunked processing** for datasets exceeding memory limits
- **Memory usage monitoring** and optimization

#### Parallel Processing
- **Multi-threaded regime detection** across timeframes
- **M1 CPU optimization** for performance and efficiency cores
- **Automatic load balancing** based on system capabilities
- **Parallel validation** for faster processing

#### GPU Acceleration
- **M1 GPU acceleration** for matrix operations
- **Automatic fallback** to CPU when GPU is unavailable
- **Optimized tensor operations** for regime detection
- **Memory-efficient GPU usage**

### 2. Advanced Validation

#### Cross-Validation
- **CVLSA cross-validation** for regime stability
- **Multiple validation folds** for robust assessment
- **Regime consistency metrics** across validation sets
- **Stability scoring** for regime quality

#### Out-of-Sample Testing
- **Temporal data splitting** for realistic validation
- **Out-of-sample regime prediction** accuracy
- **Regime similarity analysis** between train/test sets
- **Performance degradation detection**

#### Regime Persistence Analysis
- **Regime transition probabilities** calculation
- **Regime stability scores** over time
- **Regime duration statistics** analysis
- **Persistence pattern recognition**

### 3. Intelligent Caching

#### Smart Caching System
- **Automatic cache key generation** based on data and parameters
- **Persistent caching** across sessions
- **Cache hit/miss tracking** for performance monitoring
- **Automatic cache cleanup** and management

#### Performance Benefits
- **Significant speedup** for repeated analyses
- **Memory-efficient caching** with compression
- **Cache validation** to ensure data integrity
- **Configurable cache policies**

## Usage

### Basic Usage

```python
from src.utils.ml_common.optimization.tas.enhanced_regime_detection import get_enhanced_tas_regime_detection

# Initialize enhanced regime detection
regime_detector = get_enhanced_tas_regime_detection(
    enable_gpu=True,
    enable_memory_optimization=True,
    enable_parallel=True,
    cache_dir='./tas_cache',
    max_memory_gb=8.0
)

# Perform regime detection
results = regime_detector.detect_regimes_enhanced(
    data=your_data,
    timeframes=['1h', '4h', '1d'],
    methods=['unsupervised', 'clustering', 'qualification'],
    use_cache=True,
    parallel=True
)
```

### Advanced Configuration

```python
# Custom configuration for specific use cases
regime_detector = get_enhanced_tas_regime_detection(
    enable_gpu=True,                    # Enable GPU acceleration
    enable_memory_optimization=True,    # Enable memory optimization
    enable_parallel=True,               # Enable parallel processing
    cache_dir='/path/to/cache',         # Custom cache directory
    max_memory_gb=16.0                  # Memory limit in GB
)

# Optimize system resources
memory_stats = regime_detector.optimize_memory_usage()
cpu_stats = regime_detector.optimize_cpu_usage(target_utilization=0.8)

# Perform comprehensive analysis
results = regime_detector.detect_regimes_enhanced(
    data=large_dataset,
    timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
    methods=['unsupervised', 'clustering', 'qualification'],
    use_cache=True,
    parallel=True
)
```

## Performance Monitoring

### Performance Statistics

The system provides comprehensive performance monitoring:

```python
# Get performance statistics
stats = regime_detector.get_performance_stats()

print(f"Total detections: {stats['total_detections']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Parallel detections: {stats['parallel_detections']}")
print(f"Memory optimized detections: {stats['memory_optimized_detections']}")
print(f"Average detection time: {stats['average_detection_time']:.3f}s")
print(f"Peak memory usage: {stats['peak_memory_usage_mb']:.1f} MB")
```

### Hardware Information

```python
# Get hardware capabilities
stats = regime_detector.get_performance_stats()

if 'gpu_info' in stats:
    print(f"GPU Info: {stats['gpu_info']}")

if 'memory_info' in stats:
    print(f"Memory Info: {stats['memory_info']}")

if 'cpu_info' in stats:
    print(f"CPU Info: {stats['cpu_info']}")
```

## Advanced Validation Results

### Cross-Validation Results

```python
results = regime_detector.detect_regimes_enhanced(...)

# Access cross-validation results
cv_results = results['cross_validation']
for key, cv_result in cv_results.items():
    print(f"{key}: Stability = {cv_result['stability']:.3f}")
    print(f"{key}: Consistency = {cv_result['consistency']:.3f}")
```

### Out-of-Sample Results

```python
# Access out-of-sample validation results
oos_results = results['out_of_sample']
for key, oos_result in oos_results.items():
    print(f"{key}: OOS Score = {oos_result['oos_score']:.3f}")
    print(f"{key}: Similarity = {oos_result['similarity']:.3f}")
    print(f"{key}: Accuracy = {oos_result['accuracy']:.3f}")
```

### Regime Persistence Results

```python
# Access regime persistence analysis
persistence_results = results['persistence']
for key, persistence_result in persistence_results.items():
    print(f"{key}: Average Stability = {persistence_result['average_stability']:.3f}")
    print(f"{key}: Mean Duration = {persistence_result['regime_duration']['mean_duration']:.1f}")
    print(f"{key}: Max Duration = {persistence_result['regime_duration']['max_duration']:.1f}")
```

## Best Practices

### 1. Memory Management

- **Set appropriate memory limits** based on your system capabilities
- **Use memory optimization** for large datasets
- **Monitor memory usage** during processing
- **Clear cache periodically** to free up space

### 2. Parallel Processing

- **Enable parallel processing** for multiple timeframes/methods
- **Use appropriate worker counts** based on CPU cores
- **Monitor CPU utilization** to avoid overloading
- **Consider memory constraints** when using parallel processing

### 3. Caching Strategy

- **Use caching** for repeated analyses with similar parameters
- **Clear cache** when data or parameters change significantly
- **Monitor cache hit rates** to optimize cache usage
- **Use persistent cache directories** for long-running analyses

### 4. Validation

- **Always use cross-validation** for regime stability assessment
- **Perform out-of-sample testing** for realistic validation
- **Analyze regime persistence** for understanding regime behavior
- **Monitor validation metrics** to ensure quality

## Troubleshooting

### Common Issues

#### Memory Issues
```python
# Reduce memory usage
regime_detector = get_enhanced_tas_regime_detection(
    max_memory_gb=4.0,  # Reduce memory limit
    enable_memory_optimization=True
)

# Optimize memory usage
memory_stats = regime_detector.optimize_memory_usage()
```

#### Performance Issues
```python
# Disable GPU if causing issues
regime_detector = get_enhanced_tas_regime_detection(
    enable_gpu=False,
    enable_parallel=True
)

# Optimize CPU usage
cpu_stats = regime_detector.optimize_cpu_usage(target_utilization=0.6)
```

#### Cache Issues
```python
# Clear cache if corrupted
regime_detector.clear_cache()

# Disable caching if causing issues
results = regime_detector.detect_regimes_enhanced(
    use_cache=False
)
```

### Performance Optimization

#### For Large Datasets
```python
# Use memory optimization and chunked processing
regime_detector = get_enhanced_tas_regime_detection(
    enable_memory_optimization=True,
    max_memory_gb=8.0
)

# Process in smaller batches
for timeframe in ['1h', '4h', '1d']:
    results = regime_detector.detect_regimes_enhanced(
        data=large_dataset,
        timeframes=[timeframe],
        methods=['unsupervised'],
        use_cache=True,
        parallel=True
    )
```

#### For Real-time Processing
```python
# Use caching and parallel processing
regime_detector = get_enhanced_tas_regime_detection(
    enable_parallel=True,
    cache_dir='./realtime_cache'
)

# Enable caching for repeated analyses
results = regime_detector.detect_regimes_enhanced(
    data=realtime_data,
    timeframes=['1m', '5m'],
    methods=['unsupervised'],
    use_cache=True,
    parallel=True
)
```

## Examples

### Complete Example

```python
import pandas as pd
import numpy as np
from src.utils.ml_common.optimization.tas.enhanced_regime_detection import get_enhanced_tas_regime_detection

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=10000, freq='1H')
data = pd.DataFrame({
    'price': np.cumsum(np.random.normal(0.001, 0.01, 10000)),
    'volume': np.random.lognormal(10, 0.5, 10000),
    'volatility': np.random.gamma(2, 0.01, 10000)
}, index=dates)

# Initialize enhanced regime detection
regime_detector = get_enhanced_tas_regime_detection(
    enable_gpu=True,
    enable_memory_optimization=True,
    enable_parallel=True,
    cache_dir='./tas_cache',
    max_memory_gb=8.0
)

# Perform comprehensive regime detection
results = regime_detector.detect_regimes_enhanced(
    data=data,
    timeframes=['1h', '4h', '1d'],
    methods=['unsupervised', 'clustering', 'qualification'],
    use_cache=True,
    parallel=True
)

# Analyze results
print(f"Total analyses: {len(results)}")
print(f"Cross-validation results: {len(results['cross_validation'])}")
print(f"Out-of-sample results: {len(results['out_of_sample'])}")
print(f"Persistence results: {len(results['persistence'])}")

# Get performance statistics
stats = regime_detector.get_performance_stats()
print(f"Execution time: {stats['average_detection_time']:.3f}s")
print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.3f}")
```

## Conclusion

The Enhanced TAS Regime Detection system provides a comprehensive solution for regime detection with advanced performance optimizations and validation capabilities. By following the best practices outlined in this guide, you can achieve optimal performance and reliability for your regime detection analyses.

For more information, see the example files and test suites in the `examples/` and `tests/` directories.