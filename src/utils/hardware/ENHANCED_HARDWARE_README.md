# Enhanced Hardware Optimizations for Apple Silicon

This module provides comprehensive hardware optimization specifically designed for M1/M2/M3/M4 Apple Silicon chips, with full backward compatibility and advanced features for financial data processing.

## 🚀 New Features

### 🎮 GPU Acceleration for VectorBT
- **Metal Performance Shaders Integration**: Native GPU acceleration for VectorBT operations
- **Portfolio Analysis**: GPU-accelerated portfolio optimization and risk metrics
- **Signal Generation**: High-performance technical indicator calculations
- **Backward Compatibility**: Existing VectorBT code works without modification

### ⚡ Advanced CPU Optimization
- **Performance/Efficiency Core Management**: Intelligent workload distribution across M1 cores
- **Thermal Management**: Real-time thermal monitoring and throttling prevention
- **Power Management**: Dynamic power scaling based on workload requirements
- **Workload Profiling**: Learning-based optimization for different workload types

### 🧠 Unified Memory Architecture
- **Cross-Component Sharing**: Intelligent memory sharing between CPU, GPU, and Neural Engine
- **Memory Deduplication**: Automatic detection and elimination of duplicate data
- **Compression**: Advanced compression algorithms for memory efficiency
- **Memory Pooling**: Efficient memory allocation and management

### 🔄 Adaptive Optimization Engine
- **Machine Learning**: Random Forest and Neural Network-based optimization
- **Performance Learning**: Learns from execution patterns and adapts strategies
- **Workload Classification**: Automatic workload categorization and optimization
- **Real-time Adaptation**: Continuously adjusts optimization parameters

### 🗄️ Intelligent Caching System
- **Advanced Eviction Policies**: LRU, LFU, TTL, and adaptive strategies
- **Compression**: Automatic compression for large cached items
- **Memory Monitoring**: Real-time memory usage tracking and optimization
- **Statistics**: Comprehensive hit/miss rate tracking and analysis

### 📊 Performance Monitoring
- **Comprehensive Metrics**: Detailed performance tracking across all components
- **Real-time Monitoring**: Live performance metrics and system status
- **Optimization Reports**: Detailed analysis of optimization effectiveness
- **Hardware Status**: Complete hardware utilization and health monitoring

## 🛠️ Installation and Setup

### Prerequisites
```bash
# Required packages
pip install numpy pandas psutil

# Optional packages for enhanced features
pip install torch scikit-learn lz4
```

### Basic Usage
```python
from src.utils.hardware import (
    gpu_vectorbt_optimization, cpu_optimized_feature_correlation,
    unified_memory_feature_processing, adaptive_feature_selection,
    get_hardware_optimization_status
)

# GPU-accelerated VectorBT optimization
result = gpu_vectorbt_optimization(price_data, features)

# CPU-optimized feature correlation
correlation_matrix = cpu_optimized_feature_correlation(data)

# Unified memory optimization
optimized_data = unified_memory_feature_processing(data, 'feature_selection', 'gpu')

# Adaptive feature selection
selected_features = adaptive_feature_selection(data, learn_from_execution=True)

# Get system status
status = get_hardware_optimization_status()
```

## 🎯 Decorator Usage

### GPU Acceleration
```python
from src.utils.hardware import gpu_accelerated

@gpu_accelerated("matrix_multiplication")
def matrix_multiply(A, B):
    return np.dot(A, B)
```

### CPU Optimization
```python
from src.utils.hardware import optimize_cpu_execution

@optimize_cpu_execution("cpu_intensive")
def cpu_intensive_task(data):
    return np.sum(data ** 2)
```

### Unified Memory Optimization
```python
from src.utils.hardware import unified_memory_optimized

@unified_memory_optimized('feature_selection', 'gpu')
def process_features(data):
    return data * 2
```

### Adaptive Optimization
```python
from src.utils.hardware import adaptive_optimization

@adaptive_optimization(learn_from_execution=True)
def adaptive_processing(data):
    return np.sum(data, axis=1)
```

### Smart Caching
```python
from src.utils.hardware import smart_cache

@smart_cache(ttl=3600, max_size=1000, compression=True)
def expensive_calculation(data):
    return complex_computation(data)
```

### Performance Tracking
```python
from src.utils.hardware import performance_tracked

@performance_tracked(['execution_time', 'memory_usage', 'cpu_utilization'])
def tracked_function(data):
    return process_data(data)
```

### Comprehensive Memory Optimization
```python
from src.utils.hardware import comprehensive_memory_optimization

@comprehensive_memory_optimization(
    int64_to_int32=True,
    float64_to_float32=True,
    object_to_category=True,
    compression_ratio=0.7
)
def memory_optimized_function(data):
    return data * 2
```

## 📈 Real-World Examples

### Financial Portfolio Analysis
```python
import numpy as np
from src.utils.hardware import gpu_vectorbt_optimization

# Generate sample financial data
returns = np.random.normal(0.001, 0.02, (1000, 10))
weights = np.random.dirichlet(np.ones(10))

# GPU-accelerated portfolio analysis
result = gpu_vectorbt_optimization(returns, {
    'weights': weights,
    'risk_free_rate': 0.02,
    'lookback_period': 252
})

print(f"Sharpe ratio: {result['sharpe_ratio']:.3f}")
print(f"Volatility: {result['volatility']:.3f}")
```

### Feature Engineering Pipeline
```python
import pandas as pd
from src.utils.hardware import (
    cpu_optimized_feature_correlation,
    unified_memory_feature_processing,
    adaptive_feature_selection
)

# Load financial data
df = pd.read_csv('financial_data.csv')

# CPU-optimized correlation analysis
correlation_matrix = cpu_optimized_feature_correlation(df.values)

# Unified memory optimization
optimized_data = unified_memory_feature_processing(
    df.values, 'feature_selection', 'gpu'
)

# Adaptive feature selection
selected_features = adaptive_feature_selection(
    optimized_data, learn_from_execution=True
)
```

### Machine Learning Workflow
```python
from src.utils.hardware import (
    gpu_accelerated, optimize_cpu_execution,
    unified_memory_optimized, adaptive_optimization
)

@gpu_accelerated("neural_network")
@optimize_cpu_execution("cpu_intensive")
@unified_memory_optimized('neural_inference', 'gpu')
@adaptive_optimization(learn_from_execution=True)
def ml_pipeline(data):
    # Preprocessing
    processed_data = preprocess(data)
    
    # Feature engineering
    features = engineer_features(processed_data)
    
    # Model training
    model = train_model(features)
    
    # Prediction
    predictions = model.predict(features)
    
    return predictions
```

## 🔧 Configuration

### GPU Configuration
```python
from src.utils.hardware import VectorBTGPUConfig, get_vectorbt_gpu_accelerator

config = VectorBTGPUConfig(
    enable_portfolio_optimization=True,
    enable_signal_acceleration=True,
    batch_size=1000,
    max_parallel_operations=4
)

accelerator = get_vectorbt_gpu_accelerator(config)
```

### CPU Configuration
```python
from src.utils.hardware import EnhancedCPUConfig, get_enhanced_cpu_optimizer

config = EnhancedCPUConfig(
    enable_thermal_management=True,
    enable_power_management=True,
    power_mode=PowerMode.BALANCED,
    enable_workload_profiling=True
)

optimizer = get_enhanced_cpu_optimizer(config)
```

### Memory Configuration
```python
from src.utils.hardware import EnhancedUnifiedMemoryConfig, get_enhanced_unified_memory_manager

config = EnhancedUnifiedMemoryConfig(
    enable_cross_component_sharing=True,
    enable_memory_deduplication=True,
    enable_aggressive_optimization=True,
    compression_algorithm=MemoryCompressionType.LZ4
)

manager = get_enhanced_unified_memory_manager(config)
```

## 📊 Performance Monitoring

### Get System Status
```python
from src.utils.hardware import get_hardware_optimization_status

status = get_hardware_optimization_status()
print(f"GPU available: {status['vectorbt_gpu_metrics']['gpu_available']}")
print(f"Memory usage: {status['unified_memory_metrics']['current_usage_mb']:.1f} MB")
```

### Performance Metrics
```python
from src.utils.hardware import (
    get_vectorbt_gpu_performance_metrics,
    get_enhanced_cpu_performance_metrics,
    get_enhanced_unified_memory_stats,
    get_adaptive_optimization_metrics
)

# GPU metrics
gpu_metrics = get_vectorbt_gpu_performance_metrics()
print(f"GPU operations: {gpu_metrics['vectorbt_gpu_metrics']['total_operations']}")

# CPU metrics
cpu_metrics = get_enhanced_cpu_performance_metrics()
print(f"CPU optimizations: {cpu_metrics['enhanced_cpu_metrics']['total_operations']}")

# Memory metrics
memory_metrics = get_enhanced_unified_memory_stats()
print(f"Memory allocations: {memory_metrics['enhanced_allocations']}")

# Adaptive optimization metrics
adaptive_metrics = get_adaptive_optimization_metrics()
print(f"Learning enabled: {adaptive_metrics['learning_enabled']}")
```

## 🔄 Backward Compatibility

All existing code continues to work without modification. The new features are available through:

1. **Same function names**: `gpu_vectorbt_optimization`, `cpu_optimized_feature_correlation`, etc.
2. **Enhanced implementations**: Automatic fallback to CPU when GPU is not available
3. **Progressive enhancement**: New features are opt-in through decorators and configuration
4. **Error handling**: Graceful degradation when advanced features are not available

### Migration Guide

#### Before (Old Code)
```python
# Old VectorBT code
def portfolio_analysis(returns, weights):
    portfolio_returns = np.sum(returns * weights, axis=1)
    return {
        'mean_return': np.mean(portfolio_returns),
        'volatility': np.std(portfolio_returns)
    }
```

#### After (Enhanced Code)
```python
# Enhanced code with GPU acceleration
from src.utils.hardware import gpu_vectorbt_optimization

def portfolio_analysis(returns, weights):
    return gpu_vectorbt_optimization(returns, {'weights': weights})
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU not available**
   - Check MPS availability: `torch.backends.mps.is_available()`
   - System automatically falls back to CPU

2. **Memory issues**
   - Enable memory optimization: `comprehensive_memory_optimization`
   - Use unified memory: `unified_memory_feature_processing`

3. **Performance issues**
   - Enable performance tracking: `@performance_tracked`
   - Check system status: `get_hardware_optimization_status()`

### Debug Mode
```python
import logging
logging.getLogger('src.utils.hardware').setLevel(logging.DEBUG)
```

## 📚 API Reference

### Core Functions
- `gpu_vectorbt_optimization(price_data, features)` - GPU-accelerated VectorBT operations
- `cpu_optimized_feature_correlation(data)` - CPU-optimized correlation analysis
- `unified_memory_feature_processing(data, operation_type, component)` - Unified memory optimization
- `adaptive_feature_selection(data, learn_from_execution)` - Adaptive feature selection

### Decorators
- `@gpu_accelerated(operation_type)` - GPU acceleration decorator
- `@optimize_cpu_execution(workload_type)` - CPU optimization decorator
- `@unified_memory_optimized(operation_type, component)` - Unified memory decorator
- `@adaptive_optimization(learn_from_execution)` - Adaptive optimization decorator
- `@smart_cache(ttl, max_size, compression)` - Intelligent caching decorator
- `@performance_tracked(metrics)` - Performance tracking decorator
- `@comprehensive_memory_optimization(...)` - Memory optimization decorator

### Configuration Classes
- `VectorBTGPUConfig` - GPU acceleration configuration
- `EnhancedCPUConfig` - CPU optimization configuration
- `EnhancedUnifiedMemoryConfig` - Memory management configuration

### Manager Classes
- `VectorBTGPUAccelerator` - GPU acceleration manager
- `EnhancedCPUOptimizer` - CPU optimization manager
- `EnhancedUnifiedMemoryManager` - Memory management manager
- `AdaptiveOptimizationEngine` - Adaptive optimization manager

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure backward compatibility
5. Submit a pull request

## 📄 License

This module is part of the Ares Trading System and is subject to the same license terms.

## 🙏 Acknowledgments

- Apple Silicon optimization techniques
- Metal Performance Shaders framework
- PyTorch MPS backend
- Scikit-learn machine learning algorithms
- NumPy and Pandas data processing libraries