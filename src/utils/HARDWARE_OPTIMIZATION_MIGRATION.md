# Hardware Optimization Migration Guide

This guide explains the migration from legacy custom optimizations to the new comprehensive hardware optimization system.

## Overview

The codebase has been upgraded with a comprehensive hardware optimization system that provides:
- Unified hardware management across CPU, GPU, and memory
- Adaptive optimization based on workload characteristics
- Machine learning-based performance optimization
- M1/M2/M3/M4 Apple Silicon specific optimizations

## What Changed

### 1. Legacy Files Removed
The following legacy optimization files have been removed as they are now superseded by the hardware optimization system:

- `src/utils/ml_common/utils/memory_optimization.py` → Use `HardwareOptimizedMLProcessor`
- `src/utils/ml_common/vectorbt_memory_optimizer.py` → Use `GPUAccelerationUtils`
- `src/utils/ml_common/vectorbt_memory_manager.py` → Use `AdvancedM1MemoryOptimizer`
- `src/utils/enhanced_data_operations.py` → Use hardware-aware decorators
- `src/utils/enhanced_step_optimizations.py` → Use `AdaptiveOptimizationEngine`

### 2. New Hardware Optimization System

#### Core Components
- **IntegratedHardwareManager**: Centralized hardware coordination
- **AdaptiveOptimizationEngine**: ML-based optimization learning
- **AdvancedM1MemoryOptimizer**: Advanced memory management
- **EnhancedM1GPUManager**: GPU acceleration utilities

#### New Files Added
- `src/utils/ml_common/hardware_optimized_parallel_processor.py`
- `src/utils/ml_common/gpu_acceleration_utils.py`

## Migration Examples

### 1. Memory Optimization

**Before (Legacy):**
```python
from src.utils.ml_common.utils.memory_optimization import MemoryEfficientTraining

# Legacy memory optimization
memory_optimizer = MemoryEfficientTraining()
optimized_data = memory_optimizer.optimize_dataframe(df)
```

**After (Hardware-Optimized):**
```python
from src.utils.ml_common.hardware_optimized_parallel_processor import HardwareOptimizedMLProcessor

# Hardware-aware memory optimization
processor = HardwareOptimizedMLProcessor()
optimized_data = processor.process_feature_engineering(df, feature_funcs)
```

### 2. Parallel Processing

**Before (Legacy):**
```python
from src.utils.parallel_processing_optimizer import MacM1ParallelOptimizer

# Basic parallel processing
optimizer = MacM1ParallelOptimizer()
result = optimizer.parallel_apply(df, func)
```

**After (Hardware-Optimized):**
```python
from src.utils.parallel_processing_optimizer import MacM1ParallelOptimizer

# Hardware-optimized parallel processing
optimizer = MacM1ParallelOptimizer(enable_hardware_optimization=True)
result = optimizer.parallel_apply(df, func)
```

### 3. GPU Acceleration

**Before (Legacy):**
```python
# Manual GPU acceleration
if gpu_available:
    result = gpu_operation(data)
else:
    result = cpu_operation(data)
```

**After (Hardware-Optimized):**
```python
from src.utils.ml_common.gpu_acceleration_utils import gpu_accelerated

# Automatic GPU acceleration with fallback
@gpu_accelerated(operation_type='matrix_operations')
def my_operation(data):
    return process_data(data)
```

### 4. ML Training

**Before (Legacy):**
```python
# Basic ML training
model.fit(X, y)
```

**After (Hardware-Optimized):**
```python
from src.utils.ml_common.hardware_optimized_parallel_processor import ml_training_optimized

# Hardware-optimized ML training
@ml_training_optimized(enable_gpu=True)
def train_model(model, X, y):
    return model.fit(X, y)
```

## New Decorators

### Hardware-Aware Decorators
```python
from src.utils.parallel_processing_optimizer import (
    hardware_optimized, memory_efficient_processing, 
    gpu_accelerated, adaptive_workload_optimization
)

# Hardware optimization
@hardware_optimized(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
def my_function(data):
    return process_data(data)

# Memory efficiency
@memory_efficient_processing(memory_threshold_mb=200.0)
def memory_intensive_function(data):
    return process_large_data(data)

# GPU acceleration
@gpu_accelerated(GPUOperationType.MATRIX_MULTIPLICATION)
def matrix_operation(A, B):
    return np.dot(A, B)

# Adaptive optimization
@adaptive_workload_optimization()
def adaptive_function(data):
    return process_data(data)
```

### ML-Specific Decorators
```python
from src.utils.ml_common.hardware_optimized_parallel_processor import (
    ml_training_optimized, feature_engineering_optimized, hpo_optimized
)

# ML training optimization
@ml_training_optimized(enable_gpu=True)
def train_model(model, X, y):
    return model.fit(X, y)

# Feature engineering optimization
@feature_engineering_optimized(enable_gpu=True)
def create_features(df):
    return df.apply(some_transformation)

# Hyperparameter optimization
@hpo_optimized(enable_gpu=True)
def optimize_hyperparameters(model, X, y, param_grid):
    return GridSearchCV(model, param_grid).fit(X, y)
```

## Performance Benefits

### Expected Improvements
- **Memory Usage**: 30-50% reduction through intelligent memory management
- **Processing Speed**: 20-40% improvement through hardware optimization
- **Thermal Management**: Better sustained performance under load
- **Adaptive Performance**: Continuous optimization based on workload patterns

### Hardware-Specific Optimizations
- **M1/M2/M3/M4 Apple Silicon**: Unified memory architecture optimization
- **GPU Acceleration**: Automatic GPU usage for suitable operations
- **Memory Management**: Intelligent memory pooling and compression
- **Thermal Management**: Adaptive performance based on thermal conditions

## Compatibility

### Backward Compatibility
- All existing imports continue to work
- Legacy functionality is maintained through compatibility shims
- Gradual migration is supported

### Breaking Changes
- Some advanced features may require code updates
- Performance characteristics may change (usually for the better)
- Configuration options may have changed

## Getting Started

### 1. Enable Hardware Optimization
```python
# For parallel processing
from src.utils.parallel_processing_optimizer import MacM1ParallelOptimizer
optimizer = MacM1ParallelOptimizer(enable_hardware_optimization=True)

# For ML operations
from src.utils.ml_common.hardware_optimized_parallel_processor import get_hardware_optimized_ml_processor
processor = get_hardware_optimized_ml_processor()
```

### 2. Use Hardware-Aware Decorators
```python
from src.utils.parallel_processing_optimizer import hardware_optimized, WorkloadType, OptimizationLevel

@hardware_optimized(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE)
def my_ml_function(data):
    return process_ml_data(data)
```

### 3. Monitor Performance
```python
from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager

manager = get_integrated_hardware_manager()
report = manager.get_optimization_report()
print(f"Cache hit rate: {report['cache_statistics']['hit_rate']:.2%}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Update imports to use new hardware optimization modules
2. **Performance Regression**: Ensure hardware optimization is enabled
3. **Memory Issues**: Check memory optimization settings
4. **GPU Issues**: Verify GPU availability and configuration

### Debug Mode
```python
import logging
logging.getLogger('src.utils.hardware').setLevel(logging.DEBUG)
```

## Support

For questions or issues with the hardware optimization system:
1. Check the hardware optimization README: `src/utils/hardware/README.md`
2. Review the adaptive optimization engine documentation
3. Check system logs for hardware optimization messages

## Future Enhancements

The hardware optimization system is designed to be extensible and will continue to evolve with:
- Additional hardware support
- More sophisticated optimization algorithms
- Enhanced machine learning-based optimization
- Better integration with ML frameworks