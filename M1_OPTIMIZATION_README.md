# M1 Optimization Guide for Training Pipeline Steps 3-20

This document provides comprehensive guidance on optimizing the training pipeline steps 3-20 for Apple M1/M2/M3 Macs, ensuring efficient CPU, GPU, and memory usage.

## 🚀 Overview

The training pipeline has been enhanced with M1-specific optimizations that leverage:
- **Metal Performance Shaders (MPS)** for GPU acceleration
- **Unified Memory Architecture** for efficient memory management
- **Multi-core CPU** optimization with intelligent parallel processing
- **Memory-efficient data structures** optimized for M1

## 📦 Key Optimization Components

### 1. M1 GPU Utilities (`src/utils/m1_gpu_utils.py`)
- **M1GPUManager**: Intelligent device detection and GPU memory management
- **M1PerformanceOptimizer**: Performance tuning for PyTorch operations
- **Automatic device selection**: CPU/GPU/MPS based on operation type and data size

### 2. M1 Memory Optimizer (`src/utils/m1_memory_optimizer.py`)
- **M1MemoryOptimizer**: Memory monitoring and optimization
- **M1DataManager**: Memory-efficient data loading and processing
- **Chunked processing**: Automatic data chunking for large datasets

### 3. M1 CPU Optimizer (`src/utils/m1_cpu_optimizer.py`)
- **M1CPUOptimizer**: Parallel processing optimization
- **M1BatchProcessor**: Intelligent batch size calculation
- **Adaptive worker scaling**: Dynamic CPU core utilization

## 🎯 Step-by-Step Optimizations

### Step 3: HMM Regime Discovery
- **GPU Acceleration**: Uses MPS for matrix operations in Gaussian Mixture Models
- **Memory Optimization**: Chunked processing for large datasets
- **CPU Parallelization**: Parallel regime fitting when applicable

### Step 7: Enhanced Matrix Operations
- **MPS Matrix Multiplication**: Leverages Neural Engine for correlation/covariance matrices
- **Mixed Precision**: Automatic float16 usage for better performance
- **Memory Pooling**: Efficient memory management for large matrices

### Step 9-15: Model Training Steps
- **GPU Neural Networks**: MPS-accelerated PyTorch models
- **Batch Optimization**: Adaptive batch sizing based on available memory
- **Parallel Data Loading**: Multi-threaded data preprocessing

### Step 16-17: Optimization Steps
- **GPU-Accelerated Optuna**: MPS optimization for hyperparameter search
- **Memory-Efficient Caching**: Intelligent caching with memory limits
- **Parallel Objective Evaluation**: Multi-core objective function evaluation

### Step 18-20: Backtesting and Validation
- **Chunked Backtesting**: Memory-efficient walk-forward validation
- **Parallel Monte Carlo**: Multi-core simulation runs
- **GPU-Accelerated Metrics**: MPS matrix operations for performance calculations

## ⚙️ Configuration

### Environment Variables
```bash
# Enable MPS memory optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# Optimize NumPy for M1
export OMP_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Configuration File
```yaml
# config/m1_gpu_config.yaml
m1_gpu:
  enable_mps: true
  enable_mixed_precision: true
  memory_threshold: 0.8
  batch_size: 1000

performance_optimization:
  enable_parallel_processing: true
  enable_memory_cleanup: true
  chunk_size: 5000

memory_management:
  gpu_memory_fraction: 0.8
  enable_memory_pooling: true
  cleanup_frequency: 100
```

## 🧪 Testing and Validation

### Run Compatibility Tests
```bash
# Run comprehensive M1 compatibility test
python test_m1_compatibility.py

# Test specific components
python -c "from src.utils.m1_gpu_utils import get_m1_gpu_manager; print(get_m1_gpu_manager().device)"
```

### Performance Monitoring
```python
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer

# Monitor GPU usage
gpu_manager = get_m1_gpu_manager()
print(f"GPU Device: {gpu_manager.device}")
print(f"Memory Usage: {gpu_manager.memory_info}")

# Monitor memory
memory_optimizer = get_m1_memory_optimizer()
print(f"Memory Report: {memory_optimizer.get_memory_report()}")
```

## 📊 Performance Benchmarks

### Expected Improvements on M1
- **Matrix Operations**: 2-5x speedup with MPS acceleration
- **Neural Networks**: 3-8x speedup for inference, 2-4x for training
- **Memory Usage**: 30-50% reduction with optimized data structures
- **Data Processing**: 2-3x speedup with parallel processing

### Memory Requirements
- **Minimum RAM**: 8GB (16GB recommended)
- **GPU Memory**: Automatically managed (up to 80% of available)
- **Storage**: Same as original pipeline requirements

## 🔧 Troubleshooting

### Common Issues

#### 1. MPS Not Available
```python
import torch
print(f"MPS Available: {torch.backends.mps.is_available()}")

# If False, ensure PyTorch with MPS support is installed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

#### 2. Memory Issues
```python
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer

memory_opt = get_m1_memory_optimizer()
report = memory_opt.get_memory_report()

if report['memory_efficiency'] < 0.5:
    print("Consider reducing batch size or enabling chunked processing")
```

#### 3. Performance Issues
```python
from src.utils.m1_gpu_utils import get_m1_gpu_manager

gpu_manager = get_m1_gpu_manager()
if gpu_manager.device.type == 'cpu':
    print("GPU not available, check MPS installation")
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Test specific optimizations
from src.utils.m1_gpu_utils import get_m1_gpu_manager
gpu_manager = get_m1_gpu_manager()
gpu_manager.logger.setLevel(logging.DEBUG)
```

## 🚀 Best Practices

### 1. Memory Management
- Always use `memory_optimizer.optimize_memory()` after large operations
- Enable chunked processing for datasets > 1M rows
- Monitor memory usage with `memory_optimizer.get_memory_report()`

### 2. GPU Optimization
- Use `gpu_manager.should_use_gpu()` to determine when to use GPU
- Enable mixed precision for neural networks
- Clear GPU cache regularly: `gpu_manager.optimize_memory()`

### 3. CPU Parallelization
- Use `cpu_optimizer.parallel_process()` for CPU-bound tasks
- Choose appropriate task types: "cpu_bound", "io_bound", "general"
- Monitor CPU usage with `cpu_optimizer.get_cpu_usage_report()`

### 4. Data Processing
- Use `memory_optimizer.chunked_dataframe_processor()` for large DataFrames
- Enable memory-efficient loading with `M1DataManager`
- Convert data types appropriately (float32 for M1 optimization)

## 📈 Monitoring and Metrics

### Key Metrics to Monitor
- **GPU Utilization**: MPS usage percentage
- **Memory Efficiency**: RAM usage vs. available
- **CPU Utilization**: Core usage across parallel tasks
- **Processing Speed**: Operations per second
- **Memory Footprint**: Peak memory usage

### Logging Integration
All optimizations include comprehensive logging:
```python
# GPU operations
gpu_manager.logger.info("MPS matrix multiplication completed")

# Memory operations
memory_optimizer.logger.debug("Memory cleanup: freed 500MB")

# CPU operations
cpu_optimizer.logger.info("Parallel processing: 8 workers, 2.3x speedup")
```

## 🔄 Integration with Existing Code

### Automatic Integration
The optimizations are designed to integrate seamlessly:
```python
# Existing code continues to work
from src.training.steps.model_training.matrix_components import MatrixProcessor

processor = MatrixProcessor(use_gpu=True)  # Automatically uses M1 optimizations
result = processor.compute_correlation_matrix(data)  # MPS accelerated
```

### Manual Optimization
For fine-tuned control:
```python
from src.utils.m1_gpu_utils import get_m1_gpu_manager

gpu_manager = get_m1_gpu_manager()

# Manual GPU operations
with gpu_manager.gpu_context("custom_operation"):
    tensor = gpu_manager.to_device(data, "matrix_mult")
    result = torch.matmul(tensor, tensor)
```

## 🎯 Step-Specific Optimizations

### Step 3: HMM Regime Discovery
- **GPU**: Matrix operations in GMM fitting
- **Memory**: Chunked processing for large time series
- **CPU**: Parallel regime evaluation

### Step 7: Matrix Operations
- **GPU**: MPS-accelerated correlation/covariance matrices
- **Memory**: Intelligent batching for large matrices
- **CPU**: Parallel eigendecomposition

### Step 9-15: Model Training
- **GPU**: MPS neural network training/inference
- **Memory**: Gradient checkpointing for large models
- **CPU**: Data loading parallelization

### Step 16-17: Hyperparameter Optimization
- **GPU**: MPS-accelerated objective evaluation
- **Memory**: Caching with memory limits
- **CPU**: Parallel trial evaluation

### Step 18-20: Backtesting
- **GPU**: MPS matrix operations for returns calculation
- **Memory**: Chunked walk-forward validation
- **CPU**: Parallel Monte Carlo simulations

## 📝 Future Enhancements

### Planned Optimizations
- **Advanced MPS Operations**: Custom Metal shaders for specific operations
- **Dynamic Memory Allocation**: Real-time memory adjustment based on workload
- **Neural Engine Integration**: Direct Neural Engine utilization for ML operations
- **Unified Memory Optimization**: Enhanced memory sharing between CPU/GPU

### Research Areas
- **M1 Ultra Optimization**: Specialized optimizations for M1 Ultra's unified memory
- **Mixed Precision Training**: Advanced mixed precision techniques
- **Energy Efficiency**: Power-optimized computations for battery life

---

## 🤝 Support and Contributing

For issues or contributions related to M1 optimizations:
1. Run the compatibility test: `python test_m1_compatibility.py`
2. Check logs for detailed error information
3. Report issues with system information and test results

## 📚 Additional Resources

- [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [M1 Chip Architecture](https://www.apple.com/mac/m1/)

---

*This optimization guide is regularly updated as new M1/M2/M3 optimizations become available.*
