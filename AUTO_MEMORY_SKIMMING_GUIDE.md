# Auto Memory Skimming Guide

## Overview
This guide explains how to use the automatic memory skimming capabilities in the M1 Memory Optimizer to automatically free unused memory whenever more memory is needed.

## Key Features

### 🧠 **Smart Memory Allocation**
Automatically checks available memory and skims unused memory when needed.

### 🔍 **Auto Memory Skimming**
Three-tier memory cleanup system:
1. **Light Cleanup** - Quick and safe operations
2. **Moderate Cleanup** - More aggressive but safe operations  
3. **Aggressive Cleanup** - Maximum memory recovery

### 🎯 **Context Managers**
Easy-to-use context managers for automatic memory management.

### 🏷️ **Decorators**
Automatic memory skimming for functions with decorators.

## Usage Examples

### 1. **Basic Auto Memory Skimming**

```python
from src.utils.m1_memory_optimizer import auto_skim_memory

# Automatically skim memory when you need 2GB
result = auto_skim_memory(required_memory_mb=2048, operation_type="neural_net")

if result['skimming_needed']:
    print(f"✅ Freed {result['memory_freed_mb']:.1f}MB")
    print(f"📊 Available: {result['available_mb']:.1f}MB")
else:
    print("✅ Sufficient memory available")
```

### 2. **Smart Memory Allocation**

```python
from src.utils.m1_memory_optimizer import smart_memory_allocation

# Smart allocation with automatic skimming
allocation = smart_memory_allocation(required_memory_mb=1500, operation_type="matrix_mult")

if allocation['allocation_successful']:
    print("✅ Memory allocation successful")
    if allocation['skimming_performed']:
        print(f"🧹 Skimming freed {allocation['skimming_results']['memory_freed_mb']:.1f}MB")
else:
    print("⚠️ Insufficient memory even after skimming")
```

### 3. **Context Managers**

#### **Auto Memory Skim Context**
```python
from src.utils.m1_memory_optimizer import auto_memory_skim_context

# Automatically skim memory before operation
with auto_memory_skim_context(required_memory_mb=1000, operation_type="data_processing") as context:
    if context['skimming_performed']:
        print(f"🧹 Skimming performed: {context['skimming_results']['memory_freed_mb']:.1f}MB freed")
    
    # Your memory-intensive operation here
    large_data = process_large_dataset()
    result = perform_heavy_computation(large_data)
```

#### **Smart Memory Context**
```python
from src.utils.m1_memory_optimizer import smart_memory_context

# Smart memory allocation with automatic skimming
with smart_memory_context(required_memory_mb=2000, operation_type="neural_net") as allocation:
    if allocation['allocation_successful']:
        print("✅ Memory allocation successful")
        
        # Your memory-intensive operation here
        model = train_neural_network()
        predictions = model.predict(test_data)
    else:
        print("⚠️ Insufficient memory")
```

### 4. **Decorators**

#### **Memory Skim Decorator**
```python
from src.utils.m1_memory_optimizer import memory_skim_decorator

# Automatically skim memory before function execution
@memory_skim_decorator(required_memory_mb=1500, operation_type="matrix_mult")
def matrix_multiplication_large(a, b):
    """Large matrix multiplication with automatic memory skimming."""
    return np.dot(a, b)

# Function automatically skims memory before execution
result = matrix_multiplication_large(matrix_a, matrix_b)
```

#### **Auto Memory Skim Decorator**
```python
from src.utils.m1_memory_optimizer import auto_memory_skim_decorator

# Automatically estimates memory requirements and skims if needed
@auto_memory_skim_decorator(operation_type="neural_net")
def train_model(data, labels):
    """Model training with automatic memory estimation and skimming."""
    model = create_model()
    model.fit(data, labels)
    return model

# Function automatically estimates memory needs and skims if necessary
model = train_model(training_data, training_labels)
```

### 5. **Advanced Usage**

#### **Custom Memory Cleanup**
```python
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer

optimizer = get_m1_memory_optimizer()

# Light cleanup
light_result = optimizer._light_memory_cleanup()
print(f"Light cleanup freed {light_result['memory_freed_mb']:.1f}MB")

# Moderate cleanup
moderate_result = optimizer._moderate_memory_cleanup()
print(f"Moderate cleanup freed {moderate_result['memory_freed_mb']:.1f}MB")

# Aggressive cleanup
aggressive_result = optimizer._aggressive_memory_cleanup()
print(f"Aggressive cleanup freed {aggressive_result['memory_freed_mb']:.1f}MB")
```

#### **Memory Monitoring**
```python
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer

optimizer = get_m1_memory_optimizer()

# Check current memory usage
memory_info = optimizer.get_memory_usage()
print(f"Current memory: {memory_info['rss_gb']:.1f}GB")
print(f"Available memory: {memory_info['available_gb']:.1f}GB")

# Check if chunking is needed
should_chunk = optimizer.should_chunk_data(data_size_mb=1000, operation_type="general")
print(f"Chunking needed: {should_chunk}")
```

## Memory Cleanup Levels

### **Light Cleanup** 🧹
- Single garbage collection cycle
- Basic MPS cache clearing
- NumPy memory trimming
- **Use case**: Quick memory recovery for small operations

### **Moderate Cleanup** 🧹🧹
- Multiple garbage collection cycles (3x)
- Aggressive MPS cache clearing (2x)
- CUDA cache clearing
- Internal PyTorch cache clearing
- **Use case**: Medium memory recovery for moderate operations

### **Aggressive Cleanup** 🧹🧹🧹
- Multiple aggressive garbage collection cycles (5x)
- Multiple MPS cache clears (5x)
- Multiple CUDA cache clears (3x)
- Multiple internal PyTorch cache clears (3x)
- System memory cleanup
- Swap optimization
- Memory compression optimization
- **Use case**: Maximum memory recovery for large operations

## Operation Types

| Operation Type | Base Memory (MB) | Use Case |
|----------------|------------------|----------|
| `matrix_mult` | 1000 | Matrix operations, linear algebra |
| `neural_net` | 2000 | Neural network training, deep learning |
| `data_processing` | 500 | Data manipulation, pandas operations |
| `general` | 200 | General operations, default |

## Best Practices

### 1. **Use Context Managers**
```python
# ✅ Good: Use context managers for automatic cleanup
with auto_memory_skim_context(1000, "data_processing"):
    result = process_data()

# ❌ Avoid: Manual memory management
result = process_data()
# Manual cleanup needed
```

### 2. **Choose Appropriate Operation Types**
```python
# ✅ Good: Specify operation type for better estimation
@auto_memory_skim_decorator("neural_net")
def train_model():
    pass

# ❌ Avoid: Using generic operation type for specific operations
@auto_memory_skim_decorator("general")  # Less accurate estimation
def train_model():
    pass
```

### 3. **Monitor Memory Usage**
```python
# ✅ Good: Check memory before large operations
memory_info = optimizer.get_memory_usage()
if memory_info['available_gb'] < 2.0:
    auto_skim_memory(2048, "neural_net")
```

### 4. **Use Decorators for Functions**
```python
# ✅ Good: Use decorators for automatic memory management
@memory_skim_decorator(1500, "matrix_mult")
def large_matrix_operation():
    pass
```

## Error Handling

The auto memory skimming system includes comprehensive error handling:

```python
try:
    result = auto_skim_memory(1000, "general")
except Exception as e:
    print(f"Memory skimming failed: {e}")
    # Fallback to manual cleanup
    optimizer.optimize_memory()
```

## Performance Considerations

- **Light cleanup**: ~10-50ms, minimal impact
- **Moderate cleanup**: ~50-200ms, moderate impact  
- **Aggressive cleanup**: ~200-1000ms, higher impact

Choose the appropriate cleanup level based on your performance requirements.

## Integration with Other Utilities

The auto memory skimming integrates seamlessly with other M1 optimizers:

```python
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import auto_skim_memory

# Coordinate GPU and memory optimization
gpu_manager = get_m1_gpu_manager()
memory_result = auto_skim_memory(1000, "neural_net")

if memory_result['skimming_needed']:
    # Ensure GPU memory is also optimized
    gpu_manager.optimize_memory()
```

## Conclusion

The auto memory skimming system provides:
- **Automatic memory management** - No manual intervention needed
- **Intelligent cleanup levels** - Light, moderate, and aggressive options
- **Easy integration** - Context managers and decorators
- **Comprehensive monitoring** - Detailed memory usage tracking
- **Error handling** - Robust fallback mechanisms

Use these tools to ensure your M1 Mac has optimal memory management for memory-intensive operations! 🚀
