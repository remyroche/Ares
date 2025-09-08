# Step06 Utility Integration Guide

## Overview

This guide documents the extensive utility integration implemented in step06 components, providing a comprehensive framework for dependency injection, utility service management, and performance optimization.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Utility Container System](#utility-container-system)
3. [Dependency Injection Pattern](#dependency-injection-pattern)
4. [Utility Services](#utility-services)
5. [Integration Patterns](#integration-patterns)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling and Graceful Degradation](#error-handling-and-graceful-degradation)
8. [Testing and Validation](#testing-and-validation)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Architecture Overview

The step06 utility integration follows a modular, dependency-injection-based architecture that provides:

- **Centralized Utility Management**: All utility services are managed through a single container
- **Dependency Injection**: Services are automatically injected into methods that need them
- **Configuration-Driven**: Utilities can be enabled/disabled via configuration
- **Performance Optimization**: M1-specific optimizations for Apple Silicon
- **Graceful Degradation**: System continues to work even if some utilities fail
- **Comprehensive Monitoring**: Detailed performance metrics and health reporting

### Key Components

```
Step06UtilityContainer
├── CommonOperationsService
├── DataProcessingService
├── MathValidationService
├── ParquetService
├── SerializationService
├── M1GPUService
├── M1MemoryService
└── M1CPUService
```

## Utility Container System

### Core Container

The `Step06UtilityContainer` is the central hub for managing all utility services:

```python
from src.training.steps.step06_utility_container import (
    Step06UtilityContainer, UtilityConfig, get_utility_container, 
    utility_container_context, inject_utilities
)

# Create configuration
utility_config = UtilityConfig(
    enable_common_operations=True,
    enable_data_processing=True,
    enable_math_validation=True,
    enable_parquet_utils=True,
    enable_serialization=True,
    enable_m1_gpu=True,
    enable_m1_memory=True,
    enable_m1_cpu=True,
    data_processing_chunk_size=10000,
    m1_memory_limit_gb=8.0,
    m1_max_workers=8
)

# Initialize container
container = await get_utility_container(utility_config)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_common_operations` | bool | True | Enable common operations service |
| `enable_data_processing` | bool | True | Enable data processing service |
| `enable_math_validation` | bool | True | Enable math validation service |
| `enable_parquet_utils` | bool | True | Enable parquet utilities service |
| `enable_serialization` | bool | True | Enable serialization service |
| `enable_m1_gpu` | bool | True | Enable M1 GPU optimization |
| `enable_m1_memory` | bool | True | Enable M1 memory optimization |
| `enable_m1_cpu` | bool | True | Enable M1 CPU optimization |
| `data_processing_chunk_size` | int | 10000 | Chunk size for data processing |
| `m1_memory_limit_gb` | float | 8.0 | Memory limit for M1 optimization |
| `m1_max_workers` | int | 8 | Maximum workers for M1 CPU optimization |

## Dependency Injection Pattern

### Decorator-Based Injection

The `@inject_utilities` decorator automatically injects required services into methods:

```python
@inject_utilities('common_ops', 'data_proc', 'math_val', 'parquet', 'serialization')
async def process_data_with_utilities(self, data: pd.DataFrame,
                                     common_ops, data_proc, math_val, parquet, serialization):
    """Process data using injected utility services."""
    
    # Use common operations for validation
    if common_ops:
        validation_result = common_ops.get_operation('validation', 'validate_dataframe')(data)
    
    # Use data processing utilities
    if data_proc and data_proc.validator:
        quality_report = data_proc.validator.validate_dataframe(data)
    
    # Use math validation for safe operations
    if math_val:
        safe_result = math_val.safe_divide(numerator, denominator, default=0.0)
    
    # Use parquet utilities for I/O
    if parquet and parquet.parquet_utils:
        validation_result = parquet.parquet_utils.validate_parquet_file(file_path)
    
    # Use serialization utilities
    if serialization:
        serialization.serializers['json'].save(data, file_path)
```

### Service Access Patterns

Services can be accessed in multiple ways:

```python
# Direct service access
common_ops = container.get_common_operations()
data_proc = container.get_data_processing()
math_val = container.get_math_validation()

# Service-specific methods
common_ops.get_operation('datetime', 'get_current_datetime')()
data_proc.validator.validate_dataframe(data)
math_val.safe_divide(a, b, default=0.0)
```

## Utility Services

### 1. Common Operations Service

Provides common operations for data validation, datetime handling, and basic utilities:

```python
# Data validation
validation_result = common_ops.get_operation('validation', 'validate_dataframe')(data, required_columns)

# Datetime operations
current_time = common_ops.get_operation('datetime', 'get_current_datetime')()

# File operations
file_exists = common_ops.get_operation('file', 'file_exists')(file_path)
```

### 2. Data Processing Service

Advanced data processing capabilities with validation, cleaning, and transformation:

```python
# Data validation
quality_report = data_proc.validator.validate_dataframe(data)

# Data cleaning
cleaned_data = data_proc.cleaner.clean_dataframe(data)

# Data transformation
transformed_data = data_proc.transformer.add_column(data, 'new_column', values)
```

### 3. Math Validation Service

Safe mathematical operations with error handling:

```python
# Safe division
result = math_val.safe_divide(numerator, denominator, default=0.0)

# Safe logarithm
result = math_val.safe_log(value, default=0.0)

# Safe square root
result = math_val.safe_sqrt(value, default=0.0)

# Validation functions
is_finite = math_val.validate_finite(value)
is_positive = math_val.validate_positive(value)
is_in_range = math_val.validate_range(value, min_val, max_val)
```

### 4. Parquet Service

Efficient Parquet file operations with validation:

```python
# Validate parquet file
validation_result = parquet.parquet_utils.validate_parquet_file(file_path)

# Safe read with fallback engines
data = parquet.parquet_utils.safe_read_parquet(file_path)

# Repair corrupted parquet file
repair_result = parquet.parquet_utils.repair_parquet_file(file_path)
```

### 5. Serialization Service

Multi-format serialization with compression support:

```python
# JSON serialization
serialization.serializers['json'].save(data, file_path)
loaded_data = serialization.serializers['json'].load(file_path)

# Pickle serialization
serialization.serializers['pickle'].save(data, file_path)
loaded_data = serialization.serializers['pickle'].load(file_path)

# Universal serialization (auto-detect format)
serialization.serializers['universal'].save(data, file_path)
loaded_data = serialization.serializers['universal'].load(file_path)
```

### 6. M1 GPU Service

Apple Silicon GPU optimization for PyTorch operations:

```python
# GPU device management
tensor = m1_gpu.manager.to_device(tensor, "feature_processing")

# Performance optimization
optimal_batch_size = m1_gpu.optimizer.calculate_optimal_batch_size(data_size)

# Memory optimization
m1_gpu.manager.optimize_memory()
```

### 7. M1 Memory Service

Memory optimization for Apple Silicon unified memory:

```python
# Chunked processing
chunks = list(m1_memory.optimizer.chunked_dataframe_processor(data, chunk_size))

# Memory optimization
m1_memory.optimizer.optimize_memory()

# Memory leak detection
leak_report = m1_memory.optimizer.detect_memory_leaks()
```

### 8. M1 CPU Service

CPU optimization for parallel processing:

```python
# Optimal worker calculation
optimal_workers = m1_cpu.optimizer.calculate_optimal_workers()

# Parallel processing
results = m1_cpu.optimizer.parallel_process(data, process_function)

# Batch processing
optimal_batch_size = m1_cpu.batch_processor.calculate_optimal_batch_size(data_size)
```

## Integration Patterns

### 1. Initialization Pattern

```python
class MyStep06Component:
    def __init__(self, config: Dict[str, Any], utility_config: Optional[UtilityConfig] = None):
        self.config = config
        self.utility_config = utility_config or UtilityConfig()
        self.utility_container = None
    
    async def initialize_utilities(self) -> None:
        """Initialize utility services."""
        self.utility_container = await get_utility_container(self.utility_config)
        
        # Test services
        if self.utility_config.enable_common_operations:
            common_ops = self.utility_container.get_common_operations()
        
        # Get health report
        health_report = self.utility_container.get_health_report()
```

### 2. Context Manager Pattern

```python
async with utility_container_context(utility_config) as container:
    # Use container services
    common_ops = container.get_common_operations()
    data_proc = container.get_data_processing()
    
    # Process data
    result = await process_with_utilities(data, common_ops, data_proc)
```

### 3. Method Injection Pattern

```python
@inject_utilities('common_ops', 'math_val', 'data_proc')
async def process_data(self, data: pd.DataFrame,
                      common_ops, math_val, data_proc):
    """Process data with injected utilities."""
    
    # Validate data
    if common_ops:
        validation_result = common_ops.get_operation('validation', 'validate_dataframe')(data)
    
    # Safe mathematical operations
    if math_val:
        result = math_val.safe_divide(a, b, default=0.0)
    
    # Data processing
    if data_proc and data_proc.validator:
        quality_report = data_proc.validator.validate_dataframe(data)
```

### 4. Cleanup Pattern

```python
async def cleanup(self) -> None:
    """Clean up utility services."""
    if self.utility_container:
        await self.utility_container.cleanup()
        self.utility_container = None
```

## Performance Optimization

### M1-Specific Optimizations

The utility integration includes comprehensive M1 optimization:

#### GPU Optimization
- Automatic MPS (Metal Performance Shaders) detection
- Adaptive precision (fp16, bf16, fp32)
- Memory optimization for unified memory architecture
- Batch size optimization

#### Memory Optimization
- Intelligent memory management
- Chunked processing for large datasets
- Memory leak detection
- Cache clearing and garbage collection tuning

#### CPU Optimization
- Optimal worker calculation for M1 cores
- Parallel processing for CPU-bound tasks
- NumPy optimization
- Batch processing optimization

### Performance Monitoring

```python
# Get performance metrics
performance_metrics = implementation.get_performance_metrics()

# Key metrics
print(f"Total execution time: {performance_metrics['total_execution_time']:.2f}s")
print(f"Utility initialization: {performance_metrics['utility_initialization_time']:.2f}s")
print(f"Data processing: {performance_metrics['data_processing_time']:.2f}s")
print(f"Utility operations: {performance_metrics['utility_operations_count']}")
print(f"Utility errors: {performance_metrics['utility_errors']}")
```

## Error Handling and Graceful Degradation

### Error Handling Strategy

The system implements comprehensive error handling:

1. **Service-Level Errors**: Individual service failures don't crash the system
2. **Graceful Degradation**: Missing services are handled gracefully
3. **Error Tracking**: All errors are logged and tracked in metrics
4. **Fallback Mechanisms**: Alternative approaches when utilities fail

### Error Handling Patterns

```python
@inject_utilities('common_ops', 'data_proc', 'math_val')
async def robust_processing(self, data: pd.DataFrame,
                           common_ops, data_proc, math_val):
    """Robust processing with error handling."""
    
    try:
        # Try to use utilities
        if common_ops:
            validation_result = common_ops.get_operation('validation', 'validate_dataframe')(data)
        else:
            # Fallback validation
            validation_result = self._basic_validation(data)
        
        if math_val:
            result = math_val.safe_divide(a, b, default=0.0)
        else:
            # Fallback safe division
            result = a / b if b != 0 else 0.0
        
        if data_proc and data_proc.validator:
            quality_report = data_proc.validator.validate_dataframe(data)
        else:
            # Fallback quality check
            quality_report = self._basic_quality_check(data)
            
    except Exception as e:
        self.logger.error(f"Processing failed: {e}")
        self.performance_metrics['utility_errors'] += 1
        raise
```

## Testing and Validation

### Comprehensive Test Suite

The utility integration includes a comprehensive test suite:

```python
# Run all tests
python -m pytest src/training/steps/test_step06_utility_integration.py -v

# Run specific test
python -m pytest src/training/steps/test_step06_utility_integration.py::TestStep06UtilityIntegration::test_utility_container_initialization -v
```

### Test Categories

1. **Container Tests**: Utility container initialization and management
2. **Service Tests**: Individual utility service functionality
3. **Integration Tests**: End-to-end integration testing
4. **Performance Tests**: Performance metrics and optimization
5. **Error Handling Tests**: Error scenarios and graceful degradation
6. **Dependency Injection Tests**: Decorator and injection functionality

### Health Monitoring

```python
# Get health report
health_report = container.get_health_report()

# Check status
if health_report['status'] == 'healthy':
    print("All services healthy")
elif health_report['status'] == 'degraded':
    print(f"Some services unhealthy: {health_report['healthy_services']}/{health_report['total_services']}")
else:
    print("System unhealthy")
```

## Best Practices

### 1. Configuration Management

```python
# Use environment-specific configurations
if environment == 'development':
    utility_config = UtilityConfig(
        enable_m1_gpu=False,  # Disable GPU in development
        data_processing_chunk_size=1000
    )
elif environment == 'production':
    utility_config = UtilityConfig(
        enable_m1_gpu=True,
        data_processing_chunk_size=10000
    )
```

### 2. Resource Management

```python
# Always use context managers or cleanup
async with utility_container_context(utility_config) as container:
    # Process data
    result = await process_with_utilities(data)

# Or explicit cleanup
try:
    result = await process_with_utilities(data)
finally:
    await cleanup()
```

### 3. Error Handling

```python
# Always handle missing services gracefully
if service and service.is_available():
    result = service.process(data)
else:
    result = fallback_process(data)
```

### 4. Performance Monitoring

```python
# Track performance metrics
start_time = time.time()
result = await process_with_utilities(data)
processing_time = time.time() - start_time

# Update metrics
self.performance_metrics['processing_time'] += processing_time
self.performance_metrics['utility_operations_count'] += 1
```

### 5. Logging

```python
# Use structured logging
self.logger.info(f"Processing data with utilities: {len(data)} rows")
self.logger.debug(f"Using services: {list(available_services.keys())}")
self.logger.error(f"Utility error: {error}", exc_info=True)
```

## Examples

### Complete Example: Feature Engineering with Utilities

```python
import asyncio
from src.training.steps.step06_utility_container import (
    Step06UtilityContainer, UtilityConfig, get_utility_container, inject_utilities
)
from src.training.steps.step06_enhanced_feature_engineering import EnhancedFeatureEngineering

async def main():
    # Create configurations
    utility_config = UtilityConfig(
        enable_common_operations=True,
        enable_data_processing=True,
        enable_math_validation=True,
        enable_parquet_utils=True,
        enable_serialization=True,
        enable_m1_gpu=True,
        enable_m1_memory=True,
        enable_m1_cpu=True
    )
    
    step06_config = {
        'step06_feature_engineering': {
            'chunk_size': 1000,
            'max_features': 100,
            'polynomial_degree': 2
        }
    }
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    market_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Initialize feature engineering with utilities
    feature_engineering = EnhancedFeatureEngineering(step06_config, utility_config)
    
    try:
        # Initialize utilities
        await feature_engineering.initialize_utilities()
        
        # Create enhanced features
        enhanced_features = await feature_engineering.create_enhanced_features_with_utilities(market_data)
        
        # Get performance metrics
        stats = feature_engineering.get_processing_stats()
        
        print(f"✅ Created {len(enhanced_features.columns)} features")
        print(f"   Utility operations: {stats['utility_operations_count']}")
        print(f"   Processing time: {stats['processing_time']:.2f}s")
        
    finally:
        # Cleanup
        await feature_engineering.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: Comprehensive Implementation

```python
async def run_comprehensive_pipeline():
    """Run comprehensive step06 pipeline with utility integration."""
    
    # Create configurations
    utility_config = UtilityConfig(
        enable_common_operations=True,
        enable_data_processing=True,
        enable_math_validation=True,
        enable_parquet_utils=True,
        enable_serialization=True,
        enable_m1_gpu=True,
        enable_m1_memory=True,
        enable_m1_cpu=True
    )
    
    step06_config = {
        'step06_feature_engineering': {
            'chunk_size': 5000,
            'max_features': 200,
            'polynomial_degree': 2
        }
    }
    
    # Initialize comprehensive implementation
    implementation = Step06ComprehensiveImplementation(step06_config, utility_config)
    
    try:
        # Run pipeline
        results = await implementation.run_comprehensive_pipeline(market_data)
        
        # Check results
        print(f"Pipeline status: {results['pipeline_status']}")
        print(f"Features created: {results['feature_engineering_results']['features_created']}")
        print(f"Labels generated: {results['labeling_results']['labels_generated']}")
        print(f"Utility operations: {results['performance_metrics']['utility_operations_count']}")
        
        # Check utility health
        health_report = results['utility_health_report']
        print(f"Utility health: {health_report['status']}")
        
    finally:
        # Cleanup
        await implementation.cleanup()

if __name__ == "__main__":
    asyncio.run(run_comprehensive_pipeline())
```

## Conclusion

The step06 utility integration provides a robust, scalable, and performant framework for integrating utility services into machine learning pipelines. The dependency injection pattern, comprehensive error handling, and M1-specific optimizations make it suitable for production environments while maintaining flexibility and ease of use.

Key benefits:

- **Modularity**: Services can be enabled/disabled as needed
- **Performance**: M1-specific optimizations for Apple Silicon
- **Reliability**: Comprehensive error handling and graceful degradation
- **Monitoring**: Detailed performance metrics and health reporting
- **Testability**: Comprehensive test suite and validation framework
- **Maintainability**: Clean separation of concerns and dependency injection

This integration framework serves as a foundation for building robust, scalable machine learning systems that can leverage the full power of modern hardware while maintaining code quality and reliability.