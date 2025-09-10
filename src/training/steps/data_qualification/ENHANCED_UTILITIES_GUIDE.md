# Enhanced Data Qualification Utilities Guide

## Overview

This guide demonstrates how to use the new enhanced utilities for data qualification steps, providing a modern, robust, and maintainable approach to building data qualification pipelines.

## Key Features

### 1. **Unified Import Management** 🔧
- Centralized import management with automatic fallback handling
- ML Commons integration with graceful degradation
- M1 optimization utilities with CPU fallback
- Comprehensive error handling and logging

### 2. **Standardized Configuration System** ⚙️
- Type-safe configuration with validation
- Environment-specific configuration support
- Configuration inheritance and composition
- JSON/YAML configuration file support

### 3. **Comprehensive Error Handling** 🛡️
- Automatic error classification and recovery
- Retry strategies with exponential backoff
- Circuit breaker pattern for failing services
- Detailed error analytics and reporting

### 4. **Enhanced Documentation & Type Safety** 📚
- Comprehensive type hints throughout
- Detailed docstrings with examples
- Protocol definitions for interfaces
- Performance monitoring and metrics

## Quick Start

### Basic Usage

```python
from src.training.steps.data_qualification import (
    DataQualificationConfig,
    DataQualificationStep,
    DataQualificationResult,
    EnhancedSROptimizationStep
)

# Create configuration
config = DataQualificationConfig(
    symbol="AAPL",
    exchange="NASDAQ",
    timeframe="1m",
    data_dir="./data"
)

# Create and execute step
step = EnhancedSROptimizationStep(config)
result = await step.execute({"data": your_dataframe})

print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.2f}s")
```

### Advanced Configuration

```python
from src.training.steps.data_qualification import (
    DataQualificationConfig,
    PerformanceConfig,
    SROptimizationConfig,
    MLCommonsConfig
)

# Create advanced configuration
config = DataQualificationConfig(
    symbol="AAPL",
    exchange="NASDAQ",
    timeframe="1m",
    data_dir="./data",
    
    # Performance settings
    performance=PerformanceConfig(
        enable_m1_optimization=True,
        enable_gpu_acceleration=True,
        max_workers=4,
        memory_limit_gb=8.0
    ),
    
    # Step-specific settings
    sr_optimization=SROptimizationConfig(
        min_touch_count=3,
        max_touch_count=10,
        strength_threshold=0.5,
        enable_ml_commons=True
    ),
    
    # ML Commons integration
    ml_commons=MLCommonsConfig(
        enable_ml_commons=True,
        enable_fallback=True
    )
)
```

## Creating Custom Steps

### 1. Inherit from Base Class

```python
from src.training.steps.data_qualification import (
    DataQualificationStep,
    DataQualificationResult,
    ValidationResult
)

class MyCustomStep(DataQualificationStep):
    """Custom data qualification step."""
    
    async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        """Execute the custom step."""
        # Your implementation here
        return DataQualificationResult(
            success=True,
            data=processed_data,
            step_name=self.__class__.__name__
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """Validate input data."""
        # Your validation logic here
        return ValidationResult(is_valid=True)
```

### 2. Use Utilities with Error Handling

```python
class MyStep(DataQualificationStep):
    async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        try:
            # Get ML Commons utilities
            ml_commons = self.get_utility('ml_common')
            if ml_commons:
                data_quality = ml_commons.get('data_quality')
                # Use ML Commons utilities
            else:
                # Use fallback utilities
                pass
            
            # Get M1 optimization utilities
            m1_optimizers = self.get_utility('m1_optimizers')
            if m1_optimizers:
                memory_optimizer = m1_optimizers.get('memory_optimizer')
                # Use memory optimization
                
        except Exception as e:
            # Automatic error handling
            return self.handle_error(e, "my_operation")
```

## Error Handling Patterns

### 1. Automatic Error Handling

```python
from src.training.steps.data_qualification import handle_utility_failure

try:
    result = some_utility_function()
except Exception as e:
    result = handle_utility_failure(
        step_name="my_step",
        utility_name="some_utility",
        error=e,
        fallback_func=my_fallback_function
    )
```

### 2. Error Context Manager

```python
from src.training.steps.data_qualification import error_context

with error_context("my_step", "my_operation", user_id="123"):
    # Your code here
    risky_operation()
```

### 3. Decorator-based Error Handling

```python
from src.training.steps.data_qualification import with_error_recovery

@with_error_recovery(
    step_name="my_step",
    utility_name="my_utility",
    fallback_func=my_fallback
)
def my_function():
    # Your implementation
    pass
```

## Configuration Management

### 1. Load from File

```python
from src.training.steps.data_qualification import DataQualificationConfig

# Load from JSON
config = DataQualificationConfig.from_json_file("config.json")

# Load from YAML
config = DataQualificationConfig.from_yaml_file("config.yaml")
```

### 2. Environment Variables

```python
# Set environment variables: DQ_SYMBOL=AAPL, DQ_EXCHANGE=NASDAQ, etc.
config = DataQualificationConfig.from_environment()
```

### 3. Configuration Manager

```python
from src.training.steps.data_qualification import get_config_manager

config_manager = get_config_manager()
config = config_manager.load_config("my_config", environment="production")
```

## Pipeline Orchestration

### 1. Sequential Execution

```python
from src.training.steps.data_qualification import DataQualificationPipeline

pipeline = DataQualificationPipeline(config)
pipeline.add_step(EnhancedSROptimizationStep(config))
pipeline.add_step(HMMRegimeDiscoveryStep(config))

result = await pipeline.execute(input_data, mode="sequential")
```

### 2. Parallel Execution

```python
result = await pipeline.execute(input_data, mode="parallel")
```

## Performance Monitoring

### 1. Step Metrics

```python
# Metrics are automatically collected
result = await step.execute(input_data)
metrics = result.metrics

print(f"Execution time: {metrics.execution_time:.2f}s")
print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
print(f"Success rate: {metrics.success_rate:.2f}")
```

### 2. Pipeline Statistics

```python
stats = pipeline.get_pipeline_statistics()
print(f"Total execution time: {stats['total_execution_time']:.2f}s")
print(f"Average success rate: {stats['average_success_rate']:.2f}")
```

## Best Practices

### 1. **Always Use Type Hints**
```python
async def my_function(self, data: pd.DataFrame) -> DataQualificationResult:
    """Function with proper type hints."""
    pass
```

### 2. **Validate Input Data**
```python
def validate_input(self, input_data: Dict[str, Any]) -> ValidationResult:
    """Always validate input data."""
    # Check required fields
    # Validate data types
    # Check data quality
    return ValidationResult(is_valid=True)
```

### 3. **Handle Errors Gracefully**
```python
try:
    result = risky_operation()
except Exception as e:
    return self.handle_error(e, "operation_name", fallback_func)
```

### 4. **Use Configuration System**
```python
# Don't hardcode values
config = self.get_step_config()
threshold = config.get('threshold', 0.5)
```

### 5. **Monitor Performance**
```python
# Metrics are collected automatically
# Use them for optimization
if result.metrics.execution_time > 60:
    self.logger.warning("Step execution time exceeded 60s")
```

## Migration Guide

### From Legacy Steps

1. **Update Imports**
```python
# Old
from src.training.steps.data_qualification import SROptimizationStep

# New
from src.training.steps.data_qualification import (
    DataQualificationStep,
    DataQualificationConfig,
    EnhancedSROptimizationStep
)
```

2. **Update Configuration**
```python
# Old
config = {"symbol": "AAPL", "exchange": "NASDAQ"}

# New
config = DataQualificationConfig(
    symbol="AAPL",
    exchange="NASDAQ",
    timeframe="1m",
    data_dir="./data"
)
```

3. **Update Step Implementation**
```python
# Old
class MyStep:
    def __init__(self, config):
        self.config = config
    
    def execute(self, data):
        # Implementation
        pass

# New
class MyStep(DataQualificationStep):
    async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        # Implementation with proper error handling
        return DataQualificationResult(success=True, data=result)
    
    def validate_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        # Validation logic
        return ValidationResult(is_valid=True)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Check if ML Commons utilities are available
   - Verify fallback mechanisms are working
   - Check import paths

2. **Configuration Errors**
   - Validate configuration with `config.validate()`
   - Check required fields
   - Verify data types

3. **Performance Issues**
   - Monitor metrics collection
   - Check memory usage
   - Optimize chunk sizes

4. **Error Handling Issues**
   - Check error classification
   - Verify fallback functions
   - Review error logs

### Debug Mode

```python
config = DataQualificationConfig(
    symbol="AAPL",
    exchange="NASDAQ",
    timeframe="1m",
    data_dir="./data",
    enable_debugging=True,
    debug_output_dir="./debug"
)
```

## Examples

See `step_example_enhanced.py` for a complete example of how to implement an enhanced data qualification step using all the new utilities.

## Support

For issues or questions:
1. Check the error logs
2. Review the configuration validation
3. Test with fallback utilities
4. Check the example implementations