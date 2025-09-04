# Enhanced Pipeline System Guide

## Overview

The Enhanced Pipeline System provides a comprehensive framework for executing the Ares trading pipeline with robust validation, monitoring, error handling, and data protection. This system ensures that each step of the pipeline is validated, monitored, and protected throughout execution.

## Key Features

### 🔍 **Comprehensive Validation**
- **Data Format Validation**: Ensures data structure and schema compliance
- **Data Quality Validation**: Checks for missing values, outliers, and data integrity
- **Step Dependency Validation**: Verifies prerequisites before step execution
- **Output Validation**: Validates results after each step

### 🛡️ **Data Protection**
- **Automatic Backups**: Creates backups before data modifications
- **Data Integrity Checks**: Validates data integrity with checksums
- **Access Control**: Controls data access with operation restrictions
- **Safe Data Operations**: Context managers for safe data operations

### 📊 **Performance Monitoring**
- **Real-time Metrics**: Tracks execution time, memory usage, CPU usage
- **System Resource Monitoring**: Monitors disk I/O, network I/O, thread count
- **Performance Alerts**: Warns when performance thresholds are exceeded
- **Comprehensive Logging**: Structured logging with context and metadata

### 🔄 **State Management**
- **Checkpoint System**: Creates and validates checkpoints for recovery
- **Pipeline State Tracking**: Tracks pipeline execution state and progress
- **Error Recovery**: Automatic recovery from checkpoints on failure
- **Progress Monitoring**: Real-time progress tracking and reporting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Pipeline System                 │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Integration Layer                                 │
│  ├── EnhancedPipelineIntegration                           │
│  └── run_enhanced_pipeline()                               │
├─────────────────────────────────────────────────────────────┤
│  Validation & Monitoring Layer                              │
│  ├── PipelineValidatorOrchestrator                         │
│  ├── PipelineMonitor                                       │
│  └── PipelineStateManager                                  │
├─────────────────────────────────────────────────────────────┤
│  Data Operations Layer                                      │
│  ├── DataFormatManager                                     │
│  ├── DataAnalysisManager                                   │
│  ├── DataManipulationManager                               │
│  └── DataProtectionManager                                 │
├─────────────────────────────────────────────────────────────┤
│  Decorators & Utilities Layer                               │
│  ├── Pipeline Decorators                                   │
│  ├── Common Operations                                     │
│  └── Error Handling                                        │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
import asyncio
from src.utils.pipeline_integration import run_enhanced_pipeline

# Define pipeline steps
pipeline_steps = [
    {
        "name": "data_collection",
        "function": data_collection_function,
        "dependencies": [],
        "validation_level": "critical",
        "critical": True
    },
    {
        "name": "data_processing",
        "function": data_processing_function,
        "dependencies": ["data_collection"],
        "validation_level": "critical",
        "critical": True
    },
    {
        "name": "model_training",
        "function": model_training_function,
        "dependencies": ["data_processing"],
        "validation_level": "critical",
        "critical": True
    }
]

# Run enhanced pipeline
success = await run_enhanced_pipeline(
    symbol="ETHUSDT",
    exchange="BINANCE",
    pipeline_steps=pipeline_steps,
    timeframe="1m",
    data_dir="data_cache"
)
```

### Using Decorators

```python
from src.utils.pipeline_decorators import (
    pipeline_step,
    data_reader,
    data_writer,
    data_transformer,
    data_analyzer
)

@pipeline_step("my_step", ValidationLevel.CRITICAL)
@data_reader(validate_schema=True)
@data_transformer(validate_schema=True)
@data_writer(validate_schema=True)
def my_pipeline_step(symbol: str, exchange: str, **kwargs) -> bool:
    """My pipeline step with comprehensive validation."""
    # Your step logic here
    return True
```

### Manual Component Usage

```python
from src.utils.pipeline_utilities import pipeline_utilities
from src.utils.pipeline_validator_framework import validator_orchestrator

# Data operations with protection
with pipeline_utilities.safe_data_operation("my_operation", "data_file.parquet"):
    data = pipeline_utilities.format_manager.read_data("input.parquet")
    processed_data = pipeline_utilities.manipulation_manager.clean_data(data)
    pipeline_utilities.format_manager.write_data(processed_data, "output.parquet")

# Validation
validation_results = await validator_orchestrator.validate_pipeline_step(
    step_name="my_step",
    data=processed_data,
    context={"validation_level": "critical"},
    validators_to_run=["data_format", "data_quality"]
)
```

## Configuration

### Pipeline Step Configuration

```python
{
    "name": "step_name",                    # Unique step identifier
    "function": step_function,              # Function to execute
    "dependencies": ["step1", "step2"],     # Required previous steps
    "validation_level": "critical",         # critical, warning, info, debug
    "critical": True,                       # Whether step failure stops pipeline
    "custom_config": {...}                  # Step-specific configuration
}
```

### Validation Levels

- **CRITICAL**: Must pass for pipeline to continue
- **WARNING**: Can continue with warnings
- **INFO**: Informational only
- **DEBUG**: Debug information

### Data Format Support

- **Parquet**: Recommended for large datasets
- **CSV**: Human-readable format
- **JSON**: Structured data format
- **HDF5**: High-performance format
- **Feather**: Fast columnar format

## Monitoring and Logging

### Real-time Monitoring

```python
from src.utils.pipeline_monitoring import pipeline_monitor

# Get monitoring summary
summary = pipeline_monitor.get_monitoring_summary()
print(f"Total log entries: {summary['total_log_entries']}")
print(f"Total metrics: {summary['total_metrics']}")

# Record custom events
pipeline_monitor.record_pipeline_event(
    event_type="custom_event",
    pipeline_id="my_pipeline",
    message="Custom event occurred",
    context={"custom_data": "value"}
)
```

### Performance Metrics

```python
from src.utils.pipeline_monitoring import MetricType

# Record custom metrics
pipeline_monitor.metrics_collector.record_metric(
    "custom_metric",
    MetricType.GAUGE,
    42.0,
    tags={"component": "my_component"},
    unit="units"
)
```

## Error Handling and Recovery

### Automatic Recovery

The system automatically:
- Creates checkpoints after each successful step
- Validates checkpoint integrity
- Recovers from the last valid checkpoint on failure
- Provides detailed error reporting

### Manual Recovery

```python
from src.utils.pipeline_state_manager import pipeline_state_manager

# Resume from checkpoint
success = pipeline_state_manager.resume_pipeline_from_checkpoint(
    pipeline_id="my_pipeline",
    step_name="failed_step"
)
```

## Best Practices

### 1. **Step Design**
- Keep steps focused and atomic
- Use appropriate validation levels
- Define clear dependencies
- Handle errors gracefully

### 2. **Data Management**
- Use appropriate data formats
- Validate data at each step
- Create backups before modifications
- Monitor data quality

### 3. **Performance**
- Monitor resource usage
- Use efficient data formats
- Implement proper error handling
- Optimize for your use case

### 4. **Monitoring**
- Record important events
- Monitor performance metrics
- Set up alerts for critical issues
- Review logs regularly

## Testing

### Running Tests

```bash
# Run comprehensive test suite
python test_enhanced_pipeline.py

# Test individual components
python -c "
import asyncio
from test_enhanced_pipeline import test_individual_components
asyncio.run(test_individual_components())
"
```

### Test Coverage

The test suite covers:
- ✅ Data format validation
- ✅ Data quality analysis
- ✅ Data manipulation operations
- ✅ Pipeline step execution
- ✅ Checkpoint creation and recovery
- ✅ Monitoring and logging
- ✅ Error handling and recovery

## Troubleshooting

### Common Issues

1. **Validation Failures**
   - Check data format and schema
   - Verify required columns exist
   - Review data quality metrics

2. **Performance Issues**
   - Monitor resource usage
   - Check for memory leaks
   - Optimize data operations

3. **Checkpoint Issues**
   - Verify checkpoint integrity
   - Check file permissions
   - Review dependency validation

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("pipeline").setLevel(logging.DEBUG)

# Get detailed validation results
validation_results = await validator_orchestrator.validate_pipeline_step(
    step_name="debug_step",
    data=your_data,
    context={"debug": True},
    validators_to_run=["data_format", "data_quality"]
)
```

## Integration with Existing Pipeline

### Updating Existing Steps

```python
# Before: Basic step
def my_step(symbol, exchange, **kwargs):
    # Step logic
    return True

# After: Enhanced step
@pipeline_step("my_step", ValidationLevel.CRITICAL)
@data_reader(validate_schema=True)
@data_writer(validate_schema=True)
def my_enhanced_step(symbol, exchange, **kwargs):
    # Same step logic with added validation
    return True
```

### Gradual Migration

1. Start with critical steps
2. Add validation gradually
3. Monitor performance impact
4. Migrate all steps over time

## Performance Considerations

### Memory Usage
- Use efficient data formats (Parquet recommended)
- Process data in chunks for large datasets
- Monitor memory usage with built-in metrics

### Execution Time
- Use appropriate validation levels
- Cache validation results when possible
- Optimize data operations

### Storage
- Clean up old checkpoints regularly
- Use compression for log files
- Monitor disk usage

## Security

### Data Protection
- Automatic backups before modifications
- Checksum validation for data integrity
- Access control for data operations
- Audit logging for all operations

### Error Handling
- No sensitive data in error messages
- Secure error logging
- Proper exception handling

## Conclusion

The Enhanced Pipeline System provides a robust, scalable, and maintainable framework for executing complex data pipelines. With comprehensive validation, monitoring, and protection mechanisms, it ensures reliable execution while providing detailed insights into pipeline performance and data quality.

For more information, refer to the individual component documentation and the test suite for usage examples.