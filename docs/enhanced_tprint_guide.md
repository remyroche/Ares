# Enhanced TPrint Utility Guide

## Overview

The Enhanced TPrint Utility is a production-ready, feature-rich logging system that provides timestamped print functionality with advanced features including configuration, thread safety, performance optimization, and integration with existing logging systems.

## Key Features

### 🚀 **Core Features**
- **Configurable timestamp formats** - Simple, detailed, with microseconds (default), or ISO format
- **Color-coded output** - Visual distinction between log levels
- **File logging** - Output to files with single file per run support
- **Performance optimization** - Timestamp caching and lazy evaluation
- **Structured logging** - JSON and custom format support

### 🔧 **Advanced Features**
- **Context managers** - Temporary configuration changes
- **Timer context** - Automatic performance measurement
- **Logging decorators** - Automatic function call logging
- **Batch logging** - Efficient multiple message logging
- **Numba compatibility** - Integration with existing numba timestamps
- **Log level filtering** - Configurable minimum log levels
- **Single file per run** - Unique log files for each application run

## Quick Start

### Basic Usage

```python
from src.utils.tprint import tprint, tprint_info, tprint_error

# Basic logging
tprint("Hello, world!")
tprint_info("Process started")
tprint_error("Something went wrong")
```

### Configuration

```python
from src.utils.tprint import configure_tprint, TPrintConfig, TimestampFormat

# Configure global settings
config = TPrintConfig(
    timestamp_format=TimestampFormat.DETAILED,
    use_colors=True,
    output_to_file=True,
    output_file="app.log"
)
configure_tprint(config)
```

## API Reference

### Core Functions

#### `tprint(*args, **kwargs)`
Basic timestamped print with INFO level.

```python
tprint("User logged in")  # [2025-01-11 06:30:15] INFO: User logged in
tprint("Value:", 42)      # [2025-01-11 06:30:15] INFO: Value: 42
```

#### `tprint_debug(*args, **kwargs)`
Print with DEBUG level.

```python
tprint_debug("Processing data")  # [2025-01-11 06:30:15] DEBUG: Processing data
```

#### `tprint_info(*args, **kwargs)`
Print with INFO level.

```python
tprint_info("Operation completed")  # [2025-01-11 06:30:15] INFO: Operation completed
```

#### `tprint_warning(*args, **kwargs)`
Print with WARNING level.

```python
tprint_warning("Low memory")  # [2025-01-11 06:30:15] WARNING: Low memory
```

#### `tprint_error(*args, **kwargs)`
Print with ERROR level.

```python
tprint_error("Connection failed")  # [2025-01-11 06:30:15] ERROR: Connection failed
```

#### `tprint_success(*args, **kwargs)`
Print with SUCCESS level.

```python
tprint_success("Data saved")  # [2025-01-11 06:30:15] SUCCESS: Data saved
```

### Specialized Functions

#### `tprint_progress(step, total, message="", **kwargs)`
Print progress information.

```python
tprint_progress(3, 10, "Processing data")  # [2025-01-11 06:30:15] PROGRESS: 3/10 (30.0%) Processing data
```

#### `tprint_performance(operation, duration, **kwargs)`
Print performance metrics.

```python
tprint_performance("Data processing", 2.5)  # [2025-01-11 06:30:15] PERFORMANCE: Data processing took 2.5s
```

#### `tprint_structured(data, level=LogLevel.INFO, **kwargs)`
Print structured data (JSON format).

```python
data = {"user_id": 123, "action": "login"}
tprint_structured(data)  # [2025-01-11 06:30:15] INFO: {"user_id": 123, "action": "login"}
```

### Configuration

#### `TPrintConfig`
Configuration class for tprint settings.

```python
config = TPrintConfig(
    # Timestamp configuration
    timestamp_format=TimestampFormat.WITH_MICROSECONDS,  # Default
    timezone=None,  # Use system timezone
    include_microseconds=True,
    
    # Output configuration
    use_colors=True,
    output_file="app.log",
    output_to_console=True,
    output_to_file=False,
    
    # Logging configuration
    min_log_level=LogLevel.DEBUG,
    
    # Performance configuration
    enable_lazy_evaluation=True,
    cache_timestamps=True,
    timestamp_cache_duration=0.001,  # 1ms
    
    # File logging configuration - single file per run
    single_file_per_run=True,
    run_id=None,  # Auto-generated if not provided
    
    # Structured logging
    enable_structured_logging=False,
    structured_format="json",
    
    # Integration
    integrate_with_logging=True,
    log_to_python_logger=False,
)
```

#### `configure_tprint(config)`
Configure global tprint settings.

```python
configure_tprint(config)
```

#### `get_tprint_config()`
Get current configuration.

```python
current_config = get_tprint_config()
```

### Context Managers

#### `tprint_context(config)`
Temporary configuration context.

```python
with tprint_context(TPrintConfig(timestamp_format=TimestampFormat.SIMPLE)):
    tprint("This uses simple timestamp format")
# Back to previous configuration
```

#### `tprint_timer(operation, level=LogLevel.PERFORMANCE)`
Timer context manager.

```python
with tprint_timer("Data processing"):
    # ... do work ...
    pass  # Will automatically log the duration
```

### Decorators

#### `@tprint_logged(level, include_args, include_result)`
Automatic function logging decorator.

```python
@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
def my_function(x, y):
    return x + y
```

### Advanced Functions

#### `tprint_batch(messages, **kwargs)`
Batch logging for performance.

```python
messages = [
    (LogLevel.INFO, "Message 1"),
    (LogLevel.WARNING, "Message 2"),
    (LogLevel.ERROR, "Message 3"),
]
tprint_batch(messages)
```

#### `tprint_numba_compatible(*args, **kwargs)`
Numba-compatible version using existing numba_timestamps.

```python
tprint_numba_compatible("Numba compatible message")
```

## Examples

### Basic Logging

```python
from src.utils.tprint import tprint, tprint_info, tprint_error

# Simple logging
tprint("Application started")
tprint_info("User authentication successful")
tprint_error("Database connection failed")
```

### Progress Tracking

```python
from src.utils.tprint import tprint_progress

# Track progress
for i in range(1, 101):
    tprint_progress(i, 100, f"Processing item {i}")
    # ... do work ...
```

### Performance Monitoring

```python
from src.utils.tprint import tprint_timer, tprint_performance

# Using timer context
with tprint_timer("Data processing"):
    # ... do work ...
    pass

# Manual performance logging
tprint_performance("Model training", 45.5)
```

### Structured Logging

```python
from src.utils.tprint import tprint_structured, LogLevel

# Log structured data
user_data = {
    "user_id": 12345,
    "action": "login",
    "timestamp": "2025-01-11T10:30:00Z"
}
tprint_structured(user_data, LogLevel.INFO)
```

### Configuration Management

```python
from src.utils.tprint import configure_tprint, TPrintConfig, TimestampFormat

# Configure for development
dev_config = TPrintConfig(
    timestamp_format=TimestampFormat.DETAILED,
    use_colors=True,
    min_log_level=LogLevel.DEBUG
)
configure_tprint(dev_config)

# Configure for production
prod_config = TPrintConfig(
    timestamp_format=TimestampFormat.ISO,
    use_colors=False,
    output_to_file=True,
    output_file="production.log",
    min_log_level=LogLevel.INFO
)
configure_tprint(prod_config)
```

### File Logging

```python
from src.utils.tprint import TPrintConfig, tprint_context

# Enable file logging with single file per run
config = TPrintConfig(
    output_to_file=True,
    output_file="app.log",
    output_to_console=True,
    single_file_per_run=True
)

with tprint_context(config):
    tprint("This goes to both console and file")
    tprint_info("File logging enabled")
    # Creates: app_20250111_143052_123.log
```

### Single File Per Run

The single file per run feature ensures each application run gets a unique log file:

```python
# Automatic run ID generation
config = TPrintConfig(
    output_to_file=True,
    output_file="app.log",
    single_file_per_run=True
    # run_id will be auto-generated: 20250111_143052_123
)

# Manual run ID
config = TPrintConfig(
    output_to_file=True,
    output_file="app.log",
    single_file_per_run=True,
    run_id="production_run_001"
    # Creates: app_production_run_001.log
)
```

### Performance Optimization

```python
from src.utils.tprint import TPrintConfig

# Optimize for high-frequency logging
config = TPrintConfig(
    cache_timestamps=True,
    timestamp_cache_duration=0.001,  # 1ms cache
    enable_lazy_evaluation=True
)
configure_tprint(config)

# Use batch logging for multiple messages
from src.utils.tprint import tprint_batch, LogLevel

messages = [
    (LogLevel.INFO, "Message 1"),
    (LogLevel.WARNING, "Message 2"),
    (LogLevel.ERROR, "Message 3"),
]
tprint_batch(messages)
```

### Integration with Existing Code

```python
from src.utils.tprint import tprint_numba_compatible

# Use in numba-compatible contexts
tprint_numba_compatible("Processing in numba context")
```

## Best Practices

### 1. **Configuration Management**
- Use different configurations for development and production
- Enable file logging in production environments
- Use appropriate log levels to control verbosity

### 2. **Performance Considerations**
- Use batch logging for high-frequency messages
- Enable timestamp caching for better performance
- Use appropriate log levels to reduce overhead

### 3. **Thread Safety**
- TPrint is thread-safe by default
- Use context managers for temporary configuration changes
- Clean up resources when done

### 4. **Error Handling**
- Always use appropriate log levels for different types of messages
- Use structured logging for complex data
- Include relevant context in error messages

### 5. **Integration**
- Use decorators for automatic function logging
- Integrate with existing logging frameworks when needed
- Use timer contexts for performance monitoring

## Migration from Basic TPrint

If you're migrating from the basic tprint implementation:

1. **Import changes**: No changes needed for basic functions
2. **Configuration**: Add configuration for advanced features
3. **New features**: Gradually adopt new features as needed
4. **Backward compatibility**: All existing code continues to work

## Troubleshooting

### Common Issues

1. **Colors not showing**: Install colorama or disable colors
2. **File logging not working**: Check file permissions and paths
3. **Performance issues**: Enable timestamp caching and use appropriate log levels
4. **Thread safety issues**: Ensure thread safety is enabled in configuration

### Debug Mode

Enable debug mode for troubleshooting:

```python
config = TPrintConfig(min_log_level=LogLevel.DEBUG)
configure_tprint(config)
```

## Dependencies

- **colorama**: For colored output (optional)
- **numba**: For numba compatibility (optional)
- **Standard library**: datetime, threading, json, pathlib

## License

This utility is part of the project and follows the same license terms.