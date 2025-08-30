# Enhanced Decorator System

## Overview

The Enhanced Decorator System is a comprehensive improvement to the existing decorator infrastructure that provides:

- **Better Performance**: Intelligent caching and optimized validation
- **Enhanced Error Handling**: Automatic recovery and graceful degradation
- **Unified Configuration**: Centralized control over all decorator behavior
- **Improved Type Safety**: Protocol-based validation with better type hints
- **Backwards Compatibility**: Existing code continues to work unchanged
- **Registry Management**: Centralized decorator discovery and versioning

## Architecture

```
src/utils/
├── decorator_config.py          # Global configuration management
├── decorator_registry.py        # Central decorator registry
├── enhanced_decorators.py       # New enhanced decorators
├── decorator_compatibility.py   # Backwards compatibility layer
├── centralized_decorators_v2.py # Updated centralized interface
└── decorators.py               # Original decorators (unchanged)
```

## Key Components

### 1. Configuration System (`decorator_config.py`)

Centralized configuration for all decorators with environment-specific settings.

```python
from src.utils.decorator_config import global_config, ValidationMode, PerformanceMode

# Configure global behavior
global_config.validation_mode = ValidationMode.STRICT
global_config.enable_data_quality_checks = True
global_config.cache_enabled = True
global_config.max_retries = 5
```

**Available Configuration Options:**
- `validation_mode`: STRICT, WARNING, PERMISSIVE
- `performance_mode`: DISABLED, BASIC, DETAILED, PROFILING
- `enable_data_quality_checks`: Enable/disable data quality validation
- `enable_performance_monitoring`: Enable/disable performance tracking
- `enable_error_recovery`: Enable/disable automatic error recovery
- `cache_enabled`: Enable/disable caching
- `cache_size`: Maximum cache entries
- `cache_ttl`: Cache time-to-live in seconds
- `max_retries`: Maximum retry attempts for error recovery
- `backoff_factor`: Exponential backoff multiplier

### 2. Decorator Registry (`decorator_registry.py`)

Central registry for all decorators with metadata, versioning, and discovery.

```python
from src.utils.decorator_registry import decorator_registry

# List all available decorators
decorators = decorator_registry.list_decorators()

# Search decorators by tags or description
validation_decorators = decorator_registry.search("validation")

# Get usage statistics
usage_stats = decorator_registry.get_usage_stats()

# Export configuration
config = decorator_registry.export_config()
```

**Registry Features:**
- Automatic decorator registration with metadata
- Version tracking and history
- Tag-based categorization
- Usage statistics
- Deprecation management
- Alias support

### 3. Enhanced Decorators (`enhanced_decorators.py`)

New decorators with improved functionality and performance.

#### Smart Error Recovery

```python
from src.utils.enhanced_decorators import smart_error_recovery

@smart_error_recovery(max_retries=3, fallback_strategy="graceful_degradation")
def function_with_errors(x):
    if x < 0:
        raise ValueError("Negative number not allowed")
    return x * 2

# Automatically retries and applies fallback strategies
result = function_with_errors(-3)  # Returns None with graceful degradation
```

**Features:**
- Automatic retry with exponential backoff
- Configurable fallback strategies
- Exception type filtering
- Async/sync support

#### Cached Validation

```python
from src.utils.enhanced_decorators import cached_validation

@cached_validation(cache_size=128, ttl_seconds=3600)
def expensive_validation_function(data):
    # Expensive validation logic
    time.sleep(1)
    return validation_result

# First call: executes validation
result1 = expensive_validation_function(data)

# Second call: uses cached result
result2 = expensive_validation_function(data)  # Much faster!
```

**Features:**
- Intelligent cache key generation
- Configurable cache size and TTL
- Memory-efficient LRU eviction
- Async/sync support

#### Enhanced Validation

```python
from src.utils.enhanced_decorators import enhanced_validation, ValidatableData

class DataValidator(ValidatableData):
    def validate(self) -> bool:
        # Custom validation logic
        return True
    
    def get_validation_errors(self) -> List[str]:
        # Return validation errors
        return []

@enhanced_validation(validator=DataValidator(), auto_fix=True)
def process_data(data):
    return data * 2
```

**Features:**
- Protocol-based validation interface
- Automatic error fixing
- Pre/post validation hooks
- Configurable strictness levels

#### Performance Monitoring v2

```python
from src.utils.enhanced_decorators import performance_monitor_v2

@performance_monitor_v2(
    level="detailed",
    track_memory=True,
    track_cpu=True,
    track_io=True
)
def resource_intensive_function():
    # Function implementation
    pass
```

**Features:**
- Multiple monitoring levels
- Memory and CPU tracking
- I/O operation monitoring
- Structured metric logging

### 4. Backwards Compatibility (`decorator_compatibility.py`)

Ensures existing code continues to work while providing access to new features.

```python
from src.utils.decorator_compatibility import (
    # Legacy names (with deprecation warnings)
    validate_call,      # Maps to validate_call_or_runtime_types
    check_input,        # Maps to pa_check_input
    check_output,       # Maps to pa_check_output
    
    # Enhanced decorator aliases
    smart_recovery,     # Alias for smart_error_recovery
    cached,             # Alias for cached_validation
    validation,         # Alias for enhanced_validation
    performance         # Alias for performance_monitor_v2
)

# Configuration helpers
from src.utils.decorator_compatibility import (
    get_decorator_config,
    set_decorator_config,
    list_available_decorators,
    get_decorator_usage_stats,
    search_decorators
)
```

### 5. Centralized Decorators v2 (`centralized_decorators_v2.py`)

Updated centralized interface with enhanced functionality.

```python
from src.utils.centralized_decorators_v2 import (
    validate_data_quality_v2,
    quality_gate_v2,
    step_specific_ml_validation_v2,
    auto_fix_data_quality_issues_v2,
    monitor_feature_engineering_v2,
    monitor_data_collection_v2
)

@validate_data_quality_v2(
    validation_level="WARNING",
    auto_fix=True,
    context="data processing"
)
def process_data(df):
    return df * 2

@quality_gate_v2(
    min_quality_score=0.8,
    required_grade="B",
    action_on_failure="degrade"
)
def quality_assessment(data):
    return calculate_quality_score(data)

@step_specific_ml_validation_v2(
    step_name="feature_engineering",
    adaptive_thresholds=True
)
def feature_engineering_step(data):
    return engineer_features(data)
```

## Migration Guide

### Phase 1: Immediate Benefits (No Code Changes)

Existing code automatically benefits from:
- Improved error handling
- Better performance monitoring
- Enhanced logging and tracing
- Centralized configuration

### Phase 2: Gradual Enhancement

Replace legacy decorators with enhanced versions:

```python
# Before (still works with deprecation warnings)
@validate_call()
@check_input(schema)
@check_output(schema)

# After (recommended)
@validate_call_or_runtime_types()
@pa_check_input(schema)
@pa_check_output(schema)
```

### Phase 3: Advanced Features

Adopt new enhanced decorators:

```python
# Add smart error recovery
@smart_error_recovery(max_retries=3)
@validate_call_or_runtime_types()
def function(x):
    return x * 2

# Add caching for expensive operations
@cached_validation(cache_size=100)
def expensive_validation(data):
    # Expensive validation logic
    pass

# Add performance monitoring
@performance_monitor_v2(level="detailed")
def resource_intensive_function():
    # Function implementation
    pass
```

### Phase 4: Configuration-Driven Behavior

Use centralized configuration:

```python
from src.utils.decorator_config import global_config

# Configure behavior globally
global_config.validation_mode = ValidationMode.STRICT
global_config.enable_performance_monitoring = True
global_config.cache_enabled = True

# All decorators automatically use these settings
```

## Best Practices

### 1. Decorator Composition

```python
# Good: Logical order of concerns
@smart_error_recovery(max_retries=3)
@cached_validation(cache_size=100)
@performance_monitor_v2(level="basic")
@validate_call_or_runtime_types()
def process_data(data):
    return data * 2

# Avoid: Performance monitoring before caching
@performance_monitor_v2(level="basic")
@cached_validation(cache_size=100)  # Wrong order
@smart_error_recovery(max_retries=3)
def process_data(data):
    return data * 2
```

### 2. Configuration Management

```python
# Good: Environment-specific configuration
if os.getenv("ENVIRONMENT") == "production":
    global_config.validation_mode = ValidationMode.STRICT
    global_config.enable_performance_monitoring = True
else:
    global_config.validation_mode = ValidationMode.WARNING
    global_config.enable_performance_monitoring = False

# Good: Per-function overrides
@smart_error_recovery(max_retries=5)  # Override global setting
def critical_function():
    pass
```

### 3. Error Handling Strategy

```python
# Good: Appropriate fallback strategies
@smart_error_recovery(
    max_retries=3,
    fallback_strategy="graceful_degradation"
)
def user_facing_function():
    pass

@smart_error_recovery(
    max_retries=1,
    fallback_strategy="default_return"
)
def internal_function():
    pass
```

### 4. Caching Strategy

```python
# Good: Appropriate cache settings
@cached_validation(
    cache_size=1000,      # Large cache for expensive operations
    ttl_seconds=3600      # 1 hour TTL for stable data
)
def expensive_stable_validation(data):
    pass

@cached_validation(
    cache_size=100,       # Smaller cache for frequent operations
    ttl_seconds=300       # 5 minutes TTL for dynamic data
)
def frequent_validation(data):
    pass
```

## Performance Considerations

### 1. Caching Impact

- **Memory Usage**: Each cached decorator maintains its own cache
- **Cache Size**: Balance between memory usage and performance
- **TTL**: Set appropriate expiration for your use case

### 2. Error Recovery Overhead

- **Retry Logic**: Adds latency for failed operations
- **Fallback Strategies**: May have performance implications
- **Logging**: Increased logging volume in error scenarios

### 3. Validation Overhead

- **Pre/Post Validation**: Adds execution time
- **Auto-fixing**: May be expensive for large datasets
- **Strict Mode**: Fails fast but may be slower for complex validation

## Monitoring and Debugging

### 1. Usage Statistics

```python
from src.utils.decorator_compatibility import get_decorator_usage_stats

stats = get_decorator_usage_stats()
for decorator, count in stats.items():
    print(f"{decorator}: {count} uses")
```

### 2. Registry Discovery

```python
from src.utils.decorator_compatibility import search_decorators

# Find decorators by functionality
performance_decorators = search_decorators("performance")
validation_decorators = search_decorators("validation")
```

### 3. Configuration Inspection

```python
from src.utils.decorator_compatibility import get_decorator_config

config = get_decorator_config()
print(f"Validation mode: {config.validation_mode}")
print(f"Cache enabled: {config.cache_enabled}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Circular Imports**: Use the compatibility layer for imports
3. **Performance Degradation**: Check cache settings and validation levels
4. **Memory Issues**: Monitor cache sizes and TTL settings

### Debug Mode

Enable debug logging for detailed decorator behavior:

```python
import logging
logging.getLogger("src.utils.enhanced_decorators").setLevel(logging.DEBUG)
logging.getLogger("src.utils.decorator_registry").setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Distributed Caching**: Redis/Memcached integration
2. **Metrics Export**: Prometheus/InfluxDB integration
3. **Dynamic Configuration**: Runtime configuration updates
4. **Machine Learning**: Adaptive threshold adjustment
5. **Plugin System**: Custom decorator registration

### Extension Points

The system is designed for easy extension:

```python
from src.utils.decorator_registry import register_decorator

@register_decorator(
    name="custom_decorator",
    version="1.0",
    description="Custom functionality",
    tags=["custom", "extension"]
)
def custom_decorator(func):
    # Custom decorator implementation
    pass
```

## Conclusion

The Enhanced Decorator System provides a robust, performant, and maintainable foundation for your codebase while ensuring complete backwards compatibility. By following the migration guide and best practices, you can gradually adopt new features and improve your application's reliability and performance.

For questions or issues, refer to the test files and examples provided in the codebase.