# ML Optimization Utilities Migration Guide

## Overview

The ML optimization utilities have been refactored to improve maintainability, performance, and reliability while maintaining full backward compatibility. This guide explains the changes and how to migrate if needed.

## What Changed

### 1. Architecture Improvements

**Before**: Monolithic `ConsolidatedHPO` class with 1,827 lines
**After**: Focused components with single responsibilities

- `HPOEngine`: Core optimization orchestration
- `OptimizationStrategy`: Strategy pattern for different optimization approaches
- `OptimizationMonitor`: Monitoring and diagnostics
- `OptimizationCache`: Intelligent caching system
- `PrunerFactory`: Factory for creating pruners

### 2. Error Handling

**Before**: Generic exception handling with inconsistent patterns
**After**: Comprehensive exception hierarchy with specific error types

```python
# New exception types
from src.utils.ml_common.optimization import (
    OptimizationError, ConfigurationError, ModelEvaluationError,
    HardwareOptimizationError, PruningError, SearchSpaceError,
    ConvergenceError, TimeoutError, ValidationError, CacheError,
    MonitoringError, VectorBTError, AresModeError
)
```

### 3. Configuration Validation

**Before**: No validation of configuration parameters
**After**: Pydantic-based validation with type safety

```python
# New validated configuration
from src.utils.ml_common.optimization import HPOConfig, validate_hpo_config

# Automatic validation
config = HPOConfig(n_trials=100, strategy='bayesian')  # Validated automatically

# Or validate existing dict
config = validate_hpo_config({'n_trials': 100, 'strategy': 'bayesian'})
```

## Migration Guide

### For Existing Code (No Changes Required)

**Good news**: All existing code will continue to work without any changes!

```python
# This still works exactly the same
from src.utils.ml_common.optimization import ConsolidatedHPO, HPOConfig

config = HPOConfig(n_trials=100, strategy='bayesian')
hpo = ConsolidatedHPO(config)

result = hpo.optimize(model_factory, X, y, search_space)
```

### For New Code (Recommended)

Use the new, more focused components for better performance and maintainability:

```python
# New approach - more focused and efficient
from src.utils.ml_common.optimization import HPOEngine, HPOConfig

config = HPOConfig(n_trials=100, strategy='bayesian')
engine = HPOEngine(config)

result = engine.optimize(model_factory, X, y, search_space)
```

### Configuration Improvements

**Old way**:
```python
config = {
    'n_trials': 100,
    'strategy': 'bayesian',
    'timeout': 3600
}
hpo = ConsolidatedHPO(config)
```

**New way** (recommended):
```python
from src.utils.ml_common.optimization import HPOConfig

config = HPOConfig(
    n_trials=100,
    strategy='bayesian',
    timeout=3600
)
hpo = ConsolidatedHPO(config)
```

### Error Handling Improvements

**Old way**:
```python
try:
    result = hpo.optimize(model_factory, X, y, search_space)
except Exception as e:
    print(f"Optimization failed: {e}")
```

**New way** (recommended):
```python
from src.utils.ml_common.optimization import (
    OptimizationError, ModelEvaluationError, ConfigurationError
)

try:
    result = hpo.optimize(model_factory, X, y, search_space)
except ModelEvaluationError as e:
    print(f"Model evaluation failed: {e}")
    print(f"Context: {e.context}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except OptimizationError as e:
    print(f"General optimization error: {e}")
```

### Using New Components Directly

For advanced use cases, you can use the new components directly:

```python
from src.utils.ml_common.optimization import (
    HPOEngine, BayesianStrategy, OptimizationMonitor, OptimizationCache
)

# Create components
config = HPOConfig(n_trials=100, strategy='bayesian')
monitor = OptimizationMonitor(enable_detailed_logging=True)
cache = OptimizationCache(max_size=1000, ttl_seconds=3600)

# Create engine with custom components
engine = HPOEngine(
    config=config,
    monitor=monitor,
    cache=cache
)

# Use strategy directly
strategy = BayesianStrategy(config)
# ... use strategy
```

## Performance Improvements

### 1. Better Caching
- Intelligent TTL-based caching
- Memory-aware eviction policies
- Hash-based keys for complex objects

### 2. Improved Error Handling
- Faster error detection and reporting
- Better debugging information
- Reduced silent failures

### 3. Memory Optimization
- Lazy evaluation where possible
- Better memory management
- Reduced memory footprint

## Testing

The refactored system includes comprehensive test coverage:

```python
# Example test
def test_bayesian_optimization():
    config = HPOConfig(strategy="bayesian", n_trials=10)
    hpo = ConsolidatedHPO(config)
    
    def objective(params):
        return -(params['x'] - 2) ** 2
    
    search_space = {'x': {'type': 'float', 'low': 0, 'high': 5}}
    result = hpo.optimize(objective, search_space)
    
    assert result.best_score > -1.0
    assert 1.5 <= result.best_params['x'] <= 2.5
```

## Backward Compatibility

### What's Preserved
- All existing class names and methods
- All existing function signatures
- All existing configuration options
- All existing return types

### What's Improved
- Better error messages
- More robust validation
- Improved performance
- Better monitoring and diagnostics

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're importing from the correct module
   ```python
   # Correct
   from src.utils.ml_common.optimization import ConsolidatedHPO
   
   # Incorrect (old path)
   from src.utils.ml_common.optimization.consolidated_hpo import ConsolidatedHPO
   ```

2. **Configuration Errors**: Use the new validation
   ```python
   # This will give you better error messages
   from src.utils.ml_common.optimization import validate_hpo_config
   
   try:
       config = validate_hpo_config(your_config_dict)
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   ```

3. **Performance Issues**: Use the new engine directly
   ```python
   # For better performance
   from src.utils.ml_common.optimization import HPOEngine
   
   engine = HPOEngine(config)
   result = engine.optimize(model_factory, X, y, search_space)
   ```

## Support

If you encounter any issues during migration:

1. Check this migration guide
2. Review the error messages (they're much more helpful now!)
3. Use the new validation functions to debug configuration issues
4. Consider using the new components directly for better control

## Future Plans

- Async optimization support
- Advanced performance optimizations
- Enhanced monitoring and observability
- Integration with more hardware optimization systems