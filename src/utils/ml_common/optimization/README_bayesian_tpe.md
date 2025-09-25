# Bayesian TPE Optimizer with Automatic Grid Search Integration

A comprehensive Bayesian TPE optimization system that automatically calls your dedicated grid utils (coarse then fine grid) as a first step, followed by Bayesian TPE optimization for final refinement.

## 🚀 Key Features

- **Automatic Grid Search Integration**: Automatically performs coarse → fine → TPE optimization
- **Comprehensive Logging**: Detailed logging with configurable levels and file output
- **Error Handling**: Robust error handling with graceful fallbacks
- **Parallel Processing**: Support for parallel evaluation of grid points
- **Multiple Backends**: Support for both Optuna and scikit-optimize
- **Memory Management**: Efficient memory usage with configurable history limits
- **Performance Monitoring**: Built-in performance and convergence monitoring
- **Early Stopping**: Configurable early stopping and convergence detection

## 📦 Installation

The module requires the following dependencies:

```bash
# Required
numpy
pandas

# Optional (for TPE optimization)
optuna>=3.0.0
scikit-optimize>=0.9.0
```

## 🎯 Quick Start

### Basic Usage

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, BayesianTPEConfig

# Define your objective function
def objective_function(params, **kwargs):
    x, y = params['x'], params['y']
    return -(x - 1)**2 - (y - 2)**2  # Maximize this

# Define search space
search_space = {
    'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
    'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
}

# Configure optimizer
config = BayesianTPEConfig(
    n_trials=50,
    coarse_grid_points=5,
    fine_grid_points=8,
    enable_grid_search=True
)

# Optimize
optimizer = BayesianTPEOptimizer(config)
result = optimizer.optimize(objective_function, search_space)

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
```

### Convenience Function

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import optimize_with_bayesian_tpe

result = optimize_with_bayesian_tpe(
    objective_function=objective_function,
    search_space=search_space,
    config=BayesianTPEConfig(n_trials=30)
)
```

## 🔧 Configuration Options

### BayesianTPEConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_grid_search` | bool | True | Enable automatic grid search |
| `coarse_grid_points` | int | 5 | Points per parameter for coarse grid |
| `fine_grid_points` | int | 8 | Points per parameter for fine grid |
| `n_trials` | int | 50 | Number of TPE trials |
| `timeout_seconds` | Optional[int] | None | Maximum optimization time |
| `random_state` | int | 42 | Random seed for reproducibility |
| `backend` | str | 'optuna' | Optimization backend ('optuna' or 'skopt') |
| `enable_parallel` | bool | True | Enable parallel processing |
| `max_workers` | int | 4 | Maximum parallel workers |
| `enable_early_stopping` | bool | True | Enable early stopping |
| `early_stopping_patience` | int | 10 | Early stopping patience |
| `enable_convergence_detection` | bool | True | Enable convergence detection |
| `convergence_threshold` | float | 0.01 | Convergence threshold |
| `log_level` | str | 'INFO' | Logging level |
| `log_file` | Optional[str] | None | Log file path |

## 📊 Search Space Definition

### Parameter Types

#### Float Parameters
```python
'param_name': {
    'type': 'float',
    'low': 0.0,
    'high': 10.0,
    'log': False  # Optional: use log scale
}
```

#### Integer Parameters
```python
'param_name': {
    'type': 'int',
    'low': 1,
    'high': 100
}
```

#### Categorical Parameters
```python
'param_name': {
    'type': 'categorical',
    'choices': ['option1', 'option2', 'option3']
}
```

### Creating Search Space from Bounds

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import create_search_space_from_bounds

bounds = {
    'x': (-5.0, 5.0),
    'y': (1, 10),
    'method': (0, 2)
}

param_types = {
    'x': 'float',
    'y': 'int', 
    'method': 'categorical'
}

search_space = create_search_space_from_bounds(bounds, param_types)
```

## 🎯 Advanced Usage

### Machine Learning Hyperparameter Optimization

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Generate data
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

def ml_objective(params, X, y, **kwargs):
    model = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    
    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    return np.mean(scores)

search_space = {
    'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10}
}

config = BayesianTPEConfig(
    n_trials=30,
    enable_grid_search=True,
    backend='optuna'
)

optimizer = BayesianTPEOptimizer(config)
result = optimizer.optimize(ml_objective, search_space, X=X, y=y)
```

### Advanced Configuration

```python
config = BayesianTPEConfig(
    # Grid search settings
    enable_grid_search=True,
    coarse_grid_points=5,
    fine_grid_points=8,
    
    # TPE settings
    n_trials=50,
    backend='optuna',
    
    # Parallel processing
    enable_parallel=True,
    max_workers=4,
    
    # Early stopping
    enable_early_stopping=True,
    early_stopping_patience=10,
    
    # Convergence detection
    enable_convergence_detection=True,
    convergence_threshold=0.01,
    
    # Logging
    log_level='DEBUG',
    log_file='optimization.log',
    
    # Performance monitoring
    enable_performance_monitoring=True,
    monitor_memory=True,
    monitor_time=True
)
```

## 📈 Optimization Results

### OptimizationResult Object

```python
result = optimizer.optimize(objective_function, search_space)

# Access results
print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score}")
print(f"Optimization time: {result.optimization_time:.2f}s")
print(f"Number of trials: {result.n_trials}")
print(f"Success: {result.success}")

# Convergence information
print(f"Best method: {result.convergence_info['best_method']}")
print(f"Grid search used: {result.convergence_info['grid_search_used']}")
print(f"TPE optimization used: {result.convergence_info['tpe_optimization_used']}")

# Optimization history
for entry in result.optimization_history:
    print(f"Stage: {entry['stage']}, Score: {entry['best_score']:.4f}")
```

## 🔍 Optimization Process

The optimizer follows a three-stage process:

1. **Coarse Grid Search**: Explores the entire search space with a coarse grid
2. **Fine Grid Search**: Refines around the best coarse results
3. **Bayesian TPE**: Performs final optimization using Tree-structured Parzen Estimator

### Stage Details

#### Stage 1: Coarse Grid Search
- Uses `build_coarse_grid_from_search_space` from your grid utils
- Evaluates all combinations of coarse parameter values
- Identifies promising regions of the search space

#### Stage 2: Fine Grid Search  
- Uses `build_fine_grid_around_best` from your grid utils
- Focuses on the best region found in coarse search
- Provides a good starting point for TPE

#### Stage 3: Bayesian TPE
- Uses Optuna or scikit-optimize for final optimization
- Leverages the good starting point from grid search
- Efficiently explores the refined search space

## 🛠️ Error Handling

The optimizer includes comprehensive error handling:

- **Objective Function Errors**: Continues optimization even if some evaluations fail
- **Memory Management**: Automatic cleanup of optimization history
- **Backend Fallbacks**: Graceful handling of missing optimization backends
- **Logging**: Detailed error logging for debugging

```python
def risky_objective(params, **kwargs):
    if np.random.random() < 0.1:  # 10% failure rate
        raise ValueError("Simulated error")
    return -(params['x'] - 1)**2 - (params['y'] - 2)**2

# Optimizer will handle errors gracefully
result = optimizer.optimize(risky_objective, search_space)
```

## 📊 Performance Monitoring

Enable performance monitoring to track optimization metrics:

```python
config = BayesianTPEConfig(
    enable_performance_monitoring=True,
    monitor_memory=True,
    monitor_time=True
)

optimizer = BayesianTPEOptimizer(config)
result = optimizer.optimize(objective_function, search_space)

# Access performance metrics
if hasattr(optimizer, 'performance_metrics'):
    print(f"Memory usage: {optimizer.performance_metrics['memory_usage']}")
    print(f"Execution times: {optimizer.performance_metrics['execution_times']}")
```

## 🧪 Testing

Run the comprehensive test suite:

```python
from src.utils.ml_common.optimization.test_bayesian_tpe_optimizer import run_tests

success = run_tests()
if success:
    print("✅ All tests passed!")
```

## 📚 Examples

See `bayesian_tpe_examples.py` for comprehensive examples:

- Simple function optimization
- Machine learning hyperparameter optimization
- Advanced configuration examples
- Error handling demonstrations
- Performance benchmarking

## 🔧 Integration with Existing Code

The optimizer seamlessly integrates with your existing grid utilities:

```python
# Your existing grid utils are automatically used
from src.utils.ml_common.optimization.grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)

# The BayesianTPEOptimizer automatically calls these functions
optimizer = BayesianTPEOptimizer(config)
result = optimizer.optimize(objective_function, search_space)
```

## 🚀 Best Practices

1. **Start with Grid Search**: Always enable grid search for initial exploration
2. **Use Appropriate Grid Sizes**: Balance between exploration and computation time
3. **Monitor Performance**: Enable performance monitoring for large optimizations
4. **Handle Errors**: Implement robust objective functions with error handling
5. **Log Progress**: Use detailed logging for debugging and monitoring
6. **Parallel Processing**: Enable parallel processing for faster optimization
7. **Early Stopping**: Use early stopping to avoid over-optimization

## 📝 Logging

Configure detailed logging for monitoring and debugging:

```python
config = BayesianTPEConfig(
    log_level='DEBUG',
    log_file='optimization.log',
    enable_progress_logging=True
)
```

The optimizer provides comprehensive logging including:
- Optimization progress
- Performance metrics
- Error messages
- Convergence information
- Memory usage

## 🤝 Contributing

The Bayesian TPE optimizer is designed to be extensible and maintainable. Key areas for contribution:

- Additional optimization backends
- Enhanced convergence detection
- Advanced parallel processing strategies
- Integration with more ML frameworks
- Performance optimizations

## 📄 License

This module is part of the ml_common utilities and follows the same licensing terms.