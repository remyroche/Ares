# Bayesian TPE Optimizer

A comprehensive hyperparameter optimization module that automatically uses coarse grid search followed by fine grid search as initialization stages before Bayesian Tree-structured Parzen Estimator (TPE) optimization.

## Overview

The `BayesianTPEOptimizer` provides a robust, production-ready solution for hyperparameter optimization that combines the reliability of grid search with the efficiency of Bayesian optimization. This hybrid approach ensures better convergence and more reliable results across different types of machine learning problems.

## Key Features

### 🎯 Automatic Pipeline
- **Stage 1**: Coarse grid search to identify promising regions
- **Stage 2**: Fine grid search around best coarse parameters
- **Stage 3**: Bayesian TPE optimization for final refinement

### 🔧 Robust Error Handling
- Comprehensive logging with detailed error tracking
- Graceful degradation when optimization stages fail
- Automatic fallback to default parameters
- Transfer learning support between similar datasets

### 📊 Monitoring & Validation
- Built-in convergence detection and monitoring
- Support for custom evaluation functions
- Performance metrics and optimization history tracking
- Results persistence and analysis

### 🚀 Flexibility
- Support for multiple model types (XGBoost, LightGBM, Random Forest, Neural Networks)
- Custom search spaces and evaluation functions
- Parallel optimization support
- Configurable optimization strategies

## Installation

The optimizer is part of the `src.utils.ml_common.optimization` package. Required dependencies:

```bash
pip install optuna scikit-learn numpy pandas
```

## Quick Start

### Basic Usage

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    create_optimization_config
)

# Create model factory
def create_model(params):
    from xgboost import XGBClassifier
    return XGBClassifier(**params)

# Configure optimization
config = create_optimization_config(
    n_trials=50,
    coarse_grid_points=5,
    fine_grid_points=10
)

# Run optimization
optimizer = BayesianTPEOptimizer(config=config, model_type='xgboost')
results = optimizer.optimize(create_model, X_train, y_train)

print(f"Best score: {results.best_score:.4f}")
print(f"Best parameters: {results.best_params}")
```

### Simple API

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import optimize_hyperparameters

results = optimize_hyperparameters(
    model_factory=create_model,
    X=X_train,
    y=y_train,
    model_type='xgboost',
    n_trials=50
)
```

## Configuration

### OptimizationConfig

```python
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import OptimizationConfig, TPEConfig, GridConfig

config = OptimizationConfig(
    tpe_config=TPEConfig(
        n_trials=50,
        timeout_seconds=3600,
        acquisition_function='ucb',
        pruner='median',
        enable_parallel=True,
        max_workers=4
    ),
    grid_config=GridConfig(
        coarse_enabled=True,
        coarse_grid_points=5,
        fine_enabled=True,
        fine_grid_points=10,
        subsample_rate=0.3
    ),
    validation_config={
        'cv_folds': 5,
        'scoring': 'balanced_accuracy',
        'test_size': 0.2,
        'random_state': 42
    },
    enable_monitoring=True,
    fast_fail_on_error=False,
    save_results=True,
    results_path='./optimization_results'
)
```

### Search Spaces

#### XGBoost Example
```python
search_space = {
    'max_depth': {'type': 'int', 'low': 3, 'high': 12},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
    'gamma': {'type': 'float', 'low': 0, 'high': 5},
    'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
    'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
}
```

#### Neural Network Example
```python
search_space = {
    'hidden_layer_sizes': {
        'type': 'categorical',
        'choices': [(50,), (100,), (50, 25), (100, 50)]
    },
    'activation': {
        'type': 'categorical',
        'choices': ['relu', 'tanh', 'logistic']
    },
    'learning_rate_init': {
        'type': 'float',
        'low': 0.001,
        'high': 0.1,
        'log': True
    },
    'max_iter': {'type': 'int', 'low': 100, 'high': 500}
}
```

## Advanced Usage

### Custom Evaluation Functions

```python
def custom_business_metric(model, X, y):
    """Custom evaluation combining accuracy and model simplicity."""
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')

    # Penalize model complexity
    n_estimators = getattr(model, 'n_estimators', 50)
    complexity_penalty = n_estimators / 100.0

    return 0.7 * accuracy + 0.3 * f1 - 0.1 * complexity_penalty

# Use custom evaluation
results = optimizer.optimize(
    create_model,
    X_train,
    y_train,
    custom_evaluation_fn=custom_business_metric
)
```

### Transfer Learning

```python
# Previous optimization results
transfer_data = {
    'best_params': {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
    'best_score': 0.85,
    'n_samples': 1000,
    'n_features': 20,
    'n_classes': 3
}

# New optimizer with transfer learning
optimizer = BayesianTPEOptimizer(
    config=OptimizationConfig(
        transfer_learning_threshold=0.8,  # Similarity threshold
        grid_config=GridConfig(coarse_enabled=False)  # Skip grid for transfer
    ),
    model_type='xgboost'
)

results = optimizer.optimize(
    create_model,
    X_new,
    y_new,
    transfer_learning_data=transfer_data
)
```

### Multi-Objective Optimization

```python
def multi_objective_evaluator(model, X, y):
    """Evaluate multiple objectives."""
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')

    # Training time as second objective
    import time
    start_time = time.time()
    model.predict(X)  # Quick prediction for timing
    training_time = time.time() - start_time

    # Combine objectives (example: weighted sum)
    return 0.8 * accuracy + 0.2 * f1 - 0.1 * training_time

# Use multi-objective evaluation
results = optimizer.optimize(
    create_model,
    X_train,
    y_train,
    custom_evaluation_fn=multi_objective_evaluator
)
```

## Model Support

The optimizer includes built-in support for:

### Tree-Based Models
- **XGBoost**: Gradient boosting with extensive hyperparameter space
- **LightGBM**: Fast gradient boosting with advanced features
- **Random Forest**: Ensemble learning with bagging

### Neural Networks
- **MLPClassifier/MLPRegressor**: Multi-layer perceptron
- Configurable architecture (hidden layers, activation functions)
- Learning rate and regularization optimization

### Custom Models
Any model that follows scikit-learn API can be optimized by providing a model factory function.

## Monitoring and Analysis

### Logging
The optimizer provides comprehensive logging at different levels:

```python
import logging

# Enable detailed logging
logging.getLogger('MLCommon.BayesianTPEOptimizer').setLevel(logging.DEBUG)

# Log file output
logging.basicConfig(
    filename='optimization.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Results Analysis
```python
# Access optimization history
print(f"Total trials: {results.n_trials_total}")
print(f"Coarse grid trials: {results.n_trials_coarse}")
print(f"Fine grid trials: {results.n_trials_fine}")
print(f"TPE trials: {results.n_trials_tpe}")

# Performance metrics
print(f"Best stage: {results.best_stage}")
print(f"Convergence info: {results.convergence_info}")
print(f"Performance metrics: {results.performance_metrics}")
```

### Results Persistence
```python
# Configure results saving
config = OptimizationConfig(
    save_results=True,
    results_path='./my_optimization_results'
)

# Results are automatically saved as JSON files
# - results_{study_id}.json: Main results
# - history_{study_id}.json: Optimization history
# - errors_{study_id}.json: Error summary
```

## Error Handling

### Robust Error Recovery
The optimizer handles various failure scenarios:

```python
# Graceful degradation when stages fail
config = OptimizationConfig(
    fast_fail_on_error=False  # Continue with remaining stages
)

# Automatic fallback to default parameters
try:
    results = optimizer.optimize(create_model, X_train, y_train)
except Exception as e:
    logger.error(f"Optimization failed: {e}")
    # Results still contain fallback parameters
    print(f"Fallback parameters: {results.best_params}")
```

### Error Types Handled
- **Model creation failures**: Invalid parameter combinations
- **Evaluation failures**: Cross-validation errors
- **Dependency issues**: Missing optional libraries
- **Timeout errors**: Long-running evaluations
- **Memory errors**: Out of memory during optimization

## Performance Tuning

### Parallel Optimization
```python
config = OptimizationConfig(
    tpe_config=TPEConfig(
        enable_parallel=True,
        max_workers=4  # Adjust based on CPU cores
    )
)
```

### Memory Optimization
```python
# Reduce grid search density for large search spaces
config = OptimizationConfig(
    grid_config=GridConfig(
        coarse_grid_points=3,  # Fewer points for coarse search
        fine_grid_points=5,    # Fewer points for fine search
        subsample_rate=0.2     # Subsample data for coarse stage
    )
)
```

### Convergence Settings
```python
# Custom validation configuration
config = OptimizationConfig(
    validation_config={
        'cv_folds': 3,           # Fewer folds for speed
        'scoring': 'f1_macro',   # Different scoring metric
        'test_size': 0.3         # Larger test set
    }
)
```

## Best Practices

### 1. Search Space Design
- Start with reasonable parameter ranges
- Use logarithmic scaling for learning rates
- Include categorical choices for discrete parameters
- Balance search space size with computational budget

### 2. Evaluation Strategy
- Use cross-validation for robust evaluation
- Consider custom metrics for business objectives
- Balance accuracy with model complexity
- Validate on holdout set after optimization

### 3. Resource Management
- Monitor memory usage for large datasets
- Use parallel optimization judiciously
- Set reasonable timeouts for long-running models
- Save intermediate results for long optimizations

### 4. Transfer Learning
- Apply when datasets are similar in characteristics
- Use higher similarity thresholds for reliability
- Skip grid stages when transferring from known good results
- Validate transfer learning results carefully

## Integration Examples

### Integration with Existing Codebases
```python
# Wrapper for existing optimization workflows
class ModelOptimizer:
    def __init__(self, model_class, default_params=None):
        self.model_class = model_class
        self.default_params = default_params or {}

    def optimize(self, X, y, search_space=None):
        def model_factory(params):
            all_params = {**self.default_params, **params}
            return self.model_class(**all_params)

        optimizer = BayesianTPEOptimizer(model_type='auto')
        return optimizer.optimize(model_factory, X, y, search_space)

# Usage
optimizer = ModelOptimizer(RandomForestClassifier)
results = optimizer.optimize(X_train, y_train)
```

### Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(params):
    model_params = {k.replace('model__', ''): v
                   for k, v in params.items() if k.startswith('model__')}
    preprocessor_params = {k.replace('preprocessor__', ''): v
                          for k, v in params.items() if k.startswith('preprocessor__')}

    return Pipeline([
        ('preprocessor', StandardScaler(**preprocessor_params)),
        ('model', RandomForestClassifier(**model_params))
    ])

# Optimize entire pipeline
results = optimize_hyperparameters(
    create_pipeline,
    X_train,
    y_train,
    search_space={
        'model__n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'model__max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'preprocessor__with_mean': {'type': 'categorical', 'choices': [True, False]}
    }
)
```

## Troubleshooting

### Common Issues

#### 1. Optuna Import Error
```bash
pip install optuna
# or
conda install -c conda-forge optuna
```

#### 2. Memory Issues
- Reduce `coarse_grid_points` and `fine_grid_points`
- Set `subsample_rate` to a lower value
- Use `enable_parallel=False` for memory-constrained environments

#### 3. Poor Convergence
- Increase `n_trials` for TPE stage
- Expand search space ranges
- Check custom evaluation function for consistency
- Verify data quality and preprocessing

#### 4. Slow Optimization
- Enable parallel processing
- Reduce cross-validation folds
- Use smaller search spaces
- Consider transfer learning from similar problems

### Debug Mode
```python
import logging

# Enable debug logging
logging.getLogger('MLCommon.BayesianTPEOptimizer').setLevel(logging.DEBUG)

# Run optimization with detailed output
results = optimizer.optimize(create_model, X_train, y_train)
```

## Contributing

To extend the optimizer:

1. **Add Model Support**: Extend `_get_default_search_space()` method
2. **Custom Optimizers**: Add new optimization strategies to the pipeline
3. **Evaluation Metrics**: Implement additional evaluation functions
4. **Integration**: Add support for additional ML frameworks

## License

This module is part of the ML Common utilities package and follows the same licensing terms.

---

For more examples and advanced usage, see `bayesian_tpe_example.py`.