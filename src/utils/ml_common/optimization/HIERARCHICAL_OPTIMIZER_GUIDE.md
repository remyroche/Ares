# Hierarchical Parameter Optimizer - User Guide

## Overview

The Hierarchical Parameter Optimizer is a general-purpose optimization framework designed to efficiently tune hyperparameters without needing to optimize all parameters simultaneously. This addresses the **curse of dimensionality** in hyperparameter optimization.

### Key Features

1. **Parameter Grouping**: Organize parameters into logical groups and optimize them sequentially
2. **Multi-Round Optimization**: Default 2 rounds (exploration + refinement) to capture parameter interactions
3. **Staged Optimization**: Coarse grid → Fine grid → Advanced methods (TPE, BOHB)
4. **Backend Agnostic**: Works with Optuna TPE, BOHB, Random Search, etc.
5. **Memory Efficient**: Reduces search space complexity
6. **Integration Ready**: Compatible with existing optimization tools

### When to Use

Use hierarchical optimization when:
- You have **many parameters** to tune (>10)
- Parameters can be grouped by **purpose** or **importance**
- You want to **save computation time** by not exploring the full cartesian product
- Some parameters are **more critical** than others
- You want a **principled approach** to staged optimization
- Parameter interactions exist between groups (captured via 2+ rounds)

## Quick Start

### Basic Example: LightGBM Optimization

```python
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
from src.utils.ml_common.optimization import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
    default_objective_function
)

# Generate sample data
X_train, y_train = make_regression(n_samples=1000, n_features=20, random_state=42)

# Define parameter groups
param_groups = [
    # Group 1: Model structure (optimize first - highest priority)
    create_param_group(
        name="structure",
        params={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "num_leaves": {"type": "int", "low": 20, "high": 150},
            "max_depth": {"type": "int", "low": 3, "high": 12}
        },
        priority=1,
        description="Core model structure parameters"
    ),
    
    # Group 2: Learning rate (depends on structure)
    create_param_group(
        name="learning",
        params={
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
            "min_child_samples": {"type": "int", "low": 5, "high": 100}
        },
        priority=2,
        depends_on=["structure"],
        description="Learning rate and related parameters"
    ),
    
    # Group 3: Regularization (fine-tune last)
    create_param_group(
        name="regularization",
        params={
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0}
        },
        priority=3,
        depends_on=["structure", "learning"],
        description="Regularization parameters"
    )
]

# Create model
model = LGBMRegressor(random_state=42, verbose=-1)

# Create optimizer
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    stages=[
        OptimizationStage.COARSE_GRID,
        OptimizationStage.FINE_GRID,
        OptimizationStage.TPE
    ],
    cv_folds=5,
    scoring_metric='neg_mean_squared_error',
    direction='maximize',
    enable_final_refinement=True,
    final_refinement_trials=50,
    random_state=42,
    verbose=True
)

# Run optimization
result = optimizer.optimize(
    X_train=X_train,
    y_train=y_train,
    model=model
)

# Access results
print("Best parameters:", result.best_params)
print("Best score:", result.best_score)
print("Total time:", result.total_time)
print("Total trials:", result.total_trials)

# Use optimized parameters
model.set_params(**result.best_params)
model.fit(X_train, y_train)
```

## Advanced Usage

### Multiple Optimization Rounds

The optimizer performs **2 rounds by default** to capture parameter interactions:

**Round 1 (Exploration)**:
- Full search space exploration
- Optimizes each group with coarse → fine → TPE
- Establishes baseline parameter values

**Round 2 (Refinement)**:
- Narrowed search space (±15% around Round 1 results)
- Re-optimizes groups with updated context
- Captures interactions: optimal values for Group A may change after Group B is optimized

```python
# Use default 2 rounds (recommended)
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    n_rounds=2  # Default
)

# Single round (faster but may miss interactions)
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    n_rounds=1
)

# Three rounds (thorough but slower)
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    n_rounds=3
)
```

**Why 2 Rounds?**
- **Parameter Interactions**: Group A optimal values may depend on Group B
- **Iterative Refinement**: Second pass improves convergence
- **Balanced Trade-off**: Good results without excessive computation
- **Empirical Best Practice**: Most models converge well in 2 rounds

### Custom Objective Function

If you need custom evaluation logic, define your own objective function:

```python
def custom_objective(
    params,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    model=None,
    cv_folds=5,
    scoring_metric='neg_mean_squared_error'
):
    """
    Custom objective function with additional logic.
    
    Args:
        params: Parameter dictionary to evaluate
        X_train, y_train: Training data
        X_val, y_val: Validation data (optional)
        model: Model instance
        cv_folds: Number of CV folds
        scoring_metric: Metric to optimize
    
    Returns:
        float: Score to maximize/minimize
    """
    # Set parameters
    model.set_params(**params)
    
    # Add custom preprocessing
    # ... your code here ...
    
    # Evaluate with cross-validation
    if X_val is None:
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=cv_folds)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric)
        return np.mean(scores)
    else:
        # Evaluate on validation set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        from sklearn.metrics import mean_squared_error
        return -mean_squared_error(y_val, y_pred)

# Use custom objective
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=custom_objective,  # Use custom function
    # ... other parameters ...
)
```

### Different Optimization Stages

You can customize which stages to use:

```python
from src.utils.ml_common.optimization import OptimizationStage

# Quick optimization: skip fine grid
stages_quick = [
    OptimizationStage.COARSE_GRID,
    OptimizationStage.TPE
]

# Thorough optimization: all stages
stages_thorough = [
    OptimizationStage.COARSE_GRID,
    OptimizationStage.FINE_GRID,
    OptimizationStage.TPE
]

# Random baseline
stages_random = [
    OptimizationStage.RANDOM
]

optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    stages=stages_thorough,  # Choose your stages
    # ...
)
```

### Custom Stage Configuration

Control the behavior of each optimization stage:

```python
from src.utils.ml_common.optimization import StageConfig, OptimizationStage

stage_configs = {
    OptimizationStage.COARSE_GRID: StageConfig(
        stage=OptimizationStage.COARSE_GRID,
        n_trials=30,
        grid_points=3,  # 3 points per parameter
        enable_pruning=False
    ),
    OptimizationStage.FINE_GRID: StageConfig(
        stage=OptimizationStage.FINE_GRID,
        n_trials=50,
        grid_points=5,  # 5 points per parameter
        enable_pruning=False
    ),
    OptimizationStage.TPE: StageConfig(
        stage=OptimizationStage.TPE,
        n_trials=200,  # More trials for TPE
        n_startup_trials=20,
        n_ei_candidates=24,
        enable_pruning=True,
        timeout_seconds=3600  # 1 hour timeout
    )
}

optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    stages=[OptimizationStage.COARSE_GRID, OptimizationStage.FINE_GRID, OptimizationStage.TPE],
    stage_configs=stage_configs,  # Use custom configs
    # ...
)
```

### Disable Final Refinement

If you don't want joint optimization of all parameters at the end:

```python
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    enable_final_refinement=False,  # Skip final refinement
    # ...
)
```

## Parameter Group Design

### Grouping Strategies

**1. By Purpose**
```python
# Structure parameters
structure_group = create_param_group(
    name="structure",
    params={"n_estimators": ..., "max_depth": ...},
    priority=1
)

# Regularization parameters
regularization_group = create_param_group(
    name="regularization",
    params={"reg_alpha": ..., "reg_lambda": ...},
    priority=2
)
```

**2. By Importance**
```python
# Critical parameters (optimize first)
critical_group = create_param_group(
    name="critical",
    params={"learning_rate": ..., "n_estimators": ...},
    priority=1
)

# Fine-tuning parameters (optimize last)
finetuning_group = create_param_group(
    name="finetuning",
    params={"subsample": ..., "colsample_bytree": ...},
    priority=3
)
```

**3. By Computational Cost**
```python
# Expensive parameters (optimize early with few trials)
expensive_group = create_param_group(
    name="expensive",
    params={"n_estimators": ..., "max_depth": ...},
    priority=1
)

# Cheap parameters (can afford more trials)
cheap_group = create_param_group(
    name="cheap",
    params={"learning_rate": ..., "reg_alpha": ...},
    priority=2
)
```

**4. With Dependencies**
```python
# Base architecture must be optimized first
architecture_group = create_param_group(
    name="architecture",
    params={"hidden_size": ..., "num_layers": ...},
    priority=1
)

# Training parameters depend on architecture
training_group = create_param_group(
    name="training",
    params={"learning_rate": ..., "batch_size": ...},
    priority=2,
    depends_on=["architecture"]  # Will be optimized after architecture
)
```

## Real-World Examples

### Example 1: XGBoost Classifier

```python
from xgboost import XGBClassifier

param_groups = [
    create_param_group(
        name="tree_structure",
        params={
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "n_estimators": {"type": "int", "low": 50, "high": 300}
        },
        priority=1
    ),
    create_param_group(
        name="learning",
        params={
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0}
        },
        priority=2,
        depends_on=["tree_structure"]
    ),
    create_param_group(
        name="regularization",
        params={
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0}
        },
        priority=3,
        depends_on=["tree_structure", "learning"]
    )
]

model = XGBClassifier(random_state=42)
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=default_objective_function,
    stages=[OptimizationStage.COARSE_GRID, OptimizationStage.TPE],
    cv_folds=5,
    scoring_metric='accuracy',
    direction='maximize'
)

result = optimizer.optimize(X_train, y_train, model=model)
```

### Example 2: Neural Network with Multiple Modules

```python
# For a neural network with embedding, encoder, and decoder
param_groups = [
    create_param_group(
        name="embedding",
        params={
            "embedding_dim": {"type": "int", "low": 64, "high": 512},
            "dropout_embed": {"type": "float", "low": 0.0, "high": 0.5}
        },
        priority=1
    ),
    create_param_group(
        name="encoder",
        params={
            "hidden_size": {"type": "int", "low": 128, "high": 1024},
            "num_layers": {"type": "int", "low": 1, "high": 4},
            "dropout_encoder": {"type": "float", "low": 0.0, "high": 0.5}
        },
        priority=2,
        depends_on=["embedding"]
    ),
    create_param_group(
        name="training",
        params={
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1}
        },
        priority=3,
        depends_on=["embedding", "encoder"]
    )
]
```

### Example 3: Time Series Model

```python
# For a time series forecasting model
param_groups = [
    create_param_group(
        name="lookback",
        params={
            "lookback_window": {"type": "int", "low": 5, "high": 50},
            "stride": {"type": "int", "low": 1, "high": 5}
        },
        priority=1
    ),
    create_param_group(
        name="model_capacity",
        params={
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 3, "high": 12}
        },
        priority=2,
        depends_on=["lookback"]
    ),
    create_param_group(
        name="regularization",
        params={
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0}
        },
        priority=3,
        depends_on=["lookback", "model_capacity"]
    )
]
```

## Understanding the Results

The optimizer returns a `HierarchicalOptimizationResult` object with:

```python
result = optimizer.optimize(...)

# Best parameters (combined from all groups + refinement)
print(result.best_params)  # Dict[str, Any]

# Best score achieved
print(result.best_score)  # float

# Total optimization time
print(result.total_time)  # float (seconds)

# Total number of trials across all groups and stages
print(result.total_trials)  # int

# Results for each parameter group
for group_result in result.group_results:
    print(f"Group: {group_result.group_name}")
    print(f"  Best params: {group_result.best_params}")
    print(f"  Best score: {group_result.best_score}")
    print(f"  Trials: {group_result.n_trials}")
    print(f"  Time: {group_result.optimization_time}s")

# Final refinement result (if enabled)
if result.final_refinement_result:
    print("Final refinement:")
    print(f"  Score: {result.final_refinement_result.best_score}")
    print(f"  Trials: {result.final_refinement_result.n_trials}")

# Save results to JSON
result_dict = result.to_dict()
import json
with open('optimization_results.json', 'w') as f:
    json.dump(result_dict, f, indent=2)
```

## Performance Tips

### 1. Start with Coarse Grid
Always start with a coarse grid search to quickly identify promising regions:
```python
stages = [OptimizationStage.COARSE_GRID, OptimizationStage.TPE]
```

### 2. Use Parameter Dependencies
Define dependencies to ensure parameters are optimized in a logical order:
```python
create_param_group(
    name="advanced",
    depends_on=["basic"],  # Wait for basic params to be optimized first
    # ...
)
```

### 3. Prioritize Important Parameters
Optimize high-impact parameters first (lower priority number):
```python
create_param_group(name="critical", priority=1, ...)  # Optimize first
create_param_group(name="minor", priority=3, ...)     # Optimize last
```

### 4. Use Logarithmic Scale for Learning Rates
```python
params = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 0.1, "log": True}
}
```

### 5. Limit Trials for Expensive Operations
For computationally expensive evaluations, reduce the number of trials:
```python
stage_configs = {
    OptimizationStage.TPE: StageConfig(
        stage=OptimizationStage.TPE,
        n_trials=50,  # Fewer trials for expensive models
        timeout_seconds=1800  # 30 minute timeout
    )
}
```

### 6. Cache Results
Enable result caching to avoid re-running optimizations:
```python
optimizer = HierarchicalParameterOptimizer(
    cache_dir="./optimization_cache",  # Results saved here
    # ...
)
```

## Comparison with Traditional Approaches

### Traditional Grid Search
```python
# Problem: Cartesian product of all parameters
# If you have 5 params with 10 values each: 10^5 = 100,000 combinations!
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'reg_alpha': [0.0, 0.25, 0.5, 0.75, 1.0],
    'reg_lambda': [0.0, 0.25, 0.5, 0.75, 1.0]
}
# Total: 5 * 5 * 5 * 5 * 5 = 3,125 combinations
```

### Hierarchical Optimization
```python
# Solution: Optimize groups sequentially
# Group 1: n_estimators, max_depth (5*5 = 25 combinations)
# Group 2: learning_rate (5 combinations, with best from Group 1 fixed)
# Group 3: reg_alpha, reg_lambda (5*5 = 25 combinations, with best from Groups 1-2 fixed)
# Total: 25 + 5 + 25 = 55 combinations (98% reduction!)
```

## Troubleshooting

### Issue: Optimization takes too long
**Solution**: Reduce number of trials or skip fine grid stage
```python
optimizer = HierarchicalParameterOptimizer(
    stages=[OptimizationStage.COARSE_GRID, OptimizationStage.TPE],  # Skip fine grid
    stage_configs={
        OptimizationStage.TPE: StageConfig(n_trials=30)  # Fewer trials
    }
)
```

### Issue: Coarse grid finds no good parameters
**Solution**: Expand search space or increase grid points
```python
stage_configs = {
    OptimizationStage.COARSE_GRID: StageConfig(
        grid_points=5  # More points per parameter (default is 3)
    )
}
```

### Issue: Objective function fails
**Solution**: Add error handling to objective function
```python
def robust_objective(params, X_train, y_train, **kwargs):
    try:
        # Your evaluation code
        return score
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return float('-inf')  # Return worst possible score
```

### Issue: Dependencies not satisfied
**Solution**: Check dependency names match group names exactly
```python
# Correct
group1 = create_param_group(name="structure", ...)
group2 = create_param_group(name="learning", depends_on=["structure"], ...)

# Incorrect (typo in dependency name)
group2 = create_param_group(name="learning", depends_on=["strucure"], ...)  # Will raise error
```

## API Reference

See the module docstring in `hierarchical_parameter_optimizer.py` for complete API documentation.

## Integration with Other Tools

The hierarchical optimizer integrates seamlessly with:
- **Optuna**: Used for TPE optimization
- **scikit-learn**: Compatible with sklearn estimators
- **LightGBM, XGBoost**: Full support
- **Custom models**: Works with any model that has `fit()`, `predict()`, and `set_params()` methods

## Conclusion

The Hierarchical Parameter Optimizer provides a principled and efficient approach to hyperparameter tuning. By organizing parameters into groups and using staged optimization, you can:

1. **Reduce computation time** by orders of magnitude
2. **Improve optimization quality** through focused search
3. **Gain insights** into parameter importance and interactions
4. **Scale** to high-dimensional parameter spaces

For questions or issues, refer to the module source code or contact the development team.
