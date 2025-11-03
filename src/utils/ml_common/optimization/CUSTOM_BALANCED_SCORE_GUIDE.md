# Custom Balanced Score - Default HPO Metric for ML Trading Models

## Overview

As of this update, **`custom_balanced_score`** is now the **default scoring metric** for all ML-related trading models in HPO (Hyperparameter Optimization). This comprehensive metric balances:

- **Financial Performance (45%)**: Sharpe ratio, profit factor, drawdown, returns
- **Statistical Accuracy (35%)**: F1 score, accuracy, R²
- **Regime Awareness (10%)**: Market regime adaptation and stability
- **Economic Viability (10%)**: Trading practicality and implementation feasibility

## Key Benefits

1. **Holistic Evaluation**: Considers both predictive accuracy AND financial viability
2. **Regime-Aware**: Adapts to different market conditions
3. **Practical Focus**: Includes economic constraints and trading costs
4. **Multi-Objective**: Can decompose into component objectives for analysis
5. **Pareto-Optimal**: Optional integration with Pareto optimization tools

## Quick Start

### 1. Using with HierarchicalParameterOptimizer (Recommended)

```python
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_custom_balanced_score_objective,
    ParameterGroup
)

# Define your model trainer
def train_my_model(params, X_train, y_train, X_val, y_val):
    model = MyModel(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return model, predictions

# Create objective function using custom_balanced_score
objective_func = create_custom_balanced_score_objective(
    model_trainer=train_my_model,
    use_returns=True,  # Calculate financial metrics
    use_regime_labels=False  # Set True if regime labels available
)

# Define parameter groups
param_groups = [
    ParameterGroup(
        name="model_structure",
        params={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 12}
        },
        priority=1
    )
]

# Create optimizer (uses custom_balanced_score by default)
optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=objective_func,
    direction='maximize',  # custom_balanced_score should be maximized
    n_rounds=2
)

# Run optimization
result = optimizer.optimize(X_train, y_train, X_val, y_val)
```

### 2. Direct Usage (Without Wrapper)

```python
from src.utils.ml_common.optimization.shared_utils.evaluation_metrics import (
    calculate_custom_balanced_score_for_hpo
)

def my_objective_function(params, X_train, y_train, X_val, y_val, **kwargs):
    # Train model
    model = create_model(params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    # Calculate returns (example)
    returns = predictions * y_val  # Simplified example
    
    # Calculate custom_balanced_score
    score = calculate_custom_balanced_score_for_hpo(
        predictions=predictions,
        targets=y_val,
        returns=returns,
        regime_labels=kwargs.get('regime_labels', None)
    )
    
    return score
```

### 3. Advanced Usage with Component Analysis

```python
from src.utils.ml_common.optimization.shared_utils.evaluation_metrics import (
    create_unified_evaluator
)

evaluator = create_unified_evaluator()
result = evaluator.evaluate(predictions, targets, returns, regime_labels)

# Access all components
print(f"Overall Score: {result.custom_balanced_score:.4f}")
print(f"Financial Component: {result.financial_metrics.sharpe_ratio:.4f}")
print(f"Statistical Component: {result.statistical_metrics.f1_score:.4f}")
print(f"Regime Component: {result.regime_metrics.regime_accuracy:.4f}")
print(f"Economic Component: {result.economic_metrics.trading_viability:.4f}")
```

## Score Components Breakdown

### Financial Metrics (45%)
- **Sharpe Ratio** (35%): Risk-adjusted returns
- **Profit Factor** (25%): Gross profit / gross loss
- **Max Drawdown** (20%): Largest peak-to-trough decline (inverted)
- **Sortino Ratio** (10%): Downside risk-adjusted returns
- **Total Return** (5%): Cumulative returns
- **Calmar Ratio** (5%): Return / max drawdown

### Statistical Metrics (35%)
- **F1 Score** (50%): Harmonic mean of precision and recall
- **Accuracy** (25%): Correct predictions / total predictions
- **R² Score** (15%): Coefficient of determination
- **Precision** (5%): True positives / (true positives + false positives)
- **Recall** (5%): True positives / (true positives + false negatives)

### Regime-Aware Metrics (10%)
- **Regime Accuracy** (50%): Prediction accuracy per regime
- **Regime Stability** (30%): Consistency across regime transitions
- **Regime Consistency** (20%): Variance in performance across regimes

### Economic Metrics (10%)
- **Economic Significance** (60%): Statistical vs practical significance
- **Trading Viability** (40%): Implementation feasibility with costs

## Configuration Options

### Custom Weights

```python
custom_weights = {
    'sharpe': 0.30,
    'max_drawdown': 0.20,
    'profit_factor': 0.15,
    'f1_score': 0.20,
    'accuracy': 0.10,
    'r2_score': 0.05
}

result = evaluator.evaluate(
    predictions, targets, returns,
    custom_weights=custom_weights
)
```

### Pareto-Optimal Scalarization

```python
from src.utils.ml_common.optimization.shared_utils.evaluation_metrics import (
    UnifiedEvaluator
)

evaluator = UnifiedEvaluator()
result = evaluator.evaluate(
    predictions, targets, returns,
    use_pareto_scalarization=True  # Uses advanced Pareto optimization
)
```

### Sample Count Penalty

```python
# Automatically penalizes scores when sample count is low
score = calculate_custom_balanced_score_for_hpo(
    predictions=predictions,
    targets=targets,
    returns=returns,
    sample_count=len(predictions),
    sample_count_min=30,  # Minimum samples before penalty
    apply_sample_penalty=True
)
```

## Integration with Existing Code

### For New Models
Simply use `HierarchicalParameterOptimizer` with default settings - it will automatically use `custom_balanced_score`.

### For Existing Models
If you have custom objective functions, you can:

1. **Wrap your existing function**:
```python
objective_func = create_custom_balanced_score_objective(your_trainer)
```

2. **Keep custom logic but use score calculation**:
```python
def my_objective(params, X_train, y_train, X_val, y_val, **kwargs):
    # Your custom training logic
    predictions = train_and_predict(params, X_train, y_train, X_val)
    
    # Use custom_balanced_score for evaluation
    return calculate_custom_balanced_score_for_hpo(
        predictions, y_val, returns=calc_returns(predictions, y_val)
    )
```

## Backward Compatibility

- Existing code using `scoring_metric='neg_mean_squared_error'` continues to work
- To opt-out of custom_balanced_score: `use_custom_balanced_score=False`
- Default direction changed to `'maximize'` (appropriate for custom_balanced_score)

## Performance Considerations

### Computational Efficiency
- **Fast**: Vectorized operations using NumPy
- **Memory-efficient**: Streaming calculations for large datasets
- **Scalable**: O(n) complexity for n predictions

### Hardware Optimization
- Automatically uses available GPU acceleration (via Pareto tools)
- Supports batch processing for large-scale HPO
- Integrates with VectorBT for rolling window optimizations

## Troubleshooting

### Issue: Score is always 0
**Solution**: Check that financial_metrics and statistical_metrics are being calculated correctly. The score requires at least some metrics to be non-None.

### Issue: Score seems too low
**Solution**: Review the normalization ranges in the config. Default ranges are:
- Sharpe: -1.0 to 3.0
- Max Drawdown: 0.0 to 0.6
- Profit Factor: 0.0 to 5.0

### Issue: Want to emphasize specific metrics
**Solution**: Pass custom weights to adjust the importance of different components.

## Examples

### Example 1: LGBM Trading Model
```python
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_custom_balanced_score_objective,
    ParameterGroup
)
import lightgbm as lgb

def train_lgbm(params, X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(**params, verbose=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return model, predictions

objective = create_custom_balanced_score_objective(train_lgbm)

param_groups = [
    ParameterGroup(
        name="boosting",
        params={
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True}
        },
        priority=1
    )
]

optimizer = HierarchicalParameterOptimizer(
    param_groups=param_groups,
    objective_func=objective,
    n_rounds=2
)

result = optimizer.optimize(X_train, y_train, X_val, y_val)
print(f"Best Score: {result.best_score:.4f}")
print(f"Best Params: {result.best_params}")
```

### Example 2: Regime-Aware Optimization
```python
# With regime labels
regime_labels = detect_market_regimes(data)

def train_with_regimes(params, X_train, y_train, X_val, y_val):
    model = MyRegimeAwareModel(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return model, predictions

objective = create_custom_balanced_score_objective(
    train_with_regimes,
    use_regime_labels=True
)

# Pass regime_labels in kwargs during optimization
result = optimizer.optimize(
    X_train, y_train, X_val, y_val,
    regime_labels=regime_labels
)
```

## API Reference

See:
- `evaluation_metrics.py` - Core implementation
- `hierarchical_parameter_optimizer.py` - Main HPO interface
- `pareto.py` - Multi-objective optimization tools

## Migration Guide

### From `neg_mean_squared_error` to `custom_balanced_score`

**Before:**
```python
optimizer = HierarchicalParameterOptimizer(
    param_groups=groups,
    objective_func=lambda p, X, y, **kw: -mse(y, predict(p, X)),
    scoring_metric='neg_mean_squared_error',
    direction='maximize'
)
```

**After:**
```python
optimizer = HierarchicalParameterOptimizer(
    param_groups=groups,
    objective_func=create_custom_balanced_score_objective(train_fn),
    # scoring_metric='custom_balanced_score' is now default
    # direction='maximize' is now default
)
```

## Best Practices

1. **Always maximize**: `custom_balanced_score` should be maximized (it's normalized to [0, 1])
2. **Provide returns**: Include return calculations for accurate financial metrics
3. **Use regime labels**: If available, pass regime labels for regime-aware scoring
4. **Monitor components**: Use `return_components=True` to analyze score breakdown
5. **Validate ranges**: Check that metric values fall within expected normalization ranges

## Future Enhancements

- [ ] Integration with risk management constraints
- [ ] Dynamic weight adjustment based on market conditions
- [ ] Support for multi-asset portfolio optimization
- [ ] Enhanced Pareto front analysis
- [ ] Custom metric plugins

## Support

For questions or issues:
1. Check this guide first
2. Review `evaluation_metrics.py` implementation
3. See examples in `example_hierarchical_optimization.py`
4. Check inline documentation in source files

