# HPO & Pareto Optimization Integration Summary

## Overview

This document explains how Hyperparameter Optimization (HPO) and Pareto optimization are integrated for regime models training, focusing on transition-aware metrics and multi-objective optimization.

## HPO with Transition-Aware Metrics

### Current Implementation

The regime models training now uses **transition-aware scoring** for HPO, which balances:
- **Accuracy** (70% weight): Classification performance
- **Stability** (30% weight): Temporal smoothness and transition rate

### Transition-Aware Scorer

**Location**: `src/utils/ml_common/optimization/transition_aware_scoring.py`

**Function**: `create_transition_aware_scorer()`

**Formula**:
```
score = accuracy_weight * accuracy - stability_weight * (transition_rate + smoothness_penalty)
```

**Default Parameters**:
- `alpha = 0.1`: Temporal smoothness penalty weight
- `accuracy_weight = 0.7`: Weight for accuracy component
- `stability_weight = 0.3`: Weight for stability component

### Integration in Training

**Location**: `regime_models_training.py` → `_train_models_with_hpo()`

**Usage**:
```python
from src.utils.ml_common.optimization.transition_aware_scoring import create_transition_aware_scorer

scorer = create_transition_aware_scorer(
    alpha=0.1,
    accuracy_weight=0.7,
    stability_weight=0.3
)

hpo_result = self.hpo_optimizer.optimize(
    model_factory=create_model,
    X=X_train,
    y=y_train,
    cv_folds=3,
    scoring=scorer,  # Transition-aware scorer
    n_trials=15
)
```

## Pareto Optimization for Multi-Objective HPO

### Overview

Pareto optimization finds a **Pareto front** of solutions that represent optimal trade-offs between multiple objectives. Instead of combining objectives into a single score, Pareto optimization identifies solutions where improving one objective worsens another.

### Available Tools

**Location**: `src/utils/ml_common/optimization/pareto.py`

**Classes**:
- `ParetoFront`: Main Pareto front computation
- `ParetoOptimizer`: Enhanced wrapper with non-linear transformations
- `Solution`: Container for solution metrics and parameters

**Functions**:
- `compute_pareto_front()`: Compute Pareto front from solutions
- `select_knee_point()`: Select knee point from Pareto front
- `compute_hypervolume()`: Calculate hypervolume indicator

### Multi-Objective Optimization

**Location**: `src/utils/ml_common/optimization/hpo_utils.py`

**Method**: `multi_objective_optimization()`

**Capabilities**:
- Optimizes multiple objectives simultaneously
- Uses Optuna for multi-objective optimization
- Supports objectives: 'accuracy', 'f1', 'auc', 'speed'

### Integration Strategy

For regime models, we optimize for:
1. **Accuracy** (maximize): Classification accuracy
2. **Transition Rate** (minimize): Regime switches per unit time
3. **Mean Episode Length** (maximize): Average consecutive bars in same regime

**Objectives Dictionary**:
```python
objectives: ObjectiveDirection = {
    'accuracy': 'max',
    'transition_rate': 'min',
    'mean_episode_length': 'max'
}
```

### Pareto-Based HPO Implementation

**Location**: `src/utils/ml_common/optimization/transition_aware_scoring.py`

**Function**: `create_pareto_multi_objective_hpo()`

**Process**:
1. Run HPO trials evaluating multiple objectives
2. Collect solutions with their metrics
3. Compute Pareto front from all solutions
4. Return Pareto-optimal solutions

**Usage** (Future Integration):
```python
from src.utils.ml_common.optimization.transition_aware_scoring import create_pareto_multi_objective_hpo

pareto_result = create_pareto_multi_objective_hpo(
    model_factory=create_model,
    X=X_train,
    y=y_train,
    cv_folds=3,
    n_trials=50,
    use_pareto_optimization=True
)

# Get Pareto-optimal solutions
pareto_solutions = pareto_result['pareto_solutions']
```

## Current vs. Future Implementation

### Current (Single-Objective Composite Score)

**Status**: ✅ Implemented

- Uses `create_transition_aware_scorer()` 
- Combines accuracy and stability into single score
- Works with existing HPO infrastructure
- Fast and simple

**Trade-off**: Requires manual weight tuning

### Future (Pareto Multi-Objective)

**Status**: ⚠️ Partially Implemented

- Framework exists in `transition_aware_scoring.py`
- Requires Optuna multi-objective study integration
- Provides Pareto front of trade-off solutions
- Allows selection of solution based on specific needs

**Benefits**: 
- No manual weight tuning needed
- See full trade-off space
- Select solution based on deployment requirements

## Recommended Approach

### For Production Use:

1. **Start with transition-aware composite scorer** (current)
   - Fast and reliable
   - Works with existing HPO infrastructure
   - Tune weights based on validation results

2. **Use Pareto optimization for exploration** (when available)
   - Run full Pareto optimization occasionally
   - Understand trade-off space
   - Select best solution based on requirements

### Configuration

Enable/disable optimization features:
```python
# In component initialization
self.enable_multi_objective_hpo = True  # Enable multi-objective
self.use_pareto_optimization = PARETO_AVAILABLE  # Use Pareto if available
self.temporal_smoothing_alpha = 0.1  # Smoothness penalty weight
```

## Feature Pre-Selection Before ML Training

See `FEATURE_PRE_SELECTION_EXPLANATION.md` for detailed explanation.

### Summary:

1. **Feature Generation**: Generate comprehensive features from feature bank
2. **Feature Selection**: Use LightGBM-based `SelectFromModel` to select informative features
3. **Selection Criteria**: Features above median importance threshold
4. **Fallback**: Variance-based selection if model-based fails
5. **Result**: Reduced feature set (typically 75-90% of original) for faster, more stable training

**Key Points**:
- Happens **before** model training
- Reduces dimensionality to prevent overfitting
- Uses model-based importance (better than correlation-based)
- Retains most informative features while removing noise

## Integration Flow

```
1. Generate Features (feature bank)
   ↓
2. Add Smoothed Features (if enabled)
   ↓
3. Feature Pre-Selection (LightGBM SelectFromModel)
   ↓
4. Train Models with HPO
   ├─ Transition-Aware Scorer (accuracy + stability)
   └─ Optional: Pareto Multi-Objective (future)
   ↓
5. Evaluate with Comprehensive Metrics
   ├─ Classification Metrics
   ├─ Temporal/Stability Metrics
   └─ Regime-Persistence Metrics
```

## Next Steps

1. **Complete Pareto Integration**: Integrate Optuna multi-objective study with Pareto front computation
2. **Knee Point Selection**: Automatically select knee point from Pareto front
3. **Visualization**: Add Pareto front visualization for analysis
4. **A/B Testing**: Compare single-objective vs. Pareto optimization results
