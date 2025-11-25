## Advanced Feature Selection for XGBoost Models

This guide documents the advanced feature selection system for XGBoost specialist models, including monotonic constraints, zero gain pruning, and null importance testing.

## Overview

The adaptive feature selection system optimizes XGBoost models through:

1. **Monotonic Constraints**: Based on Spearman correlation with target
2. **Zero Gain Pruning**: Remove features with 0 importance after each retraining
3. **Null Importance Test**: Validate feature significance via target shuffling
4. **Monthly Full Selection**: Comprehensive feature selection every ~6 retrainings
5. **Quick Updates**: Apply only zero gain pruning between full selections

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Adaptive Feature Selection Cycle                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
    ┌───▼────┐                            ┌────▼─────┐
    │ Monthly│                            │  5-Day   │
    │  Full  │                            │  Quick   │
    │Selection│                           │Selection │
    └───┬────┘                            └────┬─────┘
        │                                      │
   ┌────▼─────────────────────┐          ┌────▼─────┐
   │ 1. Monotonic Constraints │          │ Zero Gain│
   │ 2. Null Importance Test  │          │ Pruning  │
   │ 3. Zero Gain Pruning     │          └────┬─────┘
   └────┬─────────────────────┘               │
        │                                      │
        └──────────────┬───────────────────────┘
                       │
               ┌───────▼────────┐
               │   XGBoost      │
               │   Training     │
               │ with optimized │
               │   features &   │
               │  constraints   │
               └────────────────┘
```

## Module: `adaptive_feature_selection.py`

### 1. Monotonic Constraints

#### Purpose
Enforce directional relationships between features and target based on correlation analysis.

#### Thresholds
- **corr > 0.06**: Constraint = +1 (Force increasing)
- **corr < -0.06**: Constraint = -1 (Force decreasing)
- **-0.06 < corr < 0.06**: Constraint = 0 (No constraint)

#### Usage

```python
from src.utils.ml_common.adaptive_feature_selection import MonotonicConstraintCalculator

# Initialize calculator
calc = MonotonicConstraintCalculator(
    positive_threshold=0.06,
    negative_threshold=-0.06
)

# Calculate constraints
constraints = calc.calculate_constraints(X_train, y_train, method='spearman')

# Get tuple for XGBoost
constraint_tuple = calc.get_constraint_tuple(X_train.columns.tolist())

# Use in XGBoost
model = xgb.XGBRegressor(
    monotone_constraints=constraint_tuple,
    **other_params
)
```

#### Benefits
- Improved interpretability
- Prevents counterintuitive relationships
- Reduces overfitting on spurious correlations
- Maintains domain knowledge in model

### 2. Zero Gain Pruning

#### Purpose
Remove features that contribute nothing to model predictions (importance = 0).

#### Timing
Applied after **every retraining** (5-day cycle for XGB specialist models).

#### Usage

```python
from src.utils.ml_common.adaptive_feature_selection import ZeroGainPruner

# Initialize pruner
pruner = ZeroGainPruner()

# After training, identify zero-gain features
zero_gain_features = pruner.identify_zero_gain_features(
    model=trained_xgb_model,
    feature_names=X_train.columns.tolist(),
    importance_type='gain'
)

# Remove from DataFrame
X_pruned = pruner.prune_dataframe(X_train)

# For next retraining, use X_pruned
```

#### Benefits
- Reduces model complexity
- Faster training (fewer features)
- Lower memory usage
- Cleaner feature space

### 3. Null Importance Test

#### Purpose
Validate feature importance significance by comparing against null distribution from shuffled targets.

#### Method
1. Train model with real target → record feature importances
2. Shuffle target randomly (break X-y relationship)
3. Train model with shuffled target → record "null" importances
4. Repeat shuffle N times (default: 10)
5. Compare real importance vs null distribution
6. Keep features where real importance exceeds 95th percentile of null

#### Timing
Applied during **monthly full selection** (~every 6 retrainings).

#### Usage

```python
from src.utils.ml_common.adaptive_feature_selection import NullImportanceTest

# Initialize tester
tester = NullImportanceTest(
    n_shuffles=10,
    significance_threshold=0.95
)

# Run test
model_params = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    # ... other params
}

significant_features, significance_scores = tester.run_test(
    X=X_train,
    y=y_train,
    model_params=model_params
)

# Filter to significant features
X_significant = X_train[significant_features]
```

#### Benefits
- Removes features with spurious importance
- Reduces false positives from correlated noise
- More robust feature selection
- Prevents overfitting

### 4. Integrated AdaptiveFeatureSelector

#### Purpose
Orchestrates full monthly selection and quick retraining updates.

#### Usage

```python
from src.utils.ml_common.adaptive_feature_selection import AdaptiveFeatureSelector
import xgboost as xgb

# Initialize selector
selector = AdaptiveFeatureSelector(
    cache_dir=Path("cache/feature_selection"),
    correlation_threshold=0.06,
    n_null_shuffles=10
)

# Try to load previous state
model_id = f"xgb_{symbol}_{timeframe}"
selector.load_state(model_id)

# Determine if full or quick selection
if selector.should_do_full_selection(retrainings_per_month=6):
    # Full monthly selection
    logger.info("Performing FULL monthly feature selection...")

    X_selected, constraints = selector.full_monthly_selection(
        X=X_train,
        y=y_train,
        model_params=base_xgb_params
    )

    # Get XGBoost params with constraints
    xgb_params = selector.get_xgboost_params(X_selected.columns.tolist())

    # Train model
    model = xgb.XGBRegressor(**{**base_xgb_params, **xgb_params})
    model.fit(X_selected, y_train)

else:
    # Quick retraining (use existing features/constraints)
    logger.info("Performing QUICK retraining selection...")

    # Filter to previously selected features
    X_filtered = X_train[selector.selected_features]

    # Train model with existing constraints
    xgb_params = selector.get_xgboost_params(X_filtered.columns.tolist())
    model = xgb.XGBRegressor(**{**base_xgb_params, **xgb_params})
    model.fit(X_filtered, y_train)

    # Apply zero gain pruning
    X_filtered = selector.quick_retraining_selection(X_filtered, model)

    # Retrain with pruned features
    xgb_params = selector.get_xgboost_params(X_filtered.columns.tolist())
    model = xgb.XGBRegressor(**{**base_xgb_params, **xgb_params})
    model.fit(X_filtered, y_train)

# Save state for next retraining
selector.save_state(model_id)
```

## Integration with Specialist Models

### XGBoost Specialist Models (5-day retraining)

The following specialist models should use adaptive feature selection:

1. `ml_smc_regime_step.py` - SMC regime detection
2. `ml_reversion_regime_step.py` - Mean reversion (student XGB)
3. `ml_liquidity_regime_step.py` - Liquidity regime
4. `ml_breakout_bounce_regime_step.py` - Breakout/bounce patterns

### Integration Pattern

```python
# In specialist model training code

from src.utils.ml_common.adaptive_feature_selection import AdaptiveFeatureSelector
from src.utils.ml_common.training_optimizations import get_default_xgboost_params

# Initialize feature selector
selector = AdaptiveFeatureSelector(
    correlation_threshold=config.get('monotonic_threshold', 0.06),
    n_null_shuffles=config.get('null_shuffles', 10)
)

# Load previous state
model_id = f"xgb_{symbol}_{timeframe}_{regime_type}"
selector.load_state(model_id)

# Get default optimized XGBoost params
base_params = get_default_xgboost_params(
    n_samples=len(X_train),
    n_features=len(X_train.columns)
)

# Check if full or quick selection
if selector.should_do_full_selection():
    # Monthly full selection
    X_selected, _ = selector.full_monthly_selection(X_train, y_train, base_params)
else:
    # Quick retraining with zero gain pruning
    X_selected = X_train[selector.selected_features]

# Get XGBoost params with monotonic constraints
xgb_params = selector.get_xgboost_params(X_selected.columns.tolist())

# Train with HPO
from src.utils.ml_common.optimization.local_search_hpo import AdaptiveGrid, HPOConfig

hpo_grid = AdaptiveGrid(HPOConfig())

def objective(trial_params):
    model = xgb.XGBRegressor(**{**base_params, **xgb_params, **trial_params})
    model.fit(X_selected, y_train)
    return model.score(X_val, y_val)

best_params, _ = hpo_grid.optimize(model_id, objective)

# Train final model
final_model = xgb.XGBRegressor(**{**base_params, **xgb_params, **best_params})
final_model.fit(X_selected, y_train)

# If not full selection, apply zero gain pruning
if not selector.should_do_full_selection():
    X_selected = selector.quick_retraining_selection(X_selected, final_model)
    # Retrain with pruned features
    xgb_params = selector.get_xgboost_params(X_selected.columns.tolist())
    final_model = xgb.XGBRegressor(**{**base_params, **xgb_params, **best_params})
    final_model.fit(X_selected, y_train)

# Save state
selector.save_state(model_id)
```

## Retraining Schedule

### 5-Day Retraining Cycle (XGB Specialist Models)

| Retraining | Day | Action | Features Updated |
|------------|-----|--------|------------------|
| 1 | 0 | Full monthly selection | All (constraints + null test + pruning) |
| 2 | 5 | Quick + zero gain | Remove 0-importance features |
| 3 | 10 | Quick + zero gain | Remove 0-importance features |
| 4 | 15 | Quick + zero gain | Remove 0-importance features |
| 5 | 20 | Quick + zero gain | Remove 0-importance features |
| 6 | 25 | Quick + zero gain | Remove 0-importance features |
| 7 | 30 | **Full monthly selection** | All (constraints + null test + pruning) |

### Monthly Cycle Benefits

- **Full selection** every ~30 days ensures constraints stay current
- **Quick updates** maintain performance without expensive recomputation
- **Zero gain pruning** continuously refines feature set
- **Adaptive to market**: Monotonic constraints update with regime changes

## Performance Impact

### Computational Cost

**Monthly Full Selection:**
- Monotonic constraints: O(n_features) - Fast (correlation calculation)
- Null importance test: O(n_shuffles × training_time) - Expensive (10× training)
- Zero gain pruning: O(1) - Very fast (read model importance)
- **Total**: ~10-15× single training time (once per month)

**Quick Retraining:**
- Zero gain pruning: O(1) - Very fast
- **Total**: Negligible overhead

### Feature Reduction

Expected reductions based on typical trading features:

- **After null importance**: 20-30% feature reduction
- **After zero gain** (cumulative over month): Additional 10-20% reduction
- **Total**: 30-50% fewer features vs. no selection

### Training Speed Impact

With 50% feature reduction:
- **Training time**: ~2x faster
- **Memory usage**: ~40-50% lower
- **Prediction time**: ~1.5x faster

## Best Practices

### 1. Correlation Thresholds

**Default (0.06)** works well for most cases. Adjust based on:
- **Higher threshold (0.08-0.10)**: More strict, fewer constraints
- **Lower threshold (0.04-0.05)**: More lenient, more constraints

**When to adjust:**
- High noise: Increase threshold (more strict)
- Strong regime structure: Decrease threshold (capture weaker signals)

### 2. Null Shuffles

**Default (10)** provides good balance. Adjust based on:
- **More shuffles (15-20)**: More robust, but slower
- **Fewer shuffles (5-7)**: Faster, less robust

**When to adjust:**
- Large dataset: Can use fewer shuffles
- Small dataset: Use more shuffles for reliability

### 3. Monitoring

**Track these metrics:**
- Number of features selected
- Number of features pruned (zero gain)
- Constraint distribution (increasing/decreasing/none)
- Feature importance stability across retrainings

**Warning signs:**
- Rapid feature reduction: May indicate data quality issues
- All features constrained: Threshold too low
- No features constrained: Threshold too high
- High importance instability: Consider longer retraining intervals

## Troubleshooting

### Issue: Null importance test too slow

**Solution:**
- Reduce n_shuffles (from 10 to 5-7)
- Use smaller validation set for test
- Run test less frequently (skip some monthly cycles)

### Issue: Too many features pruned

**Solution:**
- Check if features are truly uninformative
- Verify target quality (may have issues)
- Consider using 'weight' or 'cover' importance instead of 'gain'

### Issue: Monotonic constraints too restrictive

**Solution:**
- Increase correlation threshold (0.06 → 0.08)
- Review features flagged as constrained
- Consider feature engineering to reduce multicollinearity

### Issue: Features keep changing

**Solution:**
- Increase retraining interval (5 days → 7 days)
- Use rolling correlation for smoother constraint updates
- Implement feature importance smoothing across retrainings

## Files

- **Implementation**: `/home/user/Ares/src/utils/ml_common/adaptive_feature_selection.py`
- **Optimization**: `/home/user/Ares/src/utils/ml_common/training_optimizations.py`
- **HPO Integration**: `/home/user/Ares/src/utils/ml_common/optimization/local_search_hpo.py`

## Examples

See integration examples in:
- `docs/ML_TRAINING_OPTIMIZATIONS_GUIDE.md`
- Specialist model files (after integration)

## Related

- [ML Training Optimizations Guide](ML_TRAINING_OPTIMIZATIONS_GUIDE.md)
- [Retraining Scheduler](../src/utils/ml_common/retraining_scheduler.py)
- [Adaptive HPO](../src/utils/ml_common/optimization/local_search_hpo.py)
