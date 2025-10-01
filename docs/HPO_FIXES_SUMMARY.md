# HPO Optimization Fixes - Summary

## Problem: Identical Scores Across Trials

The hyperparameter optimization was producing identical scores across all trials (e.g., 0.8, 0.6, or 0.0), indicating that the model was not actually learning different patterns with different hyperparameters.

## Root Causes Identified

1. **Severe Class Imbalance**: 80% of samples in one class leads to 0.8 accuracy by always predicting majority class
2. **Inappropriate Scoring Metric**: Using `accuracy` on imbalanced data masks the problem
3. **Inadequate Data Variance Checks**: No validation before starting HPO
4. **Suboptimal Search Spaces**: Too wide ranges (max_depth 5-50) causing inefficient exploration
5. **No Trial Monitoring**: Issues weren't detected in real-time

## Fixes Implemented

### 1. Data Diagnostics (`hpo_diagnostics_and_fixes.py`)

**New Module**: Comprehensive pre-HPO validation

#### Features:
- **Class Distribution Analysis**
  - Detects severe imbalance (>80% in one class)
  - Calculates class percentages and balance scores
  - Warns about potential constant prediction issues

- **Feature Variance Checks**
  - Identifies zero-variance features
  - Detects low-variance features (<1e-6)
  - Checks for NaN and infinite values

- **Scoring Metric Recommendations**
  - Automatically suggests `balanced_accuracy` for imbalanced binary classification
  - Recommends `f1_macro` for multi-class problems
  - Falls back to `f1` for balanced binary classification

- **Real-time HPO Monitoring**
  - Tracks trial scores and detects identical results
  - Warns when all recent scores are the same
  - Detects zero scores (complete training failure)
  - Calculates score variance to ensure exploration

#### Usage:
```python
from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import apply_hpo_fixes

# Automatically diagnose and get improved configuration
search_space, hpo_params = apply_hpo_fixes(X, y, model_type="random_forest")
```

### 2. Improved Scoring Metrics

**Changed**: Default scoring from `accuracy` to `balanced_accuracy`

**Why**:
- `accuracy` on 80/20 split → constant 0.8 score
- `balanced_accuracy` averages recall per class → detects constant predictions
- For imbalanced data: `balanced_accuracy`, `f1_weighted`, or `roc_auc`

**Auto-Detection**:
```python
# Now automatically switches if imbalance detected
if scoring == 'accuracy' and max_class_percentage > 70%:
    scoring = 'balanced_accuracy'  # Auto-switch with warning
```

### 3. Expanded & Improved Search Spaces

#### RandomForest Search Space Changes:

| Parameter | Old Range | New Range | Reason |
|-----------|-----------|-----------|--------|
| `n_estimators` | 50-500 | 100-500 | More trees = better regime detection |
| `max_depth` | 5-50 | 5-15 | 50 was too deep, causing overfitting |
| `max_features` | ['sqrt', 'log2'] | ['sqrt', 'log2', 0.5] | Added float option |
| `class_weight` | N/A | ['balanced', 'balanced_subsample', None] | **NEW** - handles imbalance |

**Key Addition**: `class_weight='balanced'` automatically adjusts for imbalanced classes

### 4. Reduced Trial Counts with Early Stopping

**Changed**: Trial management for efficiency

- **Initial Trials**: Reduced to 10 (was unlimited/15)
- **Early Stopping**: Stop after 5 trials with no improvement
- **Timeout**: 10 minutes per optimization run
- **Pruning**: MedianPruner to kill bad trials early

**Rationale**:
- When all scores are identical, 15+ trials is wasteful
- Early stopping detects convergence issues faster
- Users can iterate and fix problems sooner

### 5. Better Acquisition Function

**Changed**: Default from `ucb` to `ei` in recommendations

**Why**:
- **UCB (Upper Confidence Bound)**: Struggles when scores are identical (no uncertainty)
- **EI (Expected Improvement)**: Better exploration even with flat landscapes
- **POI (Probability of Improvement)**: Alternative for exploitation-focused search

### 6. Improved Cross-Validation Strategy

**Auto-Detection**: Chooses appropriate CV based on data

```python
if n_samples > 1000:
    cv = TimeSeriesSplit(n_splits=5)  # For time-series regime data
else:
    cv = StratifiedKFold(n_splits=3, shuffle=True)  # For small datasets
```

**Why**:
- Regime data is time-series → `TimeSeriesSplit` prevents lookahead bias
- `StratifiedKFold` ensures all classes in each fold
- Removes `random_state` from CV to get real variation across trials

### 7. Enhanced Monitoring & Logging

**Added**: Real-time diagnostic logging

```
⚠️  Trial 5: Model predicting CONSTANT class! Score: 0.8000
   Predictions: [0, 0, 0, 0, 0, 0, ...]  # All same class
   
⚠️  ALL RECENT SCORES IDENTICAL: 0.8000
   This suggests:
   1. Model is predicting constant class (check class imbalance)
   2. Scoring metric may be inappropriate (try balanced_accuracy)
   3. Features may have no signal (check feature variance)
```

## Updated Workflow

### Before (Problems):
```python
# Old way - no diagnostics
hpo = HyperparameterOptimization()
results = hpo.bayesian_optimization(
    model_factory=RandomForestClassifier,
    X=X, y=y,
    search_space=search_space,
    n_trials=15,
    scoring='accuracy'  # ❌ Bad for imbalanced data
)
# Result: All trials get 0.8 (predicting majority class)
```

### After (Fixed):
```python
# New way - automatic diagnostics & fixes
hpo = HyperparameterOptimization()
results = hpo.bayesian_optimization(
    model_factory=RandomForestClassifier,
    X=X, y=y,
    search_space=search_space,
    n_trials=10,  # Reduced
    scoring='accuracy',  # ✅ Auto-switches to balanced_accuracy if needed
    enable_diagnostics=True  # ✅ NEW - validates data first
)
# Now: Diagnostic warnings appear, scoring auto-fixed, real variance in scores
```

## Example Output with Fixes

### Diagnostics Output:
```
================================================================================
📊 HPO DIAGNOSTICS: Training Data
================================================================================

📈 Dataset Stats:
  • Samples: 1000
  • Features: 50
  • Classes: 3

🎯 Class Distribution:
  • Class 0: 800 samples (80.0%)
  • Class 1: 150 samples (15.0%)
  • Class 2: 50 samples (5.0%)

⚠️  WARNINGS (1):
  High class imbalance: 80.0% in majority class. Consider using balanced_accuracy

✅ Data validation PASSED - safe to proceed with HPO
================================================================================

⚠️  Using 'accuracy' with imbalanced data (80.0% majority class)!
   This may cause constant predictions. Recommended: 'balanced_accuracy'
   Automatically switching to 'balanced_accuracy'

🎲 Starting enhanced Bayesian optimization with ucb acquisition
[I 2025-10-01 10:15:00] Trial 0 finished with value: 0.65 and parameters: {...}
[I 2025-10-01 10:15:05] Trial 1 finished with value: 0.72 and parameters: {...}
[I 2025-10-01 10:15:10] Trial 2 finished with value: 0.68 and parameters: {...}
```

## Files Modified

1. **`/src/utils/ml_common/optimization/hpo_utils.py`**
   - Added `enable_diagnostics` parameter to `bayesian_optimization()`
   - Updated `random_forest` search space
   - Added automatic scoring metric switching
   - Integrated real-time monitoring

2. **`/src/utils/ml_common/optimization/hpo_diagnostics_and_fixes.py`** (NEW)
   - `HPODiagnostics` class for data validation
   - `ImprovedHPOConfig` for better hyperparameter ranges
   - `HPOMonitor` for real-time trial monitoring
   - `apply_hpo_fixes()` convenience function

## Migration Guide

### For Existing Code:

**Option 1**: Enable diagnostics (recommended)
```python
# Just add one parameter
results = hpo.bayesian_optimization(
    ...,
    enable_diagnostics=True  # Add this
)
```

**Option 2**: Manual fixes
```python
from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import apply_hpo_fixes

# Get improved configuration
search_space, hpo_params = apply_hpo_fixes(X, y, "random_forest")

# Use improved params
results = hpo.bayesian_optimization(
    model_factory=RandomForestClassifier,
    X=X, y=y,
    search_space=search_space,
    n_trials=hpo_params['n_trials'],
    scoring=hpo_params['scoring'],
    cv=hpo_params['cv_strategy']
)
```

## Testing Recommendations

1. **Run diagnostics on your regime data**:
   ```python
   from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import HPODiagnostics
   
   diagnostics = HPODiagnostics.check_data_variance(X, y, "My Regime Data")
   HPODiagnostics.print_diagnostics(diagnostics)
   ```

2. **Check class distribution**:
   ```python
   import pandas as pd
   print(pd.Series(y).value_counts(normalize=True))
   ```

3. **Verify HPO variance**:
   - Run 5-10 trials
   - Check if scores vary (should see differences > 0.01)
   - If identical → investigate data/features

## Performance Impact

- **Speed**: Faster (fewer trials with early stopping)
- **Quality**: Better (appropriate metrics, better search spaces)
- **Debugging**: Much easier (diagnostic output identifies issues immediately)

## Next Steps if Issues Persist

If you still see identical scores after applying these fixes:

1. **Check feature engineering**: Features may have no signal
   ```python
   # Check feature importances
   rf = RandomForestClassifier().fit(X, y)
   print(rf.feature_importances_)
   ```

2. **Verify regime labels**: Labels may be incorrect
   ```python
   # Check regime transitions
   transitions = (y[1:] != y[:-1]).sum()
   print(f"Regime transitions: {transitions}/{len(y)}")
   ```

3. **Try simpler model**: Test with LogisticRegression to validate pipeline
   ```python
   from sklearn.linear_model import LogisticRegression
   lr = LogisticRegression(class_weight='balanced')
   scores = cross_val_score(lr, X, y, cv=5, scoring='balanced_accuracy')
   print(f"LR scores: {scores}")  # Should vary
   ```

## References

- Sklearn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Imbalanced Learning: https://imbalanced-learn.org/
- Optuna Best Practices: https://optuna.readthedocs.io/en/stable/tutorial/index.html

