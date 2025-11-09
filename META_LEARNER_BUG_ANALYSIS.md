# Meta-Learner R² = 0.0 Bug Analysis

**Date:** 2025-11-09  
**Issue:** Meta-learner showing R² = 0.0 despite base models performing well

---

## 🐛 Root Cause Identified

### The Bug

**File:** `src/training/steps/models_training/core/ensemble_trainer.py`  
**Lines:** 434-450

```python
if self.config.role == TrainingRole.ANALYST:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    binary_predictions = (meta_predictions > 0.5).astype(int)
    metrics = {
        'meta_accuracy': accuracy_score(targets, binary_predictions),
        'meta_precision': precision_score(targets, binary_predictions),
        'meta_recall': recall_score(targets, binary_predictions),
        'meta_f1_score': f1_score(targets, binary_predictions)
    }
else:  # TACTICIAN role
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics = {
        'meta_mse': mean_squared_error(targets, meta_predictions),
        'meta_mae': mean_absolute_error(targets, meta_predictions),
        'meta_r2': r2_score(targets, meta_predictions),
        'meta_rmse': np.sqrt(mean_squared_error(targets, meta_predictions))
    }
```

### The Problem

1. **ANALYST role uses CLASSIFICATION metrics** (accuracy, precision, recall, F1)
2. **TACTICIAN role uses REGRESSION metrics** (MSE, MAE, R², RMSE)
3. **R² is NEVER calculated for ANALYST role!**

This is why we see:
- `meta_r2: 0.0` in the metrics (default value, never actually calculated)
- The ensemble appears to not be learning

### Why This Is Wrong

**Analyst targets are CONTINUOUS, not binary!**

From `feature_generation_labeling_integration_step.py`:
```python
# Lines 1683-1684
target_df['target_long'] = labels_aligned['target_long'].fillna(0.0).astype(np.float32)
target_df['target_short'] = labels_aligned['target_short'].fillna(0.0).astype(np.float32)
```

The targets are:
- **Type:** `np.float32` (continuous values)
- **Range:** Volume-normalized continuous values (not 0/1 binary)
- **Purpose:** Predict the magnitude of price movement, not just direction

### The Incorrect Assumption

The code assumes:
- ANALYST = Classification task (binary: long/short)
- TACTICIAN = Regression task (continuous values)

**Reality:**
- ANALYST = **REGRESSION** task (predicting continuous volume-normalized targets)
- TACTICIAN = **REGRESSION** task (predicting continuous values)

Both are regression tasks!

---

## 📊 Impact Analysis

### What's Actually Happening

1. **Meta-learner trains correctly** as a regressor (LGBMRegressor)
2. **Predictions are continuous** (e.g., 0.234, 0.567, etc.)
3. **Metrics calculation is wrong:**
   - Converts continuous predictions to binary (0 or 1) with threshold 0.5
   - Compares binary predictions to continuous targets
   - Calculates classification metrics (meaningless for continuous targets)
   - **Never calculates R², MSE, MAE, RMSE**

4. **Result:** We have no idea how well the meta-learner is actually performing!

### Why We See R² = 0.0

The R² value in the report is **not calculated** - it's just a default/placeholder value. The actual regression performance is unknown.

### Why Classification Metrics Are Misleading

```python
# Example:
# True target: 0.234 (continuous)
# Meta prediction: 0.567 (continuous)
# Binary prediction: 1 (because 0.567 > 0.5)
# Binary target: 0 (because 0.234 < 0.5)
# Accuracy: Wrong! (but this comparison is meaningless)
```

The classification metrics are comparing:
- **Binary predictions** (0 or 1)
- **Continuous targets** (0.0 to 1.0+)

This is fundamentally incorrect!

---

## ✅ The Fix

### Change Required

Replace the role-based metric calculation with **regression metrics for both roles**:

```python
# BEFORE (WRONG):
if self.config.role == TrainingRole.ANALYST:
    # Classification metrics (WRONG!)
    binary_predictions = (meta_predictions > 0.5).astype(int)
    metrics = {
        'meta_accuracy': accuracy_score(targets, binary_predictions),
        ...
    }
else:
    # Regression metrics (CORRECT)
    metrics = {
        'meta_mse': mean_squared_error(targets, meta_predictions),
        'meta_r2': r2_score(targets, meta_predictions),
        ...
    }

# AFTER (CORRECT):
# Both ANALYST and TACTICIAN use regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
metrics = {
    'meta_mse': mean_squared_error(targets, meta_predictions),
    'meta_mae': mean_absolute_error(targets, meta_predictions),
    'meta_r2': r2_score(targets, meta_predictions),
    'meta_rmse': np.sqrt(mean_squared_error(targets, meta_predictions))
}
```

### Additional Metrics to Add

For better analysis, we should also calculate:

```python
# Ensemble vs base model comparison
ensemble_predictions = meta_model.predict(oof_predictions)
base_avg_predictions = oof_predictions.mean(axis=1)  # Simple average of base models

metrics.update({
    'ensemble_r2': r2_score(targets, ensemble_predictions),
    'ensemble_mse': mean_squared_error(targets, ensemble_predictions),
    'base_avg_r2': r2_score(targets, base_avg_predictions),
    'base_avg_mse': mean_squared_error(targets, base_avg_predictions),
    'improvement_over_avg': r2_score(targets, ensemble_predictions) - r2_score(targets, base_avg_predictions)
})
```

---

## 🎯 Expected Results After Fix

### Before Fix
- `meta_r2`: 0.0 (not calculated)
- `meta_accuracy`: 0.88 (meaningless - comparing binary to continuous)
- **No way to assess actual performance**

### After Fix
- `meta_r2`: Actual R² score (e.g., 0.65, 0.70, etc.)
- `meta_mse`: Actual mean squared error
- `meta_mae`: Actual mean absolute error
- `meta_rmse`: Actual root mean squared error
- **Can properly compare to base models**

### Performance Comparison

With correct metrics, we can answer:
1. Is the meta-learner better than the best base model?
2. Is the meta-learner better than a simple average?
3. How much improvement does stacking provide?

---

## 🔍 Why This Wasn't Caught Earlier

1. **Silent failure:** The code runs without errors
2. **Metrics look plausible:** 88% accuracy seems good (but it's meaningless)
3. **R² = 0.0 was overlooked:** Assumed it meant "no learning" rather than "not calculated"
4. **Role assumption:** The ANALYST/TACTICIAN split seemed logical but was incorrect

---

## 📝 Recommendations

### Immediate Actions

1. **Fix the metrics calculation** (remove role-based logic)
2. **Re-run ensemble training** with correct metrics
3. **Compare actual performance** to base models

### Long-term Improvements

1. **Add validation:** Check that target types match model types
2. **Add warnings:** Alert if using classification metrics on continuous targets
3. **Document target types:** Clearly specify ANALYST and TACTICIAN target types
4. **Add unit tests:** Test metric calculation for both roles

---

## 🎓 Key Learnings

1. **Both ANALYST and TACTICIAN are regression tasks**
2. **Role ≠ Task type** (ANALYST doesn't mean classification)
3. **Always validate metric calculations** against actual data types
4. **R² = 0.0 can mean "not calculated"** not just "no learning"

---

**Next Step:** Implement the fix and re-run ensemble training to get actual performance metrics.
