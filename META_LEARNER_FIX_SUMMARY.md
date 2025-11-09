# Meta-Learner R² = 0.0 - Fix Summary

**Date:** 2025-11-09  
**Status:** ✅ FIXED

---

## 🐛 Bug Summary

**Problem:** Meta-learner showing R² = 0.0, making it impossible to assess actual performance

**Root Cause:** Incorrect assumption that ANALYST role = classification task

**Reality:** Both ANALYST and TACTICIAN are **regression tasks** (predicting continuous targets)

---

## ✅ Fix Applied

**File:** `src/training/steps/models_training/core/ensemble_trainer.py`  
**Lines:** 431-449

### Before (WRONG)
```python
if self.config.role == TrainingRole.ANALYST:
    # Classification metrics (WRONG!)
    binary_predictions = (meta_predictions > 0.5).astype(int)
    metrics = {
        'meta_accuracy': accuracy_score(targets, binary_predictions),
        'meta_precision': precision_score(targets, binary_predictions),
        'meta_recall': recall_score(targets, binary_predictions),
        'meta_f1_score': f1_score(targets, binary_predictions)
    }
else:
    # Regression metrics
    metrics = {
        'meta_mse': mean_squared_error(targets, meta_predictions),
        'meta_mae': mean_absolute_error(targets, meta_predictions),
        'meta_r2': r2_score(targets, meta_predictions),
        'meta_rmse': np.sqrt(mean_squared_error(targets, meta_predictions))
    }
```

### After (CORRECT)
```python
# Both ANALYST and TACTICIAN use regression metrics (both predict continuous targets)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
metrics = {
    'meta_mse': mean_squared_error(targets, meta_predictions),
    'meta_mae': mean_absolute_error(targets, meta_predictions),
    'meta_r2': r2_score(targets, meta_predictions),
    'meta_rmse': np.sqrt(mean_squared_error(targets, meta_predictions))
}

# Also calculate comparison to simple average of base models
if oof_predictions.shape[1] > 1:
    base_avg_predictions = oof_predictions.mean(axis=1)
    metrics['base_avg_r2'] = r2_score(targets, base_avg_predictions)
    metrics['base_avg_mse'] = mean_squared_error(targets, base_avg_predictions)
    metrics['improvement_over_avg'] = metrics['meta_r2'] - metrics['base_avg_r2']
    tprint_info(f"📊 Meta-learner R²: {metrics['meta_r2']:.4f} vs Base Average R²: {metrics['base_avg_r2']:.4f} (Δ: {metrics['improvement_over_avg']:+.4f})")
```

---

## 🎯 What Changed

### Removed
- ❌ Classification metrics (accuracy, precision, recall, F1)
- ❌ Binary prediction conversion (threshold 0.5)
- ❌ Role-based metric selection

### Added
- ✅ Regression metrics for both roles (MSE, MAE, R², RMSE)
- ✅ Comparison to simple base model average
- ✅ Improvement delta calculation
- ✅ Informative logging of performance comparison

---

## 📊 Expected Results

### Before Fix
```json
{
  "meta_r2": 0.0,           // Not calculated!
  "meta_accuracy": 0.88,    // Meaningless (binary vs continuous)
  "meta_precision": 0.74,   // Meaningless
  "meta_recall": 0.80,      // Meaningless
  "meta_f1_score": 0.76     // Meaningless
}
```

### After Fix
```json
{
  "meta_r2": 0.XXX,              // Actual R² score
  "meta_mse": 0.XXX,             // Actual MSE
  "meta_mae": 0.XXX,             // Actual MAE
  "meta_rmse": 0.XXX,            // Actual RMSE
  "base_avg_r2": 0.XXX,          // Simple average performance
  "base_avg_mse": 0.XXX,         // Simple average MSE
  "improvement_over_avg": ±0.XXX // How much better/worse than average
}
```

---

## 🔍 Why This Matters

### Can Now Answer
1. **Is the meta-learner actually learning?**
   - R² > 0 means yes, R² ≤ 0 means no

2. **Is it better than the best base model?**
   - Compare meta_r2 to individual base model R² scores

3. **Is it better than a simple average?**
   - Check `improvement_over_avg` (positive = better, negative = worse)

4. **Should we use the ensemble or just the best base model?**
   - If meta_r2 < best_base_r2, use the base model
   - If meta_r2 > best_base_r2, use the ensemble

---

## 📝 Next Steps

1. **Re-run ensemble training** to get actual metrics
2. **Compare performance:**
   - Meta-learner R² vs LightGBM R² (0.683)
   - Meta-learner R² vs CatBoost R² (0.316)
   - Meta-learner R² vs simple average
3. **Make decision:**
   - If ensemble is better → use it
   - If base model is better → use that instead

---

## 🎓 Key Insights

1. **ANALYST targets are continuous** (volume-normalized float32)
2. **TACTICIAN targets are continuous** (price predictions)
3. **Both roles use regression**, not classification
4. **Role ≠ Task type** (common misconception)
5. **Always validate metrics match data types**

---

## 🚀 Test Command

```bash
python3 src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --execution-mode light
```

Look for the new log line:
```
📊 Meta-learner R²: 0.XXXX vs Base Average R²: 0.XXXX (Δ: ±0.XXXX)
```

This will tell us immediately if the ensemble is learning and how it compares to a simple average.

---

**Status:** Ready to test ✅
