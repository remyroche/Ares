# Ensemble Training Fix - Current Status

**Date:** 2025-11-09  
**Status:** 🟡 PARTIALLY FIXED - New Issue Discovered

---

## ✅ What We Fixed

### 1. Critical Bug: Re-training on Tiny Subsets
**Problem:** Ensemble trainer was ignoring pre-trained base models and re-training from scratch on 20-sample subsets.

**Fix Applied:**
- Modified `ensemble_trainer.py` to accept `base_predictions` parameter
- Added fast-fail if base predictions not provided
- Removed `_train_individual_models()` and `_generate_oof_predictions()` calls
- Pass base predictions directly to meta-learner

**Result:** ✅ No longer re-training models

### 2. Metrics Calculation Bug
**Problem:** Using classification metrics for ANALYST role instead of regression.

**Fix Applied:**
- Changed to use regression metrics for both ANALYST and TACTICIAN
- Added comparison to base model average
- Updated diversity metrics to work with DataFrame

**Result:** ✅ Correct metrics being calculated

### 3. Model Saving Warning
**Problem:** "Model path not found" warning.

**Fix Applied:**
- Added model saving logic in `unified_models_training_step.py`

**Result:** ⚠️ Still shows warning (minor issue)

---

## 🔴 New Issue Discovered

### Base Predictions Only Have 1 Column

**Evidence from logs:**
```
[2025-11-09 15:23:44.130] INFO: 🔍 Pre-HPO fold 1: X_train shape=(20, 1), X_val shape=(5, 1), features=1
```

**Expected:** `shape=(25, 2)` - 25 samples, 2 base models (LightGBM + CatBoost)  
**Actual:** `shape=(25, 1)` - 25 samples, only 1 model

**Impact:**
- Meta-learner only receives predictions from 1 base model
- Cannot learn to combine multiple models
- R² still 0.0 because there's nothing to combine!

---

## 🔍 Root Cause Analysis

### Why Only 1 Column?

The base predictions artifact (`analyst_base_predictions`) likely contains:
1. **Option A:** Only predictions from the last trained model (CatBoost overwrites LightGBM)
2. **Option B:** Predictions are saved separately per model, not combined
3. **Option C:** The DataFrame structure is wrong (single column instead of multi-column)

### Where to Investigate

**File:** `src/training/steps/model_training/unified_models_training_step.py`  
**Lines:** ~2503-2525 (where analyst_base_predictions are saved)

```python
# Save predictions for ensemble training (analyst_base only)
if training_type == 'analyst_base' and 'predictions' in result and result['predictions'] is not None:
    try:
        predictions_path = self._save_artifact(
            data=result['predictions'],
            artifact_name='analyst_base_predictions',
            artifact_type='data',
            data_category='predictions'
        )
```

**Question:** What is `result['predictions']`?
- Is it a DataFrame with columns for each model?
- Or is it overwritten for each model in the loop?

---

## 📊 Current Performance

| Metric | Value | Status |
|--------|-------|--------|
| **meta_r2** | 0.0 | ❌ Still broken |
| **meta_mse** | 0.106 | Poor |
| **meta_mae** | 0.211 | Poor |
| **meta_rmse** | 0.325 | Poor |

**Why R² = 0.0:**
- With only 1 input feature, meta-learner has nothing to learn
- It's essentially just passing through the single prediction
- No combination/weighting to optimize

---

## 🎯 Next Steps

### Immediate Action Required

1. **Investigate base predictions saving:**
   - Check how `analyst_base_predictions` are being saved
   - Verify it contains predictions from ALL base models
   - Should be DataFrame with columns: `['lightgbm', 'catboost']`

2. **Fix the saving logic:**
   - Accumulate predictions from all base models
   - Save as single DataFrame with multiple columns
   - Each column = one base model's predictions

3. **Expected structure:**
   ```python
   # analyst_base_predictions should be:
   pd.DataFrame({
       'lightgbm': [0.234, 0.567, ...],  # 25 predictions
       'catboost': [0.198, 0.543, ...]   # 25 predictions
   })
   # Shape: (25, 2)
   ```

### Testing

After fix, we should see:
```
✅ Using base model predictions: (25, 2)
📊 Using 2 base model predictions for meta-learning
📊 Meta-learner R²: 0.5XXX vs Base Average R²: 0.XXXX
```

---

## 📝 Files Modified So Far

1. **`ensemble_trainer.py`**
   - Added `base_predictions` parameter
   - Removed individual model training
   - Fixed metrics calculation
   - Added diversity metrics from DataFrame

2. **`pipeline_orchestrator.py`**
   - Pass base_predictions directly to ensemble trainer
   - Removed data enhancement logic

3. **`unified_models_training_step.py`**
   - Added model saving for ensemble
   - ⚠️ Need to fix base predictions saving

---

## 🎓 Key Learnings

1. **Fix one bug, find another**
   - The re-training bug was hiding the predictions bug
   - Now that we're using base predictions, we can see they're wrong

2. **Always validate intermediate data**
   - Should have checked base_predictions shape earlier
   - Logs showed `(20, 1)` but we missed it

3. **Multi-model ensembles need all models**
   - Can't combine 1 model (nothing to combine!)
   - Need at least 2 models for meaningful ensemble

---

## 🚀 Expected Final Result

Once base predictions are fixed:

| Model | Current R² | Expected R² |
|-------|-----------|-------------|
| LightGBM (base) | 0.545 | 0.545 |
| CatBoost (base) | 0.474 | 0.474 |
| **Meta-learner** | **0.0** ❌ | **0.50-0.55** ✅ |

**Improvement:** From R² = 0.0 to R² ≈ 0.52 (52% variance explained)

---

**Status:** Critical fix applied, but new issue discovered  
**Priority:** 🔴 HIGH - Fix base predictions structure  
**Next:** Investigate and fix `analyst_base_predictions` saving logic
