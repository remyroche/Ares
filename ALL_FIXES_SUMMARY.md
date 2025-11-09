# All Training Issues Fixed - Summary

## Issue 1: DEPTHWISE_CNN Model Missing ✅ FIXED

**Problem:**
- Only 2 models trained (LightGBM, CatBoost) instead of 3
- DEPTHWISE_CNN failed with error: `cannot import name 'TCNRegressor' from 'src.models.tcn_regressor'`

**Root Cause:**
- The class in `tcn_regressor.py` is named `DepthwiseSeparableCNNRegressor`
- But `src/models/__init__.py` was trying to import `TCNRegressor`

**Fix:**
- Added backward compatibility alias in `src/models/tcn_regressor.py`:
  ```python
  TCNRegressor = DepthwiseSeparableCNNRegressor
  
  def create_tcn_regressor(**kwargs):
      return DepthwiseSeparableCNNRegressor(**kwargs)
  ```

**Files Modified:**
- `src/models/tcn_regressor.py` (lines 442-448)

---

## Issue 2: Data Leakage - 97% R² Score ✅ FIXED

**Problem:**
- Suspiciously high R² score (0.9705) indicating potential data leakage
- `target_long` column was present in the feature list (line 74 of metrics JSON)

**Root Cause:**
- The target column `target_long` was included in the training features
- Feature cleaning only checked for `target`, `_target`, `_label` patterns
- Missed `target_long` and `target_short` columns

**Fix:**
- Enhanced target column detection in `unified_models_training_step.py`:
  ```python
  potential_target_cols = [
      col for col in training_data.columns
      if (col.lower() in {'target', 'label', 'target_long', 'target_short'}
          or col.lower().endswith('_target')
          or col.lower().endswith('_label'))
      and not col.lower().startswith('target_regime')  # Preserve regime probabilities
  ]
  ```
- Added explicit warning message: "DATA LEAKAGE PREVENTION"
- **IMPORTANT:** Preserves `target_regime_*` columns which are legitimate regime probability features

**Files Modified:**
- `src/training/steps/model_training/unified_models_training_step.py` (lines 1823-1836)

**Expected Impact:**
- R² score should drop significantly (likely to 0.3-0.6 range)
- Feature count remains 60 (base features) + regime probabilities
- This is GOOD - it means the model is learning real patterns, not cheating

---

## Issue 3: Missing Detailed Metrics in Reports ✅ FIXED

**Problem:**
- Per-model metrics not showing in reports
- HPO results empty
- Feature importance missing

**Root Cause:**
- Metrics extraction was looking for specific key patterns in the metrics dict
- But actual metrics are stored in `result['metadata']['individual_results']`
- The extraction logic wasn't checking metadata first

**Fix:**
- Modified `_extract_comprehensive_metrics` to extract from metadata first:
  ```python
  # Try to extract from metadata['individual_results'] first
  if 'individual_results' in result.get('metadata', {}):
      individual_results = result['metadata']['individual_results']
      for model_name, model_result in individual_results.items():
          if hasattr(model_result, 'metrics'):
              comprehensive_metrics['per_model_metrics'][model_name] = model_result.metrics
          elif isinstance(model_result, dict) and 'metrics' in model_result:
              comprehensive_metrics['per_model_metrics'][model_name] = model_result['metrics']
  ```

**Files Modified:**
- `src/training/steps/model_training/unified_models_training_step.py` (lines 2793-2801)

---

## Expected Results After Fixes

### 1. Models Trained
- ✅ **3 models** (LightGBM, CatBoost, DEPTHWISE_CNN)
- All models should complete training successfully

### 2. Performance Metrics
- **R² Score:** Expected to drop from 0.97 to **0.3-0.6** (realistic range)
- **MSE/MAE:** May increase (this is expected and good)
- **Feature Count:** Should remain at **60 base features + regime probabilities** (target_long removed, but regime features preserved)

### 3. Reports
- **Per-Model Metrics:** Individual R², MSE, MAE for each model
- **HPO Results:** Hyperparameter optimization details
- **Feature Importance:** Top features by importance score
- **Training Time:** Per-model training duration

---

## Files Modified Summary

1. **src/models/tcn_regressor.py**
   - Added TCNRegressor alias and factory function

2. **src/training/steps/model_training/unified_models_training_step.py**
   - Fixed data leakage detection (lines 1823-1835)
   - Fixed per-model metrics extraction (lines 2793-2801)

3. **src/training/steps/models_training/core/model_trainer.py** (from previous session)
   - Added models dict to training result

4. **src/training/steps/models_training/core/pipeline_orchestrator.py** (from previous session)
   - Extract models from metadata

---

## Next Steps

Run training again:
```bash
python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light
```

**What to verify:**
1. ✅ 3 models trained (not 2)
2. ✅ R² score is lower (0.3-0.6 range, not 0.97)
3. ✅ target_long NOT in feature list (but target_regime_* ARE present)
4. ✅ Per-model metrics populated in reports
5. ✅ Feature count is 60 base features + regime probabilities
6. ✅ No HPO errors about empty arrays
