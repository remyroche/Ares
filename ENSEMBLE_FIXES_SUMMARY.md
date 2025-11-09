# Ensemble Training Fixes - Complete

## 🔧 Issues Fixed

### 1. ✅ Post-HPO Metrics Collection Error
**Error:** `k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=1`

**Fix:** Modified `training_metrics_collector.py` line 312-315
```python
# Ensure n_folds is at least 2 for cross-validation
if n_folds < 2:
    self.logger.warning(f"⚠️ n_folds={n_folds} is too small, using n_folds=2")
    n_folds = 2
```

**File:** `src/training/steps/models_training/core/training_metrics_collector.py`

---

### 2. ✅ Diversity Metrics Calculation Error
**Error:** `unsupported operand type(s) for /: 'NoneType' and 'int'`

**Fix:** Modified `ensemble_trainer.py` line 526
```python
# Get predictions from all models
all_predictions = []
for result in individual_results.values():
    if result.success and hasattr(result, 'predictions') and result.predictions is not None:
        all_predictions.append(result.predictions)
```

**File:** `src/training/steps/models_training/core/ensemble_trainer.py`

---

### 3. ✅ Artifact Manager Import Error
**Error:** `No module named 'src.utils.ml_common.artifact_manager'`

**Fix:** 
1. Removed incorrect import from `pipeline_orchestrator.py` (line 638-655)
2. Added proper artifact saving in `unified_models_training_step.py` (line 2527-2539)

```python
# Save predictions for tactician training (analyst_ensemble only)
if training_type == 'analyst_ensemble' and 'predictions' in result and result['predictions'] is not None:
    try:
        predictions_path = self._save_artifact(
            data=result['predictions'],
            artifact_name='analyst_ensemble_outputs',
            artifact_type='data',
            data_category='predictions'
        )
        artifacts['analyst_ensemble_outputs'] = predictions_path
        tprint_success(f"✅ Saved analyst_ensemble_outputs: {result['predictions'].shape}")
    except Exception as e:
        tprint_warning(f"⚠️ Failed to save analyst_ensemble_outputs: {e}")
```

**Files:**
- `src/training/steps/models_training/core/pipeline_orchestrator.py`
- `src/training/steps/model_training/unified_models_training_step.py`

---

## 📋 Changes Summary

### Modified Files
1. **training_metrics_collector.py**
   - Added n_folds validation (minimum 2)
   - Prevents KFold error

2. **ensemble_trainer.py**
   - Added None check for predictions
   - Prevents division by None error

3. **pipeline_orchestrator.py**
   - Removed incorrect artifact saving code
   - Delegated to unified_models_training_step

4. **unified_models_training_step.py**
   - Added analyst_ensemble_outputs saving
   - Uses base class _save_artifact method
   - Saves to versioned HDF5 storage

---

## ✅ Verification

### Expected Behavior
1. **Post-HPO metrics** - Should collect with n_folds=2 minimum
2. **Diversity metrics** - Should skip if predictions are None
3. **Artifact saving** - Should save analyst_ensemble_outputs to HDF5

### Test Command
```bash
python3 src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --execution-mode light
```

### Expected Output
```
✅ Saved analyst_ensemble_outputs: (25, 1)
✅ Post-HPO metrics collected
✅ Diversity metrics calculated (or skipped if no predictions)
```

---

## 🎯 Status

- ✅ **Fix 1:** n_folds validation - Complete
- ✅ **Fix 2:** Predictions None check - Complete
- ✅ **Fix 3:** Artifact saving - Complete
- ⏳ **Testing:** Ready to run ensemble training

All fixes use proper base class methods and follow existing patterns in the codebase.
