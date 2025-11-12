# Training Validation Fixes - Implementation Plan

**Date**: 2025-11-11  
**Priority**: CRITICAL

---

## 🎯 Issues to Fix

### 1. ⚠️ **Force TimeSeriesSplit for All Time Series Data**
**Current Issue**: HPO uses regular `KFold` unless objective function name contains "temporal"
**Location**: `src/utils/ml_common/optimization/hierarchical_parameter_optimizer.py:1403-1409`
**Fix**: Always use `TimeSeriesSplit` for financial time series data

### 2. ❌ **Add Test Set Evaluation to Reports**
**Current Issue**: Reports only show CV metrics, no held-out test set evaluation
**Location**: Multiple files in `src/training/steps/models_training/`
**Fix**: Add explicit train/val/test split evaluation and reporting

### 3. 🤖 **Fix DepthwiseCNN for Tabular Data**
**Current Issue**: CNN treats features as time sequence (R² ≈ 0)
**Location**: `src/models/tcn_regressor.py`
**Fix**: Either disable CNN or adapt architecture for tabular data

---

## 📝 Implementation Steps

### Step 1: Force TimeSeriesSplit in HPO ✅

**File**: `src/utils/ml_common/optimization/hierarchical_parameter_optimizer.py`

**Change lines 1401-1409**:

```python
# BEFORE (lines 1401-1409):
cv = None
if hasattr(self.objective_func, '__name__') and 'temporal' in self.objective_func.__name__.lower():
    cv = TimeSeriesSplit(n_splits=self.cv_folds)
else:
    if is_classification:
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=False)
    else:
        cv = KFold(n_splits=self.cv_folds, shuffle=False)

# AFTER:
# CRITICAL FIX: Always use TimeSeriesSplit for financial time series data
# Regular KFold can cause data leakage by using future data to predict past
cv = TimeSeriesSplit(n_splits=self.cv_folds)
logger.info(f"🕐 Using TimeSeriesSplit for temporal data (n_splits={self.cv_folds})")
```

**Rationale**: Financial time series ALWAYS requires temporal ordering. Using regular KFold causes look-ahead bias.

---

### Step 2: Add Test Set Evaluation ✅

**Files to Modify**:
1. `src/training/steps/models_training/core/model_trainer.py`
2. `src/training/steps/model_training/unified_models_training_step.py`

**Changes Needed**:

#### A. Split data into train/val/test BEFORE HPO

```python
# In unified_models_training_step.py, after loading data:

# Create temporal splits (70/15/15)
from sklearn.model_selection import TimeSeriesSplit

n_samples = len(training_data)
train_end = int(n_samples * 0.70)
val_end = int(n_samples * 0.85)

X_train = training_data.iloc[:train_end]
y_train = analyst_targets.iloc[:train_end]

X_val = training_data.iloc[train_end:val_end]
y_val = analyst_targets.iloc[train_end:val_end]

X_test = training_data.iloc[val_end:]
y_test = analyst_targets.iloc[val_end:]

logger.info(f"📊 Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
```

#### B. Evaluate on all three sets after training

```python
# After model training, evaluate on all sets:

def evaluate_on_all_splits(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on train, validation, and test sets."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
    
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        
        # Determine if classification or regression
        unique_vals = len(np.unique(y))
        is_classification = unique_vals <= 10
        
        if is_classification:
            results[f'{split_name}_accuracy'] = accuracy_score(y, y_pred)
            results[f'{split_name}_precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            results[f'{split_name}_recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            results[f'{split_name}_f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Always compute regression metrics
        results[f'{split_name}_r2'] = r2_score(y, y_pred)
        results[f'{split_name}_mse'] = mean_squared_error(y, y_pred)
        results[f'{split_name}_rmse'] = np.sqrt(mean_squared_error(y, y_pred))
        results[f'{split_name}_mae'] = mean_absolute_error(y, y_pred)
    
    # Calculate overfitting metrics
    results['train_val_gap'] = results['train_r2'] - results['val_r2']
    results['train_test_gap'] = results['train_r2'] - results['test_r2']
    results['overfitting_ratio'] = results['train_test_gap'] / max(results['train_r2'], 0.01)
    
    return results
```

#### C. Add to report generation

```python
# In report generation code, add sections:

## Train/Val/Test Performance Comparison

| Metric | Train | Validation | Test | Train-Test Gap |
|--------|-------|------------|------|----------------|
| Accuracy | {train_acc:.4f} | {val_acc:.4f} | {test_acc:.4f} | {gap:.4f} |
| R² | {train_r2:.4f} | {val_r2:.4f} | {test_r2:.4f} | {gap:.4f} |
| RMSE | {train_rmse:.4f} | {val_rmse:.4f} | {test_rmse:.4f} | - |

### Overfitting Analysis

- **Train-Test R² Gap**: {gap:.4f} ({gap_pct:.1f}%)
- **Overfitting Ratio**: {ratio:.4f}
- **Status**: {'✅ Good' if ratio < 0.1 else '⚠️ Moderate' if ratio < 0.2 else '❌ High'}
```

---

### Step 3: Fix or Disable DepthwiseCNN ✅

**Option A: Disable CNN** (Quick fix)

In `src/training/steps/model_training/analyst_base_config.yaml`:
```yaml
models:
  lightgbm:
    enabled: true
  catboost:
    enabled: true
  depthwise_cnn:
    enabled: false  # Disabled - not suitable for tabular data
```

**Option B: Adapt CNN Architecture** (Better fix)

Modify `src/models/tcn_regressor.py` to handle tabular data properly:

```python
def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
    # ... existing code ...
    
    # For tabular data, create artificial time dimension
    # Treat each sample as a single timestep with multiple features
    if len(X.shape) == 2:
        # Reshape: (samples, features) -> (samples, 1, features)
        # This treats features as channels, not time steps
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    else:
        X_reshaped = X
    
    # ... rest of code ...
```

And modify the architecture to use 1D convolutions across features:

```python
def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential()
    
    # Use Conv1D across features (not time)
    model.add(Conv1D(
        filters=self.filters,
        kernel_size=min(3, input_shape[1]),  # Limit kernel to feature count
        activation='relu',
        input_shape=input_shape,
        padding='valid'  # No padding for feature dimension
    ))
    # ... rest of architecture ...
```

**Recommendation**: Use Option A (disable) for now, implement Option B later if needed.

---

## 🧪 Testing Plan

### Test 1: Verify TimeSeriesSplit Usage
```bash
# Run training and check logs for:
grep "Using TimeSeriesSplit" logs/unified_*.log
```

### Test 2: Verify Test Set Evaluation
```bash
# Check that reports include test metrics:
grep -A 10 "Test Set Metrics" outcomes/analyst_base_*.md
```

### Test 3: Verify No Overfitting
```bash
# Check train/test gap is reasonable (<20%):
grep "train_test_gap\|overfitting_ratio" outcomes/analyst_base_*.json
```

---

## 📊 Expected Outcomes

### Before Fixes:
- ✅ Accuracy: 96% (CV only, suspicious)
- ❌ Test accuracy: Unknown
- ❌ CV method: KFold (data leakage risk)
- ❌ CNN R²: 0.0006 (failed)

### After Fixes:
- ✅ Train accuracy: ~85-95%
- ✅ Val accuracy: ~80-90%
- ✅ Test accuracy: ~75-85% (realistic)
- ✅ CV method: TimeSeriesSplit (no leakage)
- ✅ CNN: Disabled or fixed
- ✅ Overfitting ratio: <0.2

---

## 🚀 Implementation Order

1. **CRITICAL**: Fix TimeSeriesSplit (5 min)
2. **CRITICAL**: Add test set evaluation (30 min)
3. **IMPORTANT**: Disable CNN (2 min)
4. **TEST**: Run full training (90 min)
5. **VERIFY**: Check reports and metrics (10 min)

**Total Time**: ~2.5 hours

---

## ⚠️ Risks & Mitigations

### Risk 1: Test accuracy much lower than CV
- **Cause**: Previous CV was leaking future data
- **Mitigation**: This is EXPECTED and CORRECT
- **Action**: Accept lower but honest metrics

### Risk 2: Training takes longer
- **Cause**: TimeSeriesSplit has smaller validation sets
- **Mitigation**: Adjust cv_folds if needed (3-5 folds)
- **Action**: Monitor training time

### Risk 3: Breaking existing code
- **Cause**: Changing evaluation logic
- **Mitigation**: Add backward compatibility
- **Action**: Test thoroughly before deployment

---

## 📝 Checklist

- [ ] Backup current code
- [ ] Implement TimeSeriesSplit fix
- [ ] Implement test set evaluation
- [ ] Disable CNN model
- [ ] Run test training
- [ ] Verify reports include test metrics
- [ ] Verify no data leakage warnings
- [ ] Document changes
- [ ] Update user documentation

---

**Status**: Ready to implement  
**Priority**: CRITICAL - Affects model validity  
**Estimated Time**: 2.5 hours
