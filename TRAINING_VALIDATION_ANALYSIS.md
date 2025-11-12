# Training Validation & Data Leakage Analysis

**Date**: 2025-11-11  
**Training Session**: analyst_base_ETHUSDT_20251111_182448

---

## 🚨 CRITICAL FINDINGS

### 1. ⚠️ METRICS ARE FROM CROSS-VALIDATION, NOT HELD-OUT TEST SET

**Issue**: The reported 96% accuracy is from **5-fold cross-validation during HPO**, NOT from a held-out test set.

**Evidence**:
- Report shows `accuracy_mean: 0.9603` and `accuracy_std: 0.0527`
- The `_std` (standard deviation) indicates this is averaged across CV folds
- No separate "Test Set Metrics" section in the report
- Report explicitly states: `*No test metrics available.*`

**What This Means**:
- ✅ The model performs well during training/validation
- ❌ **We don't know how it performs on truly unseen data**
- ⚠️ **High risk of overfitting** - 96% accuracy without test set validation is suspicious

---

### 2. 🔍 DATA LEAKAGE INVESTIGATION

#### A. Temporal Split Configuration ✅

**Code Review** (`temporal_splits.py` lines 140-165):
```python
train_days = int(total_days * train_pct)  # 70%
val_days = int(total_days * val_pct)      # 15%
test_days = total_days - train_days - val_days - 2 * embargo_days  # 15%

# Embargo periods between splits
val_start = train_end + timedelta(days=embargo_days)  # 1-day gap
test_start = val_end + timedelta(days=embargo_days)   # 1-day gap
```

**Status**: ✅ **Proper temporal separation with embargo periods**
- Train → 1-day embargo → Validation
- Validation → 1-day embargo → Test
- No overlap between periods

#### B. Feature Leakage Check ⚠️

**Potential Issues**:
1. **Regime Probabilities**: 6 regime features included
   - ⚠️ If regime models were trained on full dataset → **LEAKAGE**
   - ✅ If regime models used proper temporal splits → OK

2. **Target Column Removal**: Code shows target filtering
   ```python
   # Line 2066-2071: Removes target columns but preserves regime probabilities
   potential_target_cols = [col for col in training_data.columns
       if (col.lower() in {'target', 'label', ...})
       and not col.lower().startswith('target_regime')]
   ```
   - ✅ Target columns are filtered out
   - ⚠️ But regime probabilities are kept (could be leakage if not properly generated)

3. **Look-Ahead Bias**: 
   - ✅ Embargo periods prevent using future data
   - ✅ Temporal ordering preserved
   - ⚠️ **BUT**: We need to verify regime models didn't use future data

---

### 3. 🤖 DepthwiseCNN Accuracy Missing

**Finding**: DepthwiseCNN shows metrics but no accuracy in the report.

**From Report**:
```
DEPTHWISE_CNN
- Mae: 0.317902
- Mse: 0.164013
- R2: 0.000645  ← Nearly ZERO!
- Rmse: 0.404986
- Training Time: 100.787241
```

**Analysis**:
- ✅ Model trained successfully
- ❌ **R² = 0.0006** means model explains almost NO variance
- ❌ Model is essentially predicting the mean
- ⚠️ Accuracy not reported because **regression task**, not classification
- **Conclusion**: CNN model failed to learn meaningful patterns

---

### 4. ✅ HPO INCLUDES REGULARIZATION

**Confirmed from `hpo_config.py`**:

#### LightGBM Regularization (lines 138-148):
```python
ParameterGroup(
    name="regularization_subsampling",
    params={
        "reg_lambda": {"type": "float", "low": 0.0, "high": 5.0},
        "sampling_rate": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 10, "high": 100}
    },
    priority=2
)
```

**Optimized Parameters**:
- `reg_lambda`: Not in final params (likely 0.0)
- `feature_fraction`: 0.6095 (60% features per tree)
- `bagging_fraction`: 0.7482 (75% samples per tree)
- `min_child_samples`: 41 (prevents overfitting)

#### CatBoost Regularization (lines 174-182):
```python
ParameterGroup(
    name="regularization",
    params={
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
        "sampling_rate": {"type": "float", "low": 0.6, "high": 1.0}
    }
)
```

**Optimized Parameters**:
- `l2_leaf_reg`: 1.67 ✅ (L2 regularization active)
- `depth`: 8 (controlled tree depth)

**Status**: ✅ **HPO includes proper regularization parameters**

---

### 5. ⚠️ TEMPORAL SEPARATION VERIFICATION

#### A. Split Configuration ✅

**From Code**:
- **Train**: 70% of data (9,816 samples)
- **Validation**: 15% of data (2,103 samples)
- **Test**: 15% of data (2,104 samples)
- **Embargo**: 1 day (96 candles @ 15m) between each split

**Calculation**:
```
Total: 14,023 samples (146 days)
Train: 102 days (9,816 samples)
Val: 22 days (2,103 samples)
Test: 22 days (2,104 samples)
```

#### B. Embargo Period Analysis ✅

**1-day embargo = 96 candles @ 15m**:
- Prevents using data from day N to predict day N+1
- Sufficient for 15-minute timeframe
- ✅ **Adequate separation**

#### C. Cross-Validation During HPO ⚠️

**From Report**:
```
Post-HPO Metrics:
- accuracy_mean: 0.9603
- accuracy_std: 0.0527  ← Variation across 5 folds
```

**Issue**: 
- HPO uses 5-fold CV on **training set only** (9,816 samples)
- Each fold: ~7,853 train / ~1,963 validation
- ⚠️ **No guarantee of temporal ordering within CV folds**
- ⚠️ **Could have data leakage if CV doesn't respect time**

---

## 🎯 RECOMMENDATIONS

### CRITICAL - Must Fix:

1. **✅ ADD HELD-OUT TEST SET EVALUATION**
   ```python
   # After training, evaluate on test set:
   test_predictions = model.predict(X_test)
   test_accuracy = accuracy_score(y_test, test_predictions)
   test_r2 = r2_score(y_test, test_predictions)
   ```
   - Report test metrics separately
   - This is the ONLY way to detect overfitting

2. **⚠️ VERIFY REGIME MODEL TRAINING**
   - Check if regime models used proper temporal splits
   - If not, regime probabilities are leaking future information
   - Consider retraining regime models with strict temporal validation

3. **⚠️ IMPLEMENT TEMPORAL CV FOR HPO**
   - Use `TimeSeriesSplit` instead of regular K-Fold
   - Ensures CV respects temporal ordering
   - Prevents look-ahead bias during HPO

### IMPORTANT - Should Implement:

4. **📊 ADD WALK-FORWARD VALIDATION**
   - More robust than single train/val/test split
   - Simulates real trading conditions
   - Detects regime changes and model degradation

5. **🔍 ADD OVERFITTING DETECTION**
   - Compare train vs validation vs test metrics
   - Calculate train/test gap: `(train_acc - test_acc) / train_acc`
   - Flag if gap > 10%

6. **📈 ADD LEARNING CURVES**
   - Plot accuracy vs training set size
   - Detect if model needs more data
   - Identify if model is overfitting or underfitting

---

## 📊 CURRENT STATUS SUMMARY

| Aspect | Status | Confidence |
|--------|--------|------------|
| **Temporal Separation** | ✅ Good | High |
| **Embargo Periods** | ✅ Adequate | High |
| **Regularization in HPO** | ✅ Present | High |
| **Target Column Removal** | ✅ Implemented | High |
| **Test Set Evaluation** | ❌ **MISSING** | **CRITICAL** |
| **Temporal CV** | ⚠️ Unknown | Medium |
| **Regime Feature Leakage** | ⚠️ Unknown | Medium |
| **CNN Model Performance** | ❌ Failed | High |

---

## 🚨 CONCLUSION

**The 96% accuracy is SUSPICIOUS because**:
1. ❌ No held-out test set evaluation
2. ⚠️ Metrics are from CV, which may not respect temporal ordering
3. ⚠️ Regime features may contain future information
4. ⚠️ Very high accuracy suggests possible data leakage

**Next Steps**:
1. **IMMEDIATELY**: Add test set evaluation to reports
2. **HIGH PRIORITY**: Verify regime model training methodology
3. **HIGH PRIORITY**: Implement temporal CV for HPO
4. **MEDIUM PRIORITY**: Add walk-forward validation
5. **MEDIUM PRIORITY**: Investigate why CNN failed (R² ≈ 0)

**Until test set metrics are available, treat the 96% accuracy as UNVALIDATED.**
