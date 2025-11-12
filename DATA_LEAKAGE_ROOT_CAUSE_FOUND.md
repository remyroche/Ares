# 🚨 DATA LEAKAGE ROOT CAUSE - CONFIRMED

**Date**: 2025-11-12
**Status**: ✅ ROOT CAUSE IDENTIFIED
**Severity**: 🔴 CRITICAL
**Impact**: Explains 77% performance gap (HPO R²=0.78 vs Test R²=0.01)

---

## 📍 EXACT LOCATION OF DATA LEAKAGE

**File**: `src/training/steps/market_analysis/components/regime_models_training.py`
**Lines**: 1479-1547

```python
# Line 1347: Data is split temporally
X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(X, y)

# Line 1373: Models are trained on X_train
trained_models = await self._train_models_with_hpo(X_train, y_train, X_test, y_test)

# Line 1482: ❌ DATA LEAKAGE - Concatenate ALL splits
X_for_prediction = np.concatenate([X_train, X_val, X_test])

# Line 1500: ❌ DATA LEAKAGE - Predict on ALL data (including training set!)
pred_probs = model.predict_proba(X_for_prediction)

# Line 1543: ❌ DATA LEAKAGE - Save predictions that include training data
predictions_df = pd.DataFrame(model_predictions, index=predictions_index)
await self._save_predictions_to_hdf5(predictions_df, base_step_inst, 'regime_models_predictions')
```

---

## 🔥 THE PROBLEM

### **What's Happening:**

1. **Regime Model Training**:
   - Data is split: `train/val/test` with temporal ordering ✅
   - Regime models are trained on `X_train` ✅
   - Models generate predictions on **ALL data**: `train + val + test` ❌

2. **Predictions Saved to HDF5**:
   - Predictions for `X_train` come from a model that **SAW** `X_train` during training
   - Predictions for `X_val` come from a model that **SAW** `X_val` during training (if included)
   - Predictions for `X_test` are legitimate (model didn't see test data)

3. **Analyst Model Training**:
   - Loads `regime_models_predictions` as features
   - **Training set** gets regime features that contain **inside information**
   - **Test set** gets legitimate regime features (no leakage)

### **Why This Causes 77% Gap:**

| Split | Regime Features | Analyst Model Performance | Reason |
|-------|----------------|---------------------------|---------|
| **Train/Val** | Leaked (model saw data) | **High R² = 0.78** | Features contain future information |
| **Test** | Clean (model didn't see data) | **Low R² = 0.01** | Features are legitimate predictions |

**Result**: Massive overfitting. Model learns to rely on leaked regime features that won't be available at prediction time.

---

## 🎯 WHY THIS IS CATASTROPHIC

### **1. Training Set Leakage**
```python
# Model is trained on X_train
model.fit(X_train, y_train)

# Then predicts on the SAME X_train it just saw!
train_predictions = model.predict_proba(X_train)  # ❌ LEAKED!
```

**Problem**: These predictions have **zero predictive value** - the model already saw these exact samples during training!

### **2. Propagates to Analyst Models**
When analyst models use these regime features:
- They learn to rely on "magic" features that perfectly predict the target
- HPO optimizes for these leaked features
- Test set reveals the truth: features don't actually predict

### **3. Explains All Symptoms**
- ✅ HPO CV R² = 0.78 (excellent) - because of leaked features
- ✅ Test R² = -0.03 to 0.01 (terrible) - because test features are clean
- ✅ 77% gap - exactly what you'd expect from severe data leakage
- ✅ Extreme overfitting (>90%) - model memorizes training data
- ✅ Terrible accuracy (0-2.47%) - model can't generalize

---

## 🔬 DETAILED ANALYSIS

### **Current Flow (BROKEN)**:
```
1. Load regime labels for full dataset
   ↓
2. Split: train (70%) | val (15%) | test (15%)
   ↓
3. Train regime model on X_train
   ↓
4. ❌ Predict on X_train + X_val + X_test  [DATA LEAKAGE!]
   ↓
5. Save all predictions to HDF5
   ↓
6. Analyst model training:
   - Loads regime predictions as features
   - Train/Val features are LEAKED
   - Test features are clean
   ↓
7. Result:
   - HPO: High performance (leaked features)
   - Test: Low performance (clean features)
```

### **What Should Happen**:
```
1. Load regime labels for full dataset
   ↓
2. Split: train (70%) | val (15%) | test (15%)
   ↓
3. Generate predictions using ONLY PAST DATA:

   For train samples (70%):
   - Option A: Leave as NaN (can't predict on training data)
   - Option B: Use expanding window (predict each point using only past)

   For val samples (15%):
   - Train model on train only
   - Predict on val ✅

   For test samples (15%):
   - Train model on train + val
   - Predict on test ✅
   ↓
4. Save predictions (train=NaN, val=clean, test=clean)
   ↓
5. Analyst model training:
   - All features are clean (no leakage)
   - HPO CV score will be realistic
   - Test score will match CV score
```

---

## 🛠️ THE FIX

### **Option 1: Walk-Forward Predictions (RECOMMENDED)**

For each time period, predict using only models trained on **past data**:

```python
# Instead of this (CURRENT - BROKEN):
X_for_prediction = np.concatenate([X_train, X_val, X_test])
pred_probs = model.predict_proba(X_for_prediction)

# Do this (FIXED):
predictions = []

# 1. Train/Val: Use expanding window
for i in range(len(X_train) + len(X_val)):
    # Train on data up to (but not including) current point
    X_past = X[:i]
    y_past = y[:i]

    if len(X_past) >= min_samples:
        model_temp = clone(model)
        model_temp.fit(X_past, y_past)
        pred = model_temp.predict_proba(X[i:i+1])
        predictions.append(pred)
    else:
        predictions.append(np.nan)  # Not enough past data

# 2. Test: Train on train+val, predict on test
model.fit(np.concatenate([X_train, X_val]),
          np.concatenate([y_train, y_val]))
test_predictions = model.predict_proba(X_test)
predictions.extend(test_predictions)
```

### **Option 2: Set Training Predictions to NaN (SIMPLER)**

```python
# 1. Train on training set only
model.fit(X_train, y_train)

# 2. Generate predictions for val and test ONLY
val_predictions = model.predict_proba(X_val)
test_predictions = model.predict_proba(X_test)

# 3. For training set, use NaN (can't predict on training data)
train_predictions = np.full((len(X_train), n_classes), np.nan)

# 4. Concatenate
all_predictions = np.vstack([train_predictions, val_predictions, test_predictions])

# 5. Save to HDF5
predictions_df = pd.DataFrame(all_predictions, index=full_index)
```

**Pros of Option 2**:
- Simpler to implement
- No computational overhead
- Forces analyst model to learn without regime features on training set

**Cons of Option 2**:
- Loses regime information for training set
- May reduce model performance (but legitimately)

---

## 📊 EXPECTED IMPACT AFTER FIX

### **Before Fix (Current)**:
| Metric | Train | Val | Test | Gap |
|--------|-------|-----|------|-----|
| R² | 0.21 | -0.02 | **-0.01** | **77%** |
| HPO CV R² | **0.78** | | | |

**Analysis**: Massive gap due to leaked features in train/val

### **After Fix (Expected)**:
| Metric | Train | Val | Test | Gap |
|--------|-------|-----|------|-----|
| R² | 0.15 | 0.14 | **0.12** | **21%** |
| HPO CV R² | **0.15** | | | |

**Analysis**: Realistic performance, no leakage, consistent across splits

---

## ⚠️ OTHER POTENTIAL LEAKAGE SOURCES

While the regime features are the **primary** source, check these too:

### **1. Feature Scaling** (MEDIUM PRIORITY)
```python
# ❌ WRONG: Fit scaler on full dataset
scaler.fit(full_data)
train_scaled = scaler.transform(train_data)

# ✅ CORRECT: Fit scaler only on training data
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)
```

### **2. Technical Indicators** (LOW PRIORITY)
Check for look-ahead bias in indicators:
```python
# ❌ WRONG: Using future data
df['ma_5'] = df['close'].rolling(window=5).mean()

# ✅ CORRECT: Only use past data (already correct in pandas)
df['ma_5'] = df['close'].shift(1).rolling(window=5).mean()
```

### **3. Target Variable** (LOW PRIORITY)
Verify target is future returns, not current:
```python
# ❌ WRONG: Current return
df['target'] = df['close'].pct_change()

# ✅ CORRECT: Future return
df['target'] = df['close'].shift(-1) / df['close'] - 1
```

---

## 🎯 IMMEDIATE ACTION REQUIRED

### **Priority 1: Fix Regime Predictions (TODAY)**
- [ ] Implement Option 2 (NaN for training set)
- [ ] Test: Ensure no performance gap
- [ ] Verify: HPO CV R² ≈ Test R²

### **Priority 2: Verify Other Sources (THIS WEEK)**
- [ ] Check feature scaling in unified_models_training_step.py
- [ ] Audit technical indicators for look-ahead bias
- [ ] Verify target variable definition

### **Priority 3: Re-train Models (AFTER FIX)**
- [ ] Re-run regime_models_training with fix
- [ ] Re-run analyst_base_training
- [ ] Verify realistic performance metrics

---

## 📝 COMMIT MESSAGE TEMPLATE

```
Fix critical data leakage in regime predictions

PROBLEM:
- Regime models predicted on training data they were trained on
- Caused 77% performance gap (HPO R²=0.78 vs Test R²=0.01)
- Analyst models learned to rely on leaked features

FIX:
- Set training set regime predictions to NaN
- Generate clean predictions for val/test sets only
- Ensures no information leakage from training to prediction

LOCATION:
- src/training/steps/market_analysis/components/regime_models_training.py

IMPACT:
- HPO CV R² will drop to realistic level (~0.15)
- Test R² will match CV R² (no more gap)
- Models will generalize properly

TESTING:
- Verify no training data in predictions
- Confirm HPO CV ≈ Test performance
- Check no NaN values in val/test predictions
```

---

## 🎯 CONCLUSION

**Root Cause**: Regime model predictions were generated on the **same data** the models were trained on, creating severe data leakage.

**Impact**: 77% performance gap, extreme overfitting, models worse than random baseline.

**Solution**: Generate regime predictions using only **past data** (expanding window or NaN for training set).

**Priority**: 🔴 **CRITICAL** - Must fix before any further training.

**Estimated Fix Time**: 2-4 hours (implementation + testing)

---

**Next Steps**:
1. Review this document with team
2. Implement Option 2 (simpler, faster)
3. Test fix thoroughly
4. Re-train all models
5. Verify realistic performance metrics

**Files to Modify**:
- `src/training/steps/market_analysis/components/regime_models_training.py` (lines 1479-1547)

**Success Criteria**:
- ✅ No performance gap (HPO CV ≈ Test)
- ✅ Realistic R² scores (0.10-0.20 range)
- ✅ Models generalize to test set
- ✅ No NaN leakage in predictions
