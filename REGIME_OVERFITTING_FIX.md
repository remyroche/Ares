# Regime Ensemble Overfitting Fix

## Problem Summary

The regime ensemble training was achieving unrealistic perfect scores (99.9%+ accuracy, 1.0 ROC-AUC) due to **data leakage** in the base model training process.

### Evidence from Training Report

File: `outcomes/regime_ensemble_training_report_ETHUSDT_20251112_193453.md`

```
Accuracy: 0.9990 (99.9%)
All ROC-AUC scores: 1.0000 (perfect)
All PR-AUC scores: 1.0000 (perfect)
Log Loss: 0.0048 (extremely low)
Change-Point Detection: 1.0000 (perfect)
```

These metrics are classic indicators of data leakage, not genuine predictive performance.

## Root Cause Analysis

### The Data Leakage Mechanism

**Location:** `src/training/steps/market_analysis/components/regime_ensemble_training.py`

**Original Execution Order (INCORRECT):**

1. **Line 884:** Prepare full dataset `X, y` (100% of data)
2. **Lines 897-901:** If base models not found, train them on **FULL dataset** `X, y`
   ```python
   base_models = self._train_base_models(X, y, regime_labels)  # ⚠️ Uses 100%
   ```
3. **Lines 938-940:** Split data into train/val/test AFTER base model training
   ```python
   X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(...)
   ```
4. **Line 994:** Train meta-learner on train+val
5. **Line 998:** Evaluate on test set

**The Problem:**
- Base models were trained on indices 0-100% (including the future test set)
- Test set was indices 85-100%
- During evaluation, base models predicted on test set they had already memorized
- Meta-learner achieved perfect scores because base models "knew" the answers

### Detailed Leakage Flow

```
Training Phase:
├─ base_models.fit(X[0:100%], y[0:100%])  # Memorizes ALL data including test
└─ Models learn patterns from indices 85-100%

Evaluation Phase:
├─ X_test = X[85:100%]  # Test set extraction
├─ base_predictions = base_models.predict(X_test)  # Models have SEEN this data!
├─ meta_features = generate_from(base_predictions)  # Perfect predictions leak
└─ meta_learner.predict(meta_features)  # Perfect scores (0.999+ accuracy)
```

## The Fix

### New Execution Order (CORRECT)

**Files Modified:**
- `src/training/steps/market_analysis/components/regime_ensemble_training.py` (lines 897-974)

**Corrected Flow:**

1. **Prepare full dataset** `X, y`
2. **Split data FIRST** (BEFORE training base models)
   ```python
   X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(...)
   ```
3. **Train base models ONLY on training data**
   ```python
   regime_labels_train = regime_labels_processed[:len(y_train)]
   base_models = self._train_base_models(X_train, y_train, regime_labels_train)  # ✅ Train only
   ```
4. **Train meta-learner** on train+val
5. **Evaluate** on test set

### Key Changes

```python
# BEFORE (WRONG):
base_models = self._train_base_models(X, y, regime_labels)  # All data
X_train, X_val, X_test, ... = split(X, y)  # Split after training

# AFTER (CORRECT):
X_train, X_val, X_test, ... = split(X, y)  # Split FIRST
base_models = self._train_base_models(X_train, y_train, regime_labels_train)  # Train only
```

### Code Location

**File:** `src/training/steps/market_analysis/components/regime_ensemble_training.py`

**Modified Section:** Lines 897-974

**Key Changes:**
1. Moved data splitting (lines 909-940) BEFORE base model training
2. Changed base model training to use only `X_train, y_train` instead of full `X, y`
3. Added clear warnings and documentation about the fix

## Expected Impact

### Before Fix
- **Training Accuracy:** 99.9%
- **Test Accuracy:** 99.9% (false - due to leakage)
- **ROC-AUC:** 1.0 (impossible)
- **Generalization:** Poor (model hasn't learned, just memorized)

### After Fix
- **Training Accuracy:** 70-85% (realistic)
- **Test Accuracy:** 60-75% (true out-of-sample performance)
- **ROC-AUC:** 0.75-0.85 (realistic)
- **Generalization:** Good (model learns patterns, not samples)

## Verification

To verify the fix worked:

1. **Run regime_models_training** to generate base model predictions
2. **Run regime_ensemble_training** with the fix
3. **Check the training report:**
   - Accuracy should be 60-85% (not 99%+)
   - ROC-AUC should be 0.70-0.90 (not 1.0)
   - Log Loss should be 0.3-0.8 (not 0.005)

## Prevention

### For Future Development

**Rule:** Always split data BEFORE training ANY model that will be evaluated on test data.

**Checklist:**
- [ ] Data split happens first
- [ ] Models trained only on training partition
- [ ] Test set never seen until final evaluation
- [ ] Cross-validation uses proper temporal/purged splits
- [ ] Meta-features generated using same split logic

## Related Issues

This fix addresses the same class of issue that was previously fixed in:
- `regime_models_training.py` - OOF predictions for training set (lines 1840-1880)
- Temporal cross-validation with embargo gaps

## Timeline

- **Issue Discovered:** 2025-11-12 (report showed 99.9% accuracy)
- **Root Cause Identified:** 2025-11-13 (data leakage in base model training)
- **Fix Implemented:** 2025-11-13 (reordered operations)

## Testing Recommendations

1. **Smoke Test:** Run full pipeline and verify accuracy < 95%
2. **Leakage Test:** Check that base model predictions on test set aren't perfect
3. **Temporal Test:** Verify test set timestamps > training set timestamps
4. **Comparison Test:** Compare to previous regime detection methods (should be similar accuracy)

---

**Status:** ✅ Fixed
**Priority:** Critical (P0)
**Component:** Regime Ensemble Training
**Impact:** High (affects all regime detection downstream)
