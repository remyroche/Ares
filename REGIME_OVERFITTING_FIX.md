# Regime Ensemble Overfitting Fix

## Problem Summary

The regime ensemble training was achieving unrealistic perfect scores (99.9%+ accuracy, 1.0 ROC-AUC) due to **TWO data leakage issues** in the base model training and meta-learner training processes.

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

## Phase 2: Meta-Learner Data Leakage Fix

### The Second Leakage Problem

After fixing Phase 1 (base model training), a **second data leakage issue** was discovered in the meta-learner training:

**Location:** `src/training/steps/market_analysis/components/regime_ensemble_training.py:2448+`

**The Problem:**
1. Base models trained on `X_train` (70%)
2. Meta-learner receives `X_train_full = [X_train (70%) + X_val (15%)]` = 85%
3. **Meta-feature generation:** `base_models.predict_proba(X_train_full)` directly predicts on full data

**Leakage Mechanism:**
```python
# Base models trained on X_train (70%)
base_models.fit(X_train, y_train)

# Meta-features for meta-learner
meta_features = base_models.predict_proba(X_train_full)  # ⚠️ LEAKAGE!
# X_train_full contains:
#   - X_train (70%): Base models trained on this → overly confident predictions
#   - X_val (15%): Base models never saw → realistic predictions

# Meta-learner sees mixed data:
#   - 70% artificially confident (leakage)
#   - 15% realistic
```

### Phase 2 Fix: OOF Meta-Features

**Solution:** Use Out-Of-Fold (OOF) predictions for the training portion, similar to `regime_models_training.py`.

**Implementation:**

1. **Modified `_train_stacker_lgbm_calibrated`** to accept `train_size` parameter
2. **Created `_generate_oof_meta_features`** method for leak-free predictions
3. **Split meta-feature generation:**
   - Train portion (70%): Generate OOF predictions using 5-fold time series CV
   - Val portion (15%): Generate clean predictions (base models never saw this)
   - Combine for meta-learner training

**Code Flow:**
```python
# Phase 2 CORRECT approach:
if train_size is not None:
    # 1. OOF predictions for train portion (no leakage)
    train_meta_features = _generate_oof_meta_features(
        base_models, X_train_portion, y_train_portion, n_splits=5
    )

    # 2. Clean predictions for val portion
    val_meta_features = base_models.predict_proba(X_val_portion)

    # 3. Combine for leak-free meta-learner training
    meta_features = np.vstack([train_meta_features, val_meta_features])

meta_learner.fit(meta_features, y_train_full)
```

**Key Changes:**
- **Line 1026-1029:** Pass `train_size` to `_train_stacker_lgbm_calibrated`
- **Lines 2497-2549:** Conditional OOF vs clean meta-feature generation
- **Lines 1872-2031:** New `_generate_oof_meta_features` method

### OOF Generation Details

The `_generate_oof_meta_features` method:
1. **Time Series CV:** Uses 5-fold `TimeSeriesSplit` for temporal validation
2. **Retraining:** For each fold, recreates models and trains on fold's train data
3. **OOF Predictions:** Predicts on fold's validation data (never seen)
4. **Coverage:** Handles early folds with no predictions (fills with uniform probs)
5. **Meta-Features:** Generates uncertainty, confidence, and disagreement features from OOF predictions

### Combined Fix Impact

**Before (Both Leakages):**
- Base models: Memorized test set → 99.9% on test
- Meta-learner: Trained on leaked predictions → Perfect ensemble

**After Phase 1 Only:**
- Base models: Never see test set → Realistic test performance
- Meta-learner: Still sees leaked train predictions → Slightly inflated

**After Phase 1 + Phase 2:**
- Base models: Never see test set → Realistic test performance
- Meta-learner: Trained on OOF + clean predictions → True generalization

## Related Issues

This fix addresses the same class of issue that was previously fixed in:
- `regime_models_training.py` - OOF predictions for training set (lines 1840-1880)
- Temporal cross-validation with embargo gaps

## Timeline

- **Issue Discovered:** 2025-11-12 (report showed 99.9% accuracy)
- **Phase 1 Root Cause:** 2025-11-13 (data leakage in base model training)
- **Phase 1 Fix:** 2025-11-13 (reordered operations - train base models after split)
- **Phase 2 Root Cause:** 2025-11-13 (data leakage in meta-feature generation)
- **Phase 2 Fix:** 2025-11-13 (OOF meta-features for training portion)

## Testing Recommendations

1. **Smoke Test:** Run full pipeline and verify accuracy < 95%
2. **Leakage Test:** Check that base model predictions on test set aren't perfect
3. **Temporal Test:** Verify test set timestamps > training set timestamps
4. **Comparison Test:** Compare to previous regime detection methods (should be similar accuracy)

---

**Status:** ✅ Fixed (Phase 1 + Phase 2)
**Priority:** Critical (P0)
**Component:** Regime Ensemble Training
**Impact:** High (affects all regime detection downstream)

### Fixes Summary

**Phase 1: Base Model Training**
- File: `regime_ensemble_training.py`
- Lines: 897-974
- Change: Split data BEFORE training base models

**Phase 2: Meta-Learner Training**
- File: `regime_ensemble_training.py`
- Lines: 1026-1029, 1872-2031, 2448-2549
- Change: Use OOF predictions for train portion + clean predictions for val portion
