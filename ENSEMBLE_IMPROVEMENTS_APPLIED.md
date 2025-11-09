# Regime Ensemble Performance Improvements

**Date:** November 9, 2025  
**Status:** ✅ Critical Improvements Applied

## Issues Addressed

### 1. ✅ Custom Class Weights (Combat Imbalance)

**Problem:**  
- Model was biased toward Regime 2 (96% of predictions)
- `class_weight='balanced'` was insufficient
- Majority class dominated predictions

**Solution Implemented:**
```python
# Calculate custom class weights with extra penalty for majority classes
class_weights = {}
max_count = max(counts)
for regime_id, count in zip(unique, counts):
    # Base balanced weight
    balanced_weight = total_samples / (n_classes * count)
    # Extra penalty for majority classes (inverse frequency squared)
    majority_penalty = (max_count / count) ** 1.5
    # Combined weight
    class_weights[int(regime_id)] = balanced_weight * majority_penalty
```

**Expected Impact:**
- Minority classes get **much higher weights** (e.g., 3-4x)
- Majority class gets **lower weight** (e.g., 0.5x)
- Forces model to pay attention to all regimes

---

### 2. ✅ Simplified Meta-Features (Reduce Noise)

**Problem:**  
- Enhanced features added 13 extra features (40 → 53)
- Many features were noise rather than signal:
  - Class-specific features dependent on training labels
  - Regime transition features
  - Complex interaction features

**Solution Implemented:**
```python
# CRITICAL FIX: Use ONLY base meta-features (no enhancement)
simplified_meta_features = meta_features  # Use original 40 features
```

**Features Removed:**
- ❌ Max probabilities
- ❌ Entropy features
- ❌ Variance features
- ❌ Confidence gap
- ❌ Prediction margin
- ❌ Regime stability
- ❌ Regime changes
- ❌ Class-specific features (6 features that depend on training labels)

**Features Kept (40 total):**
- ✅ Base model predictions (12 features: 2 models × 6 regimes)
- ✅ Uncertainty features (11 features)
- ✅ Confidence features (7 features)
- ✅ Disagreement features (10 features)

**Expected Impact:**
- **Cleaner signal** - only features that generalize well
- **Better generalization** - no overfitting to training-specific patterns
- **Faster training** - fewer features to process

---

## Implementation Details

### Files Modified

**File:** `src/training/steps/market_analysis/components/regime_ensemble_training.py`

**Changes:**

1. **Lines 1310-1333:** Calculate custom class weights
   ```python
   # Calculate balanced weights with extra penalty for majority classes
   class_weights = {}
   max_count = max(counts)
   for regime_id, count in zip(unique, counts):
       balanced_weight = total_samples / (n_classes * count)
       majority_penalty = (max_count / count) ** 1.5
       class_weights[int(regime_id)] = balanced_weight * majority_penalty
   ```

2. **Lines 1328-1333:** Use simplified features instead of enhanced
   ```python
   simplified_meta_features = meta_features  # Use original 40 features
   ```

3. **Line 1355:** Apply custom weights to model factory
   ```python
   class_weight=class_weights,  # Use custom weights instead of 'balanced'
   ```

4. **Line 1367:** Train HPO with simplified features
   ```python
   X=simplified_meta_features,  # Use simplified features
   ```

5. **Line 1389:** Train model with simplified features
   ```python
   meta_learner.fit(simplified_meta_features, y, sample_weight=sample_weight)
   ```

6. **Line 1433:** Calibrate with simplified features
   ```python
   calibrated_meta_learner.fit(simplified_meta_features, y, sample_weight=sample_weight)
   ```

7. **Lines 1441-1449:** Update FeatureContract metadata
   ```python
   expected_shape=(None, simplified_meta_features.shape[1]),
   'simplified_feature_count': simplified_meta_features.shape[1],
   'feature_simplification': 'enabled'
   ```

---

## Expected Performance Improvements

### Before (Current State)
- **Accuracy:** 30.13%
- **Balanced Accuracy:** 17.08%
- **Regimes Predicted:** 2 out of 6 (Regime 2 and 4 only)
- **Regime 2 Bias:** 96.4% of predictions
- **Confidence:** 32.45% average

### After (Expected)
- **Accuracy:** >45% (target: beat best base model)
- **Balanced Accuracy:** >35% (2x improvement)
- **Regimes Predicted:** All 6 regimes
- **Regime Distribution:** More balanced predictions
- **Confidence:** >45% average

### Key Metrics to Watch

1. **Per-Regime Recall:**
   - All regimes should have >10% recall (currently 4 regimes have 0%)
   - No regime should dominate (currently Regime 2 has 96% recall)

2. **Balanced Accuracy:**
   - Should be >35% (currently 17%)
   - Closer to raw accuracy indicates better balance

3. **Confusion Matrix:**
   - Should show predictions across diagonal
   - Currently shows all predictions in Regime 2 column

4. **Class Weights Applied:**
   - Check logs for actual weights calculated
   - Minority classes should have 3-5x higher weights

---

## Testing Instructions

### Run Training
```bash
python3 src/launcher/ares_launcher.py regime_ensemble_training \
  --symbol ETHUSDT \
  --timeframe 1h \
  --execution-mode blank
```

### Check Logs
Look for these new log messages:
```
📊 [REGIME_ENSEMBLE] Training class distribution: {0: X, 1: Y, ...}
🎯 [REGIME_ENSEMBLE] Custom class weights: {0: W1, 1: W2, ...}
🔧 [REGIME_ENSEMBLE] Using simplified meta-features (base predictions only)
📊 [REGIME_ENSEMBLE] Simplified meta-features shape: (N, 40)
📊 [REGIME_ENSEMBLE] Feature reduction: 53 -> 40 (removed noisy enhanced features)
```

### Verify Results
```bash
# Check latest report
cat outcomes/regime_ensemble_training_report_ETHUSDT_*.md | grep -A 10 "Overall Performance"

# Check per-regime performance
cat outcomes/regime_ensemble_training_report_ETHUSDT_*.md | grep -A 10 "Per-Regime Performance"
```

---

## Rollback Plan

If performance degrades:

1. **Revert to balanced weights only:**
   ```python
   class_weight='balanced'  # Instead of custom weights
   ```

2. **Re-enable enhanced features:**
   ```python
   enhanced_meta_features = self._create_enhanced_meta_features(meta_features, y)
   # Use enhanced_meta_features instead of simplified_meta_features
   ```

---

## Additional Recommendations

### If Performance Still Poor (<40% Accuracy)

1. **Check Base Model Quality:**
   ```bash
   # Find latest regime_models_training report
   ls -lt outcomes/regime_models_training_report_*.md | head -1
   ```
   - If base models are <40% accurate, fix them first
   - Ensemble can't be better than its components

2. **Try Different Meta-Learner:**
   - XGBoost with `scale_pos_weight`
   - CatBoost with `auto_class_weights`
   - Neural network with focal loss

3. **Disable Calibration:**
   - Calibration on imbalanced data can hurt
   - Try `calibration_method='none'`

4. **Add SMOTE/ADASYN:**
   - Synthetic oversampling for minority classes
   - Only if class weights aren't enough

---

## Success Criteria

✅ **Minimum Requirements:**
- Accuracy > 40%
- Balanced Accuracy > 30%
- All 6 regimes predicted (>0% recall each)
- No single regime >60% of predictions

✅ **Target Performance:**
- Accuracy > 50%
- Balanced Accuracy > 40%
- Ensemble beats best base model by >5%
- Confidence > 50% average

✅ **Excellent Performance:**
- Accuracy > 60%
- Balanced Accuracy > 50%
- All regimes >20% recall
- Confidence > 60% average
