# 🎯 SR Quality Model - Ranking Validation Status

## 📊 **Current Status: RETRAINING IN PROGRESS**

Model is being retrained with fixed approach to resolve model collapse issue.

---

## 🔴 **Previous Results (Model Collapse)**

### What Went Wrong
- **Precision@K**: 100% (perfect) ✅
- **Spearman ρ**: 0.386 (poor) ❌
- **Separation**: 0.004 (virtually none) ❌
- **Future R²**: -0.538 (worse than mean) ❌

### Root Cause
**Triple filtering removed variance:**
1. Top 30% filtering
2. Quality < 0.25 filter
3. Strength < 0.4 filter
4. Extreme weights (0.1x to 3.0x)

**Result:** Model predicted ~0.81 for everything (the mean of filtered data)

---

## ✅ **Fix Applied**

### Changes Made

1. **NO Hard Filtering**
   - Changed `filter_percentile` from 70.0 to 100.0
   - Keep ALL data to preserve variance
   
2. **Gentler Confidence Weights**
   - Reduced range from 30x (0.1-3.0) to 6.7x (0.3-2.0)
   - Preserves signal while still emphasizing quality

3. **Removed Additional Filters**
   - No quality < 0.25 filter
   - No strength < 0.4 filter

### Why This Works

**Before:**
```
Raw data → Filter top 30% → Remove weak → Remove low strength → Narrow range (0.7-1.0)
         → Model learns: "Predict 0.81" → Collapse
```

**After:**
```
Raw data → Gentle weighting (0.3x to 2.0x) → Full range (0.0-1.0) preserved
         → Model learns: Features discriminate quality → Success
```

---

## 📈 **Expected Improvements**

Based on the fix, we expect:

| Metric | Before | Expected After | Target |
|--------|--------|----------------|--------|
| Precision@5 | 100% | 90-100% | >80% |
| Precision@10 | 100% | 85-95% | >75% |
| Spearman ρ | 0.386 | 0.55-0.70 | >0.60 |
| Separation | 0.004 | 0.25-0.40 | >0.35 |
| Future R² | -0.538 | 0.30-0.50 | >0.45 |

**Key Improvement:** Model will actually learn to discriminate instead of predicting the mean.

---

## 🔬 **Validation Steps**

Once retraining completes:

1. **Check Training Metrics**
   - Avg Val R² should be positive (20-40%)
   - Not collapsed (-60% like before)

2. **Run Ranking Validation**
   ```bash
   python3 scripts/validate_sr_ranking_metrics.py
   ```

3. **Verify Separation**
   - Mean strong predictions >> Mean weak predictions
   - Should differ by >0.25 (not 0.004)

4. **Check Variance**
   - Predictions should have std > 0.10
   - Not all ~0.81

---

## 📝 **Files Modified**

1. **`src/tactician/sr_levels/ml_quality/sr_quality_model.py`**
   - Lines 289-313: New filtering logic (preserve data)
   - Added safeguards against aggressive filtering

2. **`src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`**
   - Lines 220-239: Gentler tiered weights

3. **`scripts/run_sr_workflow.py`**
   - Line 665: Changed filter_percentile to 100.0
   - Updated logging messages

4. **`train_sr_quality_model.py`**
   - Lines 95-109: No hard filtering

---

## ⏰ **Next Steps**

1. ⏳ **Wait for retraining** to complete (~2-5 minutes)
2. ✅ **Validate** with ranking metrics
3. 📊 **Verify** separation and variance
4. 🎯 **Confirm** production-ready

---

**Bottom Line:** The fix addresses the fundamental issue - preserving variance is critical for learning to discriminate, not just memorize the mean.

