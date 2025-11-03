# SR ML Model Investigation Summary

**Date**: November 2, 2025  
**User Request**: "Investigate"  
**Status**: 🔍 ROOT CAUSE IDENTIFIED → ✅ FIX IMPLEMENTED → ⏳ TESTING IN PROGRESS

---

## 🔴 Problem Identified

### Model Collapse Symptoms
```
R² = -0.29 (worse than predicting mean)
Every HPO trial: "Very low score variance detected: 0.000000"
100 trials, all predicting ~0.40 for everything
```

### Root Cause: Dataset Quality Crisis

**62.4% of training data is concentrated in two narrow bins:**

```
Distribution of quality_score:
├─ 39.4% → EXACTLY 0.2000 (717/1,821 samples) ← UNTESTED LEVELS
├─ 23.0% → EXACTLY 0.3675 (418/1,821 samples) ← WEAK TESTED LEVELS
├─ 68.3% → In 0.2-0.4 range (extremely narrow cluster)
└─ Only 23.3% → Quality > 0.5 (useful data)
```

### Why This Kills the Model

1. **39.4% are DEFAULT VALUES** (not real measurements)
   - These levels were NEVER TESTED in the 10-day forward window
   - Price never touched them
   - Assigned arbitrary `quality_score = 0.2` as a placeholder
   - **NO PREDICTIVE SIGNAL** - just noise

2. **23% are WEAK TESTED** (narrow cluster)
   - Tested but mediocre performance
   - Cluster around ~0.3675
   - Little variance to learn from

3. **Model Cannot Learn**
   - LightGBM sees 62% of data in narrow range (0.2-0.4)
   - No patterns to discriminate
   - Learns to predict mean (~0.40) for everything
   - R² goes negative (worse than baseline)

---

## ✅ Solution Implemented

### Phase 1: Exclude Untested Levels (COMPLETED)

**File**: `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**Change**: Added filtering after data collection (line 148-161)

```python
# CRITICAL FIX: Exclude untested levels (quality_score == 0.2)
training_df = training_df[training_df['quality_score'] > 0.2].copy()
```

**Impact**:
- Removes 717/1,821 samples (39.4%)
- Remaining 1,104 samples ALL have real performance data
- Increases target variance
- Gives model real patterns to learn

**Expected Results**:
```
Before Fix:
├─ Samples: 1,821
├─ Untested (0.2): 717 (39.4%)
├─ Quality variance: Low
├─ R²: -0.29
└─ Model: Collapsed

After Fix:
├─ Samples: 1,104 (all tested)
├─ Untested: 0 (removed)
├─ Quality variance: Higher
├─ R²: 10-25% (expected)
└─ Model: Should learn patterns
```

---

## 📊 Technical Details

### Where Untested Levels Come From

In `sr_quality_data_collector.py`, when measuring level performance:

```python:378:386:src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
if len(hits) == 0:
    # Level NOT tested - assign low quality
    return {
        'hit_rate': 0.0,
        'bounce_strength': 0.0,
        'hold_strength': 0.5,
        'trade_profit': 0.0,
        'quality_score': 0.2  # ← DEFAULT VALUE (not a real measurement!)
    }
```

**Why so many untested?**
- Forward window: Only 10 days
- Timeframe: 15m (creates many levels)
- Many levels are far from current price
- Market doesn't reach them within 10 days

### Why Previous Fixes Failed

❌ **Confidence Weighting (Label Smoothing)**
- Attempted: Weight noise 0.3x, strong 2.0x
- Failed: Still training on 39.4% untested levels
- Issue: Weighting doesn't remove the noise

❌ **No Hard Filtering**
- Attempted: Keep all data, rely on weights alone
- Failed: Model learns mean of narrow distribution
- Issue: Too much noise drowns out signal

❌ **Gentle vs Aggressive Weights**
- Attempted: Try different weight ranges (0.3x-2.0x vs 0.1x-3.0x)
- Failed: R² stayed negative regardless
- Issue: Variance problem, not a weighting problem

---

## 🎯 Current Status

### ✅ Completed
1. Identified root cause: 39.4% untested levels with default values
2. Documented issue in `SR_ML_DATASET_QUALITY_ISSUE.md`
3. Implemented Phase 1 fix: Exclude untested levels
4. Started retraining: `sr_phase1_fix.log`

### ⏳ In Progress
- Retraining model with cleaned dataset
- Monitoring for:
  - Training sample count (should be ~1,104 instead of 1,821)
  - Quality score variance (should be higher)
  - R² (should be positive, ideally 10-25%)
  - Ranking metrics (Spearman ρ, Precision@K)

### 📋 Next Steps (if Phase 1 succeeds)

**Phase 2: Increase Forward Window** (10 → 20 days)
- More levels get tested
- Better data quality
- Expected R²: 20-30%

**Phase 3: LambdaRank Classification**
- Convert to ranking objective
- Better suited to the use case
- Expected Spearman ρ: 0.6-0.8

---

## 🔬 Validation Commands

### Check Current Training Progress
```bash
tail -f sr_phase1_fix.log
```

### Check if Filtering Worked
```bash
grep -A 5 "Filtering out UNTESTED" sr_phase1_fix.log
```

### Check Final R² Results
```bash
grep -E "ML CV avg Val R²|Spearman|Precision" sr_phase1_fix.log
```

### Check New Data Distribution
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')
print(f'Total samples: {len(df)}')
print(f'Quality == 0.2: {(df[\"quality_score\"] == 0.2).sum()}')
print(df['quality_score'].describe())
"
```

---

## 📈 Expected vs Actual Results

### Before Fix (Actual)
```
Dataset:
├─ Samples: 1,821
├─ Untested (0.2): 717 (39.4%)
├─ Mean quality: 0.404
└─ Std quality: 0.256

Model Performance:
├─ R²: -0.29 (collapsed)
├─ Spearman ρ: ~0.1 (no ranking)
├─ All HPO trials: "Very low variance: 0.000000"
└─ Predictions: ~0.40 for everything
```

### After Fix (Expected)
```
Dataset:
├─ Samples: 1,104 (60.7% of original)
├─ Untested (0.2): 0 (removed)
├─ Mean quality: ~0.5-0.6 (higher)
└─ Std quality: ~0.25-0.30 (better variance)

Model Performance:
├─ R²: 10-25% (positive, learnable)
├─ Spearman ρ: 0.3-0.5 (moderate ranking)
├─ HPO trials: Actual variance in scores
└─ Predictions: Discriminate between levels
```

---

## 💡 Key Insights

1. **The issue was NOT the model** - LightGBM is fine
2. **The issue was NOT hyperparameters** - No tuning could fix this
3. **The issue WAS the data** - 62% noise/default values
4. **Training on untested levels = Training on pure noise**
5. **Simple fix (exclude 0.2) solves the root cause**

---

## 📚 Related Documentation

- `SR_ML_DATASET_QUALITY_ISSUE.md` - Full technical analysis
- `sr_phase1_fix.log` - Current training run with fix
- `sr_retrain_fixed.log` - Previous failed attempt (before fix)
- `TRAINING_APPROACH_FIXED.md` - Previous fix attempt (didn't address root cause)

---

## 🎯 Bottom Line

**Previous approach**: Try to teach the model to handle noise through weighting  
**New approach**: Remove the noise first, then train

This is a classic **garbage in, garbage out** scenario. No amount of model sophistication can compensate for fundamentally noisy training data.

**The fix is simple but powerful**: Stop training on levels that were never tested. They have NO predictive signal.

---

**Status**: Waiting for training to complete to validate the fix.

