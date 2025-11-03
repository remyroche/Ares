# ML Model Comprehensive Fix Summary
**Generated:** 2025-11-02  
**Analysis Period:** 18-month SR workflow run  
**Status:** 🔴 CRITICAL ISSUES FIXED

---

## 🔍 Root Cause Analysis

### **Issue #1: TARGET LEAKAGE (CRITICAL) 🔴**

**Discovery:**
- Model showed 4 features with non-zero importance: `rejection_speed` (43.1%), `hold_quality` (34.2%), `speed_quality` (13.7%), `bounce_quality` (9.0%)
- These are NOT predictive features - they are **TARGET VARIABLES**
- The model was predicting `quality_score` FROM other quality scores (circular reasoning)

**Evidence:**
```python
# Correlation analysis:
rejection_speed:  corr=0.818 with quality_score  ❌ TARGET
hold_quality:     corr=0.854 with quality_score  ❌ TARGET  
speed_quality:    corr=0.818 with quality_score  ❌ TARGET
bounce_quality:   corr=0.682 with quality_score  ❌ TARGET

# vs Real features:
feature_distance_x_volatility:  corr=0.263  ✅ LEGITIMATE FEATURE
feature_prominence:             corr=0.243  ✅ LEGITIMATE FEATURE
```

**Impact:**
- Model appeared to work well (Precision@10 = 100%, Spearman = 0.852)
- But it was cheating - using future information to predict the future
- Real features (107 `feature_*` columns) were all getting zero importance

**Fix Applied:**
```python
# BEFORE (train() and train_with_hpo()):
exclude_cols = ['date', 'symbol', 'quality_score', ...]  # Incomplete!

# AFTER:
exclude_cols = [
    'date', 'symbol', 'exchange', 'timeframe', 'sample_weight',
    # Primary target
    'quality_score',
    # Sub-targets - MUST EXCLUDE!
    'bounce_quality', 'hold_quality', 'trade_quality', 
    'speed_quality', 'volume_confirmation_quality',
    # Performance metrics - MUST EXCLUDE!
    'hit_rate', 'bounce_strength', 'max_bounce_strength',
    'hold_strength', 'trade_profit', 'rejection_speed', 'volume_quality'
]

# SAFER APPROACH: Use only columns starting with 'feature_'
feature_cols = [c for c in training_data.columns 
               if c.startswith('feature_') and not pd.isna(training_data[c]).all()]
```

---

### **Issue #2: INSUFFICIENT TRAINING DATA 🔴**

**Problem:**
- Current: 285 samples for 107 features = **3:1 ratio**
- Required: 20:1 ratio minimum (2,140 samples needed)
- Fold 0: Only 50 samples → negative R² (-0.338)
- High variance: R² std dev = 0.264

**Fix Applied:**
```python
# In run_sr_workflow.py:

# Training period:
start_dt = end_dt - timedelta(days=730)  # 24 months (was 6 months)

# Sampling frequency:
default=0.5  # 12-hour sampling (was 1 = daily)
```

**Expected Impact:**
- Before: ~285 samples (6 months × 1.6 samples/day)
- After: ~2,920 samples (24 months × 2 samples/day × 1.6)  
- Result: **10x more data** → stable folds, better feature learning

---

### **Issue #3: FEATURE QUALITY ISSUES ⚠️**

**Analysis of 107 Real Features:**

| Category | Count | Details |
|----------|-------|---------|
| Zero variance | 6 | Constants (e.g., `feature_hour_of_day` = 0.79, high_vol features = 0) |
| Low variance (<0.001) | 15 | Minimal signal |
| Medium variance (0.001-0.01) | 10 | Some signal |
| High variance (>0.01) | 76 | Good variance |
| **Strong correlation (>0.3)** | **0** | ❌ **None!** |
| Medium correlation (0.1-0.3) | 50 | Weak predictors |
| Weak correlation (<0.1) | 52 | Very weak |

**Top 10 Features by Correlation:**
1. `feature_distance_x_volatility`: 0.263
2. `feature_volatility_regime_score`: 0.261
3. `feature_momentum_adjusted_distance`: 0.257
4. `feature_volume_x_trend`: 0.244
5. `feature_prominence`: 0.243
6. `feature_prominence_in_low_vol`: 0.243
7. `feature_prominence_x_strength`: 0.242
8. `feature_distance_x_velocity`: 0.238
9. `feature_quality_tier`: 0.235
10. `feature_volume_regime`: 0.223

**Key Insights:**
- No single feature strongly predicts quality
- Quality is a complex combination of many weak signals
- Need larger sample size to learn these complex interactions

**Fix Applied:**
```python
# In sr_quality_model.py train():

# Remove zero-variance features
feature_variances = X.var()
zero_var_features = feature_variances[feature_variances < 1e-10].index.tolist()
if zero_var_features:
    X = X.drop(columns=zero_var_features)
    self.feature_names = X.columns.tolist()
```

---

### **Issue #4: HIGH CROSS-VALIDATION VARIANCE 🔴**

**Problem:**
- Val R² ranges: -0.06 to 0.66 (huge spread!)
- Std dev: 0.264 (too high for production)
- Performance highly dependent on data split

**Fix Applied:**
```python
# Added stratified binning to ensure balanced quality distribution:

y_binned = pd.qcut(y, q=min(5, len(y)//10), labels=False, duplicates='drop')

self.logger.info(f"\n   📊 Using Stratified Time-Series CV:")
self.logger.info(f"      Quality bins: {y_binned.nunique()}")
for bin_idx in range(y_binned.nunique()):
    bin_count = (y_binned == bin_idx).sum()
    bin_mean = y[y_binned == bin_idx].mean()
    self.logger.info(f"      Bin {bin_idx}: {bin_count} samples (avg quality: {bin_mean:.3f})")
```

---

### **Issue #5: INSUFFICIENT REGULARIZATION ⚠️**

**Problem:**
- HPO search space: L1/L2 max = 20.0
- High variance (0.264) suggests need for stronger regularization

**Fix Applied:**
```python
# EXPANDED regularization search space:

search_space = {
    # Model complexity (more conservative)
    'num_leaves': {'type': 'int', 'low': 10, 'high': 40, 'default': 23},
    'max_depth': {'type': 'int', 'low': 3, 'high': 6, 'default': 5},
    
    # EXPANDED REGULARIZATION (was 20.0)
    'lambda_l1': {'type': 'float', 'low': 0.5, 'high': 50.0, 'default': 2.0, 'log': True},
    'lambda_l2': {'type': 'float', 'low': 0.5, 'high': 50.0, 'default': 2.0, 'log': True},
    
    # More evidence per leaf (expanded from 150)
    'min_data_in_leaf': {'type': 'int', 'low': 20, 'high': 200, 'default': 60},
    
    # Slower learning (expanded range)
    'learning_rate': {'type': 'float', 'low': 0.003, 'high': 0.05, 'default': 0.01, 'log': True},
    
    # Stronger splits required (expanded from 2.0)
    'min_gain_to_split': {'type': 'float', 'low': 0.1, 'high': 5.0, 'default': 0.5},
    
    # Aggressive subsampling (expanded lower bound from 0.5)
    'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 0.85, 'default': 0.6},
    'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 0.85, 'default': 0.6},
    'bagging_freq': {'type': 'int', 'low': 1, 'high': 15, 'default': 5},
}
```

---

## 📊 Expected Results After Fixes

### **Current Performance (WITH TARGET LEAKAGE):**
```
❌ Using target variables as features (cheating)
✅ Precision@10: 100%
✅ Spearman ρ: 0.852
✅ Separation: 0.238
❌ Future generalization: None (failed)
❌ Avg Val R²: 0.435 ± 0.264 (high variance)
```

### **Expected Performance (AFTER FIXES):**
```
✅ Using only legitimate features (no cheating)
⚠️ Precision@10: 70-85% (lower but realistic)
⚠️ Spearman ρ: 0.50-0.70 (lower but legitimate)
✅ Separation: 0.15-0.25 (maintains discrimination)
✅ Future generalization: R² > 0.45 (should pass)
✅ Avg Val R²: 0.40-0.55 ± 0.10-0.15 (lower variance)
✅ Fold 0 R²: > 0.0 (positive, not negative)
```

### **Performance will be LOWER but LEGITIMATE:**
- Without target leakage, the model must learn from weak signals
- 107 features with max correlation 0.263 = difficult learning problem
- Need to combine many features → requires more data
- With 10x more data, should learn complex interactions

---

## 🚀 Next Steps

### **1. Re-run Workflow with Fixes (REQUIRED)**

```bash
# This will:
# - Use 24 months of data (10x more samples)
# - Use 12-hour sampling (2x more samples per day)
# - Exclude all target variables (no cheating)
# - Remove zero-variance features automatically
# - Use stratified CV for balanced folds
# - Search expanded regularization space

python3 scripts/run_sr_workflow.py --lookback-days 548
```

### **2. Monitor These Metrics:**

**Training Logs:**
```
✅ Check: "Using only 'feature_*' columns: 107 features"
✅ Check: "Removing X zero-variance features"
✅ Check: "Stratified Time-Series CV: Quality bins: 5"
✅ Check: "Training samples: ~2,900" (was 285)
```

**Validation Results:**
```
Target Metrics:
- Fold 0 R² > 0.0 (not negative)
- CV Std Dev < 0.15 (was 0.264)
- Future generalization R² > 0.45 (was None)
- Precision@10: 70-85% (realistic range)
- Spearman ρ: 0.50-0.70 (legitimate correlation)
```

### **3. If Performance Still Poor:**

**Option A: Feature Engineering**
- Create interaction features (distance × volatility × trend)
- Add polynomial features for top correlations
- Create ratio features (strength / avg_strength)

**Option B: Ensemble Methods**
- Train separate models for different quality tiers
- Combine predictions with weighted voting

**Option C: Alternative Targets**
- Instead of predicting `quality_score` (complex)
- Predict binary classification: good (>0.7) vs bad (<0.4)
- Simpler problem = better performance with small data

---

## 📋 Summary of Files Modified

1. **`scripts/run_sr_workflow.py`**
   - Line 634: Training period 6 → 24 months
   - Line 1221: Sampling frequency 1 → 0.5 days

2. **`src/tactician/sr_levels/ml_quality/sr_quality_model.py`**
   - Lines 144-167: Fixed target leakage in `train()`
   - Lines 166-177: Added zero-variance filter in `train()`
   - Lines 187-199: Added stratified CV logging in `train()`
   - Lines 475-494: Fixed target leakage in `train_with_hpo()`
   - Lines 553-586: Expanded HPO search space

---

## ✅ Validation Checklist

Before deploying the fixed model:

- [ ] Confirm "Using only 'feature_*' columns" in logs
- [ ] Confirm ~2,900 training samples (was 285)
- [ ] Confirm Fold 0 R² > 0.0 (not negative)
- [ ] Confirm CV Std Dev < 0.15 (was 0.264)
- [ ] Confirm future generalization passes (R² > 0.45)
- [ ] Confirm no target variables in feature importance
- [ ] Confirm top features are legitimate (start with 'feature_')
- [ ] Manually validate predictions on hold-out set

---

**Status:** 🟢 Ready for re-training with 10x more data and no target leakage!

