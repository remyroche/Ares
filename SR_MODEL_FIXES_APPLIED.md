# 🔧 SR Model Fixes Applied - Complete Overhaul

**Date:** November 2, 2025  
**Status:** ✅ All Fixes Implemented

---

## 🚨 **Problems Identified**

### **Problem 1: Model Predicting Constant Value (0.742)**
```
BEFORE FIXES:
  Spearman ρ: nan (undefined!)
  Separation: -0.000 (no separation)
  Mean strong: 0.742
  Mean weak: 0.742  ← Same prediction for everything!
  Prediction variance: 0.000000
```

### **Problem 2: Too Few Training Samples**
```
730 days (2 years) → Only 255 samples
180 days → Only 499 samples

Why? Weekly sampling (sample_freq_days=7) + sparse SR detections
```

### **Problem 3: 12 Constant Features (13% of features)**
```
Features with ZERO variance:
  - feature_strong_bounce_count
  - feature_strong_bounce_ratio
  - feature_rejection_velocity
  - feature_recency_weighted_strength
  - feature_dwell_time
  - feature_multi_tf_score
  - feature_multi_tf_confirmations
  - feature_cluster_x_multi_tf
  - feature_hour_of_day
  - feature_day_of_week
  - feature_method_diversity
  - feature_mtf_x_prominence
```

### **Problem 4: Weak Feature Correlations**
```
Top feature correlation: 0.2554 (very weak!)
Mean correlation: 0.0822
18 features with correlation < 0.05
```

### **Problem 5: Too Many Features for Sample Size**
```
89 features for 255 samples = 2.9 samples/feature
Rule of thumb: Need 10-20 samples/feature
You need 890-1780 samples for 89 features!
```

---

## ✅ **Fixes Applied**

### **Fix 1: Change to DAILY Sampling (More Samples)**

**Files Modified:**
- `scripts/run_sr_workflow.py`

**Changes:**
```python
# BEFORE:
ml_sample_freq_days: int = 7  # Weekly sampling

# AFTER:
ml_sample_freq_days: int = 1  # DAILY sampling for more training data
```

**Expected Impact:**
- **730 days × daily sampling** = ~730 potential sample dates (vs 104 with weekly)
- Should get **1000-2000+ samples** instead of 255
- Better coverage of different market conditions

---

### **Fix 2: Remove Constant Features**

**File Modified:**
- `src/tactician/sr_levels/ml_quality/sr_quality_model.py` - `train_with_hpo()` method

**Changes:**
```python
# NEW CODE (lines 335-357):
self.logger.info(f"\n🔧 FIX 1: Removing constant/low-variance features...")
feature_std = X.std()
constant_features = feature_std[feature_std <= 0.001].index.tolist()

# Remove constant features
valid_features = feature_std[feature_std > 0.001].index.tolist()
X = X[valid_features]

self.logger.info(f"   ✅ Features after removing constants: {len(valid_features)}")
```

**Impact:**
- ✅ Removes 12 useless features with std ≤ 0.001
- ✅ Prevents LightGBM from using zero-information features
- ✅ Reduces feature count: 89 → ~77 features

---

### **Fix 3: Aggressive Feature Selection (Top 50 by LGBM Importance)**

**File Modified:**
- `src/tactician/sr_levels/ml_quality/sr_quality_model.py` - `train_with_hpo()` method

**Changes:**
```python
# NEW CODE (lines 359-407):
if len(valid_features) > 50:
    self.logger.info(f"\n🔧 FIX 2: Selecting top 50 features using LightGBM importance...")
    
    # Train a quick model to get feature importance
    quick_model = lgb.train(quick_params, train_data, num_boost_round=50)
    
    # Get feature importance (using 'gain' metric)
    importance = quick_model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Select top 50
    top_50_features = importance_df.head(50)['feature'].tolist()
    X = X[top_50_features]
    self.feature_names = top_50_features
```

**Impact:**
- ✅ Selects BEST 50 features based on actual predictive power
- ✅ Uses LightGBM 'gain' importance (better than correlation)
- ✅ Sample-to-feature ratio: **255/50 = 5.1** (better, still need more samples)
- ✅ Target ratio with more samples: **1500/50 = 30** ✅ (will be healthy)

---

### **Fix 4: Mandatory HPO Regularization**

**File Modified:**
- `src/tactician/sr_levels/ml_quality/sr_quality_model.py` - `train_with_hpo()` method

**Changes:**
```python
# BEFORE:
'lambda_l1': {'type': 'float', 'low': 0.1, 'high': 10.0}  # Could be 0.1 (weak)
'lambda_l2': {'type': 'float', 'low': 0.1, 'high': 10.0}  # Could be 0.1 (weak)
'num_leaves': {'low': 20, 'high': 80}  # Too high max
'min_data_in_leaf': {'low': 20, 'high': 100}  # Too low min
'learning_rate': {'low': 0.01, 'high': 0.05}  # Too fast
'feature_fraction': {'low': 0.6, 'high': 0.9}  # Too high

# AFTER (lines 413-437):
'lambda_l1': {'low': 0.5, 'high': 20.0}  # FORCED min 0.5 (5x stronger)
'lambda_l2': {'low': 0.5, 'high': 20.0}  # FORCED min 0.5 (5x stronger)
'num_leaves': {'low': 15, 'high': 50}  # Reduced max (simpler trees)
'max_depth': {'low': 3, 'high': 7}  # Reduced max depth
'min_data_in_leaf': {'low': 30, 'high': 150}  # Increased min (50% more)
'learning_rate': {'low': 0.005, 'high': 0.03}  # Slower (2x slower)
'min_gain_to_split': {'low': 0.2, 'high': 2.0}  # Prevent weak splits
'feature_fraction': {'low': 0.5, 'high': 0.8}  # More aggressive subsampling
'bagging_fraction': {'low': 0.5, 'high': 0.8}  # More aggressive subsampling
'bagging_freq': {'low': 1, 'high': 10}  # Added bagging frequency
```

**Impact:**
- ✅ **Mandatory L1/L2 regularization** (min 0.5, was 0.1)
- ✅ **Simpler models** (max 50 leaves, was 80)
- ✅ **More evidence required** (min 30 samples/leaf, was 20)
- ✅ **Slower learning** (max 0.03 LR, was 0.05)
- ✅ **Aggressive subsampling** (50-80%, was 60-90%)
- ✅ **Cannot turn off regularization** - always > 0

---

### **Fix 5: Use Pre-Selected Features in train()**

**File Modified:**
- `src/tactician/sr_levels/ml_quality/sr_quality_model.py` - `train()` method

**Changes:**
```python
# NEW CODE (lines 140-150):
# If feature_names already set by train_with_hpo(), use those
if self.feature_names is not None:
    self.logger.info(f"   ℹ️  Using pre-selected features from HPO: {len(self.feature_names)}")
    feature_cols = self.feature_names
else:
    # Original feature preparation logic
    exclude_cols = [...]
    feature_cols = [c for c in training_data.columns if c not in exclude_cols...]
    self.feature_names = feature_cols
```

**Impact:**
- ✅ train() now respects feature selection from train_with_hpo()
- ✅ Ensures consistent feature set throughout training
- ✅ Prevents re-adding constant features removed earlier

---

## 📊 **Expected Results After Fixes**

### **Sample Size:**
```
BEFORE: 255 samples (2 years, weekly)
AFTER:  1500-2000+ samples (2 years, daily) ← 6-8x increase!
```

### **Features:**
```
BEFORE: 89 features (12 constant, 18 weak correlation)
AFTER:  50 features (all non-constant, top by importance)
```

### **Sample-to-Feature Ratio:**
```
BEFORE: 255/89 = 2.9 samples/feature (TERRIBLE!)
AFTER:  1500/50 = 30 samples/feature (EXCELLENT!)
```

### **Model Predictions:**
```
BEFORE:
  All predictions = 0.742 (constant)
  Variance = 0.000
  Separation = 0.000
  Spearman ρ = nan

AFTER (Expected):
  Predictions: 0.3 to 0.9 range (varied)
  Variance > 0.05
  Separation > 0.15 (strong-weak)
  Spearman ρ > 0.50
```

### **Ranking Metrics:**
```
Target Metrics:
  ✅ Precision@10: >75% (already 100%)
  ✅ Spearman ρ: >0.60 (from nan)
  ✅ Separation: >0.25 (from 0.000)
  ✅ Future R²: >0.0 (from -0.399)
```

---

## 🎯 **Summary of All Changes**

### **3 Files Modified:**

1. **`scripts/run_sr_workflow.py`**
   - Changed default `ml_sample_freq_days` from 7 → 1
   - Updated help text to reflect daily sampling

2. **`src/tactician/sr_levels/ml_quality/sr_quality_model.py` - `train_with_hpo()`**
   - Added constant feature removal (std ≤ 0.001)
   - Added top-50 feature selection using LGBM importance
   - Strengthened regularization constraints (min 0.5 → 20.0)
   - Reduced model complexity (max leaves 50, max depth 7)
   - Slower learning rate (0.005-0.03)
   - Aggressive subsampling (50-80%)

3. **`src/tactician/sr_levels/ml_quality/sr_quality_model.py` - `train()`**
   - Respects pre-selected features from `train_with_hpo()`
   - Prevents re-adding removed constant features

---

## 🚀 **Next Steps**

Run the complete workflow with all fixes:

```bash
# Retrain with daily sampling + all fixes
python3 scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m --lookback-days 730

# Validate the new model
python3 scripts/validate_sr_ranking_metrics.py --symbol ETHUSDT --timeframe 15m
```

**Expected improvements:**
1. ✅ **6-8x more training samples** (255 → 1500-2000)
2. ✅ **Model will predict varied values** (not constant 0.742)
3. ✅ **Better separation** between strong and weak levels
4. ✅ **Spearman ρ > 0.50** (actual ranking correlation)
5. ✅ **Future R² > 0** (can generalize to new data)

---

## 🔑 **Key Principles Applied**

1. **More Data > Complex Models** - Daily sampling for more samples
2. **Remove Noise** - Constant features provide zero information
3. **Feature Selection** - Keep only what matters (top 50)
4. **Mandatory Regularization** - Force L1/L2 > 0.5, can't disable
5. **Simpler Models** - Fewer leaves, shallower trees
6. **Aggressive Dropout** - 50-80% subsampling prevents memorization

---

**Ready to test!** All fixes are in place. The model should now learn meaningful patterns instead of collapsing to the mean.

