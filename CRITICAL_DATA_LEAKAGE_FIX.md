# 🚨 CRITICAL: Data Leakage Issue Found and Solution

## Problem Identified

### Issue #1: Missing Features in Parquet File
```
📂 nas_tas_regime_assignments_20250928_231946.parquet
📊 Shape: (960, 2)  ← Only 2 columns!
📋 Columns: ['regime_id', 'regime_prob']  ← NO FEATURES!

❌ Missing: nas_feature_*, tas_feature_* columns
```

### Issue #2: Perfect Scores = Data Leakage
```
Trial 0: value: 1.0000  🚨 PERFECT!
CV fold [1]: 1.0000     🚨 ONE FOLD PERFECT!
Trial scores: 0.9958, 0.9937, 0.9937  🚨 ALL 99%+
```

## Root Cause

**The regime_assignments parquet file is incomplete!**

1. It only has `regime_id` and `regime_prob`
2. No NAS/TAS features are saved
3. Yet HPO is training on 21 features somehow
4. This means features are being generated ELSEWHERE
5. **Temporal misalignment** is causing perfect predictions

## Where Are the 21 Features Coming From?

Based on your logs showing 21 features during HPO, they must be:

1. **Generated on-the-fly** during training (but from what data?)
2. **Loaded from a different file** (not the regime_assignments)
3. **Created incorrectly** with temporal leakage

## The Fix

### Step 1: Update Clustering Pipeline to Save Features

The `nas_tas_clustering.py` component must save features WITH regime assignments:

```python
# In NASTASClusteringComponent - where results are saved

# ❌ CURRENT (incomplete):
result_df = pd.DataFrame({
    'regime_id': final_assignments,
    'regime_prob': probabilities
}, index=market_data.index)

# ✅ FIXED (with features):
result_df = pd.DataFrame({
    'regime_id': final_assignments,
    'regime_prob': probabilities
}, index=market_data.index)

# Add NAS features
for i, feature_name in enumerate(nas_feature_names):
    result_df[f'nas_feature_{i}'] = nas_features[:, i]

# Add TAS features  
for i, feature_name in enumerate(tas_feature_names):
    result_df[f'tas_feature_{i}'] = tas_features[:, i]

# Save with features
result_df.to_parquet(output_path)
```

### Step 2: Ensure Temporal Separation

**CRITICAL**: Features at time T should predict regime at T+1, not T!

```python
# ❌ WRONG: Same timestamp
df['regime_id'] = assignments  # Current regime
features = market_features      # Features from current time
# Predicting: Given features NOW, what is regime NOW? (easy = leakage!)

# ✅ CORRECT: Forward shift
df['regime_id_current'] = assignments
df['regime_id_future'] = df['regime_id_current'].shift(-1)  # Next period
features = market_features  # Features from current time  
labels = df['regime_id_future'].dropna()  # Regime in NEXT period
features = features[:-1]  # Align with shifted labels

# Predicting: Given features NOW, what regime will we be in NEXT? (useful!)
```

### Step 3: Run Diagnostic to Verify Fix

After applying fixes:

```bash
python3 scripts/diagnose_regime_data_leakage.py
```

Expected output after fix:
```
✅ Temporal alignment correct
📊 Accuracy: 0.72 (not 0.99!)
✅ No perfect CV folds
```

## Immediate Action Required

### Option A: Find Where Features Are Added (Recommended)

```bash
# Search for where nas_feature_ columns should be created
grep -r "nas_feature_" src/training/steps/market_analysis/components/
```

### Option B: Check Alternative Data Source

Maybe features come from a different file:
```bash
# Check if there's a features file separate from assignments
ls -la data_cache/nas_tas_clustering/ETHUSDT/
```

### Option C: Re-run Clustering with Features

If the clustering pipeline doesn't save features, you need to:

1. **Find the clustering execution code** (likely in `ares_launcher.py` or a training step)
2. **Update it to save features** along with regime_id
3. **Re-run the clustering** to generate proper files

## What to Look For

### In nas_tas_clustering.py:

Search for where the final dataframe is created before saving:

```python
# Look for patterns like:
- "to_parquet"
- "save.*regime"
- "output_path"
- "result_df"
- "assignments"
```

The code should include features before saving, like:

```python
# Add all features to the dataframe before saving
for col_idx in range(nas_features.shape[1]):
    result_df[f'nas_feature_{col_idx}'] = nas_features[:, col_idx]
```

## Why Perfect Scores Indicate Leakage

On balanced data (25% each class), you should get:
- **Random guessing**: 25% accuracy
- **Good model**: 60-75% accuracy  
- **Excellent model**: 75-85% accuracy
- **99%+ accuracy**: 🚨 **IMPOSSIBLE without leakage!**

Your scores of 99%+ prove features contain the answer (regime_id itself or perfect correlation).

## Next Steps

1. ✅ **Added error message** to data_access.py (done)
2. 🔍 **Find where features should be added** to clustering pipeline
3. 🔧 **Update clustering to save features** with regime_id
4. ⏭️ **Add temporal shift** (labels = regime at T+1, features from T)
5. 🔄 **Re-run clustering** to generate proper parquet files
6. ✅ **Verify with diagnostic** script

---

**The parquet file is missing features - this is why you can't train properly!** Find where `nas_tas_regime_assignments*.parquet` is created and ensure features are included.

