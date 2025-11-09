# Training Issues Summary - Nov 9, 2025

## Run Configuration
- **Validation Step**: `feature_generation_final_validation_step` in **blank mode** (180 days)
- **Training Step**: `train-analyst-base` in **light mode** (20 days)
- **Exit Code**: 0 (Success)
- **Report**: `reports/analyst_base/ETHUSDT_15m_long/20251109_193052/`

## ✅ What Worked
1. Training completed successfully without crashes
2. All 3 models (LightGBM, DepthwiseCNN, CatBoost) were trained
3. Models saved to artifacts directory
4. Comprehensive reports generated
5. Tactician target loading fix working (no more warnings about tactician artifacts)

## 🔴 Critical Issues

### Issue 1: Only CatBoost Generates Predictions
**Status**: CRITICAL
**Evidence**:
- Predictions artifact: `analyst_base_predictions_20251109_193052_138`
- Shape: (75, 1) - Only 1 column instead of 3
- Columns: ['catboost'] - Missing 'lightgbm' and 'depthwise_cnn'

**Impact**: LightGBM and DepthwiseCNN models train but don't generate predictions for the ensemble.

### Issue 2: Only 60 Samples Used for Training
**Status**: CRITICAL
**Evidence**:
```
logs/unified_20251109_192642.log:
📊 Adjusted LightGBM params for small dataset (60 samples): num_leaves=20, max_depth=-1, min_child_samples=6
```

**Expected**: Light mode (20 days on 15m) should give ~1,920 candles
**Actual**: Only 60 samples used

**Root Cause**: Feature-target index mismatch. The selected features and targets have different indices, resulting in only 60 overlapping rows.

### Issue 3: Data Range Mismatch
**Evidence**:
- Predictions index range: `2025-08-30 22:00:00 to 2025-08-31 22:00:00`
- This is only 1-2 days of data, not 20 days

**Root Cause**: The temporal filtering is reducing the data to a very small window.

### Issue 4: Empty Performance Metrics
**Status**: MODERATE
**Evidence**: Report shows:
- "No overall performance metrics available"
- "No training metrics available"
- "No validation metrics available"  
- "No test metrics available"
- "No HPO results available"
- "No feature importance data available"

**Impact**: Cannot evaluate model quality or compare models.

### Issue 5: Perfect R² Scores (Overfitting Warning)
**Status**: WARNING
**Evidence**:
- LightGBM: R² = 1.000000 (perfect fit)
- CatBoost: R² = 0.999926 (near perfect)
- DepthwiseCNN: R² = -0.004389 (poor fit)

**Analysis**: Perfect R² on such small data (60 samples) indicates severe overfitting. The models are memorizing the training data.

## 📊 Model Metrics Summary

| Model | R² | MAE | RMSE | Status |
|-------|-----|-----|------|--------|
| LightGBM | 1.0000 | 0.000028 | 0.000043 | ⚠️ Overfitting |
| DepthwiseCNN | -0.0044 | 0.148765 | 0.325674 | ❌ Poor fit |
| CatBoost | 0.9999 | 0.000815 | 0.002787 | ⚠️ Overfitting |

## 🔍 Root Cause Analysis

### Primary Issue: Feature-Target Index Mismatch
The fundamental problem is that the selected features and targets don't have matching indices:

1. **Feature Selection** (blank mode, 180 days):
   - Should generate ~17,280 candles (180 days × 96 candles/day on 15m)
   - Actually saved: 300 rows (May 30 - Aug 31)
   - **Problem**: Feature selection is truncating data

2. **Training** (light mode, 20 days):
   - Should use ~1,920 candles (20 days × 96 candles/day)
   - Actually uses: 60 samples (Aug 30 - Aug 31)
   - **Problem**: Temporal filtering reduces to tiny window

3. **Index Overlap**:
   - Features: 300 rows (May-Aug)
   - Targets: Different time range
   - Intersection: Only 60 rows match

### Why Only CatBoost Works
CatBoost is more robust to data issues and can handle:
- Small sample sizes
- Missing data
- Index misalignment

LightGBM and DepthwiseCNN are more sensitive and fail silently when data quality is poor.

## 🛠️ Required Fixes

### Fix 1: Ensure Feature Selection Saves Full Dataset
**Problem**: Feature selection only saves 300 rows instead of full 180 days
**Solution**: Modify `feature_generation_final_feature_selection_step.py` to:
- Load the FULL labeled_data (not truncated)
- Save ALL rows after feature selection
- Verify the saved artifact has correct row count

### Fix 2: Fix Temporal Filtering Alignment
**Problem**: Training applies different temporal filters than feature selection
**Solution**: Either:
- A) Make training use the SAME temporal window as feature selection
- B) Make feature selection apply the SAME filters as training
- C) Disable temporal filtering when using pre-selected features

### Fix 3: Investigate Why LightGBM/DepthwiseCNN Don't Generate Predictions
**Problem**: Models train but don't save predictions
**Solution**: Add error handling and logging in the prediction generation code to capture why these models fail.

### Fix 4: Add Data Validation
**Problem**: No validation that features and targets align
**Solution**: Add explicit checks:
```python
assert len(features.index.intersection(targets.index)) == len(features), \
    f"Feature-target mismatch: {len(features)} features, {len(targets)} targets, {len(common)} overlap"
```

## 📝 Recommended Next Steps

1. **Immediate**: Run feature generation + labeling + feature selection in the SAME mode (light) to ensure alignment
2. **Debug**: Add logging to show exactly what data is being loaded and filtered at each step
3. **Verify**: Check that `labeled_data` artifact has the correct number of rows for the execution mode
4. **Test**: Run training with explicit data validation to catch misalignment early

## 🎯 Success Criteria

Training is successful when:
- ✅ All 3 models generate predictions (3 columns in predictions artifact)
- ✅ Training uses correct number of samples (~1,920 for light mode, not 60)
- ✅ Performance metrics are populated in reports
- ✅ R² scores are reasonable (not perfect 1.0)
- ✅ Predictions cover the full time range (20 days, not 1-2 days)
