# Fixes Applied and Remaining Issues

## ✅ Fixes Applied

### 1. Fixed Tactician Target Loading in Analyst Mode
**File**: `src/training/steps/model_training/unified_models_training_step.py`
**Lines**: 1911-1932
**Fix**: Only load tactician targets when `training_type` contains "tactician"
**Status**: ✅ WORKING

### 2. Fixed labeled_data Loading in Feature Selection  
**File**: `src/training/steps/pre_training/feature_generation_final_feature_selection_step.py`
**Lines**: 223-250
**Fix**: Load FULL labeled_data (173,434 rows) from versioned artifacts instead of 300-row subset
**Status**: ✅ PARTIALLY WORKING - loads 173,434 rows but output is still 300 rows

### 3. Fixed Syntax Error in Interaction Generation
**File**: `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`
**Lines**: 4297-4302
**Fix**: Moved comments inside else block to fix syntax error
**Status**: ✅ WORKING

## 🔴 Remaining Issues

### Issue 1: Feature Selection Still Outputs 300 Rows
**Current State**:
- ✅ labeled_data loads: 173,434 rows
- ✅ generated_features loads: 16,201 rows  
- ❌ Final output: 300 rows (May 30 - Aug 31)

**Root Cause**: The `_combine_features` or `_perform_multi_size_selection` method is reducing the data from 16,201 rows to 300 rows.

**Likely Culprit**: 
- Execution mode filtering being applied AGAIN after data is already filtered
- Light mode sampling being applied in blank mode
- Index intersection reducing data unexpectedly

**Next Steps**:
1. Add logging to `_combine_features` to see where rows are lost
2. Check if execution mode is being misidentified
3. Verify no additional sampling is applied

### Issue 2: Temporal Filtering Alignment
**Problem**: Training applies different temporal filters than feature selection
**Impact**: Even if we fix the 300-row issue, training may still use different data ranges
**Fix Needed**: Ensure consistent temporal windowing across all steps

### Issue 3: LightGBM/DepthwiseCNN Don't Generate Predictions
**Problem**: Only CatBoost generates predictions
**Root Cause**: Likely due to insufficient data (60 samples) or data quality issues
**Expected Resolution**: Should resolve once data size issues are fixed

## 📊 Data Flow Summary

| Step | Expected Rows | Actual Rows | Status |
|------|---------------|-------------|--------|
| Data Validation | 173,434 | 173,434 | ✅ |
| Labeling | 173,434 | 173,434 | ✅ |
| Feature Generation | ~17,280 (180 days) | 16,201 | ✅ (93.8%) |
| Feature Selection Input | 16,201 | 173,434 (labeled) + 16,201 (features) | ✅ |
| Feature Selection Output | 16,201 | **300** | ❌ |
| Training | ~1,920 (20 days) | 60 | ❌ |

## 🎯 Priority Actions

1. **HIGH**: Debug `_combine_features` to find where 16,201 → 300 reduction happens
2. **HIGH**: Add data validation checks at each step to catch reductions early
3. **MEDIUM**: Fix temporal filtering alignment
4. **LOW**: Investigate LightGBM/DepthwiseCNN (should auto-resolve with data fixes)

## 💡 Hypothesis

The 300-row output (May 30 - Aug 31) suggests:
- This is exactly 3 months of data
- Likely a "light mode" or "sample" configuration being applied
- The date range (May-Aug) doesn't match the expected range (last 180 days from Oct 31)
- This suggests an OLD cached artifact or incorrect execution mode detection

**Recommendation**: Check if the feature selection step is:
1. Loading an old cached 300-row artifact instead of generating new one
2. Misidentifying execution mode as "light" instead of "blank"
3. Applying additional sampling after combining features
