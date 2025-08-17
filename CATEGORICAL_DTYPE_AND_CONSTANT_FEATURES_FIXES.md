# Categorical Dtype and Constant Features Fixes

## Issues Fixed

### 1. ❌ Validation error: 'Categorical' with dtype category does not support reduction 'var'

**Problem**: The feature output validator was trying to calculate variance on categorical columns, which don't support the `.var()` operation.

**Root Cause**: The validation code was calling `.var()` on all columns without checking their data type first.

**Solution**: Added data type checks before calling `.var()`:

```python
# Before (causing error):
if series.var() == 0:
    results["warnings"].append(f"Zero variance feature detected: {col}")

# After (fixed):
if pd.api.types.is_numeric_dtype(series.dtype):
    if series.var() == 0:
        results["warnings"].append(f"Zero variance feature detected: {col}")
else:
    # For categorical columns, check if all values are the same
    if series.nunique() <= 1:
        results["warnings"].append(f"Constant categorical feature detected: {col}")
```

**Files Modified**:
- `src/utils/feature_output_validator.py` - Fixed all instances of `.var()` calls to check data types first

### 2. 248 constant features

**Problem**: Too many features had zero variance (constant values), particularly HMM cluster features.

**Root Cause**: 
1. HMM cluster features were being generated for all possible cluster IDs (0-19), even if some clusters didn't exist in the data
2. The constant feature detection was using `nunique()` which doesn't work well with categorical data

**Solution**: 
1. **Fixed HMM cluster generation** to only create intensity features for clusters that actually exist in the data
2. **Improved constant feature detection** to handle different data types properly

```python
# Before (generating features for all possible clusters):
for cluster_id in range(cluster_labels.max() + 1):
    cluster_mask = cluster_labels == cluster_id
    intensity = cluster_mask.astype(float)
    composite_df[f'intensity_cluster_{cluster_id}'] = intensity

# After (only generating features for existing clusters):
unique_clusters = sorted(cluster_labels.unique())
for cluster_id in unique_clusters:
    cluster_mask = cluster_labels == cluster_id
    intensity = cluster_mask.astype(float)
    composite_df[f'intensity_cluster_{cluster_id}'] = intensity
```

**Files Modified**:
- `src/training/steps/step3_hmm_regime_discovery.py` - Fixed HMM cluster feature generation
- `src/training/steps/step2_feature_engineering.py` - Improved constant feature detection

## Technical Details

### Categorical Dtype Handling

The validation system now properly handles different data types:

- **Numeric columns**: Use `.var()` to check for zero variance
- **Categorical columns**: Use `.nunique()` to check for constant values
- **Mixed data types**: Apply appropriate checks based on column type

### HMM Cluster Feature Optimization

- **Before**: Generated 20 intensity features (0-19) regardless of actual clusters
- **After**: Only generates intensity features for clusters that exist in the data
- **Result**: Significantly fewer constant features, better feature quality

### Constant Feature Detection

The new detection logic:

```python
# Check for constant features more intelligently
low_var_cols = []
for col in df.columns:
    series = df[col]
    if pd.api.types.is_numeric_dtype(series.dtype):
        # For numeric columns, check variance
        if series.var() == 0:
            low_var_cols.append(col)
    else:
        # For categorical columns, check unique values
        if series.nunique() <= 1:
            low_var_cols.append(col)
```

## Expected Results

1. **No more categorical dtype errors** in feature output validation
2. **Significantly fewer constant features** (should drop from 248 to a much lower number)
3. **Better feature quality** with only meaningful HMM cluster features
4. **Improved validation accuracy** with proper data type handling

## Status: ✅ FIXED

Both issues have been resolved:
- Categorical dtype validation errors eliminated
- Constant feature count should be dramatically reduced
- Feature quality improved through better HMM cluster generation
