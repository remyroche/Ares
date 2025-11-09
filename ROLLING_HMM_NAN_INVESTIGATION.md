# Rolling HMM NaN Issues Investigation

## Problem Summary
When running `python3 src/launcher/ares_launcher.py --rolling-hmm-regime-discovery --symbol ETHUSDT --execution-mode full`, we encounter two issues:

1. **All-NaN columns**: 11 columns become all-NaN for EWMA 8+16 configuration
   - `volatility_16`, `volatility_ratio_8+16`, `log_volatility_16`
   - `sma_16`, `sma_diff_8+16`, `price_to_sma_16`, `sma_slope_16`
   - `rolling_sharpe_8`, `rolling_zscore_8`
   - `avg_volume_16`

2. **Excessive NaN rows**: ~50% of rows (16,808 out of 33,898) contain NaN values

## Data Context
- **Dataset**: 3 years of ETHUSDT 1h data (~33,898 rows)
- **EWMA configs**: 8+16, 8+20, 8+24, 12+16, 12+20, 12+24
- **Issue**: Only affects EWMA 8+16 (and likely 12+16, 12+20, 12+24)

## Fixes Applied

### 1. Infinity Handling (✅ WORKING)
**Files Modified**:
- `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`
- `src/features_common/transforms/scaling_normalization.py`

**Changes**:
- Added infinity detection and replacement before PCA
- Added infinity detection and replacement before scaling
- Infinities are replaced with finite min/max values

**Result**: Infinity warnings are now showing, indicating the fix is working.

### 2. Returns Calculation Fix (✅ APPLIED)
**File**: `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`

**Problem**: Using `np.where()` to create returns was creating numpy arrays instead of pandas Series, breaking rolling operations.

**Fix**: Changed from:
```python
returns = np.where(close_shifted > 0, np.log(close / close_shifted), 0.0)
returns = pd.Series(returns, index=close.index)
```

To:
```python
returns = np.log(close / close.shift(1))
returns = returns.fillna(0.0)
```

**Result**: Maintains pandas Series type throughout, ensuring rolling operations work correctly.

### 3. Aggressive min_periods (✅ APPLIED)
**File**: `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`

**Changes**:
- Set `min_periods=1` for all `.mean()` operations
- Set `min_periods=2` for all `.std()` operations
- Set `min_periods=1` for all `.ewm()` operations

**Rationale**: With 33,898 rows, even window=24 should produce values starting from row 24 with min_periods=2.

### 4. Adaptive Normalization Window (✅ APPLIED)
**File**: `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`

**Changes**:
- Made rolling normalization window adaptive based on data size
- Window = `min(config.rolling_normalize_window, max(20, data_size // 10))`
- For 33,898 rows: window = min(100, max(20, 3389)) = 100

**Result**: Should reduce NaN rows from normalization step.

## Current Status: Issue Persists ❌

Despite all fixes, the warnings still appear:
```
[2025-11-09 01:31:55.436] WARNING: ⚠️  EWMA 8+16: Dropping 11 all-NaN columns: volatility_16, ...
[2025-11-09 01:31:55.656] WARNING: ⚠️  EWMA 8+16: Found 16808 NaN rows out of 33898
```

## Root Cause Analysis

### Why These Specific Features?
The all-NaN columns are specifically for **long window** features (16, 20, 24):
- `volatility_16` uses `rolling(16).std()` on returns
- `sma_16` uses `rolling(16).mean()` on close prices
- `avg_volume_16` uses `rolling(16).mean()` on volume

### Hypothesis
The issue is NOT in the feature generation itself, but in how the features are being **combined into a DataFrame** or **processed during normalization**.

Evidence:
1. Infinity warnings appear AFTER feature generation (from scaling step)
2. The features use standard pandas operations that should work
3. Only affects certain EWMA configs (8+16, not 8+24)

### Next Steps to Investigate

1. **Check if features are NaN before DataFrame construction**:
   - Add logging in `_generate_features_internal` to check each feature Series before returning
   - Verify that `features[f'volatility_{ewma_config.long_window}']` is not all-NaN when assigned

2. **Check DataFrame construction**:
   - The dict-to-DataFrame conversion might be causing issues
   - Check if index alignment is correct

3. **Check normalization process**:
   - The `_normalize_features` method might be creating NaNs
   - The rolling normalization with window=100 might still be too aggressive

4. **Verify rolling operations are actually working**:
   - Add debug logging to print first 50 values of `vol_long` after calculation
   - Check if `returns` Series is valid before rolling operations

## Recommended Fix

Add comprehensive debug logging to trace where NaNs are introduced:

```python
# In _generate_volatility_features, after calculating vol_long:
tprint_debug(f"vol_long stats: min={vol_long.min()}, max={vol_long.max()}, "
             f"nan_count={vol_long.isna().sum()}, total={len(vol_long)}")

# In _generate_features_internal, before returning:
for key, series in features.items():
    if series.isna().all():
        tprint_warning(f"Feature {key} is all-NaN BEFORE DataFrame construction!")
```

This will help identify exactly where the NaN values are being introduced.
