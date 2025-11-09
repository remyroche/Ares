# Rolling HMM NaN Issues - RESOLVED ✅

## Problem Summary
When running the rolling HMM regime discovery, we encountered:
1. **11 all-NaN columns** for certain EWMA configurations (8+16, 8+20, 8+24, etc.)
2. **~50% NaN rows** (16,808 out of 33,898)

## Root Cause Identified
The batch rolling optimizer (`ConsolidatedRollingOptimizer`) was returning `None` for certain operations. The `_ensure_series()` method was converting these `None` values into Series filled with `np.nan`, resulting in all-NaN feature columns.

**Evidence**:
```python
@staticmethod
def _ensure_series(value: Any, index: pd.Index, name: str) -> pd.Series:
    if value is None:
        series = pd.Series(np.nan, index=index, name=name)  # ← Creates all-NaN Series!
```

Debug logging confirmed that features like `volatility_16`, `sma_16`, `rolling_sharpe_8`, etc. were **all-NaN before DataFrame construction**, indicating the issue was in the feature generation itself, not in normalization or DataFrame construction.

## Solution Applied
**Disabled the batch rolling optimizer** and reverted to standard pandas operations:

**File**: `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`

```python
# DISABLED: Batch optimizer returns None for some operations, causing all-NaN features
# if config.enable_vectorbt_optimization:
#     tprint_info("🚀 Initializing VectorBT optimizers")
#     self.rolling_optimizer = ConsolidatedRollingOptimizer(...)
# else:
self.rolling_optimizer = None
self.statistical_optimizer = None
tprint_info("ℹ️  Using standard pandas operations (batch optimizer disabled)")
```

## Additional Fixes Applied

### 1. Returns Calculation Fix
**Problem**: Using `np.where()` created numpy arrays instead of pandas Series.

**Fix**: Use native pandas operations:
```python
# Before:
returns = np.where(close_shifted > 0, np.log(close / close_shifted), 0.0)
returns = pd.Series(returns, index=close.index)

# After:
returns = np.log(close / close.shift(1))
returns = returns.fillna(0.0)
```

### 2. Aggressive min_periods
Set more permissive `min_periods` to ensure features are generated:
- `min_periods=1` for all `.mean()` operations
- `min_periods=2` for all `.std()` operations  
- `min_periods=1` for all `.ewm()` operations

### 3. Adaptive Normalization Window
Made rolling normalization window adaptive based on data size:
```python
adaptive_window = min(config.rolling_normalize_window, max(20, data_size // 10))
```

### 4. Infinity Handling
Added infinity detection and replacement before PCA and scaling:
- Detect infinity values with `np.isinf()`
- Replace with finite min/max values
- Log warnings for transparency

## Results ✅

After applying the fix:
- ✅ **NO all-NaN column warnings**
- ✅ **NO excessive NaN row warnings**
- ✅ **PCA input has no NaNs across 33,898 rows**
- ✅ **HMM models training successfully**
- ✅ **Quality assessment working correctly**
- ✅ **Script running to completion**

## Performance Impact

**Trade-off**: Standard pandas operations are slower than the batch optimizer, but they are:
- ✅ **More reliable** - No None returns
- ✅ **More predictable** - Standard pandas behavior
- ✅ **Easier to debug** - Clear execution path
- ✅ **Sufficient for the use case** - 3 years of 1h data processes in reasonable time

## Files Modified

1. **src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py**
   - Disabled batch rolling optimizer
   - Fixed returns calculation
   - Added aggressive min_periods
   - Added adaptive normalization window
   - Added infinity handling
   - Added comprehensive debug logging

2. **src/features_common/transforms/scaling_normalization.py**
   - Added infinity handling before scaling

## Verification

Run the following command to verify the fix:
```bash
python3 src/launcher/ares_launcher.py --rolling-hmm-regime-discovery --symbol ETHUSDT --execution-mode full
```

Expected output:
- No warnings about "Dropping X all-NaN columns"
- No warnings about "Found X NaN rows" (or minimal NaN rows from edge cases)
- Successful HMM training and quality assessment
- Script completes without errors

## Future Improvements

1. **Fix the batch optimizer**: Investigate why `ConsolidatedRollingOptimizer` returns `None` for certain operations
2. **Add unit tests**: Test feature generation with various EWMA configurations
3. **Monitor performance**: Compare execution time with/without batch optimizer
4. **Consider alternatives**: Evaluate other optimization libraries (e.g., numba, cython)

## Conclusion

The issue was successfully resolved by identifying that the batch rolling optimizer was the root cause of all-NaN features. By disabling it and using standard pandas operations, all features are now generated correctly, and the rolling HMM regime discovery process completes successfully.
