# Remaining Issues to Fix

## ✅ FIXED
1. **tprint color parameter** - Removed color parameters from all tprint calls in data loading logic

## ⚠️ TO INVESTIGATE

### 2. Non-finite values in regime probabilities (43,423 NaN values)
**Issue:** All regime probability columns have 43,423 non-finite values
```
regime_0_prob through regime_6_prob: 43,423 NaN values each
```

**Cause:** The regime probabilities are loaded from a previous run with different data/timeframe, so they don't align with the current ETHUSDT 1h data (43,423 rows).

**Impact:** These NaN values are expected when regime probabilities from a different run don't match the current data. The system handles this by:
- Dropping these columns during feature generation
- Or filling NaNs with appropriate values

**Action Required:** 
- Either run the regime discovery step first to generate matching regime probabilities
- Or modify the component to skip loading regime probabilities when they don't match the data dimensions

### 3. Non-finite values in basic columns (17,150 NaN values)
**Issue:** Many basic columns have ~17,150 NaN values
```
quote_volume, trades, open_time, close_time, close_return, close_log_return, 
volume_return, volume_log_return, price_range, price_range_pct, body_size, 
body_size_pct, hour, day_of_week, is_weekend
```

**Cause:** These columns are likely:
- Calculated features that have NaN values at the beginning due to rolling windows
- Or columns that weren't present in the original data and were added with NaN values

**Action Required:**
- Investigate where these columns are being added
- Ensure proper forward-filling or interpolation for calculated features
- Consider dropping rows with NaN values before training

### 4. Duplicate index labels causing reindex errors
**Issue:** Multiple warnings about "cannot reindex on an axis with duplicate labels"
```
VectorBT rolling optimizer failed: cannot reindex on an axis with duplicate labels
VectorBT volatility calculation failed: name 'rolling_std' is not defined
Failed to coerce feature output to Series: cannot reindex on an axis with duplicate labels
```

**Cause:** The DataFrame index has duplicate timestamp values, which causes:
- VectorBT operations to fail
- Pandas reindex operations to fail
- Feature generation to use fallback methods

**Root Cause:** The data loaded from `historical_data/binance/ethusdt/processed/` might have:
- Duplicate timestamps in the parquet files
- Or the join operation with regime probabilities creates duplicates

**Action Required:**
1. Check for duplicate timestamps in the loaded data:
   ```python
   duplicates = data.index.duplicated()
   if duplicates.any():
       print(f"Found {duplicates.sum()} duplicate timestamps")
       print(data[duplicates].index)
   ```

2. Remove duplicates before feature generation:
   ```python
   data = data[~data.index.duplicated(keep='first')]
   ```

3. Or investigate why the processed data has duplicates and fix the data generation step

### 5. VectorBT attribute errors
**Issue:** VectorBT indicators failing with attribute errors
```
module 'vectorbt' has no attribute 'EMA'
module 'vectorbt' has no attribute 'ADX'
```

**Cause:** VectorBT API has changed or the indicators are accessed incorrectly

**Action Required:**
- Update VectorBT indicator access to use correct API
- Or ensure fallback to pandas/numpy implementations works correctly

## Priority Order
1. **HIGH:** Fix duplicate index labels (causes multiple downstream issues)
2. **MEDIUM:** Investigate non-finite values in basic columns
3. **LOW:** Handle regime probability NaN values (expected behavior)
4. **LOW:** Update VectorBT indicator access (fallbacks are working)

## Next Steps
1. Add duplicate index check and removal in the data loading logic
2. Add logging to identify where NaN values are introduced
3. Consider running regime discovery before regime models training to get matching probabilities
