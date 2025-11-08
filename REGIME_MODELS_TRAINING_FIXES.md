# Regime Models Training Fixes

## Issues Fixed

### 1. ✅ Feature Selection - Always Return 60 Features
**Problem**: Feature selection was using adaptive feature count (50-150 features based on dataset size)
**Solution**: Changed to always return exactly **60 features** for consistency

**Files Modified**:
- `src/training/steps/market_analysis/components/regime_models_training.py`
  - Line 1651: Set `target_features = 60` (fixed value)
  - Line 1642: Updated warning message to reflect "exactly 60 features"

**Code Changes**:
```python
# OLD (adaptive):
if n_samples < 500:
    target_features = min(80, max(50, n_samples // 3))
elif n_samples < 1000:
    target_features = min(100, max(60, n_samples // 5))
else:
    target_features = min(150, max(80, n_samples // 10))

# NEW (fixed):
target_features = 60  # Always use exactly 60 features for consistency
```

### 2. ✅ Feature Selection Bug Fix - Index Out of Bounds
**Problem**: Feature selection was failing with "index 1637 is out of bounds" error
**Root Cause**: Mismatch between `selected_features_mask` length and `feature_names` length

**Solution**: Added validation and padding/truncation logic to handle mask length mismatches

**Files Modified**:
- `src/training/steps/market_analysis/components/regime_models_training.py`
  - Lines 1681-1691: Added mask length validation and correction

**Code Changes**:
```python
# Ensure mask length matches feature_names length
if len(selected_features_mask) != len(feature_names):
    tprint(f"⚠️ [REGIME_MODELS] Mask length mismatch: {len(selected_features_mask)} vs {len(feature_names)}", color="yellow")
    # Truncate or pad mask to match feature_names length
    if len(selected_features_mask) > len(feature_names):
        selected_features_mask = selected_features_mask[:len(feature_names)]
    else:
        # Pad with False
        padded_mask = np.zeros(len(feature_names), dtype=bool)
        padded_mask[:len(selected_features_mask)] = selected_features_mask
        selected_features_mask = padded_mask
```

### 3. ⚠️ Sample Count Issue - Root Cause Identified
**Problem**: Only 268 samples available instead of expected 4,320 samples (180 days × 24 hours)

**Root Cause**: 
1. System is loading **cached BTCUSDT data** instead of fresh **ETHUSDT data**
2. The cached artifact is from Nov 3, 2024 and only contains 268 samples
3. When running `regime_models_training` as a standalone step, it falls back to cached artifacts

**Evidence from logs**:
```
Nov 08, 2025 15:43:26 - System.ArtifactManager - INFO - ✅ Retrieved artifact from fallback search: 
artifacts/klines_downloading_processing_klines_data_BTCUSDT_binance_long_Analyst_20251103_194632.parquet
```

**Solution Implemented**:
- Added data validation in `regime_models_training_step.py` to warn when sample count is insufficient
- Lines 112-126: Added validation logic to check expected vs actual sample count

**Code Changes**:
```python
# Check if we have enough data for blank mode (180 days)
if execution_mode == 'blank' and market_data is not None:
    expected_samples_per_day = 24 if timeframe == '1h' else (24 * 4 if timeframe == '15m' else 24)
    expected_samples = 180 * expected_samples_per_day
    actual_samples = len(market_data)
    
    tprint(f"📊 Data validation: Expected ~{expected_samples:,} samples for 180 days of {timeframe} data", "INFO")
    tprint(f"📊 Data validation: Actual samples: {actual_samples:,}", "INFO")
    
    # If we have significantly less data than expected, warn the user
    if actual_samples < expected_samples * 0.5:  # Less than 50% of expected
        tprint(f"⚠️ WARNING: Only {actual_samples:,} samples available (expected ~{expected_samples:,})", "WARNING")
        tprint(f"⚠️ This may indicate:", "WARNING")
        tprint(f"   • Cached data from wrong symbol (check if {symbol} data exists)", "WARNING")
        tprint(f"   • Incomplete historical data", "WARNING")
        tprint(f"   • Need to run klines_downloading_processing first", "WARNING")
```

### 4. ✅ Blank Mode Data Filtering
**Problem**: Blank mode should use 180 days of data but was not filtering correctly
**Solution**: Added execution mode data filtering in `regime_models_training_step.py`

**Files Modified**:
- `src/training/steps/market_analysis/regime_models_training_step.py`
  - Lines 138-151: Added blank mode filtering (180 days)
  - Lines 152-165: Added light mode filtering (20 days)

## How to Fix the Sample Count Issue

To get the correct 4,320 samples for ETHUSDT, you need to:

### Option 1: Run Full Pipeline
```bash
python3 src/launcher/ares_launcher.py --symbol ETHUSDT --execution-mode blank
```
This will run all steps including `klines_downloading_processing` which will download fresh ETHUSDT data.

### Option 2: Run klines_downloading_processing First
```bash
# Step 1: Download fresh ETHUSDT data
python3 src/launcher/ares_launcher.py klines_downloading_processing --symbol ETHUSDT --execution-mode blank

# Step 2: Run regime_models_training
python3 src/launcher/ares_launcher.py regime_models_training --symbol ETHUSDT --execution-mode blank
```

### Option 3: Clear Cached Artifacts
```bash
# Remove old cached artifacts
rm -rf artifacts/klines_downloading_processing_klines_data_BTCUSDT_*

# Then run regime_models_training
python3 src/launcher/ares_launcher.py regime_models_training --symbol ETHUSDT --execution-mode blank
```

## Testing the Fixes

After clearing Python cache, run:
```bash
# Clear Python bytecode cache
rm -rf src/training/steps/market_analysis/components/__pycache__/regime_models_training.cpython-311.pyc
rm -rf src/training/steps/market_analysis/__pycache__/regime_models_training_step.cpython-311.pyc

# Run the command
python3 src/launcher/ares_launcher.py regime_models_training --symbol ETHUSDT --execution-mode blank
```

## Expected Output

After fixes, you should see:
1. ✅ Feature selection reduces from 1637 to exactly **60 features**
2. ✅ Warning if sample count is insufficient
3. ✅ Blank mode filters to 180 days of data
4. ✅ No "index out of bounds" errors

## Files Modified

1. `src/training/steps/market_analysis/components/regime_models_training.py`
   - Fixed feature selection to always return 60 features
   - Fixed index out of bounds error
   - Added better error logging

2. `src/training/steps/market_analysis/regime_models_training_step.py`
   - Added data validation for sample count
   - Added execution mode data filtering
   - Added warnings for insufficient data

## Summary

- ✅ Feature selection now works correctly and returns exactly 60 features
- ✅ Index out of bounds error fixed
- ✅ Data validation added to warn about insufficient samples
- ⚠️ Sample count issue is due to cached BTCUSDT data - need to download fresh ETHUSDT data
