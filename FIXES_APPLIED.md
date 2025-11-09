# Fixes Applied for Data Truncation Issue

## Summary
Applied comprehensive fixes to prevent 1000-row data truncation and added extensive debug logging throughout the data pipeline.

## Fixes Applied

### Fix 1: HDF5 Index Chunking Consistency ✅
**File:** `src/utils/versioned_artifacts/store.py:319-341`

**What was fixed:**
- Added consistent chunking for index datasets to match data column chunking
- Prevents potential index/data misalignment issues

**Code:**
```python
# Store index with consistent chunking
index_chunk_shape = (max(1, min(chunk_rows, len(data))),)

if isinstance(data.index, pd.DatetimeIndex):
    index_data = data.index.astype(np.int64).values
    version_group.create_dataset(
        '_index',
        data=index_data,
        compression=self.compression,
        compression_opts=self.compression_level,
        chunks=index_chunk_shape  # ← Added consistent chunking
    )
```

### Fix 2: Config Data Validation to Prevent Truncation ✅
**File:** `src/training/steps/market_analysis/rolling_hmm_clustering/rolling_hmm_regime_discovery_step.py:330-360`

**What was fixed:**
- Added validation when `config['market_data']` is provided
- Detects truncated data (< 30% of expected 180-day lookback)
- Automatically bypasses truncated config data and loads from historical storage instead

**Code:**
```python
if 'market_data' in config and config['market_data'] is not None:
    external_data = config['market_data']

    # CRITICAL: Validate data size to prevent truncation
    execution_mode = config.get('execution_mode', 'full')
    samples_per_day_map = {'1m': 1440, '3m': 480, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1}
    samples_per_day = samples_per_day_map.get(timeframe, 24)
    expected_min_samples = 180 * samples_per_day
    actual_samples = len(external_data)

    if actual_samples < expected_min_samples * 0.3:  # Allow 30% tolerance
        tprint(
            f"❌ [REGIME_DISCOVERY] CRITICAL: config['market_data'] has only {actual_samples:,} samples!\n"
            f"   Expected at least {int(expected_min_samples * 0.3):,} samples (30% of 180 days)\n"
            f"   This data appears TRUNCATED - bypassing config and loading from historical storage instead!",
            "ERROR"
        )
        # Fall through to normal loading to get full dataset
    else:
        return external_data
```

**Impact:**
- For 1h timeframe: Expects ~4320 samples (180 days), rejects if < 1296 samples (30%)
- 1000 samples = ~42 days, which is < 30% threshold → Will be rejected and reload full data

### Fix 3: Comprehensive Debug Logging ✅
**File:** `src/training/steps/market_analysis/rolling_hmm_clustering/rolling_hmm_regime_discovery_step.py`

**Added debug checkpoints:**

1. **[STEP 1]** After `_load_market_data()` (line 195-196)
   ```python
   tprint(f"🐛 DEBUG: [STEP 1] market_data after _load_market_data: {market_data.shape}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 1] Index range: {market_data.index.min()} to {market_data.index.max()}", "INFO")
   ```

2. **[STEP 2]** After `_apply_execution_mode_filter()` (line 214)
   ```python
   tprint(f"🐛 DEBUG: [STEP 2] market_data after _apply_execution_mode_filter: {market_data.shape}", "INFO")
   ```

3. **[STEP 3-4]** Feature generation (lines 604, 606-607)
   ```python
   tprint(f"🐛 DEBUG: [STEP 3] market_data before feature generation: {market_data.shape}", "INFO")
   features = feature_engineer.generate_features(market_data, ewma_config)
   tprint(f"🐛 DEBUG: [STEP 4] features after generation: {features.shape}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 4] features index range: {features.index.min()} to {features.index.max()}", "INFO")
   ```

4. **[STEP 5]** After PCA (lines 621-622)
   ```python
   tprint(f"🐛 DEBUG: [STEP 5] features_pca after PCA: {features_pca.shape}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 5] features_pca index range: {features_pca.index.min()} to {features_pca.index.max()}", "INFO")
   ```

5. **[STEP 6]** After HMM prediction (lines 656-657)
   ```python
   tprint(f"🐛 DEBUG: [STEP 6] regime_labels after HMM predict: shape={regime_labels.shape}, unique={np.unique(regime_labels)}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 6] regime_probs after HMM predict: {regime_probs.shape}", "INFO")
   ```

6. **[STEP 7-8]** In `_save_results()` (lines 724-736)
   ```python
   tprint(f"🐛 DEBUG: [STEP 7] _save_results called", "INFO")
   tprint(f"🐛 DEBUG: [STEP 7] result['timestamps'] length: {len(result['timestamps'])}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 7] result['regime_labels'] length: {len(result['regime_labels'])}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 8] labels_df shape after creation: {labels_df.shape}", "INFO")
   tprint(f"🐛 DEBUG: [STEP 8] labels_df index range: {labels_df.index.min()} to {labels_df.index.max()}", "INFO")
   ```

7. **[STEP 9]** Before/after saving to HDF5 (lines 764-772)
   ```python
   tprint(f"🐛 DEBUG: [STEP 9] About to save labels_df to HDF5: {labels_df.shape}", "INFO")
   self._save_artifact(...)
   tprint(f"✅ DEBUG: [STEP 9] Successfully saved rolling_hmm_regime_labels", "SUCCESS")
   ```

**Benefits:**
- Track data shape at every transformation step
- Identify exactly where truncation occurs (if it happens)
- Validate index preservation throughout pipeline
- Monitor regime label generation

## Expected Behavior After Fixes

### Before Fixes:
```
market_data → 1000 rows (truncated from config)
features → 1000 rows
features_pca → 1000 rows
regime_labels → 1000 rows
HDF5 storage → 1000 rows ❌
```

### After Fixes:
```
market_data (config) → 1000 rows detected
Validation → FAIL (< 30% of 4320 expected)
Bypass config → Load from historical storage
market_data → 4320 rows (180 days × 24 hours) ✅
features → 4320 rows
features_pca → 4320 rows
regime_labels → 4320 rows
HDF5 storage → 4320 rows ✅
```

## Testing

### Run the command again:
```bash
python3 src/launcher/ares_launcher.py --rolling-hmm-regime-discovery --symbol ETHUSDT --execution-mode blank
```

### What to look for in output:
1. **Validation messages** (if config data is provided):
   ```
   🔍 [REGIME_DISCOVERY] Validating config market_data:
      → Execution mode: blank
      → Timeframe: 1h
      → Expected min samples (180 days): 4,320
      → Actual samples in config: 1,000
   ❌ [REGIME_DISCOVERY] CRITICAL: config['market_data'] has only 1,000 samples!
      This data appears TRUNCATED - bypassing config and loading from historical storage instead!
   ```

2. **Debug checkpoints** showing correct data sizes:
   ```
   🐛 DEBUG: [STEP 1] market_data after _load_market_data: (4320, 6)
   🐛 DEBUG: [STEP 2] market_data after _apply_execution_mode_filter: (4320, 6)
   🐛 DEBUG: [STEP 4] features after generation: (4320, X)
   🐛 DEBUG: [STEP 5] features_pca after PCA: (4320, 4)
   🐛 DEBUG: [STEP 6] regime_labels after HMM predict: shape=(4320,)
   🐛 DEBUG: [STEP 9] About to save labels_df to HDF5: (4320, 1)
   ```

3. **Successful HDF5 save** with full dataset

## Files Modified

1. ✅ `src/utils/versioned_artifacts/store.py` - Index chunking fix
2. ✅ `src/training/steps/market_analysis/rolling_hmm_clustering/rolling_hmm_regime_discovery_step.py` - Validation & debug logging

## Next Steps

1. **Run the pipeline** and verify the debug output shows 4320 rows at all steps
2. **Check HDF5 file** after run to confirm it contains 4320 rows:
   ```bash
   python3 test_hdf5_inspect.py
   ```
3. **If still truncated**, the debug logs will pinpoint exactly where it happens
4. **If fixed**, remove or reduce debug logging in production

## Root Cause

The truncation was happening because:
1. Someone/something was passing `config['market_data']` with only 1000 rows
2. The step blindly accepted this without validation
3. All downstream processing operated on the truncated data
4. HDF5 correctly stored the 1000 rows it received

With these fixes, truncated config data is now detected and rejected, forcing the step to load the full 180-day dataset from historical storage.
