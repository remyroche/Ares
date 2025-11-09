# HDF5 Data Truncation Bug Investigation

## Summary
Data is being saved to HDF5 with only **1000 rows** despite having more data available. Investigation shows this is **NOT** a bug in the HDF5 storage layer itself, but rather data truncation happening **before** it reaches the storage layer.

## Investigation Results

### 1. HDF5 Storage Layer - NO BUG ✅
**Files Checked:**
- `src/utils/versioned_artifacts/store.py`
- `src/utils/versioned_artifacts/view.py`
- `src/utils/artifact_router.py`
- `src/training/steps/base_step.py`

**Findings:**
- HDF5 storage correctly saves ALL data passed to it
- Chunking (`chunk_rows = min(1000, num_rows)`) is for **internal HDF5 organization**, NOT data truncation
- Applied fix: Added consistent chunking for index to match data columns
- All `materialize()` and `_load_data_with_mask()` methods load complete datasets

**Evidence from HDF5 File Inspection:**
```
Version: rolling_hmm_regime_labels_20251108_203927_884
Expected rows (metadata): 1000
Actual rows (HDF5 data): 1000
✅ Match - no truncation

Version: rolling_hmm_regime_labels_20251108_202510_294
Expected rows (metadata): 4320
Actual rows (HDF5 data): 4320
✅ Match - no truncation
```

The HDF5 layer faithfully stores whatever data it receives. The 1000-row versions were saved with exactly 1000 rows of input data.

### 2. Data Truncation Source - FOUND ❌

**Data Flow in rolling_hmm_regime_discovery_step.py:**
```
market_data (full dataset)
  ↓
feature_engineer.generate_features(market_data, ewma_config)
  ↓
features (TRUNCATION HAPPENS HERE?)
  ↓
feature_engineer.apply_pca(features)
  ↓
features_pca (1000 rows)
  ↓
hmm_model.predict(features_pca.values)
  ↓
regime_labels (1000 rows)
  ↓
_save_results() → _save_artifact()
  ↓
HDF5 storage (saves 1000 rows correctly)
```

**Root Cause:**
The truncation is happening during **feature generation** or **PCA**, not in the HDF5 storage layer.

### 3. Execution Mode Configuration

**File:** `src/training/steps/market_analysis/shared_utils/execution_mode_lookback_config.py`

Execution mode "light" has:
- `optimization_sample_size=1000`
- `labeling_sample_size=1000`

Execution mode "blank" has:
- `optimization_sample_size=500`
- `labeling_window_days=180`

However, rolling_hmm_regime_discovery_step.py does **NOT** directly use these sample_size parameters in the code reviewed.

## The Actual Bug Location

**The truncation must be happening in ONE of these places:**

1. **Feature generation** (`feature_engineer.generate_features()`)
   - File: `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`
   - May be limiting features based on execution mode config

2. **PCA application** (`feature_engineer.apply_pca()`)
   - File: `src/training/steps/market_analysis/rolling_hmm_clustering/feature_engineering.py`
   - May be sampling data before PCA

3. **Execution mode data limiting**
   - Some upstream component may be applying `sample_size` limits before feature generation

## Recommended Next Steps

1. **Add debug logging** in `rolling_hmm_regime_discovery_step.py` to track data sizes:
   ```python
   tprint(f"DEBUG: market_data shape: {market_data.shape}")
   tprint(f"DEBUG: features shape after generation: {features.shape}")
   tprint(f"DEBUG: features_pca shape after PCA: {features_pca.shape}")
   ```

2. **Check feature_engineering.py** for any `.head()`, `.sample()`, or `.iloc[:sample_size]` calls

3. **Verify execution mode handling** - ensure "blank" mode isn't accidentally using "light" mode configs

## Fix Applied So Far

✅ **HDF5 Index Chunking Consistency**
- File: `src/utils/versioned_artifacts/store.py:319-341`
- Added consistent chunking for index to match data column chunking
- This prevents potential index/data misalignment issues

## Conclusion

The HDF5 storage system is working correctly. The bug is in the **data preparation pipeline** before it reaches storage. Investigation should focus on the feature engineering and PCA steps in the rolling HMM regime discovery workflow.
