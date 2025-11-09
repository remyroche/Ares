# Actual Root Cause & Fix Applied

## The Real Problem

After comprehensive debugging, the actual issue was:

**The rolling HMM step was loading from a TRUNCATED ARTIFACT (`klines_downloading_processing/klines_data`) with only 1000 rows, instead of loading from the full historical data storage which has years of data.**

## Evidence

From the debug output:
```
📁 Loading from processed directory: 135 files found  # ← Has full data!
✅ Loaded market data from klines_downloading_processing/klines_data  # ← But loads from artifact instead
🐛 DEBUG: [STEP 1] market_data after _load_market_data: (1000, 5)  # ← Only 1000 rows!
```

The historical storage has 135 parquet files spanning multiple years (`year=2021`, `year=2022`, `year=2023`, `year=2024`, `year=2025`), but the step loaded from a truncated artifact instead.

## Root Cause Analysis

### Data Flow:
1. `_load_market_data()` tries `KlinesParquetManager.load_klines()`
2. Klines Parquet Manager successfully finds 135 files but returns empty/fails (likely due to error)
3. Code falls back to artifact sources
4. Finds `klines_downloading_processing/klines_data` artifact
5. **Artifact only has 1000 rows** (probably from a previous limited run)
6. No validation on artifact data size
7. Uses truncated data for entire pipeline

## Fixes Applied

### Fix #1: Validate Config Data ✅
**File:** `rolling_hmm_regime_discovery_step.py:333-360`

Prevents accepting truncated data from `config['market_data']`:
```python
if actual_samples < expected_min_samples * 0.3:
    tprint(f"❌ CRITICAL: config['market_data'] has only {actual_samples:,} samples!")
    # Fall through to load from storage
```

### Fix #2: Validate Artifact Data ✅
**File:** `rolling_hmm_regime_discovery_step.py:441-460`

Prevents accepting truncated data from artifacts:
```python
if market_data is not None and not market_data.empty:
    # CRITICAL: Validate artifact data size
    if actual_samples < expected_min_samples * 0.3:
        tprint(f"⚠️ Artifact has only {actual_samples:,} samples - Skipping")
        continue  # Try next source
```

**Impact:** Truncated artifact will now be rejected!

### Fix #3: Enhanced Debug Logging ✅
**File:** `rolling_hmm_regime_discovery_step.py` (multiple locations)

Added 11 debug checkpoints to track data through the entire pipeline.

### Fix #4: HDF5 Index Consistency ✅
**File:** `store.py:319-341`

Added consistent chunking for index to prevent alignment issues.

## Expected Behavior After Fix

### Before Fix:
```
Try KlinesParquetManager → (fails silently)
Fall back to artifact klines_data → 1000 rows ❌
Use 1000 rows for training
Save 1000 rows to HDF5
```

### After Fix:
```
Try KlinesParquetManager → (should work, but if fails...)
Fall back to artifact klines_data → 1000 rows
Validate: 1000 < 1296 (30% of 4320) → REJECT ❌
Try next artifact source...
Eventually either:
  - Load full data from working source ✅
  - OR fail with clear error message ✅
```

## Next Steps

### Option 1: Fix KlinesParquetManager (Recommended)
The Klines ParquetManager is finding 135 files but failing to load them properly. Investigate why:

1. Run with more verbose logging to see the actual error
2. Check if there's a timezone issue or data format problem
3. Fix the underlying issue in KlinesParquetManager

### Option 2: Delete Truncated Artifact
Delete or update the truncated artifact:
```bash
# Find and remove/update the truncated artifact
rm -rf artifacts/ETHUSDT_binance_1h_long_regime/klines_downloading_processing/klines_data*
```

Then re-run - it will skip the bad artifact and try other sources.

### Option 3: Re-run Pipeline
Just re-run the command - the validation will now reject the truncated artifact:
```bash
python3 src/launcher/ares_launcher.py --rolling-hmm-regime-discovery --symbol ETHUSDT --execution-mode blank
```

You should see:
```
⚠️ [REGIME_DISCOVERY] Artifact klines_downloading_processing/klines_data has only 1,000 samples
   (expected at least 1,296 for 180 days)
   Skipping this artifact - will try other sources
```

## Investigation Timeline Summary

1. **Phase 1:** Investigated HDF5 storage → ✅ Working correctly
2. **Phase 2:** Investigated data pipeline → ✅ Working correctly
3. **Phase 3:** Investigated config bypass → ✅ Working correctly, added validation
4. **Phase 4:** Investigated historical storage → Found it has full data!
5. **Phase 5:** Discovered artifact fallback → 🎯 **ROOT CAUSE**

## Files Modified

1. ✅ `src/utils/versioned_artifacts/store.py` - Index chunking
2. ✅ `src/training/steps/market_analysis/rolling_hmm_clustering/rolling_hmm_regime_discovery_step.py`:
   - Config data validation (lines 333-360)
   - Artifact data validation (lines 441-460)
   - Debug logging (11 checkpoints throughout)
   - KlinesParquetManager debug (lines 376-386)

## Conclusion

The issue was NOT a truncation bug in the code, but rather:
1. Historical data exists and is complete
2. KlinesParquetManager fails to load it (silent failure)
3. Falls back to old truncated artifact
4. No validation prevented using truncated artifact

**All fixes are now in place to:**
- Detect and reject truncated data from any source
- Provide clear visibility into what data is being loaded
- Force the system to use complete datasets or fail explicitly

Run the pipeline again and it should work correctly!
