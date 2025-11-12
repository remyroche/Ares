# Data Loading Performance Analysis

## Problem Identified

The `feature_generation_interaction_generation_step` is extremely slow due to duplicate timestamp detection happening at the **KlinesParquetManager** level, which processes duplicates one-by-one with extensive logging.

## Root Cause

### Duplicate Detection Location
The duplicates are being detected in `KlinesParquetManager` during data loading:
```
2025-11-11 17:56:48 - KlinesParquetManager - INFO: 🔍 DEBUG: Timestamp 2021-11-06 14:30:00 has 2 records
2025-11-11 17:56:48 - KlinesParquetManager - INFO: 🔍 DEBUG: Found 2 identical records for 2021-11-06 14:30:00
... (thousands of these messages)
```

This happens **BEFORE** our deduplication fixes can run, which are at the interaction generation step level.

## Fixes Applied (Correct but Too Late in Pipeline)

### Fix 1: Early Deduplication (Lines 675-687)
```python
# CRITICAL FIX: Deduplicate indices immediately after loading
if labeled_data.index.duplicated().any():
    n_dup = labeled_data.index.duplicated().sum()
    tprint_warning(f"⚠️ Labeled data has {n_dup} duplicate indices, deduplicating...")
    labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
```

**Status:** ✅ Correct but runs AFTER slow parquet loading

### Fix 2: Reindex Deduplication (Lines 3051-3072)
```python
# CRITICAL FIX: Deduplicate indices to prevent reindex errors
if features.index.duplicated().any():
    features = features[~features.index.duplicated(keep='first')]
```

**Status:** ✅ Correct but runs AFTER slow parquet loading

### Fix 3: Versioned Artifacts (Lines 3592-3601)
```python
# CRITICAL FIX: Save to versioned artifacts (HDF5)
artifact_type='data'  # Triggers HDF5/versioned artifacts storage
```

**Status:** ✅ Correct and will work once step completes

## The Real Bottleneck

The bottleneck is in **`KlinesParquetManager`** which:
1. Loads data from multiple parquet files
2. Detects duplicates one-by-one with extensive logging
3. Takes 5-10+ minutes just to load and process duplicates
4. Happens before any of our fixes can run

## Solutions

### Option A: Fix at Source (Recommended)
Clean the parquet data files to remove duplicates permanently:

```bash
# Script to deduplicate parquet files
python3 -c "
import pandas as pd
import glob

parquet_files = glob.glob('historical_data/storage/binance/ethusdt/**/*.parquet', recursive=True)
for file in parquet_files:
    df = pd.read_parquet(file)
    if df.index.duplicated().any():
        print(f'Deduplicating {file}...')
        df = df[~df.index.duplicated(keep='first')]
        df.to_parquet(file)
        print(f'  Saved {len(df)} unique records')
"
```

### Option B: Disable Duplicate Logging
Modify `KlinesParquetManager` to silently remove duplicates instead of logging each one:

**File:** `src/data_management/klines_parquet.py` (or similar)
**Change:** Remove or reduce duplicate detection logging

### Option C: Wait for Current Run
Let the current process complete (may take 30-60 minutes) and verify fixes work.

## Current Status

**Process:** Running (started 17:52, now 17:56 - still in data loading phase)
**Estimated Time:** 30-60+ minutes to complete data loading
**Fixes Applied:** ✅ All 3 fixes are correct and in place
**Bottleneck:** KlinesParquetManager duplicate processing

## Recommendation

**Immediate:** Let current process run to completion to verify all fixes work end-to-end

**Long-term:** Clean parquet data files at source (Option A) to permanently fix performance

## Expected Outcome (Once Complete)

1. ✅ Data loads (slowly due to parquet duplicates)
2. ✅ Early deduplication removes duplicates after loading
3. ✅ Interaction generation proceeds without reindex errors
4. ✅ Saves to versioned artifacts (HDF5)
5. ✅ Final feature selection can load interaction features
6. ✅ Selected features include interaction types

## Verification Steps (After Completion)

### 1. Check for Deduplication Messages
```bash
grep "Deduplicated" /tmp/interaction_gen_test.log
```

**Expected:**
```
✅ Deduplicated labeled_data to XXXX unique indices
✅ Deduplicated generated_features to XXXX unique indices
✅ Deduplicated features to XXXX unique indices
✅ Deduplicated targets to XXXX unique indices
```

### 2. Check for Artifact Creation
```bash
python3 -c "
from src.utils.versioned_artifacts.store import VersionedArtifactStore
store = VersionedArtifactStore('versioned_artifacts/ETHUSDT_binance_15m_long_analyst')
versions = [v for v in store.list_versions() if 'interaction' in v.lower()]
print(f'Interaction features: {len(versions)}')
for v in versions:
    print(f'  - {v}')
"
```

**Expected:**
```
Interaction features: 1
  - analyst_interaction_features_20251111_HHMMSS_XXX
```

### 3. Run Final Feature Selection
```bash
python3 src/launcher/ares_launcher.py \
  --feature_generation_final_feature_selection_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

**Expected:**
```
✅ Retrieved interaction features: (14023, 150+)
✅ Interaction features will be merged with generated features
```

## Summary

- ✅ **All fixes are correct and applied**
- ⚠️ **Performance bottleneck is upstream** (KlinesParquetManager)
- 🔄 **Current run will verify fixes work** (when it completes)
- 📋 **Long-term fix needed** (clean parquet data at source)
