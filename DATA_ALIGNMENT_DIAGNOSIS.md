# Data Alignment Issue - Root Cause Analysis

**Date:** 2025-11-09
**Issue:** Feature selection showing 0 stability windows and poor baseline performance
**Symptoms:** 26,274 samples reduced to 100 after alignment

---

## Critical Discovery

### The Data Loss
```
Base Features (labeled_df):     26,274 samples × 29 columns
Generated Features:               1,790 samples × 341 columns
Common Index Intersection:          100 samples (only 0.38%!)
```

### Impact
- **Stability Analysis:** 100 samples ÷ 5 windows = 20 samples/window
- **Threshold:** Need 50+ samples per window → ALL windows skipped
- **Result:** 0 time windows analyzed, 0 stable features
- **Baseline:** Too few samples for meaningful feature selection

---

## Root Cause Analysis

### 1. Labeled Data Source
- **File:** `labeled_df` loaded in step initialization
- **Samples:** 26,274 rows
- **Date Range:** Full historical dataset
- **Target Column:** `price_target_vol_normalized`
- **Columns:** 29 features (OHLCV + basic features)

### 2. Generated Features Source
- **Artifact:** `generated_features_15m_20251108_171705_707`
- **Created:** 2024-11-08 at 17:17:05
- **Samples:** 1,790 rows (only 6.8% of labeled data!)
- **Columns:** 341 advanced features
- **Store:** `ETHUSDT_binance_15m_long_analyst`

### 3. The Mismatch
```python
# Line 576-584 in feature_generation_final_feature_selection_step.py
if generated_features.shape[0] != base_features.shape[0]:
    # Shape mismatch: 26,274 vs 1,790
    common_index = base_features.index.intersection(generated_features.index)
    # Only 100 common indices found!
    generated_features = generated_features.loc[common_index]
    base_features = base_features.loc[common_index]
    # Both now reduced to 100 samples
```

---

## Why Only 100 Common Indices?

The index intersection is based on **timestamps**. The generated features artifact:
1. Was created on a **different date** (Nov 8) than the labeled data
2. Only processed **1,790 samples** instead of the full dataset
3. Likely used a different time window or filtering

When aligning by timestamp index:
- Labeled data: Has timestamps from [earliest] to [latest] (26,274 points)
- Generated features: Has only 1,790 timestamps
- Overlap: Only 100 timestamps exist in both datasets

---

## Investigation Questions

### Q1: Why Does Generated Features Have Only 1,790 Samples?

**Possible Causes:**
1. **Execution Mode Filter:** Feature generation used a restrictive execution mode
2. **Lookback Window:** Advanced features require significant lookback, reducing valid samples
3. **Data Filtering:** NaN/invalid data removal was too aggressive
4. **Date Range:** Feature generation was run on a subset of data
5. **Feature Requirements:** Complex features (vectorbt, interactions) may require minimum data

**Evidence Needed:**
- Check the feature generation step logs from Nov 8, 2024 17:17
- Review execution mode configuration at that time
- Examine lookback requirements for the 341 generated features

### Q2: Why Use This Specific Generated Features Version?

**Current Behavior:**
```python
# ArtifactManager retrieves the LATEST versioned artifact matching the pattern
artifact = artifact_manager.retrieve_artifact('generated_features')
# Returns: generated_features_15m_20251108_171705_707 (latest in metadata)
```

**Available Versions:**
- `generated_features_15m_20251108_171705_707` (1,790 samples) ← Currently used
- Multiple other versions from Nov 8, 2024
- Some may have different sample counts

**Issue:** No validation that generated features align with labeled data!

---

## Solutions

### Solution 1: Regenerate Features for Full Dataset (RECOMMENDED)

**Approach:** Run feature generation step on the FULL 26,274 samples

```bash
python3 src/launcher/ares_launcher.py \
  --feature_generation_step \
  --symbol ETHUSDT \
  --execution-mode blank \
  --force-regenerate
```

**Benefits:**
- Generates features for all 26,274 labeled samples
- Perfect alignment between labeled data and generated features
- Enables proper stability analysis (5 windows × ~5,254 samples each)
- Meaningful baseline comparison

**Requirements:**
- Ensure labeled data is loaded from correct source
- Use same date range for both labeled and generated features
- Verify sufficient memory for 26,274 × 341 feature matrix

### Solution 2: Filter Labeled Data to Match Generated Features

**Approach:** Use only the 1,790 samples that have generated features

```python
# In _combine_features()
if generated_features is not None:
    # Instead of intersection, use generated_features index as source of truth
    base_features = base_features.loc[base_features.index.intersection(generated_features.index)]
    # This gives us 100 samples - still too few!
```

**Issues:**
- ❌ Still only 100 samples (not enough for stability analysis)
- ❌ Wastes 26,174 labeled samples
- ❌ Not addressing root cause

**Verdict:** NOT RECOMMENDED

### Solution 3: Intelligent Artifact Version Selection

**Approach:** Find a generated features version that matches labeled data size

```python
# Modify artifact retrieval to prefer versions with matching sample count
def retrieve_matching_artifact(artifact_name, target_sample_count):
    """Retrieve artifact version with closest sample count to target."""
    all_versions = artifact_store.list_versions(artifact_name)

    best_match = None
    min_diff = float('inf')

    for version in all_versions:
        metadata = artifact_store.get_metadata(version)
        sample_count = metadata.get('row_count', 0)
        diff = abs(sample_count - target_sample_count)

        if diff < min_diff:
            min_diff = diff
            best_match = version

    return artifact_store.load(best_match)
```

**Benefits:**
- Automatically selects best matching version
- No need to regenerate if matching version exists
- Reduces data loss

**Drawbacks:**
- Complex implementation
- May not find perfect match
- Doesn't guarantee features are computed on same data

### Solution 4: Add Alignment Validation

**Approach:** Add validation step before feature selection

```python
def validate_feature_alignment(labeled_df, generated_features):
    """Validate that features align properly with labeled data."""

    # Check sample count match
    if generated_features.shape[0] != labeled_df.shape[0]:
        raise ValueError(
            f"Feature alignment mismatch: "
            f"labeled_df has {labeled_df.shape[0]} samples, "
            f"generated_features has {generated_features.shape[0]} samples. "
            f"Please regenerate features on the full dataset."
        )

    # Check index overlap
    common_idx = labeled_df.index.intersection(generated_features.index)
    overlap_pct = len(common_idx) / len(labeled_df) * 100

    if overlap_pct < 95:
        raise ValueError(
            f"Insufficient index overlap: {overlap_pct:.1f}% "
            f"({len(common_idx)}/{len(labeled_df)} samples). "
            f"Features and labels must share >95% of indices."
        )

    return True
```

**Benefits:**
- Prevents silent failures
- Forces regeneration when alignment is poor
- Clear error messages for debugging

---

## Recommended Action Plan

### Step 1: Verify Current State ✅ DONE
- ✅ Confirmed labeled_df: 26,274 samples
- ✅ Confirmed generated_features: 1,790 samples
- ✅ Confirmed common indices: 100 samples
- ✅ Identified artifact version: `generated_features_15m_20251108_171705_707`

### Step 2: Regenerate Features for Full Dataset
```bash
# Run feature generation on full labeled dataset
python3 src/launcher/ares_launcher.py \
  --feature_generation_step \
  --symbol ETHUSDT \
  --exchange binance \
  --timeframe 15m \
  --execution-mode blank \
  --direction long \
  --model analyst
```

**Expected Output:**
- New `generated_features_15m_YYYYMMDD_HHMMSS` with 26,274+ samples
- Perfect alignment with labeled data
- Enables proper feature selection

### Step 3: Add Alignment Validation
- Add validation function to `_combine_features()` in [feature_generation_final_feature_selection_step.py](src/training/steps/pre_training/feature_generation_final_feature_selection_step.py#L554)
- Fail fast if alignment < 95%
- Provide clear error messages

### Step 4: Re-run Feature Selection
```bash
python3 src/launcher/ares_launcher.py \
  --feature_generation_final_feature_selection_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

**Expected Results:**
- All 26,274 samples used
- 5 windows × ~5,254 samples each
- Meaningful stability analysis (>40% stable features)
- Valid baseline comparison (>1.2x improvement)

---

## Technical Details

### Index Alignment Logic
```python
# Current implementation (line 580-586)
common_index = base_features.index.intersection(generated_features.index)
if len(common_index) > 0:
    generated_features = generated_features.loc[common_index]
    base_features = base_features.loc[common_index]
```

**Issue:** Silent data loss - reduces from 26,274 to 100 samples without warning

**Proposed Fix:**
```python
# Add validation before alignment
min_overlap_pct = 95.0
common_index = base_features.index.intersection(generated_features.index)
overlap_pct = len(common_index) / len(base_features) * 100

if overlap_pct < min_overlap_pct:
    raise ValueError(
        f"Critical: Only {overlap_pct:.1f}% index overlap "
        f"({len(common_index)}/{len(base_features)} samples).\n"
        f"This indicates generated features were created on different data.\n"
        f"Required: >{min_overlap_pct}% overlap.\n\n"
        f"Action: Regenerate features using same dataset as labeled data."
    )

if len(common_index) > 0:
    generated_features = generated_features.loc[common_index]
    base_features = base_features.loc[common_index]
```

### Artifact Version Metadata

Check artifact metadata to understand what data was used:
```python
import json

# Load metadata
with open('versioned_artifacts/ETHUSDT_binance_15m_long_analyst/metadata.json') as f:
    metadata = json.load(f)

# Find our version
version_info = metadata['versions']['generated_features_15m_20251108_171705_707']
print(f"Row count: {version_info.get('row_count')}")
print(f"Created: {version_info.get('created_at')}")
print(f"Config: {version_info.get('config', {})}")
```

---

## Expected Behavior After Fix

### Before (Current State)
```
📊 Using labeled dataframe as base: (26274, 29)
⚠️ Shape mismatch: base_features (26274, 29) vs generated_features (1790, 341)
📊 Aligning dataframes using 100 common indices
📊 After alignment - base_features shape: (100, 29)
📊 Combined feature matrix: 469 features, 100 samples

Stability Analysis:
- Time Windows: 0 (all skipped - need 50+ samples per window)
- Stable Features: 0
- Average Stability: 0.0

Baseline Comparison:
- Improvement Ratio: 0.76x (regression due to tiny dataset)
```

### After Fix (Expected)
```
📊 Using labeled dataframe as base: (26274, 29)
✅ Generated features aligned: (26274, 341)
📊 Combined feature matrix: 370 features, 26274 samples

Stability Analysis:
- Time Windows: 5 (5254 samples per window)
- Stable Features: 35-45 out of 60 (58-75%)
- Average Stability: 0.65-0.75

Baseline Comparison:
- Improvement Ratio: 1.8-2.5x (selected features outperform baseline)
```

---

## Files to Investigate

1. **Feature Generation Config:**
   - [src/training/steps/pre_training/feature_generation_step.py](src/training/steps/pre_training/feature_generation_step.py)
   - Check why only 1,790 samples were generated

2. **Execution Mode Config:**
   - [src/training/steps/market_analysis/shared_utils/execution_mode_lookback_config.py](src/training/steps/market_analysis/shared_utils/execution_mode_lookback_config.py)
   - Verify "blank" mode uses full 180 days

3. **Artifact Loading:**
   - [src/training/steps/base_step.py](src/training/steps/base_step.py)
   - Check artifact retrieval logic

4. **Data Alignment:**
   - [src/training/steps/pre_training/feature_generation_final_feature_selection_step.py](src/training/steps/pre_training/feature_generation_final_feature_selection_step.py#L554-L824)
   - Add validation before alignment

---

## Logs Evidence

```
[2025-11-09 15:20:07.399] INFO: 🔍 DEBUG: labeled_df shape: (26274, 29)
[2025-11-09 15:20:07.544] INFO: ✅ Retrieved main generated features: (1790, 341)
[2025-11-09 15:20:11.699] WARNING: ⚠️ Shape mismatch: base_features (26274, 29) vs generated_features (1790, 341)
[2025-11-09 15:20:11.699] INFO: 📊 Aligning dataframes using 100 common indices
[2025-11-09 15:20:11.701] INFO: 📊 After alignment - base_features shape: (100, 29)
[2025-11-09 15:20:12.005] INFO: 📊 Combined feature matrix: 469 features, 100 samples
```

---

## Summary

**Root Cause:** Generated features artifact contains only 1,790 samples instead of 26,274, with only 100 timestamps overlapping with labeled data.

**Impact:** Feature selection operates on 100 samples instead of 26,274, making stability analysis and baseline comparison meaningless.

**Solution:** Regenerate features on the full labeled dataset to ensure perfect alignment.

**Priority:** **CRITICAL** - This blocks all downstream feature selection and model training.

---

**Next Steps:**
1. Run feature generation step on full dataset
2. Add alignment validation (fail fast on < 95% overlap)
3. Re-run feature selection with properly aligned data
4. Verify stability > 40% and baseline > 1.2x
