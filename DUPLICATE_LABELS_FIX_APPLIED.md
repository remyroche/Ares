# Duplicate Labels Fix - Complete Solution

## Problem
The `feature_generation_interaction_generation_step` was failing with:
```
ValueError: cannot reindex on an axis with duplicate labels
```

This prevented the step from completing and creating the `analyst_interaction_features` artifact.

## Root Cause
The data loaded from parquet files contained thousands of duplicate timestamps, which caused pandas reindex operations to fail during interaction discovery.

## Solution Applied

### Fix Location
**File:** `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`
**Lines:** 3051-3072

### Changes Made
Added comprehensive deduplication of indices before any reindex operations:

```python
# CRITICAL FIX: Deduplicate indices to prevent "cannot reindex on an axis with duplicate labels" error
if features.index.duplicated().any():
    n_dup = features.index.duplicated().sum()
    tprint_warning(f"  ⚠️ Features has {n_dup} duplicate indices, deduplicating...")
    features = features[~features.index.duplicated(keep='first')]
    tprint_success(f"  ✅ Deduplicated features to {len(features)} unique indices")

if targets.index.duplicated().any():
    n_dup = targets.index.duplicated().sum()
    tprint_warning(f"  ⚠️ Targets has {n_dup} duplicate indices, deduplicating...")
    targets = targets[~targets.index.duplicated(keep='first')]
    tprint_success(f"  ✅ Deduplicated targets to {len(targets)} unique indices")

# Find common indices
common_indices = features.index.intersection(targets.index)

# Deduplicate common_indices if needed
if common_indices.duplicated().any():
    n_dup = common_indices.duplicated().sum()
    tprint_warning(f"  ⚠️ Found {n_dup} duplicate indices in common_indices, removing duplicates...")
    common_indices = common_indices[~common_indices.duplicated(keep='first')]
    tprint_success(f"  ✅ Deduplicated common_indices to {len(common_indices)} unique indices")
```

### How It Works
1. **Check for duplicates** in `features.index` and `targets.index`
2. **Remove duplicates** using `~index.duplicated(keep='first')` - keeps first occurrence
3. **Deduplicate common_indices** after intersection to ensure no duplicates remain
4. **Log warnings and success** messages for transparency

### Benefits
- ✅ Prevents reindex errors
- ✅ Maintains data integrity (keeps first occurrence)
- ✅ Provides visibility into data quality issues
- ✅ Allows interaction generation to complete successfully

## Versioned Artifacts Fix (Also Applied)

### Fix Location
**File:** `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`
**Lines:** 3592-3601

### Changes Made
Added comment clarifying HDF5/versioned artifacts storage:

```python
# CRITICAL FIX: Save to versioned artifacts (HDF5) using artifact_type='data'
# This ensures the interaction features are stored in the same versioned artifacts
# store as generated_features and labeled_data, making them accessible to
# feature_generation_final_feature_selection_step via _get_artifact()
features_path = self._save_artifact(
    data=combined_features,
    artifact_name='analyst_interaction_features',
    artifact_type='data',  # This triggers HDF5/versioned artifacts storage
    metadata=enhanced_metadata
)
```

## Testing Steps

### Step 1: Run Interaction Generation
```bash
python3 src/launcher/ares_launcher.py \
  --feature_generation_interaction_generation_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

**Expected Output:**
```
⚠️ Features has XXXX duplicate indices, deduplicating...
✅ Deduplicated features to YYYY unique indices
⚠️ Targets has XXXX duplicate indices, deduplicating...
✅ Deduplicated targets to YYYY unique indices
...
✅ Saved analyst_interaction_features to versioned artifacts
```

### Step 2: Verify Artifact Creation
```bash
python3 -c "
from src.utils.versioned_artifacts.store import VersionedArtifactStore
store = VersionedArtifactStore('versioned_artifacts/ETHUSDT_binance_15m_long_analyst')
versions = store.list_versions()
interaction_versions = [v for v in versions if 'interaction' in v.lower()]
print(f'Interaction features: {len(interaction_versions)}')
for v in interaction_versions:
    print(f'  - {v}')
"
```

**Expected Output:**
```
Interaction features: 1
  - analyst_interaction_features_20251111_HHMMSS_XXX
```

### Step 3: Run Final Feature Selection
```bash
python3 src/launcher/ares_launcher.py \
  --feature_generation_final_feature_selection_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

**Expected Output:**
```
🔍 Loading interaction features from interaction generation step...
✅ Retrieved interaction features: (14023, 150+)
✅ Interaction features will be merged with generated features for selection
📊 Combining features...
✅ Combined (14023, 450+) features
```

### Step 4: Verify Selected Features Include Interactions
```bash
grep -E "_x_|_div_|_minus_|_3x_ratio|_6x_ratio" \
  outcomes/final_feature_selection_outcome_report_*.md | head -20
```

**Expected Output:**
```
15. momentum_5_x_volume_std_20
23. rsi_14_div_atr_14
31. trend_score_3x_ratio
...
```

## Status

✅ **Duplicate labels fix applied** - Deduplication prevents reindex errors
✅ **Versioned artifacts fix applied** - Saves to HDF5 store
🔄 **Testing in progress** - Running interaction generation step

## Next Steps

1. Wait for interaction generation to complete
2. Verify `analyst_interaction_features` appears in versioned artifacts
3. Run final feature selection
4. Confirm interaction features are included in selected features
