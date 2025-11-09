# Data Loss Analysis - Feature Generation Pipeline

## Executive Summary
The feature generation pipeline is losing 99.8% of the data through multiple reduction steps.

## Data Flow Through Pipeline

| Step | Output Rows | % Retained | Issue |
|------|-------------|------------|-------|
| 1. Data Validation | 173,434 | 100% | ✅ GOOD |
| 2. Labeling Integration | 173,434 | 100% | ✅ GOOD |
| 3. Feature Generation | **16,201** | **9.3%** | 🔴 **MAJOR LOSS** |
| 4. Lookback Optimization | 21,333 | 12.3% | ⚠️ Partial recovery |
| 5. Interaction Generation | 8,000 | 4.6% | ❌ FAILED + more loss |
| 6. Final Feature Selection | **300** | **0.17%** | 🔴 **CRITICAL LOSS** |

## Root Causes Identified

### Issue 1: Feature Generation Step (173,434 → 16,201 rows)
**Loss: 157,233 rows (90.7%)**

**Cause**: The feature generation step is likely:
1. Dropping rows with NaN values created by rolling window calculations
2. Using a lookback window that requires ~157k rows of history
3. Not properly handling the beginning of the time series

**Evidence**:
```
Generated features: 16,201 rows, 327 columns
Date range: Likely starts ~157k candles after the beginning
```

**Fix Required**: 
- Reduce lookback periods for blank mode
- Use forward-fill or interpolation for initial NaN values
- Configure feature generation to preserve more data

### Issue 2: Final Feature Selection (16,201 → 300 rows)
**Loss: 15,901 rows (98.1%)**

**Cause**: The `labeled_df` being loaded in feature selection is truncated to 300 rows.

**Evidence**:
```python
# In feature_generation_final_feature_selection_step.py line 223:
labeled_df = self._get_artifact('labeled_data')
# This is loading a 300-row subset, not the full 173,434 rows
```

**Why 300 rows?**
- Likely a "light mode" or "sample" version of the data
- The artifact loading is getting a preview/subset instead of full data
- Execution mode configuration may be limiting data size

**Fix Required**:
- Ensure `labeled_data` artifact loads the FULL dataset
- Check execution mode configuration for data size limits
- Verify artifact loading doesn't apply sampling

### Issue 3: Interaction Generation Failed
**Status**: Crashed during processing

**Cause**: Error in LGBM SHAP pipeline with 8,000 samples

**Impact**: Cannot generate interaction features, reducing model quality

## Expected vs Actual Data Sizes

### For Blank Mode (180 days, 15m timeframe):
- **Expected**: 180 days × 96 candles/day = **17,280 candles**
- **After Feature Gen**: 16,201 candles (93.8% - acceptable)
- **After Selection**: 300 candles (1.7% - **UNACCEPTABLE**)

### For Light Mode (20 days, 15m timeframe):
- **Expected**: 20 days × 96 candles/day = **1,920 candles**
- **Actual in training**: 60-75 candles (3-4% - **UNACCEPTABLE**)

## Critical Fixes Needed

### Priority 1: Fix labeled_data Loading in Feature Selection
**File**: `src/training/steps/pre_training/feature_generation_final_feature_selection_step.py`
**Line**: ~223

**Current**:
```python
labeled_df = self._get_artifact('labeled_data')
```

**Problem**: This loads a 300-row subset

**Solution**: Ensure it loads the full dataset for the execution mode:
```python
# Option 1: Load from the correct store with full data
labeled_df = self._get_artifact('labeled_data_ETHUSDT_15m')  # Full artifact name

# Option 2: Load directly from versioned artifacts
from src.utils.versioned_artifacts.store import VersionedArtifactStore
store = VersionedArtifactStore(f'versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_analyst')
versions = [v for v in store.list_versions() if 'labeled_data' in v]
latest = sorted(versions)[-1]
labeled_df = store.get_view(latest).materialize()

# Option 3: Verify no sampling is applied
labeled_df = self._get_artifact('labeled_data', apply_sampling=False)
```

### Priority 2: Reduce Feature Generation Lookback
**File**: `src/training/steps/pre_training/feature_generation_feature_generation_step.py`

**Problem**: Using lookback windows that require 157k rows of history

**Solution**: 
- Configure shorter lookback periods for blank mode
- Use adaptive lookback based on available data
- Implement proper NaN handling at series start

### Priority 3: Fix Interaction Generation
**File**: `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`
**Line**: ~2478

**Problem**: LGBM SHAP pipeline crashes

**Solution**: Add error handling and fallback logic

## Verification Steps

After fixes, verify:
1. ✅ Feature generation preserves >90% of data (>15,500 rows for blank mode)
2. ✅ Feature selection uses FULL labeled_data (16,201 rows, not 300)
3. ✅ Final selected features have >15,000 rows
4. ✅ Training uses correct number of samples (~1,920 for light mode)
5. ✅ All 3 models (LightGBM, DepthwiseCNN, CatBoost) generate predictions

## Impact on Training

The current data loss explains ALL the training issues:
- ❌ Only 60 samples used (from 300-row subset intersecting with targets)
- ❌ Perfect R² scores (overfitting on tiny dataset)
- ❌ LightGBM/DepthwiseCNN fail (insufficient data)
- ❌ Only CatBoost works (more robust to small data)
- ❌ Predictions cover 1-2 days (from 300-row window)

**Once the data loss is fixed, all training issues should resolve.**
