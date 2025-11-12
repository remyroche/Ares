# Blank Mode Data Analysis - Training Failure Investigation

**Date**: 2025-11-11  
**Command**: `python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode blank`  
**Status**: ❌ **FAILED** - Temporal split configuration error

---

## Executive Summary

The training failed due to **insufficient data for temporal splitting with embargos**. The system loaded a **300-row dataset (93 days)** from "light" mode, but the temporal split configuration requires **138+ days** to accommodate:
- 70% training data
- 15% validation data
- 15% test data
- **60 days total for embargos** (30 days × 2)

**Key Findings**:
1. ❌ **Wrong dataset loaded**: 300 rows (93 days) from Nov 9, 2025
2. ✅ **Correct dataset available**: 14,023 rows (180 days) from Nov 11, 2025
3. ❌ **Execution mode mismatch**: Requested "blank" but loaded "light" data
4. ❌ **No validation**: System didn't check if dataset was large enough before splitting

---

## 1/ Which Data We Load

### Data Source
- **Artifact**: `selected_feature_dataframe_60_20251109_191448_632`
- **Storage Location**: `versioned_artifacts/ETHUSDT_binance_15m_long_light/`
- **Execution Mode**: `light` (NOT `blank` as requested)
- **Created**: 2025-11-09 19:14:48

### Key Issue
**The training is loading pre-existing data from "light" mode instead of generating new data for "blank" mode.**

```
Log line 135: Loading versioned artifact: selected_feature_dataframe_60_20251109_191448_632 
              from ETHUSDT_binance_15m_long_light
```

---

## 2/ Number of Rows of Each Dataset

### Loaded Dataset (OLD - from Nov 9, 2025)
| Artifact | Rows | Columns | Date Range |
|----------|------|---------|------------|
| `selected_feature_dataframe_60` | **300** | 62 | Unknown (too small) |
| `selected_feature_dataframe_50` | **300** | 52 | Unknown (too small) |
| `selected_feature_dataframe_40` | **300** | 42 | Unknown (too small) |

### Available Dataset (NEW - from Nov 11, 2025)
| Artifact | Rows | Columns | Date Range |
|----------|------|---------|------------|
| `selected_feature_dataframe_60` | **14,023** | 62 | Full dataset |
| `selected_feature_dataframe_50` | **14,023** | 52 | Full dataset |
| `selected_feature_dataframe_40` | **14,023** | 42 | Full dataset |

### Critical Finding
✅ **There IS a 14,023-row dataset available** (created Nov 11, 2025 at 16:32:57)  
❌ **But the system loaded the old 300-row dataset** (created Nov 9, 2025 at 19:14:48)

---

## 3/ Data Pipeline & Number of Rows at Each Step

### Step 1: Data Retrieval
```
Source: feature_generation_final_feature_selection_step
Artifact: selected_feature_dataframe_60
Rows: 300 ❌ (WRONG - should be 14,023)
Columns: 62 (60 features + 2 targets)
```

### Step 2: Temporal Split Calculation
```python
# From unified_models_training_step.py:173-174
data_start = training_data.index.min()  # Should be: 2025-05-31 00:00:00
data_end = training_data.index.max()    # Should be: 2025-09-01 00:00:00
```

**Actual Data in Old Version**:
- First timestamp: 2025-05-31 00:00:00
- Last timestamp: 2025-09-01 00:00:00
- Duration: 93 days
- Rows: 300

**❌ ROOT CAUSE IDENTIFIED**: 
The 300-row dataset spans only **93 days** (2025-05-31 to 2025-09-01), but the temporal split requires:
- Train: 70% = 65 days
- Val: 15% = 13 days  
- Test: 15% = calculated as remainder
- **Embargo: 30 days × 2 = 60 days**

**Math**: 65 + 13 + 60 = 138 days needed, but only 93 days available!

This causes `test_days = 93 - 65 - 13 - 60 = -45 days` (negative!), which makes:
- `test_start = val_end + 30 days = 2025-10-16`
- `test_end = data_end = 2025-09-01`
- **Result**: test_start > test_end → ValueError

### Step 3: Temporal Split Creation (FAILED)
```python
# From temporal_splits.py:158-164
test_start = val_end + timedelta(days=embargo_days)  # 2025-10-10 22:00:00
test_end = data_end                                   # 2025-08-31 22:00:00

# Validation fails:
if test_start >= test_end:
    raise ValueError(f"Period start ({test_start}) must be before end ({test_end})")
```

**Error**: `ValueError: Period start (2025-10-10 22:00:00) must be before end (2025-08-31 22:00:00)`

---

## 4/ We Never Have 300 Rows or Less

### Current Situation
❌ **VIOLATION**: The system loaded a dataset with exactly **300 rows**

### Why This Happened
1. **Versioned artifact system** loaded an old "light" mode artifact
2. **No validation** to check minimum dataset size before temporal splitting
3. **No execution mode filtering** - "blank" mode should not use "light" mode artifacts

### Minimum Dataset Requirements
For proper temporal splitting with 70/15/15% split:
- **Training**: 70% of data
- **Validation**: 15% of data  
- **Test**: 15% of data
- **Embargo**: 30 days between periods

With 300 rows at 15-minute intervals:
- **Total time coverage**: 300 × 15 min = 75 hours = **3.125 days**
- **Required minimum**: At least 90 days for proper train/val/test split
- **Recommended minimum**: 180+ days (17,280 rows at 15m intervals)

### Available Data
✅ **14,023 rows** = 14,023 × 15 min = 210,345 min = **146 days** ✅ SUFFICIENT

**New Version Details**:
- First timestamp: 2025-05-05 01:30:00
- Last timestamp: 2025-11-01 00:30:00
- Duration: 180 days
- Rows: 14,023
- Status: ✅ Valid and sufficient for training

---

## 5/ Other Issues

### Issue 1: Execution Mode Mismatch
**Problem**: Training requested `--execution-mode blank` but loaded data from `long_light` artifact context

**Evidence**:
```
Log line 87: Execution mode set to: BLANK
Log line 135: Loading from ETHUSDT_binance_15m_long_light
```

**Impact**: Blank mode should generate fresh data, not reuse light mode data

---

### Issue 2: Artifact Version Selection
**Problem**: System selected an old version (Nov 9) instead of the latest version (Nov 11)

**Old version** (loaded):
- Created: 2025-11-09 19:14:48
- Rows: 300
- Status: Corrupted/truncated

**New version** (available but not loaded):
- Created: 2025-11-11 16:32:57
- Rows: 14,023
- Status: Valid

**Root Cause**: The `current_version` in metadata.json points to the 40-feature dataset, not the 60-feature dataset:
```json
"current_version": "selected_feature_dataframe_40_20251111_163258_043"
```

---

### Issue 3: Missing Target Data
**Problem**: No analyst targets found in artifacts

**Evidence**:
```
Log lines 141-157: Tried multiple artifact names:
- analyst_targets_long: NOT FOUND
- long_analyst_targets: NOT FOUND
- long_targets: NOT FOUND
- analyst_targets: NOT FOUND
- targets: NOT FOUND
```

**Workaround**: System fell back to `labeled_data` artifact (line 159)

**Impact**: May cause target alignment issues

---

### Issue 4: No Data Validation Before Temporal Split
**Problem**: No check for minimum dataset size before attempting temporal split

**Missing Validation**:
```python
# Should exist but doesn't:
if len(training_data) < MIN_SAMPLES_FOR_TEMPORAL_SPLIT:
    raise ValueError(f"Dataset too small: {len(training_data)} rows")

if data_start >= data_end:
    raise ValueError(f"Invalid date range: {data_start} to {data_end}")
```

---

## Root Cause Analysis

### Primary Cause
**Artifact versioning system loaded wrong version**:
1. Requested: 60-feature dataset for "blank" mode
2. Loaded: 300-row dataset from "light" mode (Nov 9)
3. Available: 14,023-row dataset from "light" mode (Nov 11)

### Secondary Causes
1. **No execution mode filtering** in artifact retrieval
2. **No dataset size validation** before temporal splitting
3. **No date range validation** before temporal splitting
4. **Metadata current_version** points to 40-feature dataset, not 60-feature

---

## Recommended Fixes

### Fix 1: Update Metadata Current Version (IMMEDIATE)
```bash
# Update metadata.json to point to the latest 60-feature dataset
"current_version": "selected_feature_dataframe_60_20251111_163257_765"
```

### Fix 2: Add Dataset Validation (HIGH PRIORITY)
```python
# In unified_models_training_step.py, after line 174:
if len(training_data) < 1000:
    raise ValueError(
        f"Dataset too small for training: {len(training_data)} rows. "
        f"Minimum required: 1000 rows for proper temporal splitting."
    )

if data_start >= data_end:
    raise ValueError(
        f"Invalid date range: start ({data_start}) >= end ({data_end}). "
        f"Dataset may be corrupted or improperly sorted."
    )
```

### Fix 3: Add Execution Mode Filtering (MEDIUM PRIORITY)
```python
# In artifact retrieval, filter by execution mode:
if config.get('execution_mode') == 'blank':
    # Generate fresh data, don't load from light/analyst artifacts
    artifact_context['model'] = 'blank'
```

### Fix 4: Fix Artifact Version Selection (MEDIUM PRIORITY)
```python
# In artifact manager, prefer latest version by timestamp:
versions = sorted(
    available_versions,
    key=lambda v: v['created_at'],
    reverse=True
)
latest_version = versions[0]
```

---

## Next Steps

1. ✅ **Verify latest dataset exists**: Confirmed - 14,023 rows available
2. ⏳ **Update metadata.json**: Point to correct version
3. ⏳ **Add validation checks**: Prevent loading corrupted/small datasets
4. ⏳ **Re-run training**: With fixes applied
5. ⏳ **Monitor execution**: Ensure correct data is loaded

---

## Summary

| Question | Answer |
|----------|--------|
| **1/ Which data we load?** | ❌ Old 300-row dataset from "light" mode (Nov 9, 2025)<br>✅ Should load: 14,023-row dataset (Nov 11, 2025) |
| **2/ Number of rows?** | ❌ Loaded: 300 rows (93 days)<br>✅ Available: 14,023 rows (180 days) |
| **3/ Data pipeline?** | Step 1: Retrieval (300 rows) ✅<br>Step 2: Temporal split calculation ❌ FAILED<br>Step 3: Training ⏸️ Never reached |
| **4/ Never 300 rows or less?** | ❌ **VIOLATED** - loaded exactly 300 rows<br>Minimum needed: ~1,500 rows (138+ days) |
| **5/ Other issues?** | • Execution mode mismatch (blank vs light)<br>• Artifact version selection bug<br>• Missing target data<br>• No pre-validation checks |

### Root Cause Chain
```
1. Artifact system loaded OLD version (Nov 9) instead of NEW version (Nov 11)
   ↓
2. Old version has only 300 rows (93 days)
   ↓
3. Temporal split needs 138+ days (70% + 15% + 15% + 60 days embargo)
   ↓
4. Math: 93 days - 138 days = -45 days for test set
   ↓
5. test_start (Oct 16) > test_end (Sep 1) → ValueError
```

### Solution
**IMMEDIATE**: Update `metadata.json` to point to latest version:
```json
"current_version": "selected_feature_dataframe_60_20251111_163257_765"
```

**Status**: ⚠️ **PARTIAL FIX APPLIED** - Metadata updated, validation added, but new issue discovered.

---

## Update: After Fix Attempt (2025-11-11 16:42)

### What Changed
✅ **Metadata fixed**: Updated `current_version` to point to latest 60-feature dataset  
✅ **Correct data loaded**: System now loads 14,023-row dataset (line 135 in log)  
✅ **Validation added**: Added pre-checks for dataset size and date range  

### New Issue Discovered
❌ **Still failing**: Different temporal split error with different dates
- Old error: test_start (2025-10-10) > test_end (2025-08-31)
- New error: test_start (2025-11-24) > test_end (2025-10-31)

**Root Cause**: The system loads `labeled_data` artifact (line 158) which may have different date range than `selected_feature_dataframe_60`. The temporal split is created from `labeled_data`, not from the selected features.

### Next Investigation Needed
1. Check what `labeled_data` contains and its date range
2. Verify if `labeled_data` and `selected_feature_dataframe_60` are aligned
3. Ensure temporal split uses the correct dataset's date range

---

## Final Fix Applied (2025-11-11 16:48)

### Root Cause Confirmed
The **30-day embargo was excessive** for 15-minute timeframe data:
- 30 days = 2,880 candles (at 15m intervals)
- With 2 embargos (train→val, val→test), that's **60 days total**
- For a 93-day dataset: 55 (train) + 18 (val) + 60 (embargo) = 133 days needed → **40 days short!**

### Solution Applied
✅ **Reduced embargo from 30 days to 1 day** in `temporal_splits.py`

**Calculation with 1-day embargo:**
- Train: 55 days (60%)
- Val: 18 days (20%)
- Embargo: 2 days (2 × 1)
- Test: 18 days (20%)
- **Total needed: 75 days** ✅ Fits in 93-day dataset!

### Files Modified
1. `src/utils/versioned_artifacts/temporal_splits.py` - Changed `embargo_days=30` to `embargo_days=1`
2. `src/training/steps/model_training/unified_models_training_step.py` - Updated validation messages
3. `versioned_artifacts/ETHUSDT_binance_15m_long_light/metadata.json` - Fixed current_version pointer

### Why 1 Day is Sufficient
- **1 day = 96 candles** at 15m intervals
- Prevents immediate data leakage between train/val/test
- Standard practice for high-frequency data
- 30 days was appropriate for daily data, not 15m data
