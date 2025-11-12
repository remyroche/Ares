# Training Issues & Fixes - ETHUSDT Analyst Base (Light Mode)

**Date**: 2025-11-11  
**Command**: `python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light`  
**Status**: 🔄 RUNNING (with issues)

---

## Issues Detected

### ❌ Issue 1: Data Shape Mismatch in HPO
**Error**: `Found input variables with inconsistent numbers of samples: [1200, 200]`

**Location**: HPO cross-validation during hyperparameter optimization

**Root Cause**: 
- CV expects 800 samples total (2 folds × 400 samples each)
- But receiving mismatched arrays: 1200 vs 200 samples
- This suggests features (X) and targets (y) are not aligned

**Impact**: 
- All HPO trials failing
- Model cannot be trained without successful hyperparameter optimization

**Fix Priority**: 🔴 CRITICAL - Blocks training

**Potential Causes**:
1. **Target misalignment**: `labeled_data` has different length than `selected_feature_dataframe_60`
2. **Index mismatch**: Features and targets have different timestamps
3. **NaN filtering**: Features were filtered for NaNs but targets weren't (or vice versa)
4. **Temporal split issue**: Train/val/test splits created different sized datasets

**Fix Approach**:
```python
# Need to verify in _retrieve_training_data():
# 1. Check shapes after loading
print(f"Features shape: {training_data.shape}")
print(f"Targets shape: {analyst_targets.shape}")

# 2. Verify indices match
assert training_data.index.equals(analyst_targets.index), "Index mismatch!"

# 3. Align if needed
common_idx = training_data.index.intersection(analyst_targets.index)
training_data = training_data.loc[common_idx]
analyst_targets = analyst_targets.loc[common_idx]
```

---

### ⚠️ Issue 2: High CPU Usage
**Warning**: `cpu_usage (98.5) exceeds threshold (85.0)`

**Impact**: 
- System slowdown
- Potential thermal throttling
- May cause timeouts or crashes

**Fix Priority**: 🟡 MEDIUM - Doesn't block but reduces performance

**Fix Approach**:
- Already using light mode (reduced trials: 100 → 10)
- Consider reducing parallel workers
- May need to add cooling delays between trials

---

### 🔍 Issue 3: Repeated CV Failures
**Pattern**: Same error repeating for every HPO trial

**Observation**:
```
Trial 1: Failed - [1200, 200] mismatch
Trial 2: Failed - [1200, 200] mismatch  
Trial 3: Failed - [1200, 200] mismatch
...
```

**Impact**: 
- No successful trials = no optimal parameters found
- Will likely fail or use default parameters

**Fix Priority**: 🔴 CRITICAL - Same as Issue 1

---

## Predicted Issues (Not Yet Encountered)

### 📊 Issue 4: Temporal Split Config May Not Exist
**Potential Error**: `Temporal split config not found at config/temporal_splits/ETHUSDT_binance_15m.json`

**Why It Might Happen**:
- First run after embargo change
- Config file needs to be regenerated with new 1-day embargo

**Fix**: 
- Delete old config: `rm config/temporal_splits/ETHUSDT_binance_15m.json`
- Let system regenerate with new embargo settings

---

### 📊 Issue 5: Labeled Data May Have Different Date Range
**Potential Error**: Temporal split dates don't match between features and targets

**Why It Might Happen**:
- `labeled_data` artifact created at different time than `selected_feature_dataframe_60`
- Different filtering/processing applied

**Fix**:
- Ensure both artifacts use same date range
- Add validation to check date range alignment

---

## Immediate Action Plan

### Step 1: Stop Current Run (if needed)
The current run will likely fail due to Issue 1. Consider stopping it to save resources.

### Step 2: Investigate Data Alignment
```bash
# Check what's in the artifacts
python3 -c "
from src.utils.versioned_artifacts import VersionedArtifactManager
import pandas as pd

# Load features
mgr = VersionedArtifactManager('ETHUSDT', 'binance', '15m', 'long', 'light')
features = mgr.load_artifact('selected_feature_dataframe_60')
print(f'Features: {features.shape}')
print(f'Features index: {features.index[0]} to {features.index[-1]}')

# Load targets (if available)
try:
    targets = mgr.load_artifact('labeled_data')
    print(f'Targets: {targets.shape}')
    print(f'Targets index: {targets.index[0]} to {targets.index[-1]}')
    
    # Check alignment
    if 'target_long' in targets.columns:
        target_col = targets['target_long']
        print(f'Target column: {target_col.shape}')
        
        # Check for common indices
        common = features.index.intersection(targets.index)
        print(f'Common indices: {len(common)}')
        print(f'Features only: {len(features.index.difference(targets.index))}')
        print(f'Targets only: {len(targets.index.difference(features.index))}')
except Exception as e:
    print(f'Error loading targets: {e}')
"
```

### Step 3: Fix Data Alignment in Code
Add alignment logic in `unified_models_training_step.py` after loading data:

```python
# After line 1960 (after loading training_data and analyst_targets)
if training_data is not None and analyst_targets is not None:
    # Ensure indices are aligned
    if not training_data.index.equals(analyst_targets.index):
        tprint_warning(f"⚠️ Index mismatch detected!")
        tprint_info(f"   Features: {len(training_data)} samples")
        tprint_info(f"   Targets: {len(analyst_targets)} samples")
        
        # Find common indices
        common_idx = training_data.index.intersection(analyst_targets.index)
        
        if len(common_idx) == 0:
            raise ValueError("No common indices between features and targets!")
        
        tprint_info(f"   Aligning to {len(common_idx)} common samples")
        training_data = training_data.loc[common_idx]
        analyst_targets = analyst_targets.loc[common_idx]
        
        tprint_success(f"✅ Data aligned: {training_data.shape}")
```

### Step 4: Delete Old Temporal Config
```bash
rm -f config/temporal_splits/ETHUSDT_binance_15m.json
```

### Step 5: Re-run Training
```bash
python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light
```

---

## Root Cause Analysis

The **[1200, 200]** mismatch suggests:

### Hypothesis 1: Light Mode Filtering
- Light mode may filter features more aggressively than targets
- Features: 14,023 → filtered → 1,200 samples
- Targets: 14,023 → filtered → 200 samples (or vice versa)

### Hypothesis 2: Temporal Split Mismatch
- Features using one temporal split
- Targets using different temporal split
- Results in different sample counts

### Hypothesis 3: NaN Handling
- Features: NaNs removed → 1,200 valid samples
- Targets: Different NaN pattern → 200 valid samples
- No alignment step to ensure both have same indices

---

## Success Criteria

Training will succeed when:
1. ✅ Features and targets have matching shapes
2. ✅ Features and targets have matching indices
3. ✅ HPO trials complete without shape mismatch errors
4. ✅ Model training completes with valid metrics
5. ✅ Artifacts saved successfully

---

## Monitoring Commands

```bash
# Watch log for errors
tail -f logs/unified_*.log | grep -E "ERROR|WARNING|Failed"

# Check HPO progress
grep "Objective evaluation" logs/unified_*.log | tail -20

# Check data shapes
grep "shape\|samples" logs/unified_*.log | tail -30
```
