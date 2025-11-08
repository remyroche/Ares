# Regime Probabilities Loading Fix

## Problem
Regime models training was failing to load regime probabilities from versioned artifacts, resulting in 100% NaN values. This prevented the ML models from training properly since they need regime probabilities as target labels.

## Root Cause
1. **Wrong storage flag**: `use_versioned_artifacts=False` was set, preventing access to HDF5 storage
2. **Wrong timeframe**: Code was trying to resample from 1h to 15m, but we're training on 1h data
3. **Missing data_category hint**: Not providing `data_category='features'` hint for HDF5 routing

## Solution

### Changes Made

#### 1. Enable Versioned Artifacts
**File:** `src/training/steps/market_analysis/components/regime_models_training.py`

**Before:**
```python
base_step_inst = _ArtifactLoaderStep(
    "regime_models_training_loader",
    use_versioned_artifacts=False,  # ❌ Wrong!
)
```

**After:**
```python
base_step_inst = _ArtifactLoaderStep(
    "regime_models_training_loader",
    use_versioned_artifacts=True,  # ✅ Correct!
)
```

#### 2. Use Correct Timeframe
**Before:**
```python
# Set context with 15m timeframe
base_step_inst.set_context(
    symbol=symbol,
    exchange=exchange,
    timeframe='15m',  # ❌ Wrong! Doesn't match regime discovery output
    direction='long',
    model='regime',
)
```

**After:**
```python
# Set context to match regime discovery output (1h)
base_step_inst.set_context(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,  # ✅ Correct! Use same timeframe as training (1h)
    direction='long',
    model='regime',
)
```

#### 3. Remove Unnecessary Resampling
**Before:**
```python
# Resample from 1h to 15m using forward-fill
regime_probs_15m = regime_probs_1h.resample('15T').ffill()
```

**After:**
```python
# No resampling needed - load at same timeframe as training
return regime_probs
```

#### 4. Add data_category Hint
**Before:**
```python
regime_probs = base_step._get_artifact(
    'rolling_hmm_regime_probabilities',
    artifact_type='data'
)
```

**After:**
```python
regime_probs = base_step._get_artifact(
    'rolling_hmm_regime_probabilities',
    artifact_type='data',
    data_category='features'  # ✅ Hint for HDF5 routing
)
```

#### 5. Add Validation and Error Handling
Added comprehensive validation to detect when regime probabilities are completely mismatched:

```python
if all_nan_cols:
    error_msg = (
        f"❌ [REGIME_MODELS] CRITICAL: Regime probabilities have completely mismatched timestamps!\n"
        f"   All {len(all_nan_cols)} regime probability columns are 100% NaN.\n"
        f"   This means the regime discovery data doesn't match the current training data.\n"
        f"   \n"
        f"   SOLUTION: Run regime discovery FIRST with the same symbol and timeframe:\n"
        f"   python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery --symbol {symbol} --timeframe {timeframe} --execution-mode blank\n"
        f"   \n"
        f"   Then run regime models training again."
    )
    tprint(error_msg)
    raise ValueError(f"Regime probabilities completely mismatched - cannot train without valid regime labels.")
```

## How Regime Discovery Saves Data

**File:** `src/training/steps/market_analysis/rolling_hmm_clustering/rolling_hmm_regime_discovery_step.py`

```python
# Save probabilities to versioned artifacts
probs_df = pd.DataFrame(
    result['regime_probs'],
    index=result['timestamps'],
    columns=[f'regime_{i}_prob' for i in range(result['n_regimes'])]
)

self._save_artifact(
    data=probs_df,
    artifact_name='rolling_hmm_regime_probabilities',
    artifact_type='data',
    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
)
```

This uses `BaseStep._save_artifact()` which routes to:
- **Versioned artifacts (HDF5)** for feature DataFrames
- Stored in: `versioned_artifacts/{SYMBOL}_{EXCHANGE}_{TIMEFRAME}_{DIRECTION}_{MODEL}/`

## Correct Pipeline Order

### Step 1: Run Regime Discovery
```bash
python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery \
    --symbol ETHUSDT \
    --timeframe 1h \
    --execution-mode blank
```

**Output:**
- `rolling_hmm_regime_probabilities` → HDF5 storage
- `rolling_hmm_regime_labels` → HDF5 storage
- Timestamps match the market data

### Step 2: Run Regime Models Training
```bash
python3 src/launcher/ares_launcher.py regime_models_training \
    --symbol ETHUSDT \
    --timeframe 1h \
    --execution-mode blank
```

**Now it will:**
- ✅ Load regime probabilities from versioned artifacts
- ✅ Timestamps will match (both 1h)
- ✅ No NaN values
- ✅ Train ML models successfully

### Step 3: Run Regime Ensemble Training
```bash
python3 src/launcher/ares_launcher.py regime_ensemble_training \
    --symbol ETHUSDT \
    --timeframe 1h \
    --execution-mode blank
```

**Goal:** Create ensemble model that generates probabilities for each regime

## Expected Behavior

### Before Fix:
```
⚠️ Found 43423 non-finite values in column 'regime_0_prob'
⚠️ Found 43423 non-finite values in column 'regime_1_prob'
...
❌ Training with 100% NaN target labels
```

### After Fix:
```
✅ Loaded regime probabilities: (43423, 7)
📊 Columns: ['regime_0_prob', 'regime_1_prob', ..., 'regime_6_prob']
📊 Index range: 2024-05-01 00:00:00 to 2024-11-08 00:00:00
✅ Regime probabilities successfully joined
✅ Training with valid regime labels
```

## Verification

To verify the fix is working:

```bash
# 1. Check if regime discovery artifacts exist
ls -la versioned_artifacts/ETHUSDT_binance_1h_long_regime/

# 2. Run regime models training and check logs
tail -f logs/unified_*.log | grep "REGIME_MODELS"

# 3. Look for these success messages:
# ✅ Loaded regime probabilities: (N, 7)
# ✅ Regime probabilities successfully joined
# ✅ Training with valid regime labels
```

## Summary

- ✅ Fixed versioned artifacts loading
- ✅ Fixed timeframe matching
- ✅ Removed unnecessary resampling
- ✅ Added proper error handling
- ✅ Added validation for NaN detection
- ✅ Documented correct pipeline order

The regime models training will now properly load regime probabilities from versioned artifacts and train ML models to predict regime labels!
