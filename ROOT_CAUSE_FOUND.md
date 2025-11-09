# ROOT CAUSE FOUND: Data Truncation to 1000 Rows

## The Bug

**File:** `src/training/steps/market_analysis/rolling_hmm_clustering/rolling_hmm_regime_discovery_step.py:330-333`

```python
def _load_market_data(...):
    if 'market_data' in config and config['market_data'] is not None:
        external_data = config['market_data']
        tprint(f"✅ Using market data from config ({len(external_data)} samples)", "SUCCESS")
        return external_data  # ← Returns whatever is in config, could be 1000 rows!
```

**The Issue:**
When `config['market_data']` is provided, the step bypasses all normal data loading logic and uses whatever DataFrame is in the config. If that DataFrame has only 1000 rows, that's what gets used - **NO VALIDATION, NO FILTERING, NO CHECKS**.

## Evidence

### From HDF5 Inspection:
```
Version: rolling_hmm_regime_labels_20251108_202510_294
Expected rows: 4320  ✅ Correct (180 days * 24 hours/day)

Version: rolling_hmm_regime_labels_20251108_203927_884
Expected rows: 1000  ❌ Truncated (only 41.67 days)

Version: rolling_hmm_regime_labels_20251108_204120_651
Expected rows: 1000  ❌ Truncated (only 41.67 days)
```

### Timeline Analysis:
1. **20:25:10** - Run with 4320 rows (correct full dataset)
2. **20:39:27** - Run with 1000 rows (truncated)
3. **20:41:20** - Run with 1000 rows (truncated)
4. **20:46:14** - Run with 1000 rows (truncated)
5. **20:52:12** - Run with 1000 rows (truncated)

**Something changed between 20:25 and 20:39 that started passing truncated data in config!**

## Data Flow

### Normal Flow (Working):
```
_load_market_data()
  → KlinesParquetManager.load_klines(180 days)
  → Returns 4320 rows for 1h timeframe
  → _apply_execution_mode_filter(blank mode, 180 days)
  → Still 4320 rows
  → Feature generation
  → PCA
  → HMM prediction
  → Save 4320 rows to HDF5 ✅
```

### Broken Flow (Truncated):
```
_load_market_data()
  → config['market_data'] exists with 1000 rows
  → Returns 1000 rows immediately (BYPASS!)
  → _apply_execution_mode_filter() receives 1000 rows
  → Still 1000 rows (within limit)
  → Feature generation on 1000 rows
  → PCA on 1000 rows
  → HMM prediction on 1000 rows
  → Save 1000 rows to HDF5 ❌
```

## Who Is Passing Truncated Data?

Possible culprits:
1. **HPO (Hyperparameter Optimization)** - Might be sampling data to 1000 rows for faster optimization
2. **Execution mode config** - Might have a setting that pre-truncates data before passing to step
3. **Previous step in pipeline** - Might be passing its output as input to rolling HMM step
4. **Test/Debug code** - Someone might have added a `.head(1000)` somewhere for testing

## The Fix

### Option 1: Remove the bypass (RECOMMENDED)
Force all data to go through proper loading and filtering:

```python
def _load_market_data(...):
    # Remove these lines:
    # if 'market_data' in config and config['market_data'] is not None:
    #     external_data = config['market_data']
    #     return external_data

    # Always load from historical storage
    ...
```

### Option 2: Add validation
Keep the bypass but validate the data size:

```python
def _load_market_data(...):
    if 'market_data' in config and config['market_data'] is not None:
        external_data = config['market_data']

        # VALIDATE SIZE
        expected_min_samples = self._calculate_expected_samples(timeframe, execution_mode)
        if len(external_data) < expected_min_samples:
            tprint_warning(
                f"⚠️  Config market_data has only {len(external_data)} samples, "
                f"expected at least {expected_min_samples} for {execution_mode} mode"
            )
            # Fall through to normal loading
        else:
            tprint(f"✅ Using market data from config ({len(external_data)} samples)", "SUCCESS")
            return external_data

    # Normal loading logic...
```

### Option 3: Document and warn
Add clear logging when using config data:

```python
if 'market_data' in config and config['market_data'] is not None:
    external_data = config['market_data']
    tprint(f"⚠️  WARNING: Using market_data from config ({len(external_data)} samples)", "WARNING")
    tprint(f"    This bypasses normal data loading and execution mode filtering!", "WARNING")
    tprint(f"    Expected samples for {execution_mode} mode: {expected_samples}", "INFO")
    return external_data
```

## Recommendation

**Investigate who/what is populating `config['market_data']` with only 1000 rows**, then either:
1. Fix the upstream component to NOT truncate data
2. Remove the config bypass entirely and always load from historical storage
3. Add proper validation before accepting config data

## Next Steps

1. Add debug logging to see when `config['market_data']` is being used:
   ```python
   if 'market_data' in config and config['market_data'] is not None:
       tprint(f"🐛 DEBUG: config['market_data'] shape: {config['market_data'].shape}", "INFO")
       tprint(f"🐛 DEBUG: config keys: {list(config.keys())}", "INFO")
   ```

2. Check the ares_launcher or calling code to see what's populating this config

3. Search for `.head(1000)` or `.sample(1000)` in files that might call rolling_hmm_regime_discovery_step
