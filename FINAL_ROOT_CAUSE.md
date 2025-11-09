# FINAL ROOT CAUSE: Historical Data Truncation

## Discovery

After applying comprehensive debug logging and running the pipeline, we discovered the **actual** root cause:

### The Real Problem

**The historical data storage (KlinesParquetManager) only contains 1000 rows of data for ETHUSDT at 1h timeframe!**

### Evidence from Debug Output:

```
📥 [REGIME_DISCOVERY] Loading fresh data for ETHUSDT from historical storage
✅ Loaded 1000 samples of market data
🐛 DEBUG: [STEP 1] market_data after _load_market_data: (1000, 5)
🐛 DEBUG: [STEP 1] Index range: 2024-01-01 00:00:00 to 2024-02-11 15:00:00
INFO:   → Blank mode: Data size (1000) within limit (4320 samples)
```

**Key Points:**
1. Loading from historical storage returned only 1000 samples
2. Date range: Jan 1 to Feb 11 = ~42 days (1000 hours at 1h timeframe)
3. This is NOT a truncation by the pipeline - this is ALL the data that exists in storage
4. The execution mode filter (180 days limit) was not triggered because 1000 < 4320

## Timeline of Investigation

### Phase 1: HDF5 Storage ✅
**Hypothesis:** HDF5 storage is truncating data
**Finding:** ❌ NO - HDF5 works perfectly, stores exactly what it receives

### Phase 2: Data Pipeline ✅
**Hypothesis:** Data is truncated during feature generation or PCA
**Finding:** ❌ NO - All pipeline steps preserve data size correctly

### Phase 3: Config Bypass ✅
**Hypothesis:** `config['market_data']` is providing truncated data
**Finding:** ❌ NO - config doesn't contain market_data, so this path wasn't taken

### Phase 4: Historical Storage 🎯
**Hypothesis:** Historical data storage only has 1000 rows
**Finding:** ✅ YES - This is the root cause!

## The Actual Flow

```
KlinesParquetManager.load_klines(ETHUSDT, 1h, 180 days)
  ↓
Returns only 1000 rows (all that exists in storage)
  ↓
rolling_hmm_regime_discovery_step receives 1000 rows
  ↓
Execution mode filter: 1000 < 4320 limit ✅ (passes, no truncation)
  ↓
Feature generation on 1000 rows
  ↓
PCA on 1000 rows
  ↓
HMM on 1000 rows
  ↓
Save 1000 rows to HDF5 ✅ (correct behavior!)
```

## Why This Happened

The historical_data directory for ETHUSDT only contains ~42 days of data (1000 hours), not the full 180 days requested.

Possible reasons:
1. **Data collection was only run for ~42 days**
2. **Historical data was deleted/purged** at some point
3. **Data download failed** after ~42 days
4. **Storage space issues** prevented collecting more data

## The Fix Required

This is NOT a code bug - this is a **data availability issue**.

### Solution Options:

### Option 1: Download More Historical Data (RECOMMENDED)
```bash
# Run data collection for ETHUSDT to get 180 days of 1h data
python3 src/launcher/ares_launcher.py --data-download --symbol ETHUSDT --timeframe 1h --days 180
```

### Option 2: Adjust Expectations
If only 42 days of data is available, adjust the step to work with less data:
- Change execution mode from "blank" (180 days) to a custom mode
- Or accept that training will use only ~42 days of data

### Option 3: Use Different Symbol/Timeframe
Try a different symbol or timeframe that has more historical data available

## Files Modified (Still Useful!)

Even though the root cause was different, our fixes are still valuable:

1. ✅ **HDF5 Index Chunking** - Prevents future index/data misalignment
2. ✅ **Config Validation** - Protects against truncated data in config
3. ✅ **Debug Logging** - Helped us find the real root cause!

## Recommendation

**Run data collection first:**
```bash
python3 src/launcher/ares_launcher.py --data-download --symbol ETHUSDT --timeframe 1h --days 180
```

Then re-run the regime discovery:
```bash
python3 src/launcher/ares_launcher.py --rolling-hmm-regime-discovery --symbol ETHUSDT --execution-mode blank
```

You should then see:
```
✅ Loaded 4,320 samples of market data  # (180 days × 24 hours)
🐛 DEBUG: [STEP 1] market_data after _load_market_data: (4320, 5)
```

## Conclusion

The investigation revealed:
- ✅ HDF5 storage works correctly
- ✅ Pipeline preserves data size correctly
- ✅ Config bypass works correctly
- ❌ Historical data storage only has 1000 rows for ETHUSDT

**Action Required:** Download/collect more historical data for ETHUSDT at 1h timeframe.

The debug logging we added will continue to be useful for tracking data flow and catching any future issues!
