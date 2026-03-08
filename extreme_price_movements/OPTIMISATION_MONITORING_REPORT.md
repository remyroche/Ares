# Optimisation Step Monitoring Report

## Date: 2026-03-08

## Summary

Monitored the `optimise` step of the extreme_price_movements pipeline for bugs, computation bottlenecks, and optimization opportunities.

## Bug Found and Fixed

### Issue
The optimized meta prediction was failing repeatedly with the error:
```
IndexError: only integers, slices (:), ellipsis (...), numpy.newaxis (None) and integer or boolean arrays are valid indices
```

### Root Cause
The `BatchPredictor.predict_batched` method was converting the input DataFrame to a numpy array before passing it to the model's `predict` method. However, the meta_model's `predict` method expects a DataFrame because it uses `self.selected_features` to select specific columns:

```python
# meta_model.py line 583
X = X_meta[self.selected_features].to_numpy(dtype=float)
```

When `X_meta` is a numpy array, attempting to use `self.selected_features` (a list of column names) as an index raises an IndexError.

### Solution
1. Added a `keep_dataframe` parameter to `BatchPredictor.predict_batched` to control whether the DataFrame should be kept as-is or converted to numpy
2. Updated the method to handle both DataFrame and numpy array inputs correctly
3. Updated calls to `predict_batched` in both `engine.py` and `optimized_predictions.py` to pass `keep_dataframe=True` when calling with meta models
4. Updated data preparation to keep the DataFrame instead of converting to numpy before passing to the batch predictor

### Files Modified
1. `extreme_price_movements/optimized_predictions.py`
   - Modified `BatchPredictor.predict_batched` to add `keep_dataframe` parameter
   - Updated `OptimizedPredictionPipeline.predict_meta_fast` to use `keep_dataframe=True`
   - Updated data preparation to keep DataFrame format

2. `extreme_price_movements/engine.py`
   - Updated `_meta_predict_or_fallback` to pass `keep_dataframe=True` to batch predictor
   - Updated data preparation to keep DataFrame format
   - Added detailed error logging with traceback

## Performance Observations

### 1. Feature Loading (54.2 seconds)
- **Current:** Loading 260 feature files takes ~54 seconds
- **Bottleneck:** This is a significant portion of the initialization time
- **Opportunity:** Use `OptimizedFeatureLoader` with chunking and memory mapping to speed this up
- **Expected Improvement:** 2-5x faster feature loading

### 2. Feature Materialization (34 seconds)
- **Current:** Materializing 110 feature matrices takes ~34 seconds
- **Observation:** Progress shown at 18.2%, 36.4%, 54.5%, 72.7%, 90.9%, 100.0%
- **Opportunity:** Could be optimized with parallel processing or lazy evaluation
- **Expected Improvement:** 1.5-2x faster with parallel processing

### 3. Signal Generation
- **Current Progress:** 38.2% (820/2146 hours)
- **Status:** Running smoothly with optimized meta prediction
- **Performance:** No errors, stable execution
- **Opportunity:** The optimized meta prediction is now working correctly and should provide 5-20x speedup in the meta prediction phase

### 4. Data Load (19.45 seconds)
- **Current:** Loading backtest data takes ~19 seconds
- **Observation:** Reasonable for 274 symbols over 17,497 hours
- **Opportunity:** Could use memory mapping for faster subsequent loads

## Optimization Opportunities

### 1. Feature Loading Optimization
**Current Issue:** Sequential loading of 260 feature files takes ~54 seconds

**Solution:** Implement `OptimizedFeatureLoader` with:
- Chunked loading (configurable chunk size)
- Memory mapping for large files
- Parallel file reading
- Automatic fallback to chunked loading if memory mapping fails

**Expected Impact:** 2-5x faster feature loading

### 2. Feature Caching
**Current Issue:** Features are recomputed on every run

**Solution:** Implement `FeatureComputationCache` with:
- LRU cache with configurable max size
- Cache keys based on feature name, timestamp, symbol, and parameters hash
- Thread-safe cache access
- Global cache instance shared across pipeline

**Expected Impact:** Eliminate redundant feature computations for repeated runs

### 3. Batch Size Tuning
**Current Issue:** Fixed batch size of 10,000 may not be optimal for all scenarios

**Solution:** Implement adaptive batch sizing:
- Start with default batch size
- Monitor memory usage and prediction speed
- Dynamically adjust batch size based on system resources
- Consider different batch sizes for different model types

**Expected Impact:** 10-20% improvement in prediction throughput

### 4. Numba JIT Acceleration
**Current Issue:** Some numerical operations are not JIT-compiled

**Solution:** Accelerate critical operations with Numba JIT:
- Sizing functions (sigmoid, tanh, concave)
- Disagreement calculations
- Regime interaction computations
- Clip and scale operations

**Expected Impact:** 10-50x speedup for JIT-compiled functions

### 5. Vectorized Meta Feature Preparation
**Current Status:** Already implemented in `OptimizedMetaCombiner`

**Verification:** Confirm that the vectorized implementation is being used
- Check if `OptimizedMetaCombiner.prepare_meta_features_fast` is being called
- Monitor performance of meta feature preparation
- Compare with legacy implementation

**Expected Impact:** 5-20x faster meta feature preparation

## Computation Bottlenecks Identified

### 1. Feature Loading (54.2 seconds)
**Priority:** High
**Impact:** Affects every pipeline run
**Solution:** Use `OptimizedFeatureLoader`

### 2. Feature Materialization (34 seconds)
**Priority:** Medium
**Impact:** Affects every pipeline run
**Solution:** Implement parallel materialization

### 3. Meta Feature Preparation
**Priority:** Low (already optimized)
**Impact:** Affects every prediction
**Solution:** Verify vectorized implementation is being used

## Recommendations

### Immediate Actions
1. ✅ Fix the optimized meta prediction bug (COMPLETED)
2. Implement `OptimizedFeatureLoader` for feature loading
3. Implement `FeatureComputationCache` for feature caching
4. Verify that all optimized components are being used

### Short-term Improvements
1. Tune batch sizes for optimal performance
2. Implement adaptive batch sizing
3. Add performance metrics logging
4. Benchmark optimized vs. legacy implementations

### Long-term Enhancements
1. GPU acceleration for suitable operations
2. Distributed inference for large-scale predictions
3. Advanced caching strategies (e.g., persistent cache)
4. Model quantization for faster inference

## Conclusion

The optimisation step is now running successfully with the bug fix. The optimized meta prediction is working correctly and should provide significant speedup. Several optimization opportunities have been identified, with feature loading being the highest priority bottleneck.

### Backtest Completion Status
- ✅ Backtest completed successfully (100% - 2146/2146 hours)
- ✅ Optimized meta prediction worked perfectly throughout the entire backtest
- ✅ No errors during signal generation or meta prediction
- ⚠️ Post-processing error encountered (timezone issue in datetime subtraction)

### Additional Bug Found
**Error:** `TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects`
**Location:** `pipeline_steps.py` line 2183
**Context:** Calculating hold times in post-processing
**Impact:** Does not affect the optimized meta prediction functionality
**Priority:** Medium - needs to be fixed for full pipeline functionality

The next steps should focus on:
1. Implementing the `OptimizedFeatureLoader` to speed up feature loading
2. Implementing the `FeatureComputationCache` to eliminate redundant computations
3. Benchmarking the performance improvements from all optimizations
4. Fixing the timezone bug in the post-processing phase
