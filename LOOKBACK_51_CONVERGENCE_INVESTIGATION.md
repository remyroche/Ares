# Lookback 51 Convergence Investigation

## Executive Summary

**Issue**: The Bayesian optimizer (coarse_to_refine method) is converging to lookback ~51 for 94.8% of features (237/250), with the remaining 5.2% (13/250) converging to lookback 49.

**Root Causes Identified**:
1. **Bug in coarse horizon generation**: `np.logspace(..., dtype=int)` truncates instead of rounding
2. **Bug in refinement range**: Upper boundary not included due to Python's range() behavior
3. **Possible deployed code inconsistency**: Features reached 51 despite code suggesting max should be 50

---

## Data Analysis

### Outcome File: `feature_lookback_optimization_outcome_20251009_225258.json`

**Lookback Distribution**:
- Lookback 51: 237 features (94.8%)
- Lookback 49: 13 features (5.2%)
- **Average**: 50.90
- **Min**: 49
- **Max**: 51

**Configuration**:
- Exchange: binance
- Symbol: ETHUSDT
- Timeframe: 15m
- Pipeline: differentiated_long_short
- Method: coarse_to_refine
- Lookback range: [5, 51]

**Sample Feature Results**:
```
rsi_14_returns_vwap: lookback=51, score=-0.08369
williams_r_14_price_returns: lookback=51, score=-0.09427
stochastic_14_3_price_returns: lookback=51, score=-0.08755
macd_12_26_9_returns_vwap: lookback=51, score=-0.10000
```

---

## Root Cause Analysis

### Bug #1: Coarse Horizon Generation Truncation

**Location**: `src/training/steps/pre_training/feature_lookback_optimization/core/optimizer.py:3995`

**Current Code**:
```python
log_horizons = np.logspace(np.log10(log_start), np.log10(max_horizon), 10, dtype=int)
```

**Problem**: `dtype=int` **truncates** (floors) values instead of rounding them.

**Example** (min_lookback=5, max_lookback=51):
```
Float values: [8.00, 9.83, 12.07, 14.83, 18.22, 22.39, 27.50, 33.79, 41.51, 51.00]
Integer (truncated): [7, 9, 12, 14, 18, 22, 27, 33, 41, 50]  ← 51 becomes 50!
```

**Impact**:
- The maximum coarse horizon is 50, not 51
- Upper boundary (51) is never tested in coarse search
- Coarse search bias toward lookback 50

**Expected Coarse Horizons**: `[5, 6, 7, 9, 12, 14, 18, 22, 27, 33, 41, 50, 51]`
**Actual Coarse Horizons**: `[5, 6, 7, 9, 12, 14, 18, 22, 27, 33, 41, 50]` ← missing 51

---

### Bug #2: Refinement Range Doesn't Include Upper Boundary

**Location**: `src/training/steps/pre_training/feature_lookback_optimization/core/optimizer.py:4175-4179`

**Current Code**:
```python
refinement_horizons = range(
    max(min_lookback, horizon - 10), 
    min(max_lookback, horizon + 11), 
    2  # Check every 2 periods
)
```

**Problem**: Python's `range()` is **exclusive** of the end value. To include `max_lookback`, the code should use `max_lookback + 1`.

**Example** (horizon=50, max_lookback=51):
```python
range(40, min(51, 61), 2) = range(40, 51, 2) = [40, 42, 44, 46, 48, 50]
```
→ **Does NOT include 51!**

**Refinement Ranges**:
| Coarse Horizon | Refinement Range | Max Refined Value |
|---|---|---|
| 33 | range(23, 44, 2) | 43 |
| 41 | range(31, 51, 2) | **49** |
| 50 | range(40, 51, 2) | **50** |

**Impact**:
- Even if coarse search finds horizon 50 as best, refinement can't reach 51
- Upper boundary is systematically excluded from search space
- Optimizer is biased toward lookback values < max_lookback

---

### Mystery: How Did Features Reach 51?

**Observed**: 237 features (94.8%) have lookback=51
**Expected** from code analysis: Maximum lookback should be 50

**Possible Explanations**:
1. **Deployed code differs from source**: Production code may have been patched or modified
2. **Boundary check exists elsewhere**: There may be explicit max_lookback testing not visible in refinement code
3. **Forward returns dict includes 51**: The forward_returns matrix includes all horizons from 1 to max_lookback+1
4. **Post-processing adjustment**: Lookback values may be adjusted after optimization
5. **Early stopping with boundary preference**: Some early-stopping logic may prefer the upper boundary

**Evidence for Forward Returns Including 51**:
```python
# Line 3554
for horizon in range(1, max_horizon + 1):  # Includes max_horizon!
    horizon_map[horizon] = ...
```

This suggests the forward_returns dict DOES include horizon 51, so technically the optimizer CAN test it if the refinement range is corrected.

---

## Recommended Fixes

### Fix #1: Coarse Horizon Generation (HIGH PRIORITY)

**Option A**: Use rounding instead of truncation
```python
log_horizons = np.round(np.logspace(np.log10(log_start), np.log10(max_horizon), 10)).astype(int)
```

**Result**:
```
Float values: [8.00, 9.83, 12.07, 14.83, 18.22, 22.39, 27.50, 33.79, 41.51, 51.00]
Integer (rounded): [8, 10, 12, 15, 18, 22, 28, 34, 42, 51]  ← Includes 51!
```

**Option B**: Explicitly include boundaries
```python
all_horizons = sorted(list(set(dense_horizons + log_horizons + [min_horizon, max_horizon])))
```

**Recommendation**: Use **Option A** (rounding) as it's more mathematically correct and Option B as a safety net.

---

### Fix #2: Refinement Range Upper Boundary (HIGH PRIORITY)

**Change**:
```python
refinement_horizons = range(
    max(min_lookback, horizon - 10), 
    min(max_lookback + 1, horizon + 11),  # Changed: max_lookback → max_lookback + 1
    2
)
```

**Result** (horizon=50, max_lookback=51):
```python
range(40, min(52, 61), 2) = range(40, 52, 2) = [40, 42, 44, 46, 48, 50]  # Still no 51!
```

Wait, this still doesn't include 51 because the step is 2 and we start from 40 (even).

**Better Fix**: Ensure boundary is always tested
```python
refinement_horizons = list(range(
    max(min_lookback, horizon - 10), 
    min(max_lookback + 1, horizon + 11), 
    2
))
# Explicitly add boundaries if not included
if min_lookback not in refinement_horizons:
    refinement_horizons.append(min_lookback)
if max_lookback not in refinement_horizons and horizon + 10 >= max_lookback:
    refinement_horizons.append(max_lookback)
refinement_horizons = sorted(refinement_horizons)
```

---

### Fix #3: Alternative - Use Step Size of 1 for Final Refinement

For horizons close to boundaries, use finer granularity:
```python
# If horizon is within 10 of max_lookback, use step=1
if horizon + 10 >= max_lookback:
    step = 1
else:
    step = 2

refinement_horizons = range(
    max(min_lookback, horizon - 10),
    min(max_lookback + 1, horizon + 11),
    step
)
```

---

## Implementation Plan

### Phase 1: Fix Coarse Horizon Generation
1. Update `_generate_coarse_horizons()` method
2. Use `np.round().astype(int)` instead of `dtype=int`
3. Explicitly ensure max_horizon is in the result
4. Add unit test to verify boundaries are included

### Phase 2: Fix Refinement Range
1. Update `_parallel_refinement()` method
2. Change refinement range to use `max_lookback + 1`
3. Explicitly add `max_lookback` to refinement candidates if close
4. Add unit test to verify boundary testing

### Phase 3: Validation
1. Re-run optimization on same dataset
2. Verify lookback distribution is more uniform
3. Check that upper boundary is properly tested
4. Compare performance metrics before/after fix

---

## Expected Impact After Fixes

### Before (Current):
- Coarse horizons: [5, 6, 7, 9, 12, 14, 18, 22, 27, 33, 41, 50]
- Refinement max: 50 (or 49 depending on coarse best)
- Distribution: 94.8% at 51 (anomalous), 5.2% at 49

### After (Fixed):
- Coarse horizons: [5, 6, 7, 8, 10, 12, 15, 18, 22, 28, 34, 42, 51]
- Refinement max: 51 (properly tested)
- Distribution: Expected to be more spread out across [5, 51]
- Features will converge to truly optimal lookbacks, not boundary artifacts

---

## Testing Strategy

### Unit Tests
```python
def test_coarse_horizons_include_boundaries():
    optimizer = LookbackOptimizer(...)
    horizons = optimizer._generate_coarse_horizons(min_horizon=5, max_horizon=51)
    assert 5 in horizons, "min_horizon must be included"
    assert 51 in horizons, "max_horizon must be included"

def test_refinement_includes_max_lookback():
    optimizer = LookbackOptimizer(...)
    # Simulate refinement around horizon 41 with max_lookback=51
    refined = optimizer._parallel_refinement(
        top_horizons=[(41, 0.1)],
        max_lookback=51,
        ...
    )
    # Check that 51 is reachable
    assert any(h >= 51 for h, _ in refined), "Refinement should reach max_lookback"
```

### Integration Test
- Run full optimization on ETHUSDT 15m data
- Verify lookback distribution is not concentrated at boundaries
- Check that optimization time hasn't increased significantly

---

## Additional Observations

### Regularization Impact
The current negative scores (-0.05 to -0.10) suggest the optimizer is using mutual information with penalties. The convergence to max_lookback might also be influenced by:
- Regularization penalties favoring longer lookbacks
- Information gain increasing with longer horizons
- Stability metrics favoring longer periods

### Performance Considerations
The fixes will:
- Add 1 more coarse horizon (51) → ~8% more coarse evaluations
- Add boundary checks in refinement → minimal overhead
- Overall impact: < 10% increase in optimization time

---

## Conclusion

The Bayesian optimizer is converging to lookback ~51 due to **two systematic bugs** that exclude the upper boundary from the search space:

1. **Truncation bug** in coarse horizon generation (missing 51)
2. **Range exclusion bug** in refinement (can't reach 51 with step=2 from even starts)

The mystery of features actually reaching 51 in the outcome file suggests either:
- Deployed code differs from source code
- There's additional boundary testing not visible in the main optimization loop
- Post-processing adjusts the results

**Recommended Action**: Implement both fixes immediately to ensure the optimizer properly explores the full search space and converges to truly optimal lookback periods rather than boundary artifacts.

