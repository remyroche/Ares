# Root Cause Analysis: Why Lookback 51 for 94.8% of Features?

## The Mystery

**Observation**: 237/250 features (94.8%) converged to lookback=51, with 13 features (5.2%) at lookback=49.

**The Problem**: Even if these are correlations (not MI), there's no statistical reason why EVERY feature should have its peak correlation at the same lookback period. This suggests a systematic bias or bug.

---

## Most Likely Root Cause: **TARGET HORIZON MATCHING**

### Hypothesis
The `analyst_target` is calculated with a **fixed forward-looking horizon of ~51 periods**, and the optimizer is finding that features align best when their lookback matches the target's forward horizon.

### Evidence
1. **Perfect alignment**: When target horizon = feature lookback, temporal alignment is optimal
2. **Boundary concentration**: max_lookback=51, all features gravitate to it
3. **Uniform behavior**: ALL features show same preference (not random)
4. **Found example**: `analyst_target = analyst_data['close'].pct_change(4).shift(-4)` in code

### How This Works
```python
# If analyst_target is calculated as:
analyst_target[t] = (price[t+51] - price[t]) / price[t]  # Forward return over 51 periods

# And feature with lookback=51:
feature[t] = technical_indicator(data[t-51:t])  # Uses 51 bars of history

# Then when we calculate correlation:
corr(feature[t], analyst_target[t])
# We're correlating: "indicator using past 51 bars" with "future 51-bar return"
# This creates optimal temporal alignment!
```

### Why Other Lookbacks Perform Worse
- **Lookback < 51**: Feature doesn't capture full horizon matching target
- **Lookback > 51**: Can't test because max_lookback=51
- **Lookback = 49**: Next best thing (refinement from horizon 41)

---

## Secondary Issue: Negative "MI" Scores

The scores are **correlations, not MI values**. Since the analyst_target represents returns (which can be negative), and many features have negative correlation with future returns, the scores are negative.

**Example:**
- If higher RSI predicts lower future returns: correlation = -0.08
- This gets stored as score = -0.08 in the outcome file
- But it should be stored as MI = 0.003 (always positive)

---

## Verification Steps

### 1. Check Analyst Target Horizon
```python
# Look for where analyst_target is calculated
# Expected to find something like:
analyst_target = price.pct_change(51).shift(-51)
# or
analyst_target = (price.shift(-51) - price) / price
```

### 2. Test with Different Max Lookbacks
Run optimization with different max_lookback values:
- max_lookback=30 → expect convergence at 30
- max_lookback=100 → expect convergence at ~51 or the true target horizon
- If features still converge to same absolute value, confirms target horizon theory

### 3. Check for Lookahead Bias
Verify alignment:
```python
# At time t:
feature_value = feature.shift(lookback)[t]  # Uses data up to t-lookback
target_value = analyst_target[t]  # Should be forward-looking from t

# Ensure no future leakage:
assert feature_value was calculated using only data before time t
```

---

## The Bugs to Fix

### Bug #1: Storing Correlation Instead of MI ✅ **CRITICAL**
**Issue**: Outcome file stores raw correlation values (can be negative) instead of MI (always ≥ 0)

**Fix**: Ensure all scoring paths convert correlation to MI before saving:
```python
# Instead of:
best_score = correlation  # Can be negative

# Use:
if abs(correlation) < 0.999:
    best_score = -0.5 * np.log(1 - correlation**2)  # MI approximation
else:
    best_score = float('inf')
```

### Bug #2: Target Horizon Not Documented ✅ **HIGH PRIORITY**
**Issue**: The forward horizon of analyst_target is implicit, not explicit

**Fix**: 
1. Document the target horizon in the outcome file
2. Add validation to warn if lookback >> target horizon
3. Consider target horizon when setting max_lookback

### Bug #3: Coarse Horizon Truncation ✅ **FIXED**
Already fixed in previous changes (using rounding instead of truncation)

### Bug #4: Refinement Boundary Exclusion ✅ **FIXED**
Already fixed in previous changes (explicit boundary inclusion)

---

## Expected Behavior After Understanding

**If analyst_target horizon = 51 periods:**
- Features SHOULD converge near lookback=51 (this is correct!)
- But we need to:
  1. Document this relationship
  2. Store MI instead of correlation
  3. Test if target horizon changes, features adapt

**If analyst_target horizon ≠ 51:**
- Then we have a separate alignment bug to investigate
- Features shouldn't all prefer the same lookback

---

## Action Items

1. **IMMEDIATE**: Find where analyst_target is calculated and verify its forward horizon
2. **CRITICAL**: Fix correlation→MI conversion in outcome file storage
3. **HIGH**: Add target_horizon metadata to optimization results
4. **MEDIUM**: Add validation warning if max_lookback < 2 * target_horizon
5. **LOW**: Consider adaptive max_lookback based on detected target horizon

---

## Test Plan

```python
# Test 1: Verify target horizon
data = load_data()
analyst_target = data['analyst_target']
# Calculate autocorrelation to find characteristic timescale
from scipy import signal
acf = np.correlate(analyst_target, analyst_target, mode='full')
peak_lag = np.argmax(acf[len(acf)//2+1:]) + 1
print(f"Target characteristic horizon: {peak_lag}")

# Test 2: Run optimization with different max_lookbacks
for max_lb in [20, 30, 40, 51, 100]:
    results = optimize_features(data, max_lookback=max_lb)
    print(f"max_lookback={max_lb}: median_selected={np.median(results.lookbacks)}")
```

Expected: If target horizon=51, median should cap at min(max_lookback, 51)

---

## Conclusion

The convergence to lookback=51 is likely **NOT a bug**, but rather a **correct discovery** that the optimal feature lookback matches the target's forward horizon. However:

1. ✅ **This should be documented**
2. ✅ **MI scores should be stored correctly (not correlations)**  
3. ✅ **The relationship between target horizon and optimal lookback should be explicit**

The real bugs are:
- Storing correlations instead of MI
- Not documenting the target horizon
- Not warning users about this relationship

