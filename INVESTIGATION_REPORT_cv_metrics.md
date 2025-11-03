# Investigation Report: cv_order_flow and economic_cv_ratio Always 0.0

## Date: 2025-11-02
## File Analyzed: `outcomes/stage1_checkpoint_50_20251102_172245.csv`

---

## Executive Summary

Two critical metrics are consistently returning 0.0 across all 50 test runs:
1. **`cv_order_flow`**: Always 0.0
2. **`economic_cv_ratio`**: Always 0.0

### Root Causes Identified

---

## Issue 1: `cv_order_flow` Always 0.0

### **Root Cause**
No order flow features exist in the feature cache (`hdp_hmm_features_cache.pkl`).

### **Evidence**
```python
# hdp_hmm_single_test.py lines 272-293
categories = {
    'order_flow': ['order_flow_imbalance', 'order_flow_momentum', 'buy_sell'],
    'microstructure': ['price_impact', 'vw_price_range', ...],
    # ... other categories
}

for cat_name, patterns in categories.items():
    cat_indices = []
    for idx, fname in enumerate(feature_names):
        if any(pattern in fname for pattern in patterns):
            cat_indices.append(idx)
    
    if not cat_indices:  # <-- THIS TRIGGERS FOR order_flow
        category_cvs[cat_name] = 0.0  # <-- RETURNS 0.0
        continue
```

### **Cache Analysis**
Feature cache contains **100 features** but **ZERO order flow features**:

**Features present:**
- `regime_duration_short/long`
- `lagged_volatility_*`
- `lagged_range_*`
- `lagged_volume_ma_*`
- `price_zscore_*`
- `momentum_*`
- `volatility_*`
- `volume_*`
- `trend_*`

**Features missing:**
- ❌ `order_flow_imbalance`
- ❌ `order_flow_momentum`
- ❌ `buy_sell_*`
- ❌ `taker_*` (taker buy/sell features)
- ❌ `aggression_*`
- ❌ Any order flow proxies

### **Impact**
- Order flow separation between regimes is not being measured
- Cannot evaluate if regimes capture different order flow dynamics
- Missing a critical dimension of market microstructure

### **Fix Required**
Add order flow features to the feature generation pipeline before caching:

```python
# Potential features to add:
- order_flow_imbalance (from MicrostructureFeatureGenerator)
- market_aggression_index
- taker_buy_ratio / taker_sell_ratio
- buy_sell_pressure
- volume-price correlation (as order flow proxy)
```

---

## Issue 2: `economic_cv_ratio` Always 0.0

### **Root Cause**
Incorrect dictionary key path in `hdp_hmm_single_test.py` when extracting economic CV metrics.

### **Evidence**

**What the code expects (lines 335-341):**
```python
economic_cv = safe_nested_get(
    qa, 'economic_cv_metrics', 'economic_cv_ratio', 'mean_return',
    #                           ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^
    #                           Looking for nested dict structure
    default=0.0
)
# Expects: qa['economic_cv_metrics']['economic_cv_ratio']['mean_return']
```

**What is actually returned (`cluster_quality_assessor.py` lines 1377-1385):**
```python
# _calculate_economic_cv_metrics returns:
metrics_data = {
    'economic_avg_within_cv_fwd_return': 0.123,
    'economic_between_cv_mean_return': 0.456,
    'economic_cv_ratio_mean_return': 3.71,  # <-- FLAT KEY, not nested!
    # ... other metrics
}
```

**Actual structure:**
```
qa['economic_cv_metrics']['economic_cv_ratio_mean_return']  # ✅ Correct
qa['economic_cv_metrics']['economic_cv_ratio']['mean_return']  # ❌ Wrong (what code looks for)
```

### **Impact**
- Economic regime separation is not being measured
- Cannot evaluate if regimes have different forward return distributions
- Missing critical validation metric for trading utility

### **Fix Required**
Change the key lookup in `hdp_hmm_single_test.py`:

```python
# BEFORE (lines 335-341):
economic_cv = safe_nested_get(
    qa, 'economic_cv_metrics', 'economic_cv_ratio', 'mean_return',
    default=0.0
)

# AFTER:
economic_cv = safe_nested_get(
    qa, 'economic_cv_metrics', 'economic_cv_ratio_mean_return',
    default=0.0
)
```

---

## Verification Steps

### **Check if economic metrics are actually calculated**
Run a single test with debug output to inspect the full `quality_assessment` dict:

```python
# Add to hdp_hmm_single_test.py after line 320:
import json
print(f"DEBUG: quality_assessment keys: {list(qa.keys())}", file=sys.stderr)
if 'economic_cv_metrics' in qa:
    print(f"DEBUG: economic_cv_metrics: {json.dumps(qa['economic_cv_metrics'], indent=2)}", 
          file=sys.stderr)
```

### **Check if forward_returns are being passed correctly**
```python
# Add to hdp_hmm_single_test.py after line 207:
print(f"DEBUG: forward_returns shape: {forward_returns.shape if forward_returns is not None else None}", 
      file=sys.stderr)
print(f"DEBUG: forward_returns non-zero: {(forward_returns != 0).sum() if forward_returns is not None else 0}", 
      file=sys.stderr)
```

---

## Recommended Fixes

### **Priority 1: Fix economic_cv_ratio extraction (Quick Win)**

**File:** `hdp_hmm_single_test.py`  
**Lines:** 335-341

```python
# CURRENT CODE:
economic_cv = safe_nested_get(
    qa, 'economic_cv_metrics', 'economic_cv_ratio', 'mean_return',
    default=0.0
)
if economic_cv == 0.0:
    # Fallback: try alternative path
    economic_cv = safe_nested_get(qa, 'economic_cv_metrics', 'mean_return_cv', default=0.0)

# FIXED CODE:
economic_cv = safe_nested_get(
    qa, 'economic_cv_metrics', 'economic_cv_ratio_mean_return',
    default=0.0
)
if economic_cv == 0.0:
    # Fallback: try between/within ratio calculation
    between_cv = safe_nested_get(qa, 'economic_cv_metrics', 'economic_between_cv_mean_return', default=0.0)
    within_cv = safe_nested_get(qa, 'economic_cv_metrics', 'economic_avg_within_cv_fwd_return', default=1.0)
    if within_cv > 1e-9:
        economic_cv = between_cv / within_cv
```

### **Priority 2: Add order flow features to feature generation (Longer Term)**

**Option A: Quick Fix - Use existing features as proxies**

Modify pattern matching in `hdp_hmm_single_test.py` to capture volume-price correlation features:

```python
categories = {
    'order_flow': [
        'order_flow_imbalance', 
        'order_flow_momentum', 
        'buy_sell',
        'vol_price_corr',  # ADD: Volume-price correlation as proxy
        'obv',              # ADD: On-Balance Volume as proxy
        'ad_line'           # ADD: Accumulation/Distribution as proxy
    ],
    # ... other categories
}
```

**Option B: Proper Fix - Generate real order flow features**

1. Check which order flow generators are available:
```bash
grep -r "class.*OrderFlow.*Generator" src/feature_generation/
```

2. Add order flow features to `RegimeFeatureIntegration._generate_regime_features()`

3. Regenerate cache with:
```bash
python hdp_hmm_prepare_data.py
```

### **Priority 3: Verify economic metrics calculation**

Add comprehensive logging to verify economic CV metrics are being calculated:

```python
# In cluster_quality_assessor.py after line 1387:
self.logger.info(f"📊 Economic CV Metrics: {metrics_data}")
```

---

## Expected Impact After Fixes

### **economic_cv_ratio fix:**
- Should see values in range [0.1, 10.0] typically
- Values > 1.0 indicate regimes have distinct economic outcomes
- Values < 0.5 indicate poor economic separation

### **order_flow features fix:**
- Should see cv_order_flow values in range [0.0, 5.0]
- Non-zero values indicate regimes capture different order flow patterns
- High values (>2.0) suggest strong order flow regime separation

### **Re-run checkpoint analysis:**
After fixes, the best configurations should show:
- `economic_cv_ratio > 0.5` (meaningful economic separation)
- `cv_order_flow > 0.0` (order flow dynamics captured)
- Better composite scores with these dimensions included

---

## Files to Modify

1. **`hdp_hmm_single_test.py`** (lines 335-341)
   - Fix economic_cv_ratio key path ✅ CRITICAL

2. **`hdp_hmm_single_test.py`** (lines 272-280)
   - Update order_flow patterns to match available features ⚠️ WORKAROUND

3. **`src/feature_generation/categories/regime_feature_integration.py`**
   - Add order flow feature generation 📋 PROPER FIX (longer term)

4. **`hdp_hmm_prepare_data.py`**
   - Regenerate cache with order flow features 📋 AFTER #3

---

## Testing Plan

1. **Quick test:** Fix economic_cv_ratio and run single test
```bash
python hdp_hmm_single_test.py 1.0 32.5 8.0 30
```

2. **Verify output:** Check that economic_cv_ratio is non-zero

3. **Add order flow proxy patterns:** Modify categories dict

4. **Re-run stage 1:** Run small grid (3x3) to verify both metrics

5. **Full re-run:** If successful, re-run full grid search

---

## Conclusion

Both issues are solvable:
- **economic_cv_ratio**: Simple key path fix (5 minutes)
- **cv_order_flow**: Requires either pattern update (5 minutes) or feature addition (1-2 hours)

Recommend implementing **Priority 1** fix immediately to recover economic validation metrics from existing runs.

