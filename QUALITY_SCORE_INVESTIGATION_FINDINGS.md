# 🔍 Quality Score Investigation - Complete Findings

**Date:** November 2, 2025  
**Dataset:** 499 samples, ETHUSDT 15m (May-Sept 2025)  
**Investigation Target:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

---

## 📊 Executive Summary

### ✅ **GOOD NEWS:**
1. **Forward-looking formula** - Quality score uses FUTURE data (bounce, hold, profit)
2. **Decent distribution** - Mean 0.557, Std 0.246, IQR 0.407 (not binary)
3. **Proper filtering** - Untested levels (quality=0.2) correctly removed
4. **Moderate correlations** - Top feature (prominence) correlates at 0.357

### ⚠️ **CONCERNS IDENTIFIED:**

#### 🚨 **CRITICAL ISSUE #1: Bounce Strength Saturation**
- **bounce_strength mean: 0.9757** (almost maxed out!)
- **bounce_strength median: 1.0000** (50% of samples at max)
- **Problem:** Almost every level shows "perfect" bounce, diluting signal
- **Impact:** 35% of quality score is nearly constant (no discriminative power)

#### 🚨 **CRITICAL ISSUE #2: Negative Trade Profit**
- **trade_profit mean: -0.0523** (average trade LOSES money)
- **trade_profit median: -0.5000** (most trades hit SL)
- **Problem:** 30% of quality score is negative/zero on average
- **Impact:** Quality formula penalizes most levels unfairly

#### ⚠️ **ISSUE #3: Weak Feature Correlations**
- Only **3 features** have strong correlation (>0.3)
- **28 features** have weak correlation (<0.1)
- Top correlation only **0.357** (moderate at best)
- **Impact:** Limited predictive power from features

#### ⚠️ **ISSUE #4: Distribution Skew**
- **14% of samples at exactly 1.0** (70 out of 499)
- **Positive skew (0.78)** - tail towards high quality
- **25th percentile: 0.3675** (lower quartile very compressed)

---

## 🧩 Component Breakdown

### Quality Score Formula:
```python
quality_score = (
    bounce_strength * 0.35 +    # Future bounce after hit
    hold_strength * 0.35 +      # How long level holds
    trade_profit * 0.30         # Simulated trade P&L
)
```

### Component Analysis:

| Component | Mean | Median | Std | Min | Max | **Issue** |
|-----------|------|--------|-----|-----|-----|-----------|
| bounce_strength | 0.9757 | 1.0000 | 0.1028 | 0.35 | 1.00 | ❌ **Saturated!** |
| hold_strength | 0.3965 | 0.1000 | 0.4219 | 0.05 | 1.00 | ✅ Good variance |
| trade_profit | -0.0523 | -0.5000 | 0.6157 | -0.50 | 1.00 | ❌ **Negative!** |

**Effective formula after averaging:**
```
quality_score ≈ 0.9757 * 0.35 + hold_strength * 0.35 + (-0.05) * 0.30
quality_score ≈ 0.341 + hold_strength * 0.35 - 0.015
quality_score ≈ 0.326 + hold_strength * 0.35
```

**Interpretation:** Quality score is **DOMINATED by hold_strength** due to bounce saturation and negative trade profit!

---

## 🔗 Feature Correlation Analysis

### Top 20 Features by Correlation:

| Rank | Feature | Correlation | Category |
|------|---------|-------------|----------|
| 1 | prominence | **0.357** | 🟢 Strong |
| 2 | prominence_x_strength | **0.341** | 🟢 Strong |
| 3 | failure_count | **0.312** | 🟢 Strong |
| 4 | recency_x_strength | 0.240 | 🟡 Moderate |
| 5 | time_adjusted_strength | 0.234 | 🟡 Moderate |
| 6 | vol_adjusted_strength | 0.234 | 🟡 Moderate |
| 7 | strength | 0.233 | 🟡 Moderate |
| 8 | success_rate | 0.198 | 🟡 Moderate |
| 9 | prominence_x_width | 0.193 | 🟡 Moderate |
| 10 | success_x_strength | 0.173 | 🟡 Moderate |
| 11-20 | ... | 0.14-0.11 | 🟠 Weak |

### Correlation with Historical Features:
- **touch_count**: -0.103 (negative!)
- **strength**: 0.233 (moderate)
- **age_bars**: -0.073 (negative)
- **failure_count**: -0.312 (strong negative!)

**✅ Good:** Quality is NOT just historical strength (avoids overfitting to past)  
**⚠️ Concern:** Negative correlation with failure_count seems counterintuitive

---

## 🚨 Red Flags Summary

| Check | Result | Status | Details |
|-------|--------|--------|---------|
| Binary distribution | 15.8% at extremes | ✅ OK | <50% threshold |
| Distribution width | IQR = 0.407 | ✅ OK | >0.1 threshold |
| Max correlation | 0.357 | ✅ OK | >0.2 threshold |
| Variance | std = 0.246 | ✅ OK | >0.05 threshold |
| **Bounce saturation** | **mean = 0.976** | ❌ **FAIL** | Nearly constant |
| **Trade profit** | **mean = -0.05** | ❌ **FAIL** | Negative |
| **Weak features** | **28 features <0.1** | ⚠️ **WARN** | Limited utility |

---

## 💡 Root Cause Analysis

### Why is bounce_strength saturated?

Looking at the code (`sr_quality_data_collector.py`, lines 410-420):

```python
# 1. Bounce Strength
if level_type == 'support':
    future_highs = future_data.loc[first_hit_idx:, 'high']
    max_bounce = future_highs.max() - hit_bar['low']
    bounce_pct = max_bounce / level_price
else:  # resistance
    future_lows = future_data.loc[first_hit_idx:, 'low']
    max_bounce = hit_bar['high'] - future_lows.min()
    bounce_pct = max_bounce / level_price

bounce_strength = min(bounce_pct / 0.02, 1.0)  # 2% bounce = 1.0
```

**Problem:** Uses `max()` over entire forward window → almost always finds a 2%+ move  
**Result:** bounce_strength caps at 1.0 for ~50% of samples

### Why is trade_profit negative?

Looking at the trade simulation (`sr_quality_data_collector.py`, lines 470-505):

```python
def _simulate_trade(self, level_type: str, entry_price: float,
                   future_data: pd.DataFrame, hit_idx) -> float:
    if level_type == 'support':
        stop_loss = entry_price * 0.99     # 1% SL
        take_profit = entry_price * 1.02   # 2% TP
```

**Problem:** 2:1 R/R (2% TP vs 1% SL) is too aggressive for 15m timeframe  
**Result:** Stop loss hit more frequently than take profit → negative expectancy

---

## 🔧 Recommended Fixes

### **FIX #1: Improve Bounce Strength Calculation** (HIGH PRIORITY)

**Current Issue:** Uses max bounce over entire window → saturated at 1.0

**Proposed Fix:**
```python
# BEFORE (saturated):
bounce_strength = min(bounce_pct / 0.02, 1.0)  # 2% = max

# AFTER (better discrimination):
# Option A: Use FIRST significant bounce (not max)
first_bounce_bars = 10  # First 10 bars after hit
early_future = future_data.loc[first_hit_idx:].iloc[:first_bounce_bars]
if level_type == 'support':
    max_bounce = early_future['high'].max() - hit_bar['low']
else:
    max_bounce = hit_bar['high'] - early_future['low'].min()
bounce_pct = max_bounce / level_price
bounce_strength = min(bounce_pct / 0.03, 1.0)  # 3% = max (harder threshold)

# Option B: Use median bounce instead of max
all_bounces = []
for idx, bar in future_data.loc[first_hit_idx:].iloc[:20].iterrows():
    if level_type == 'support':
        bounce = bar['high'] - hit_bar['low']
    else:
        bounce = hit_bar['high'] - bar['low']
    all_bounces.append(bounce / level_price)
bounce_strength = min(np.median(all_bounces) / 0.02, 1.0)

# Option C: Weighted by time decay (recent bounces matter more)
bounce_strength = 0
total_weight = 0
for i, (idx, bar) in enumerate(future_data.loc[first_hit_idx:].iloc[:20].iterrows()):
    weight = np.exp(-i / 10)  # Exponential decay
    if level_type == 'support':
        bounce = (bar['high'] - hit_bar['low']) / level_price
    else:
        bounce = (hit_bar['high'] - bar['low']) / level_price
    bounce_strength += bounce * weight
    total_weight += weight
bounce_strength = min(bounce_strength / total_weight / 0.02, 1.0)
```

### **FIX #2: Adjust Trade Simulation** (HIGH PRIORITY)

**Current Issue:** 2:1 R/R too aggressive → -0.05 average (losing)

**Proposed Fix:**
```python
# BEFORE (losing):
if level_type == 'support':
    stop_loss = entry_price * 0.99     # 1% SL
    take_profit = entry_price * 1.02   # 2% TP (2:1 R/R)

# AFTER (more realistic for 15m):
if level_type == 'support':
    stop_loss = entry_price * 0.995    # 0.5% SL (tighter)
    take_profit = entry_price * 1.01   # 1% TP (2:1 R/R maintained)
    
# OR: Use 1:1 R/R for 15m timeframe
if level_type == 'support':
    stop_loss = entry_price * 0.99     # 1% SL
    take_profit = entry_price * 1.01   # 1% TP (1:1 R/R)
```

**Alternative:** Remove trade_profit entirely, increase bounce/hold weights:
```python
quality_score = (
    bounce_strength * 0.50 +    # 50% (was 35%)
    hold_strength * 0.50        # 50% (was 35%)
    # Remove trade_profit (unrealistic simulation)
)
```

### **FIX #3: Rebalance Quality Formula Weights** (MEDIUM PRIORITY)

**Current Issue:** Bounce saturated (0.98) + trade negative (-0.05) → only hold_strength matters

**Proposed Fix:**
```python
# AFTER fixing bounce and trade issues above, consider:

# Option A: Equal weights
quality_score = (
    bounce_strength * 0.333 +
    hold_strength * 0.333 +
    trade_profit * 0.333
)

# Option B: Emphasize hold (most predictive currently)
quality_score = (
    bounce_strength * 0.25 +
    hold_strength * 0.50 +  # Emphasize
    trade_profit * 0.25
)

# Option C: Add recency factor (levels tested recently = better)
recency_factor = np.exp(-bars_since_hit / 50)  # Decay over 50 bars
quality_score = (
    bounce_strength * 0.30 +
    hold_strength * 0.30 +
    trade_profit * 0.25 +
    recency_factor * 0.15  # Recent test = higher quality
)
```

### **FIX #4: Feature Engineering** (LOW PRIORITY)

**Current Issue:** Only 3 strong correlations, many weak features

**Proposed Actions:**
1. **Drop weak features** (<0.1 correlation): Reduces noise, improves training
2. **Create new interaction features** based on top correlators:
   - `prominence_x_hold` (prominence correlated, hold correlated)
   - `failure_x_bounce` (failure negatively correlated)
3. **Add regime-specific features:**
   - `quality_in_volatile_regime` (separate quality for high volatility)
   - `quality_in_trending_regime` (separate for strong trends)

---

## 📋 Action Plan (Prioritized)

### **IMMEDIATE (Critical):**
1. ✅ ~~Investigate quality score calculation~~ ← **DONE**
2. ⚠️ **Fix bounce_strength saturation** → Use Option A or C above
3. ⚠️ **Fix trade_profit negativity** → Adjust R/R or remove

### **SHORT-TERM (Important):**
4. Rebalance quality formula weights after fixes
5. Recollect training data with new formula
6. Re-run validation to verify improvements
7. Check if correlations improve

### **MEDIUM-TERM (Improvement):**
8. Feature selection: Drop weak features (<0.1 correlation)
9. Add new engineered features
10. Retrain model with cleaned dataset

### **LONG-TERM (Optimization):**
11. Test different forward_days windows (currently 10 days)
12. A/B test formula variations (equal weights vs emphasized hold)
13. Consider separate models for different regimes

---

## 📊 Expected Improvements After Fixes

### Current State:
- bounce_strength: 97.6% of samples near 1.0 → **No discrimination**
- trade_profit: -5.2% average → **Penalizes unfairly**
- Quality dominated by hold_strength only

### After Fix #1 (Bounce):
- bounce_strength: Expected mean ~0.5-0.6, std ~0.3 → **Better spread**
- Strong/weak bounces differentiated → **Discriminative power**

### After Fix #2 (Trade):
- trade_profit: Expected mean ~0.1-0.2 (positive) → **Reward good levels**
- Win rate ~40-50% (realistic for 2:1 R/R) → **Fair assessment**

### After Fix #3 (Rebalance):
- quality_score: All components contribute meaningfully
- Distribution: Less skewed, more normal
- Correlations: Expected to improve to 0.4-0.5 range

---

## 🎯 Validation Checklist (After Fixes)

Run `python3 validate_quality_score.py` again and check:

- [ ] bounce_strength mean < 0.8 (not saturated)
- [ ] trade_profit mean > 0 (positive expectancy)
- [ ] quality_score std > 0.25 (good variance)
- [ ] Top feature correlation > 0.4 (stronger)
- [ ] <10% samples at extremes (not binary)
- [ ] IQR > 0.4 (good spread)
- [ ] No component dominates (balanced)

---

## 📎 Files Generated

1. `analysis_output/quality_score_distribution.png` - Distribution plots
2. `analysis_output/quality_components.png` - Component analysis
3. `analysis_output/feature_correlations.png` - Feature correlation heatmap
4. `analysis_output/quality_score_investigation_report.txt` - Text summary
5. `QUALITY_SCORE_INVESTIGATION_FINDINGS.md` - **This document**

---

## 🔗 Related Code Files

- **Main collector:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`
  - Quality calculation: Lines 442-446
  - Bounce calculation: Lines 410-420
  - Trade simulation: Lines 470-505
  
- **Model training:** `src/tactician/sr_levels/ml_quality/sr_quality_model.py`
  
- **Validation script:** `validate_quality_score.py` (newly created)

---

## 💭 Conclusion

The quality score formula is **conceptually sound** (forward-looking, multi-component), but has **implementation issues**:

1. **Bounce strength is saturated** → Fix threshold and use early/median bounce
2. **Trade profit is negative** → Adjust R/R or remove component
3. **Distribution is skewed** → Will improve after above fixes

**Recommended approach:** Implement Fix #1 and #2, recollect data, validate improvements.

---

**Investigation completed:** November 2, 2025  
**Next step:** Implement recommended fixes and recollect training data

